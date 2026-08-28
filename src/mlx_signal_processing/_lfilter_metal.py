"""Batched-channel IIR (transfer-function form) Metal kernel.

The order-N direct-form-II-transposed recurrence of ``scipy.signal.lfilter``
(``len(a) > 1``), generalized from the second-order-section kernels in
``_sosfilt_metal.py``: per-filter coefficients and the length-N state live in
registers (the order is baked into the kernel source so the compiler fully
unrolls the state update), with the same two implementations:

- sequential: one thread per channel — short signals with many channels;
- block-parallel scan: parallel over time as well as channels — long signals,
  faster than scipy from a single channel up.

Exactness: scipy's compiled C contracts the recurrence into fused
multiply-adds (measured op by op against scipy 1.18 on arm64 — 100% bitwise
over 20k single-sample probes per line):

    y      = fma(b[0], x, z[0])
    z[k]   = fma(-a[k+1], y, fma(b[k+1], x, z[k+1]))   (middle states)
    z[N-1] = fma(b[N], x, -(a[N] * y))                 (product single-rounded)

The kernel spells those ``fma()`` calls out explicitly rather than relying on
the Metal compiler contracting plain expressions the same way clang did, so
the structure is pinned regardless of fast-math settings. Coefficients arrive
pre-normalized by ``a[0]`` in float32 — bit-identical to scipy's in-loop
``b[k] / a0`` division, which yields the same f32 quotient every sample.
One hardware caveat: Apple GPUs flush float32 denormals to zero while
scipy's CPU recurrence keeps them, so bit-identity holds for normal-range
signal/coefficient/state values (below ~1.2e-38 the routes diverge).

The kernels read and write float32 and take/return the (B, N) filter state,
so scipy's ``zi``/``zf`` contract is supported natively. Complex signals with
real coefficients are handled by the caller as two launches (filtering is
linear).
"""

from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np

MAX_ORDER = 16  # register-resident state/coefficients; more falls back to scipy

#: samples per scan block (L); shared with the SOS kernels' measured optimum
SCAN_BLOCK = 1024

# {N}: filter order (state length); {P} = N + 1 taps per side. `ba` is the
# flat [b[0..N], a[0..N]] pair, normalized so aa[0] == 1 (and unused).
_TF_STEP = """
        float yv = fma(bb[0], v, z[0]);
        for (int k = 0; k < {N} - 1; ++k)
            z[k] = fma(-aa[k + 1], yv, fma(bb[k + 1], v, z[k + 1]));
        z[{N} - 1] = fma(bb[{N}], v, -(aa[{N}] * yv));
"""

_SRC = """
    uint chan = thread_position_in_grid.x;
    int B = params[0];
    int n = params[1];
    if ((int)chan >= B) return;

    long row = (long)chan * n;

    float bb[{P}], aa[{P}], z[{N}];
    for (int k = 0; k < {P}; ++k) {{
        bb[k] = ba[k];
        aa[k] = ba[{P} + k];
    }}
    for (int k = 0; k < {N}; ++k) z[k] = zi[(long)chan * {N} + k];

    for (int t = 0; t < n; ++t) {{
        float v = x[row + t];
{STEP}
        y[row + t] = yv;
    }}

    for (int k = 0; k < {N}; ++k) zf[(long)chan * {N} + k] = z[k];
"""


@functools.lru_cache(maxsize=32)
def _kernel(order: int):
    return mx.fast.metal_kernel(
        name=f"mlx_signal_processing_lfilter_{order}",
        input_names=["x", "ba", "zi", "params"],
        output_names=["y", "zf"],
        source=_SRC.format(N=order, P=order + 1, STEP=_TF_STEP.format(N=order)),
    )


def lfilter_gpu(x2d: mx.array, ba_flat: mx.array, zi: mx.array):
    """Run the recurrence on float32 planes.

    x2d: (B, n); ba_flat: (2*(N+1),) normalized [b, a]; zi: (B, N).
    Returns (y (B, n), zf (B, N)).
    """
    B, n = x2d.shape
    order = ba_flat.shape[0] // 2 - 1
    params = mx.array([B, n], dtype=mx.int32)
    y, zf = _kernel(order)(
        inputs=[x2d, ba_flat, zi, params],
        grid=(B, 1, 1),
        threadgroup=(min(256, max(32, B)), 1, 1),
        output_shapes=[(B, n), (B, order)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return y, zf


# ---------------------------------------------------------------------------
# block-parallel (associative-scan) variant for few channels / long signals
# ---------------------------------------------------------------------------
#
# Same construction as the SOS scan: the state z (length N) evolves as
# z' = A z + B x with constant A. Blocks of length L compute their
# zero-entry-state contributions d_k in parallel (phase 1); a tiny sequential
# pass composes entry states through the host-precomputed A^L (phase 2); each
# block re-runs the exact recurrence from its true entry state (phase 3).
# A here is the companion-like map of the DF2T update — non-normal, so the
# dispatch gate checks the norm of A^L directly rather than only pole radii.

_P1_SRC = """
    uint blk  = thread_position_in_grid.x;
    uint chan = thread_position_in_grid.y;
    int B = params[0];
    int n = params[1];
    int K = params[2];
    if ((int)blk >= K || (int)chan >= B) return;

    long xoff = (long)chan * n + (long)((int)blk * {L});
    int start = (int)blk * {L};
    int len = min({L}, n - start);

    float bb[{P}], aa[{P}], z[{N}];
    for (int k = 0; k < {P}; ++k) {{
        bb[k] = ba[k];
        aa[k] = ba[{P} + k];
    }}
    for (int k = 0; k < {N}; ++k) z[k] = 0.0f;

    for (int t = 0; t < len; ++t) {{
        float v = x[xoff + t];
{STEP}
        (void)yv;
    }}
    long doff = ((long)chan * K + blk) * {N};
    for (int k = 0; k < {N}; ++k) d[doff + k] = z[k];
"""

_P2_SRC = """
    uint chan = thread_position_in_grid.x;
    int B = params[0];
    int K = params[2];
    if ((int)chan >= B) return;

    float z[{N}];
    for (int i = 0; i < {N}; ++i) z[i] = zi[(long)chan * {N} + i];
    for (int k = 0; k < K; ++k) {{
        long zoff = ((long)chan * K + k) * {N};
        for (int i = 0; i < {N}; ++i) zin[zoff + i] = z[i];
        if (k < K - 1) {{
            long doff = ((long)chan * K + k) * {N};
            float zn[{N}];
            for (int r = 0; r < {N}; ++r) {{
                float acc = d[doff + r];
                for (int c = 0; c < {N}; ++c) acc += AL[r * {N} + c] * z[c];
                zn[r] = acc;
            }}
            for (int i = 0; i < {N}; ++i) z[i] = zn[i];
        }}
    }}
"""

_P3_SRC = """
    uint blk  = thread_position_in_grid.x;
    uint chan = thread_position_in_grid.y;
    int B = params[0];
    int n = params[1];
    int K = params[2];
    if ((int)blk >= K || (int)chan >= B) return;

    long row = (long)chan * n;
    int start = (int)blk * {L};
    int len = min({L}, n - start);

    float bb[{P}], aa[{P}], z[{N}];
    long zoff = ((long)chan * K + blk) * {N};
    for (int k = 0; k < {P}; ++k) {{
        bb[k] = ba[k];
        aa[k] = ba[{P} + k];
    }}
    for (int k = 0; k < {N}; ++k) z[k] = zin[zoff + k];

    for (int t = 0; t < len; ++t) {{
        float v = x[row + start + t];
{STEP}
        y[row + start + t] = yv;
    }}
    if ((int)blk == K - 1) {{
        long zfoff = (long)chan * {N};
        for (int k = 0; k < {N}; ++k) zf[zfoff + k] = z[k];
    }}
"""


@functools.lru_cache(maxsize=32)
def _scan_kernels(order: int, L: int):
    def make(name, src, outs):
        return mx.fast.metal_kernel(
            name=f"mlx_signal_processing_lfilter_scan_{name}_{order}_{L}",
            input_names={
                "p1": ["x", "ba", "params"],
                "p2": ["zi", "d", "AL", "params"],
                "p3": ["x", "ba", "zin", "params"],
            }[name],
            output_names=outs,
            source=src,
        )

    step = _TF_STEP.format(N=order)
    p1 = make("p1", _P1_SRC.format(N=order, P=order + 1, L=L, STEP=step), ["d"])
    p2 = make("p2", _P2_SRC.format(N=order), ["zin"])
    p3 = make("p3", _P3_SRC.format(N=order, P=order + 1, L=L, STEP=step), ["y", "zf"])
    return p1, p2, p3


@functools.lru_cache(maxsize=64)
def _transition_np(a_bytes: bytes, order: int, L: int):
    """(A^L, max_t ||A^t||_inf) for the DF2T state, probed in float64.

    ``a_bytes`` is the float64 view of the f32-rounded normalized denominator
    (a[0] == 1), so the transition models exactly the filter the float32
    kernels run. The scan-safety gate needs both values: the companion form is
    non-normal, so the pole radius alone misses transient amplification. The
    returned gain correlates with measured scan-vs-sequential drift (the
    honest calibration lives on filtering._TF_SCAN_MAX_GAIN: ~2-5e-8 x gain
    for well-damped filters, up to ~5e-7 x gain for near-gate resonators).
    """
    a = np.frombuffer(a_bytes, dtype=np.float64)

    def step(z):
        zn = np.empty(order)
        y = z[0]
        for k in range(order - 1):
            zn[k] = z[k + 1] - a[k + 1] * y
        zn[order - 1] = -a[order] * y
        return zn

    A = np.stack([step(np.eye(order)[i]) for i in range(order)], axis=1)
    gain, M = 1.0, np.eye(order)
    for _ in range(L):
        M = A @ M
        if not np.all(np.isfinite(M)):
            return M, float("inf")
        gain = max(gain, float(np.abs(M).sum(axis=1).max()))
    return M, gain


def lfilter_scan_gpu(x2d: mx.array, ba_np, zi: mx.array, L: int = SCAN_BLOCK):
    """Block-parallel recurrence on float32 planes.

    x2d: (B, n); ba_np: float32 (2, N+1) host array of normalized [b, a];
    zi: (B, N). Returns (y (B, n), zf (B, N)). Requires n > L (otherwise use
    the sequential kernel).
    """
    B, n = x2d.shape
    order = int(ba_np.shape[1]) - 1
    K = -(-n // L)
    p1, p2, p3 = _scan_kernels(order, L)
    ba_flat = mx.array(np.ascontiguousarray(ba_np, dtype=np.float32).reshape(-1))
    a64 = np.ascontiguousarray(ba_np[1], dtype=np.float64)
    AL = mx.array(
        np.ascontiguousarray(
            _transition_np(a64.tobytes(), order, L)[0], dtype=np.float32
        ).reshape(-1)
    )
    params = mx.array([B, n, K], dtype=mx.int32)

    (d,) = p1(
        inputs=[x2d, ba_flat, params],
        grid=(K, B, 1),
        threadgroup=(min(256, max(32, K)), 1, 1),
        output_shapes=[(B * K, order)],
        output_dtypes=[mx.float32],
    )
    (zin,) = p2(
        inputs=[zi, d, AL, params],
        grid=(B, 1, 1),
        threadgroup=(min(256, max(32, B)), 1, 1),
        output_shapes=[(B * K, order)],
        output_dtypes=[mx.float32],
    )
    y, zf = p3(
        inputs=[x2d, ba_flat, zin, params],
        grid=(K, B, 1),
        threadgroup=(min(256, max(32, K)), 1, 1),
        output_shapes=[(B, n), (B, order)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return y, zf
