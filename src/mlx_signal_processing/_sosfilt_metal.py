"""Batched-channel IIR (second-order sections) Metal kernel.

Two GPU implementations of scipy's direct-form-II-transposed biquad cascade,
with per-section coefficients and state held in registers (the section count
is baked into the kernel source so the compiler fully unrolls the cascade):

- sequential: one thread per channel — used for short signals with many
  channels;
- block-parallel scan (below): parallel over time as well as channels — used
  for anything longer than a couple of blocks, and faster than scipy from a
  single channel up.

The kernel reads and writes float32 and takes/returns the (B, S, 2) filter
state, so scipy's ``zi``/``zf`` contract is supported natively. Complex
signals with real coefficients are handled by the caller as two launches
(filtering is linear).
"""

from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np

MAX_SECTIONS = 16  # register-resident state/coefficients; more falls back to scipy

_SRC = """
    uint chan = thread_position_in_grid.x;
    int B = params[0];
    int n = params[1];
    if ((int)chan >= B) return;

    long row = (long)chan * n;

    float b0[{S}], b1[{S}], b2[{S}], a1[{S}], a2[{S}], z1[{S}], z2[{S}];
    for (int s = 0; s < {S}; ++s) {{
        b0[s] = sos[s * 6 + 0];
        b1[s] = sos[s * 6 + 1];
        b2[s] = sos[s * 6 + 2];
        a1[s] = sos[s * 6 + 4];
        a2[s] = sos[s * 6 + 5];
        z1[s] = zi[((long)chan * {S} + s) * 2 + 0];
        z2[s] = zi[((long)chan * {S} + s) * 2 + 1];
    }}

    for (int t = 0; t < n; ++t) {{
        float v = x[row + t];
        for (int s = 0; s < {S}; ++s) {{
            float yv = b0[s] * v + z1[s];         // direct form II transposed
            z1[s] = b1[s] * v - a1[s] * yv + z2[s];
            z2[s] = b2[s] * v - a2[s] * yv;
            v = yv;
        }}
        y[row + t] = v;
    }}

    for (int s = 0; s < {S}; ++s) {{
        zf[((long)chan * {S} + s) * 2 + 0] = z1[s];
        zf[((long)chan * {S} + s) * 2 + 1] = z2[s];
    }}
"""


@functools.lru_cache(maxsize=32)
def _kernel(n_sections: int):
    return mx.fast.metal_kernel(
        name=f"mlx_signal_processing_sosfilt_{n_sections}",
        input_names=["x", "sos", "zi", "params"],
        output_names=["y", "zf"],
        source=_SRC.format(S=n_sections),
    )


def sosfilt_gpu(x2d: mx.array, sos_flat: mx.array, zi: mx.array):
    """Run the cascade on float32 planes.

    x2d: (B, n); sos_flat: (S*6,); zi: (B, S, 2). Returns (y (B, n), zf (B, S, 2)).
    """
    B, n = x2d.shape
    S = sos_flat.shape[0] // 6
    params = mx.array([B, n], dtype=mx.int32)
    y, zf = _kernel(S)(
        inputs=[x2d, sos_flat, zi, params],
        grid=(B, 1, 1),
        threadgroup=(min(256, max(32, B)), 1, 1),
        output_shapes=[(B, n), (B, S, 2)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return y, zf


# ---------------------------------------------------------------------------
# block-parallel (associative-scan) variant for few channels / long signals
# ---------------------------------------------------------------------------
#
# The cascade is a linear system: the stacked section states z (length 2S)
# evolve as z' = A z + B x with constant A. Split the time axis into K blocks
# of length L: each block's zero-entry-state run yields its input contribution
# d_k in parallel (phase 1); a tiny sequential pass composes entry states
# through the host-precomputed A^L (phase 2, exact for full blocks); each
# block then re-runs the exact recurrence from its true entry state (phase 3).
# Total work is ~2x the sequential kernel but parallel over B*K blocks, which
# is what wins when the channel count alone can't fill the GPU. Stable filters
# keep A^L well-behaved (spectral radius < 1).

SCAN_BLOCK = 1024  # samples per block (L); measured optimum on M4 Max

_CASCADE_STEP = """
            float yv = b0[s] * v + z1[s];
            z1[s] = b1[s] * v - a1[s] * yv + z2[s];
            z2[s] = b2[s] * v - a2[s] * yv;
            v = yv;
"""

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

    float b0[{S}], b1[{S}], b2[{S}], a1[{S}], a2[{S}], z1[{S}], z2[{S}];
    for (int s = 0; s < {S}; ++s) {{
        b0[s] = sos[s * 6 + 0]; b1[s] = sos[s * 6 + 1]; b2[s] = sos[s * 6 + 2];
        a1[s] = sos[s * 6 + 4]; a2[s] = sos[s * 6 + 5];
        z1[s] = 0.0f; z2[s] = 0.0f;
    }}
    for (int t = 0; t < len; ++t) {{
        float v = x[xoff + t];
        for (int s = 0; s < {S}; ++s) {{
{STEP}
        }}
    }}
    long doff = ((long)chan * K + blk) * (2 * {S});
    for (int s = 0; s < {S}; ++s) {{
        d[doff + 2 * s]     = z1[s];
        d[doff + 2 * s + 1] = z2[s];
    }}
"""

_P2_SRC = """
    uint chan = thread_position_in_grid.x;
    int B = params[0];
    int K = params[2];
    if ((int)chan >= B) return;

    float z[2 * {S}];
    for (int i = 0; i < 2 * {S}; ++i) z[i] = zi[(long)chan * 2 * {S} + i];
    for (int k = 0; k < K; ++k) {{
        long zoff = ((long)chan * K + k) * (2 * {S});
        for (int i = 0; i < 2 * {S}; ++i) zin[zoff + i] = z[i];
        if (k < K - 1) {{
            long doff = ((long)chan * K + k) * (2 * {S});
            float zn[2 * {S}];
            for (int r = 0; r < 2 * {S}; ++r) {{
                float acc = d[doff + r];
                for (int c = 0; c < 2 * {S}; ++c) acc += AL[r * 2 * {S} + c] * z[c];
                zn[r] = acc;
            }}
            for (int i = 0; i < 2 * {S}; ++i) z[i] = zn[i];
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

    float b0[{S}], b1[{S}], b2[{S}], a1[{S}], a2[{S}], z1[{S}], z2[{S}];
    long zoff = ((long)chan * K + blk) * (2 * {S});
    for (int s = 0; s < {S}; ++s) {{
        b0[s] = sos[s * 6 + 0]; b1[s] = sos[s * 6 + 1]; b2[s] = sos[s * 6 + 2];
        a1[s] = sos[s * 6 + 4]; a2[s] = sos[s * 6 + 5];
        z1[s] = zin[zoff + 2 * s];
        z2[s] = zin[zoff + 2 * s + 1];
    }}
    for (int t = 0; t < len; ++t) {{
        float v = x[row + start + t];
        for (int s = 0; s < {S}; ++s) {{
{STEP}
        }}
        y[row + start + t] = v;
    }}
    if ((int)blk == K - 1) {{
        long zfoff = (long)chan * (2 * {S});
        for (int s = 0; s < {S}; ++s) {{
            zf[zfoff + 2 * s]     = z1[s];
            zf[zfoff + 2 * s + 1] = z2[s];
        }}
    }}
"""


@functools.lru_cache(maxsize=32)
def _scan_kernels(n_sections: int, L: int):
    def make(name, src, outs):
        return mx.fast.metal_kernel(
            name=f"mlx_signal_processing_sos_scan_{name}_{n_sections}_{L}",
            input_names={
                "p1": ["x", "sos", "params"],
                "p2": ["zi", "d", "AL", "params"],
                "p3": ["x", "sos", "zin", "params"],
            }[name],
            output_names=outs,
            source=src,
        )

    step = _CASCADE_STEP
    p1 = make("p1", _P1_SRC.format(S=n_sections, L=L, STEP=step), ["d"])
    p2 = make("p2", _P2_SRC.format(S=n_sections), ["zin"])
    p3 = make("p3", _P3_SRC.format(S=n_sections, L=L, STEP=step), ["y", "zf"])
    return p1, p2, p3


@functools.lru_cache(maxsize=64)
def _transition_power(sos_bytes: bytes, n_sections: int, L: int) -> mx.array:
    """A^L for the stacked-section state, probed numerically in float64."""
    sos = np.frombuffer(sos_bytes, dtype=np.float64).reshape(n_sections, 6)
    dim = 2 * n_sections

    def step(z):
        zn = np.empty(dim)
        v = 0.0
        for s in range(n_sections):
            b0, b1, b2, _, a1, a2 = sos[s]
            yv = b0 * v + z[2 * s]
            zn[2 * s] = b1 * v - a1 * yv + z[2 * s + 1]
            zn[2 * s + 1] = b2 * v - a2 * yv
            v = yv
        return zn

    A = np.stack([step(np.eye(dim)[i]) for i in range(dim)], axis=1)
    AL = np.linalg.matrix_power(A, L)
    return mx.array(np.ascontiguousarray(AL, dtype=np.float32).reshape(-1))


def sosfilt_scan_gpu(x2d: mx.array, sos_np, zi: mx.array, L: int = SCAN_BLOCK):
    """Block-parallel cascade on float32 planes.

    x2d: (B, n); sos_np: float64 (S, 6) host array; zi: (B, S, 2).
    Returns (y (B, n), zf (B, S, 2)). Requires n > L (otherwise use the
    sequential kernel).
    """
    B, n = x2d.shape
    S = int(sos_np.shape[0])
    K = -(-n // L)
    p1, p2, p3 = _scan_kernels(S, L)
    sos_flat = mx.array(np.ascontiguousarray(sos_np, dtype=np.float32).reshape(-1))
    # derive A^L from the float32-ROUNDED coefficients so the block transition
    # models exactly the filter the float32 kernels run
    sos32 = np.ascontiguousarray(sos_np, np.float64).astype(np.float32).astype(np.float64)
    AL = _transition_power(sos32.tobytes(), S, L)
    params = mx.array([B, n, K], dtype=mx.int32)

    (d,) = p1(
        inputs=[x2d, sos_flat, params],
        grid=(K, B, 1),
        threadgroup=(min(256, max(32, K)), 1, 1),
        output_shapes=[(B * K, 2 * S)],
        output_dtypes=[mx.float32],
    )
    (zin,) = p2(
        inputs=[zi.reshape(B, 2 * S), d, AL, params],
        grid=(B, 1, 1),
        threadgroup=(min(256, max(32, B)), 1, 1),
        output_shapes=[(B * K, 2 * S)],
        output_dtypes=[mx.float32],
    )
    y, zf = p3(
        inputs=[x2d, sos_flat, zin, params],
        grid=(K, B, 1),
        threadgroup=(min(256, max(32, K)), 1, 1),
        output_shapes=[(B, n), (B, 2 * S)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return y, zf.reshape(B, S, 2)
