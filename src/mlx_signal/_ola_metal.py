"""Gather-based overlap-add for blocked convolution reassembly.

oaconvolve produces per-block full convolutions of length N placed every
``step`` samples (step = N - n_taps + 1, so at most ceil(N/step) = 2 blocks
overlap any output sample). Scatter-adds serialize those collisions; this
kernel inverts the dependency — one thread per output sample sums its (at
most two) overlapping block entries — which is deterministic, atomics-free,
and coalesced.

Complex segments are passed as float32 views and fetched as pairs (no
address-space pointer casts: MLX places small buffers in ``constant`` space).
"""

from __future__ import annotations

import functools

import mlx.core as mx

_SRC = """
    uint i = thread_position_in_grid.x;
    uint b = thread_position_in_grid.y;
    int full_len = params[0];
    int K        = params[1];
    int step     = params[2];
    int B        = params[3];
    if ((int)i >= full_len || (int)b >= B) return;

    int s_hi = (int)i / step;
    if (s_hi > K - 1) s_hi = K - 1;
    int s_lo = ((int)i >= {N}) ? (((int)i - {N}) / step + 1) : 0;

    {ACCT} acc = {ZERO};
    long base = (long)b * K * {N} * {W};
    for (int s = s_lo; s <= s_hi; ++s) {{
        long off = base + ((long)s * {N} + ((int)i - s * step)) * {W};
        {FETCH}
    }}
    {STORE}
"""


@functools.lru_cache(maxsize=32)
def _kernel(N: int, cplx: bool):
    if cplx:
        acct, zero = "float2", "float2(0.0f, 0.0f)"
        fetch = "acc += float2(segs[off], segs[off + 1]);"
        store = ("long o = ((long)b * full_len + i) * 2; "
                 "out[o] = acc.x; out[o + 1] = acc.y;")
    else:
        acct, zero = "float", "0.0f"
        fetch = "acc += segs[off];"
        store = "out[(long)b * full_len + i] = acc;"
    return mx.fast.metal_kernel(
        name=f"mlx_signal_conv_ola_{N}_{int(cplx)}",
        input_names=["segs", "params"],
        output_names=["out"],
        source=_SRC.format(N=N, W=2 if cplx else 1, ACCT=acct, ZERO=zero,
                           FETCH=fetch, STORE=store),
    )


def ola_gather(segs: mx.array, step: int, full_len: int) -> mx.array:
    """Overlap-add (B, K, N) block outputs into (B, full_len) with hop ``step``."""
    B, K, N = segs.shape
    cplx = segs.dtype == mx.complex64
    sin = mx.view(segs, mx.float32) if cplx else segs
    params = mx.array([full_len, K, step, B], dtype=mx.int32)
    width = 2 * full_len if cplx else full_len
    (out,) = _kernel(N, cplx)(
        inputs=[sin, params],
        grid=(full_len, B, 1),
        threadgroup=(min(256, max(32, full_len)), 1, 1),
        output_shapes=[(B, width)],
        output_dtypes=[mx.float32],
    )
    return mx.view(out, mx.complex64) if cplx else out
