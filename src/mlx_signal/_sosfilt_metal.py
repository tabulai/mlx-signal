"""Batched-channel IIR (second-order sections) Metal kernel.

An IIR recurrence is serial in time, so one GPU thread processes one channel
sequentially — exactly scipy's direct-form-II-transposed biquad cascade, with
per-section coefficients and state held in registers (the section count is
baked into the kernel source so the compiler fully unrolls the cascade). With
tens to hundreds of channels the lanes run the recurrence in lockstep and beat
scipy's one-core channel loop; for few channels the caller routes to scipy
instead (a block-parallel associative-scan kernel for the single-channel case
is on the roadmap).

The kernel reads and writes float32 and takes/returns the (B, S, 2) filter
state, so scipy's ``zi``/``zf`` contract is supported natively. Complex
signals with real coefficients are handled by the caller as two launches
(filtering is linear).
"""

from __future__ import annotations

import functools

import mlx.core as mx

MAX_SECTIONS = 16  # register-resident state/coefficients; more falls back to scipy

_SRC = """
    uint chan = thread_position_in_grid.x;
    int B = params[0];
    int n = params[1];
    if ((int)chan >= B) return;

    const device float* xrow = x + (long)chan * n;
    device float* yrow = y + (long)chan * n;

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
        float v = xrow[t];
        for (int s = 0; s < {S}; ++s) {{
            float yv = b0[s] * v + z1[s];         // direct form II transposed
            z1[s] = b1[s] * v - a1[s] * yv + z2[s];
            z2[s] = b2[s] * v - a2[s] * yv;
            v = yv;
        }}
        yrow[t] = v;
    }}

    for (int s = 0; s < {S}; ++s) {{
        zf[((long)chan * {S} + s) * 2 + 0] = z1[s];
        zf[((long)chan * {S} + s) * 2 + 1] = z2[s];
    }}
"""


@functools.lru_cache(maxsize=32)
def _kernel(n_sections: int):
    return mx.fast.metal_kernel(
        name=f"mlx_signal_sosfilt_{n_sections}",
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
