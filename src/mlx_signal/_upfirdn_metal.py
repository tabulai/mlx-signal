"""Custom Metal kernel for upfirdn: one thread per output sample.

Each output sample of upfirdn (upsample -> FIR -> downsample) is an independent
dot product over one polyphase branch of the filter, which maps perfectly onto
one GPU thread. Filter taps are staged into threadgroup memory when they fit
(<= 8 KB); larger filters read taps through the device cache.

The kernel is float32-only; complex signals/filters are handled by the caller
with component-wise launches (convolution is linear).
"""

from __future__ import annotations

import functools

import mlx.core as mx

_SMEM_CAP = 2048  # taps staged in threadgroup memory when they fit (8 KB of 32 KB)

_SOURCE = f"""
    uint i = thread_position_in_grid.x;
    uint b = thread_position_in_grid.y;
    int n_in   = params[0];
    int n_out  = params[1];
    int up     = params[2];
    int down   = params[3];
    int n_taps = params[4];

    threadgroup float sh[{_SMEM_CAP}];
    bool use_sh = n_taps <= {_SMEM_CAP};
    if (use_sh) {{
        for (uint t = thread_index_in_threadgroup; t < (uint)n_taps;
             t += threads_per_threadgroup.x) {{
            sh[t] = h[t];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (i >= (uint)n_out) return;

    long m  = (long)i * down;      // position in the upsampled, filtered stream
    int  p  = (int)(m % up);       // polyphase branch
    long j  = m / up;              // newest contributing input sample
    long t0 = j >= n_in ? j - (n_in - 1) : 0;  // skip taps past the input tail
    long k  = p + t0 * (long)up;
    long jj = j - t0;
    float acc = 0.0f;
    const device float* xrow = x + (long)b * n_in;
    for (; k < n_taps && jj >= 0; k += up, --jj) {{
        acc += (use_sh ? sh[(int)k] : h[(int)k]) * xrow[jj];
    }}
    out[(long)b * n_out + i] = acc;
"""


@functools.cache
def _kernel():
    return mx.fast.metal_kernel(
        name="mlx_signal_upfirdn",
        input_names=["x", "h", "params"],
        output_names=["out"],
        source=_SOURCE,
    )


def upfirdn_gpu(x2d: mx.array, h: mx.array, up: int, down: int, n_out: int) -> mx.array:
    """Run the kernel on float32 planes. x2d: (batch, n_in), h: (n_taps,)."""
    batch, n_in = x2d.shape
    params = mx.array([n_in, n_out, up, down, h.shape[0]], dtype=mx.int32)
    outputs = _kernel()(
        inputs=[x2d, h, params],
        grid=(n_out, batch, 1),
        threadgroup=(min(256, max(32, n_out)), 1, 1),
        output_shapes=[(batch, n_out)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]
