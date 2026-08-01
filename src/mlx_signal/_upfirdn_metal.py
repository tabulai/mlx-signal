"""Custom Metal kernel for upfirdn: one thread per output sample.

Each output sample of upfirdn (upsample -> FIR -> downsample) is an independent
dot product over one polyphase branch of the filter, which maps perfectly onto
one GPU thread. Filter taps stream through threadgroup memory in fixed-size
tiles (with barriers all threads participate in), so arbitrarily long filters
stay in fast memory — the 3201-tap 48 kHz -> 44.1 kHz resampler included.

The kernel is generated per real/complex combination of signal and taps
(complex arrays are passed as float32 views and addressed as ``float2``), so a
complex IQ stream with real taps — the SDR decimation case — is a single
launch instead of separate real/imaginary passes, and complex-taps-times-
complex-signal is one launch instead of four. Per-thread tap windows are
computed exactly from the polyphase geometry, so the inner loop has no bounds
checks.
"""

from __future__ import annotations

import functools

import mlx.core as mx

_TILE_REAL = 2048  # taps per threadgroup-memory tile (8 KB as float)
_TILE_CPLX = 1024  # 8 KB as float2
#: above this upsampling factor each tap is reused by too few threads per
#: threadgroup for staging to pay; read taps straight through the device cache
_DIRECT_UP = 32

_HEADER = """
static inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
"""

# NOTE: no address-space pointer casts — MLX places small input buffers in the
# `constant` address space (size-dependent!), so complex data passed as float32
# views is fetched with explicit paired indexing instead of float2* casts.
_SRC = """
    uint i = thread_position_in_grid.x;
    uint b = thread_position_in_grid.y;
    int n_in   = params[0];
    int n_out  = params[1];
    int up     = params[2];
    int down   = params[3];
    int n_taps = params[4];

    threadgroup {HT} sh[{TILE}];
    long xbase = (long)b * {XW} * n_in;

    // every thread participates in the tile-staging barriers, so inactive
    // threads may not return before the loop
    bool active = (int)i < n_out;
    int p = 0;
    long j = 0, k_lo = 0, k_hi = -1;
    if (active) {{
        long m = (long)i * down;
        p = (int)(m % up);
        j = m / up;
        long t_lo = (j - n_in + 1 > 0) ? (j - n_in + 1) : 0;
        k_lo = p + (long)up * t_lo;  // tap pairing with input index n_in-1
        k_hi = p + (long)up * j;     // tap pairing with input index 0 (incl.)
    }}
    {ACCT} acc = {ZERO};

    for (int tile = 0; tile < n_taps; tile += {TILE}) {{
        int tlen = min({TILE}, n_taps - tile);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int t = thread_index_in_threadgroup; t < tlen;
             t += threads_per_threadgroup.x) {{
            {LOADH}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!active) continue;
        long lo = k_lo > (long)tile ? k_lo : (long)tile;
        long hi = k_hi < (long)(tile + tlen - 1) ? k_hi : (long)(tile + tlen - 1);
        if (lo > hi) continue;
        long rem = (p - lo) % up;
        if (rem < 0) rem += up;
        // one division per tile; jj then decrements as k steps by up
        int k = (int)(lo + rem);
        int khi = (int)hi;
        int jj = (int)(j - ((long)k - p) / up);
        for (; k <= khi; k += up, --jj) {{
            {XFETCH}
            acc += {MULT};
        }}
    }}
    if (active) {{
        {STORE}
    }}
"""


_SRC_DIRECT = """
    uint i = thread_position_in_grid.x;
    uint b = thread_position_in_grid.y;
    int n_in   = params[0];
    int n_out  = params[1];
    int up     = params[2];
    int down   = params[3];
    int n_taps = params[4];
    if ((int)i >= n_out) return;

    long xbase = (long)b * {XW} * n_in;
    long m = (long)i * down;
    int p = (int)(m % up);
    long j = m / up;
    long t_lo = (j - n_in + 1 > 0) ? (j - n_in + 1) : 0;
    long lo = p + (long)up * t_lo;
    long hi = p + (long)up * j;
    if (hi > (long)n_taps - 1) hi = (long)n_taps - 1;
    {ACCT} acc = {ZERO};
    if (lo <= hi) {{
        long rem = (p - lo) % up;
        if (rem < 0) rem += up;
        int k = (int)(lo + rem);
        int khi = (int)hi;
        int jj = (int)(j - ((long)k - p) / up);
        for (; k <= khi; k += up, --jj) {{
            {HFETCH}
            {XFETCH}
            acc += {MULT};
        }}
    }}
    {STORE}
"""


@functools.lru_cache(maxsize=16)
def _kernel(cx: bool, ch: bool, direct: bool):
    ht = "float2" if ch else "float"
    complex_out = cx or ch
    loadh = (
        "sh[t] = float2(h[2 * (tile + t)], h[2 * (tile + t) + 1]);"
        if ch else "sh[t] = h[tile + t];"
    )
    hfetch = (
        "float2 hv = float2(h[2 * k], h[2 * k + 1]);"
        if ch else "float hv = h[k];"
    )
    xfetch = (
        "float2 xv = float2(x[xbase + 2 * jj], x[xbase + 2 * jj + 1]);"
        if cx else "float xv = x[xbase + jj];"
    )
    hval = "hv" if direct else "sh[k - tile]"
    if cx and ch:
        mult = f"cmul({hval}, xv)"
    else:
        # float2 * float (or float * float) is componentwise in MSL
        mult = f"{hval} * xv"
    if complex_out:
        acct = "float2"
        zero = "float2(0.0f, 0.0f)"
        store = ("long o = ((long)b * n_out + i) * 2; "
                 "out[o] = acc.x; out[o + 1] = acc.y;")
    else:
        acct = "float"
        zero = "0.0f"
        store = "out[(long)b * n_out + i] = acc;"
    common = dict(XW=2 if cx else 1, XFETCH=xfetch, ACCT=acct, ZERO=zero,
                  MULT=mult, STORE=store)
    if direct:
        src = _SRC_DIRECT.format(HFETCH=hfetch, **common)
    else:
        src = _SRC.format(HT=ht, LOADH=loadh,
                          TILE=_TILE_CPLX if ch else _TILE_REAL, **common)
    return mx.fast.metal_kernel(
        name=f"mlx_signal_upfirdn_{int(cx)}{int(ch)}{int(direct)}",
        input_names=["x", "h", "params"],
        output_names=["out"],
        source=src,
        header=_HEADER,
    )


def upfirdn_gpu(x2d: mx.array, h: mx.array, up: int, down: int, n_out: int) -> mx.array:
    """Run the polyphase kernel. x2d: (B, n_in) f32/c64; h: (n_taps,) f32/c64."""
    B, n_in = x2d.shape
    n_taps = h.shape[0]
    cx = x2d.dtype == mx.complex64
    ch = h.dtype == mx.complex64
    complex_out = cx or ch

    xin = mx.view(x2d, mx.float32) if cx else x2d
    hin = mx.view(h, mx.float32) if ch else h
    params = mx.array([n_in, n_out, up, down, n_taps], dtype=mx.int32)
    width = 2 * n_out if complex_out else n_out
    (out,) = _kernel(cx, ch, up > _DIRECT_UP)(
        inputs=[xin, hin, params],
        grid=(n_out, B, 1),
        threadgroup=(min(256, max(32, n_out)), 1, 1),
        output_shapes=[(B, width)],
        output_dtypes=[mx.float32],
    )
    if complex_out:
        out = mx.view(out, mx.complex64)
    return out
