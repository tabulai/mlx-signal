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

Apple GPUs have no hardware integer divide: the 64-bit div/mod in the
per-thread polyphase setup is emulated in hundreds of instructions and, with
one thread per output sample, dominates the kernel at high ``up`` (measured
~3x of the whole 147/160 resampler kernel). Whenever every index fits 32-bit
range the dispatch therefore uses a uint-arithmetic direct-read variant; the
long-arithmetic kernels remain the fallback for larger-than-2^31 geometries,
and the taps-staged tiled kernel keeps one measured niche — pure decimation
(``up == 1``) with a cache-line-or-wider gather stride and a long filter,
where every thread walks all taps and staging them beats re-fetching through
a cache already thrashed by the strided signal reads. All variants accumulate
taps in the same ascending-k order, so results are bit-identical and the
routing is purely a performance choice.
"""

from __future__ import annotations

import functools

import mlx.core as mx

_TILE_REAL = 2048  # taps per threadgroup-memory tile (8 KB as float)
_TILE_CPLX = 1024  # 8 KB as float2
#: on the long-arithmetic fallback path only: above this upsampling factor
#: each tap is reused by too few threads per threadgroup for staging to pay;
#: read taps straight through the device cache
_DIRECT_UP = 32
#: ceiling for the quantities the u32 dispatch guard bounds: flat x and out
#: indices (including their float2 doubling) and the tap-window geometry
#: i*down plus the loop's + up overshoot. 2^31 rather than 2^32 on purpose:
#: overflow here is silent wrong output, not an error, so keep a full
#: factor-of-two margin. The one index the guard does NOT bound is the
#: complex-taps fetch 2k+1, which may pass 2^31 and is safe only because
#: int32 n_taps caps it below 2^32 — hence its explicit uint indexing.
_U32_LIMIT = 1 << 31
#: pure-decimation staging niche (measured on M4 Max): at up == 1 the tiled
#: kernel still wins once the per-tap signal gather stride reaches a cache
#: line (down * floats-per-element >= 16, i.e. 64 B) AND the filter is long
#: enough to amortize the tile barriers. Mispredicting costs at most ~1.3x
#: either way; the u32 direct wins everywhere else by 1.3-4x.
_STAGE_STRIDE_FLOATS = 16
_STAGE_MIN_TAPS = 128

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


# Same polyphase geometry as _SRC_DIRECT with the div/mod pair — the entire
# cost being removed — in uint32. The window bookkeeping stays in signed int
# (perf-identical, measured) so that n_taps == 0 empties the window via
# hi = -1 exactly like the long kernels, instead of a wrapped uint clamp.
# k starts exactly at p + up*t_lo, so the (p - lo) % up remainder and the
# (k - p) / up division of the tiled kernel are identities here and disappear.
_SRC_DIRECT_U32 = """
    uint i = thread_position_in_grid.x;
    uint b = thread_position_in_grid.y;
    int n_in   = params[0];
    int n_out  = params[1];
    int n_taps = params[4];
    if ((int)i >= n_out) return;
    uint up   = (uint)params[2];
    uint down = (uint)params[3];

    uint xbase = b * {XW}u * (uint)n_in;
    uint m = i * down;
    int p = (int)(m % up);
    int j = (int)(m / up);
    int t_lo = (j >= n_in) ? (j - n_in + 1) : 0;
    int k  = p + (int)up * t_lo;   // may exceed hi: the k <= hi test guards
    int hi = (int)m;               // p + up*j == m by the div/mod identity
    if (hi > n_taps - 1) hi = n_taps - 1;
    {ACCT} acc = {ZERO};
    int jj = j - t_lo;
    for (; k <= hi; k += (int)up, --jj) {{
        {HFETCH}
        {XFETCH}
        acc += {MULT};
    }}
    {STORE}
"""


@functools.lru_cache(maxsize=16)
def _kernel(cx: bool, ch: bool, direct: bool, u32: bool = False):
    ht = "float2" if ch else "float"
    complex_out = cx or ch
    loadh = (
        "sh[t] = float2(h[2 * (tile + t)], h[2 * (tile + t) + 1]);"
        if ch else "sh[t] = h[tile + t];"
    )
    # complex taps are the one index the u32 dispatch guard does not bound:
    # 2k+1 can pass 2^31 (k caps at int32 n_taps - 1, so it stays < 2^32),
    # which is safe in uint indexing and signed-overflow UB in int
    hfetch = (
        ("float2 hv = float2(h[2u * (uint)k], h[2u * (uint)k + 1u]);" if u32
         else "float2 hv = float2(h[2 * k], h[2 * k + 1]);")
        if ch else "float hv = h[k];"
    )
    xfetch = (
        "float2 xv = float2(x[xbase + 2 * jj], x[xbase + 2 * jj + 1]);"
        if cx else "float xv = x[xbase + jj];"
    )
    hval = "hv" if (direct or u32) else "sh[k - tile]"
    if cx and ch:
        mult = f"cmul({hval}, xv)"
    else:
        # float2 * float (or float * float) is componentwise in MSL
        mult = f"{hval} * xv"
    if complex_out:
        acct = "float2"
        zero = "float2(0.0f, 0.0f)"
        if u32:
            store = ("uint o = (b * (uint)n_out + i) * 2u; "
                     "out[o] = acc.x; out[o + 1] = acc.y;")
        else:
            store = ("long o = ((long)b * n_out + i) * 2; "
                     "out[o] = acc.x; out[o + 1] = acc.y;")
    else:
        acct = "float"
        zero = "0.0f"
        if u32:
            store = "out[b * (uint)n_out + i] = acc;"
        else:
            store = "out[(long)b * n_out + i] = acc;"
    common = dict(XW=2 if cx else 1, XFETCH=xfetch, ACCT=acct, ZERO=zero,
                  MULT=mult, STORE=store)
    if u32:
        src = _SRC_DIRECT_U32.format(HFETCH=hfetch, **common)
    elif direct:
        src = _SRC_DIRECT.format(HFETCH=hfetch, **common)
    else:
        src = _SRC.format(HT=ht, LOADH=loadh,
                          TILE=_TILE_CPLX if ch else _TILE_REAL, **common)
    return mx.fast.metal_kernel(
        name=f"mlx_signal_upfirdn_{int(cx)}{int(ch)}{int(direct)}"
             + ("_u32" if u32 else ""),
        input_names=["x", "h", "params"],
        output_names=["out"],
        source=src,
        header=_HEADER,
    )


def _fits_u32(B: int, n_in: int, n_out: int, up: int, down: int,
              cx: bool, complex_out: bool) -> bool:
    """True when the flat x/out indices and window geometry fit u32 safely.

    The float32-view factors are the trap: complex input doubles the flat x
    index and complex *output* (complex signal OR complex taps) doubles the
    store offset, so each side gets its own factor. The geometry bound covers
    m = i*down plus the loop's k = hi + up overshoot on its final test.
    Complex taps also double the flat h index; that one is deliberately not
    bounded here — the kernel fetches it in uint, where int32 n_taps caps it
    below 2^32.
    """
    xw = 2 if cx else 1
    ow = 2 if complex_out else 1
    return (B * n_in * xw < _U32_LIMIT
            and B * n_out * ow < _U32_LIMIT
            and n_out * down + 2 * up < _U32_LIMIT)


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
    xw = 2 if cx else 1
    stage_wins = (up == 1 and n_taps >= _STAGE_MIN_TAPS
                  and down * xw >= _STAGE_STRIDE_FLOATS)
    if stage_wins or not _fits_u32(B, n_in, n_out, up, down, cx, complex_out):
        kern = _kernel(cx, ch, direct=up > _DIRECT_UP)
    else:
        kern = _kernel(cx, ch, direct=True, u32=True)
    (out,) = kern(
        inputs=[xin, hin, params],
        grid=(n_out, B, 1),
        threadgroup=(min(256, max(32, n_out)), 1, 1),
        output_shapes=[(B, width)],
        output_dtypes=[mx.float32],
    )
    if complex_out:
        out = mx.view(out, mx.complex64)
    return out
