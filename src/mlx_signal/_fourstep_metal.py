"""Fused three-pass Metal FFT for large power-of-two lengths.

The composed four-step in ``_fourstep`` moves ~2x the bytes the transform
needs (materialized transposes around each sub-FFT plus a full-size twiddle
table). This kernel pipeline runs the same mathematics in three fused passes
— each one: tiled coalesced load, a multi-column radix-2 FFT in threadgroup
memory, on-the-fly inter-pass twiddles, tiled store — moving read+write only.
Measured on M4 Max (complex64, device-resident): 1.8-2.0x over the composed
path at 2^23 and 4-5.5x at 2^26 (where the composed transposes fall off a
cache cliff), while being ~2-3x MORE accurate against float64 numpy (three
table-free radix-2 passes beat MLX's own large-n FFT error). The inverse
costs the same as the forward: conjugated stage tables, positive twiddle
sign, and 1/N folded into pass 1 — no conjugation passes.

Decomposition (the ``_fourstep`` identity applied twice), n = R3*R2*R1 with
power-of-two digits <= 1024:

    n = (m3*R2 + m2)*R1 + r          k = kr*(R2*R3) + km2*R3 + km3
    pass 1: FFT_R3 over m3 -> km3, twiddle W_M^(m2*km3),  M = R2*R3
    pass 2: FFT_R2 over m2 -> km2, twiddle W_N^(r*(km2*R3 + km3))
    pass 3: FFT_R1 over r  -> kr, permuted final write

Intermediate layouts keep km3 innermost from pass 1 on, so every strided
access is a CT-wide contiguous chunk staged through threadgroup memory:

    x  [m3][m2][r] -> S1 [m2][r][km3] -> S2 [km2][r][km3] -> out [kr][km2][km3]

The threadgroup FFT is an in-place radix-2 DIT with the bit reversal folded
into the store index of the (already strided) global load: single buffer, so
the 16 KB threadgroup budget holds L*CT = 2048 points — double the tile width
of a ping-pong Stockham, which measured 15-25% slower here. (32 KB is the
known-unbuildable boundary; the documented mirrored-read compiler crash needs
the buf[k]/buf[M-k] pattern and does not fire for the disjoint DIT butterfly.
``half`` is a reserved MSL type — do not use it as an identifier.)

Twiddle phases are exact: integer products reduced by a power-of-two mask in
uint (max product < n <= 2^26 < 2^32), converted to float32 and scaled, then
``precise::sin/cos`` — the same recipe as ``_fourstep._twiddle``, immune to
any fast-math default in kernel compilation.
"""

from __future__ import annotations

import functools
import math

import mlx.core as mx
import numpy as np

#: served length range: power-of-two n with 2^21 <= n <= 2^26 (the measured/
#: verified envelope; safe lengths never reach here, larger fall back to the
#: composed path)
_MIN_LOG2 = 21
_MAX_LOG2 = 26
#: L*CT threadgroup points per pass (16 KB as single-buffered float2)
_BUDGET = 2048

_HDR = """
static inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
"""

# in-place multi-column radix-2 DIT; expects bit-reversed rows in buf0 and
# leaves natural-order results in A (== buf0)
_FFT_BLOCK_DIT = """
    threadgroup float2* A = buf0;
    {
        for (int s = 0; s < {S}; s++) {
            int hlf = 1 << s;
            for (int p = (int)tid; p < ({L} / 2) * {CT}; p += {T}) {
                int u = p % {CT};
                int q = p / {CT};
                int j = q / hlf;
                int t = q - j * hlf;
                int i0 = (j * 2 * hlf + t) * {CT} + u;
                int i1 = i0 + hlf * {CT};
                int ti = t * ({L} / (2 * hlf));
                float2 w = float2(twt[2 * ti], twt[2 * ti + 1]);
                float2 x0 = A[i0];
                float2 x1 = cmul(w, A[i1]);
                A[i0] = x0 + x1;
                A[i1] = x0 - x1;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
"""

_P1_SRC = """
    uint tid = thread_index_in_threadgroup;
    int gy = (int)threadgroup_position_in_grid.y;
    const int tiles_r = {R1} / {CT};
    int b   = gy / ({R2} * tiles_r);
    int rem = gy - b * ({R2} * tiles_r);
    int m2  = rem / tiles_r;
    int r0  = (rem - m2 * tiles_r) * {CT};
    const device float2* xin = (const device float2*)x;
    device float2* sout = (device float2*)s1;
    long base = (long)b * {N};

    threadgroup float2 buf0[{L} * {CT}];

    for (int p = (int)tid; p < {L} * {CT}; p += {T}) {
        int u  = p % {CT};
        int m3 = p / {CT};
        buf0[RIDX(m3) * {CT} + u] = xin[base + ((long)m3 * {R2} + m2) * {R1} + r0 + u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
{FFT}
    for (int p = (int)tid; p < {L} * {CT}; p += {T}) {
        int km3 = p % {L};
        int u   = p / {L};
        float2 v = A[km3 * {CT} + u];
        uint ph = ((uint)m2 * (uint)km3) & {MMASK}u;
        float th = {W1}f * (float)ph;
        float cw = precise::cos(th);
        float sw = precise::sin(th);
        float2 vv = cmul(float2(cw, sw), v);
        sout[base + ((long)m2 * {R1} + (r0 + u)) * {R3} + km3] =
            float2(vv.x * {SCALE}f, vv.y * {SCALE}f);
    }
"""

_P2_SRC = """
    uint tid = thread_index_in_threadgroup;
    int gy = (int)threadgroup_position_in_grid.y;
    const int tiles_k = {R3} / {CT};
    int b   = gy / ({R1} * tiles_k);
    int rem = gy - b * ({R1} * tiles_k);
    int r   = rem / tiles_k;
    int k0  = (rem - r * tiles_k) * {CT};
    const device float2* xin = (const device float2*)s1;
    device float2* sout = (device float2*)s2;
    long base = (long)b * {N};

    threadgroup float2 buf0[{L} * {CT}];

    for (int p = (int)tid; p < {L} * {CT}; p += {T}) {
        int u  = p % {CT};
        int m2 = p / {CT};
        buf0[RIDX(m2) * {CT} + u] = xin[base + ((long)m2 * {R1} + r) * {R3} + k0 + u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
{FFT}
    for (int p = (int)tid; p < {L} * {CT}; p += {T}) {
        int u   = p % {CT};
        int km2 = p / {CT};
        float2 v = A[km2 * {CT} + u];
        uint km = (uint)km2 * {R3}u + (uint)(k0 + u);
        uint ph = ((uint)r * km) & {NMASK}u;
        float th = {W2}f * (float)ph;
        float cw = precise::cos(th);
        float sw = precise::sin(th);
        sout[base + ((long)km2 * {R1} + r) * {R3} + k0 + u] = cmul(float2(cw, sw), v);
    }
"""

_P3_SRC = """
    uint tid = thread_index_in_threadgroup;
    int gy = (int)threadgroup_position_in_grid.y;
    const int tiles_k = {R3} / {CT};
    int b   = gy / ({R2} * tiles_k);
    int rem = gy - b * ({R2} * tiles_k);
    int km2 = rem / tiles_k;
    int k0  = (rem - km2 * tiles_k) * {CT};
    const device float2* xin = (const device float2*)s2;
    device float2* xout = (device float2*)out;
    long base = (long)b * {N};

    threadgroup float2 buf0[{L} * {CT}];

    for (int p = (int)tid; p < {L} * {CT}; p += {T}) {
        int u  = p % {CT};
        int rr = p / {CT};
        buf0[RIDX(rr) * {CT} + u] = xin[base + ((long)km2 * {R1} + rr) * {R3} + k0 + u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
{FFT}
    for (int p = (int)tid; p < {L} * {CT}; p += {T}) {
        int u  = p % {CT};
        int kr = p / {CT};
        xout[base + (long)kr * {MM} + (long)km2 * {R3} + k0 + u] = A[kr * {CT} + u];
    }
"""


def _subst(src: str, **kw) -> str:
    # token substitution rather than str.format: the MSL braces stay sane
    for k, v in kw.items():
        src = src.replace("{" + k + "}", str(v))
    return src


def eligible(n: int) -> bool:
    """True if the fused pipeline serves this transform length."""
    if not (mx.metal.is_available() and hasattr(mx, "view")):
        return False
    if n & (n - 1):
        return False
    e = n.bit_length() - 1
    return _MIN_LOG2 <= e <= _MAX_LOG2


@functools.lru_cache(maxsize=16)
def _digits(n: int) -> tuple[int, int, int]:
    """(R3, R2, R1) for n = 2^e: measured-best splits, mild sensitivity.

    2^21 prefers the balanced (128, 128, 128); from 2^22 up, (2^(e-16),
    256, 256) matches the measured winners at 2^23 and 2^26 exactly.
    """
    e = n.bit_length() - 1
    if e == 21:
        return (128, 128, 128)
    return (1 << (e - 16), 256, 256)


def _ct(L: int, cap: int) -> int:
    return max(1, min(_BUDGET // L, cap))


@functools.lru_cache(maxsize=64)
def _stage_table(L: int, sign: int) -> mx.array:
    tw = np.exp(sign * 2j * np.pi * np.arange(L // 2) / L).astype(np.complex64)
    t = mx.array(np.ascontiguousarray(tw.view(np.float32)))
    mx.eval(t)
    return t


@functools.lru_cache(maxsize=32)
def _kernels(R3: int, R2: int, R1: int, sign: int):
    """sign = -1 forward, +1 inverse (inverse folds 1/N into pass 1)."""
    N = R1 * R2 * R3
    M = R2 * R3
    ct1 = _ct(R3, R1)
    ct2 = _ct(R2, R3)
    ct3 = _ct(R1, R3)
    scale = 1.0 if sign < 0 else 1.0 / N
    kerns = []
    for pid, (src, L, CT) in enumerate(
        [(_P1_SRC, R3, ct1), (_P2_SRC, R2, ct2), (_P3_SRC, R1, ct3)], start=1
    ):
        T = min(256, max(32, (L * CT) // 2))
        S = int(math.log2(L))
        fft = _subst(_FFT_BLOCK_DIT, L=L, CT=CT, T=T, S=S)
        ridx = f"#define RIDX(i) ((int)(reverse_bits((uint)(i)) >> (32 - {S})))"
        body = _subst(
            src,
            L=L, CT=CT, T=T, FFT=fft,
            R1=R1, R2=R2, R3=R3, N=N, MM=M,
            MMASK=M - 1, NMASK=N - 1,
            W1=f"{sign * 2.0 * np.pi / M:.10e}",
            W2=f"{sign * 2.0 * np.pi / N:.10e}",
            SCALE=f"{scale:.10e}",
        )
        names = {
            1: (["x", "twt"], ["s1"]),
            2: (["s1", "twt"], ["s2"]),
            3: (["s2", "twt"], ["out"]),
        }[pid]
        kern = mx.fast.metal_kernel(
            name=f"mlx_signal_fft3_p{pid}_{R3}_{R2}_{R1}_{'f' if sign < 0 else 'i'}",
            input_names=names[0],
            output_names=names[1],
            source=body,
            header=_HDR + "\n" + ridx + "\n",
        )
        kerns.append((kern, L, CT, T))
    return kerns, (ct1, ct2, ct3)


def fft3_last(a: mx.array, inverse: bool = False) -> mx.array:
    """FFT/IFFT along the last axis of complex64 ``a`` (eligible length)."""
    n = a.shape[-1]
    batch_shape = a.shape[:-1]
    a2 = a.reshape(-1, n)
    B = a2.shape[0]
    R3, R2, R1 = _digits(n)
    sign = 1 if inverse else -1
    (k1, k2, k3), (ct1, ct2, ct3) = _kernels(R3, R2, R1, sign)
    xf = mx.view(a2, mx.float32)  # (B, 2n), free reinterpretation
    (s1,) = k1[0](
        inputs=[xf, _stage_table(R3, sign)],
        grid=(k1[3], B * R2 * (R1 // ct1), 1),
        threadgroup=(k1[3], 1, 1),
        output_shapes=[(B, 2 * n)],
        output_dtypes=[mx.float32],
    )
    (s2,) = k2[0](
        inputs=[s1, _stage_table(R2, sign)],
        grid=(k2[3], B * R1 * (R3 // ct2), 1),
        threadgroup=(k2[3], 1, 1),
        output_shapes=[(B, 2 * n)],
        output_dtypes=[mx.float32],
    )
    (o,) = k3[0](
        inputs=[s2, _stage_table(R1, sign)],
        grid=(k3[3], B * R2 * (R3 // ct3), 1),
        threadgroup=(k3[3], 1, 1),
        output_shapes=[(B, 2 * n)],
        output_dtypes=[mx.float32],
    )
    return mx.view(o, mx.complex64).reshape(batch_shape + (n,))
