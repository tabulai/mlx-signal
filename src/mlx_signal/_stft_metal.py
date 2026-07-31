"""Fused frame+detrend+window+rfft Metal kernel for the spectral hot path.

One threadgroup per segment: load the segment straight from the (strided)
signal, optionally subtract its mean (detrend='constant'), apply the window,
pack even/odd samples into M = nperseg/2 complex points, run a radix-2
Stockham FFT in threadgroup memory with host-precomputed twiddle tables, and
untangle to the length-nperseg real-input rfft (M+1 bins). The whole per-
segment pipeline of welch/stft touches the signal once and writes the spectrum
once — there is no materialized frames array at all.

Two output variants: complex spectrum (stft/csd) written as interleaved float
pairs and reinterpreted with ``mx.view``, or |X|^2 float32 directly (welch's
same-signal PSD), which halves the write traffic and removes the separate
power pass.

Eligibility (checked by :func:`eligible`): Metal available, real float32
input, one-sided FFT, nfft == nperseg, power-of-two nperseg in [64, 2048]
(threadgroup memory bound), detrend in (False, 'constant'). Everything else
uses the composed MLX path in ``spectral._fft_frames``.
"""

from __future__ import annotations

import functools
import math

import mlx.core as mx
import numpy as np

_MIN_N = 64
_MAX_N = 2048  # 2 * (N/2) complex64 threadgroup buffers = 16 KB at N=2048

_LOAD_PLAIN = """
    for (int i = tid; i < {M}; i += {T}) {{
        buf0[i] = float2(xseg[2 * i] * win[2 * i], xseg[2 * i + 1] * win[2 * i + 1]);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""

_LOAD_DETREND = """
    threadgroup float red[{T}];
    float acc = 0.0f;
    for (int i = tid; i < {M}; i += {T}) {{
        float a = xseg[2 * i];
        float b = xseg[2 * i + 1];
        acc += a + b;
        buf0[i] = float2(a, b);
    }}
    red[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int off = {T} / 2; off > 0; off >>= 1) {{
        if ((int)tid < off) red[tid] += red[tid + off];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float mean = red[0] / (float){N};
    for (int i = tid; i < {M}; i += {T}) {{
        float2 v = buf0[i];
        buf0[i] = float2((v.x - mean) * win[2 * i], (v.y - mean) * win[2 * i + 1]);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""

_OUT_COMPLEX = """
    device float* orow = out + (long)seg * (2 * ({M} + 1));
    for (int k = tid; k <= {M}; k += {T}) {{
        float2 Xk = untangle(k, A, utf);
        orow[2 * k]     = Xk.x;
        orow[2 * k + 1] = Xk.y;
    }}
"""

_OUT_POWER = """
    device float* orow = out + (long)seg * ({M} + 1);
    for (int k = tid; k <= {M}; k += {T}) {{
        float2 Xk = untangle(k, A, utf);
        orow[k] = Xk.x * Xk.x + Xk.y * Xk.y;
    }}
"""

_HEADER = """
static inline float2 cmul(float2 a, float2 b) {{
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}}

// rfft bin k of the real length-N signal from the FFT Z of its packed
// even/odd complex form: X[k] = E[k] + e^(-2*pi*i*k/N) * O[k]
static inline float2 untangle(int k, threadgroup const float2* Z,
                              const device float2* utf) {{
    if (k == 0) {{
        return float2(Z[0].x + Z[0].y, 0.0f);
    }}
    if (k == {M}) {{
        return float2(Z[0].x - Z[0].y, 0.0f);
    }}
    float2 zk  = Z[k];
    float2 zmk = Z[{M} - k];
    float2 ze = float2(0.5f * (zk.x + zmk.x), 0.5f * (zk.y - zmk.y));
    float2 zo = float2(0.5f * (zk.x - zmk.x), 0.5f * (zk.y + zmk.y));
    float2 wzo = cmul(utf[k], zo);
    return float2(ze.x + wzo.y, ze.y - wzo.x);  // ze + (-i) * wzo
}}
"""

_SRC = """
    uint tid = thread_index_in_threadgroup;
    uint seg = threadgroup_position_in_grid.y;
    int n      = params[0];
    int hop    = params[1];
    int nseg   = params[2];
    int total  = params[3];
    if ((int)seg >= total) return;

    int row  = (int)seg / nseg;
    int s_in = (int)seg % nseg;
    const device float* xseg = x + (long)row * n + (long)s_in * hop;
    const device float2* twf = (const device float2*)tw;
    const device float2* utf = (const device float2*)ut;

    threadgroup float2 buf0[{M}];
    threadgroup float2 buf1[{M}];

{LOAD_BLOCK}

    threadgroup float2* A = buf0;
    threadgroup float2* B = buf1;
    int l = {M} / 2;
    int stride = 1;
    int m = 1;
    for (int s = 0; s < {S}; s++) {{
        for (int p = tid; p < {M} / 2; p += {T}) {{
            int j = p / m;
            int k = p - j * m;
            float2 c0 = A[k + j * m];
            float2 c1 = A[k + j * m + l * m];
            float2 d = c0 - c1;
            B[k + 2 * j * m]     = c0 + c1;
            B[k + 2 * j * m + m] = cmul(twf[j * stride], d);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float2* tmp = A; A = B; B = tmp;
        l >>= 1;
        stride <<= 1;
        m <<= 1;
    }}

{OUT_BLOCK}
"""


def eligible(x: mx.array, nperseg: int, nfft: int, sides: str, detrend) -> bool:
    """True if the fused kernel can compute this windowed rfft."""
    if not (mx.metal.is_available() and hasattr(mx, "view")):
        return False
    if sides != "onesided" or x.dtype != mx.float32:
        return False
    if nfft != nperseg:
        return False
    if nperseg < _MIN_N or nperseg > _MAX_N or nperseg & (nperseg - 1):
        return False
    return detrend in (False, None, "constant", "c")


@functools.lru_cache(maxsize=32)
def _kernel(N: int, detrend: bool, power: bool):
    M = N // 2
    S = int(math.log2(M))
    T = min(256, max(32, M // 2))
    load = (_LOAD_DETREND if detrend else _LOAD_PLAIN).format(N=N, M=M, T=T)
    out = (_OUT_POWER if power else _OUT_COMPLEX).format(M=M, T=T)
    src = _SRC.format(M=M, S=S, T=T, LOAD_BLOCK=load, OUT_BLOCK=out)
    kern = mx.fast.metal_kernel(
        name=f"mlx_signal_stft_{N}_{int(detrend)}_{int(power)}",
        input_names=["x", "win", "tw", "ut", "params"],
        output_names=["out"],
        source=src,
        header=_HEADER.format(M=M),
    )
    return kern, T, M


@functools.lru_cache(maxsize=32)
def _tables(N: int):
    M = N // 2
    tw = np.exp(-2j * np.pi * np.arange(M // 2) / M).astype(np.complex64)
    ut = np.exp(-1j * np.pi * np.arange(M + 1) / M).astype(np.complex64)
    return (
        mx.array(np.ascontiguousarray(tw.view(np.float32))),
        mx.array(np.ascontiguousarray(ut.view(np.float32))),
    )


def rfft_frames(x: mx.array, win: mx.array, nperseg: int, hop: int, detrend,
                power: bool) -> mx.array:
    """Windowed, optionally mean-detrended, framed rfft of the last axis.

    Returns ``(..., nseg, nperseg//2 + 1)`` — complex64, or float32 |X|^2 when
    ``power`` is set. ``win`` must already carry any scale factor.
    """
    n = x.shape[-1]
    nseg = (n - nperseg) // hop + 1
    batch_shape = x.shape[:-1]
    x2 = x.reshape(-1, n)
    total = x2.shape[0] * nseg

    do_detrend = detrend in ("constant", "c")
    kern, T, M = _kernel(nperseg, do_detrend, power)
    tw, ut = _tables(nperseg)
    params = mx.array([n, hop, nseg, total], dtype=mx.int32)

    width = (M + 1) if power else 2 * (M + 1)
    (out,) = kern(
        inputs=[x2, win, tw, ut, params],
        grid=(T, total, 1),
        threadgroup=(T, 1, 1),
        output_shapes=[(total, width)],
        output_dtypes=[mx.float32],
    )
    if not power:
        out = mx.view(out, mx.complex64)
    return out.reshape(batch_shape + (nseg, M + 1))


# ---------------------------------------------------------------------------
# two-signal variant: csd / coherence in one sweep
# ---------------------------------------------------------------------------

_PAIR_FFT = """
    {{
        threadgroup float2* A = buf0;
        threadgroup float2* B = buf1;
        int l = {M} / 2;
        int stride = 1;
        int m = 1;
        for (int s = 0; s < {S}; s++) {{
            for (int p = tid; p < {M} / 2; p += {T}) {{
                int j = p / m;
                int k = p - j * m;
                float2 c0 = A[k + j * m];
                float2 c1 = A[k + j * m + l * m];
                float2 d = c0 - c1;
                B[k + 2 * j * m]     = c0 + c1;
                B[k + 2 * j * m + m] = cmul(twf[j * stride], d);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup float2* tmp = A; A = B; B = tmp;
            l >>= 1;
            stride <<= 1;
            m <<= 1;
        }}
        Z = A;
    }}
"""


def _pair_load(sig: str, N: int, M: int, T: int, detrend: bool) -> str:
    if not detrend:
        return f"""
    for (int i = tid; i < {M}; i += {T}) {{
        buf0[i] = float2({sig}[2 * i] * win[2 * i], {sig}[2 * i + 1] * win[2 * i + 1]);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""
    return f"""
    {{
        float acc = 0.0f;
        for (int i = tid; i < {M}; i += {T}) {{
            float a = {sig}[2 * i];
            float b = {sig}[2 * i + 1];
            acc += a + b;
            buf0[i] = float2(a, b);
        }}
        red[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int off = {T} / 2; off > 0; off >>= 1) {{
            if ((int)tid < off) red[tid] += red[tid + off];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float mean = red[0] / (float){N};
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < {M}; i += {T}) {{
            float2 v = buf0[i];
            buf0[i] = float2((v.x - mean) * win[2 * i], (v.y - mean) * win[2 * i + 1]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
"""


_PAIR_OUT_CSD = """
    for (int i = 0; i < {KMAX}; ++i) {{
        int k = (int)tid + i * {T};
        if (k > {M}) continue;
        float2 Xk = hold[i];
        float2 Yk = untangle(k, Z, utf);
        // conj(X) * Y
        float2 P = float2(Xk.x * Yk.x + Xk.y * Yk.y, Xk.x * Yk.y - Xk.y * Yk.x);
        device float* orow = out + (long)seg * (2 * ({M} + 1));
        orow[2 * k]     = P.x;
        orow[2 * k + 1] = P.y;
    }}
"""

_PAIR_OUT_TRIO = """
    for (int i = 0; i < {KMAX}; ++i) {{
        int k = (int)tid + i * {T};
        if (k > {M}) continue;
        float2 Xk = hold[i];
        float2 Yk = untangle(k, Z, utf);
        float2 P = float2(Xk.x * Yk.x + Xk.y * Yk.y, Xk.x * Yk.y - Xk.y * Yk.x);
        pxx[(long)seg * ({M} + 1) + k] = Xk.x * Xk.x + Xk.y * Xk.y;
        pyy[(long)seg * ({M} + 1) + k] = Yk.x * Yk.x + Yk.y * Yk.y;
        device float* orow = pxy + (long)seg * (2 * ({M} + 1));
        orow[2 * k]     = P.x;
        orow[2 * k + 1] = P.y;
    }}
"""

_PAIR_SRC = """
    uint tid = thread_index_in_threadgroup;
    uint seg = threadgroup_position_in_grid.y;
    int n      = params[0];
    int hop    = params[1];
    int nseg   = params[2];
    int total  = params[3];
    if ((int)seg >= total) return;

    int row  = (int)seg / nseg;
    int s_in = (int)seg % nseg;
    const device float* xseg = xa + (long)row * n + (long)s_in * hop;
    const device float* yseg = xb + (long)row * n + (long)s_in * hop;
    const device float2* twf = (const device float2*)tw;
    const device float2* utf = (const device float2*)ut;

    threadgroup float2 buf0[{M}];
    threadgroup float2 buf1[{M}];
    threadgroup float red[{T}];
    threadgroup float2* Z;
    float2 hold[{KMAX}];

{LOAD_X}
{FFT}
    for (int i = 0; i < {KMAX}; ++i) {{
        int k = (int)tid + i * {T};
        hold[i] = (k <= {M}) ? untangle(k, Z, utf) : float2(0.0f, 0.0f);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

{LOAD_Y}
{FFT}
{OUT_BLOCK}
"""


@functools.lru_cache(maxsize=32)
def _pair_kernel(N: int, detrend: bool, trio: bool):
    M = N // 2
    S = int(math.log2(M))
    T = min(256, max(32, M // 2))
    kmax = -(-(M + 1) // T)
    fft = _PAIR_FFT.format(M=M, S=S, T=T)
    out = (_PAIR_OUT_TRIO if trio else _PAIR_OUT_CSD).format(M=M, T=T, KMAX=kmax)
    src = _PAIR_SRC.format(
        M=M, T=T, KMAX=kmax, FFT=fft,
        LOAD_X=_pair_load("xseg", N, M, T, detrend),
        LOAD_Y=_pair_load("yseg", N, M, T, detrend),
        OUT_BLOCK=out,
    )
    kern = mx.fast.metal_kernel(
        name=f"mlx_signal_csd_{N}_{int(detrend)}_{int(trio)}",
        input_names=["xa", "xb", "tw", "ut", "win", "params"],
        output_names=["pxx", "pyy", "pxy"] if trio else ["out"],
        source=src,
        header=_HEADER.format(M=M),
    )
    return kern, T, M


def _pair_launch(x, y, win, nperseg, hop, detrend, trio):
    n = x.shape[-1]
    nseg = (n - nperseg) // hop + 1
    batch_shape = x.shape[:-1]
    x2 = x.reshape(-1, n)
    y2 = y.reshape(-1, n)
    total = x2.shape[0] * nseg

    do_detrend = detrend in ("constant", "c")
    kern, T, M = _pair_kernel(nperseg, do_detrend, trio)
    tw, ut = _tables(nperseg)
    params = mx.array([n, hop, nseg, total], dtype=mx.int32)

    if trio:
        shapes = [(total, M + 1), (total, M + 1), (total, 2 * (M + 1))]
        dtypes = [mx.float32, mx.float32, mx.float32]
    else:
        shapes = [(total, 2 * (M + 1))]
        dtypes = [mx.float32]
    outs = kern(
        inputs=[x2, y2, tw, ut, win, params],
        grid=(T, total, 1),
        threadgroup=(T, 1, 1),
        output_shapes=shapes,
        output_dtypes=dtypes,
    )
    if trio:
        pxx = outs[0].reshape(batch_shape + (nseg, M + 1))
        pyy = outs[1].reshape(batch_shape + (nseg, M + 1))
        pxy = mx.view(outs[2], mx.complex64).reshape(batch_shape + (nseg, M + 1))
        return pxx, pyy, pxy
    return mx.view(outs[0], mx.complex64).reshape(batch_shape + (nseg, M + 1))


def csd_frames(x, y, win, nperseg, hop, detrend):
    """conj(F(x_seg)) * F(y_seg) for all segments, one fused sweep."""
    return _pair_launch(x, y, win, nperseg, hop, detrend, trio=False)


def coherence_frames(x, y, win, nperseg, hop, detrend):
    """(|Fx|^2, |Fy|^2, conj(Fx)*Fy) per segment in one fused sweep."""
    return _pair_launch(x, y, win, nperseg, hop, detrend, trio=True)


# ---------------------------------------------------------------------------
# inverse: per-segment inverse rfft + windowing, then gather-based overlap-add
# ---------------------------------------------------------------------------

_ISTFT_SRC = """
    uint tid = thread_index_in_threadgroup;
    uint seg = threadgroup_position_in_grid.y;
    int total = params[0];
    if ((int)seg >= total) return;

    const device float2* xrow = ((const device float2*)zin) + (long)seg * ({M} + 1);
    const device float2* twf = (const device float2*)tw;
    const device float2* utf = (const device float2*)ut;

    threadgroup float2 buf0[{M}];
    threadgroup float2 buf1[{M}];
    threadgroup float2* Z;

    // rebuild conj(Z[k]) of the packed even/odd spectrum from X[0..M]:
    // E = (X[k] + conj(X[M-k]))/2, O = conj(ut[k]) * (X[k] - conj(X[M-k]))/2
    for (int k = tid; k < {M}; k += {T}) {{
        float2 Xk  = xrow[k];
        float2 Xmk = xrow[{M} - k];
        if (k == 0) {{
            // irfft semantics: DC and Nyquist bins are taken as real
            Xk.y = 0.0f;
            Xmk.y = 0.0f;
        }}
        float2 E  = float2(0.5f * (Xk.x + Xmk.x), 0.5f * (Xk.y - Xmk.y));
        float2 wO = float2(0.5f * (Xk.x - Xmk.x), 0.5f * (Xk.y + Xmk.y));
        float2 u  = utf[k];
        float2 O  = cmul(float2(u.x, -u.y), wO);
        buf0[k] = float2(E.x - O.y, -(E.y + O.x));  // conj(E + i*O)
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

{FFT}

    // z[t] = conj(fft(conj(Z)))[t] / M; the 1/M and the istft scale are folded
    // into wineff, so this just writes windowed interleaved time samples
    device float* orow = out + (long)seg * {N};
    for (int t = tid; t < {M}; t += {T}) {{
        float2 a = Z[t];
        orow[2 * t]     =  a.x * wineff[2 * t];
        orow[2 * t + 1] = -a.y * wineff[2 * t + 1];
    }}
"""

_OLA_SRC = """
    uint i = thread_position_in_grid.x;
    uint b = thread_position_in_grid.y;
    int outlen = params[0];
    int nseg   = params[1];
    int hop    = params[2];
    int B      = params[3];
    if ((int)i >= outlen || (int)b >= B) return;

    int s_hi = min(nseg - 1, (int)i / hop);
    int s_lo = ((int)i >= {N}) ? (((int)i - {N}) / hop + 1) : 0;

    float acc = 0.0f;
    float nrm = 0.0f;
    const device float* srow = segs + (long)b * nseg * {N};
    for (int s = s_lo; s <= s_hi; ++s) {{
        int off = (int)i - s * hop;
        acc += srow[(long)s * {N} + off];
        float wv = win[off];
        nrm += wv * wv;
    }}
    out[(long)b * outlen + i] = acc / (nrm > 1e-10f ? nrm : 1.0f);
    if (b == 0) normout[i] = nrm;
"""


def eligible_istft(nperseg: int) -> bool:
    """True if the inverse kernel pair can run this one-sided istft."""
    if not (mx.metal.is_available() and hasattr(mx, "view")):
        return False
    return _MIN_N <= nperseg <= _MAX_N and not (nperseg & (nperseg - 1))


@functools.lru_cache(maxsize=32)
def _istft_kernels(N: int):
    M = N // 2
    S = int(math.log2(M))
    T = min(256, max(32, M // 2))
    inv = mx.fast.metal_kernel(
        name=f"mlx_signal_istft_{N}",
        input_names=["zin", "wineff", "tw", "ut", "params"],
        output_names=["out"],
        source=_ISTFT_SRC.format(N=N, M=M, T=T, FFT=_PAIR_FFT.format(M=M, S=S, T=T)),
        header=_HEADER.format(M=M),
    )
    ola = mx.fast.metal_kernel(
        name=f"mlx_signal_ola_{N}",
        input_names=["segs", "win", "params"],
        output_names=["out", "normout"],
        source=_OLA_SRC.format(N=N),
    )
    return inv, ola, T, M


def istft_frames(z3: mx.array, win_np, scale: float, nperseg: int, hop: int):
    """One-sided istft core: (..., nseg, M+1) spectra -> (..., outlen) signal.

    Returns ``(x, norm)`` where ``x`` is already divided by the window-power
    norm (scipy's exact where(norm > 1e-10, norm, 1) rule) and ``norm`` is the
    1-D window-power array for the caller's NOLA check.
    """
    M = nperseg // 2
    nseg = z3.shape[-2]
    batch_shape = z3.shape[:-2]
    z2 = z3.reshape(-1, nseg, M + 1)
    B = z2.shape[0]
    total = B * nseg
    outlen = nperseg + (nseg - 1) * hop

    inv, ola, T, _ = _istft_kernels(nperseg)
    tw, ut = _tables(nperseg)
    win_eff = mx.array((np.asarray(win_np, dtype=np.float64) * (scale / M))
                       .astype(np.float32))
    win_raw = mx.array(np.asarray(win_np, dtype=np.float32))

    z_f = mx.view(z2.reshape(total, M + 1), mx.float32)
    (segs,) = inv(
        inputs=[z_f, win_eff, tw, ut, mx.array([total], dtype=mx.int32)],
        grid=(T, total, 1),
        threadgroup=(T, 1, 1),
        output_shapes=[(total, nperseg)],
        output_dtypes=[mx.float32],
    )
    out, norm = ola(
        inputs=[segs, win_raw, mx.array([outlen, nseg, hop, B], dtype=mx.int32)],
        grid=(outlen, B, 1),
        threadgroup=(min(256, max(32, outlen)), 1, 1),
        output_shapes=[(B, outlen), (outlen,)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return out.reshape(batch_shape + (outlen,)), norm
