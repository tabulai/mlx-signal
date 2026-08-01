"""Convolution and correlation on the GPU via padded batched FFTs.

``fftconvolve``/``oaconvolve``/``correlate``/``convolve`` mirror scipy.signal.
Internally every FFT length is padded to the next power of two
(:func:`mlx_signal.next_fast_len`), where MLX's Metal FFT is fastest, then the
result is sliced back — so prime and odd lengths don't fall off a performance
cliff the way raw non-power-of-two FFTs do.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from . import _fft_core as _sfft
from . import _fourstep, _ola_metal, _stft_metal
from ._array import result_to_mlx, signal_np, to_mlx
from ._cache import TWIDDLES
from ._config import use_mlx
from ._fft import next_fast_len

__all__ = ["convolve", "correlate", "correlation_lags", "fftconvolve", "oaconvolve"]


# ---------------------------------------------------------------------------
# shape/axis helpers (ported from scipy.signal._signaltools semantics)
# ---------------------------------------------------------------------------


def _normalize_axes(ndim: int, axes) -> list[int]:
    if axes is None:
        return list(range(ndim))
    if np.isscalar(axes):
        axes = [axes]
    checked = []
    for a in axes:
        if not isinstance(a, int | np.integer):
            raise ValueError("axes must be a scalar or iterable of integers")
        a = int(a)
        if not -ndim <= a < ndim:
            raise ValueError("axes exceeds dimensionality of input")
        checked.append(a % ndim)
    if len(set(checked)) != len(checked):
        raise ValueError("all axes must be unique")
    return checked


def _inputs_swap_needed(mode, shape1, shape2, axes=None) -> bool:
    """For 'valid' mode one input must cover the other on every conv axis."""
    if mode != "valid":
        return False
    if not shape1:
        return False
    if axes is None:
        axes = range(len(shape1))
    ok1 = all(shape1[i] >= shape2[i] for i in axes)
    ok2 = all(shape2[i] >= shape1[i] for i in axes)
    if not (ok1 or ok2):
        raise ValueError(
            "For 'valid' mode, one must be at least as large as the other in every dimension"
        )
    return not ok1


def _init_conv_axes(a1, a2, mode, axes, sorted_axes=False):
    s1, s2 = a1.shape, a2.shape
    noaxes = axes is None
    axes = _normalize_axes(a1.ndim, axes)
    if not noaxes and not len(axes):
        raise ValueError("when provided, axes cannot be empty")
    # length-1 conv axes rely on broadcasting, no FFT needed
    axes = [a for a in axes if s1[a] != 1 and s2[a] != 1]
    if sorted_axes:
        axes.sort()
    if not all(s1[a] == s2[a] or s1[a] == 1 or s2[a] == 1
               for a in range(a1.ndim) if a not in axes):
        raise ValueError(f"incompatible shapes for in1 and in2: {s1} and {s2}")
    if _inputs_swap_needed(mode, s1, s2, axes=axes):
        a1, a2 = a2, a1
    return a1, a2, axes


def _centered(arr: mx.array, newshape) -> mx.array:
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    return arr[tuple(slice(int(s), int(e)) for s, e in zip(startind, endind, strict=True))]


def _apply_conv_mode(ret, s1, s2, mode, axes):
    if mode == "full":
        return ret
    if mode == "same":
        return _centered(ret, s1)
    if mode == "valid":
        shape_valid = [
            ret.shape[a] if a not in axes else s1[a] - s2[a] + 1 for a in range(ret.ndim)
        ]
        return _centered(ret, shape_valid)
    raise ValueError("acceptable mode flags are 'valid', 'same', or 'full'")


def _promote_pair(a1: mx.array, a2: mx.array):
    complex_result = mx.complex64 in (a1.dtype, a2.dtype)
    if complex_result:
        a1 = a1.astype(mx.complex64)
        a2 = a2.astype(mx.complex64)
    return a1, a2, complex_result


# ---------------------------------------------------------------------------
# public functions
# ---------------------------------------------------------------------------


def fftconvolve(in1, in2, mode="full", axes=None):
    """Convolve two N-dimensional arrays using padded GPU FFTs (scipy-compatible)."""
    a1 = to_mlx(in1)
    a2 = to_mlx(in2)
    if a1.ndim == a2.ndim == 0:
        return a1 * a2
    if a1.ndim != a2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    if a1.size == 0 or a2.size == 0:
        return mx.zeros((0,), dtype=a1.dtype)

    same_input = in1 is in2
    axes_arg = axes
    a1, a2, axes = _init_conv_axes(a1, a2, mode, axes, sorted_axes=False)
    s1, s2 = a1.shape, a2.shape
    shape = [
        max(s1[i], s2[i]) if i not in axes else s1[i] + s2[i] - 1 for i in range(a1.ndim)
    ]

    if not use_mlx(int(np.prod(shape))):
        import scipy.signal as sps

        out = sps.fftconvolve(signal_np(in1), signal_np(in2), mode=mode,
                              axes=axes_arg)
        return result_to_mlx(out)

    # A long x short convolution runs as blocked overlap-add when the padded
    # FFT would land on an MLX-0.32 broken Metal size, or is simply big enough
    # (>= 2^19) that many small block FFTs beat one 2x-padded giant one.
    if len(axes) == 1:
        ax = axes[0]
        fl = next_fast_len(shape[ax])
        if (
            min(s1[ax], s2[ax]) <= (1 << 16)
            and max(s1[ax], s2[ax]) > (1 << 16)
            and (_sfft.metal_fft_broken(fl) or fl >= (1 << 19))
        ):
            return oaconvolve(a1, a2, mode=mode, axes=list(axes))

    ret = _freq_domain_conv(a1, a2, axes, shape, same_input=same_input)
    return _apply_conv_mode(ret, s1, s2, mode, axes)


def _freq_domain_conv(a1, a2, axes, shape, same_input=False):
    """FFT-multiply-IFFT over ``axes`` at padded fast lengths, sliced to ``shape``.

    Single-axis transforms go through the 1-D FFT wrappers so broken Metal
    lengths run as GPU four-step decompositions rather than on the CPU stream
    (a 2^20-sample pair convolution is ~15x faster for it). ``same_input``
    (auto-convolution) computes one forward transform and squares it.
    """
    if not len(axes):
        return a1 * a2
    a1, a2, complex_result = _promote_pair(a1, a2)
    fshape = [next_fast_len(shape[a]) for a in axes]
    if len(axes) == 1:
        ax, fl = axes[0], fshape[0]
        if complex_result:
            sp1 = _sfft.fft(a1, n=fl, axis=ax)
            sp2 = sp1 if same_input else _sfft.fft(a2, n=fl, axis=ax)
            ret = _sfft.ifft(sp1 * sp2, n=fl, axis=ax)
        else:
            # at direct Metal sizes the plain path wins (the FFT is launch-
            # bound, so packing's extra passes cost more than they save); at
            # broken lengths the packed pair replaces the rfft_large/
            # irfft_large untangle machinery and runs ~2x faster
            ret = None
            if _sfft.metal_fft_broken(fl):
                ret = _fourstep.rfft_conv_pair(a1, a1 if same_input else a2, fl, axis=ax)
            if ret is None and same_input:
                sp1 = _sfft.rfft(a1, n=fl, axis=ax)
                ret = _sfft.irfft(sp1 * sp1, n=fl, axis=ax)
            elif ret is None:
                sp1 = _sfft.rfft(a1, n=fl, axis=ax)
                sp2 = _sfft.rfft(a2, n=fl, axis=ax)
                ret = _sfft.irfft(sp1 * sp2, n=fl, axis=ax)
    elif complex_result:
        sp1 = _sfft.fftn(a1, s=fshape, axes=axes)
        sp2 = sp1 if same_input else _sfft.fftn(a2, s=fshape, axes=axes)
        ret = _sfft.ifftn(sp1 * sp2, s=fshape, axes=axes)
    else:
        sp1 = _sfft.rfftn(a1, s=fshape, axes=axes)
        sp2 = sp1 if same_input else _sfft.rfftn(a2, s=fshape, axes=axes)
        ret = _sfft.irfftn(sp1 * sp2, s=fshape, axes=axes)
    fslice = tuple(
        slice(shape[a]) if a in axes else slice(None) for a in range(a1.ndim)
    )
    return ret[fslice]


def oaconvolve(in1, in2, mode="full", axes=None):
    """Convolve using the overlap-add method (scipy-compatible results).

    Beneficial when one input is much longer than the other along one axis
    (e.g. streaming FIR filtering). When shapes differ along more than one
    convolution axis this implementation delegates to :func:`fftconvolve`
    (identical results, possibly slower than scipy's fully-blocked version).
    """
    a1 = to_mlx(in1)
    a2 = to_mlx(in2)
    if a1.ndim == a2.ndim == 0:
        return a1 * a2
    if a1.ndim != a2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    if a1.size == 0 or a2.size == 0:
        return mx.zeros((0,), dtype=a1.dtype)
    if a1.shape == a2.shape:
        return fftconvolve(in1, in2, mode=mode, axes=axes)

    axes_arg = axes
    a1, a2, axes = _init_conv_axes(a1, a2, mode, axes, sorted_axes=True)
    s1, s2 = a1.shape, a2.shape

    split_axes = [a for a in axes if s1[a] != s2[a]]
    if len(split_axes) != 1:
        ret = _freq_domain_conv(
            a1, a2, axes, [max(s1[i], s2[i]) if i not in axes else s1[i] + s2[i] - 1
                           for i in range(a1.ndim)]
        )
        return _apply_conv_mode(ret, s1, s2, mode, axes)
    ax = split_axes[0]

    if not use_mlx(int(np.prod([max(s1[i], s2[i]) for i in range(a1.ndim)]))):
        import scipy.signal as sps

        out = sps.oaconvolve(signal_np(in1), signal_np(in2), mode=mode,
                              axes=axes_arg)
        return result_to_mlx(out)

    # long input first for the blocked computation (convolution commutes; the
    # output mode is still applied with the possibly-swapped-for-valid s1/s2)
    b1, b2 = (a1, a2) if s1[ax] >= s2[ax] else (a2, a1)
    n1, n2 = b1.shape[ax], b2.shape[ax]

    nfull = next_fast_len(n1 + n2 - 1)
    nblock = next_fast_len(max(8 * n2, 4096))
    if nblock >= nfull:
        ret = _freq_domain_conv(
            a1, a2, axes, [max(s1[i], s2[i]) if i not in axes else s1[i] + s2[i] - 1
                           for i in range(a1.ndim)]
        )
        return _apply_conv_mode(ret, s1, s2, mode, axes)

    # conv over the remaining (equal-length) axes happens inside each block FFT
    other_axes = [a for a in axes if a != ax]
    if other_axes:
        # rare mixed case: fall back to the plain FFT path for correctness
        ret = _freq_domain_conv(
            a1, a2, axes, [max(s1[i], s2[i]) if i not in axes else s1[i] + s2[i] - 1
                           for i in range(a1.ndim)]
        )
        return _apply_conv_mode(ret, s1, s2, mode, axes)

    # fully fused kernel path: real data with one shared filter that fits a
    # block — forward FFT, spectrum multiply, and inverse all in threadgroup
    # memory, then gather overlap-add; no padded signal copy, no giant FFTs
    if (
        mx.metal.is_available()
        and b1.dtype == mx.float32
        and b2.dtype == mx.float32
        and n2 <= _stft_metal.FFTCONV_N // 2 + 1
        and n1 + n2 - 1 > _stft_metal.FFTCONV_N
        and all(d == 1 for i, d in enumerate(b2.shape) if i != ax)
    ):
        b1m = mx.moveaxis(b1, ax, -1)
        blocks, step = _stft_metal.fftconv_blocks(
            b1m.reshape(-1, n1), b2.reshape(n2)
        )
        full_len = (blocks.shape[1] - 1) * step + _stft_metal.FFTCONV_N
        out = _ola_metal.ola_gather(blocks, step, full_len)[..., : n1 + n2 - 1]
        out = out.reshape(b1m.shape[:-1] + (n1 + n2 - 1,))
        out = mx.moveaxis(out, -1, ax)
        return _apply_conv_mode(out, s1, s2, mode, axes)

    b1, b2, complex_result = _promote_pair(b1, b2)
    b1 = mx.moveaxis(b1, ax, -1)
    b2 = mx.moveaxis(b2, ax, -1)

    batch = np.broadcast_shapes(b1.shape[:-1], b2.shape[:-1])
    b1 = mx.broadcast_to(b1, tuple(batch) + (n1,))
    b2 = mx.broadcast_to(b2, tuple(batch) + (n2,))

    step = nblock - n2 + 1
    nblocks = math.ceil(n1 / step)
    pad = nblocks * step - n1
    if pad:
        b1 = mx.concatenate([b1, mx.zeros(b1.shape[:-1] + (pad,), dtype=b1.dtype)], axis=-1)
    blocks = b1.reshape(tuple(batch) + (nblocks, step))

    if complex_result:
        hf = _sfft.fft(b2, n=nblock, axis=-1)[..., None, :]
        yf = _sfft.fft(blocks, n=nblock, axis=-1) * hf
        y = _sfft.ifft(yf, n=nblock, axis=-1)
    else:
        hf = _sfft.rfft(b2, n=nblock, axis=-1)[..., None, :]
        yf = _sfft.rfft(blocks, n=nblock, axis=-1) * hf
        y = _sfft.irfft(yf, n=nblock, axis=-1)

    # overlap-add the per-block full convolutions (each length nblock, hop step)
    full_len = (nblocks - 1) * step + nblock
    if mx.metal.is_available():
        # gather kernel: one thread per output sample, no scatter collisions
        out = _ola_metal.ola_gather(y.reshape((-1, nblocks, nblock)), step, full_len)
    else:
        pos = (
            (step * mx.arange(nblocks, dtype=mx.int32))[:, None]
            + mx.arange(nblock, dtype=mx.int32)[None, :]
        ).reshape(-1)
        v2 = y.reshape((-1, nblocks * nblock))
        if complex_result:
            re = mx.zeros((v2.shape[0], full_len)).at[:, pos].add(mx.real(v2))
            im = mx.zeros((v2.shape[0], full_len)).at[:, pos].add(mx.imag(v2))
            out = re.astype(mx.complex64) + im.astype(mx.complex64) * mx.array(1j)
        else:
            out = mx.zeros((v2.shape[0], full_len)).at[:, pos].add(v2)
    out = out.reshape(tuple(batch) + (full_len,))[..., : n1 + n2 - 1]

    out = mx.moveaxis(out, -1, ax)
    return _apply_conv_mode(out, s1, s2, mode, axes)


def convolve(in1, in2, mode="full", method="auto"):
    """Convolve two N-dimensional arrays (scipy-compatible).

    ``method="auto"`` and ``"fft"`` run the GPU FFT path; ``method="direct"``
    is an explicit request for time-domain convolution and routes to
    scipy.signal (documented, silent).
    """
    if method not in ("auto", "fft", "direct"):
        raise ValueError("Acceptable method flags are 'auto', 'direct', or 'fft'.")
    if method == "direct":
        import scipy.signal as sps

        return result_to_mlx(sps.convolve(signal_np(in1), signal_np(in2), mode=mode,
                                          method="direct"))
    return fftconvolve(in1, in2, mode=mode)


def _reverse_and_conj(a: mx.array) -> mx.array:
    rev = a[(slice(None, None, -1),) * a.ndim]
    if rev.dtype == mx.complex64:
        rev = mx.conj(rev)
    return rev


def correlate(in1, in2, mode="full", method="auto"):
    """Cross-correlate two N-dimensional arrays (scipy-compatible).

    Computed as ``convolve(in1, reversed(conj(in2)))`` on the GPU FFT path.
    """
    if method not in ("auto", "fft", "direct"):
        raise ValueError("Acceptable method flags are 'auto', 'direct', or 'fft'.")
    a1 = to_mlx(in1)
    a2 = to_mlx(in2)
    if a1.ndim != a2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    if method == "direct":
        import scipy.signal as sps

        return result_to_mlx(sps.correlate(signal_np(in1), signal_np(in2), mode=mode,
                                           method="direct"))
    if in1 is in2:
        return _autocorrelate(a1, in1, mode)
    return fftconvolve(a1, _reverse_and_conj(a2), mode=mode)


def _rev_twiddle(fl: int, length: int, half: bool) -> mx.array:
    """Linear phase relating a reversed signal's spectrum to conj(X).

    FFT(rev(conj(x)), fl)[k] = exp(-2j*pi*k*(length-1)/fl) * conj(X[k]); the
    phase is reduced mod fl in exact int64 before the complex exponential.
    Cached under a byte budget: per-(fl, length) vectors under an entry-count
    LRU once retained hundreds of MiB across autocorrelation lengths.
    """
    key = ("rev", fl, length, half)
    t = TWIDDLES.get(key)
    if t is not None:
        return t
    k = mx.arange(fl // 2 + 1 if half else fl, dtype=mx.int64)
    theta = ((k * (length - 1)) % fl).astype(mx.float32) * mx.array(
        -2.0 * np.pi / fl, dtype=mx.float32
    )
    t = mx.cos(theta).astype(mx.complex64) + mx.sin(theta).astype(
        mx.complex64
    ) * mx.array(1j)
    mx.eval(t)
    TWIDDLES.put(key, t, t.size * 8)
    return t


def _autocorrelate(x: mx.array, orig, mode: str) -> mx.array:
    """correlate(x, x): one forward FFT instead of two plus a reversed copy.

    The reversed-conjugate input's spectrum is conj(X) times a linear phase
    (:func:`_rev_twiddle`), so the second transform is never computed.
    """
    if x.ndim == 0:
        return x * mx.conj(x) if x.dtype == mx.complex64 else x * x
    if x.size == 0:
        return mx.zeros((0,), dtype=x.dtype)
    s1 = x.shape
    axes = [a for a in range(x.ndim) if s1[a] != 1]
    shape = [s1[a] if a not in axes else 2 * s1[a] - 1 for a in range(x.ndim)]
    if not use_mlx(int(np.prod(shape))):
        import scipy.signal as sps

        out = sps.correlate(signal_np(orig), signal_np(orig), mode=mode, method="fft")
        return result_to_mlx(out)
    if not axes:
        ret = x * mx.conj(x) if x.dtype == mx.complex64 else x * x
        return _apply_conv_mode(ret, s1, s1, mode, axes)
    complex_input = x.dtype == mx.complex64
    fshape = [next_fast_len(shape[a]) for a in axes]
    fslice = tuple(slice(shape[a]) if a in axes else slice(None) for a in range(x.ndim))
    if len(axes) == 1:
        ax, fl = axes[0], fshape[0]
        if not complex_input and _sfft.metal_fft_broken(fl):
            # at broken lengths the packed autocorrelation (one half-length
            # FFT, fused product, one inverse) beats rfft+twiddle+irfft,
            # whose rfft_large/irfft_large untangle passes it never runs
            ret = _fourstep.rfft_autocorr(x, fl, axis=ax)
            if ret is not None:
                return _apply_conv_mode(ret[fslice], s1, s1, mode, axes)
        sp = _sfft.fft(x, n=fl, axis=ax) if complex_input else _sfft.rfft(x, n=fl, axis=ax)
    else:
        sp = (_sfft.fftn(x, s=fshape, axes=axes) if complex_input
              else _sfft.rfftn(x, s=fshape, axes=axes))
    sp2 = mx.conj(sp)
    for i, (a, fl) in enumerate(zip(axes, fshape, strict=True)):
        half = (not complex_input) and i == len(axes) - 1  # rfftn halves axes[-1]
        tw = _rev_twiddle(fl, s1[a], half)
        bshape = [1] * sp2.ndim
        bshape[a] = tw.shape[0]
        sp2 = sp2 * tw.reshape(bshape)
    prod = sp * sp2
    if len(axes) == 1:
        ax, fl = axes[0], fshape[0]
        ret = _sfft.ifft(prod, n=fl, axis=ax) if complex_input else _sfft.irfft(prod, n=fl, axis=ax)
    else:
        ret = (_sfft.ifftn(prod, s=fshape, axes=axes) if complex_input
               else _sfft.irfftn(prod, s=fshape, axes=axes))
    return _apply_conv_mode(ret[fslice], s1, s1, mode, axes)


def correlation_lags(in1_len, in2_len, mode="full") -> np.ndarray:
    """Lag indices for 1-D cross-correlation (scipy-compatible).

    Returns a NumPy integer array: lags are axis metadata, not GPU data.
    """
    in1_len = int(in1_len)
    in2_len = int(in2_len)
    if mode == "full":
        lags = np.arange(-in2_len + 1, in1_len)
    elif mode == "same":
        lags = np.arange(-in2_len + 1, in1_len)
        mid = lags.size // 2
        lag_bound = in1_len // 2
        if in1_len % 2 == 0:
            lags = lags[(mid - lag_bound) : (mid + lag_bound)]
        else:
            lags = lags[(mid - lag_bound) : (mid + lag_bound) + 1]
    elif mode == "valid":
        lag_bound = in1_len - in2_len
        if lag_bound >= 0:
            lags = np.arange(lag_bound + 1)
        else:
            lags = np.arange(lag_bound, 1)
    else:
        raise ValueError(f"Mode {mode} is invalid")
    return lags
