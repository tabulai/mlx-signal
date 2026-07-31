"""Filtering: FIR design (host-side), FIR application (GPU), hilbert, filtfilt.

The FIR fast paths run entirely on the GPU via FFT convolution. General IIR
filtering (``a`` with more than one tap) is inherently recursive and is
deferred to a future release (planned: batched IIR via associative scan);
those calls fall back to scipy with a :class:`~mlx_signal.FallbackWarning`.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from . import _fft_core as _sfft
from ._array import input_size, result_to_mlx, to_mlx, to_numpy
from ._arraytools import const_ext, even_ext, odd_ext
from ._config import capability_fallback, use_mlx
from .convolution import oaconvolve

__all__ = ["filtfilt", "firwin", "firwin2", "hilbert", "lfilter"]


def hilbert(x, N=None, axis=-1):
    """Analytic signal via the Hilbert transform (scipy-compatible).

    FFT -> zero negative frequencies / double positive -> inverse FFT, all on
    the GPU. Returns complex64.
    """
    xa = to_mlx(x)
    if xa.dtype == mx.complex64:
        raise ValueError("x must be real.")
    if N is None:
        N = xa.shape[axis]
    N = int(N)
    if N <= 0:
        raise ValueError("N must be positive.")

    if not use_mlx(max(input_size(x), N)):
        import scipy.signal as sps

        return result_to_mlx(sps.hilbert(to_numpy(x), N=N, axis=axis))

    xa = mx.moveaxis(xa, axis, -1) if xa.ndim > 1 else xa
    Xf = _sfft.fft(xa.astype(mx.complex64), n=N, axis=-1)
    h = np.zeros(N, dtype=np.float32)
    h[0] = 1.0
    if N % 2 == 0:
        h[1 : N // 2] = 2.0
        h[N // 2] = 1.0
    else:
        h[1 : (N + 1) // 2] = 2.0
    out = _sfft.ifft(Xf * mx.array(h), axis=-1)
    if out.ndim > 1:
        out = mx.moveaxis(out, -1, axis)
    return out


# ---------------------------------------------------------------------------
# FIR design (host-side, tiny: delegate to scipy and convert)
# ---------------------------------------------------------------------------


def firwin(numtaps, cutoff, **kwargs) -> mx.array:
    """FIR filter design using the window method (scipy.signal.firwin).

    Design happens host-side in float64 (it is tiny); the returned taps are a
    float32 MLX array ready for :func:`lfilter`/:func:`filtfilt`/`upfirdn`.
    """
    from scipy.signal import firwin as _firwin

    return mx.array(np.asarray(_firwin(numtaps, cutoff, **kwargs), dtype=np.float32))


def firwin2(numtaps, freq, gain, **kwargs) -> mx.array:
    """FIR filter design by frequency sampling (scipy.signal.firwin2)."""
    from scipy.signal import firwin2 as _firwin2

    return mx.array(np.asarray(_firwin2(numtaps, freq, gain, **kwargs), dtype=np.float32))


# ---------------------------------------------------------------------------
# FIR application
# ---------------------------------------------------------------------------


def _as_fir_taps(b, a):
    """Return float32/complex64 taps b/a[0] if (b, a) describe an FIR filter, else None."""
    a_np = np.atleast_1d(np.asarray(to_numpy(a)))
    if a_np.ndim != 1 or a_np.size != 1:
        return None
    ba = to_mlx(b)
    if ba.ndim != 1:
        return None
    a0 = complex(a_np[0]) if np.iscomplexobj(a_np) else float(a_np[0])
    if a0 == 0:
        raise ValueError("a[0] must be nonzero")
    if a0 != 1:
        ba = ba / mx.array(a0)
    return ba


def _fir_causal(b: mx.array, x: mx.array) -> mx.array:
    """Causal FIR filtering along the last axis, zero initial conditions.

    Equals ``lfilter(b, 1, x)``: the first len(x) samples of the full
    convolution.
    """
    n = x.shape[-1]
    y = oaconvolve(x, b.reshape((1,) * (x.ndim - 1) + (b.shape[0],)), mode="full",
                   axes=[x.ndim - 1])
    return y[..., :n]


def lfilter(b, a, x, axis=-1, zi=None):
    """Filter data with an IIR or FIR filter (scipy-compatible signature).

    The FIR case (``len(a) == 1``) runs on the GPU as a truncated FFT
    convolution. IIR filters and ``zi`` initial conditions fall back to scipy
    with a FallbackWarning (IIR-via-associative-scan is on the roadmap).
    """
    taps = _as_fir_taps(b, a)
    work = input_size(x)
    if taps is None or zi is not None or not use_mlx(work):
        if use_mlx(work):  # not a size decision: explain why we left the GPU
            reason = "zi initial conditions" if taps is not None else "IIR filtering (len(a) > 1)"
            capability_fallback("lfilter", reason)
        import scipy.signal as sps

        out = sps.lfilter(to_numpy(b), to_numpy(a), to_numpy(x), axis=axis, zi=zi)
        if zi is not None:
            return result_to_mlx(out[0]), result_to_mlx(out[1])
        return result_to_mlx(out)

    xa = to_mlx(x)
    moved = xa.ndim > 1 and axis not in (-1, xa.ndim - 1)
    if moved:
        xa = mx.moveaxis(xa, axis, -1)
    y = _fir_causal(taps, xa)
    if moved:
        y = mx.moveaxis(y, -1, axis)
    return y


def filtfilt(b, a, x, axis=-1, padtype="odd", padlen=None, method="pad", irlen=None):
    """Zero-phase forward-backward filtering (scipy-compatible).

    The FIR case runs on the GPU: odd/even/constant edge extension, then a
    forward and a backward causal convolution with scipy's exact steady-state
    initialization (implemented as an (ntaps-1)-sample constant prefix, which
    is algebraically identical to ``lfilter_zi`` for FIR filters). IIR filters
    and ``method="gust"`` fall back to scipy.
    """
    if method not in ("pad", "gust"):
        raise ValueError("method must be 'pad' or 'gust'.")

    taps = _as_fir_taps(b, a)
    work = input_size(x)
    if taps is None or method == "gust" or not use_mlx(work):
        if use_mlx(work):
            reason = "method='gust'" if taps is not None else "IIR filtering (len(a) > 1)"
            capability_fallback("filtfilt", reason)
        import scipy.signal as sps

        return result_to_mlx(
            sps.filtfilt(to_numpy(b), to_numpy(a), to_numpy(x), axis=axis,
                         padtype=padtype, padlen=padlen, method=method, irlen=irlen)
        )

    if padtype not in ("even", "odd", "constant", None):
        raise ValueError(
            f"Unknown value '{padtype}' given to padtype. padtype must be 'even', "
            "'odd', 'constant', or None."
        )

    ntaps = taps.shape[0]
    if padtype is None:
        padlen = 0
    elif padlen is None:
        padlen = 3 * max(1, ntaps)
    padlen = int(padlen)

    xa = to_mlx(x)
    n = xa.shape[axis]
    if padlen >= n:
        raise ValueError(
            "The length of the input vector x must be greater than padlen, "
            f"which is {padlen}."
        )

    moved = xa.ndim > 1 and axis not in (-1, xa.ndim - 1)
    if moved:
        xa = mx.moveaxis(xa, axis, -1)

    if padlen > 0:
        ext_func = {"even": even_ext, "odd": odd_ext, "constant": const_ext}[padtype]
        ext = ext_func(xa, padlen)
    else:
        ext = xa

    def _pass(sig: mx.array) -> mx.array:
        """One causal pass with steady-state (lfilter_zi-equivalent) init."""
        if ntaps == 1:
            return sig * taps
        prefix = mx.broadcast_to(sig[..., :1], sig.shape[:-1] + (ntaps - 1,))
        padded = mx.concatenate([prefix, sig], axis=-1)
        kernel = taps.reshape((1,) * (padded.ndim - 1) + (ntaps,))
        return oaconvolve(padded, kernel, mode="valid", axes=[padded.ndim - 1])

    y = _pass(ext)
    y = _pass(y[..., ::-1])[..., ::-1]

    if padlen > 0:
        y = y[..., padlen : padlen + n]
    if moved:
        y = mx.moveaxis(y, -1, axis)
    return y
