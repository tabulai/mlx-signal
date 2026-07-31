"""Spectral estimation: periodogram, welch, csd, coherence, spectrogram, stft, istft.

All six functions share one MLX core (:func:`_spectral_helper`): frame the
signal into overlapping segments with a gather, detrend and window every
segment, run one batched FFT over all segments at once, then scale/reduce.
scipy loops over segments serially; here a 64-channel Welch PSD is a single
batched GPU FFT.

Semantics mirror ``scipy.signal`` (same defaults, shapes, and edge handling).
Known deviations are documented per function; the big global one is dtype:
computation is float32/complex64 (Metal has no float64).
"""

from __future__ import annotations

import functools
import warnings

import mlx.core as mx
import numpy as np

from . import _fft_core as _sfft
from ._array import input_size, result_to_mlx, to_mlx, to_numpy
from ._arraytools import const_ext, even_ext, odd_ext, zero_ext
from ._config import capability_fallback, use_mlx
from .windows import _window_np

__all__ = ["coherence", "csd", "istft", "periodogram", "spectrogram", "stft", "welch"]

_BOUNDARY_FUNCS = {
    "even": even_ext,
    "odd": odd_ext,
    "constant": const_ext,
    "zeros": zero_ext,
    None: None,
}


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------


def _triage_segments(window, nperseg, input_length):
    """Resolve (window values, nperseg) exactly like scipy's _triage_segments."""
    if isinstance(window, str | tuple):
        nperseg = 256 if nperseg is None else int(nperseg)
        if nperseg > input_length:
            warnings.warn(
                f"nperseg = {nperseg:d} is greater than input length "
                f" = {input_length:d}, using nperseg = {input_length:d}",
                stacklevel=4,
            )
            nperseg = input_length
        key = tuple(window) if isinstance(window, tuple) else window
        win_np = _window_np(key, nperseg)
    else:
        win_np = np.asarray(to_numpy(window), dtype=np.float64)
        if win_np.ndim != 1:
            raise ValueError("window must be 1-D")
        if input_length < win_np.shape[0]:
            raise ValueError("window is longer than input signal")
        if nperseg is None:
            nperseg = win_np.shape[0]
        elif int(nperseg) != win_np.shape[0]:
            raise ValueError("value specified for nperseg is different from length of window")
        nperseg = int(nperseg)
    return win_np, nperseg


def _detrend_segments(d: mx.array, kind) -> mx.array:
    """Detrend each length-nperseg segment (last axis) in one vectorized shot."""
    if kind is False or kind is None:
        return d
    if not isinstance(kind, str):
        raise ValueError("Trend type must be 'linear' or 'constant'.")
    if kind in ("constant", "c"):
        return d - mx.mean(d, axis=-1, keepdims=True)
    if kind in ("linear", "l"):
        n = d.shape[-1]
        t = mx.arange(n, dtype=mx.float32) - (n - 1) / 2.0
        denom = n * (n * n - 1) / 12.0  # sum of centered t**2
        if denom == 0.0:  # n == 1: mean removal is exact
            return d - mx.mean(d, axis=-1, keepdims=True)
        slope = mx.sum(d * t, axis=-1, keepdims=True) / denom
        return d - mx.mean(d, axis=-1, keepdims=True) - slope * t
    raise ValueError("Trend type must be 'linear' or 'constant'.")


def _fft_frames(x, win, detrend, nperseg, nstep, nfft, sides):
    """Frame -> detrend -> window -> batched FFT. Returns (..., nseg, nfreq)."""
    n = x.shape[-1]
    nseg = (n - nperseg) // nstep + 1
    idx = (
        mx.arange(nperseg, dtype=mx.int32)[None, :]
        + (nstep * mx.arange(nseg, dtype=mx.int32))[:, None]
    )
    frames = mx.take(x, idx, axis=-1)
    frames = _detrend_segments(frames, detrend)
    frames = frames * win
    if sides == "onesided":
        return _sfft.rfft(frames, n=nfft, axis=-1)
    if frames.dtype != mx.complex64:
        frames = frames.astype(mx.complex64)
    return _sfft.fft(frames, n=nfft, axis=-1)


@functools.lru_cache(maxsize=64)
def _onesided_factor(nfft: int) -> mx.array:
    """[1, 2, 2, ..., 2, (1 if nfft even else 2)] — doubling for one-sided PSDs."""
    f = np.full(nfft // 2 + 1, 2.0, dtype=np.float32)
    f[0] = 1.0
    if nfft % 2 == 0:
        f[-1] = 1.0
    return mx.array(f)


def _median_bias(n: int) -> float:
    ii_2 = 2 * np.arange(1.0, (n - 1) // 2 + 1)
    return float(1 + np.sum(1.0 / (ii_2 + 1) - 1.0 / ii_2))


def _unwrap(p: mx.array, axis: int = -1) -> mx.array:
    """np.unwrap (period 2*pi) in MLX ops."""
    p = mx.moveaxis(p, axis, -1)
    d = p[..., 1:] - p[..., :-1]
    two_pi = 2.0 * np.pi
    ddmod = d + np.pi - mx.floor((d + np.pi) / two_pi) * two_pi - np.pi
    ddmod = mx.where((ddmod == -np.pi) & (d > 0), mx.array(np.pi, dtype=p.dtype), ddmod)
    ph_correct = mx.where(mx.abs(d) < np.pi, mx.zeros_like(d), ddmod - d)
    out = mx.concatenate([p[..., :1], p[..., 1:] + mx.cumsum(ph_correct, axis=-1)], axis=-1)
    return mx.moveaxis(out, -1, axis)


def _empty_like_shape(shape) -> mx.array:
    return mx.zeros(shape, dtype=mx.float32)


# ---------------------------------------------------------------------------
# the shared core
# ---------------------------------------------------------------------------


def _spectral_helper(
    x,
    y,
    fs=1.0,
    window="hann",
    nperseg=None,
    noverlap=None,
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="density",
    axis=-1,
    mode="psd",
    boundary=None,
    padded=False,
):
    """MLX port of scipy.signal._spectral_helper (identical layout and scaling).

    Returns ``(freqs, t, result)`` as MLX arrays; ``result`` has the frequency
    axis at the position of the input ``axis`` and segment times last.
    """
    if mode not in ("psd", "stft"):
        raise ValueError(f"Unknown value for mode {mode}, must be one of: ('psd', 'stft')")
    if boundary not in _BOUNDARY_FUNCS:
        raise ValueError(
            f"Unknown boundary option '{boundary}', must be one of: "
            f"{list(_BOUNDARY_FUNCS.keys())}"
        )
    same_data = y is x
    if not same_data and mode != "psd":
        raise ValueError("x and y must be equal if mode is 'stft'")
    axis = int(axis)

    x = to_mlx(x)
    if not same_data:
        y = to_mlx(y)
        xouter, youter = list(x.shape), list(y.shape)
        xouter.pop(axis)
        youter.pop(axis)
        try:
            outershape = np.broadcast_shapes(tuple(xouter), tuple(youter))
        except ValueError as e:
            raise ValueError("x and y cannot be broadcast together.") from e

    if same_data:
        if x.size == 0:
            e = _empty_like_shape(x.shape)
            return e, e, e
    else:
        if x.size == 0 or y.size == 0:
            outshape = outershape + (min(x.shape[axis], y.shape[axis]),)
            emptyout = mx.moveaxis(_empty_like_shape(outshape), -1, axis)
            return emptyout, emptyout, emptyout

    if x.ndim > 1 and axis != -1:
        x = mx.moveaxis(x, axis, -1)
        if not same_data and y.ndim > 1:
            y = mx.moveaxis(y, axis, -1)

    if not same_data and x.shape[-1] != y.shape[-1]:
        if x.shape[-1] < y.shape[-1]:
            pad = y.shape[-1] - x.shape[-1]
            x = mx.concatenate([x, mx.zeros(x.shape[:-1] + (pad,), dtype=x.dtype)], axis=-1)
        else:
            pad = x.shape[-1] - y.shape[-1]
            y = mx.concatenate([y, mx.zeros(y.shape[:-1] + (pad,), dtype=y.dtype)], axis=-1)

    if nperseg is not None:
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError("nperseg must be a positive integer")

    win_np, nperseg = _triage_segments(window, nperseg, input_length=x.shape[-1])

    nfft = nperseg if nfft is None else int(nfft)
    if nfft < nperseg:
        raise ValueError("nfft must be greater than or equal to nperseg.")
    noverlap = nperseg // 2 if noverlap is None else int(noverlap)
    if noverlap >= nperseg:
        raise ValueError("noverlap must be less than nperseg.")
    nstep = nperseg - noverlap

    if boundary is not None:
        ext_func = _BOUNDARY_FUNCS[boundary]
        x = ext_func(x, nperseg // 2)
        if not same_data:
            y = ext_func(y, nperseg // 2)

    if padded:
        nadd = (-(x.shape[-1] - nperseg) % nstep) % nperseg
        if nadd:
            x = mx.concatenate([x, mx.zeros(x.shape[:-1] + (nadd,), dtype=x.dtype)], axis=-1)
            if not same_data:
                y = mx.concatenate(
                    [y, mx.zeros(y.shape[:-1] + (nadd,), dtype=y.dtype)], axis=-1
                )

    x_complex = x.dtype == mx.complex64
    y_complex = (not same_data) and y.dtype == mx.complex64
    if return_onesided and (x_complex or y_complex):
        # scipy warns and switches; we switch silently (documented)
        sides = "twosided"
    elif return_onesided:
        sides = "onesided"
    else:
        sides = "twosided"

    if sides == "onesided":
        freqs_np = np.fft.rfftfreq(nfft, 1 / fs)
    else:
        freqs_np = np.fft.fftfreq(nfft, 1 / fs)

    if scaling == "density":
        scale = 1.0 / (fs * float((win_np * win_np).sum()))
    elif scaling == "spectrum":
        scale = 1.0 / float(win_np.sum()) ** 2
    else:
        raise ValueError(f"Unknown scaling: {scaling!r}")
    if mode == "stft":
        scale = float(np.sqrt(scale))

    win = mx.array(win_np.astype(np.float32))

    fx = _fft_frames(x, win, detrend, nperseg, nstep, nfft, sides)
    if not same_data:
        fy = _fft_frames(y, win, detrend, nperseg, nstep, nfft, sides)
        result = mx.conj(fx) * fy
    elif mode == "psd":
        a = mx.abs(fx)
        result = a * a  # real float32, == (conj(F) * F).real
    else:
        result = fx

    result = result * mx.array(scale, dtype=mx.float32)
    if sides == "onesided" and mode == "psd":
        result = result * _onesided_factor(nfft)

    t_np = np.arange(nperseg / 2, x.shape[-1] - nperseg / 2 + 1, nstep) / float(fs)
    if boundary is not None:
        t_np = t_np - (nperseg / 2) / fs

    # New trailing time axis: a negative frequency-axis index shifts down one.
    ax = axis - 1 if axis < 0 else axis
    result = mx.moveaxis(result, -1, ax)

    return (
        mx.array(freqs_np.astype(np.float32)),
        mx.array(t_np.astype(np.float32)),
        result,
    )


# ---------------------------------------------------------------------------
# scipy fallback plumbing
# ---------------------------------------------------------------------------


def _window_arg_np(window):
    """Pass window specs through; convert array windows for scipy calls."""
    if window is None or isinstance(window, str | tuple):
        return window
    return to_numpy(window)


def _mlx_or_fallback(func_name: str, work: int, detrend=None) -> bool:
    """Shared routing: returns True for the MLX path, False for scipy."""
    if not use_mlx(work):
        return False
    if detrend is not None and callable(detrend):
        capability_fallback(func_name, "callable detrend")
        return False
    return True


# ---------------------------------------------------------------------------
# public functions
# ---------------------------------------------------------------------------


def csd(
    x,
    y,
    fs=1.0,
    window="hann",
    nperseg=None,
    noverlap=None,
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="density",
    axis=-1,
    average="mean",
):
    """Cross power spectral density, Pxy, by Welch's method (scipy-compatible).

    Deviations from scipy: float32/complex64 compute; complex input switches to
    a two-sided spectrum silently instead of warning.
    """
    if average not in ("mean", "median"):
        raise ValueError(f"average must be 'mean' or 'median', got {average!r}")
    if not _mlx_or_fallback("csd", max(input_size(x), input_size(y)), detrend):
        import scipy.signal as sps

        f, p = sps.csd(
            to_numpy(x),
            to_numpy(y) if y is not x else to_numpy(x),
            fs=fs,
            window=_window_arg_np(window),
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            detrend=detrend,
            return_onesided=return_onesided,
            scaling=scaling,
            axis=axis,
            average=average,
        )
        return result_to_mlx(f), result_to_mlx(p)

    freqs, _, pxy = _spectral_helper(
        x,
        y,
        fs,
        window,
        nperseg,
        noverlap,
        nfft,
        detrend,
        return_onesided,
        scaling,
        axis,
        mode="psd",
    )
    if pxy.ndim >= 1 and pxy.shape[-1] > 1:
        if average == "median":
            bias = _median_bias(pxy.shape[-1])
            if pxy.dtype == mx.complex64:
                pxy = mx.median(mx.real(pxy), axis=-1).astype(mx.complex64) + mx.median(
                    mx.imag(pxy), axis=-1
                ).astype(mx.complex64) * mx.array(1j)
            else:
                pxy = mx.median(pxy, axis=-1)
            pxy = pxy / bias
        else:
            pxy = mx.mean(pxy, axis=-1)
    else:
        pxy = mx.reshape(pxy, pxy.shape[:-1])
    return freqs, pxy


def welch(
    x,
    fs=1.0,
    window="hann",
    nperseg=None,
    noverlap=None,
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="density",
    axis=-1,
    average="mean",
):
    """Welch power spectral density estimate (scipy-compatible).

    One batched GPU FFT over all segments (and all channels of an N-d input)
    replaces scipy's serial per-segment loop.
    """
    freqs, pxx = csd(
        x,
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        return_onesided=return_onesided,
        scaling=scaling,
        axis=axis,
        average=average,
    )
    if pxx.dtype == mx.complex64:
        pxx = mx.real(pxx)
    return freqs, pxx


def periodogram(
    x,
    fs=1.0,
    window="boxcar",
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="density",
    axis=-1,
):
    """Power spectral density using a periodogram (scipy-compatible)."""
    x = to_mlx(x)
    if x.size == 0:
        return _empty_like_shape(x.shape), _empty_like_shape(x.shape)

    if window is None:
        window = "boxcar"

    n_ax = x.shape[axis]
    if nfft is None:
        nperseg = n_ax
    elif nfft == n_ax:
        nperseg = nfft
    elif nfft > n_ax:
        nperseg = n_ax
    else:  # nfft < n: analyze only the first nfft samples, like scipy
        sl = [slice(None)] * x.ndim
        sl[axis] = slice(nfft)
        x = x[tuple(sl)]
        nperseg = nfft
        nfft = None

    if not isinstance(window, str | tuple) and window is not None:
        if input_size(window) != nperseg:
            raise ValueError(
                "the size of the window must be the same size of the input on the specified axis"
            )

    return welch(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=0,
        nfft=nfft,
        detrend=detrend,
        return_onesided=return_onesided,
        scaling=scaling,
        axis=axis,
    )


def coherence(
    x, y, fs=1.0, window="hann", nperseg=None, noverlap=None, nfft=None,
    detrend="constant", axis=-1,
):
    """Magnitude squared coherence estimate, |Pxy|^2/(Pxx*Pyy) (scipy-compatible)."""
    freqs, pxx = welch(
        x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
        detrend=detrend, axis=axis,
    )
    _, pyy = welch(
        y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
        detrend=detrend, axis=axis,
    )
    _, pxy = csd(
        x, y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
        detrend=detrend, axis=axis,
    )
    a = mx.abs(pxy)
    return freqs, a * a / pxx / pyy


def spectrogram(
    x,
    fs=1.0,
    window=("tukey", 0.25),
    nperseg=None,
    noverlap=None,
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="density",
    axis=-1,
    mode="psd",
):
    """Spectrogram (scipy-compatible: same defaults, incl. tukey window and nperseg//8 overlap)."""
    modelist = ["psd", "complex", "magnitude", "angle", "phase"]
    if mode not in modelist:
        raise ValueError(f"unknown value for mode {mode}, must be one of {modelist}")

    if not _mlx_or_fallback("spectrogram", input_size(x), detrend):
        import scipy.signal as sps

        f, t, s = sps.spectrogram(
            to_numpy(x),
            fs=fs,
            window=_window_arg_np(window),
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            detrend=detrend,
            return_onesided=return_onesided,
            scaling=scaling,
            axis=axis,
            mode=mode,
        )
        return result_to_mlx(f), result_to_mlx(t), result_to_mlx(s)

    x = to_mlx(x)
    # defaults resolve against the analysis-axis length
    window, nperseg = _triage_segments(window, nperseg, input_length=x.shape[axis])
    if noverlap is None:
        noverlap = nperseg // 8

    helper_mode = "psd" if mode == "psd" else "stft"
    freqs, time, sxx = _spectral_helper(
        x, x, fs, window, nperseg, noverlap, nfft, detrend,
        return_onesided, scaling, axis, mode=helper_mode,
    )

    if mode == "magnitude":
        sxx = mx.abs(sxx)
    elif mode in ("angle", "phase"):
        sxx = mx.arctan2(mx.imag(sxx), mx.real(sxx))
        if mode == "phase":
            ax = axis - 1 if axis < 0 else axis
            sxx = _unwrap(sxx, axis=ax)

    return freqs, time, sxx


def stft(
    x,
    fs=1.0,
    window="hann",
    nperseg=256,
    noverlap=None,
    nfft=None,
    detrend=False,
    return_onesided=True,
    boundary="zeros",
    padded=True,
    axis=-1,
    scaling="spectrum",
):
    """Short-time Fourier transform (legacy scipy.signal.stft-compatible)."""
    if scaling == "psd":
        scaling = "density"
    elif scaling != "spectrum":
        raise ValueError(f"Parameter {scaling=} not in ['spectrum', 'psd']!")

    if not _mlx_or_fallback("stft", input_size(x), detrend):
        import scipy.signal as sps

        f, t, z = sps.stft(
            to_numpy(x),
            fs=fs,
            window=_window_arg_np(window),
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            detrend=detrend,
            return_onesided=return_onesided,
            boundary=boundary,
            padded=padded,
            axis=axis,
            scaling="psd" if scaling == "density" else scaling,
        )
        return result_to_mlx(f), result_to_mlx(t), result_to_mlx(z)

    return _spectral_helper(
        x, x, fs, window, nperseg, noverlap, nfft, detrend,
        return_onesided, scaling=scaling, axis=axis, mode="stft",
        boundary=boundary, padded=padded,
    )


def _overlap_add(xsubs_t: mx.array, outputlength: int, nstep: int) -> mx.array:
    """Scatter-add (..., nseg, nperseg) windowed segments into (..., outputlength)."""
    nseg, nperseg = xsubs_t.shape[-2], xsubs_t.shape[-1]
    pos = (
        (nstep * mx.arange(nseg, dtype=mx.int32))[:, None]
        + mx.arange(nperseg, dtype=mx.int32)[None, :]
    ).reshape(-1)
    batch_shape = xsubs_t.shape[:-2]
    v2 = xsubs_t.reshape((-1, nseg * nperseg))
    if v2.dtype == mx.complex64:
        re = mx.zeros((v2.shape[0], outputlength)).at[:, pos].add(mx.real(v2))
        im = mx.zeros((v2.shape[0], outputlength)).at[:, pos].add(mx.imag(v2))
        out = re.astype(mx.complex64) + im.astype(mx.complex64) * mx.array(1j)
    else:
        out = mx.zeros((v2.shape[0], outputlength)).at[:, pos].add(v2)
    return out.reshape(batch_shape + (outputlength,))


def istft(
    Zxx,
    fs=1.0,
    window="hann",
    nperseg=None,
    noverlap=None,
    nfft=None,
    input_onesided=True,
    boundary=True,
    time_axis=-1,
    freq_axis=-2,
    scaling="spectrum",
):
    """Inverse short-time Fourier transform (scipy-compatible).

    Deviation: for N-d input the returned time vector always has the length of
    the time axis (scipy uses ``x.shape[0]``, which is only correct for 1-D).
    """
    if not use_mlx(input_size(Zxx)):
        import scipy.signal as sps

        t, x = sps.istft(
            to_numpy(Zxx),
            fs=fs,
            window=_window_arg_np(window),
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            input_onesided=input_onesided,
            boundary=boundary,
            time_axis=time_axis,
            freq_axis=freq_axis,
            scaling=scaling,
        )
        return result_to_mlx(t), result_to_mlx(x)

    zxx = to_mlx(Zxx)
    if zxx.dtype != mx.complex64:
        zxx = zxx.astype(mx.complex64)
    freq_axis = int(freq_axis)
    time_axis = int(time_axis)

    if zxx.ndim < 2:
        raise ValueError("Input stft must be at least 2d!")
    if freq_axis == time_axis:
        raise ValueError("Must specify differing time and frequency axes!")

    nseg = zxx.shape[time_axis]

    if input_onesided:
        n_default = 2 * (zxx.shape[freq_axis] - 1)
    else:
        n_default = zxx.shape[freq_axis]

    if nperseg is None:
        nperseg = n_default
    else:
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError("nperseg must be a positive integer")

    if nfft is None:
        if input_onesided and (nperseg == n_default + 1):
            nfft = nperseg  # odd nperseg, no FFT padding
        else:
            nfft = n_default
    elif nfft < nperseg:
        raise ValueError("nfft must be greater than or equal to nperseg.")
    else:
        nfft = int(nfft)

    noverlap = nperseg // 2 if noverlap is None else int(noverlap)
    if noverlap >= nperseg:
        raise ValueError("noverlap must be less than nperseg.")
    nstep = nperseg - noverlap

    # rearrange so the layout is (..., freq, time)
    if time_axis != zxx.ndim - 1 or freq_axis != zxx.ndim - 2:
        if freq_axis < 0:
            freq_axis = zxx.ndim + freq_axis
        if time_axis < 0:
            time_axis = zxx.ndim + time_axis
        zouter = list(range(zxx.ndim))
        for ax in sorted([time_axis, freq_axis], reverse=True):
            zouter.pop(ax)
        zxx = mx.transpose(zxx, zouter + [freq_axis, time_axis])

    if isinstance(window, str | tuple):
        key = tuple(window) if isinstance(window, tuple) else window
        win_np = _window_np(key, nperseg)
    else:
        win_np = np.asarray(to_numpy(window), dtype=np.float64)
        if win_np.ndim != 1:
            raise ValueError("window must be 1-D")
        if win_np.shape[0] != nperseg:
            raise ValueError(f"window must have length of {nperseg}")

    ifunc = _sfft.irfft if input_onesided else _sfft.ifft
    xsubs = ifunc(zxx, n=nfft, axis=-2)[..., :nperseg, :]

    if scaling == "spectrum":
        xsubs = xsubs * mx.array(float(win_np.sum()), dtype=mx.float32)
    elif scaling == "psd":
        xsubs = xsubs * mx.array(float(np.sqrt(fs * (win_np**2).sum())), dtype=mx.float32)
    else:
        raise ValueError(f"Parameter {scaling=} not in ['spectrum', 'psd']!")

    outputlength = nperseg + (nseg - 1) * nstep
    win = mx.array(win_np.astype(np.float32))
    xsubs_t = mx.swapaxes(xsubs, -2, -1) * win  # (..., nseg, nperseg)
    x = _overlap_add(xsubs_t, outputlength, nstep)

    w2 = win * win
    pos = (
        (nstep * mx.arange(nseg, dtype=mx.int32))[:, None]
        + mx.arange(nperseg, dtype=mx.int32)[None, :]
    ).reshape(-1)
    norm = mx.zeros((outputlength,)).at[pos].add(
        mx.broadcast_to(w2[None, :], (nseg, nperseg)).reshape(-1)
    )

    if boundary:
        x = x[..., nperseg // 2 : -(nperseg // 2)]
        norm = norm[nperseg // 2 : -(nperseg // 2)]

    norm_np = np.array(norm)
    if np.sum(norm_np > 1e-10) != len(norm_np):
        warnings.warn(
            "NOLA condition failed, STFT may not be invertible."
            + (" Possibly due to missing boundary" if not boundary else ""),
            stacklevel=2,
        )
    x = x / mx.where(norm > 1e-10, norm, mx.ones_like(norm))

    if input_onesided:
        x = mx.real(x)

    n_out = x.shape[-1]
    if x.ndim > 1 and time_axis != zxx.ndim - 1:
        if freq_axis < time_axis:
            time_axis -= 1
        x = mx.moveaxis(x, -1, time_axis)

    time = mx.arange(n_out, dtype=mx.float32) / float(fs)
    return time, x
