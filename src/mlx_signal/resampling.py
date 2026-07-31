"""Sample-rate conversion: resample (FFT method); polyphase functions land here too.

``resample`` is a faithful MLX port of scipy's FFT-domain algorithm, including
the unpaired-Nyquist-bin handling for even lengths and frequency-domain
windowing. Note scipy's historical default ``axis=0`` (unlike most of
scipy.signal), preserved here for compatibility.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from . import _fft_core as _sfft
from ._array import input_size, result_to_mlx, to_mlx, to_numpy
from ._config import capability_fallback, use_mlx
from ._upfirdn_metal import upfirdn_gpu
from .convolution import fftconvolve
from .windows import _window_np

__all__ = ["decimate", "resample", "resample_poly", "upfirdn"]


def _freq_window(window, n_x: int) -> np.ndarray:
    """Resolve the frequency-domain window to float64 host values (fftfreq order)."""
    if callable(window):
        w = np.asarray(window(np.fft.fftfreq(n_x)), dtype=np.float64)
        if w.shape != (n_x,):
            raise ValueError(
                f"window function returned shape {w.shape}, expected ({n_x},)"
            )
        return w
    if isinstance(window, str | tuple):
        key = tuple(window) if isinstance(window, tuple) else window
        return np.fft.fftshift(_window_np(key, n_x)).copy()
    w = np.asarray(to_numpy(window), dtype=np.float64)
    if w.shape != (n_x,):
        raise ValueError(
            f"window.shape={w.shape} != ({n_x},), i.e., window length is not "
            "equal to number of frequency bins!"
        )
    return w.copy()


def resample(x, num, t=None, axis=0, window=None, domain="time"):
    """Resample ``x`` to ``num`` samples using the Fourier method (scipy-compatible).

    Returns ``x_resampled`` or ``(x_resampled, new_t)`` when ``t`` is given.
    """
    if domain not in ("time", "freq"):
        raise ValueError(f"Parameter {domain=} not in ('time', 'freq')!")
    num = int(num)

    if not use_mlx(max(input_size(x), num)):
        import scipy.signal as sps

        win = window
        if win is not None and not (isinstance(win, str | tuple) or callable(win)):
            win = to_numpy(win)
        out = sps.resample(to_numpy(x), num, t=to_numpy(t) if t is not None else None,
                           axis=axis, window=win, domain=domain)
        if t is not None:
            return result_to_mlx(out[0]), result_to_mlx(out[1])
        return result_to_mlx(out)

    xa = to_mlx(x)
    moved = xa.ndim > 1
    if moved:
        xa = mx.moveaxis(xa, axis, -1)
    n_x = xa.shape[-1]
    s_fac = n_x / num  # sample-interval dilatation
    m = min(num, n_x)  # relevant frequency bins
    m2 = m // 2 + 1  # relevant one-sided bins

    W = None if window is None else _freq_window(window, n_x)

    real_input = xa.dtype != mx.complex64
    if domain == "time" and real_input:
        X = _sfft.rfft(xa, axis=-1)
        n_X = X.shape[-1]
        if W is not None:
            # fold the two-sided window onto the one-sided spectrum
            Wf = W.copy()
            Wf[1:n_X] += np.flip(Wf[-n_X + 1 :])
            Wf[1:n_X] /= 2
            X = X * mx.array(Wf[:n_X].astype(np.float32))
        X = X[..., :m2]
        if m % 2 == 0 and num != n_x:  # unpaired bin at m//2
            fac = np.ones(m2, dtype=np.float32)
            fac[m // 2] = 2.0 if num < n_x else 0.5
            X = X * mx.array(fac)
        x_r = _sfft.irfft(X * mx.array(1.0 / s_fac, dtype=mx.float32), n=num, axis=-1)
    else:
        if domain == "time":
            xc = xa if xa.dtype == mx.complex64 else xa.astype(mx.complex64)
            X = _sfft.fft(xc, axis=-1)
        else:
            X = xa if xa.dtype == mx.complex64 else xa.astype(mx.complex64)
        if W is not None:
            X = X * mx.array(W.astype(np.float32))

        parts: list[mx.array] = []
        first = X[..., :m2]
        if num > n_x:  # upsampling: split the unpaired bin into a pair
            if m % 2 == 0:
                fac = np.ones(m2, dtype=np.float32)
                fac[-1] = 0.5
                first = first * mx.array(fac)
                parts = [first]
                nz = num - m - 1
                if nz:
                    parts.append(mx.zeros(X.shape[:-1] + (nz,), dtype=X.dtype))
                parts.append(first[..., m2 - 1 : m2])  # mirrored half-bin
            else:
                parts = [first]
                nz = num - m
                if nz:
                    parts.append(mx.zeros(X.shape[:-1] + (nz,), dtype=X.dtype))
            if m2 < m:
                parts.append(X[..., m2 - m :])
        elif num < n_x:  # downsampling: unite the bin pair into one unpaired bin
            if m % 2 == 0:
                add = X[..., n_x - m // 2 : n_x - m // 2 + 1]
                first = mx.concatenate(
                    [first[..., : m2 - 1], first[..., m2 - 1 : m2] + add], axis=-1
                )
            parts = [first]
            if m2 < m:
                parts.append(X[..., m2 - m :])
        else:  # num == n_x: pass through (with window applied)
            parts = [first]
            if m2 < m:
                parts.append(X[..., m2 - m :])
        Y = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=-1)
        x_r = _sfft.ifft(Y * mx.array(1.0 / s_fac, dtype=mx.float32), n=num, axis=-1)

    if moved:
        x_r = mx.moveaxis(x_r, -1, axis)
    if t is not None:
        t_np = np.asarray(to_numpy(t), dtype=np.float64)
        new_t = t_np[0] + (t_np[1] - t_np[0]) * s_fac * np.arange(num)
        return x_r, mx.array(new_t.astype(np.float32))
    return x_r


# ---------------------------------------------------------------------------
# polyphase resampling
# ---------------------------------------------------------------------------


def _output_len(len_h: int, in_len: int, up: int, down: int) -> int:
    """Output length of upfirdn (scipy.signal._upfirdn._output_len)."""
    return (((in_len - 1) * up + len_h) - 1) // down + 1


def _upfirdn_composed(x2: mx.array, h: mx.array, up: int, down: int) -> mx.array:
    """upfirdn via zero-stuff + FFT convolution + strided slice, in MLX ops.

    Correct for real and complex data; used when no Metal GPU is available
    (e.g. CI) and as an independent cross-check of the custom kernel.
    """
    batch, n = x2.shape
    if up > 1:
        z = mx.zeros((batch, n, up - 1), dtype=x2.dtype)
        xu = mx.concatenate([x2[..., None], z], axis=-1).reshape(batch, n * up)
    else:
        xu = x2
    full = fftconvolve(xu, h[None, :], mode="full")
    full = full[..., : (n - 1) * up + h.shape[0]]
    return full[..., ::down]


def _upfirdn_plane_dispatch(x2: mx.array, h: mx.array, up: int, down: int,
                            n_out: int) -> mx.array:
    """Split complex signals/filters into float32 kernel launches (conv is linear)."""
    cx = x2.dtype == mx.complex64
    ch = h.dtype == mx.complex64
    j = mx.array(1j)
    if not cx and not ch:
        return upfirdn_gpu(x2, h, up, down, n_out)
    if cx and not ch:
        re = upfirdn_gpu(mx.real(x2), h, up, down, n_out)
        im = upfirdn_gpu(mx.imag(x2), h, up, down, n_out)
        return re.astype(mx.complex64) + im.astype(mx.complex64) * j
    if ch and not cx:
        re = upfirdn_gpu(x2, mx.real(h), up, down, n_out)
        im = upfirdn_gpu(x2, mx.imag(h), up, down, n_out)
        return re.astype(mx.complex64) + im.astype(mx.complex64) * j
    xr, xi = mx.real(x2), mx.imag(x2)
    hr, hi = mx.real(h), mx.imag(h)
    rr = upfirdn_gpu(xr, hr, up, down, n_out)
    ii = upfirdn_gpu(xi, hi, up, down, n_out)
    ri = upfirdn_gpu(xr, hi, up, down, n_out)
    ir = upfirdn_gpu(xi, hr, up, down, n_out)
    return (rr - ii).astype(mx.complex64) + (ri + ir).astype(mx.complex64) * j


def upfirdn(h, x, up=1, down=1, axis=-1, mode="constant", cval=0):
    """Upsample by ``up``, FIR filter with ``h``, downsample by ``down``.

    scipy-compatible. On the GPU this runs a custom Metal kernel: one thread
    per output sample, each computing one polyphase dot product, with filter
    taps staged in threadgroup memory. Signal-extension modes other than
    zero-padded ``"constant"`` fall back to scipy.
    """
    up, down = int(up), int(down)
    if up < 1 or down < 1:
        raise ValueError("Both up and down must be >= 1")

    ha = to_mlx(h)
    if ha.ndim != 1 or ha.size == 0:
        raise ValueError("h must be 1-D with non-zero length")

    if mode != "constant" or cval not in (0, 0.0, None):
        capability_fallback("upfirdn", f"signal extension mode={mode!r}")
        import scipy.signal as sps

        return result_to_mlx(
            sps.upfirdn(to_numpy(h), to_numpy(x), up=up, down=down, axis=axis,
                        mode=mode, cval=cval)
        )

    xa = to_mlx(x)
    n_in = xa.shape[axis]
    if n_in == 0:
        raise ValueError("x must have at least one sample along axis")
    n_taps = ha.shape[0]
    n_out = _output_len(n_taps, n_in, up, down)
    batch = xa.size // n_in
    # per-output-sample dot product length is ~n_taps/up
    work = batch * n_out * max(1, n_taps // up)

    if not use_mlx(work):
        import scipy.signal as sps

        return result_to_mlx(
            sps.upfirdn(to_numpy(h), to_numpy(x), up=up, down=down, axis=axis)
        )

    moved = xa.ndim > 1
    if moved:
        xa = mx.moveaxis(xa, axis, -1)
    batch_shape = xa.shape[:-1]
    x2 = xa.reshape(-1, n_in) if xa.ndim != 2 else xa

    if mx.metal.is_available():
        out = _upfirdn_plane_dispatch(x2, ha, up, down, n_out)
    else:
        out = _upfirdn_composed(x2, ha, up, down)

    out = out.reshape(batch_shape + (n_out,))
    if moved:
        out = mx.moveaxis(out, -1, axis)
    return out


def resample_poly(x, up, down, axis=0, window=("kaiser", 5.0), padtype="constant",
                  cval=None):
    """Polyphase resampling by the rational factor ``up/down`` (scipy-compatible).

    The anti-aliasing FIR filter is designed host-side exactly like scipy
    (``firwin(2*10*max(up, down)+1, 1/max(up, down), window)``), then applied
    with the GPU upfirdn kernel. ``padtype`` values other than the default
    zero-padded ``"constant"`` fall back to scipy for now.
    """
    if up != int(up):
        raise ValueError("up must be an integer")
    if down != int(down):
        raise ValueError("down must be an integer")
    up, down = int(up), int(down)
    if up < 1 or down < 1:
        raise ValueError("up and down must be >= 1")
    if cval is not None and padtype != "constant":
        raise ValueError(f"cval has no effect when padtype is {padtype!r}")

    if padtype != "constant" or cval not in (None, 0, 0.0):
        capability_fallback("resample_poly", f"padtype={padtype!r}/cval={cval!r}")
        import scipy.signal as sps

        win = window if isinstance(window, str | tuple) else to_numpy(window)
        return result_to_mlx(
            sps.resample_poly(to_numpy(x), up, down, axis=axis, window=win,
                              padtype=padtype, cval=cval)
        )

    g = math.gcd(up, down)
    up //= g
    down //= g
    xa = to_mlx(x)
    if up == down == 1:
        return xa * 1  # scipy returns a copy

    n_in = xa.shape[axis]
    n_out = n_in * up
    n_out = n_out // down + bool(n_out % down)

    if isinstance(window, str | tuple):
        from scipy.signal import firwin

        max_rate = max(up, down)
        f_c = 1.0 / max_rate
        half_len = 10 * max_rate
        h_np = np.asarray(firwin(2 * half_len + 1, f_c, window=window), dtype=np.float64)
    else:
        h_np = np.array(to_numpy(window), dtype=np.float64)
        if h_np.ndim != 1:
            raise ValueError("window must be 1-D")
        half_len = (h_np.size - 1) // 2
    h_np = h_np * up

    # zero-pad the filter so output samples land at the center
    n_pre_pad = down - half_len % down
    n_post_pad = 0
    n_pre_remove = (half_len + n_pre_pad) // down
    while _output_len(len(h_np) + n_pre_pad + n_post_pad, n_in, up, down) < (
        n_out + n_pre_remove
    ):
        n_post_pad += 1
    h_full = np.concatenate([np.zeros(n_pre_pad), h_np, np.zeros(n_post_pad)])
    n_pre_remove_end = n_pre_remove + n_out

    y = upfirdn(h_full.astype(np.float32), xa, up, down, axis=axis)
    keep = [slice(None)] * y.ndim
    keep[axis] = slice(n_pre_remove, n_pre_remove_end)
    return y[tuple(keep)]


def decimate(x, q, n=None, ftype="iir", axis=-1, zero_phase=True):
    """Downsample by an integer factor after an anti-aliasing filter.

    scipy-compatible signature. The FIR path (``ftype="fir"``) runs on the GPU
    (a hamming-window ``firwin(20*q+1, 1/q)`` filter applied via upfirdn); the
    default ``ftype="iir"`` (order-8 Chebyshev I) runs through the batched
    GPU :func:`~mlx_signal.sosfiltfilt`/:func:`~mlx_signal.sosfilt` kernel.
    ``dlti`` instances fall back to scipy.
    """
    q = int(q)
    if q < 1:
        raise ValueError("q must be a positive integer")

    if ftype == "iir":
        from scipy.signal import cheby1

        from .filtering import sosfilt, sosfiltfilt

        if n is None:
            n = 8
        sos = cheby1(int(n), 0.05, 0.8 / q, output="sos")
        if zero_phase:
            y = sosfiltfilt(sos, x, axis=axis)
        else:
            y = sosfilt(sos, x, axis=axis)
        sl = [slice(None)] * y.ndim
        sl[axis] = slice(None, None, q)
        return y[tuple(sl)]
    if ftype != "fir":
        capability_fallback("decimate", f"ftype={ftype!r} (dlti systems have no MLX path)")
        import scipy.signal as sps

        return result_to_mlx(
            sps.decimate(to_numpy(x), q, n=n, ftype=ftype, axis=axis,
                         zero_phase=zero_phase)
        )

    if n is None:
        half_len = 10 * q
        n = 2 * half_len
    n = int(n)

    from scipy.signal import firwin

    b = np.asarray(firwin(n + 1, 1.0 / q, window="hamming"), dtype=np.float64)

    xa = to_mlx(x)
    if zero_phase:
        return resample_poly(xa, 1, q, axis=axis, window=b)
    n_in = xa.shape[axis]
    n_out = n_in // q + bool(n_in % q)
    y = upfirdn(b.astype(np.float32), xa, up=1, down=q, axis=axis)
    sl = [slice(None)] * y.ndim
    sl[axis] = slice(None, n_out)
    return y[tuple(sl)]
