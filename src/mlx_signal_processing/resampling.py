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
from ._array import input_size, result_to_mlx, signal_np, to_mlx, to_numpy
from ._config import capability_fallback, use_mlx
from ._upfirdn_metal import upfirdn_gpu
from .convolution import fftconvolve
from .windows import _window_np

__all__ = ["decimate", "resample", "resample_poly", "upfirdn"]


_MLX_VERSION = tuple(int(part) for part in mx.__version__.split(".")[:2])


def _freq_window(window, n_x: int) -> np.ndarray:
    """Resolve the frequency-domain window to 64-bit host values (fftfreq order).

    Complex windows are preserved — scipy applies them as-is.
    """
    if callable(window):
        w = np.asarray(window(np.fft.fftfreq(n_x)))
        if w.shape != (n_x,):
            raise ValueError(
                f"window function returned shape {w.shape}, expected ({n_x},)"
            )
    elif isinstance(window, str | tuple):
        key = tuple(window) if isinstance(window, tuple) else window
        w = np.fft.fftshift(_window_np(key, n_x))
    else:
        w = np.asarray(to_numpy(window))
        if w.shape != (n_x,):
            raise ValueError(
                f"window.shape={w.shape} != ({n_x},), i.e., window length is not "
                "equal to number of frequency bins!"
            )
    dtype = np.complex128 if np.iscomplexobj(w) else np.float64
    return np.array(w, dtype=dtype)


def _win32(w: np.ndarray) -> mx.array:
    """Host window values as the matching 32-bit MLX array."""
    return mx.array(w.astype(np.complex64 if np.iscomplexobj(w) else np.float32))


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
        out = sps.resample(signal_np(x), num, t=to_numpy(t) if t is not None else None,
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
            X = X * _win32(Wf[:n_X])
        X = X[..., :m2]
        if m % 2 == 0 and num != n_x:  # unpaired bin at m//2
            fac = np.ones(m2, dtype=np.float32)
            fac[m // 2] = 2.0 if num < n_x else 0.5
            X = X * mx.array(fac)
        if W is not None and np.iscomplexobj(W):
            # np.irfft ignores the imaginary parts of the DC (and, for even
            # output lengths, Nyquist) bins; MLX's does not — drop them
            # explicitly so a complex window matches scipy
            head = mx.real(X[..., :1]).astype(mx.complex64)
            if num % 2 == 0 and X.shape[-1] == num // 2 + 1:
                X = mx.concatenate(
                    [head, X[..., 1:-1], mx.real(X[..., -1:]).astype(mx.complex64)],
                    axis=-1,
                )
            else:
                X = mx.concatenate([head, X[..., 1:]], axis=-1)
        x_r = _sfft.irfft(X * mx.array(1.0 / s_fac, dtype=mx.float32), n=num, axis=-1)
    else:
        if domain == "time":
            xc = xa if xa.dtype == mx.complex64 else xa.astype(mx.complex64)
            X = _sfft.fft(xc, axis=-1)
        else:
            X = xa if xa.dtype == mx.complex64 else xa.astype(mx.complex64)
        if W is not None:
            X = X * _win32(W)

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


def _has_nonfinite(source, converted: mx.array) -> bool:
    """Whether an input contains NaN/Inf after canonicalization."""
    if isinstance(source, np.ndarray):
        dtype = np.complex64 if converted.dtype == mx.complex64 else np.float32
        with np.errstate(over="ignore", invalid="ignore"):
            canonical = np.asarray(source, dtype=dtype)
        return not np.isfinite(canonical).all()
    if converted.dtype != mx.complex64:
        return bool(np.array(mx.any(~mx.isfinite(converted))))
    return bool(
        np.array(
            mx.any(
                ~mx.isfinite(mx.real(converted))
                | ~mx.isfinite(mx.imag(converted))
            )
        )
    )


def _has_complex_nonfinite(source, converted: mx.array) -> bool:
    """Whether a complex input contains NaN/Inf in either component."""
    return converted.dtype == mx.complex64 and _has_nonfinite(source, converted)


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
    """One kernel launch for every real/complex combination (dtype-templated)."""
    return upfirdn_gpu(x2, h, up, down, n_out)


#: scipy's upfirdn signal-extension modes, all served by pre-extending the
#: signal on-device and running the constant-mode kernel on the extension
_UPFIRDN_MODES = ("constant", "wrap", "edge", "smooth", "symmetric", "reflect",
                  "antisymmetric", "antireflect", "line")


def _extend_plane(x2: mx.array, L: int, R: int, mode: str, cval) -> mx.array:
    """Extend (B, n) planes left by L and right by R with scipy's upfirdn
    boundary modes. Convention trap (verified against scipy sample by
    sample): ``antireflect`` anchors an odd reflection at the edge value
    (2*edge - reflection) like numpy's reflect_type='odd', but
    ``antisymmetric`` is the plain NEGATED symmetric reflection."""
    n = x2.shape[-1]
    B = x2.shape[0]
    if mode == "constant":
        c = mx.full((1, 1), cval).astype(x2.dtype)
        left = mx.broadcast_to(c, (B, L))
        right = mx.broadcast_to(c, (B, R))
    elif mode == "edge":
        left = mx.broadcast_to(x2[:, :1], (B, L))
        right = mx.broadcast_to(x2[:, -1:], (B, R))
    elif mode in ("symmetric", "antisymmetric"):
        left = x2[:, :L][:, ::-1]
        right = x2[:, n - R:][:, ::-1]
        if mode == "antisymmetric":
            left = -left
            right = -right
    elif mode in ("reflect", "antireflect"):
        left = x2[:, 1:L + 1][:, ::-1]
        right = x2[:, n - R - 1:n - 1][:, ::-1]
        if mode == "antireflect":
            if x2.dtype == mx.complex64:
                # NumPy/scipy apply the odd-reflection affine transform to the
                # two stored components independently.  Expressing it as MLX
                # complex arithmetic would introduce generic 0*Inf cross terms.
                def _odd_reflect(edge, reflected):
                    real = 2 * mx.real(edge) - mx.real(reflected)
                    imag = 2 * mx.imag(edge) - mx.imag(reflected)
                    packed = mx.stack([real, imag], axis=-1)
                    return mx.view(packed, mx.complex64).reshape(real.shape)

                left = _odd_reflect(x2[:, :1], left)
                right = _odd_reflect(x2[:, -1:], right)
            else:
                left = 2 * x2[:, :1] - left
                right = 2 * x2[:, -1:] - right
    elif mode == "wrap":
        left = x2[:, n - L:]
        right = x2[:, :R]
    elif mode == "smooth":
        left = x2[:, :1] + (x2[:, 1:2] - x2[:, :1]) * mx.arange(-L, 0, dtype=mx.float32)
        right = x2[:, -1:] + (x2[:, -1:] - x2[:, -2:-1]) * mx.arange(1, R + 1, dtype=mx.float32)
    else:  # line
        slope = (x2[:, -1:] - x2[:, :1]) / (n - 1)
        left = x2[:, :1] + slope * mx.arange(-L, 0, dtype=mx.float32)
        right = x2[:, -1:] + slope * mx.arange(1, R + 1, dtype=mx.float32)
    return mx.concatenate([left, x2, right], axis=-1)


def _extension_lengths(n_taps: int, up: int, down: int) -> tuple[int, int]:
    """(left, right) extension lengths: enough to cover the tap window past
    each edge, with the LEFT length rounded up so left*up is divisible by
    down — otherwise the extended output grid is phase-shifted against the
    unextended one and the slice below returns subtly wrong samples."""
    base = -(-(n_taps - 1) // up)
    d_align = down // math.gcd(up, down)
    left = -(-base // d_align) * d_align
    return left, base


def upfirdn(h, x, up=1, down=1, axis=-1, mode="constant", cval=0):
    """Upsample by ``up``, FIR filter with ``h``, downsample by ``down``.

    scipy-compatible, including every signal-extension ``mode``. On the GPU
    this runs a custom Metal kernel: one thread per output sample, each
    computing one polyphase dot product. Non-constant modes pre-extend the
    signal on-device (a few boundary samples of cheap MLX ops), run the
    constant-mode kernel on the extension, and slice the aligned output
    window back out — only signals shorter than the required extension fall
    back to scipy. One f32 corner: with ``|cval|`` more than ~2^24 times the
    signal scale and partially cancelling boundary taps, the kernel's
    ascending-tap accumulation can round boundary outputs differently from
    scipy's per-phase order (both are valid f32 sums of the same terms).
    """
    up, down = int(up), int(down)
    if up < 1 or down < 1:
        raise ValueError("Both up and down must be >= 1")

    ha = to_mlx(h)
    if ha.ndim != 1 or ha.size == 0:
        raise ValueError("h must be 1-D with non-zero length")
    mode = mode.lower()  # scipy lowercases mode strings
    if mode not in _UPFIRDN_MODES:
        raise ValueError(f"Unknown mode: {mode}")
    # scipy's C-level double cast: complex (even 0j) raises TypeError
    cval = float(cval)

    xa = to_mlx(x)
    n_in = xa.shape[axis]
    n_taps = ha.shape[0]
    n_out = _output_len(n_taps, n_in, up, down)
    if n_in == 0:
        if mode == "constant" and cval != 0:
            # the formula-implied outputs are tap-window sums over the pure
            # cval extension — well-defined, and scipy computes them exactly
            import scipy.signal as sps

            return result_to_mlx(
                sps.upfirdn(to_numpy(ha), to_numpy(xa), up=up, down=down,
                            axis=axis, mode=mode, cval=cval)
            )
        # scipy returns zeros of the formula-implied output length (its
        # reflect-family output on empty input is uninitialized memory)
        shape = list(xa.shape)
        shape[axis] = max(0, n_out)
        dtype = (mx.complex64 if mx.complex64 in (xa.dtype, ha.dtype)
                 else mx.float32)
        return mx.zeros(tuple(shape), dtype=dtype)
    batch = xa.size // n_in
    # per-output-sample dot product length is ~n_taps/up
    work = batch * n_out * max(1, n_taps // up)

    if not use_mlx(work):
        import scipy.signal as sps

        return result_to_mlx(
            sps.upfirdn(to_numpy(ha), to_numpy(xa), up=up, down=down, axis=axis,
                        mode=mode, cval=cval)
        )

    if _has_nonfinite(h, ha):
        # scipy multiplies non-finite taps by the implicit zero extension past
        # each signal edge; the optimized kernel skips those mathematically-zero
        # products.  Preserve scipy's exceptional NaN/Inf masks wholesale.
        capability_fallback("upfirdn", "non-finite filter taps")
        import scipy.signal as sps

        return result_to_mlx(
            sps.upfirdn(to_numpy(ha), to_numpy(xa), up=up, down=down, axis=axis,
                        mode=mode, cval=cval)
        )

    if not mx.metal.is_available() and _has_nonfinite(x, xa):
        # The CPU-composed fallback uses FFT convolution, where a single
        # non-finite contaminates the whole transform.  scipy's direct
        # polyphase loop has local, generic-complex propagation instead.
        capability_fallback("upfirdn", "non-finite input without Metal")
        import scipy.signal as sps

        return result_to_mlx(
            sps.upfirdn(to_numpy(ha), to_numpy(xa), up=up, down=down, axis=axis,
                        mode=mode, cval=cval)
        )

    ext = not (mode == "constant" and cval == 0)
    if ext:
        L, R = _extension_lengths(n_taps, up, down)
        if L == 0 and R == 0:
            ext = False  # a 1-tap filter never reaches past the edges
        else:
            constructible = (
                mode in ("constant", "edge")
                or (mode in ("smooth", "line") and n_in >= 2)
                or (mode in ("symmetric", "antisymmetric", "wrap")
                    and max(L, R) <= n_in)
                or (mode in ("reflect", "antireflect") and max(L, R) <= n_in - 1)
            )
            if not constructible:
                # multi-fold reflection of a signal shorter than the filter's
                # boundary reach; rare, and scipy synthesizes it exactly
                capability_fallback(
                    "upfirdn",
                    f"mode={mode!r} with a signal shorter than the "
                    f"{max(L, R)}-sample boundary extension",
                )
                import scipy.signal as sps

                return result_to_mlx(
                    sps.upfirdn(to_numpy(ha), to_numpy(xa), up=up, down=down,
                                axis=axis, mode=mode, cval=cval)
                )

    moved = xa.ndim > 1
    if moved:
        xa = mx.moveaxis(xa, axis, -1)
    batch_shape = xa.shape[:-1]
    x2 = xa.reshape(-1, n_in) if xa.ndim != 2 else xa

    run_n_out = n_out
    if ext:
        x2 = _extend_plane(x2, L, R, mode, cval)
        run_n_out = _output_len(n_taps, n_in + L + R, up, down)

    if mx.metal.is_available():
        out = _upfirdn_plane_dispatch(x2, ha, up, down, run_n_out)
    else:
        out = _upfirdn_composed(x2, ha, up, down)

    if ext:
        off = (L * up) // down
        out = out[..., off:off + n_out]

    out = out.reshape(batch_shape + (n_out,))
    if moved:
        out = mx.moveaxis(out, -1, axis)
    return out


def resample_poly(x, up, down, axis=0, window=("kaiser", 5.0), padtype="constant",
                  cval=None):
    """Polyphase resampling by the rational factor ``up/down`` (scipy-compatible).

    The anti-aliasing FIR filter is designed host-side exactly like scipy
    (``firwin(2*10*max(up, down)+1, 1/max(up, down), window)``), then applied
    with the GPU upfirdn kernel. Every scipy ``padtype`` is supported: the
    statistical ones (``mean``/``median``/``maximum``/``minimum``) subtract a
    per-channel background around a zero-padded pass exactly like scipy, and
    the signal-extension ones ride upfirdn's on-device boundary extension.
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

    g = math.gcd(up, down)
    up //= g
    down //= g
    xa = to_mlx(x)
    if up == down == 1:
        return xa * 1  # scipy returns a copy (before even validating padtype)

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
        w_np = to_numpy(window)
        h_np = np.array(w_np, dtype=np.complex128 if np.iscomplexobj(w_np)
                        else np.float64)
        if h_np.ndim != 1:
            raise ValueError("window must be 1-D")
        half_len = (h_np.size - 1) // 2
    h_np = h_np * up

    # padtype is validated after axis and window, matching scipy's error order
    stat_padtypes = ("mean", "median", "maximum", "minimum")
    if padtype not in stat_padtypes and padtype not in _UPFIRDN_MODES:
        raise ValueError(
            "padtype must be one of: "
            + ", ".join(sorted(stat_padtypes) + list(_UPFIRDN_MODES))
        )

    # Zero-pad the filter so output samples land at the center.  Do this before
    # the statistical-background branch so an exceptional SciPy fallback can
    # honor the actual size/dispatch decision of the filtering work.
    n_pre_pad = down - half_len % down
    n_post_pad = 0
    n_pre_remove = (half_len + n_pre_pad) // down
    while _output_len(len(h_np) + n_pre_pad + n_post_pad, n_in, up, down) < (
        n_out + n_pre_remove
    ):
        n_post_pad += 1
    h_full = np.concatenate([
        np.zeros(n_pre_pad, dtype=h_np.dtype), h_np,
        np.zeros(n_post_pad, dtype=h_np.dtype),
    ])
    n_pre_remove_end = n_pre_remove + n_out
    filter_n_out = _output_len(len(h_full), n_in, up, down)
    batch = xa.size // n_in if n_in else 0
    filter_work = batch * filter_n_out * max(1, len(h_full) // up)

    background = None
    upfirdn_mode, upfirdn_cval = "constant", 0
    if padtype in stat_padtypes:
        if _has_complex_nonfinite(x, xa):
            if use_mlx(filter_work):
                # scipy's generic complex multiply propagates a NaN in either
                # component (including one created by infinity arithmetic) to
                # both output components.  Although the Metal polyphase kernel
                # does likewise, MLX's complex statistical reductions do not
                # match every non-finite ordering corner, so retain scipy for
                # the complete operation.
                capability_fallback(
                    "resample_poly",
                    f"complex non-finite input with padtype={padtype!r}",
                )
            import scipy.signal as sps

            win = window if isinstance(window, str | tuple) else to_numpy(window)
            return result_to_mlx(
                sps.resample_poly(
                    to_numpy(xa), up, down, axis=axis, window=win,
                    padtype=padtype, cval=cval,
                )
            )
        # scipy subtracts a per-channel statistic, runs the zero-padded
        # polyphase pass, and adds it back after the keep-slice. MLX's
        # sort/min/max order complex64 like numpy (real part, then imaginary),
        # so complex signals stay on-device too.
        if padtype == "mean":
            background = mx.mean(xa, axis=axis, keepdims=True)
        elif padtype == "maximum":
            background = mx.max(xa, axis=axis, keepdims=True)
        elif padtype == "minimum":
            background = mx.min(xa, axis=axis, keepdims=True)
        else:  # median, like np.median: mean of the two middle order statistics
            if xa.dtype == mx.complex64 and _MLX_VERSION < (0, 31):
                # MLX 0.30 advertises complex sort but its Metal block-sort
                # kernel is unavailable and aborts at evaluation time.
                background = mx.array(
                    np.median(to_numpy(xa), axis=axis, keepdims=True).astype(np.complex64)
                )
            else:
                s = mx.sort(xa, axis=axis)
                mid = mx.take(s, mx.array([n_in // 2]), axis=axis)
                if n_in % 2 == 0:
                    mid = (mid + mx.take(s, mx.array([n_in // 2 - 1]), axis=axis)) / 2
                if xa.dtype != mx.complex64:
                    # np.median propagates NaN (sorting alone hides it at the end)
                    any_nan = mx.any(mx.isnan(xa), axis=axis, keepdims=True)
                    mid = mx.where(any_nan, mx.array(np.nan, dtype=mid.dtype), mid)
                background = mid
        xa = xa - background
    else:
        upfirdn_mode = padtype
        upfirdn_cval = 0 if cval is None else cval

    h32 = h_full.astype(np.complex64 if np.iscomplexobj(h_full) else np.float32)
    y = upfirdn(h32, xa, up, down, axis=axis, mode=upfirdn_mode, cval=upfirdn_cval)
    keep = [slice(None)] * y.ndim
    keep[axis] = slice(n_pre_remove, n_pre_remove_end)
    y = y[tuple(keep)]
    return y + background if background is not None else y


def decimate(x, q, n=None, ftype="iir", axis=-1, zero_phase=True):
    """Downsample by an integer factor after an anti-aliasing filter.

    scipy-compatible signature. The FIR path (``ftype="fir"``) runs on the GPU
    (a hamming-window ``firwin(20*q+1, 1/q)`` filter applied via upfirdn); the
    default ``ftype="iir"`` (order-8 Chebyshev I) runs through the batched
    GPU :func:`~mlx_signal_processing.sosfiltfilt`/:func:`~mlx_signal_processing.sosfilt` kernel.
    ``dlti`` instances fall back to scipy.
    """
    q = int(q)
    if q < 1:
        raise ValueError("q must be a positive integer")

    if ftype == "iir":
        from scipy.signal import cheby1

        from .filtering import _validate_sos_np, sosfilt, sosfiltfilt

        if n is None:
            n = 8
        # This design is internal, not a user-supplied f64 coefficient array:
        # canonicalize and validate it without tripping float64="strict". The
        # public sosfilt call below sees the resulting f32 SOS.
        sos = _validate_sos_np(
            cheby1(int(n), 0.05, 0.8 / q, output="sos"), apply_dtype_policy=False
        )
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
            sps.decimate(signal_np(x), q, n=n, ftype=ftype, axis=axis,
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
