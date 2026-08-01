"""Filtering: FIR design (host-side), FIR/IIR application (GPU), hilbert.

FIR paths run on the GPU via FFT convolution. IIR in second-order-section form
(:func:`sosfilt`/:func:`sosfiltfilt`) runs on the GPU through two custom Metal
kernels — per-channel-sequential for short signals, block-parallel associative
scan for long ones. Transfer-function IIR (``lfilter`` with ``len(a) > 1``)
falls back to scipy with a :class:`~mlx_signal.FallbackWarning`; prefer the
better-conditioned SOS form, as scipy itself recommends.
"""

from __future__ import annotations

import functools as _functools

import mlx.core as mx
import numpy as np

from . import _fft_core as _sfft
from ._array import input_size, result_to_mlx, signal_np, to_mlx, to_numpy
from ._arraytools import const_ext, even_ext, odd_ext
from ._config import capability_fallback, get_config, use_mlx
from ._sosfilt_metal import SCAN_BLOCK as _SCAN_BLOCK_REF
from .convolution import oaconvolve

__all__ = ["filtfilt", "firwin", "firwin2", "hilbert", "lfilter", "sosfilt", "sosfiltfilt"]


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

        return result_to_mlx(sps.hilbert(signal_np(x), N=N, axis=axis))

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
    work = input_size(x) * (1 if taps is None else max(1, taps.size))
    if taps is None or zi is not None or not use_mlx(work):
        if use_mlx(work):  # not a size decision: explain why we left the GPU
            reason = "zi initial conditions" if taps is not None else "IIR filtering (len(a) > 1)"
            capability_fallback("lfilter", reason)
        import scipy.signal as sps

        out = sps.lfilter(to_numpy(b), to_numpy(a), signal_np(x), axis=axis, zi=zi)
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
    work = input_size(x) * (1 if taps is None else max(1, taps.size))
    if taps is None or method == "gust" or not use_mlx(work):
        if use_mlx(work):
            reason = "method='gust'" if taps is not None else "IIR filtering (len(a) > 1)"
            capability_fallback("filtfilt", reason)
        import scipy.signal as sps

        return result_to_mlx(
            sps.filtfilt(to_numpy(b), to_numpy(a), signal_np(x), axis=axis,
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


# ---------------------------------------------------------------------------
# IIR: second-order sections (batched-channel GPU kernel)
# ---------------------------------------------------------------------------

#: signals at least this long use the block-parallel scan kernel (parallel over
#: time as well as channels); shorter ones use the per-channel-sequential
#: kernel, which under dispatch="auto" needs this many rows to beat scipy
_SCAN_MIN_N = 2 * _SCAN_BLOCK_REF
_SOSFILT_MIN_ROWS = 32


def _validate_sos_np(sos) -> np.ndarray:
    sos = np.atleast_2d(np.asarray(to_numpy(sos)))
    # canonical coefficient dtype: everything downstream (the scan-safety
    # cache key, A^L precompute, scipy fallbacks) assumes 8-byte scalars
    sos = sos.astype(np.complex128 if np.iscomplexobj(sos) else np.float64)
    if sos.ndim != 2:
        raise ValueError("sos array must be 2D")
    if sos.shape[1] != 6:
        raise ValueError("sos array must be shape (n_sections, 6)")
    if sos.shape[0] < 1:
        raise ValueError("sos array must have at least one section")
    if not np.all(sos[:, 3] == 1):
        raise ValueError("sos[:, 3] should be all ones")
    return sos


@_functools.lru_cache(maxsize=64)
def _scan_safe(sos_bytes: bytes, n_sections: int) -> bool:
    """Scan-dispatch gate: every section's impulse response must decay to
    (near) nothing within one scan block.

    The block scan composes per-block A^L transitions in float32; when a
    pole's response outlives a block (radius**L not small), those compositions
    round differently from the sequential recurrence and the output can drift
    by percents — a radius-only near-unit-circle check misses this (e.g.
    butter(2, 2e-4): radius 0.99956, 16% L2 drift). Such filters stay on the
    exact per-channel-sequential kernel."""
    sos = np.frombuffer(sos_bytes, dtype=np.float64).reshape(n_sections, 6)
    for a1, a2 in sos[:, 4:6].real:
        radius = np.max(np.abs(np.roots([1.0, a1, a2]))) if (a1 or a2) else 0.0
        if radius >= 1.0 or radius**_SCAN_BLOCK_REF > 1e-2:
            return False
    return True


def _zi_expected_shape(n_sections, x_shape, axis):
    shape = list(x_shape)
    shape[axis] = 2
    return (n_sections, *shape)


def sosfilt(sos, x, axis=-1, zi=None):
    """Filter data along one dimension using cascaded second-order sections.

    scipy-compatible (direct form II transposed, identical ``zi``/``zf``
    contract). Signals longer than ~2k samples run the block-parallel scan
    kernel — the cascade is a linear system, so blocks compute their
    contributions in parallel and entry states compose through the
    precomputed A^L transition — which is parallel over time as well as
    channels and beats scipy from a single channel up (7x at 1ch, 100x at
    256ch on M4 Max). Short signals use a per-channel-sequential kernel when
    there are enough channels, else scipy. Coefficients are applied in
    float32 (like every array in this library) — well within tolerance for
    SOS cascades of reasonable Q, which is exactly what the SOS factorization
    is for.
    """
    from . import _sosfilt_metal

    sos_np = _validate_sos_np(sos)
    n_sections = sos_np.shape[0]
    xa = to_mlx(x)
    if xa.ndim < 1 or xa.shape[axis] == 0:
        raise ValueError("x must be at least 1-D with samples along axis")
    n = xa.shape[axis]
    batch = xa.size // n

    if zi is not None:
        zi_a = to_mlx(zi)
        expected = _zi_expected_shape(n_sections, xa.shape, axis % xa.ndim)
        if tuple(zi_a.shape) != expected:
            raise ValueError(
                f"Invalid zi shape. With axis={axis!r}, an input with shape "
                f"{tuple(xa.shape)}, and an sos array with {n_sections} "
                f"sections, zi must have shape {expected}, got {tuple(zi_a.shape)}."
            )

    complex_sos = np.iscomplexobj(sos_np)
    kernel_ok = (
        mx.metal.is_available()
        and not complex_sos
        and n_sections <= _sosfilt_metal.MAX_SECTIONS
    )

    def _scipy_path():
        import scipy.signal as sps

        out = sps.sosfilt(sos_np, signal_np(x), axis=axis,
                          zi=to_numpy(zi) if zi is not None else None)
        if zi is not None:
            return result_to_mlx(out[0]), result_to_mlx(out[1])
        return result_to_mlx(out)

    if not use_mlx(xa.size * n_sections):
        return _scipy_path()
    if not kernel_ok:
        reason = ("complex sos coefficients" if complex_sos
                  else f"more than {_sosfilt_metal.MAX_SECTIONS} sections"
                  if n_sections > _sosfilt_metal.MAX_SECTIONS
                  else "no Metal GPU")
        capability_fallback("sosfilt", reason)
        return _scipy_path()
    use_scan = n >= _SCAN_MIN_N and _scan_safe(sos_np.tobytes(), n_sections)
    if (
        get_config().dispatch == "auto"
        and not use_scan
        and batch < _SOSFILT_MIN_ROWS
    ):
        # short signal, few channels: scipy's single fast core wins
        return _scipy_path()

    sos_flat = mx.array(np.ascontiguousarray(sos_np, dtype=np.float32).reshape(-1))

    def _run(x2p, zi2p):
        if use_scan:
            return _sosfilt_metal.sosfilt_scan_gpu(x2p, sos_np, zi2p)
        return _sosfilt_metal.sosfilt_gpu(x2p, sos_flat, zi2p)

    ax = axis % xa.ndim
    moved = xa.ndim > 1 and ax != xa.ndim - 1
    if moved:
        xa = mx.moveaxis(xa, ax, -1)
    batch_shape = xa.shape[:-1]
    x2 = xa.reshape(-1, n)

    if zi is not None:
        # (S, ..., 2-at-axis) -> (B, S, 2)
        zi_m = zi_a
        if moved:
            zi_m = mx.moveaxis(zi_m, ax + 1, -1)
        zi_m = mx.moveaxis(zi_m, 0, -2)  # (..., S, 2)
        zi2 = zi_m.reshape(-1, n_sections, 2)
    else:
        zi2 = mx.zeros((x2.shape[0], n_sections, 2))

    if x2.dtype == mx.complex64:
        if zi is not None and zi2.dtype != mx.complex64:
            zi2 = zi2.astype(mx.complex64)
        yr, zr = _run(mx.real(x2), mx.real(zi2))
        yi, zim = _run(mx.imag(x2), mx.imag(zi2))
        j = mx.array(1j)
        y2 = yr.astype(mx.complex64) + yi.astype(mx.complex64) * j
        zf2 = zr.astype(mx.complex64) + zim.astype(mx.complex64) * j
    else:
        if zi is not None and zi2.dtype == mx.complex64:
            raise ValueError("complex zi requires complex x")
        y2, zf2 = _run(x2, zi2)

    y = y2.reshape(batch_shape + (n,))
    if moved:
        y = mx.moveaxis(y, -1, ax)
    if zi is None:
        return y
    zf = zf2.reshape(batch_shape + (n_sections, 2))
    zf = mx.moveaxis(zf, -2, 0)  # (S, ..., 2)
    if moved:
        zf = mx.moveaxis(zf, -1, ax + 1)
    return y, zf


def sosfiltfilt(sos, x, axis=-1, padtype="odd", padlen=None):
    """Zero-phase forward-backward IIR filtering with second-order sections.

    scipy-compatible port: odd/even/constant edge extension, ``sosfilt_zi``
    steady-state initialization scaled by the edge samples, forward and
    reverse passes, trim. Both passes run through :func:`sosfilt`, so the GPU
    kernel (or the scipy routing) applies to each.
    """
    sos_np = _validate_sos_np(sos)
    n_sections = sos_np.shape[0]

    if np.iscomplexobj(sos_np):
        capability_fallback("sosfiltfilt", "complex sos coefficients")
    if np.iscomplexobj(sos_np) or not use_mlx(input_size(x) * n_sections):
        import scipy.signal as sps

        return result_to_mlx(
            sps.sosfiltfilt(sos_np, signal_np(x), axis=axis, padtype=padtype,
                            padlen=padlen)
        )

    if padtype not in ("even", "odd", "constant", None):
        raise ValueError(
            f"Unknown value '{padtype}' given to padtype. padtype must be 'even', "
            "'odd', 'constant', or None."
        )

    ntaps = 2 * n_sections + 1
    ntaps -= min(int((sos_np[:, 2] == 0).sum()), int((sos_np[:, 5] == 0).sum()))
    if padtype is None:
        edge = 0
    elif padlen is None:
        edge = ntaps * 3
    else:
        edge = int(padlen)

    xa = to_mlx(x)
    ax = axis % xa.ndim
    if xa.shape[ax] <= edge:
        raise ValueError(
            f"The length of the input vector x must be greater than padlen, "
            f"which is {edge}."
        )

    moved = xa.ndim > 1 and ax != xa.ndim - 1
    if moved:
        xa = mx.moveaxis(xa, ax, -1)

    if edge > 0:
        ext_func = {"even": even_ext, "odd": odd_ext, "constant": const_ext}[padtype]
        ext = ext_func(xa, edge)
    else:
        ext = xa

    from scipy.signal import sosfilt_zi as _sosfilt_zi

    from . import _sosfilt_metal

    # the GPU kernel runs float32-rounded coefficients; the steady-state zi
    # must describe THAT filter, or narrowband cascades start visibly wrong
    sos_zi = sos_np
    if mx.metal.is_available() and n_sections <= _sosfilt_metal.MAX_SECTIONS:
        sos_zi = sos_np.astype(np.float32).astype(np.float64)
    zi_np = np.asarray(_sosfilt_zi(sos_zi), dtype=np.float64)  # (S, 2)
    zi_shape = [1] * ext.ndim
    zi_shape[-1] = 2
    zi = mx.array(zi_np.astype(np.float32)).reshape([n_sections] + zi_shape)

    x_0 = ext[..., :1]
    y, _ = sosfilt(sos_np, ext, axis=-1, zi=zi * x_0)
    y_0 = y[..., -1:]
    y, _ = sosfilt(sos_np, y[..., ::-1], axis=-1, zi=zi * y_0)
    y = y[..., ::-1]
    if edge > 0:
        y = y[..., edge:-edge]
    if moved:
        y = mx.moveaxis(y, -1, ax)
    return y
