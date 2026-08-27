"""Filtering: FIR design (host-side), FIR/IIR application (GPU), hilbert.

FIR paths run on the GPU via FFT convolution. IIR runs on the GPU through
custom Metal kernels — per-channel-sequential for short signals,
block-parallel associative scan for long ones — in both second-order-section
form (:func:`sosfilt`/:func:`sosfiltfilt`) and transfer-function form
(:func:`lfilter`/:func:`filtfilt` with ``len(a) > 1``, up to order 16, real
coefficients). The sequential kernels reproduce scipy's float32 recurrence
bit-for-bit for normal-range data (Apple GPUs flush float32 denormals to
zero; scipy's CPU keeps them); the block-parallel scan matches to ~1e-6 for
typical wideband filters and ~1e-5 worst-case near its dispatch gate. The
SOS form remains the better-conditioned choice for high orders, as scipy
itself recommends; complex coefficients and orders past the register cap
fall back to scipy with a :class:`~mlx_signal.FallbackWarning`.
"""

from __future__ import annotations

import functools as _functools

import mlx.core as mx
import numpy as np

from . import _fft_core as _sfft
from ._array import (
    _downcast_notice,
    check_strict,
    input_size,
    result_to_mlx,
    signal_np,
    to_mlx,
    to_numpy,
)
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
    # rfft halves the forward transform (the input is real); the analytic
    # spectrum is then assembled directly — doubled positive bins, zeroed
    # negatives — instead of masking a full complex-FFT spectrum
    Xr = _sfft.rfft(xa, n=N, axis=-1)
    if N % 2 == 0:
        parts = [Xr[..., :1], 2.0 * Xr[..., 1 : N // 2], Xr[..., N // 2 : N // 2 + 1]]
        n_zero = N // 2 - 1
    else:
        parts = [Xr[..., :1], 2.0 * Xr[..., 1 : (N + 1) // 2]]
        n_zero = N - (N + 1) // 2
    if n_zero:
        parts.append(mx.zeros(Xr.shape[:-1] + (n_zero,), dtype=mx.complex64))
    out = _sfft.ifft(mx.concatenate(parts, axis=-1), axis=-1)
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
    if ba.ndim == 0:  # scalar b is a length-1 FIR, like scipy's atleast_1d
        ba = ba.reshape(1)
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


# ---------------------------------------------------------------------------
# IIR: transfer-function form (order-N DF2T Metal kernel)
# ---------------------------------------------------------------------------


def _validate_ba_np(b, a):
    """Canonicalize ``(b, a)`` transfer-function coefficients to 1-D f32/c64.

    Mirrors :func:`_validate_sos_np`: scipy's design routines return float64,
    which canonicalizes quietly as design metadata; explicit complex128 warns
    under the downcast policy; a filter whose stability or numerator does not
    survive 32-bit quantization fails identically on every route.
    """
    check_strict(b)
    check_strict(a)
    b_raw = np.atleast_1d(np.asarray(to_numpy(b)))
    a_raw = np.atleast_1d(np.asarray(to_numpy(a)))
    for name, arr, orig in (("b", b_raw, b), ("a", a_raw, a)):
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError(f"{name} must be 1-D with non-zero length")
        explicit_array = isinstance(orig, (np.ndarray, mx.array))
        if explicit_array and (
            (arr.dtype.kind == "c" and arr.dtype.itemsize > 8)
            or (arr.dtype.kind == "f" and arr.dtype.itemsize > 8)
        ):
            _downcast_notice(str(arr.dtype))

    dtype = (np.complex64 if (np.iscomplexobj(b_raw) or np.iscomplexobj(a_raw))
             else np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        b_np = np.ascontiguousarray(b_raw, dtype=dtype)
        a_np = np.ascontiguousarray(a_raw, dtype=dtype)
    if not (np.all(np.isfinite(b_np)) and np.all(np.isfinite(a_np))):
        raise ValueError("filter coefficients must be finite after 32-bit quantization")
    if a_np[0] == 0:
        raise ValueError("a[0] must be nonzero")
    dtype_name = np.dtype(dtype).name
    if np.any(b_raw != 0) and np.all(b_np == 0):
        raise ValueError(
            f"numerator vanished after {dtype_name} quantization; "
            "use a better-conditioned scaling"
        )

    if a_np.shape[0] > 1:
        raw_a = np.asarray(a_raw, dtype=np.complex128)
        raw_poles = np.roots(raw_a / raw_a[0])
        poles = np.roots(np.asarray(a_np, dtype=np.complex128) / complex(a_np[0]))
        raw_radius = float(np.max(np.abs(raw_poles))) if raw_poles.size else 0.0
        radius = float(np.max(np.abs(poles))) if poles.size else 0.0
        if raw_radius < 1.0 and (not np.isfinite(radius) or radius >= 1.0):
            raise ValueError(
                f"denominator is unstable after {dtype_name} quantization "
                f"(maximum pole magnitude {radius:.9g} >= 1); use the SOS form "
                "or a better-conditioned design"
            )
    return b_np, a_np


def _tf_recurrence_f32_reference(b, a, x, zi):
    """NumPy emulation of the TF kernel's exact fma rounding structure.

    Products are exact in float64 and each stored value rounds through
    float32, emulating fused multiply-adds (a true fma can disagree with this
    double-rounded emulation ~2^-29 of the time — tolerable for the probe
    below, whose false negatives only cost the conservative routing).
    """
    f32, f64 = np.float32, np.float64
    order = max(len(b), len(a)) - 1
    a0 = f32(a[0])
    bb = np.zeros(order + 1, f64)
    aa = np.zeros(order + 1, f64)
    bb[: len(b)] = (np.asarray(b, f32) / a0).astype(f32)
    aa[: len(a)] = (np.asarray(a, f32) / a0).astype(f32)
    z = np.asarray(zi, f32).astype(f64).copy()
    y = np.empty(len(x), f32)
    for t in range(len(x)):
        v = f64(f32(x[t]))
        yv = f32(bb[0] * v + z[0])                              # fma(b0, x, z0)
        for k in range(order - 1):                              # fma(-a, y, fma(b, x, z+1))
            z[k] = f64(f32(f64(f32(bb[k + 1] * v + z[k + 1])) - aa[k + 1] * f64(yv)))
        z[order - 1] = f64(f32(bb[order] * v - f64(f32(aa[order] * f64(yv)))))
        y[t] = yv
    return y, z.astype(f32)


@_functools.lru_cache(maxsize=1)
def _scipy_lfilter_f32_order_matches_metal() -> bool:
    """Whether scipy's f32 TF recurrence rounds like our kernel.

    scipy's lfilter C loop inherits the build compiler's fma-contraction
    choices (measured: clang on arm64 fuses every multiply-add), so a version
    gate cannot answer this — probe the installed build directly.
    """
    import scipy.signal as sps

    rng = np.random.default_rng(20260826)
    poles = np.asarray([0.5, 0.3 + 0.4j, 0.3 - 0.4j, -0.2 + 0.6j, -0.2 - 0.6j])
    # a[0] = 1.7 on purpose: the probe must also cover scipy's in-loop
    # normalization division, not just the a0 == 1 fast case
    a = (np.real(np.poly(poles)) * 1.7).astype(np.float32)
    b = (rng.standard_normal(6) * 1.7).astype(np.float32)
    x = rng.standard_normal(257).astype(np.float32)
    zi = (rng.standard_normal(5) * 0.1).astype(np.float32)
    y_ref, zf_ref = _tf_recurrence_f32_reference(b, a, x, zi)
    y, zf = sps.lfilter(b, a, x, zi=zi)
    return bool(np.array_equal(y, y_ref) and np.array_equal(zf, zf_ref))


#: transient-gain cap for the TF scan. Measured block-boundary drift relative
#: to peak output: ~2-5e-8 x max_t ||A^t|| for well-damped filters (butter-4:
#: 3.6e-7 at gain 18), but near-gate resonant families (repeated real poles,
#: narrow resonators at gain ~20+) reach ~5e-7 x gain — worst observed 1.0e-5
#: at this cap. 24 keeps every mainstream wideband design on the scan; a
#: guaranteed ~1e-6 would need a cap of ~4, exiling butter-4 to the
#: sequential kernel. Clustered high-order poles blow far past any cap
#: (butter-8: 275, an order-8 bandpass: 5500) and always stay sequential —
#: TF form at those orders is exactly where scipy says to use SOS instead.
_TF_SCAN_MAX_GAIN = 24.0


@_functools.lru_cache(maxsize=64)
def _tf_scan_safe(a_bytes: bytes, order: int) -> bool:
    """Scan-dispatch gate for the TF kernel.

    Same decay requirement as the SOS gate — the response must die within one
    scan block — plus bounds on the actual block transition: the DF2T
    companion form is non-normal, so clustered poles transiently amplify f32
    entry-state rounding even when every radius**L is tiny.
    ``a_bytes`` is the float64 view of the f32-rounded normalized denominator.

    The admitted drift bound is stated on _TF_SCAN_MAX_GAIN and is relative to
    the internal state scale; a numerator that cancels most of that scale in
    the passband (a deep notch) sees proportionally larger output-relative
    drift, as it does for any float32 recurrence including scipy's own.
    """
    from . import _lfilter_metal

    a = np.frombuffer(a_bytes, dtype=np.float64)
    radius = float(np.max(np.abs(np.roots(a)))) if np.any(a[1:]) else 0.0
    if radius >= 1.0 or radius ** _SCAN_BLOCK_REF > 1e-2:
        return False
    AL, gain = _lfilter_metal._transition_np(a_bytes, order, _SCAN_BLOCK_REF)
    return bool(
        np.all(np.isfinite(AL))
        and float(np.max(np.abs(AL))) <= 1e-2
        and gain <= _TF_SCAN_MAX_GAIN
    )


def _lfilter_tf(b_np, a_np, x, axis, zi):
    """Transfer-function ``lfilter`` (``len(a) > 1``): GPU kernels + routing.

    Mirrors :func:`sosfilt`'s dispatch: scipy below the size threshold or when
    the kernel lacks the capability (complex coefficients, order past the
    register cap, no Metal); the scan kernel for long scan-safe signals; the
    per-channel-sequential kernel otherwise. Every route executes the same
    f32-quantized filter; the sequential kernel reproduces scipy's recurrence
    bit-for-bit (outside the denormal range, which the GPU flushes), and the
    scan stays within the drift bound stated on _TF_SCAN_MAX_GAIN.
    """
    from . import _lfilter_metal

    order = max(b_np.shape[0], a_np.shape[0]) - 1
    xa = to_mlx(x)
    if xa.ndim < 1:
        raise ValueError("x must be at least 1-D with samples along axis")
    if not -xa.ndim <= axis < xa.ndim:
        raise ValueError(
            f"axis {axis} is out of range for an input with {xa.ndim} dimension(s)"
        )
    ax = axis % xa.ndim
    n = xa.shape[ax]

    zi_a = None
    if zi is not None:
        zi_a = to_mlx(zi)
        if zi_a.ndim != xa.ndim:
            raise ValueError(
                "Dimensions of x and zi must match, but "
                f"x.ndim={xa.ndim}, zi.ndim={zi_a.ndim}"
            )
        expected = list(xa.shape)
        expected[ax] = order
        ok = all(
            zs == e or (i != ax and zs == 1)
            for i, (zs, e) in enumerate(zip(zi_a.shape, expected, strict=True))
        )
        if not ok:
            raise ValueError(
                f"Unexpected shape for zi: expected {tuple(expected)}, "
                f"found {tuple(zi_a.shape)}."
            )
        zi_a = mx.broadcast_to(zi_a, tuple(expected))

    complex_coeffs = np.iscomplexobj(b_np)
    kernel_ok = (
        mx.metal.is_available()
        and not complex_coeffs
        and order <= _lfilter_metal.MAX_ORDER
    )

    def _scipy_path():
        import scipy.signal as sps

        # canonicalized signal/state, like sosfilt's fallback: routing by size
        # or batch count must never change the recurrence precision
        x_np = to_numpy(xa)
        zi_np = to_numpy(zi_a) if zi is not None else None
        split_complex = not complex_coeffs and (
            np.iscomplexobj(x_np) or (zi_np is not None and np.iscomplexobj(zi_np))
        )
        if split_complex:
            # mirror the Metal two-plane arithmetic (scipy's native complex64
            # recurrence rounds in a different order)
            xr = np.asarray(np.real(x_np), dtype=np.float32)
            xi = (
                np.asarray(np.imag(x_np), dtype=np.float32)
                if np.iscomplexobj(x_np)
                else np.zeros_like(xr)
            )
            j = np.complex64(1j)
            if zi_np is None:
                yr = sps.lfilter(b_np, a_np, xr, axis=axis)
                yi = sps.lfilter(b_np, a_np, xi, axis=axis)
                out = np.asarray(yr, dtype=np.complex64) + j * yi
            else:
                zir = np.asarray(np.real(zi_np), dtype=np.float32)
                zii = (
                    np.asarray(np.imag(zi_np), dtype=np.float32)
                    if np.iscomplexobj(zi_np)
                    else np.zeros_like(zir)
                )
                yr, zfr = sps.lfilter(b_np, a_np, xr, axis=axis, zi=zir)
                yi, zfi = sps.lfilter(b_np, a_np, xi, axis=axis, zi=zii)
                out = (
                    np.asarray(yr, dtype=np.complex64) + j * yi,
                    np.asarray(zfr, dtype=np.complex64) + j * zfi,
                )
        else:
            x_c = x_np if not complex_coeffs else np.asarray(x_np, dtype=np.complex64)
            zi_c = (
                None if zi_np is None
                else zi_np if not complex_coeffs
                else np.asarray(zi_np, dtype=np.complex64)
            )
            out = sps.lfilter(b_np, a_np, x_c, axis=axis, zi=zi_c)
        if zi is not None:
            return result_to_mlx(out[0]), result_to_mlx(out[1])
        return result_to_mlx(out)

    if n == 0 or not use_mlx(xa.size * order):
        return _scipy_path()
    if not kernel_ok:
        reason = ("complex filter coefficients" if complex_coeffs
                  else f"order {order} exceeds the kernel cap of "
                       f"{_lfilter_metal.MAX_ORDER}"
                  if order > _lfilter_metal.MAX_ORDER
                  else "no Metal GPU")
        capability_fallback("lfilter", reason)
        return _scipy_path()

    # pre-normalize in f32: bit-identical to scipy's in-loop f32 b[k] / a[0]
    # (which also divides silently — hence the errstate — and produces the
    # same inf/NaN coefficients for a pathological a[0])
    ba = np.zeros((2, order + 1), dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        ba[0, : b_np.shape[0]] = (b_np / a_np[0]).astype(np.float32)
        ba[1, : a_np.shape[0]] = (a_np / a_np[0]).astype(np.float32)
    a64 = np.ascontiguousarray(ba[1], dtype=np.float64)
    scan_safe = _tf_scan_safe(a64.tobytes(), order)
    use_scan = n >= _SCAN_MIN_N and scan_safe
    if (
        get_config().dispatch == "auto"
        and not scan_safe
        and not _scipy_lfilter_f32_order_matches_metal()
    ):
        # a scipy build whose f32 rounding differs from the kernel must not
        # switch recurrence with auto routing on a long-lived filter
        return _scipy_path()
    batch = xa.size // n
    if (
        get_config().dispatch == "auto"
        and not use_scan
        and batch < _SOSFILT_MIN_ROWS
    ):
        # short signal, few channels: scipy's single fast core wins
        return _scipy_path()

    ba_flat = mx.array(ba.reshape(-1))

    def _run(x2p, zi2p):
        if use_scan:
            return _lfilter_metal.lfilter_scan_gpu(x2p, ba, zi2p)
        return _lfilter_metal.lfilter_gpu(x2p, ba_flat, zi2p)

    moved = xa.ndim > 1 and ax != xa.ndim - 1
    if moved:
        xa = mx.moveaxis(xa, ax, -1)
    batch_shape = xa.shape[:-1]
    x2 = xa.reshape(-1, n)

    if zi is not None:
        zi_m = mx.moveaxis(zi_a, ax, -1) if moved else zi_a
        zi2 = zi_m.reshape(-1, order)
    else:
        zi2 = mx.zeros((x2.shape[0], order))

    complex_state = zi is not None and zi2.dtype == mx.complex64
    if x2.dtype == mx.complex64 or complex_state:
        # filtering is linear: real and imaginary parts run as two launches
        if x2.dtype != mx.complex64:
            x2 = x2.astype(mx.complex64)
        if zi is not None and zi2.dtype != mx.complex64:
            zi2 = zi2.astype(mx.complex64)
        yr, zr = _run(mx.real(x2), mx.real(zi2))
        yi, zim = _run(mx.imag(x2), mx.imag(zi2))
        j = mx.array(1j)
        y2 = yr.astype(mx.complex64) + yi.astype(mx.complex64) * j
        zf2 = zr.astype(mx.complex64) + zim.astype(mx.complex64) * j
    else:
        y2, zf2 = _run(x2, zi2)

    y = y2.reshape(batch_shape + (n,))
    if moved:
        y = mx.moveaxis(y, -1, ax)
    if zi is None:
        return y
    zf = zf2.reshape(batch_shape + (order,))
    if moved:
        zf = mx.moveaxis(zf, -1, ax)
    return y, zf


def lfilter(b, a, x, axis=-1, zi=None):
    """Filter data with an IIR or FIR filter (scipy-compatible signature).

    The FIR case (``len(a) == 1``) runs on the GPU as a truncated FFT
    convolution. Transfer-function IIR (``len(a) > 1``) runs on the GPU
    through order-N direct-form-II-transposed Metal kernels (sequential and
    block-parallel scan, like :func:`sosfilt`) that reproduce scipy's float32
    recurrence bit-for-bit, with the full ``zi``/``zf`` contract. Complex
    coefficients and orders above 16 fall back to scipy with a
    FallbackWarning; ``zi`` with ``len(a) == 1`` follows scipy's FIR
    convolution path on the CPU.
    """
    taps = _as_fir_taps(b, a)
    if taps is not None:
        work = input_size(x) * max(1, taps.size)
        if zi is not None or not use_mlx(work):
            if zi is not None and use_mlx(work):
                capability_fallback("lfilter", "zi initial conditions")
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

    b_np, a_np = _validate_ba_np(b, a)
    return _lfilter_tf(b_np, a_np, x, axis, zi)


def _filtfilt_tf(b_np, a_np, x, axis, padtype, padlen, method, irlen):
    """Transfer-function ``filtfilt``: scipy's pad method through our lfilter.

    The same construction :func:`sosfiltfilt` uses — edge extension,
    ``lfilter_zi`` steady-state initialization scaled by the edge samples, a
    forward and a reverse pass, trim — with both passes routed through
    :func:`lfilter`, so the TF Metal kernels (or the scipy routing) apply to
    each. ``method="gust"`` and complex coefficients go to scipy wholesale.
    """
    order = max(b_np.shape[0], a_np.shape[0]) - 1
    complex_coeffs = np.iscomplexobj(b_np)
    xa = to_mlx(x)
    if xa.ndim < 1:
        raise ValueError("x must be at least 1-D with samples along axis")
    if not -xa.ndim <= axis < xa.ndim:
        raise ValueError(
            f"axis {axis} is out of range for an input with {xa.ndim} dimension(s)"
        )
    # real coefficients always use the shared pad construction below (the two
    # inner lfilter calls honor size dispatch) so complex signals run the same
    # two-plane recurrence on every route — a wholesale sps.filtfilt call for
    # small inputs would switch to scipy's native complex64 recurrence and
    # make the silent size boundary visible, exactly like sosfiltfilt's fix
    if method == "gust" or complex_coeffs:
        if use_mlx(xa.size * max(1, order)):
            reason = ("method='gust'" if method == "gust"
                      else "complex filter coefficients")
            capability_fallback("filtfilt", reason)
        import scipy.signal as sps

        return result_to_mlx(
            sps.filtfilt(b_np, a_np, to_numpy(xa), axis=axis, padtype=padtype,
                         padlen=padlen, method=method, irlen=irlen)
        )

    if padtype not in ("even", "odd", "constant", None):
        raise ValueError(
            f"Unknown value '{padtype}' given to padtype. padtype must be 'even', "
            "'odd', 'constant', or None."
        )

    ntaps = max(b_np.shape[0], a_np.shape[0])
    if padtype is None:
        edge = 0
    elif padlen is None:
        edge = 3 * ntaps
    else:
        edge = int(padlen)

    ax = axis % xa.ndim
    if xa.shape[ax] <= edge:
        raise ValueError(
            "The length of the input vector x must be greater than padlen, "
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

    from scipy.signal import lfilter_zi as _lfilter_zi

    # scipy's f32 construction on the exact coefficients every route executes,
    # like sosfiltfilt: the direct scipy recurrence and the Metal kernels both
    # consume these state values
    # ascontiguousarray: lfilter_zi can return a negative-stride view, which
    # mx.array refuses through DLPack
    zi_np = np.ascontiguousarray(_lfilter_zi(b_np, a_np), dtype=np.float32)  # (order,)
    zi_shape = [1] * ext.ndim
    zi_shape[-1] = order
    zi_r = mx.array(zi_np).reshape(zi_shape)

    x_0 = ext[..., :1]
    y, _ = lfilter(b_np, a_np, ext, axis=-1, zi=zi_r * x_0)
    y_0 = y[..., -1:]
    y, _ = lfilter(b_np, a_np, y[..., ::-1], axis=-1, zi=zi_r * y_0)
    y = y[..., ::-1]
    if edge > 0:
        y = y[..., edge:-edge]
    if moved:
        y = mx.moveaxis(y, -1, ax)
    return y


def filtfilt(b, a, x, axis=-1, padtype="odd", padlen=None, method="pad", irlen=None):
    """Zero-phase forward-backward filtering (scipy-compatible).

    The FIR case runs on the GPU: odd/even/constant edge extension, then a
    forward and a backward causal convolution with scipy's exact steady-state
    initialization (implemented as an (ntaps-1)-sample constant prefix, which
    is algebraically identical to ``lfilter_zi`` for FIR filters). The IIR
    case (``len(a) > 1``) runs both passes through :func:`lfilter`'s TF Metal
    kernels with scipy's exact pad construction. ``method="gust"`` and complex
    coefficients fall back to scipy.
    """
    if method not in ("pad", "gust"):
        raise ValueError("method must be 'pad' or 'gust'.")

    taps = _as_fir_taps(b, a)
    if taps is None:
        b_np, a_np = _validate_ba_np(b, a)
        return _filtfilt_tf(b_np, a_np, x, axis, padtype, padlen, method, irlen)

    work = input_size(x) * max(1, taps.size)
    if method == "gust" or not use_mlx(work):
        if use_mlx(work):
            capability_fallback("filtfilt", "method='gust'")
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


@_functools.lru_cache(maxsize=1)
def _scipy_sosfilt_f32_order_matches_metal() -> bool:
    """Whether scipy's f32 recurrence uses the operation order in our kernel."""
    import scipy

    version = tuple(int(part) for part in scipy.__version__.split(".")[:2])
    return version >= (1, 15)


def _validate_sos_np(sos, *, apply_dtype_policy: bool = True) -> np.ndarray:
    # scipy's design routines intentionally return float64 SOS. Treat those
    # small coefficient arrays as design metadata under the default downcast
    # policy (silently canonicalize them), but honor an explicitly strict
    # policy. Signal/state arrays still use the normal DowncastWarning path.
    if apply_dtype_policy:
        check_strict(sos)
    sos_raw = np.atleast_2d(np.asarray(to_numpy(sos)))
    explicit_array = isinstance(sos, (np.ndarray, mx.array))
    if apply_dtype_policy and explicit_array and (
        (sos_raw.dtype.kind == "c" and sos_raw.dtype.itemsize > 8)
        or (sos_raw.dtype.kind == "f" and sos_raw.dtype.itemsize > 8)
    ):
        # Real float64 is the routine output dtype of scipy's SOS designers and
        # stays quiet. Explicit complex128/float128 coefficients are unusual,
        # materially lossy conversions and should follow the warning policy.
        _downcast_notice(str(sos_raw.dtype))
    if sos_raw.ndim != 2:
        raise ValueError("sos array must be 2D")
    if sos_raw.shape[1] != 6:
        raise ValueError("sos array must be shape (n_sections, 6)")
    if sos_raw.shape[0] < 1:
        raise ValueError("sos array must have at least one section")
    if not np.all(sos_raw[:, 3] == 1):
        raise ValueError("sos[:, 3] should be all ones")

    # One execution dtype on every route. Complex SOS currently has no Metal
    # kernel, but its scipy fallback must still obey the documented c64 policy
    # rather than silently computing in complex128.
    dtype = np.complex64 if np.iscomplexobj(sos_raw) else np.float32
    with np.errstate(over="ignore", invalid="ignore"):
        sos = np.ascontiguousarray(sos_raw, dtype=dtype)
    if not np.all(np.isfinite(sos)):
        raise ValueError("sos coefficients must be finite after 32-bit quantization")

    vanished = np.all(sos[:, :3] == 0, axis=1) & np.any(sos_raw[:, :3] != 0, axis=1)
    if np.any(vanished):
        section = int(np.flatnonzero(vanished)[0])
        raise ValueError(
            f"sos section {section} numerator vanished after {np.dtype(dtype).name} "
            "quantization; use a better-conditioned section scaling"
        )

    # A stable host-side design can cross the unit circle when its denominator
    # is rounded to f32/c64. Running that changed filter produces explosive,
    # route-sensitive nonsense (and a pole at exactly one makes sosfilt_zi
    # singular), so fail before either backend executes it. Deliberately
    # unstable/integrating SOS remain valid for causal sosfilt; reject only a
    # stability loss introduced by quantization. Apply the same check to
    # complex-typed copies so dtype alone cannot bypass validation.
    dtype_name = np.dtype(dtype).name
    for section, (raw, quantized) in enumerate(zip(sos_raw, sos, strict=True)):
        raw_poles = np.roots(np.asarray([1.0, raw[4], raw[5]], dtype=np.complex128))
        poles = np.roots(
            np.asarray([1.0, quantized[4], quantized[5]], dtype=np.complex128)
        )
        raw_radius = float(np.max(np.abs(raw_poles)))
        radius = float(np.max(np.abs(poles)))
        if raw_radius < 1.0 and (not np.isfinite(radius) or radius >= 1.0):
            raise ValueError(
                f"sos section {section} is unstable after {dtype_name} quantization "
                f"(maximum pole magnitude {radius:.9g} >= 1); use a "
                "better-conditioned design or a higher cutoff"
            )
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

        # Use the already-canonicalized signal/state as well as coefficients.
        # Passing the original f64/c128 objects here made routing by size or
        # batch count change the recurrence precision and broke streaming when
        # the returned f32 state was fed into the next chunk.
        x_np = to_numpy(xa)
        zi_np = to_numpy(zi_a) if zi is not None else None
        split_complex = not complex_sos and (
            np.iscomplexobj(x_np) or (zi_np is not None and np.iscomplexobj(zi_np))
        )
        if split_complex:
            # The Metal implementation runs real and imaginary float32 planes
            # independently. scipy's native complex64 recurrence rounds in a
            # different order, which made the 31/32-row routing boundary
            # visible for narrow filters. Mirror the two-plane arithmetic.
            xr = np.asarray(np.real(x_np), dtype=np.float32)
            xi = (
                np.asarray(np.imag(x_np), dtype=np.float32)
                if np.iscomplexobj(x_np)
                else np.zeros_like(xr)
            )
            if zi_np is None:
                yr = sps.sosfilt(sos_np, xr, axis=axis)
                yi = sps.sosfilt(sos_np, xi, axis=axis)
                out = np.asarray(yr, dtype=np.complex64) + np.complex64(1j) * yi
            else:
                zir = np.asarray(np.real(zi_np), dtype=np.float32)
                zii = (
                    np.asarray(np.imag(zi_np), dtype=np.float32)
                    if np.iscomplexobj(zi_np)
                    else np.zeros_like(zir)
                )
                yr, zfr = sps.sosfilt(sos_np, xr, axis=axis, zi=zir)
                yi, zfi = sps.sosfilt(sos_np, xi, axis=axis, zi=zii)
                out = (
                    np.asarray(yr, dtype=np.complex64) + np.complex64(1j) * yi,
                    np.asarray(zfr, dtype=np.complex64) + np.complex64(1j) * zfi,
                )
        else:
            out = sps.sosfilt(sos_np, x_np, axis=axis, zi=zi_np)
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
    # The safety-cache and transition-power helpers use a canonical f64 byte
    # key, but the represented coefficients have already been rounded to f32.
    sos64 = np.ascontiguousarray(sos_np, dtype=np.float64)
    scan_safe = _scan_safe(sos64.tobytes(), n_sections)
    use_scan = n >= _SCAN_MIN_N and scan_safe
    if (
        get_config().dispatch == "auto"
        and not scan_safe
        and not _scipy_sosfilt_f32_order_matches_metal()
    ):
        # scipy <1.15 rounds the f32 recurrence in a different order. For a
        # narrow/long-lived filter, switching from that fallback at 31 rows to
        # the sequential Metal kernel at 32 was visible at the percent level.
        # Keep scan-unsafe auto calls on the floor backend; explicit MLX still
        # pins the kernel, and wide scan-safe filters retain GPU acceleration.
        return _scipy_path()
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

    complex_state = zi is not None and zi2.dtype == mx.complex64
    if x2.dtype == mx.complex64 or complex_state:
        # complex signal and/or complex state: filtering is linear, so the
        # real and imaginary parts run as two launches (scipy returns complex
        # output for real x with complex zi; so do we)
        if x2.dtype != mx.complex64:
            x2 = x2.astype(mx.complex64)
        if zi is not None and zi2.dtype != mx.complex64:
            zi2 = zi2.astype(mx.complex64)
        yr, zr = _run(mx.real(x2), mx.real(zi2))
        yi, zim = _run(mx.imag(x2), mx.imag(zi2))
        j = mx.array(1j)
        y2 = yr.astype(mx.complex64) + yi.astype(mx.complex64) * j
        zf2 = zr.astype(mx.complex64) + zim.astype(mx.complex64) * j
    else:
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
    xa = to_mlx(x)

    if np.iscomplexobj(sos_np):
        # Complex coefficients have no Metal kernel. A deliberately selected
        # scipy route (or an auto-sized tiny call) is not a capability fallback
        # and should stay silent; auto-large warns and pinned MLX raises.
        if use_mlx(xa.size * n_sections):
            capability_fallback("sosfiltfilt", "complex sos coefficients")
        import scipy.signal as sps

        return result_to_mlx(
            sps.sosfiltfilt(sos_np, to_numpy(xa), axis=axis, padtype=padtype,
                            padlen=padlen)
        )

    # Real SOS always use this shared wrapper, including dispatch="scipy".
    # The two inner sosfilt calls still honor dispatch, while padding and the
    # exact f32 zi stay identical across scipy, auto, and Metal routes.
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


    # Use scipy's f32 construction directly. Promoting rounded coefficients
    # back to f64 before solving changes zi by an ulp; near-unit poles amplify
    # that into route-dependent transients. Both the direct scipy recurrence
    # and Metal kernels consume these exact f32 state values.
    zi_np = np.asarray(_sosfilt_zi(sos_np), dtype=np.float32)  # (S, 2)
    zi_shape = [1] * ext.ndim
    zi_shape[-1] = 2
    zi = mx.array(zi_np).reshape([n_sections] + zi_shape)

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
