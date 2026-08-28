"""Peak finding: GPU local-maxima prefilter + prominence scan, host refinement.

The prominence stage — scipy's dominant cost in ``find_peaks(prominence=...)``
(75% of its runtime: a sequential Cython walk from every peak) — runs on the
GPU through a per-peak two-level block-skip Metal kernel (``_peaks_metal``)
for float32 sources, bit-identical to scipy. The remaining index logic
(plateau resolution, distance/width filtering) is bandwidth-bound bookkeeping
and runs vectorized on the host, matching scipy.signal results exactly.

Unlike the rest of the library, peak indices and properties are returned as
NumPy arrays: they are host-side metadata (indices into your signal), not GPU
tensors. Routing here is quiet by design (no capability fallbacks): scipy is
the reference implementation and always available, the GPU path is an exact
accelerator for the case that dominates in practice.
"""

from __future__ import annotations

import math
import warnings

import mlx.core as mx
import numpy as np

from ._array import to_numpy
from ._config import use_mlx

__all__ = ["find_peaks", "peak_prominences", "peak_widths"]


def _as_1d_f64(x) -> np.ndarray:
    arr = np.asarray(to_numpy(x), order="C", dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("`x` must be a 1-D array")
    return arr


def _local_maxima(x: np.ndarray, gpu_ok: bool = False):
    """Exact scipy `_local_maxima_1d` semantics, vectorized.

    A maximum is a sample (or plateau) with a strict rise before and a strict
    fall after; plateaus report their midpoint plus left/right edges.
    """
    if x.size < 3:
        e = np.array([], dtype=np.intp)
        return e, e, e
    if gpu_ok and use_mlx(x.size) and mx.metal.is_available():
        # Compare directly.  Subtract-then-sign is not exact on Metal when two
        # adjacent normal values differ by a subnormal ULP: the subtraction is
        # flushed to zero even though the strict ordering is representable.
        # Encode unordered NaN pairs as a non-sign sentinel so they break, not
        # bridge, a possible rise-to-fall transition.
        xa = mx.array(x.astype(np.float32))
        left, right = xa[:-1], xa[1:]
        unordered = mx.isnan(left) | mx.isnan(right)
        d = np.array(
            mx.where(unordered, 2, mx.where(right > left, 1, mx.where(right < left, -1, 0)))
        ).astype(np.int8)
    else:
        left, right = x[:-1], x[1:]
        unordered = np.isnan(left) | np.isnan(right)
        d = np.where(
            unordered, 2, np.where(right > left, 1, np.where(right < left, -1, 0))
        )
        d = d.astype(np.int8)
    nz = np.nonzero(d)[0]
    if nz.size < 2:
        e = np.array([], dtype=np.intp)
        return e, e, e
    s = d[nz]
    trans = np.nonzero((s[:-1] == 1) & (s[1:] == -1))[0]
    left_edges = (nz[trans] + 1).astype(np.intp)
    right_edges = nz[trans + 1].astype(np.intp)
    midpoints = (left_edges + right_edges) // 2
    return midpoints, left_edges, right_edges


def _unpack_condition_args(interval, x, peaks):
    try:
        imin, imax = interval
    except (TypeError, ValueError):
        imin, imax = (interval, None)
    if isinstance(imin, np.ndarray):
        if imin.size != x.size:
            raise ValueError("array size of lower interval border must match x")
        imin = imin[peaks]
    if isinstance(imax, np.ndarray):
        if imax.size != x.size:
            raise ValueError("array size of upper interval border must match x")
        imax = imax[peaks]
    return imin, imax


def _select_by_property(peak_properties, pmin, pmax):
    keep = np.ones(peak_properties.size, dtype=bool)
    if pmin is not None:
        keep &= pmin <= peak_properties
    if pmax is not None:
        keep &= peak_properties <= pmax
    return keep


def _select_by_peak_threshold(x, peaks, tmin, tmax):
    stacked_thresholds = np.vstack([x[peaks] - x[peaks - 1], x[peaks] - x[peaks + 1]])
    keep = np.ones(peaks.size, dtype=bool)
    if tmin is not None:
        keep &= tmin <= np.min(stacked_thresholds, axis=0)
    if tmax is not None:
        keep &= np.max(stacked_thresholds, axis=0) <= tmax
    return keep, stacked_thresholds[0], stacked_thresholds[1]


def _select_by_peak_distance(peaks, priority, distance):
    """Greedy highest-priority-first suppression (scipy's exact semantics).

    The loop is inherently sequential (each survival depends on every
    higher-priority decision within range), so it belongs on the CPU —
    compiled. scipy ships this exact loop as Cython; delegate to it when
    importable (the pure-Python port below, written from that loop, is ~7x
    slower at 200k peaks and seconds at millions). Both use np.argsort on the
    same priority array, so tie ordering is identical on either route.
    """
    try:
        from scipy.signal._peak_finding_utils import (
            _select_by_peak_distance as _cy_select,
        )
    except ImportError:  # pragma: no cover - future scipy reorganizations
        _cy_select = None
    if _cy_select is not None:
        return np.asarray(
            _cy_select(
                np.ascontiguousarray(peaks, dtype=np.intp),
                np.ascontiguousarray(priority, dtype=np.float64),
                float(distance),
            ),
            dtype=bool,
        )

    # fallback-only divergence: a non-finite distance follows scipy's C cast
    # on the Cython route (NaN keeps everything) but raises in math.ceil here;
    # every finite distance produces identical keep masks on both routes
    peaks_size = peaks.shape[0]
    distance_ = math.ceil(distance)
    keep = np.ones(peaks_size, dtype=bool)
    priority_to_position = np.argsort(priority)
    for i in range(peaks_size - 1, -1, -1):
        j = priority_to_position[i]
        if not keep[j]:
            continue
        k = j - 1
        while 0 <= k and peaks[j] - peaks[k] < distance_:
            keep[k] = False
            k -= 1
        k = j + 1
        while k < peaks_size and peaks[k] - peaks[j] < distance_:
            keep[k] = False
            k += 1
    return keep


def _f32_source(x):
    """The 1-D float32 array behind ``x`` when the GPU exactness precondition
    holds (float32 values embed exactly in the float64 scipy computes with),
    else None."""
    if isinstance(x, mx.array) and x.dtype == mx.float32 and x.ndim == 1:
        return x
    if isinstance(x, np.ndarray) and x.dtype == np.float32 and x.ndim == 1:
        return x
    return None


def _has_f32_denormals(x64: np.ndarray) -> bool:
    """Whether an exactly embedded float32 signal contains a denormal value.

    Metal flushes float32 denormals during comparison, so these rare signals
    retain scipy's CPU path.  Scan in chunks to keep temporaries cache-sized.
    """
    tiny = np.finfo(np.float32).tiny
    step = 1 << 21
    for start in range(0, x64.size, step):
        a = np.abs(x64[start : start + step])
        if bool(np.any((a > 0) & (a < tiny))):
            return True
    return False


def _peaks_as_intp(peaks):
    """scipy's peaks canonicalization: same casts, same errors."""
    peaks = np.asarray(peaks)
    if peaks.size == 0:
        peaks = np.array([], dtype=np.intp)
    try:
        peaks = peaks.astype(np.intp, casting="safe")
    except TypeError as e:
        raise TypeError("cannot safely cast `peaks` to dtype('intp')") from e
    if peaks.ndim != 1:
        raise ValueError("`peaks` must be a 1-D array")
    return peaks


def _prominences(x64, f32_src, peaks, wlen):
    """(prominences, left_bases, right_bases), scipy-exact on every route.

    GPU route (float32 source, no wlen, Metal, above the size threshold): the
    kernel finds base indices with float32 comparisons — each predicate
    decides identically to scipy's float64 walk because f32 embeds exactly in
    f64 — and the prominences are one vectorized float64 subtraction, so the
    triple bit-matches scipy. Everything else delegates to scipy. The
    zero-prominence warning carries the same class and message on both
    routes; only its frame attribution differs (the GPU route warns from the
    caller's frame like scipy-direct, the delegation from this module).
    """
    import scipy.signal as sps

    from . import _peaks_metal

    if (
        f32_src is None
        or wlen is not None
        or not mx.metal.is_available()
        or not use_mlx(x64.size)
        or x64.size >= _peaks_metal.MAX_N
    ):
        return sps.peak_prominences(x64, peaks, wlen=wlen)

    # Metal compares denormals as zero (measured: 1e-40 == 0.0 is true
    # on-GPU), which would corrupt base decisions in a signal scaled into the
    # float32-denormal range; such signals keep scipy's exact CPU walk.
    # Chunked so the |x| temporaries stay cache-resident rather than making
    # three full-signal passes through DRAM.
    if _has_f32_denormals(x64):
        return sps.peak_prominences(x64, peaks, wlen=wlen)

    peaks_i = _peaks_as_intp(peaks)
    if peaks_i.size == 0:
        return (np.array([], dtype=np.float64), np.array([], dtype=np.intp),
                np.array([], dtype=np.intp))
    invalid = (peaks_i < 0) | (peaks_i >= x64.size)
    if invalid.any():
        first = peaks_i[int(np.flatnonzero(invalid)[0])]
        raise ValueError(f"peak {first} is not a valid index for `x`")

    if isinstance(f32_src, mx.array):
        x32 = f32_src
    else:
        # a reversed f32 view is ordinary numpy; mx.array refuses
        # negative-stride DLPack exports, so ensure contiguity (a no-op for
        # the common contiguous case — and only the values matter here)
        src = f32_src.copy() if any(s < 0 for s in f32_src.strides) else f32_src
        x32 = mx.array(np.ascontiguousarray(src))
    lb, rb = _peaks_metal.prominence_bases(x32, peaks_i)
    prominences = x64[peaks_i] - np.maximum(x64[lb], x64[rb])
    if np.any(prominences == 0):
        try:  # scipy keeps the class in a private module (subclasses RuntimeWarning)
            from scipy.signal._peak_finding_utils import PeakPropertyWarning
        except ImportError:  # pragma: no cover - future scipy reorganization
            PeakPropertyWarning = RuntimeWarning
        warnings.warn("some peaks have a prominence of 0",
                      PeakPropertyWarning, stacklevel=3)
    return prominences, lb, rb


def peak_prominences(x, peaks, wlen=None):
    """Prominence of each peak (scipy.signal.peak_prominences, bit-exact).

    For float32 sources the base search runs on the GPU (one thread per peak,
    two-level block skipping for the heavy-tailed scan lengths); ``wlen`` and
    other dtypes delegate to scipy.
    """
    x64 = _as_1d_f64(x)
    return _prominences(x64, _f32_source(x), np.asarray(peaks), wlen)


def peak_widths(x, peaks, rel_height=0.5, prominence_data=None, wlen=None):
    """Width of each peak (scipy.signal.peak_widths; host-side interpolation).

    When ``prominence_data`` is not supplied it is computed first — on the GPU
    for float32 sources — and handed to scipy, which then only does the
    per-peak width interpolation.
    """
    import scipy.signal as sps

    x64 = _as_1d_f64(x)
    if prominence_data is None:
        prominence_data = _prominences(x64, _f32_source(x), np.asarray(peaks), wlen)
    return sps.peak_widths(
        x64, np.asarray(peaks), rel_height=rel_height,
        prominence_data=prominence_data, wlen=wlen,
    )


def find_peaks(
    x,
    height=None,
    threshold=None,
    distance=None,
    prominence=None,
    width=None,
    wlen=None,
    rel_height=0.5,
    plateau_size=None,
):
    """Find peaks inside a signal based on peak properties (scipy-compatible).

    Returns ``(peaks, properties)`` with NumPy arrays; conditions are evaluated
    in scipy's order: plateau_size, height, threshold, distance, prominence,
    width.
    """
    orig = x
    x = _as_1d_f64(x)
    if distance is not None and distance < 1:
        raise ValueError("`distance` must be greater or equal to 1")

    f32_src = _f32_source(orig)
    if (
        f32_src is not None
        and mx.metal.is_available()
        and use_mlx(x.size)
        and _has_f32_denormals(x)
    ):
        f32_src = None
    peaks, left_edges, right_edges = _local_maxima(x, f32_src is not None)
    properties: dict[str, np.ndarray] = {}

    if plateau_size is not None:
        plateau_sizes = right_edges - left_edges + 1
        pmin, pmax = _unpack_condition_args(plateau_size, x, peaks)
        keep = _select_by_property(plateau_sizes, pmin, pmax)
        peaks = peaks[keep]
        properties["plateau_sizes"] = plateau_sizes
        properties["left_edges"] = left_edges
        properties["right_edges"] = right_edges
        properties = {key: array[keep] for key, array in properties.items()}

    if height is not None:
        peak_heights = x[peaks]
        hmin, hmax = _unpack_condition_args(height, x, peaks)
        keep = _select_by_property(peak_heights, hmin, hmax)
        peaks = peaks[keep]
        properties["peak_heights"] = peak_heights
        properties = {key: array[keep] for key, array in properties.items()}

    if threshold is not None:
        tmin, tmax = _unpack_condition_args(threshold, x, peaks)
        keep, left_thresholds, right_thresholds = _select_by_peak_threshold(
            x, peaks, tmin, tmax
        )
        peaks = peaks[keep]
        properties["left_thresholds"] = left_thresholds
        properties["right_thresholds"] = right_thresholds
        properties = {key: array[keep] for key, array in properties.items()}

    if distance is not None:
        keep = _select_by_peak_distance(peaks, x[peaks], distance)
        peaks = peaks[keep]
        properties = {key: array[keep] for key, array in properties.items()}

    if prominence is not None or width is not None:
        properties.update(
            zip(
                ["prominences", "left_bases", "right_bases"],
                _prominences(x, f32_src, peaks, wlen),
                strict=True,
            )
        )

    if prominence is not None:
        pmin, pmax = _unpack_condition_args(prominence, x, peaks)
        keep = _select_by_property(properties["prominences"], pmin, pmax)
        peaks = peaks[keep]
        properties = {key: array[keep] for key, array in properties.items()}

    if width is not None:
        import scipy.signal as sps

        properties.update(
            zip(
                ["widths", "width_heights", "left_ips", "right_ips"],
                sps.peak_widths(
                    x,
                    peaks,
                    rel_height=rel_height,
                    prominence_data=(
                        properties["prominences"],
                        properties["left_bases"],
                        properties["right_bases"],
                    ),
                ),
                strict=True,
            )
        )
        wmin, wmax = _unpack_condition_args(width, x, peaks)
        keep = _select_by_property(properties["widths"], wmin, wmax)
        peaks = peaks[keep]
        properties = {key: array[keep] for key, array in properties.items()}

    return peaks, properties
