"""Peak finding: GPU-prefiltered local-maxima scan + host-side refinement.

find_peaks is bandwidth-bound index bookkeeping, not FLOPs — the wrong shape
for a GPU win. It is included for pipeline completeness: the elementwise
difference/sign pass runs in MLX, and the index logic (plateau resolution,
distance/prominence/width filtering) runs vectorized on the host, matching
scipy.signal.find_peaks results exactly.

Unlike the rest of the library, peak indices and properties are returned as
NumPy arrays: they are host-side metadata (indices into your signal), not GPU
tensors.
"""

from __future__ import annotations

import math

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
        # exact when the source data is float32: the sign of an f32 subtraction
        # of f32 values is always the true sign (Sterbenz)
        xa = mx.array(x.astype(np.float32))
        d = np.array(mx.sign(xa[1:] - xa[:-1])).astype(np.int8)
    else:
        d = np.sign(np.diff(x)).astype(np.int8)
    nz = np.nonzero(d)[0]
    if nz.size < 2:
        e = np.array([], dtype=np.intp)
        return e, e, e
    s = d[nz]
    trans = np.nonzero((s[:-1] > 0) & (s[1:] < 0))[0]
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


def peak_prominences(x, peaks, wlen=None):
    """Prominence of each peak (scipy.signal.peak_prominences; host-side)."""
    import scipy.signal as sps

    return sps.peak_prominences(_as_1d_f64(x), np.asarray(peaks), wlen=wlen)


def peak_widths(x, peaks, rel_height=0.5, prominence_data=None, wlen=None):
    """Width of each peak (scipy.signal.peak_widths; host-side)."""
    import scipy.signal as sps

    return sps.peak_widths(
        _as_1d_f64(x), np.asarray(peaks), rel_height=rel_height,
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

    gpu_ok = (isinstance(orig, mx.array) and orig.dtype == mx.float32) or (
        isinstance(orig, np.ndarray) and orig.dtype == np.float32
    )
    peaks, left_edges, right_edges = _local_maxima(x, gpu_ok)
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
        import scipy.signal as sps

        properties.update(
            zip(
                ["prominences", "left_bases", "right_bases"],
                sps.peak_prominences(x, peaks, wlen=wlen),
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
