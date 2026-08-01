"""Correctness gates shared by the cross-library benchmarks and their tests."""

from __future__ import annotations

import numpy as np


def rel_err(out, ref, tol=1e-3):
    """Format relative error and fail when shape, finiteness, or error is invalid."""
    out = np.asarray(out)
    ref = np.asarray(ref)
    if out.shape != ref.shape:
        raise RuntimeError(f"verification failed: shape {out.shape} != {ref.shape}")
    if not np.all(np.isfinite(out)) or not np.all(np.isfinite(ref)):
        raise RuntimeError("verification failed: output or reference contains non-finite values")
    e = np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-30)
    if not e <= tol:
        raise RuntimeError(f"verification failed: rel err {e:.2e} > {tol:.0e}")
    return f"{e:.0e}"


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Signed Pearson correlation without allocating a 2x2 covariance matrix."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ac = a - a.mean()
    bc = b - b.mean()
    scale = np.linalg.norm(ac) * np.linalg.norm(bc)
    if scale == 0:
        return 1.0 if np.array_equal(a, b) else float("-inf")
    return float(np.dot(ac, bc) / scale)


def resample_quality(
    out,
    ref,
    *,
    max_lag=64,
    min_r=0.98,
    max_nrmse=0.17,
    max_gain_error=0.05,
    max_bias=0.005,
    max_length_delta=1,
):
    """Validate task-level resampling quality across every channel.

    Different libraries intentionally use different anti-aliasing filters, so
    exact elementwise parity is not expected.  We nevertheless require the
    channel shape and sample count (allowing one sample for rate rounding),
    signed correlation, gain, DC bias, and raw aligned RMS error to agree with
    scipy.  A single global lag is estimated from the
    first channel and then applied to every channel, which also catches
    channel-specific delay or corruption without repeating the expensive lag
    search sixteen times.
    """
    out = np.asarray(out)
    ref = np.asarray(ref)
    if out.ndim < 1 or ref.ndim < 1:
        raise RuntimeError("verification failed: resampler output must be at least one-dimensional")
    if out.ndim != ref.ndim or out.shape[:-1] != ref.shape[:-1]:
        raise RuntimeError(f"verification failed: shape {out.shape} != {ref.shape}")
    if abs(out.shape[-1] - ref.shape[-1]) > max_length_delta:
        raise RuntimeError(
            f"verification failed: output length {out.shape[-1]} differs from "
            f"reference length {ref.shape[-1]} by more than {max_length_delta}"
        )
    n = min(out.shape[-1], ref.shape[-1])
    if n <= 2 * max_lag:
        raise RuntimeError("verification failed: resampler output is too short to align")
    if not np.all(np.isfinite(out)) or not np.all(np.isfinite(ref)):
        raise RuntimeError("verification failed: output or reference contains non-finite values")

    a2 = out.reshape(-1, out.shape[-1])[:, :n]
    b2 = ref.reshape(-1, ref.shape[-1])[:, :n]
    lo, hi = max_lag, n - max_lag

    # Search on a strided view: the lag is common to all channels and exact
    # sample precision matters, while evaluating every sample does not.
    stride = max(1, (hi - lo) // 131_072)
    probe_ref = b2[0, lo:hi:stride]
    best_r, best_lag = float("-inf"), 0
    for lag in range(-max_lag, max_lag + 1):
        r = _corr(a2[0, lo + lag : hi + lag : stride], probe_ref)
        if r > best_r:
            best_r, best_lag = r, lag

    correlations = []
    nrmses = []
    gain_errors = []
    biases = []
    for channel in range(a2.shape[0]):
        a = np.asarray(a2[channel, lo + best_lag : hi + best_lag], dtype=np.float64)
        b = np.asarray(b2[channel, lo:hi], dtype=np.float64)
        correlations.append(_corr(a, b))

        a_centered = a - a.mean()
        b_centered = b - b.mean()
        ref_norm = np.linalg.norm(b)
        centered_ref_norm = np.linalg.norm(b_centered)
        if ref_norm == 0 or centered_ref_norm == 0:
            raise RuntimeError("verification failed: reference channel has zero energy")
        nrmses.append(float(np.linalg.norm(a - b) / ref_norm))
        gain = float(np.linalg.norm(a_centered) / centered_ref_norm)
        gain_errors.append(abs(gain - 1.0))
        ref_rms = ref_norm / np.sqrt(b.size)
        biases.append(float(abs((a - b).mean()) / ref_rms))

    worst_r = min(correlations)
    worst_nrmse = max(nrmses)
    worst_gain_error = max(gain_errors)
    worst_bias = max(biases)
    if not worst_r >= min_r:
        raise RuntimeError(f"verification failed: signed correlation {worst_r:.4f} < {min_r}")
    if not worst_nrmse <= max_nrmse:
        raise RuntimeError(f"verification failed: aligned NRMSE {worst_nrmse:.3f} > {max_nrmse}")
    if not worst_gain_error <= max_gain_error:
        raise RuntimeError(
            f"verification failed: gain error {worst_gain_error:.3f} > {max_gain_error}"
        )
    if not worst_bias <= max_bias:
        raise RuntimeError(f"verification failed: normalized bias {worst_bias:.3f} > {max_bias}")
    return f"r={worst_r:.4f}, nrmse={worst_nrmse:.3f}"
