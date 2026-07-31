"""Shared test helpers: fp32-tolerance comparison against float64 scipy references."""

import mlx.core as mx
import numpy as np


def as_np(x):
    if isinstance(x, mx.array):
        return np.array(x)
    return np.asarray(x)


def assert_close(actual, desired, rtol=1e-4, atol_frac=1e-5):
    """Compare an mlx-signal result against a scipy reference.

    fp32 pipelines can't hit fp64 pointwise relative accuracy at spectral
    nulls, so the absolute tolerance is scaled by the reference's peak
    magnitude: |err| <= rtol*|ref| + atol_frac*max|ref|.
    """
    a = as_np(actual)
    d = np.asarray(desired)
    assert a.shape == d.shape, f"shape mismatch: {a.shape} vs {d.shape}"
    scale = np.max(np.abs(d)) if d.size else 1.0
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    np.testing.assert_allclose(a, d.astype(a.dtype), rtol=rtol, atol=atol_frac * scale)


def assert_type_and_dtype(x, complex_ok=False):
    assert isinstance(x, mx.array), f"expected mx.array, got {type(x)}"
    if complex_ok:
        assert x.dtype in (mx.float32, mx.complex64), x.dtype
    else:
        assert x.dtype == mx.float32, x.dtype
