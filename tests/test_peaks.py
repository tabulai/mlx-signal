"""Golden tests: find_peaks vs scipy — exact index and property parity."""

import numpy as np
import pytest
import scipy.signal as sps

import mlx_signal as msig


def check_parity(x, **kwargs):
    p_ref, props_ref = sps.find_peaks(x, **kwargs)
    p_out, props_out = msig.find_peaks(x, **kwargs)
    np.testing.assert_array_equal(p_out, p_ref)
    assert set(props_out) == set(props_ref)
    for key in props_ref:
        np.testing.assert_allclose(props_out[key], props_ref[key], rtol=1e-12,
                                   err_msg=f"property {key!r}")


def noisy(rng, n=2000):
    t = np.arange(n) / 100.0
    return (np.sin(2 * np.pi * 0.7 * t) + 0.3 * rng.standard_normal(n)).astype(np.float64)


def test_no_conditions(rng):
    check_parity(noisy(rng))


def test_plateaus(rng):
    x = np.repeat(rng.standard_normal(300), rng.integers(1, 5, size=300))
    check_parity(x)
    check_parity(x, plateau_size=2)
    check_parity(x, plateau_size=(2, 4))


def test_height(rng):
    x = noisy(rng)
    check_parity(x, height=0.5)
    check_parity(x, height=(0.2, 1.0))
    check_parity(x, height=np.linspace(0, 1, x.size))


def test_threshold(rng):
    x = noisy(rng)
    check_parity(x, threshold=0.05)
    check_parity(x, threshold=(0.01, 0.5))


def test_distance(rng):
    x = noisy(rng)
    check_parity(x, distance=5)
    check_parity(x, distance=50.5)
    check_parity(x, distance=1)


def test_prominence(rng):
    x = noisy(rng)
    check_parity(x, prominence=0.4)
    check_parity(x, prominence=(0.1, 2.0))
    check_parity(x, prominence=0.4, wlen=101)


def test_width(rng):
    x = noisy(rng)
    check_parity(x, width=3)
    check_parity(x, width=(2, 30), rel_height=0.75)


def test_all_conditions_combined(rng):
    x = noisy(rng)
    check_parity(
        x, height=0.1, threshold=1e-4, distance=4, prominence=0.2, width=2,
        wlen=201, rel_height=0.6, plateau_size=1,
    )


def test_degenerate_signals():
    check_parity(np.zeros(100))
    check_parity(np.arange(100, dtype=float))
    check_parity(np.array([0.0, 1.0, 0.0]))
    check_parity(np.array([1.0, 0.5]))
    x = np.zeros(50)
    x[10:14] = 1.0  # single plateau
    check_parity(x)


def test_mlx_input(rng):
    import mlx.core as mx

    x = noisy(rng).astype(np.float32)
    p_ref, _ = sps.find_peaks(x, prominence=0.3)
    p_out, _ = msig.find_peaks(mx.array(x), prominence=0.3)
    np.testing.assert_array_equal(p_out, p_ref)


def test_distance_validation(rng):
    with pytest.raises(ValueError, match="distance"):
        msig.find_peaks(noisy(rng), distance=0.5)


def test_rejects_2d(rng):
    with pytest.raises(ValueError, match="1-D"):
        msig.find_peaks(rng.standard_normal((10, 10)))


def test_large_signal_gpu_prefilter(rng):
    """Above the dispatch threshold the sign-diff pass runs through MLX."""
    x = noisy(rng, n=200_000)
    check_parity(x, prominence=0.5)
