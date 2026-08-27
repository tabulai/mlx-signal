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


def test_gpu_prefilter_preserves_tiny_normal_ordering():
    """Adjacent normal float32 values can differ by a subnormal ULP; direct
    comparisons must not flush that strict peak into a plateau."""
    tiny = np.float32(np.finfo(np.float32).tiny)
    above = np.nextafter(tiny, np.float32(np.inf))
    check_parity(np.array([tiny, above, tiny], np.float32))


def test_gpu_prefilter_denormal_signal_falls_back_exact():
    tiny = np.float32(np.finfo(np.float32).smallest_subnormal)
    check_parity(np.array([0.0, tiny, 0.0], np.float32))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nan_gap_is_not_a_plateau_peak(dtype):
    """Unordered pairs must break a rise/fall sequence; treating them as an
    equality plateau incorrectly reports the NaN itself as a peak."""
    x = np.array([0, 1, np.nan, 1, 0, 0, 2, 1], dtype=dtype)
    check_parity(x)


# ---------------------------------------------------------------------------
# GPU prominence-base kernel (scipy-bit-exact; scipy fallback is trivially
# exact, so every assertion here is array_equal on all routes)
# ---------------------------------------------------------------------------


def _assert_prom_equal(x32, peaks=None, **kwargs):
    import warnings

    x64 = x32.astype(np.float64)
    if peaks is None:
        peaks, _ = sps.find_peaks(x64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sps.peak_prominences(x64, peaks, **kwargs)
        got = msig.peak_prominences(x32, peaks, **kwargs)
    for r, g, name in zip(ref, got, ["prominences", "left_bases", "right_bases"],
                          strict=True):
        # strict=True also pins the (float64, intp, intp) dtype contract
        np.testing.assert_array_equal(np.asarray(g), r, err_msg=name, strict=True)


def test_prominences_bitwise_noise(rng):
    _assert_prom_equal(rng.standard_normal(1 << 20).astype(np.float32))


def test_prominences_bitwise_tie_heavy(rng):
    """Quantized signals stress the min tie-break (closest-to-peak wins) and
    the pending-block descent of the skip scan. Coarse quantization makes the
    scipy reference walk O(P*N), so sizes stay just past a few skip blocks."""
    _assert_prom_equal(rng.integers(0, 6, size=1 << 14).astype(np.float32))
    _assert_prom_equal(rng.integers(0, 60, size=1 << 17).astype(np.float32))
    _assert_prom_equal(np.tile(np.array([0.0, 1.0, 0.0, 2.0], np.float32), 1 << 15))


def test_prominences_bitwise_nan(rng):
    """NaN terminates scipy's walk (any comparison is false); the kernel's
    !(v <= xp) predicate and NaN-propagating block maxima must agree."""
    x = rng.standard_normal(1 << 17).astype(np.float32) * 0.1
    x[65536], x[65535], x[65537] = 5.0, 0.0, 0.0
    x[30000] = np.nan  # inside an otherwise-skippable low block
    x[100000] = np.nan
    _assert_prom_equal(x)


def test_prominences_block_max_equal_to_peak_is_not_terminator():
    """A distant block whose max EQUALS the peak height must be skipped or
    descended, never treated as a terminator — scipy's walk continues through
    equal values to the true base beyond it."""
    from mlx_signal._peaks_metal import BLOCK

    x = np.full(5 * BLOCK, 4.0, np.float32)
    x[4 * BLOCK + 100] = 10.0   # queried peak (block 4)
    x[3 * BLOCK + 12] = 10.0    # block 3: bmax == xp, nothing greater
    x[2 * BLOCK + 808] = 1.0    # block 2: the true running min
    x[BLOCK + 904] = 20.0       # block 1: the true terminator
    x[4 * BLOCK + 150] = 20.0   # right-side terminator
    _assert_prom_equal(x, peaks=np.array([4 * BLOCK + 100], dtype=np.intp))


def test_prominences_negative_stride_view(rng):
    """A reversed float32 view is ordinary numpy and must not crash the GPU
    route (mx.array refuses negative-stride DLPack exports)."""
    x = rng.standard_normal(1 << 17).astype(np.float32)[::-1]
    _assert_prom_equal(x)


def test_prominences_negative_stride_singleton():
    """Singleton views can advertise C-contiguity while retaining stride -4."""
    _assert_prom_equal(
        np.array([1.0], np.float32)[::-1], peaks=np.array([0], dtype=np.intp)
    )


def test_prominences_arbitrary_indices(rng):
    """scipy accepts any valid index, not only maxima (prominence may be 0)."""
    x = rng.standard_normal(1 << 16).astype(np.float32)
    _assert_prom_equal(x, peaks=rng.integers(0, 1 << 16, 5000).astype(np.intp))


def test_prominences_mx_input(rng):
    import mlx.core as mx

    x32 = rng.standard_normal(1 << 18).astype(np.float32)
    peaks, _ = sps.find_peaks(x32.astype(np.float64))
    ref = sps.peak_prominences(x32.astype(np.float64), peaks)
    got = msig.peak_prominences(mx.array(x32), peaks)
    for r, g in zip(ref, got, strict=True):
        np.testing.assert_array_equal(np.asarray(g), r, strict=True)


def test_prominences_wlen_falls_back_exact(rng):
    x = rng.standard_normal(1 << 18).astype(np.float32)
    _assert_prom_equal(x, wlen=501)


def test_prominences_denormals_fall_back_exact(rng):
    """Metal compares denormals as zero; such signals keep scipy's walk."""
    _assert_prom_equal((rng.standard_normal(1 << 14) * 1e-40).astype(np.float32))


def test_prominences_error_parity(rng):
    x = rng.standard_normal(1 << 16).astype(np.float32)
    with pytest.raises(ValueError, match="not a valid index"):
        msig.peak_prominences(x, np.array([1 << 20]))
    with pytest.raises(ValueError, match="not a valid index"):
        msig.peak_prominences(x, np.array([-1]))
    with pytest.raises(TypeError, match="cannot safely cast"):
        msig.peak_prominences(x, np.array([1.5]))
    with pytest.raises(ValueError, match="1-D"):
        msig.peak_prominences(x, np.array([[1]]))


def test_prominences_zero_warning_parity():
    """The kernel path emits scipy's PeakPropertyWarning class for zero
    prominences, so pytest.warns filters behave identically."""
    from scipy.signal._peak_finding_utils import PeakPropertyWarning

    x = np.ones(1 << 17, np.float32)
    with pytest.warns(PeakPropertyWarning, match="prominence of 0"):
        msig.peak_prominences(x, np.array([100]))


def test_peak_widths_uses_gpu_prominence_data(rng):
    x = rng.standard_normal(1 << 19).astype(np.float32)
    peaks, _ = sps.find_peaks(x.astype(np.float64))
    ref = sps.peak_widths(x.astype(np.float64), peaks)
    got = msig.peak_widths(x, peaks)
    for r, g in zip(ref, got, strict=True):
        np.testing.assert_array_equal(np.asarray(g), r)


def test_find_peaks_prominence_width_bitwise(rng):
    x = rng.standard_normal(1 << 20).astype(np.float32)
    ref_p, ref_props = sps.find_peaks(x.astype(np.float64), prominence=0.5, width=2)
    got_p, got_props = msig.find_peaks(x, prominence=0.5, width=2)
    np.testing.assert_array_equal(got_p, ref_p)
    assert set(got_props) == set(ref_props)
    for k in ref_props:
        np.testing.assert_array_equal(got_props[k], ref_props[k], err_msg=k)
