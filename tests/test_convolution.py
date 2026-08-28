"""Golden tests: convolution family vs scipy.signal."""

import numpy as np
import pytest
import scipy.signal as sps

import mlx_signal_processing as msig
from _utils import assert_close, assert_type_and_dtype


@pytest.mark.parametrize("mode", ["full", "same", "valid"])
@pytest.mark.parametrize(
    "n1,n2",
    [(1000, 33), (33, 1000), (128, 128), (997, 61), (4096, 511)],
)
def test_fftconvolve_1d(rng, mode, n1, n2):
    a = rng.standard_normal(n1).astype(np.float32)
    b = rng.standard_normal(n2).astype(np.float32)
    ref = sps.fftconvolve(a, b, mode=mode)
    out = msig.fftconvolve(a, b, mode=mode)
    assert ref.shape == tuple(np.array(out).shape)
    assert_close(out, ref)


@pytest.mark.parametrize("mode", ["full", "same", "valid"])
def test_fftconvolve_complex(rng, mode):
    a = (rng.standard_normal(1000) + 1j * rng.standard_normal(1000)).astype(np.complex64)
    b = (rng.standard_normal(65) + 1j * rng.standard_normal(65)).astype(np.complex64)
    ref = sps.fftconvolve(a, b, mode=mode)
    out = msig.fftconvolve(a, b, mode=mode)
    assert_close(out, ref)
    assert_type_and_dtype(out, complex_ok=True)


def test_fftconvolve_complex_real_mix(rng):
    a = (rng.standard_normal(512) + 1j * rng.standard_normal(512)).astype(np.complex64)
    b = rng.standard_normal(31).astype(np.float32)
    assert_close(msig.fftconvolve(a, b), sps.fftconvolve(a, b))


@pytest.mark.parametrize("mode", ["full", "same", "valid"])
def test_fftconvolve_2d(rng, mode):
    a = rng.standard_normal((64, 64)).astype(np.float32)
    b = rng.standard_normal((8, 8)).astype(np.float32)
    ref = sps.fftconvolve(a, b, mode=mode)
    out = msig.fftconvolve(a, b, mode=mode)
    assert ref.shape == tuple(np.array(out).shape)
    assert_close(out, ref)


def test_fftconvolve_3d(rng):
    a = rng.standard_normal((12, 13, 14)).astype(np.float32)
    b = rng.standard_normal((3, 4, 5)).astype(np.float32)
    assert_close(msig.fftconvolve(a, b), sps.fftconvolve(a, b))


def test_fftconvolve_axes_batch(rng):
    a = rng.standard_normal((3, 1000)).astype(np.float32)
    b = rng.standard_normal((1, 33)).astype(np.float32)
    ref = sps.fftconvolve(a, b, axes=[1])
    out = msig.fftconvolve(a, b, axes=[1])
    assert ref.shape == tuple(np.array(out).shape)
    assert_close(out, ref)


def test_fftconvolve_valid_requires_coverage(rng):
    a = rng.standard_normal((10, 5)).astype(np.float32)
    b = rng.standard_normal((5, 10)).astype(np.float32)
    with pytest.raises(ValueError):
        msig.fftconvolve(a, b, mode="valid")


@pytest.mark.parametrize("mode", ["full", "same", "valid"])
@pytest.mark.parametrize("n1,n2", [(1 << 16, 127), (100000, 63), (5000, 4999)])
def test_oaconvolve_1d(rng, mode, n1, n2):
    a = rng.standard_normal(n1).astype(np.float32)
    b = rng.standard_normal(n2).astype(np.float32)
    ref = sps.oaconvolve(a, b, mode=mode)
    out = msig.oaconvolve(a, b, mode=mode)
    assert ref.shape == tuple(np.array(out).shape)
    assert_close(out, ref)


def test_oaconvolve_batched_axes(rng):
    a = rng.standard_normal((4, 1 << 15)).astype(np.float32)
    b = rng.standard_normal((1, 63)).astype(np.float32)
    ref = sps.oaconvolve(a, b, axes=[1])
    out = msig.oaconvolve(a, b, axes=[1])
    assert ref.shape == tuple(np.array(out).shape)
    assert_close(out, ref)


def test_oaconvolve_complex(rng):
    a = (rng.standard_normal(1 << 15) + 1j * rng.standard_normal(1 << 15)).astype(np.complex64)
    b = (rng.standard_normal(101) + 1j * rng.standard_normal(101)).astype(np.complex64)
    assert_close(msig.oaconvolve(a, b), sps.oaconvolve(a, b))


def test_oaconvolve_equal_shapes_delegates(rng):
    a = rng.standard_normal(2048).astype(np.float32)
    b = rng.standard_normal(2048).astype(np.float32)
    assert_close(msig.oaconvolve(a, b), sps.oaconvolve(a, b))


@pytest.mark.parametrize("mode", ["full", "same", "valid"])
def test_correlate_real(rng, mode):
    a = rng.standard_normal(1000).astype(np.float32)
    b = rng.standard_normal(100).astype(np.float32)
    ref = sps.correlate(a, b, mode=mode)
    out = msig.correlate(a, b, mode=mode)
    assert ref.shape == tuple(np.array(out).shape)
    assert_close(out, ref)


@pytest.mark.parametrize("mode", ["full", "same", "valid"])
def test_correlate_complex_conjugation(rng, mode):
    a = (rng.standard_normal(500) + 1j * rng.standard_normal(500)).astype(np.complex64)
    b = (rng.standard_normal(80) + 1j * rng.standard_normal(80)).astype(np.complex64)
    ref = sps.correlate(a, b, mode=mode)
    out = msig.correlate(a, b, mode=mode)
    assert_close(out, ref)


def test_correlate_2d(rng):
    a = rng.standard_normal((50, 40)).astype(np.float32)
    b = rng.standard_normal((8, 9)).astype(np.float32)
    assert_close(msig.correlate(a, b), sps.correlate(a, b, method="fft"))


def test_correlate_peak_at_shift(rng):
    x = rng.standard_normal(4096).astype(np.float32)
    y = np.roll(x, 100)
    c = np.array(msig.correlate(y, x, mode="full"))
    lags = msig.correlation_lags(len(y), len(x), mode="full")
    assert lags[np.argmax(c)] == 100


@pytest.mark.parametrize("mode", ["full", "same", "valid"])
@pytest.mark.parametrize("n1,n2", [(100, 30), (30, 100), (101, 31)])
def test_correlation_lags_exact(mode, n1, n2):
    np.testing.assert_array_equal(
        msig.correlation_lags(n1, n2, mode), sps.correlation_lags(n1, n2, mode)
    )


def test_convolve_wrapper(rng):
    a = rng.standard_normal(2000).astype(np.float32)
    b = rng.standard_normal(100).astype(np.float32)
    assert_close(msig.convolve(a, b, mode="same"), sps.convolve(a, b, mode="same"))


def test_convolve_direct_routes_to_scipy(rng):
    a = rng.standard_normal(100).astype(np.float32)
    b = rng.standard_normal(10).astype(np.float32)
    out = msig.convolve(a, b, method="direct")
    assert_type_and_dtype(out)
    assert_close(out, sps.convolve(a, b, method="direct"))


@pytest.mark.parametrize(
    "func", [msig.convolve, msig.correlate], ids=["convolve", "correlate"]
)
@pytest.mark.parametrize("method", ["auto", "fft"])
def test_convolution_wrappers_valid_require_full_coverage(func, method):
    """Unlike fftconvolve, the high-level wrappers check singleton axes too."""
    a = np.ones((1, 4), dtype=np.float32)
    b = np.ones((7, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="at least as large"):
        func(a, b, mode="valid", method=method)


@pytest.mark.parametrize(
    "func,ref_func",
    [(msig.convolve, sps.convolve), (msig.correlate, sps.correlate)],
    ids=["convolve", "correlate"],
)
@pytest.mark.parametrize("method", ["auto", "fft"])
@pytest.mark.parametrize("swap", [False, True], ids=["larger-first", "larger-second"])
def test_convolution_wrappers_valid_covered_inputs(func, ref_func, method, swap, rng):
    large = rng.standard_normal((7, 5)).astype(np.float32)
    small = rng.standard_normal((2, 3)).astype(np.float32)
    a, b = (small, large) if swap else (large, small)
    out = func(a, b, mode="valid", method=method)
    ref = ref_func(a, b, mode="valid", method=method)
    assert np.array(out).shape == ref.shape
    assert_close(out, ref)


@pytest.mark.parametrize("func", [msig.fftconvolve, msig.oaconvolve])
def test_fft_convolution_valid_keeps_singleton_axis_semantics(func):
    """The FFT-specific APIs permit mixed coverage when singleton axes broadcast."""
    a = np.ones((1, 4), dtype=np.float32)
    b = np.ones((7, 2), dtype=np.float32)
    ref_func = sps.fftconvolve if func is msig.fftconvolve else sps.oaconvolve
    assert_close(func(a, b, mode="valid"), ref_func(a, b, mode="valid"))


def test_fftconvolve_equal_pair_fourstep(rng):
    """Equal-size pair whose padded FFT lands on a broken Metal length: the
    1-D wrappers must route it through the GPU four-step path (it used to run
    on the CPU stream via rfftn)."""
    n = (1 << 19) + 21  # fl = 2^21, in the broken range
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    assert_close(msig.fftconvolve(a, b), sps.fftconvolve(a, b), rtol=2e-4, atol_frac=2e-5)


@pytest.mark.parametrize("complex_input", [False, True])
def test_fftconvolve_same_object(rng, complex_input):
    """fftconvolve(x, x) computes one forward transform and squares it."""
    if complex_input:
        x = (rng.standard_normal(30000) + 1j * rng.standard_normal(30000)).astype(np.complex64)
    else:
        x = rng.standard_normal(30000).astype(np.float32)
    assert_close(msig.fftconvolve(x, x), sps.fftconvolve(x, x), rtol=2e-4, atol_frac=2e-5)


def test_convolve_same_object(rng):
    x = rng.standard_normal(20000).astype(np.float32)
    assert_close(msig.convolve(x, x), sps.convolve(x, x, method="fft"), rtol=2e-4,
                 atol_frac=2e-5)


@pytest.mark.parametrize("mode", ["full", "same", "valid"])
@pytest.mark.parametrize("complex_input", [False, True])
def test_correlate_same_object(rng, mode, complex_input):
    """correlate(x, x) uses the conjugate-spectrum shortcut (one forward FFT)."""
    if complex_input:
        x = (rng.standard_normal(30000) + 1j * rng.standard_normal(30000)).astype(np.complex64)
    else:
        x = rng.standard_normal(30000).astype(np.float32)
    ref = sps.correlate(x, x, mode=mode, method="fft")
    out = msig.correlate(x, x, mode=mode)
    assert ref.shape == tuple(np.array(out).shape)
    assert_close(out, ref, rtol=2e-4, atol_frac=2e-5)


def test_correlate_same_object_2d(rng):
    x = rng.standard_normal((6, 4000)).astype(np.float32)
    assert_close(msig.correlate(x, x), sps.correlate(x, x, method="fft"), rtol=2e-4,
                 atol_frac=2e-5)


def test_correlate_same_object_singleton_axis(rng):
    x = rng.standard_normal((1, 2000)).astype(np.float32)
    assert_close(msig.correlate(x, x), sps.correlate(x, x, method="fft"), rtol=2e-4,
                 atol_frac=2e-5)


def test_correlate_same_object_autocorr_peak(rng):
    """Physics check: autocorrelation peaks at zero lag with energy value."""
    x = rng.standard_normal(8192).astype(np.float32)
    r = np.array(msig.correlate(x, x, mode="full"))
    assert int(np.argmax(r)) == x.size - 1
    np.testing.assert_allclose(r[x.size - 1], float(np.dot(x, x)), rtol=1e-5)


def test_fftconvolve_pair_broadcast_batch_fourstep(rng):
    """Mismatched batch dims can't stack into one batched FFT; the packed
    path must transform separately and broadcast the product."""
    n = (1 << 19) + 7  # fl = 2^21: packed-pair path
    a = rng.standard_normal((4, n)).astype(np.float32)
    b = rng.standard_normal((1, n)).astype(np.float32)
    assert_close(msig.fftconvolve(a, b, axes=[1]), sps.fftconvolve(a, b, axes=[1]),
                 rtol=2e-4, atol_frac=2e-5)


def test_fftconvolve_same_object_fourstep(rng):
    """Auto-convolution at a broken Metal length runs the packed-pair path
    with a single forward transform."""
    x = rng.standard_normal((1 << 19) + 21).astype(np.float32)
    assert_close(msig.fftconvolve(x, x), sps.fftconvolve(x, x), rtol=2e-4, atol_frac=2e-5)


@pytest.mark.parametrize("length", [(1 << 19) + 20, (1 << 19) + 21])  # even, odd
@pytest.mark.parametrize("mode", ["full", "same", "valid"])
def test_correlate_same_object_fourstep(rng, length, mode):
    """Autocorrelation at broken Metal lengths uses the packed identity
    (odd lengths take the zero-prepend realignment branch)."""
    x = rng.standard_normal(length).astype(np.float32)
    ref = sps.correlate(x, x, mode=mode, method="fft")
    out = msig.correlate(x, x, mode=mode)
    assert ref.shape == tuple(np.array(out).shape)
    assert_close(out, ref, rtol=2e-4, atol_frac=2e-5)
