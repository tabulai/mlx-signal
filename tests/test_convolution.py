"""Golden tests: convolution family vs scipy.signal."""

import numpy as np
import pytest
import scipy.signal as sps

import mlx_signal as msig
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
