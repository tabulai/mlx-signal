"""Golden tests: resample (FFT method) and hilbert vs scipy.signal."""

import numpy as np
import pytest
import scipy.signal as sps

import mlx_signal as msig
from _utils import assert_close, assert_type_and_dtype


@pytest.mark.parametrize("n,num", [
    (1000, 500), (1000, 2000), (1000, 1000), (1000, 333), (1000, 1001),
    (999, 500), (999, 1998), (999, 334), (1024, 320),
])
def test_resample_real(rng, n, num):
    x = rng.standard_normal(n).astype(np.float32)
    ref = sps.resample(x, num)
    out = msig.resample(x, num)
    assert ref.shape == tuple(np.array(out).shape)
    assert_close(out, ref)
    assert_type_and_dtype(out)


@pytest.mark.parametrize("n,num", [(1000, 500), (1000, 2000), (999, 500), (999, 2001)])
def test_resample_complex(rng, n, num):
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    ref = sps.resample(x, num)
    out = msig.resample(x, num)
    assert ref.shape == tuple(np.array(out).shape)
    assert_close(out, ref)


def test_resample_domain_freq(rng):
    x = (rng.standard_normal(512) + 1j * rng.standard_normal(512)).astype(np.complex64)
    ref = sps.resample(x, 200, domain="freq")
    out = msig.resample(x, 200, domain="freq")
    assert_close(out, ref)


@pytest.mark.parametrize("window", ["hamming", ("kaiser", 5.0)])
def test_resample_window_spec(rng, window):
    x = rng.standard_normal(1000).astype(np.float32)
    ref = sps.resample(x, 400, window=window)
    out = msig.resample(x, 400, window=window)
    assert_close(out, ref)


def test_resample_window_array_and_callable(rng):
    x = rng.standard_normal(1000).astype(np.float32)
    w = (np.abs(np.fft.fftfreq(1000)) < 0.25).astype(np.float64)
    assert_close(msig.resample(x, 400, window=w), sps.resample(x, 400, window=w))

    def wf(f):
        return (np.abs(f) < 0.3).astype(np.float64)

    assert_close(msig.resample(x, 400, window=wf), sps.resample(x, 400, window=wf))


def test_resample_axis_default_is_zero(rng):
    x = rng.standard_normal((1000, 3)).astype(np.float32)
    ref = sps.resample(x, 250)
    out = msig.resample(x, 250)
    assert ref.shape == tuple(np.array(out).shape)
    assert_close(out, ref)


def test_resample_axis1(rng):
    x = rng.standard_normal((3, 1000)).astype(np.float32)
    ref = sps.resample(x, 250, axis=1)
    out = msig.resample(x, 250, axis=1)
    assert_close(out, ref)


def test_resample_with_t(rng):
    x = rng.standard_normal(1000).astype(np.float32)
    t = np.arange(1000) / 100.0
    ref_x, ref_t = sps.resample(x, 500, t=t)
    out_x, out_t = msig.resample(x, 500, t=t)
    assert_close(out_x, ref_x)
    assert_close(out_t, ref_t)


def test_resample_tone_preserved(rng):
    fs, f0 = 1000.0, 50.0
    t = np.arange(2048) / fs
    x = np.sin(2 * np.pi * f0 * t).astype(np.float32)
    y = np.array(msig.resample(x, 1024))
    t2 = np.arange(1024) / (fs / 2)
    # interior only: FFT resampling rings at the edges of a non-periodic tone
    ref = np.sin(2 * np.pi * f0 * t2)
    np.testing.assert_allclose(y[100:-100], ref[100:-100], atol=5e-3)


@pytest.mark.parametrize("n", [1024, 999])
@pytest.mark.parametrize("N", [None, 1500, 700])
def test_hilbert_matches_scipy(rng, n, N):
    x = rng.standard_normal(n).astype(np.float32)
    ref = sps.hilbert(x, N=N)
    out = msig.hilbert(x, N=N)
    assert ref.shape == tuple(np.array(out).shape)
    assert_close(out, ref)


def test_hilbert_2d_axis(rng):
    x = rng.standard_normal((4, 1000)).astype(np.float32)
    assert_close(msig.hilbert(x, axis=-1), sps.hilbert(x, axis=-1))
    x2 = rng.standard_normal((1000, 4)).astype(np.float32)
    assert_close(msig.hilbert(x2, axis=0), sps.hilbert(x2, axis=0))


def test_hilbert_envelope():
    t = np.arange(4096, dtype=np.float32) / 4096
    x = (1.0 + 0.5 * np.sin(2 * np.pi * 3 * t)) * np.sin(2 * np.pi * 400 * t)
    env = np.abs(np.array(msig.hilbert(x)))
    expected = 1.0 + 0.5 * np.sin(2 * np.pi * 3 * t)
    np.testing.assert_allclose(env[64:-64], expected[64:-64], atol=2e-2)


def test_hilbert_rejects_complex(rng):
    x = (rng.standard_normal(100) + 1j * rng.standard_normal(100)).astype(np.complex64)
    with pytest.raises(ValueError, match="real"):
        msig.hilbert(x)
