"""Golden tests: firwin, lfilter (FIR path), filtfilt vs scipy."""

import numpy as np
import pytest
import scipy.signal as sps

import mlx_signal as msig
from _utils import assert_close, assert_type_and_dtype
from conftest import HAS_GPU


def test_firwin_matches_scipy():
    ref = sps.firwin(65, 0.3)
    out = msig.firwin(65, 0.3)
    assert_type_and_dtype(out)
    np.testing.assert_allclose(np.array(out), ref.astype(np.float32), rtol=1e-6)


def test_firwin_kwargs():
    ref = sps.firwin(64, [0.1, 0.4], pass_zero=False, window=("kaiser", 8.0))
    out = msig.firwin(64, [0.1, 0.4], pass_zero=False, window=("kaiser", 8.0))
    np.testing.assert_allclose(np.array(out), ref.astype(np.float32), rtol=1e-6)


def test_firwin2_matches_scipy():
    ref = sps.firwin2(65, [0.0, 0.3, 0.6, 1.0], [1.0, 1.0, 0.0, 0.0])
    out = msig.firwin2(65, [0.0, 0.3, 0.6, 1.0], [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(np.array(out), ref.astype(np.float32), rtol=1e-6)


@pytest.mark.parametrize("ntaps", [1, 8, 101])
@pytest.mark.parametrize("n", [1000, 997])
def test_lfilter_fir(rng, ntaps, n):
    b = rng.standard_normal(ntaps).astype(np.float32)
    x = rng.standard_normal(n).astype(np.float32)
    ref = sps.lfilter(b, [1.0], x)
    out = msig.lfilter(b, [1.0], x)
    assert ref.shape == tuple(np.array(out).shape)
    assert_close(out, ref)


def test_lfilter_fir_scalar_a(rng):
    b = rng.standard_normal(31).astype(np.float32)
    x = rng.standard_normal(500).astype(np.float32)
    assert_close(msig.lfilter(b, 2.0, x), sps.lfilter(b, 2.0, x))


def test_lfilter_fir_2d_axis(rng):
    b = np.array(msig.firwin(33, 0.2))
    x = rng.standard_normal((4, 800)).astype(np.float32)
    assert_close(msig.lfilter(b, [1.0], x, axis=-1), sps.lfilter(b, [1.0], x, axis=-1))
    x2 = rng.standard_normal((800, 4)).astype(np.float32)
    assert_close(msig.lfilter(b, [1.0], x2, axis=0), sps.lfilter(b, [1.0], x2, axis=0))


def test_lfilter_fir_complex(rng):
    b = rng.standard_normal(21).astype(np.float32)
    x = (rng.standard_normal(400) + 1j * rng.standard_normal(400)).astype(np.complex64)
    assert_close(msig.lfilter(b, [1.0], x), sps.lfilter(b, [1.0], x))


def test_lfilter_taps_longer_than_signal(rng):
    b = rng.standard_normal(64).astype(np.float32)
    x = rng.standard_normal(40).astype(np.float32)
    ref = sps.lfilter(b, [1.0], x)
    assert_close(msig.lfilter(b, [1.0], x), ref)


def test_lfilter_iir_falls_back(rng):
    x = rng.standard_normal(1000).astype(np.float32)
    b, a = sps.butter(4, 0.2)
    with msig.config_context(dispatch="auto", gpu_min_size=1, warn_on_fallback=True):
        with pytest.warns(msig.FallbackWarning):
            out = msig.lfilter(b, a, x)
    assert_close(out, sps.lfilter(b, a, x), rtol=1e-3)
    if HAS_GPU:  # fixture pins dispatch="mlx" only when Metal exists
        with pytest.raises(NotImplementedError):
            msig.lfilter(b, a, x)


def test_lfilter_zi_falls_back(rng):
    b = np.array(msig.firwin(9, 0.3))
    x = rng.standard_normal(300).astype(np.float32)
    zi = sps.lfilter_zi(b, [1.0]) * x[0]
    with msig.config_context(dispatch="auto", gpu_min_size=1, warn_on_fallback=True):
        with pytest.warns(msig.FallbackWarning):
            y, zf = msig.lfilter(b, [1.0], x, zi=zi)
    y_ref, zf_ref = sps.lfilter(b, [1.0], x, zi=zi)
    assert_close(y, y_ref)
    assert_close(zf, zf_ref)


@pytest.mark.parametrize("padtype", ["odd", "even", "constant", None])
def test_filtfilt_fir_padtypes(rng, padtype):
    b = np.array(msig.firwin(31, 0.25))
    x = rng.standard_normal(1000).astype(np.float32)
    ref = sps.filtfilt(b, [1.0], x, padtype=padtype)
    out = msig.filtfilt(b, [1.0], x, padtype=padtype)
    assert ref.shape == tuple(np.array(out).shape)
    assert_close(out, ref)


def test_filtfilt_fir_padlen(rng):
    b = np.array(msig.firwin(21, 0.3))
    x = rng.standard_normal(600).astype(np.float32)
    assert_close(msig.filtfilt(b, [1.0], x, padlen=150), sps.filtfilt(b, [1.0], x, padlen=150))
    assert_close(msig.filtfilt(b, [1.0], x, padlen=0), sps.filtfilt(b, [1.0], x, padlen=0))


def test_filtfilt_fir_2d(rng):
    b = np.array(msig.firwin(15, 0.4))
    x = rng.standard_normal((5, 700)).astype(np.float32)
    assert_close(msig.filtfilt(b, [1.0], x), sps.filtfilt(b, [1.0], x))
    x2 = rng.standard_normal((700, 5)).astype(np.float32)
    assert_close(msig.filtfilt(b, [1.0], x2, axis=0), sps.filtfilt(b, [1.0], x2, axis=0))


def test_filtfilt_padlen_too_long(rng):
    b = np.array(msig.firwin(31, 0.25))
    x = rng.standard_normal(50).astype(np.float32)
    with pytest.raises(ValueError, match="padlen"):
        msig.filtfilt(b, [1.0], x)


def test_filtfilt_iir_falls_back(rng):
    x = rng.standard_normal(1000).astype(np.float32)
    b, a = sps.butter(3, 0.1)
    with msig.config_context(dispatch="auto", gpu_min_size=1, warn_on_fallback=True):
        with pytest.warns(msig.FallbackWarning):
            out = msig.filtfilt(b, a, x)
    assert_close(out, sps.filtfilt(b, a, x), rtol=1e-3)


def test_filtfilt_zero_phase_property(rng):
    """filtfilt of a delayed impulse stays centered (no group delay)."""
    b = np.array(msig.firwin(51, 0.2))
    x = np.zeros(512, dtype=np.float32)
    x[256] = 1.0
    y = np.array(msig.filtfilt(b, [1.0], x))
    assert abs(int(np.argmax(y)) - 256) <= 1
