"""Regression tests for the audit findings: degenerate inputs, complex
coefficients/windows, dispatch parity, and config robustness.

Every test here corresponds to a case that previously crashed the interpreter,
silently produced wrong values, or diverged from scipy.
"""

import warnings

import mlx.core as mx
import numpy as np
import pytest
import scipy.signal as sps

import mlx_signal as msig
from _utils import assert_close
from conftest import HAS_GPU

# ---------------------------------------------------------------------------
# empty / degenerate inputs
# ---------------------------------------------------------------------------


def test_sosfilt_empty_sos_raises(rng):
    """0-section sos previously aborted the interpreter inside the kernel."""
    x = rng.standard_normal(100).astype(np.float32)
    empty = np.empty((0, 6))
    with pytest.raises(ValueError, match="at least one section"):
        msig.sosfilt(empty, x)
    with msig.config_context(dispatch="scipy"):
        with pytest.raises(ValueError, match="at least one section"):
            msig.sosfilt(empty, x)
    with pytest.raises(ValueError, match="at least one section"):
        msig.sosfiltfilt(empty, x)


@pytest.mark.parametrize("func", ["welch", "csd"])
def test_empty_input_matches_scipy(func, rng):
    """Empty signals must return scipy's (0,)-shaped results, not raise."""
    x = np.array([], dtype=np.float32)
    if func == "welch":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_ref, p_ref = sps.welch(x)
            f, p = msig.welch(x)
    else:
        y = rng.standard_normal(0).astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_ref, p_ref = sps.csd(x, y)
            f, p = msig.csd(x, y)
    assert np.array(f).shape == f_ref.shape
    assert np.array(p).shape == p_ref.shape


def test_empty_upfirdn_matches_scipy(rng):
    """Empty x still produces the filter-determined number of output samples."""
    h = rng.standard_normal(5).astype(np.float32)
    x = np.array([], dtype=np.float32)
    ref = sps.upfirdn(h, x, up=2, down=1)
    out = msig.upfirdn(h, x, up=2, down=1)
    assert np.array(out).shape == ref.shape
    np.testing.assert_allclose(np.array(out), ref)

    xc = np.array([], dtype=np.complex64)
    out_c = msig.upfirdn(h, xc, up=2, down=1)
    assert np.array(out_c).dtype == np.complex64


def test_empty_resample_poly_matches_scipy():
    x = np.array([], dtype=np.float32)
    ref = sps.resample_poly(x, 2, 3)
    out = msig.resample_poly(x, 2, 3)
    assert np.array(out).shape == ref.shape


# ---------------------------------------------------------------------------
# nperseg=1 spectral paths (denom == 0 in the linear-detrend closed form)
# ---------------------------------------------------------------------------


def test_welch_nperseg_1_matches_scipy(rng):
    x = rng.standard_normal(64).astype(np.float32)
    f_ref, p_ref = sps.welch(x, nperseg=1)
    f, p = msig.welch(x, nperseg=1)
    np.testing.assert_allclose(np.array(f), f_ref)
    np.testing.assert_allclose(np.array(p), p_ref, rtol=1e-5)


def test_welch_nperseg_1_linear_detrend(rng):
    """Linear detrend of a single point == mean removal (scipy semantics)."""
    x = rng.standard_normal(64).astype(np.float32)
    f_ref, p_ref = sps.welch(x, nperseg=1, detrend="linear")
    f, p = msig.welch(x, nperseg=1, detrend="linear")
    np.testing.assert_allclose(np.array(p), p_ref, rtol=1e-5, atol=1e-12)


def test_stft_nperseg_1_matches_scipy(rng):
    x = rng.standard_normal(32).astype(np.float32)
    f_ref, t_ref, z_ref = sps.stft(x, nperseg=1)
    f, t, z = msig.stft(x, nperseg=1)
    assert np.array(z).shape == z_ref.shape
    np.testing.assert_allclose(np.array(z), z_ref, rtol=1e-5, atol=1e-6)


def test_csd_nperseg_1_matches_scipy(rng):
    x = rng.standard_normal(48).astype(np.float32)
    y = rng.standard_normal(48).astype(np.float32)
    f_ref, p_ref = sps.csd(x, y, nperseg=1)
    f, p = msig.csd(x, y, nperseg=1)
    np.testing.assert_allclose(np.array(p), p_ref, rtol=1e-5, atol=1e-7)


def test_fft_length_one_axis(rng):
    """MLX's batched length-1 rfft used to fold the batch into complex pairs."""
    from mlx_signal import _fft_core

    x = mx.array(rng.standard_normal((4, 1)).astype(np.float32))
    np.testing.assert_allclose(
        np.array(_fft_core.rfft(x, axis=-1)),
        np.fft.rfft(np.array(x), axis=-1),
        rtol=1e-6,
    )
    xc = x.astype(mx.complex64)
    np.testing.assert_allclose(
        np.array(_fft_core.fft(xc, axis=-1)), np.fft.fft(np.array(x), axis=-1), rtol=1e-6
    )
    np.testing.assert_allclose(
        np.array(_fft_core.irfft(_fft_core.rfft(x, axis=-1), n=1, axis=-1)),
        np.fft.irfft(np.fft.rfft(np.array(x), axis=-1), n=1, axis=-1),
        rtol=1e-6,
        atol=1e-7,
    )


# ---------------------------------------------------------------------------
# ill-conditioned IIR: the A^L scan must not silently lose accuracy
# ---------------------------------------------------------------------------


def test_sosfilt_narrowband_long_signal_accurate(rng):
    """Poles at radius ~0.99996: the block scan is numerically unsafe here,
    so the dispatcher must route to the sequential kernel and match scipy-f32.
    """
    sos = sps.butter(2, 1e-5, output="sos")
    x = np.zeros(200_000, dtype=np.float32)
    x[0] = 1.0
    # every dispatch route executes the float32-quantized filter
    ref = sps.sosfilt(sos.astype(np.float32), x)
    out = np.array(msig.sosfilt(sos, x))
    scale = np.abs(ref).max()
    np.testing.assert_allclose(out, ref, atol=1e-4 * scale)


def test_sosfilt_wideband_still_uses_scan(rng):
    """Well-conditioned filters must keep the fast scan path on long inputs."""
    from mlx_signal.filtering import _scan_safe

    good = sps.butter(4, 0.2, output="sos")
    bad = sps.butter(2, 1e-5, output="sos")
    good = np.asarray(good, dtype=np.float64)
    bad = np.asarray(bad, dtype=np.float64)
    assert _scan_safe(good.tobytes(), good.shape[0])
    assert not _scan_safe(bad.tobytes(), bad.shape[0])


# ---------------------------------------------------------------------------
# complex coefficients / windows must not be silently degraded
# ---------------------------------------------------------------------------


def test_sosfiltfilt_complex_sos_matches_scipy(rng):
    """Complex sos previously crashed in zi construction before the fallback."""
    sos = sps.butter(2, 0.2, output="sos").astype(np.complex128)
    sos[0, 0] *= np.exp(0.3j)
    x = rng.standard_normal(2000).astype(np.float32)
    ref = sps.sosfiltfilt(sos, x)
    with msig.config_context(dispatch="auto", gpu_min_size=0, warn_on_fallback=False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = msig.sosfiltfilt(sos, x)
    assert_close(out, ref, rtol=1e-3, atol_frac=1e-4)
    if HAS_GPU:
        # under pinned dispatch="mlx" the capability gap raises cleanly instead
        # of dying inside real-valued zi construction
        with pytest.raises(NotImplementedError, match="complex sos"):
            msig.sosfiltfilt(sos, x)


def test_resample_complex_window_matches_scipy(rng):
    """A complex frequency-domain window must be applied, not real-cast."""
    x = rng.standard_normal(256).astype(np.float32)
    w = np.exp(1j * np.linspace(0, np.pi / 4, 256))  # complex, full-fft-length
    ref = sps.resample(x, 128, window=w)
    out = msig.resample(x, 128, window=w)
    assert_close(out, ref, rtol=1e-4, atol_frac=1e-5)


def test_resample_callable_complex_window(rng):
    x = rng.standard_normal(200).astype(np.float32)

    def cwin(freqs):
        return np.exp(1j * 0.1 * freqs)

    ref = sps.resample(x, 100, window=cwin)
    out = msig.resample(x, 100, window=cwin)
    assert_close(out, ref, rtol=1e-4, atol_frac=1e-5)


def test_resample_poly_complex_window_matches_scipy(rng):
    """Complex FIR window: scipy filters with the complex taps."""
    x = rng.standard_normal(1000).astype(np.float32)
    h = (sps.firwin(41, 0.4) * np.exp(1j * 0.2)).astype(np.complex128)
    ref = sps.resample_poly(x, 2, 3, window=h)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = msig.resample_poly(x, 2, 3, window=h)
    assert np.array(out).dtype == np.complex64
    assert_close(out, ref, rtol=1e-4, atol_frac=1e-5)


# ---------------------------------------------------------------------------
# convolution axes handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("conv", [msig.fftconvolve, msig.oaconvolve])
def test_convolve_singleton_axes_fallback_parity(conv, rng):
    """(1, n) with axes=[0]: every conv axis is length-1, so the result is a
    plain product. The scipy fallback must receive the user's axes, not None.
    """
    a = rng.standard_normal((1, 3)).astype(np.float32)
    b = rng.standard_normal((1, 3)).astype(np.float32)
    ref = sps.fftconvolve(a, b, axes=[0])
    for ctx in (
        msig.config_context(dispatch="scipy"),
        msig.config_context(dispatch="auto", gpu_min_size=10**9),
        msig.config_context(dispatch="mlx" if HAS_GPU else "auto", gpu_min_size=0),
    ):
        with ctx:
            out = conv(a, b, axes=[0])
        assert np.array(out).shape == ref.shape
        np.testing.assert_allclose(np.array(out), ref, rtol=1e-5)


def test_convolve_axes_validation(rng):
    a = rng.standard_normal((4, 5)).astype(np.float32)
    with pytest.raises(ValueError, match="axes exceeds dimensionality"):
        msig.fftconvolve(a, a, axes=2)
    with pytest.raises(ValueError, match="axes exceeds dimensionality"):
        msig.fftconvolve(a, a, axes=[-3])
    with pytest.raises(ValueError, match="integers"):
        msig.fftconvolve(a, a, axes=1.5)
    with pytest.raises(ValueError, match="integers"):
        msig.oaconvolve(a, a, axes=[0.5, 1])


# ---------------------------------------------------------------------------
# config robustness
# ---------------------------------------------------------------------------


def test_set_config_is_atomic():
    """A rejected call must leave every field untouched, including valid ones
    that appeared before the invalid one.
    """
    before = msig.get_config()
    with pytest.raises(ValueError):
        msig.set_config(dispatch="scipy", float64="never")
    after = msig.get_config()
    assert after == before
    with pytest.raises(ValueError):
        msig.set_config(gpu_min_size=-1)
    assert msig.get_config() == before


def test_float64_strict_raises_on_scipy_path(rng):
    """strict must reject f64 input regardless of which backend runs it."""
    x = rng.standard_normal(100)  # float64
    with msig.config_context(dispatch="scipy", float64="strict"):
        with pytest.raises(TypeError, match="strict"):
            msig.welch(x, nperseg=32)
    h = rng.standard_normal(9)
    with msig.config_context(dispatch="scipy", float64="strict"):
        with pytest.raises(TypeError, match="strict"):
            msig.fftconvolve(x, h)


# ---------------------------------------------------------------------------
# second audit round: windows, 0-d, FFT edge parity
# ---------------------------------------------------------------------------


def test_resample_complex_window_dc_nyquist(rng):
    """np.irfft drops the imaginary parts of the DC/Nyquist bins; the MLX
    path must do so explicitly or a complex window diverges completely."""
    x = np.ones(19, np.float32)
    w = np.full(19, 1j, np.complex64)
    ref = sps.resample(x, 47, window=w)
    out = np.array(msig.resample(x, 47, window=w))
    np.testing.assert_allclose(out, ref, atol=1e-5)
    x2 = rng.standard_normal(100).astype(np.float32)
    w2 = np.exp(1j * np.linspace(0, 1, 100))
    for num in (64, 63):  # even output exercises the Nyquist-bin branch
        np.testing.assert_allclose(
            np.array(msig.resample(x2, num, window=w2)),
            sps.resample(x2, num, window=w2), atol=1e-5,
        )


def test_zero_dim_inputs():
    from mlx_signal._array import to_mlx

    assert to_mlx(np.float32(3.0)).shape == ()
    assert to_mlx(np.array(2.0)).shape == ()
    out = msig.fftconvolve(np.float32(2.0), np.float32(3.0))
    ref = sps.fftconvolve(np.float32(2.0), np.float32(3.0))
    assert np.array(out).shape == ref.shape
    np.testing.assert_allclose(np.array(out), ref)


def test_next_fast_len_edge_parity():
    import scipy.fft

    assert msig.next_fast_len(0) == scipy.fft.next_fast_len(0) == 0
    assert msig.next_fast_len(1) == scipy.fft.next_fast_len(1) == 1
    with pytest.raises(ValueError):
        msig.next_fast_len(-1)


def test_nd_fft_length_one_axis(rng):
    """The Metal n-d real FFT returns garbage for a length-1 transform axis;
    the wrappers must route those to the CPU stream."""
    from mlx_signal import _fft_core

    x = mx.array(rng.standard_normal((4, 1)).astype(np.float32))
    np.testing.assert_allclose(
        np.array(_fft_core.rfftn(x, s=[1], axes=[-1])),
        np.fft.rfftn(np.array(x), s=[1], axes=[-1]), atol=1e-6,
    )
    sp = _fft_core.rfftn(x, s=[1], axes=[-1])
    np.testing.assert_allclose(
        np.array(_fft_core.irfftn(sp, s=[1], axes=[-1])),
        np.fft.irfftn(np.fft.rfftn(np.array(x), s=[1], axes=[-1]), s=[1], axes=[-1]),
        atol=1e-6,
    )


def test_zero_dim_dtype_policy():
    """0-d float64 arrays must honor strict/downcast like any other input."""
    from mlx_signal._array import to_mlx

    with msig.config_context(float64="strict"):
        with pytest.raises(TypeError, match="strict"):
            to_mlx(np.array(2.0))
    with msig.config_context(float64="downcast", warn_on_downcast=True):
        with pytest.warns(msig.DowncastWarning):
            assert to_mlx(np.array(2.0)).shape == ()


def test_direct_method_scalar_shapes():
    """method='direct' scalar results stay 0-d like the fft path."""
    a, b = np.float32(2.0), np.float32(3.0)
    assert np.array(msig.convolve(a, b, method="direct")).shape == ()
    assert np.array(msig.correlate(a, b, method="direct")).shape == ()


def test_byte_budget_cache_hard_bound():
    from mlx_signal._cache import ByteBudgetCache

    c = ByteBudgetCache(100)
    c.put(("big",), "v", 200)
    assert c.get(("big",)) is None and c._total == 0
    c.put(("a",), 1, 60)
    c.put(("b",), 2, 60)
    assert c._total <= 100
