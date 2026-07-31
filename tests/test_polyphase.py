"""Golden tests: upfirdn (custom Metal kernel), resample_poly, decimate vs scipy."""

import numpy as np
import pytest
import scipy.signal as sps

import mlx_signal as msig
from _utils import assert_close, assert_type_and_dtype
from conftest import requires_gpu
from mlx_signal.resampling import _upfirdn_composed

UPFIRDN_CASES = [
    # (n, up, down, n_taps)
    (100, 1, 1, 31),
    (1000, 3, 2, 65),
    (1000, 2, 3, 64),
    (997, 7, 5, 47),
    (512, 1, 4, 33),
    (512, 4, 1, 33),
    (256, 8, 1, 5),      # up larger than taps: some branches are empty
    (300, 160, 147, 3201),  # audio-rate 48k->44.1k style, taps > threadgroup cap
    (64, 1, 1, 1),
    (10, 5, 13, 7),      # output shorter than input
]


@pytest.mark.parametrize("n,up,down,n_taps", UPFIRDN_CASES)
def test_upfirdn_matches_scipy(rng, n, up, down, n_taps):
    x = rng.standard_normal(n).astype(np.float32)
    h = rng.standard_normal(n_taps).astype(np.float32)
    ref = sps.upfirdn(h, x, up=up, down=down)
    out = msig.upfirdn(h, x, up=up, down=down)
    assert ref.shape == tuple(np.array(out).shape)
    assert_close(out, ref)
    assert_type_and_dtype(out)


def test_upfirdn_2d_axes(rng):
    x = rng.standard_normal((5, 400)).astype(np.float32)
    h = rng.standard_normal(31).astype(np.float32)
    assert_close(msig.upfirdn(h, x, 3, 2, axis=-1), sps.upfirdn(h, x, 3, 2, axis=-1))
    x2 = rng.standard_normal((400, 5)).astype(np.float32)
    assert_close(msig.upfirdn(h, x2, 3, 2, axis=0), sps.upfirdn(h, x2, 3, 2, axis=0))


def test_upfirdn_3d(rng):
    x = rng.standard_normal((2, 3, 200)).astype(np.float32)
    h = rng.standard_normal(21).astype(np.float32)
    assert_close(msig.upfirdn(h, x, 2, 5), sps.upfirdn(h, x, 2, 5))


def test_upfirdn_complex_signal(rng):
    x = (rng.standard_normal(500) + 1j * rng.standard_normal(500)).astype(np.complex64)
    h = rng.standard_normal(41).astype(np.float32)
    assert_close(msig.upfirdn(h, x, 2, 3), sps.upfirdn(h, x, 2, 3))


def test_upfirdn_complex_filter(rng):
    x = rng.standard_normal(500).astype(np.float32)
    h = (rng.standard_normal(41) + 1j * rng.standard_normal(41)).astype(np.complex64)
    assert_close(msig.upfirdn(h, x, 2, 3), sps.upfirdn(h, x, 2, 3))


def test_upfirdn_complex_both(rng):
    x = (rng.standard_normal(500) + 1j * rng.standard_normal(500)).astype(np.complex64)
    h = (rng.standard_normal(41) + 1j * rng.standard_normal(41)).astype(np.complex64)
    assert_close(msig.upfirdn(h, x, 3, 2), sps.upfirdn(h, x, 3, 2))


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("n,up,down,n_taps", [(777, 3, 5, 129), (300, 160, 147, 3201)])
def test_kernel_agrees_with_composed_path(rng, n, up, down, n_taps):
    """The Metal kernel and the FFT-composed MLX path are independent implementations."""
    import mlx.core as mx

    from mlx_signal.resampling import _output_len, _upfirdn_plane_dispatch

    x = mx.array(rng.standard_normal((3, n)).astype(np.float32))
    h = mx.array(rng.standard_normal(n_taps).astype(np.float32))
    n_out = _output_len(n_taps, n, up, down)
    a = _upfirdn_plane_dispatch(x, h, up, down, n_out)
    b = _upfirdn_composed(x, h, up, down)
    assert_close(a, np.array(b))


def test_upfirdn_pad_mode_falls_back(rng):
    x = rng.standard_normal(300).astype(np.float32)
    h = rng.standard_normal(21).astype(np.float32)
    with msig.config_context(dispatch="auto", gpu_min_size=1):
        with pytest.warns(msig.FallbackWarning):
            out = msig.upfirdn(h, x, 2, 1, mode="smooth")
    assert_close(out, sps.upfirdn(h, x, 2, 1, mode="smooth"))
    with pytest.raises(NotImplementedError):
        msig.upfirdn(h, x, 2, 1, mode="smooth")  # dispatch="mlx" pinned by fixture


RESAMPLE_POLY_CASES = [(2, 1), (1, 2), (3, 2), (2, 3), (160, 147), (7, 3), (4, 2), (5, 5)]


@pytest.mark.parametrize("up,down", RESAMPLE_POLY_CASES)
@pytest.mark.parametrize("n", [1000, 997])
def test_resample_poly_matches_scipy(rng, up, down, n):
    x = rng.standard_normal(n).astype(np.float32)
    ref = sps.resample_poly(x, up, down)
    out = msig.resample_poly(x, up, down)
    assert ref.shape == tuple(np.array(out).shape)
    assert_close(out, ref, rtol=2e-4)


def test_resample_poly_axis_default_zero(rng):
    x = rng.standard_normal((600, 4)).astype(np.float32)
    ref = sps.resample_poly(x, 2, 3)
    out = msig.resample_poly(x, 2, 3)
    assert ref.shape == tuple(np.array(out).shape)
    assert_close(out, ref, rtol=2e-4)


def test_resample_poly_axis1(rng):
    x = rng.standard_normal((4, 600)).astype(np.float32)
    assert_close(msig.resample_poly(x, 3, 4, axis=1), sps.resample_poly(x, 3, 4, axis=1),
                 rtol=2e-4)


def test_resample_poly_window_array(rng):
    x = rng.standard_normal(800).astype(np.float32)
    w = sps.firwin(41, 0.25)
    assert_close(msig.resample_poly(x, 1, 2, window=w), sps.resample_poly(x, 1, 2, window=w),
                 rtol=2e-4)


def test_resample_poly_complex(rng):
    x = (rng.standard_normal(1000) + 1j * rng.standard_normal(1000)).astype(np.complex64)
    assert_close(msig.resample_poly(x, 2, 3), sps.resample_poly(x, 2, 3), rtol=2e-4)


def test_resample_poly_identity(rng):
    x = rng.standard_normal(256).astype(np.float32)
    out = np.array(msig.resample_poly(x, 3, 3))
    np.testing.assert_allclose(out, x)


def test_resample_poly_tone():
    fs = 48000.0
    t = np.arange(48000, dtype=np.float32) / fs
    x = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    y = np.array(msig.resample_poly(x, 147, 160))  # 48k -> 44.1k
    t2 = np.arange(len(y)) / 44100.0
    ref = np.sin(2 * np.pi * 1000 * t2)
    # tolerance bounded by the default kaiser filter's passband ripple, not fp32
    np.testing.assert_allclose(y[500:-500], ref[500:-500], atol=2e-3)


@pytest.mark.parametrize("zero_phase", [True, False])
@pytest.mark.parametrize("q", [2, 4, 13])
def test_decimate_fir(rng, zero_phase, q):
    x = rng.standard_normal(2000).astype(np.float32)
    ref = sps.decimate(x, q, ftype="fir", zero_phase=zero_phase)
    out = msig.decimate(x, q, ftype="fir", zero_phase=zero_phase)
    assert ref.shape == tuple(np.array(out).shape)
    assert_close(out, ref, rtol=2e-4)


def test_decimate_fir_custom_order_axis(rng):
    x = rng.standard_normal((3, 1500)).astype(np.float32)
    ref = sps.decimate(x, 5, n=60, ftype="fir", axis=-1)
    out = msig.decimate(x, 5, n=60, ftype="fir", axis=-1)
    assert_close(out, ref, rtol=2e-4)


def test_decimate_iir_falls_back(rng):
    x = rng.standard_normal(1000).astype(np.float32)
    with msig.config_context(dispatch="auto", gpu_min_size=1):
        with pytest.warns(msig.FallbackWarning):
            out = msig.decimate(x, 4)
    assert_close(out, sps.decimate(x, 4), rtol=1e-3)
    assert_type_and_dtype(out)
    with pytest.raises(NotImplementedError):
        msig.decimate(x, 4)  # dispatch="mlx" pinned by fixture
