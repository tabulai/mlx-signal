"""Golden tests: upfirdn (custom Metal kernel), resample_poly, decimate vs scipy."""

import numpy as np
import pytest
import scipy.signal as sps

import mlx_signal as msig
from _utils import assert_close, assert_type_and_dtype
from conftest import HAS_GPU, requires_gpu
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


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("cx,ch", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("n,up,down,n_taps", [
    (1000, 3, 2, 65), (300, 160, 147, 3201), (2000, 1, 10, 201), (1500, 2, 5, 4097),
])
def test_u32_route_bit_identical(rng, cx, ch, n, up, down, n_taps):
    """Dispatch between the u32 and long-arithmetic kernels is a pure
    performance choice: all routes accumulate in the same order, so outputs
    must be bit-identical, not merely close."""
    import mlx.core as mx

    from mlx_signal import _upfirdn_metal as um
    from mlx_signal.resampling import _output_len

    x = rng.standard_normal((3, n)).astype(np.float32)
    if cx:
        x = (x + 1j * rng.standard_normal((3, n))).astype(np.complex64)
    h = rng.standard_normal(n_taps).astype(np.float32)
    if ch:
        h = (h + 1j * rng.standard_normal(n_taps)).astype(np.complex64)
    x, h = mx.array(x), mx.array(h)
    n_out = _output_len(n_taps, n, up, down)
    complex_out = cx or ch
    xin = mx.view(x, mx.float32) if cx else x
    hin = mx.view(h, mx.float32) if ch else h
    params = mx.array([n, n_out, up, down, n_taps], dtype=mx.int32)
    width = 2 * n_out if complex_out else n_out

    def launch(kern):
        (out,) = kern(inputs=[xin, hin, params], grid=(n_out, 3, 1),
                      threadgroup=(min(256, max(32, n_out)), 1, 1),
                      output_shapes=[(3, width)], output_dtypes=[mx.float32])
        return np.array(out)

    ref = launch(um._kernel(cx, ch, direct=True, u32=True))
    np.testing.assert_array_equal(ref, launch(um._kernel(cx, ch, direct=False)))
    np.testing.assert_array_equal(ref, launch(um._kernel(cx, ch, direct=True)))


def test_u32_guard_accounts_for_float2_factors():
    """The u32 overflow guard must include the float32-view doubling: complex
    input doubles the flat x index, complex output doubles the store offset.
    A miss here is silent wrong output, so pin each term's boundary."""
    from mlx_signal._upfirdn_metal import _U32_LIMIT, _fits_u32

    assert _U32_LIMIT == 1 << 31
    # input side: B * n_in * (2 if complex signal)
    assert _fits_u32(2, 1 << 29, 1 << 20, 1, 1, cx=False, complex_out=False)
    assert not _fits_u32(2, 1 << 29, 1 << 20, 1, 1, cx=True, complex_out=True)
    # output side: B * n_out * (2 if complex signal OR complex taps)
    assert _fits_u32(2, 1 << 20, 1 << 29, 1, 1, cx=False, complex_out=False)
    assert not _fits_u32(2, 1 << 20, 1 << 29, 1, 1, cx=False, complex_out=True)
    # geometry: n_out * down + 2 * up covers m = i*down plus the loop overshoot
    assert not _fits_u32(1, 1 << 20, 1 << 30, 1, 2, cx=False, complex_out=False)
    assert _fits_u32(1, 1 << 20, 1 << 29, 1, 2, cx=False, complex_out=False)
    # ... and the overshoot term alone: only 2*up trips these
    assert not _fits_u32(1, 4, 1, 1 << 30, 1, cx=False, complex_out=False)
    assert _fits_u32(1, 4, 1, 1 << 29, 1, cx=False, complex_out=False)


@requires_gpu
@pytest.mark.gpu
def test_upfirdn_gpu_empty_taps_stores_zeros():
    """Internal contract: zero-length taps (unreachable through the public
    upfirdn(), which validates h) must empty the tap window and store exact
    zeros on the u32 route just like the long kernels — never read h."""
    import mlx.core as mx

    from mlx_signal._upfirdn_metal import upfirdn_gpu

    out = upfirdn_gpu(mx.ones((2, 100)), mx.zeros((0,)), 2, 1, 199)
    np.testing.assert_array_equal(np.array(out), np.zeros((2, 199), np.float32))


def test_upfirdn_pad_mode_falls_back(rng):
    x = rng.standard_normal(300).astype(np.float32)
    h = rng.standard_normal(21).astype(np.float32)
    with msig.config_context(dispatch="auto", gpu_min_size=1, warn_on_fallback=True):
        with pytest.warns(msig.FallbackWarning):
            out = msig.upfirdn(h, x, 2, 1, mode="smooth")
    assert_close(out, sps.upfirdn(h, x, 2, 1, mode="smooth"))
    if HAS_GPU:  # fixture pins dispatch="mlx" only when Metal exists
        with pytest.raises(NotImplementedError):
            msig.upfirdn(h, x, 2, 1, mode="smooth")


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


def test_decimate_iir_matches_scipy(rng):
    x = rng.standard_normal(1000).astype(np.float32)
    out = msig.decimate(x, 4)  # default ftype="iir", GPU sosfiltfilt path
    assert_close(out, sps.decimate(x, 4), rtol=1e-3, atol_frac=1e-4)
    assert_type_and_dtype(out)


def test_decimate_dlti_falls_back(rng):
    x = rng.standard_normal(1000).astype(np.float32)
    system = sps.dlti(*sps.cheby1(6, 0.05, 0.2))
    with msig.config_context(dispatch="auto", gpu_min_size=1, warn_on_fallback=True):
        with pytest.warns(msig.FallbackWarning):
            out = msig.decimate(x, 4, ftype=system)
    assert_close(out, sps.decimate(x, 4, ftype=system), rtol=1e-3, atol_frac=1e-4)
    if HAS_GPU:
        with pytest.raises(NotImplementedError):
            msig.decimate(x, 4, ftype=system)


@pytest.mark.parametrize("n,up,down,n_taps", [(2000, 3, 2, 6000), (1500, 2, 5, 4097)])
def test_upfirdn_multi_tile_taps(rng, n, up, down, n_taps):
    """Filters longer than one threadgroup tile stream through multiple tiles."""
    x = rng.standard_normal(n).astype(np.float32)
    h = rng.standard_normal(n_taps).astype(np.float32)
    assert_close(msig.upfirdn(h, x, up, down), sps.upfirdn(h, x, up, down),
                 rtol=2e-4, atol_frac=2e-5)


def test_upfirdn_complex_multi_tile(rng):
    """Complex taps use half-size tiles; 3000 taps spans three of them."""
    x = (rng.standard_normal(1200) + 1j * rng.standard_normal(1200)).astype(np.complex64)
    h = (rng.standard_normal(3000) + 1j * rng.standard_normal(3000)).astype(np.complex64)
    assert_close(msig.upfirdn(h, x, 2, 3), sps.upfirdn(h, x, 2, 3),
                 rtol=2e-4, atol_frac=2e-5)
