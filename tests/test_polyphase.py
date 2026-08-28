"""Golden tests: upfirdn (custom Metal kernel), resample_poly, decimate vs scipy."""

import numpy as np
import pytest
import scipy.signal as sps

import mlx_signal_processing as msig
from _utils import assert_close, assert_type_and_dtype
from conftest import HAS_GPU, requires_gpu
from mlx_signal_processing.resampling import _upfirdn_composed

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


def _assert_nonfinite_components(got, expected, *, rtol=None):
    """Compare NaN/Inf masks per component, then every finite value."""
    got = np.asarray(got)
    expected = np.asarray(expected)
    parts = ((got.real, expected.real), (got.imag, expected.imag))
    if not (np.iscomplexobj(got) or np.iscomplexobj(expected)):
        parts = ((got, expected),)
    for actual_part, expected_part in parts:
        np.testing.assert_array_equal(np.isnan(actual_part), np.isnan(expected_part))
        np.testing.assert_array_equal(np.isposinf(actual_part), np.isposinf(expected_part))
        np.testing.assert_array_equal(np.isneginf(actual_part), np.isneginf(expected_part))
        finite = np.isfinite(expected_part)
        if rtol is None:
            np.testing.assert_array_equal(actual_part[finite], expected_part[finite])
        else:
            scale = max(1.0, float(np.max(np.abs(expected_part[finite]), initial=0.0)))
            np.testing.assert_allclose(
                actual_part[finite], expected_part[finite], rtol=rtol, atol=rtol * scale
            )


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

    from mlx_signal_processing.resampling import _output_len, _upfirdn_plane_dispatch

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

    from mlx_signal_processing import _upfirdn_metal as um
    from mlx_signal_processing.resampling import _output_len

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
    params = mx.array([n, n_out, up, down, n_taps, 0], dtype=mx.int32)
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
    from mlx_signal_processing._upfirdn_metal import _U32_LIMIT, _fits_u32

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

    from mlx_signal_processing._upfirdn_metal import upfirdn_gpu

    out = upfirdn_gpu(mx.ones((2, 100)), mx.zeros((0,)), 2, 1, 199)
    np.testing.assert_array_equal(np.array(out), np.zeros((2, 199), np.float32))


@pytest.mark.parametrize("mode", ["wrap", "edge", "smooth", "symmetric", "reflect",
                                  "antisymmetric", "antireflect", "line"])
@pytest.mark.parametrize("up,down", [(2, 3), (1, 10), (7, 3)])
def test_upfirdn_modes_run_on_device(rng, mode, up, down):
    """Every scipy signal-extension mode is served by pre-extending on-device
    and running the constant kernel — no fallback, no warning."""
    import warnings as _w

    x = rng.standard_normal((3, 700)).astype(np.float32)
    h = rng.standard_normal(63).astype(np.float32)
    ref = sps.upfirdn(h, x, up, down, mode=mode)
    with _w.catch_warnings():
        _w.simplefilter("error", msig.FallbackWarning)
        out = msig.upfirdn(h, x, up, down, mode=mode)
    assert ref.shape == tuple(np.array(out).shape)
    assert_close(out, ref, rtol=2e-5, atol_frac=2e-6)


def test_upfirdn_mode_complex_signal(rng):
    x = (rng.standard_normal(600) + 1j * rng.standard_normal(600)).astype(np.complex64)
    h = rng.standard_normal(41).astype(np.float32)
    assert_close(msig.upfirdn(h, x, 3, 2, mode="reflect"),
                 sps.upfirdn(h, x, 3, 2, mode="reflect"), rtol=2e-5, atol_frac=2e-6)


def test_upfirdn_constant_cval(rng):
    x = rng.standard_normal(500).astype(np.float32)
    h = rng.standard_normal(31).astype(np.float32)
    assert_close(msig.upfirdn(h, x, 2, 3, cval=1.5),
                 sps.upfirdn(h, x, 2, 3, cval=1.5), rtol=2e-5, atol_frac=2e-6)


def test_upfirdn_mode_short_signal_falls_back(rng):
    """A signal shorter than the boundary extension can't be single-fold
    reflected; that corner keeps scipy, loudly."""
    x = rng.standard_normal(40).astype(np.float32)
    h = rng.standard_normal(255).astype(np.float32)
    with msig.config_context(dispatch="auto", gpu_min_size=1, warn_on_fallback=True):
        with pytest.warns(msig.FallbackWarning, match="shorter"):
            out = msig.upfirdn(h, x, 2, 3, mode="reflect")
    assert_close(out, sps.upfirdn(h, x, 2, 3, mode="reflect"))
    if HAS_GPU:  # fixture pins dispatch="mlx" only when Metal exists
        with pytest.raises(NotImplementedError):
            msig.upfirdn(h, x, 2, 3, mode="reflect")


def test_upfirdn_unknown_mode_raises(rng):
    with pytest.raises(ValueError, match="Unknown mode"):
        msig.upfirdn(np.ones(5, np.float32), rng.standard_normal(100).astype(np.float32),
                     2, 1, mode="bogus")


@pytest.mark.parametrize("padtype", ["mean", "median", "maximum", "minimum",
                                     "reflect", "smooth", "line", "wrap"])
def test_resample_poly_padtypes_match_scipy(rng, padtype):
    x = (rng.standard_normal((5, 2000)) + 3.0).astype(np.float32)
    ref = sps.resample_poly(x, 147, 160, axis=-1, padtype=padtype)
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error", msig.FallbackWarning)
        out = msig.resample_poly(x, 147, 160, axis=-1, padtype=padtype)
    assert ref.shape == tuple(np.array(out).shape)
    assert_close(out, ref, rtol=2e-4)


def test_resample_poly_constant_cval(rng):
    x = rng.standard_normal(2000).astype(np.float32)
    assert_close(msig.resample_poly(x, 2, 3, padtype="constant", cval=0.7),
                 sps.resample_poly(x, 2, 3, padtype="constant", cval=0.7), rtol=2e-4)


@pytest.mark.parametrize("padtype", ["median", "maximum", "minimum", "mean"])
def test_resample_poly_complex_stat_padtypes_on_device(rng, padtype):
    """MLX sorts/reduces complex64 with numpy's lexicographic convention, so
    the statistical padtypes serve complex signals without fallback."""
    import warnings as _w

    x = (rng.standard_normal(5000) + 1j * rng.standard_normal(5000)).astype(np.complex64)
    ref = sps.resample_poly(x, 3, 2, padtype=padtype)
    with _w.catch_warnings():
        _w.simplefilter("error", msig.FallbackWarning)
        out = msig.resample_poly(x, 3, 2, padtype=padtype)
    assert_close(out, ref, rtol=2e-4)


def test_resample_poly_median_even_length_averages_middles():
    """np.median of an even-length axis averages the two middle order
    statistics; a tiny skewed signal pins the exact background value."""
    x = np.array([1.0, 2.0, 3.0, 100.0], np.float32)  # median 2.5, not 3.0
    with msig.config_context(gpu_min_size=0):
        out = np.array(msig.resample_poly(x, 3, 2, padtype="median"))
    ref = sps.resample_poly(x, 3, 2, padtype="median")
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5 * np.abs(ref).max())


def test_resample_poly_median_propagates_nan(rng):
    """np.median returns NaN when any element is NaN; sorting alone would
    hide the NaN at the end and pick a finite middle."""
    x = rng.standard_normal(3000).astype(np.float32)
    x[1500] = np.nan
    with msig.config_context(gpu_min_size=0):
        out = np.array(msig.resample_poly(x, 3, 2, padtype="median"))
    ref = sps.resample_poly(x, 3, 2, padtype="median")
    np.testing.assert_array_equal(np.isnan(out), np.isnan(ref))


def test_upfirdn_mode_boundary_signal_lengths(rng):
    """The constructibility guard sits exactly where the extension slices stay
    in range: at the boundary n the mode runs on-device and matches scipy; one
    sample shorter falls back loudly. (up=2, down=3, 63 taps: left extension
    L=33 after alignment, right R=31.)"""
    import warnings as _w

    h = rng.standard_normal(63).astype(np.float32)
    for mode, n_ok in [("symmetric", 33), ("antisymmetric", 33), ("wrap", 33),
                       ("reflect", 34), ("antireflect", 34)]:
        x = rng.standard_normal(n_ok).astype(np.float32)
        with msig.config_context(dispatch="auto", gpu_min_size=1, warn_on_fallback=True):
            with _w.catch_warnings():
                _w.simplefilter("error", msig.FallbackWarning)
                out = msig.upfirdn(h, x, 2, 3, mode=mode)
        assert_close(out, sps.upfirdn(h, x, 2, 3, mode=mode), rtol=2e-5, atol_frac=2e-6)

        x_short = rng.standard_normal(n_ok - 1).astype(np.float32)
        with msig.config_context(dispatch="auto", gpu_min_size=1, warn_on_fallback=True):
            with pytest.warns(msig.FallbackWarning, match="shorter"):
                out = msig.upfirdn(h, x_short, 2, 3, mode=mode)
        assert_close(out, sps.upfirdn(h, x_short, 2, 3, mode=mode),
                     rtol=2e-5, atol_frac=2e-6)


def test_upfirdn_mode_case_insensitive(rng):
    """scipy lowercases mode strings; so do we (padtype stays exact-match,
    also like scipy)."""
    x = rng.standard_normal(500).astype(np.float32)
    h = rng.standard_normal(31).astype(np.float32)
    assert_close(msig.upfirdn(h, x, 2, 3, mode="REFLECT"),
                 sps.upfirdn(h, x, 2, 3, mode="REFLECT"), rtol=2e-5, atol_frac=2e-6)


def test_upfirdn_complex_cval_raises_like_scipy(rng):
    x = rng.standard_normal(500).astype(np.float32)
    h = rng.standard_normal(31).astype(np.float32)
    with pytest.raises(TypeError):
        msig.upfirdn(h, x, 2, 3, cval=1 + 2j)


def test_upfirdn_none_cval_raises_like_scipy(rng):
    x = rng.standard_normal(500).astype(np.float32)
    h = rng.standard_normal(31).astype(np.float32)
    with pytest.raises(TypeError):
        msig.upfirdn(h, x, 2, 3, cval=None)


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["constant", "reflect"])
@pytest.mark.parametrize(
    "bad", [np.nan, np.inf, complex(np.nan, 1), complex(3, np.nan),
            complex(np.inf, 1), complex(3, np.inf)],
    ids=["real-nan", "real-inf", "complex-real-nan", "complex-imag-nan",
         "complex-real-inf", "complex-imag-inf"],
)
def test_upfirdn_nonfinite_signal_matches_scipy_on_gpu(mode, bad):
    """The padded polyphase branches and generic complex multiply preserve
    scipy's local, component-wise exceptional-value masks on Metal."""
    import warnings as _w

    dtype = np.complex64 if isinstance(bad, complex) else np.float32
    x = np.arange(80, dtype=np.float32).astype(dtype)
    x[31] = bad
    h = np.linspace(-1.0, 1.0, 17, dtype=np.float32)
    with _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        ref = sps.upfirdn(h, x, 3, 2, mode=mode)
    with _w.catch_warnings():
        _w.simplefilter("error", msig.FallbackWarning)
        out = np.array(msig.upfirdn(h, x, 3, 2, mode=mode))
    _assert_nonfinite_components(out, ref, rtol=2e-5)


@requires_gpu
@pytest.mark.gpu
def test_upfirdn_antireflect_complex_infinite_edge_matches_scipy():
    """Odd reflection is component-wise; generic complex arithmetic would add
    spurious 0*Inf NaNs while constructing the boundary extension."""
    import warnings as _w

    x = np.array([-1.2221696 + 0.26735377j, complex(2, np.inf)], np.complex64)
    h = np.array([-0.15171018, -0.08910976], np.float32)
    with _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        ref = sps.upfirdn(h, x, mode="antireflect")
        out = np.array(msig.upfirdn(h, x, mode="antireflect"))
    _assert_nonfinite_components(out, ref, rtol=2e-5)


@pytest.mark.parametrize(
    "bad", [np.nan, np.inf, complex(np.nan, 1), complex(3, np.inf)],
    ids=["real-nan", "real-inf", "complex-nan", "complex-inf"],
)
def test_upfirdn_nonfinite_taps_fall_back_exact(bad):
    """Non-finite taps interact with scipy's implicit signal extension; retain
    the direct scipy loop rather than dropping its zero-times-nonfinite terms."""
    import warnings as _w

    dtype = np.complex64 if isinstance(bad, complex) else np.float32
    h = np.arange(1, 18, dtype=np.float32).astype(dtype)
    h[8] = bad
    x = np.arange(80, dtype=np.float32)
    with _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        ref = sps.upfirdn(h, x, 3, 2)
    with msig.config_context(dispatch="auto", gpu_min_size=0, warn_on_fallback=True):
        with pytest.warns(msig.FallbackWarning, match="non-finite filter taps"):
            with _w.catch_warnings():
                _w.simplefilter("ignore", RuntimeWarning)
                out = np.array(msig.upfirdn(h, x, 3, 2))
    _assert_nonfinite_components(out, ref)
    if HAS_GPU:
        with pytest.raises(NotImplementedError, match="non-finite filter taps"):
            msig.upfirdn(h, x, 3, 2)


def test_upfirdn_no_metal_nonfinite_fallback_is_route_invariant(monkeypatch):
    """The exceptional CPU fallback must filter the same canonical f32 values
    as an explicitly selected scipy route, not the original f64 operands."""
    import mlx.core as mx

    h = np.array([1.0, -1.0 + 1e-8], np.float64)
    x = np.full(20, 1e8, np.float64)
    x[10] = np.nan
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    with msig.config_context(dispatch="scipy", warn_on_downcast=False):
        scipy_route = np.array(msig.upfirdn(h, x))
    with msig.config_context(
        dispatch="auto", gpu_min_size=0, warn_on_downcast=False,
        warn_on_fallback=False,
    ):
        fallback_route = np.array(msig.upfirdn(h, x))
    _assert_nonfinite_components(fallback_route, scipy_route)


@pytest.mark.parametrize("padtype", ["mean", "median", "maximum", "minimum"])
@pytest.mark.parametrize(
    "bad", [complex(np.nan, 1), complex(3, np.nan), complex(np.inf, 1),
            complex(3, np.inf)],
    ids=["nan-real", "nan-imag", "inf-real", "inf-imag"],
)
@pytest.mark.parametrize("container", ["numpy", "mlx"])
def test_resample_poly_complex_nonfinite_stat_padtypes_fall_back_exact(
    padtype, bad, container
):
    """SciPy's generic complex multiplication spreads either-component
    non-finites across components; the split-plane GPU kernel must not serve
    this exceptional parity corner."""
    import warnings as _w

    x = np.array(
        [1 + 2j, 2 + 1j, bad, 4 - 1j, 5 + 0j], np.complex64
    )
    source = x
    if container == "mlx":
        import mlx.core as mx

        source = mx.array(x)
    with _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        ref = sps.resample_poly(x, 3, 2, padtype=padtype)
    with msig.config_context(dispatch="auto", gpu_min_size=0, warn_on_fallback=True):
        with pytest.warns(msig.FallbackWarning, match="complex non-finite"):
            with _w.catch_warnings():
                _w.simplefilter("ignore", RuntimeWarning)
                out = np.array(msig.resample_poly(source, 3, 2, padtype=padtype))
    _assert_nonfinite_components(out, ref)
    if HAS_GPU:
        with pytest.raises(NotImplementedError, match="complex non-finite"):
            msig.resample_poly(source, 3, 2, padtype=padtype)


@pytest.mark.parametrize(
    "config", [
        {"dispatch": "scipy"},
        {"dispatch": "auto", "gpu_min_size": 10**9},
    ],
    ids=["explicit-scipy", "auto-small"],
)
@pytest.mark.parametrize("container", ["numpy", "mlx"])
def test_resample_poly_complex_nonfinite_scipy_routes_stay_silent(config, container):
    """A deliberate or size-based scipy route is not a capability fallback."""
    import warnings as _w

    x = np.array([1 + 2j, 2 + 1j, complex(3, np.nan), 4 - 1j], np.complex64)
    source = x
    if container == "mlx":
        import mlx.core as mx

        source = mx.array(x)
    with _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        ref = sps.resample_poly(x, 3, 2, padtype="mean")
    with msig.config_context(**config, warn_on_fallback=True):
        with _w.catch_warnings():
            _w.simplefilter("error", msig.FallbackWarning)
            _w.simplefilter("ignore", RuntimeWarning)
            out = np.array(msig.resample_poly(source, 3, 2, padtype="mean"))
    _assert_nonfinite_components(out, ref)


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("padtype", ["constant", "reflect", "smooth", "wrap"])
@pytest.mark.parametrize("bad", [complex(np.nan, 1), complex(3, np.inf)])
def test_resample_poly_complex_nonfinite_extension_padtypes_on_gpu(padtype, bad):
    """Non-statistical padtypes retain their on-device route and scipy masks."""
    import warnings as _w

    x = np.full(100, 1 + 2j, np.complex64)
    x[50] = bad
    with _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        ref = sps.resample_poly(x, 3, 2, padtype=padtype)
    with _w.catch_warnings():
        _w.simplefilter("error", msig.FallbackWarning)
        out = np.array(msig.resample_poly(x, 3, 2, padtype=padtype))
    _assert_nonfinite_components(out, ref, rtol=2e-4)


def test_upfirdn_empty_input_nonzero_cval(rng):
    """Empty input with a nonzero cval still has well-defined outputs (tap
    sums over the pure-cval extension); scipy computes them and so must we."""
    h = np.array([1.0, -2.0, 3.0, 1.0, 2.0], np.float32)
    x = np.zeros(0, np.float32)
    ref = sps.upfirdn(h, x, 2, 3, cval=5.0)
    out = np.array(msig.upfirdn(h, x, 2, 3, cval=5.0))
    np.testing.assert_allclose(out, ref, rtol=1e-6)


def test_resample_poly_bad_padtype_raises_after_identity_shortcut(rng):
    x = rng.standard_normal(100).astype(np.float32)
    with pytest.raises(ValueError, match="padtype must be one of"):
        msig.resample_poly(x, 2, 3, padtype="bogus")
    # scipy validates padtype only after the up == down early return
    np.testing.assert_array_equal(np.array(msig.resample_poly(x, 3, 3, padtype="bogus")), x)


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
