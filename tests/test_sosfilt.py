"""Golden tests: sosfilt / sosfiltfilt (batched-channel IIR kernel) vs scipy.

The kernel runs scipy's exact direct-form-II-transposed recurrence in float32,
so it is compared bit-tight against scipy executing in float32, and loosely
against scipy's float64 results (the difference is float32 itself, not the
implementation).
"""

import numpy as np
import pytest
import scipy
import scipy.signal as sps

import mlx_signal_processing as msig
from _utils import assert_close, assert_type_and_dtype
from conftest import HAS_GPU

FILTERS = [
    sps.butter(4, 0.2, output="sos"),
    sps.butter(8, [0.1, 0.3], btype="band", output="sos"),
    sps.cheby1(8, 0.05, 0.25, output="sos"),
    sps.ellip(6, 0.1, 60, 0.3, output="sos"),
    sps.butter(1, 0.5, output="sos"),  # single section
]


# with a GPU the kernel is bit-identical to modern scipy-in-float32; older
# scipy's float32 recurrence rounds in a different op order (~2e-6 drift).
# Without Metal the canonical f32 scipy fallback is exact, but keep the broader
# tolerance shared by the CPU-only test environment.
_SCIPY_VER = tuple(int(p) for p in scipy.__version__.split(".")[:2])
if not HAS_GPU:
    _EXACT_RTOL = 1e-4
elif _SCIPY_VER >= (1, 15):
    _EXACT_RTOL = 1e-6
else:
    _EXACT_RTOL = 1e-5

# renamed from sosfreqz in scipy 1.15
_freqz_sos = getattr(sps, "freqz_sos", None) or sps.sosfreqz


def _f32_ref(sos, x, axis=-1, zi=None):
    return sps.sosfilt(sos.astype(np.float32), x.astype(np.float32), axis=axis, zi=zi)


@pytest.mark.parametrize("sos", FILTERS)
@pytest.mark.parametrize("shape", [(4096,), (16, 20000), (3, 5, 4000)])
def test_sosfilt_matches_scipy_f32_exact(rng, sos, shape):
    x = rng.standard_normal(shape).astype(np.float32)
    ref = _f32_ref(sos, x)
    out = msig.sosfilt(sos, x)
    assert_type_and_dtype(out)
    assert ref.shape == tuple(np.array(out).shape)
    np.testing.assert_allclose(np.array(out), ref, rtol=_EXACT_RTOL,
                               atol=_EXACT_RTOL * np.abs(ref).max())


@pytest.mark.parametrize("sos", FILTERS[:2])
def test_sosfilt_matches_scipy_f64(rng, sos):
    x = rng.standard_normal((8, 50000)).astype(np.float32)
    ref = sps.sosfilt(sos, x.astype(np.float64))
    # difference is float32 arithmetic itself, not the implementation
    assert_close(msig.sosfilt(sos, x), ref, rtol=1e-3, atol_frac=1e-4)


def test_sosfilt_axis0(rng):
    sos = FILTERS[0]
    x = rng.standard_normal((20000, 6)).astype(np.float32)
    ref = _f32_ref(sos, x, axis=0)
    np.testing.assert_allclose(np.array(msig.sosfilt(sos, x, axis=0)), ref, rtol=_EXACT_RTOL,
                               atol=_EXACT_RTOL * np.abs(ref).max())


def test_sosfilt_zi_roundtrip(rng):
    sos = FILTERS[1]
    S = sos.shape[0]
    x = rng.standard_normal((5, 8000)).astype(np.float32)
    zi = rng.standard_normal((S, 5, 2)).astype(np.float32)
    y_ref, zf_ref = _f32_ref(sos, x, zi=zi)
    y, zf = msig.sosfilt(sos, x, zi=zi)
    np.testing.assert_allclose(np.array(y), y_ref, rtol=_EXACT_RTOL,
                               atol=_EXACT_RTOL * np.abs(y_ref).max())
    np.testing.assert_allclose(np.array(zf), zf_ref, rtol=1e-5,
                               atol=1e-5 * max(np.abs(zf_ref).max(), 1e-9))


def test_sosfilt_streaming_chunks_equal_one_shot(rng):
    """Filtering in chunks with carried state must equal one-shot filtering."""
    sos = FILTERS[0]
    S = sos.shape[0]
    x = rng.standard_normal((10, 30000)).astype(np.float32)
    one = np.array(msig.sosfilt(sos, x))
    zi = np.zeros((S, 10, 2), dtype=np.float32)
    parts = []
    for chunk in np.split(x, 3, axis=-1):
        y, zi = msig.sosfilt(sos, chunk, zi=zi)
        zi = np.array(zi)
        parts.append(np.array(y))
    np.testing.assert_allclose(np.concatenate(parts, axis=-1), one, rtol=1e-5,
                               atol=1e-5 * np.abs(one).max())


def test_sosfilt_complex_x(rng):
    sos = FILTERS[0]
    x = (rng.standard_normal((9, 6000)) + 1j * rng.standard_normal((9, 6000))).astype(
        np.complex64
    )
    ref = sps.sosfilt(sos.astype(np.float32), x)
    out = msig.sosfilt(sos, x)
    np.testing.assert_allclose(np.array(out), ref, rtol=1e-5,
                               atol=1e-5 * np.abs(ref).max())


def test_sosfilt_impulse_response_matches_freqz(rng):
    """Physics check: kernel impulse response matches the design's sosfreqz."""
    sos = FILTERS[2]
    x = np.zeros((8, 4096), dtype=np.float32)
    x[:, 0] = 1.0
    h = np.array(msig.sosfilt(sos, x))[0]
    w, resp = _freqz_sos(sos, worN=2048)
    got = np.fft.rfft(h, 4096)[:2048]
    np.testing.assert_allclose(np.abs(got), np.abs(resp), atol=2e-4)


def test_sosfilt_validation(rng):
    x = rng.standard_normal(100).astype(np.float32)
    with pytest.raises(ValueError, match="shape"):
        msig.sosfilt(np.zeros((2, 5)), x)
    bad = FILTERS[0].copy()
    bad[0, 3] = 2.0
    with pytest.raises(ValueError, match="ones"):
        msig.sosfilt(bad, x)
    with pytest.raises(ValueError, match="zi"):
        msig.sosfilt(FILTERS[0], np.zeros((3, 50), np.float32), zi=np.zeros((2, 2)))


def test_sosfilt_many_sections_falls_back(rng):
    stable = sps.butter(2, 0.3, output="sos")
    sos = np.tile(stable, (20, 1))  # 20 sections > kernel cap
    x = rng.standard_normal((10, 5000)).astype(np.float32)
    with msig.config_context(dispatch="auto", gpu_min_size=1, warn_on_fallback=True):
        with pytest.warns(msig.FallbackWarning):
            out = msig.sosfilt(sos, x)
    np.testing.assert_allclose(np.array(out), _f32_ref(sos, x), rtol=1e-4,
                               atol=1e-4 * np.abs(np.array(out)).max())
    if HAS_GPU:  # fixture pins dispatch="mlx" only when Metal exists
        with pytest.raises(NotImplementedError):
            msig.sosfilt(sos, x)


def test_sosfilt_single_channel_uses_scan_under_auto(rng):
    """Long single-channel input runs the block-parallel scan kernel on GPU."""
    import mlx.core as mx

    sos = FILTERS[0]
    x = rng.standard_normal(200000).astype(np.float32)
    with msig.config_context(dispatch="auto"):
        out = msig.sosfilt(sos, x)
    assert isinstance(out, mx.array) and out.dtype == mx.float32
    np.testing.assert_allclose(np.array(out), _f32_ref(sos, x), rtol=1e-5,
                               atol=1e-5 * np.abs(np.array(out)).max())


def test_sosfilt_short_signal_sequential_kernel(rng):
    """n below the scan threshold exercises the per-channel-sequential kernel."""
    sos = FILTERS[1]
    x = rng.standard_normal((40, 1500)).astype(np.float32)
    ref = _f32_ref(sos, x)
    np.testing.assert_allclose(np.array(msig.sosfilt(sos, x)), ref, rtol=_EXACT_RTOL,
                               atol=_EXACT_RTOL * np.abs(ref).max())


def test_scan_and_sequential_kernels_agree(rng):
    """The two GPU implementations are independent; they must agree."""
    import mlx.core as mx

    from mlx_signal_processing import _sosfilt_metal

    if not mx.metal.is_available():
        pytest.skip("no Metal GPU")
    sos = np.asarray(FILTERS[2], dtype=np.float64)
    S = sos.shape[0]
    x = mx.array(rng.standard_normal((3, 50001)).astype(np.float32))
    zi = mx.array(rng.standard_normal((3, S, 2)).astype(np.float32) * 0.1)
    sflat = mx.array(sos.astype(np.float32).reshape(-1))
    y_seq, zf_seq = _sosfilt_metal.sosfilt_gpu(x, sflat, zi)
    y_scan, zf_scan = _sosfilt_metal.sosfilt_scan_gpu(x, sos, zi)
    np.testing.assert_allclose(np.array(y_scan), np.array(y_seq), rtol=1e-4,
                               atol=1e-5 * float(mx.max(mx.abs(y_seq))))
    np.testing.assert_allclose(np.array(zf_scan), np.array(zf_seq), rtol=1e-4,
                               atol=1e-5)


@pytest.mark.parametrize("padtype", ["odd", "even", "constant", None])
def test_sosfiltfilt_matches_scipy(rng, padtype):
    sos = FILTERS[0]
    x = rng.standard_normal((12, 4000)).astype(np.float32)
    ref = sps.sosfiltfilt(sos.astype(np.float32), x, padtype=padtype)
    out = msig.sosfiltfilt(sos, x, padtype=padtype)
    assert ref.shape == tuple(np.array(out).shape)
    np.testing.assert_allclose(np.array(out), ref, rtol=1e-4,
                               atol=1e-5 * np.abs(ref).max())


def test_sosfiltfilt_padlen_and_axis(rng):
    sos = FILTERS[2]
    x = rng.standard_normal((3000, 9)).astype(np.float32)
    ref = sps.sosfiltfilt(sos.astype(np.float32), x, axis=0, padlen=500)
    out = msig.sosfiltfilt(sos, x, axis=0, padlen=500)
    np.testing.assert_allclose(np.array(out), ref, rtol=1e-4,
                               atol=1e-5 * np.abs(ref).max())


def test_sosfiltfilt_padlen_too_long(rng):
    # butter(4) -> 2 sections -> default edge = 15, so 12 samples must raise
    with pytest.raises(ValueError, match="padlen"):
        msig.sosfiltfilt(FILTERS[0], rng.standard_normal(12).astype(np.float32))


def test_sosfiltfilt_zero_phase_property(rng):
    sos = FILTERS[0]
    x = np.zeros((8, 2048), dtype=np.float32)
    x[:, 1024] = 1.0
    y = np.array(msig.sosfiltfilt(sos, x))[0]
    assert abs(int(np.argmax(np.abs(y))) - 1024) <= 1


def test_decimate_iir_gpu(rng):
    """decimate's default IIR path now runs on the GPU without warnings."""
    import warnings as _w

    x = rng.standard_normal((10, 8000)).astype(np.float32)
    with _w.catch_warnings():
        _w.simplefilter("error", msig.FallbackWarning)
        out = msig.decimate(x, 4)
    ref = sps.decimate(x.astype(np.float64), 4)
    assert ref.shape == tuple(np.array(out).shape)
    assert_close(out, ref, rtol=1e-3, atol_frac=1e-4)


def test_decimate_iir_zero_phase_false(rng):
    x = rng.standard_normal((10, 8000)).astype(np.float32)
    ref = sps.decimate(x.astype(np.float64), 5, zero_phase=False)
    assert_close(msig.decimate(x, 5, zero_phase=False), ref, rtol=1e-3, atol_frac=1e-4)


# ---------------------------------------------------------------------------
# audit regressions: dtype x section count x block count x pole radius
# ---------------------------------------------------------------------------

_MATRIX_FILTERS = {
    "wide-1sec": sps.butter(2, 0.3, output="sos"),
    "wide-8sec": sps.butter(8, [0.1, 0.3], btype="band", output="sos"),
    "narrowband": sps.butter(2, 2e-4, output="sos"),  # scan-unsafe: sequential
}


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("sos_key", sorted(_MATRIX_FILTERS))
@pytest.mark.parametrize("n", [3000, 5 * 1024 + 37, 200_000])
def test_sosfilt_scan_matrix(rng, dtype, sos_key, n):
    """Every (coefficient dtype, section count, block count, pole radius)
    combination must track the scipy recurrence — float32 coefficients
    previously crashed the scan-safety cache, and narrowband poles silently
    drifted through the block-composed path. 40 channels so auto dispatch
    engages the kernels (wide filters scan; narrowband goes sequential);
    without Metal everything runs scipy on the same canonical f32 inputs."""
    sos = _MATRIX_FILTERS[sos_key].astype(dtype)
    x = rng.standard_normal((40, n)).astype(np.float32)
    ref = sps.sosfilt(sos.astype(np.float32), x)  # all routes run the f32 filter
    with msig.config_context(dispatch="auto"):
        out = np.array(msig.sosfilt(sos, x))
    tol = 1e-5 if HAS_GPU else 1e-4
    if sos_key == "narrowband" and _SCIPY_VER < (1, 15):
        # pre-1.15 scipy's float32 recurrence rounds in a different op order;
        # this filter's conditioning amplifies that to ~1e-2 relative-to-max
        # (still far below the 16% drift this test guards against)
        tol = 1e-2
    np.testing.assert_allclose(out, ref, rtol=tol,
                               atol=tol * max(np.abs(ref).max(), 1e-9))


def test_sosfilt_scan_small_launches(rng):
    """1 section x 1 channel at 2048-3072 samples previously failed Metal
    compilation (tiny scan intermediates land in constant address space)."""
    if not HAS_GPU:
        pytest.skip("scan kernel needs Metal")
    sos = sps.butter(2, 0.3, output="sos")
    for n in (2048, 2560, 3072):
        x = rng.standard_normal(n).astype(np.float32)
        ref = _f32_ref(sos, x)
        out = np.array(msig.sosfilt(sos, x))
        np.testing.assert_allclose(out, ref, rtol=1e-5,
                                   atol=1e-5 * np.abs(ref).max())


def test_sosfiltfilt_narrowband_zi_matches_executed_filter():
    """zi must describe the float32-rounded filter the kernel executes; the
    f64-designed zi started narrowband cascades visibly wrong (0.298 error)."""
    sos = sps.butter(2, 1e-4, output="sos")
    x = np.ones(60_000, np.float32)
    # the f32-rounded coefficients form a genuinely different filter (DC-gain
    # cancellation); every dispatch route now executes exactly that filter
    ref = sps.sosfiltfilt(sos.astype(np.float32), x)
    out = np.array(msig.sosfiltfilt(sos, x))
    assert np.abs(out - ref).max() < 0.02


def test_sosfilt_auto_routing_is_filter_invariant(rng):
    """Default-auto routing (scipy vs kernel, by batch size / scan safety)
    must never change WHICH filter runs: every route executes the float32-
    quantized coefficients. Previously a narrowband design differed by 4.9%
    L2 across the 31/32-channel boundary and sosfiltfilt mixed precisions
    within one call (0.76 max error from batching alone)."""
    sos = sps.butter(2, 1e-4, output="sos")
    xn = rng.standard_normal(20_000).astype(np.float32)
    with msig.config_context(dispatch="auto", warn_on_downcast=False):
        r31 = np.array(msig.sosfilt(sos, np.tile(xn, (31, 1))))[0]
        r32 = np.array(msig.sosfilt(sos, np.tile(xn, (32, 1))))[0]
    tol = 1e-6 if HAS_GPU else 0.0
    np.testing.assert_allclose(r31, r32, atol=tol * max(np.abs(r32).max(), 1e-9))

    x1 = np.ones(60_000, np.float32)
    with msig.config_context(dispatch="auto", warn_on_downcast=False):
        o1 = np.array(msig.sosfiltfilt(sos, x1))
        ob = np.array(msig.sosfiltfilt(sos, np.tile(x1, (32, 1))))[0]
    np.testing.assert_allclose(o1, ob, atol=tol * max(np.abs(ob).max(), 1e-9))
    ref = sps.sosfiltfilt(sos.astype(np.float32), x1)
    assert np.abs(o1 - ref).max() < 0.02  # matches the f32 filter, not neither


def test_old_scipy_scan_unsafe_auto_stays_on_scipy(monkeypatch, rng):
    """Before scipy 1.15 its f32 op order differs from Metal. A scan-unsafe
    filter must not switch recurrence at the 31/32-row auto boundary."""
    import mlx.core as mx

    from mlx_signal_processing import _sosfilt_metal, filtering

    monkeypatch.setattr(filtering, "_scipy_sosfilt_f32_order_matches_metal", lambda: False)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)

    def unexpected_kernel(*args, **kwargs):
        raise AssertionError("old-scipy scan-unsafe auto call reached Metal")

    monkeypatch.setattr(_sosfilt_metal, "sosfilt_gpu", unexpected_kernel)
    sos = sps.butter(2, 1e-4, output="sos")
    x = rng.standard_normal((32, 20_000)).astype(np.float32)
    ref = sps.sosfilt(sos.astype(np.float32), x)
    with msig.config_context(dispatch="auto", warn_on_downcast=False):
        out = np.array(msig.sosfilt(sos, x))
    np.testing.assert_array_equal(out, ref)


@pytest.mark.parametrize("n", [16_383, 16_384])  # straddles the auto threshold
def test_sosfilt_complex_zi_real_x(rng, n):
    """scipy accepts real x with complex zi (complex output); so must every
    dispatch route — the Metal path previously raised above the threshold."""
    sos = sps.butter(4, 0.25, output="sos")
    zi = (rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))).astype(np.complex64)
    x = rng.standard_normal(n).astype(np.float32)
    ref_y, ref_zf = sps.sosfilt(sos.astype(np.float32), x, zi=zi)
    with msig.config_context(dispatch="auto", warn_on_downcast=False):
        y, zf = msig.sosfilt(sos, x, zi=zi)
    assert np.array(y).dtype == np.complex64
    np.testing.assert_allclose(np.array(y), ref_y, rtol=1e-4,
                               atol=1e-5 * np.abs(ref_y).max())
    np.testing.assert_allclose(np.array(zf), ref_zf, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# route-invariant dtype/state regressions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("complex_state", [False, True])
def test_sosfilt_auto_boundary_canonicalizes_f64_signal_and_state(rng, complex_state):
    """The 31/32-row routing boundary must not retain f64/c128 arithmetic on
    the scipy side while the kernel side consumes f32/c64."""
    sos = sps.butter(2, 1e-4, output="sos")
    x = rng.standard_normal(20_000)  # deliberately float64
    zi = rng.standard_normal((1, 2))
    if complex_state:
        zi = zi + 1j * rng.standard_normal((1, 2))
    zi_dtype = np.complex64 if complex_state else np.float32
    ref_y, ref_zf = sps.sosfilt(
        sos.astype(np.float32), x.astype(np.float32), zi=zi.astype(zi_dtype)
    )

    rows = {}
    states = {}
    for batch in (31, 32):
        xb = np.tile(x, (batch, 1))
        zib = np.tile(zi[:, None, :], (1, batch, 1))
        with msig.config_context(dispatch="auto", warn_on_downcast=False):
            y, zf = msig.sosfilt(sos, xb, zi=zib)
        rows[batch] = np.array(y)[0]
        states[batch] = np.array(zf)[:, 0]

    tol = 1e-6 if HAS_GPU else 0.0
    for batch in (31, 32):
        np.testing.assert_allclose(rows[batch], ref_y, rtol=tol,
                                   atol=tol * max(np.abs(ref_y).max(), 1e-9))
        np.testing.assert_allclose(states[batch], ref_zf, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(rows[31], rows[32], rtol=tol,
                               atol=tol * max(np.abs(rows[32]).max(), 1e-9))


@pytest.mark.parametrize("dispatch", ["auto", "scipy"])
@pytest.mark.parametrize("complex_signal", [False, True])
def test_sosfilt_high_precision_state_streaming_is_chunk_invariant(
    rng, dispatch, complex_signal
):
    """A c128 initial state is canonicalized before the first fallback pass;
    carrying the returned c64 state must therefore equal one-shot filtering."""
    sos = sps.butter(2, 1e-4, output="sos")
    x = rng.standard_normal(60_000)
    if complex_signal:
        x = x + 1j * rng.standard_normal(60_000)
    zi0 = (
        rng.standard_normal((1, 2)) + 1j * rng.standard_normal((1, 2))
    ).astype(np.complex128)
    with msig.config_context(dispatch=dispatch, warn_on_downcast=False):
        whole, whole_zf = msig.sosfilt(sos, x, zi=zi0)
        state = zi0
        parts = []
        for lo, hi in ((0, 1000), (1000, 10_000), (10_000, 40_000), (40_000, 60_000)):
            part, state = msig.sosfilt(sos, x[lo:hi], zi=state)
            parts.append(np.array(part))
    streamed = np.concatenate(parts)
    np.testing.assert_allclose(streamed, np.array(whole), rtol=1e-6,
                               atol=1e-6 * max(np.abs(whole).max().item(), 1e-9))
    np.testing.assert_allclose(np.array(state), np.array(whole_zf), rtol=1e-6, atol=1e-6)


def test_sosfilt_complex_auto_boundary_uses_split_f32_planes(rng):
    """SciPy fallback and Metal both run real/imaginary f32 recurrences, so a
    complex signal/state cannot reveal the 31/32-row implementation boundary."""
    sos = sps.butter(2, 1e-4, output="sos")
    x = rng.standard_normal(20_000) + 1j * rng.standard_normal(20_000)
    zi = rng.standard_normal((1, 2)) + 1j * rng.standard_normal((1, 2))
    xq = x.astype(np.complex64)
    zq = zi.astype(np.complex64)
    yr, zfr = sps.sosfilt(sos.astype(np.float32), xq.real, zi=zq.real)
    yi, zfi = sps.sosfilt(sos.astype(np.float32), xq.imag, zi=zq.imag)
    ref_y = yr.astype(np.complex64) + np.complex64(1j) * yi
    ref_zf = zfr.astype(np.complex64) + np.complex64(1j) * zfi

    rows = {}
    for batch in (31, 32):
        xb = np.tile(x, (batch, 1))
        zib = np.tile(zi[:, None, :], (1, batch, 1))
        with msig.config_context(dispatch="auto", warn_on_downcast=False):
            y, zf = msig.sosfilt(sos, xb, zi=zib)
        rows[batch] = np.array(y)[0]
        np.testing.assert_allclose(rows[batch], ref_y, rtol=1e-6,
                                   atol=1e-6 * max(np.abs(ref_y).max(), 1e-9))
        np.testing.assert_allclose(np.array(zf)[:, 0], ref_zf, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(rows[31], rows[32], rtol=1e-6,
                               atol=1e-6 * max(np.abs(rows[32]).max(), 1e-9))


def test_sosfiltfilt_all_routes_share_exact_f32_zi():
    """The direct scipy and manual/kernel routes must use the same f32
    sosfilt_zi result; promoting rounded coefficients changed it by one ulp."""
    sos = sps.butter(2, 1e-4, output="sos")
    x = np.ones(60_000, np.float32)
    ref = sps.sosfiltfilt(sos.astype(np.float32), x)
    results = {}
    contexts = {
        "scipy": {"dispatch": "scipy"},
        "auto-scipy": {"dispatch": "auto", "gpu_min_size": 1 << 30},
        "auto-kernel": {"dispatch": "auto", "gpu_min_size": 0},
    }
    if HAS_GPU:
        contexts["mlx"] = {"dispatch": "mlx"}
    for name, kwargs in contexts.items():
        with msig.config_context(warn_on_downcast=False, **kwargs):
            results[name] = np.array(msig.sosfiltfilt(sos, x))

    tol = (1e-5 if _SCIPY_VER >= (1, 15) else 1e-2) if HAS_GPU else 0.0
    for out in results.values():
        np.testing.assert_allclose(out, ref, rtol=tol,
                                   atol=tol * max(np.abs(ref).max(), 1e-9))
    first = next(iter(results.values()))
    for out in results.values():
        np.testing.assert_allclose(out, first, rtol=tol,
                                   atol=tol * max(np.abs(first).max(), 1e-9))


@pytest.mark.parametrize("func", [msig.sosfilt, msig.sosfiltfilt])
@pytest.mark.parametrize("dispatch", ["auto", "mlx", "scipy"])
@pytest.mark.parametrize("complex_coefficients", [False, True])
def test_sos_rejects_filter_destabilized_by_32bit_quantization(
    func, dispatch, complex_coefficients
):
    """A stable f64 design whose f32 poles cross the unit circle must fail
    clearly and identically before any route can produce explosive output."""
    sos = sps.butter(2, 5e-5, output="sos")
    assert np.max(np.abs(np.roots([1.0, *sos[0, 4:6]]))) < 1.0
    sos32 = sos.astype(np.float32)
    assert np.max(np.abs(np.roots([1.0, *sos32[0, 4:6]]))) >= 1.0
    if complex_coefficients:
        sos = sos.astype(np.complex128)
    dtype_name = "complex64" if complex_coefficients else "float32"
    with msig.config_context(dispatch=dispatch, warn_on_downcast=False):
        with pytest.raises(ValueError, match=rf"unstable after {dtype_name} quantization"):
            func(sos, np.ones(256, np.float32))


def test_sos_rejects_numerator_lost_to_f32_underflow():
    """A nonzero section becoming the all-zero numerator would silently turn
    the entire cascade into zero, even though all quantized poles stay stable."""
    sos = sps.butter(16, 1e-4, output="sos")
    assert np.any(sos[0, :3] != 0)
    assert np.all(sos.astype(np.float32)[0, :3] == 0)
    with pytest.raises(ValueError, match="numerator vanished after float32 quantization"):
        msig.sosfilt(sos, np.ones(256, np.float32))


def test_sosfilt_preserves_explicit_integrator_semantics():
    """Only a stability loss caused by quantization is rejected; causal
    sosfilt still supports an explicitly supplied pole at one like scipy."""
    sos = np.array([[1, 0, 0, 1, -1, 0]], dtype=np.float32)
    x = np.arange(1, 33, dtype=np.float32)
    with msig.config_context(dispatch="scipy"):
        out = np.array(msig.sosfilt(sos, x))
    np.testing.assert_allclose(out, sps.sosfilt(sos, x))


def test_complex_sos_uses_c64_coefficient_policy():
    """Changing only a real SOS array's dtype to complex must not silently
    switch the scipy fallback from the documented 32-bit computation to c128."""
    real_sos = sps.butter(2, 1e-4, output="sos").astype(np.float32)
    complex_sos = real_sos.astype(np.complex128)
    x = np.ones(60_000, np.float32)
    ref = sps.sosfiltfilt(complex_sos.astype(np.complex64), x)
    with msig.config_context(dispatch="scipy", warn_on_downcast=False):
        real_out = np.array(msig.sosfiltfilt(real_sos, x))
        complex_out = np.array(msig.sosfiltfilt(complex_sos, x))
    assert complex_out.dtype == np.complex64
    np.testing.assert_allclose(complex_out, ref, rtol=1e-6)
    np.testing.assert_allclose(complex_out.real, real_out, rtol=1e-6)
    np.testing.assert_allclose(complex_out.imag, 0.0, atol=1e-7)


def test_sos_coefficient_dtype_policy_is_quiet_by_default_and_strict_on_request(rng):
    """SciPy designs are f64, so their canonicalization is quiet. Deliberate
    c128 coefficients warn; strict mode rejects both coefficient dtypes."""
    import warnings

    sos64 = sps.butter(4, 0.2, output="sos")
    x = rng.standard_normal(1000).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", msig.DowncastWarning)
        with msig.config_context(dispatch="scipy", warn_on_downcast=True):
            msig.sosfilt(sos64, x)
    with msig.config_context(dispatch="scipy", warn_on_downcast=True):
        with pytest.warns(msig.DowncastWarning, match="complex128"):
            msig.sosfilt(sos64.astype(np.complex128), x)
    with msig.config_context(dispatch="scipy", float64="strict"):
        with pytest.raises(TypeError, match="strict"):
            msig.sosfilt(sos64, x)
        with pytest.raises(TypeError, match="strict"):
            msig.sosfilt(sos64.astype(np.complex128), x)
        # Already-canonical coefficient arrays remain usable under strict mode.
        msig.sosfilt(sos64.astype(np.float32), x)
        msig.sosfilt(sos64.astype(np.complex64), x)


@pytest.mark.parametrize("n", [128, 40_000])
@pytest.mark.parametrize("dispatch", ["auto", "scipy"])
def test_sosfiltfilt_signal_warning_and_strict_are_route_independent(n, dispatch):
    """Small/direct and large/manual calls must both canonicalize a f64 signal
    once, warn once in downcast mode, and reject it in strict mode."""
    sos = sps.butter(4, 0.2, output="sos").astype(np.float32)
    x = np.ones(n, np.float64)
    with msig.config_context(dispatch=dispatch, warn_on_downcast=True):
        with pytest.warns(msig.DowncastWarning) as caught:
            msig.sosfiltfilt(sos, x)
    assert len(caught) == 1
    with msig.config_context(dispatch=dispatch, float64="strict"):
        with pytest.raises(TypeError, match="strict"):
            msig.sosfiltfilt(sos, x)


@pytest.mark.parametrize("zero_phase", [False, True])
@pytest.mark.parametrize("dispatch", ["auto", "scipy"])
def test_decimate_internal_iir_design_works_under_strict(rng, zero_phase, dispatch):
    """The f64 SOS produced internally by scipy.cheby1 is implementation
    metadata, not a forbidden f64 user input under strict mode."""
    x = rng.standard_normal(4000).astype(np.float32)
    with msig.config_context(dispatch=dispatch, float64="strict"):
        out = np.array(msig.decimate(x, 4, zero_phase=zero_phase))
    ref = sps.decimate(x, 4, zero_phase=zero_phase)
    np.testing.assert_allclose(out, ref, rtol=1e-3,
                               atol=1e-4 * max(np.abs(ref).max(), 1e-9))
