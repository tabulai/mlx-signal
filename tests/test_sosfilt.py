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

import mlx_signal as msig
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
# scipy's float32 recurrence rounds in a different op order (~2e-6 drift), and
# without Metal the scipy fallback computes in float64 — loosen accordingly
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

    from mlx_signal import _sosfilt_metal

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
    without Metal everything runs scipy in float64."""
    sos = _MATRIX_FILTERS[sos_key].astype(dtype)
    x = rng.standard_normal((40, n)).astype(np.float32)
    ref = sps.sosfilt(sos.astype(np.float32) if HAS_GPU else sos.astype(np.float64), x)
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
    # cancellation): compare against whichever filter this build executes
    ref = sps.sosfiltfilt(sos.astype(np.float32) if HAS_GPU else sos, x)
    out = np.array(msig.sosfiltfilt(sos, x))
    assert np.abs(out - ref).max() < 0.02
