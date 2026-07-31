"""Golden tests: sosfilt / sosfiltfilt (batched-channel IIR kernel) vs scipy.

The kernel runs scipy's exact direct-form-II-transposed recurrence in float32,
so it is compared bit-tight against scipy executing in float32, and loosely
against scipy's float64 results (the difference is float32 itself, not the
implementation).
"""

import numpy as np
import pytest
import scipy.signal as sps

import mlx_signal as msig
from _utils import assert_close, assert_type_and_dtype

FILTERS = [
    sps.butter(4, 0.2, output="sos"),
    sps.butter(8, [0.1, 0.3], btype="band", output="sos"),
    sps.cheby1(8, 0.05, 0.25, output="sos"),
    sps.ellip(6, 0.1, 60, 0.3, output="sos"),
    sps.butter(1, 0.5, output="sos"),  # single section
]


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
    np.testing.assert_allclose(np.array(out), ref, rtol=1e-6,
                               atol=1e-6 * np.abs(ref).max())


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
    np.testing.assert_allclose(np.array(msig.sosfilt(sos, x, axis=0)), ref, rtol=1e-6,
                               atol=1e-6 * np.abs(ref).max())


def test_sosfilt_zi_roundtrip(rng):
    sos = FILTERS[1]
    S = sos.shape[0]
    x = rng.standard_normal((5, 8000)).astype(np.float32)
    zi = rng.standard_normal((S, 5, 2)).astype(np.float32)
    y_ref, zf_ref = _f32_ref(sos, x, zi=zi)
    y, zf = msig.sosfilt(sos, x, zi=zi)
    np.testing.assert_allclose(np.array(y), y_ref, rtol=1e-6,
                               atol=1e-6 * np.abs(y_ref).max())
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
    w, resp = sps.freqz_sos(sos, worN=2048)
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
    with msig.config_context(dispatch="auto", gpu_min_size=1):
        with pytest.warns(msig.FallbackWarning):
            out = msig.sosfilt(sos, x)
    np.testing.assert_allclose(np.array(out), _f32_ref(sos, x), rtol=1e-4,
                               atol=1e-4 * np.abs(np.array(out)).max())
    with pytest.raises(NotImplementedError):
        msig.sosfilt(sos, x)  # dispatch="mlx" pinned by fixture


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
    np.testing.assert_allclose(np.array(msig.sosfilt(sos, x)), ref, rtol=1e-6,
                               atol=1e-6 * np.abs(ref).max())


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
