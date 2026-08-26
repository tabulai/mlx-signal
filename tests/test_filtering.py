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


def test_lfilter_iir_runs_without_fallback(rng):
    """Transfer-function IIR is served by the TF Metal kernel — no warning."""
    import warnings as _w

    x = rng.standard_normal((40, 3000)).astype(np.float32)
    b, a = sps.butter(4, 0.2)
    with _w.catch_warnings():
        _w.simplefilter("error", msig.FallbackWarning)
        out = msig.lfilter(b, a, x)
    ref = sps.lfilter(b.astype(np.float32), a.astype(np.float32), x)
    tol = 1e-6 if HAS_GPU else 0.0  # no-Metal route IS scipy on the same f32 inputs
    np.testing.assert_allclose(np.array(out), ref, rtol=tol,
                               atol=tol * max(np.abs(ref).max(), 1e-9))


def test_lfilter_iir_order_above_cap_falls_back(rng):
    from mlx_signal._lfilter_metal import MAX_ORDER

    x = rng.standard_normal(5000).astype(np.float32)
    # a stable high-order denominator: 17 poles well inside the unit circle
    poles = 0.5 * np.exp(1j * np.pi * np.linspace(-0.9, 0.9, MAX_ORDER + 1))
    a = np.real(np.poly(np.concatenate([poles, np.conj(poles)])))[: MAX_ORDER + 2]
    a = a / a[0]
    b = rng.standard_normal(MAX_ORDER + 2)
    with msig.config_context(dispatch="auto", gpu_min_size=1, warn_on_fallback=True):
        with pytest.warns(msig.FallbackWarning, match="order"):
            out = msig.lfilter(b, a, x)
    ref = sps.lfilter(b.astype(np.float32), a.astype(np.float32), x)
    assert_close(out, ref, rtol=1e-3)
    if HAS_GPU:  # fixture pins dispatch="mlx" only when Metal exists
        with pytest.raises(NotImplementedError):
            msig.lfilter(b, a, x)


def test_lfilter_iir_complex_coefficients_fall_back(rng):
    x = rng.standard_normal(5000).astype(np.float32)
    b = np.asarray([0.2 + 0.1j, 0.1], dtype=np.complex64)
    a = np.asarray([1.0, -0.4 + 0.2j], dtype=np.complex64)
    with msig.config_context(dispatch="auto", gpu_min_size=1, warn_on_fallback=True):
        with pytest.warns(msig.FallbackWarning, match="complex"):
            out = msig.lfilter(b, a, x)
    ref = sps.lfilter(b, a, x.astype(np.float32))
    np.testing.assert_allclose(np.array(out), ref, rtol=1e-5,
                               atol=1e-5 * np.abs(ref).max())


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


def test_filtfilt_iir_matches_scipy_f32(rng):
    """Transfer-function filtfilt runs both passes on the GPU — no warning."""
    import warnings as _w

    x = rng.standard_normal((40, 3000)).astype(np.float32)
    b, a = sps.butter(3, 0.1)
    with _w.catch_warnings():
        _w.simplefilter("error", msig.FallbackWarning)
        out = msig.filtfilt(b, a, x)
    ref = sps.filtfilt(b.astype(np.float32), a.astype(np.float32), x)
    tol = 1e-6 if HAS_GPU else 1e-4
    np.testing.assert_allclose(np.array(out), ref, rtol=tol,
                               atol=tol * max(np.abs(ref).max(), 1e-9))


def test_filtfilt_zero_phase_property(rng):
    """filtfilt of a delayed impulse stays centered (no group delay)."""
    b = np.array(msig.firwin(51, 0.2))
    x = np.zeros(512, dtype=np.float32)
    x[256] = 1.0
    y = np.array(msig.filtfilt(b, [1.0], x))
    assert abs(int(np.argmax(y)) - 256) <= 1


# ---------------------------------------------------------------------------
# transfer-function IIR (order-N DF2T Metal kernel)
# ---------------------------------------------------------------------------
#
# The kernel spells out scipy's fma-contracted float32 recurrence, so against
# a scipy build with the same contraction (probed, not version-gated) it is
# bit-identical; without Metal the fallback runs scipy on the same canonical
# f32 inputs, which is exact by construction.

from mlx_signal.filtering import _scipy_lfilter_f32_order_matches_metal

_TF_EXACT = (not HAS_GPU) or _scipy_lfilter_f32_order_matches_metal()

_TF_FILTERS = [
    sps.butter(1, 0.5),
    sps.butter(2, 0.15),
    sps.butter(4, 0.2),
    sps.cheby1(5, 0.05, 0.3),
    sps.butter(8, 0.25),
]


def _tf_f32(ba):
    b, a = ba
    return b.astype(np.float32), a.astype(np.float32)


def _assert_tf(out, ref, tol=1e-6):
    """Bit-tight against scipy-in-f32: the sequential kernel is bit-identical
    on a probe-matching scipy build; scan blocks compose entry states in f32,
    which drifts by ulps at block boundaries (same bound as the SOS suite)."""
    out = np.asarray(out)
    if not _TF_EXACT:
        tol = max(tol, 1e-4)
    np.testing.assert_allclose(out, ref, rtol=tol,
                               atol=tol * max(np.abs(np.asarray(ref)).max(), 1e-9))


@pytest.mark.parametrize("a0_scale", [1.0, 2.5])
@pytest.mark.parametrize("ba", _TF_FILTERS)
def test_lfilter_tf_sequential_bitwise(rng, ba, a0_scale):
    """Below the scan threshold the sequential kernel must be BIT-identical to
    the probed scipy build's f32 recurrence — not merely close — including
    the a[0] != 1 pre-normalization."""
    if not _TF_EXACT:
        pytest.skip("scipy build's f32 rounding differs from the kernel")
    b32, a32 = _tf_f32(ba)
    b32, a32 = (b32 * np.float32(a0_scale)), (a32 * np.float32(a0_scale))
    x = rng.standard_normal((40, 1500)).astype(np.float32)
    ref = sps.lfilter(b32, a32, x)
    np.testing.assert_array_equal(np.array(msig.lfilter(b32, a32, x)), ref)


def test_lfilter_tf_zi_sequential_short_n(rng):
    """zi must flow bit-correctly through the SEQUENTIAL kernel too (short n
    keeps the call off the scan route)."""
    b32, a32 = _tf_f32(_TF_FILTERS[3])
    order = max(len(b32), len(a32)) - 1
    x = rng.standard_normal((40, 1500)).astype(np.float32)
    zi = (rng.standard_normal((40, order)) * 0.1).astype(np.float32)
    y_ref, zf_ref = sps.lfilter(b32, a32, x, zi=zi)
    y, zf = msig.lfilter(*_TF_FILTERS[3], x, zi=zi)
    _assert_tf(y, y_ref)
    _assert_tf(zf, zf_ref, tol=1e-5)


def test_lfilter_tf_scalar_b_and_a(rng):
    """Scalar (0-d) b with scalar a is a pure gain, like scipy — and must not
    crash under pinned MLX dispatch."""
    x = rng.standard_normal(1000).astype(np.float32)
    ref = sps.lfilter(np.float32(3.0), np.float32(2.0), x)
    out = msig.lfilter(3.0, 2.0, x)
    np.testing.assert_allclose(np.array(out), ref, rtol=1e-6)
    if HAS_GPU:
        with msig.config_context(dispatch="mlx"):
            out = msig.lfilter(3.0, 2.0, x)
        np.testing.assert_allclose(np.array(out), ref, rtol=1e-6)


def test_lfilter_tf_axis_out_of_range_raises(rng):
    b32, a32 = _tf_f32(_TF_FILTERS[1])
    x = rng.standard_normal((8, 3000)).astype(np.float32)
    with pytest.raises(ValueError, match="out of range"):
        msig.lfilter(b32, a32, x, axis=5)
    with pytest.raises(ValueError, match="out of range"):
        msig.filtfilt(b32, a32, x, axis=-3)


def test_lfilter_tf_scan_composition_matters(rng):
    """A slow real pole (r = 0.99) keeps A^L far from zero (~3e-5), so the
    scan's entry-state composition genuinely contributes; breaking it (or
    transposing AL) shifts results ~2e-5 while pristine code stays ~1e-6."""
    import mlx.core as mx

    from mlx_signal import _lfilter_metal

    if not mx.metal.is_available():
        pytest.skip("scan kernel needs Metal")
    b32 = np.asarray([1.0], np.float32)
    a32 = np.asarray([1.0, -0.99], np.float32)
    ba = np.zeros((2, 2), np.float32)
    ba[0, :1], ba[1, :2] = b32, a32
    x = rng.standard_normal((1, 1 << 17)).astype(np.float32)
    ref = sps.lfilter(b32, a32, x)
    y, _ = _lfilter_metal.lfilter_scan_gpu(mx.array(x), ba, mx.zeros((1, 1)))
    np.testing.assert_allclose(np.array(y), ref, rtol=2e-6,
                               atol=2e-6 * np.abs(ref).max())


def test_lfilter_tf_scan_engages_and_drift_stays_bounded(rng, monkeypatch):
    """A gate-passing near-cap resonator must actually take the scan route
    (guards against a gate stuck False) and stay within the documented worst
    ~1e-5 drift bound."""
    import mlx.core as mx

    from mlx_signal import _lfilter_metal, filtering

    if not mx.metal.is_available():
        pytest.skip("scan kernel needs Metal")
    b, a = sps.butter(2, 0.013)  # transient gain ~23: passes the 24 cap
    calls = []
    real_scan = _lfilter_metal.lfilter_scan_gpu

    def recording_scan(*args, **kwargs):
        calls.append(1)
        return real_scan(*args, **kwargs)

    monkeypatch.setattr(_lfilter_metal, "lfilter_scan_gpu", recording_scan)
    x = rng.standard_normal((2, 1 << 18)).astype(np.float32)
    out = np.array(msig.lfilter(b, a, x))
    assert calls, "gate-passing filter did not dispatch to the scan kernel"
    ref = sps.lfilter(b.astype(np.float32), a.astype(np.float32), x)
    np.testing.assert_allclose(out, ref, rtol=2e-5,
                               atol=2e-5 * np.abs(ref).max())


def test_filtfilt_tf_complex_x_is_route_invariant(rng):
    """Complex signals run the same two-plane recurrence on every dispatch —
    the silent size boundary must not switch to scipy's native c64 filtfilt."""
    b, a = sps.butter(4, 0.03)
    x = (rng.standard_normal(8100)
         + 1j * rng.standard_normal(8100)).astype(np.complex64)
    with msig.config_context(dispatch="scipy", warn_on_downcast=False):
        y_scipy = np.array(msig.filtfilt(b, a, x))
    with msig.config_context(dispatch="auto", gpu_min_size=0, warn_on_downcast=False):
        y_auto = np.array(msig.filtfilt(b, a, x))
    tol = 1e-6 if HAS_GPU else 0.0
    np.testing.assert_allclose(y_scipy, y_auto, rtol=tol,
                               atol=tol * max(np.abs(y_auto).max(), 1e-9))


def test_filtfilt_tf_0d_x_raises_value_error():
    b32, a32 = _tf_f32(_TF_FILTERS[1])
    with msig.config_context(gpu_min_size=0):
        with pytest.raises(ValueError, match="at least 1-D"):
            msig.filtfilt(b32, a32, np.float32(1.0))


@pytest.mark.parametrize("ba", _TF_FILTERS)
@pytest.mark.parametrize("shape", [(4096,), (40, 3000), (3, 40, 4000), (2, 200_000)])
def test_lfilter_tf_matches_scipy_f32_exact(rng, ba, shape):
    """Sequential and scan routes both reproduce scipy's f32 recurrence."""
    b32, a32 = _tf_f32(ba)
    x = rng.standard_normal(shape).astype(np.float32)
    ref = sps.lfilter(b32, a32, x)
    out = msig.lfilter(*ba, x)
    assert_type_and_dtype(out)
    assert ref.shape == tuple(np.array(out).shape)
    _assert_tf(out, ref)


def test_lfilter_tf_axis0(rng):
    b32, a32 = _tf_f32(_TF_FILTERS[2])
    x = rng.standard_normal((20000, 6)).astype(np.float32)
    _assert_tf(msig.lfilter(*_TF_FILTERS[2], x, axis=0), sps.lfilter(b32, a32, x, axis=0))


def test_lfilter_tf_a0_not_one(rng):
    """scipy normalizes by a[0] with in-loop f32 division; so do we."""
    b, a = sps.butter(3, 0.2)
    b, a = b * 2.5, a * 2.5
    x = rng.standard_normal((40, 5000)).astype(np.float32)
    _assert_tf(msig.lfilter(b, a, x), sps.lfilter(b.astype(np.float32) , a.astype(np.float32), x))


def test_lfilter_tf_zi_roundtrip(rng):
    b32, a32 = _tf_f32(_TF_FILTERS[3])
    order = max(len(b32), len(a32)) - 1
    x = rng.standard_normal((5, 8000)).astype(np.float32)
    zi = (rng.standard_normal((5, order)) * 0.1).astype(np.float32)
    y_ref, zf_ref = sps.lfilter(b32, a32, x, zi=zi)
    y, zf = msig.lfilter(*_TF_FILTERS[3], x, zi=zi)
    _assert_tf(y, y_ref)
    _assert_tf(zf, zf_ref, tol=1e-5)


def test_lfilter_tf_streaming_chunks_equal_one_shot(rng):
    """Filtering in chunks with carried state must equal one-shot filtering."""
    ba = _TF_FILTERS[2]
    order = max(len(ba[0]), len(ba[1])) - 1
    x = rng.standard_normal((10, 30000)).astype(np.float32)
    one = np.array(msig.lfilter(*ba, x))
    zi = np.zeros((10, order), dtype=np.float32)
    parts = []
    for chunk in np.split(x, 3, axis=-1):
        y, zi = msig.lfilter(*ba, chunk, zi=zi)
        zi = np.array(zi)
        parts.append(np.array(y))
    np.testing.assert_allclose(np.concatenate(parts, axis=-1), one, rtol=1e-5,
                               atol=1e-5 * np.abs(one).max())


def test_lfilter_tf_complex_x(rng):
    """Real coefficients + complex signal run as two f32 planes on every route."""
    b32, a32 = _tf_f32(_TF_FILTERS[2])
    x = (rng.standard_normal((9, 6000))
         + 1j * rng.standard_normal((9, 6000))).astype(np.complex64)
    yr = sps.lfilter(b32, a32, x.real)
    yi = sps.lfilter(b32, a32, x.imag)
    ref = yr.astype(np.complex64) + np.complex64(1j) * yi
    out = msig.lfilter(*_TF_FILTERS[2], x)
    assert np.array(out).dtype == np.complex64
    _assert_tf(out, ref)


def test_lfilter_tf_complex_zi_real_x(rng):
    b32, a32 = _tf_f32(_TF_FILTERS[1])
    order = 2
    x = rng.standard_normal(20000).astype(np.float32)
    zi = (rng.standard_normal(order) + 1j * rng.standard_normal(order)).astype(np.complex64)
    yr, zfr = sps.lfilter(b32, a32, x, zi=zi.real.copy())
    yi, zfi = sps.lfilter(b32, a32, np.zeros_like(x), zi=zi.imag.copy())
    ref_y = yr.astype(np.complex64) + np.complex64(1j) * yi
    y, zf = msig.lfilter(*_TF_FILTERS[1], x, zi=zi)
    assert np.array(y).dtype == np.complex64
    _assert_tf(y, ref_y)


def test_lfilter_tf_narrowband_stays_sequential_and_tracks_scipy(rng):
    """A long-lived filter is scan-unsafe: the sequential kernel (or scipy on
    non-matching builds) runs, and the result still tracks the f32 filter."""
    b, a = sps.butter(2, 1e-3)
    b32, a32 = b.astype(np.float32), a.astype(np.float32)
    x = rng.standard_normal((40, 100_000)).astype(np.float32)
    ref = sps.lfilter(b32, a32, x)
    out = np.array(msig.lfilter(b, a, x))
    tol = 1e-5 if HAS_GPU else 0.0
    np.testing.assert_allclose(out, ref, rtol=tol,
                               atol=tol * max(np.abs(ref).max(), 1e-9))


def test_lfilter_tf_auto_routing_is_filter_invariant(rng):
    """Row-count routing (scipy vs kernel) must never change WHICH filter runs."""
    b, a = sps.butter(2, 1e-3)
    xn = rng.standard_normal(20_000).astype(np.float32)
    with msig.config_context(dispatch="auto", warn_on_downcast=False):
        r31 = np.array(msig.lfilter(b, a, np.tile(xn, (31, 1))))[0]
        r32 = np.array(msig.lfilter(b, a, np.tile(xn, (32, 1))))[0]
    tol = 1e-6 if HAS_GPU else 0.0
    np.testing.assert_allclose(r31, r32, atol=tol * max(np.abs(r32).max(), 1e-9))


def test_lfilter_tf_scan_and_sequential_kernels_agree(rng):
    """The two GPU implementations are independent; they must agree."""
    import mlx.core as mx

    from mlx_signal import _lfilter_metal

    if not mx.metal.is_available():
        pytest.skip("no Metal GPU")
    b, a = sps.cheby1(5, 0.05, 0.3)
    order = 5
    ba = np.zeros((2, order + 1), np.float32)
    ba[0, : len(b)] = (b / a[0]).astype(np.float32)
    ba[1, : len(a)] = (a / a[0]).astype(np.float32)
    x = mx.array(rng.standard_normal((3, 50001)).astype(np.float32))
    zi = mx.array((rng.standard_normal((3, order)) * 0.1).astype(np.float32))
    y_seq, zf_seq = _lfilter_metal.lfilter_gpu(x, mx.array(ba.reshape(-1)), zi)
    y_scan, zf_scan = _lfilter_metal.lfilter_scan_gpu(x, ba, zi)
    np.testing.assert_allclose(np.array(y_scan), np.array(y_seq), rtol=1e-4,
                               atol=1e-5 * float(mx.max(mx.abs(y_seq))))
    np.testing.assert_allclose(np.array(zf_scan), np.array(zf_seq), rtol=1e-4,
                               atol=1e-5)


def test_lfilter_tf_validation(rng):
    x = rng.standard_normal(1000).astype(np.float32)
    with pytest.raises(ValueError, match="nonzero"):
        msig.lfilter([1.0, 0.5], [0.0, 1.0], x)
    with pytest.raises(ValueError, match="1-D"):
        msig.lfilter(np.ones((2, 2)), [1.0, 0.5], x)
    with pytest.raises(ValueError, match="zi"):
        msig.lfilter([1.0, 0.2], [1.0, -0.5], x, zi=np.zeros(3, np.float32))


def test_lfilter_tf_rejects_filter_destabilized_by_f32_quantization():
    """Mirror of the SOS guard: a stable f64 design whose f32 poles cross the
    unit circle must fail identically before any route runs it."""
    b, a = sps.butter(2, 5e-5)
    assert np.max(np.abs(np.roots(a))) < 1.0
    assert np.max(np.abs(np.roots(a.astype(np.float32).astype(np.float64)))) >= 1.0
    for dispatch in ("auto", "scipy"):
        with msig.config_context(dispatch=dispatch, warn_on_downcast=False):
            with pytest.raises(ValueError, match="unstable after float32 quantization"):
                msig.lfilter(b, a, np.ones(256, np.float32))


def test_lfilter_tf_explicit_integrator_still_allowed(rng):
    """Only quantization-caused instability is rejected; a deliberate
    integrator works like scipy."""
    b, a = np.array([1.0, 0.0], np.float32), np.array([1.0, -1.0], np.float32)
    x = rng.standard_normal(3000).astype(np.float32)
    _assert_tf(msig.lfilter(b, a, x), sps.lfilter(b, a, x))


@pytest.mark.parametrize("padtype", ["odd", "even", "constant", None])
def test_filtfilt_tf_padtypes(rng, padtype):
    b32, a32 = _tf_f32(_TF_FILTERS[2])
    x = rng.standard_normal((12, 4000)).astype(np.float32)
    ref = sps.filtfilt(b32, a32, x, padtype=padtype)
    out = msig.filtfilt(*_TF_FILTERS[2], x, padtype=padtype)
    assert ref.shape == tuple(np.array(out).shape)
    tol = 1e-6 if HAS_GPU else 1e-4
    np.testing.assert_allclose(np.array(out), ref, rtol=tol,
                               atol=tol * max(np.abs(ref).max(), 1e-9))


def test_filtfilt_tf_padlen_and_axis(rng):
    b32, a32 = _tf_f32(_TF_FILTERS[3])
    x = rng.standard_normal((3000, 9)).astype(np.float32)
    ref = sps.filtfilt(b32, a32, x, axis=0, padlen=500)
    out = msig.filtfilt(*_TF_FILTERS[3], x, axis=0, padlen=500)
    tol = 1e-6 if HAS_GPU else 1e-4
    np.testing.assert_allclose(np.array(out), ref, rtol=tol,
                               atol=tol * max(np.abs(ref).max(), 1e-9))


def test_filtfilt_tf_padlen_too_long(rng):
    # butter(4) tf -> ntaps 5 -> default edge 15, so 12 samples must raise
    with pytest.raises(ValueError, match="padlen"):
        msig.filtfilt(*_TF_FILTERS[2], rng.standard_normal(12).astype(np.float32))


def test_filtfilt_tf_zero_phase_property(rng):
    x = np.zeros((8, 2048), dtype=np.float32)
    x[:, 1024] = 1.0
    y = np.array(msig.filtfilt(*_TF_FILTERS[2], x))[0]
    assert abs(int(np.argmax(np.abs(y))) - 1024) <= 1


def test_filtfilt_tf_gust_falls_back(rng):
    x = rng.standard_normal(5000).astype(np.float32)
    b, a = sps.butter(3, 0.2)
    with msig.config_context(dispatch="auto", gpu_min_size=1, warn_on_fallback=True):
        with pytest.warns(msig.FallbackWarning, match="gust"):
            out = msig.filtfilt(b, a, x, method="gust")
    ref = sps.filtfilt(b.astype(np.float32), a.astype(np.float32), x, method="gust")
    assert_close(out, ref, rtol=1e-3)
