"""Golden tests: spectral functions vs scipy.signal across shapes, modes, windows."""

import numpy as np
import pytest
import scipy.signal as sps

import mlx_signal_processing as msig
from _utils import assert_close, assert_type_and_dtype


def make_sig(rng, shape, complex_=False):
    x = rng.standard_normal(shape).astype(np.float32)
    # add a deterministic tone so spectra have structure, not just noise floor
    n = shape[-1] if isinstance(shape, tuple) else shape
    t = np.arange(n, dtype=np.float32)
    x = x + 0.5 * np.sin(2 * np.pi * 0.12 * t).astype(np.float32)
    if complex_:
        x = (x + 1j * rng.standard_normal(shape)).astype(np.complex64)
    return x


WELCH_CASES = [
    {},
    {"nperseg": 128},
    {"nperseg": 128, "noverlap": 96},
    {"nperseg": 128, "window": "hamming"},
    {"nperseg": 100, "window": ("tukey", 0.25)},
    {"nperseg": 128, "detrend": "linear"},
    {"nperseg": 128, "detrend": False},
    {"nperseg": 128, "scaling": "spectrum"},
    {"nperseg": 128, "return_onesided": False},
    {"nperseg": 128, "average": "median"},
    {"nperseg": 128, "nfft": 256},
    {"nperseg": 129, "noverlap": 37, "fs": 250.0},
]


@pytest.mark.parametrize("n", [1024, 999, 4093])
@pytest.mark.parametrize("case", WELCH_CASES)
def test_welch_matches_scipy(rng, n, case):
    x = make_sig(rng, (n,))
    f_ref, p_ref = sps.welch(x, **case)
    f, p = msig.welch(x, **case)
    assert_type_and_dtype(f)
    assert_type_and_dtype(p)
    assert_close(f, f_ref)
    assert_close(p, p_ref)


def test_welch_array_window(rng):
    x = make_sig(rng, (2048,))
    w = sps.get_window("hann", 200)
    f_ref, p_ref = sps.welch(x, window=w)
    f, p = msig.welch(x, window=w.astype(np.float32))
    assert_close(f, f_ref)
    assert_close(p, p_ref)


@pytest.mark.parametrize(
    "shape,axis",
    [((3, 2000), -1), ((2000, 3), 0), ((2, 3, 1500), -1), ((2, 1500, 3), 1)],
)
def test_welch_multidim(rng, shape, axis):
    x = make_sig(rng, shape)
    f_ref, p_ref = sps.welch(x, nperseg=128, axis=axis)
    f, p = msig.welch(x, nperseg=128, axis=axis)
    assert p_ref.shape == tuple(np.array(p).shape)
    assert_close(p, p_ref)
    assert_close(f, f_ref)


def test_welch_complex_input_switches_twosided(rng):
    x = make_sig(rng, (2048,), complex_=True)
    f_ref, p_ref = sps.welch(x, nperseg=256)
    f, p = msig.welch(x, nperseg=256)
    assert_close(f, f_ref)
    assert_close(p, p_ref)


def test_welch_short_signal_warns(rng):
    x = make_sig(rng, (100,))
    with pytest.warns(UserWarning, match="nperseg"):
        f_ref, p_ref = sps.welch(x)
    with pytest.warns(UserWarning, match="nperseg"):
        f, p = msig.welch(x)
    assert_close(f, f_ref)
    assert_close(p, p_ref)


@pytest.mark.parametrize(
    "case",
    [
        {},
        {"nfft": 4096},
        {"nfft": 1000},
        {"scaling": "spectrum"},
        {"detrend": False},
        {"window": "hann"},
        {"return_onesided": False},
    ],
)
def test_periodogram_matches_scipy(rng, case):
    x = make_sig(rng, (2048,))
    f_ref, p_ref = sps.periodogram(x, **case)
    f, p = msig.periodogram(x, **case)
    assert_close(f, f_ref)
    assert_close(p, p_ref)


def test_periodogram_array_window(rng):
    x = make_sig(rng, (1024,))
    w = sps.get_window("blackman", 1024)
    f_ref, p_ref = sps.periodogram(x, window=w)
    f, p = msig.periodogram(x, window=w)
    assert_close(p, p_ref)


@pytest.mark.parametrize("average", ["mean", "median"])
def test_csd_matches_scipy(rng, average):
    x = make_sig(rng, (4096,))
    y = make_sig(rng, (4096,))
    f_ref, p_ref = sps.csd(x, y, nperseg=256, average=average)
    f, p = msig.csd(x, y, nperseg=256, average=average)
    assert isinstance(p_ref[0], complex | np.complexfloating)
    assert_close(f, f_ref)
    assert_close(p, p_ref, rtol=2e-4)


def test_csd_broadcast(rng):
    x = make_sig(rng, (4096,))
    y = make_sig(rng, (3, 4096))
    f_ref, p_ref = sps.csd(x, y, nperseg=256)
    f, p = msig.csd(x, y, nperseg=256)
    assert p_ref.shape == tuple(np.array(p).shape)
    assert_close(p, p_ref, rtol=2e-4)


def test_csd_unequal_lengths(rng):
    x = make_sig(rng, (3000,))
    y = make_sig(rng, (4096,))
    f_ref, p_ref = sps.csd(x, y, nperseg=256)
    f, p = msig.csd(x, y, nperseg=256)
    assert_close(p, p_ref, rtol=2e-4)


def test_coherence_matches_scipy(rng):
    x = make_sig(rng, (8192,))
    y = 0.5 * x + 0.5 * make_sig(rng, (8192,))
    f_ref, c_ref = sps.coherence(x, y, nperseg=256)
    f, c = msig.coherence(x, y, nperseg=256)
    assert_close(c, c_ref, rtol=1e-3)


@pytest.mark.parametrize("nperseg", [64, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("detrend", [False, "constant"])
def test_grouped_spectral_means_match_per_frame_and_scipy(rng, nperseg, detrend):
    """Grouped-frame Metal accumulation for welch/csd/coherence must agree with
    both the per-frame kernels and scipy across all eligible FFT sizes."""
    import mlx.core as mx

    from mlx_signal_processing import _stft_metal

    if not mx.metal.is_available():
        pytest.skip("Metal GPU not available")

    # Non-multiple of 64 segments so the trailing partial group is exercised
    n = nperseg + (nperseg // 2) * 130
    x = make_sig(rng, (16, n))
    y = 0.5 * x + 0.5 * make_sig(rng, (16, n))

    _, pw = msig.welch(x, nperseg=nperseg, detrend=detrend)
    _, pw_ref = sps.welch(x, nperseg=nperseg, detrend=detrend)
    assert_close(pw, pw_ref, rtol=1e-4)

    _, pc = msig.csd(x, y, nperseg=nperseg, detrend=detrend)
    _, pc_ref = sps.csd(x, y, nperseg=nperseg, detrend=detrend)
    assert_close(pc, pc_ref, rtol=2e-4)

    _, co = msig.coherence(x, y, nperseg=nperseg, detrend=detrend)
    _, co_ref = sps.coherence(x, y, nperseg=nperseg, detrend=detrend)
    assert_close(co, co_ref, rtol=1e-3)

    # Also cross-check grouped helpers directly against per-frame kernels
    xm, ym = mx.array(x), mx.array(y)
    win = mx.array(sps.get_window("hann", nperseg).astype(np.float32))
    hop = nperseg // 2
    p_grp = _stft_metal.rfft_power_mean(xm, win, nperseg, hop, detrend)
    p_frm = mx.mean(_stft_metal.rfft_frames(xm, win, nperseg, hop, detrend, power=True), axis=-2)
    assert_close(p_grp, np.array(p_frm), rtol=1e-5, atol_frac=1e-6)

    c_grp = _stft_metal.csd_mean(xm, ym, win, nperseg, hop, detrend)
    c_frm = mx.mean(_stft_metal.csd_frames(xm, ym, win, nperseg, hop, detrend), axis=-2)
    assert_close(c_grp, np.array(c_frm), rtol=1e-5, atol_frac=1e-6)

    pxx_g, pyy_g, pxy_g = _stft_metal.coherence_mean(xm, ym, win, nperseg, hop, detrend)
    pxx_f, pyy_f, pxy_f = _stft_metal.coherence_frames(xm, ym, win, nperseg, hop, detrend)
    assert_close(pxx_g, np.array(mx.mean(pxx_f, axis=-2)), rtol=1e-5, atol_frac=1e-6)
    assert_close(pyy_g, np.array(mx.mean(pyy_f, axis=-2)), rtol=1e-5, atol_frac=1e-6)
    assert_close(pxy_g, np.array(mx.mean(pxy_f, axis=-2)), rtol=1e-5, atol_frac=1e-6)


def test_grouped_spectral_single_group_and_nonlast_axis(rng):
    """Covers ngroups==1 with nseg>1 (high channel count) and non-default axes."""
    x_many = make_sig(rng, (2048, 320))  # nperseg=64, hop=32 -> nseg=9, ngroups=1
    y_many = 0.5 * x_many + 0.5 * make_sig(rng, (2048, 320))
    assert_close(msig.welch(x_many, nperseg=64)[1], sps.welch(x_many, nperseg=64)[1], rtol=1e-4)
    assert_close(
        msig.coherence(x_many, y_many, nperseg=64)[1],
        sps.coherence(x_many, y_many, nperseg=64)[1],
        rtol=1e-3,
    )

    # Non-last axis for welch, csd, coherence
    x_ax = make_sig(rng, (2, 4096, 3))
    y_ax = 0.5 * x_ax + 0.5 * make_sig(rng, (2, 4096, 3))
    assert_close(
        msig.welch(x_ax, nperseg=128, axis=1)[1],
        sps.welch(x_ax, nperseg=128, axis=1)[1],
        rtol=1e-4,
    )
    assert_close(
        msig.csd(x_ax, y_ax, nperseg=128, axis=1)[1],
        sps.csd(x_ax, y_ax, nperseg=128, axis=1)[1],
        rtol=2e-4,
    )
    assert_close(
        msig.coherence(x_ax, y_ax, nperseg=128, axis=1)[1],
        sps.coherence(x_ax, y_ax, nperseg=128, axis=1)[1],
        rtol=1e-3,
    )


def test_grouped_spectral_reduces_peak_memory(rng):
    """Grouped accumulation avoids materializing per-frame spectral tensors."""
    import mlx.core as mx

    if not mx.metal.is_available():
        pytest.skip("Metal GPU not available")

    # 64ch x 2^19, nperseg=1024 -> nseg=1023 -> ungrouped coherence would be ~512 MiB
    x = mx.array(rng.standard_normal((64, 1 << 19)).astype(np.float32))
    y = mx.array(rng.standard_normal((64, 1 << 19)).astype(np.float32))
    mx.eval(x, y)

    # Warm kernel caches before measuring allocation delta
    mx.eval(*msig.welch(x, nperseg=1024))
    mx.eval(*msig.coherence(x, y, nperseg=1024))
    mx.synchronize()

    base = mx.get_active_memory()
    mx.reset_peak_memory()
    mx.eval(*msig.welch(x, nperseg=1024))
    mx.synchronize()
    welch_mib = (mx.get_peak_memory() - base) / (1024 * 1024)

    base = mx.get_active_memory()
    mx.reset_peak_memory()
    mx.eval(*msig.coherence(x, y, nperseg=1024))
    mx.synchronize()
    coh_mib = (mx.get_peak_memory() - base) / (1024 * 1024)

    assert welch_mib < 16.0, f"welch peak memory too high: {welch_mib:.1f} MiB"
    assert coh_mib < 64.0, f"coherence peak memory too high: {coh_mib:.1f} MiB"


@pytest.mark.parametrize("mode", ["psd", "complex", "magnitude", "angle", "phase"])
def test_spectrogram_matches_scipy(rng, mode):
    x = make_sig(rng, (4096,))
    f_ref, t_ref, s_ref = sps.spectrogram(x, fs=100.0, mode=mode)
    f, t, s = msig.spectrogram(x, fs=100.0, mode=mode)
    assert s_ref.shape == tuple(np.array(s).shape)
    assert_close(f, f_ref)
    assert_close(t, t_ref)
    if mode in ("angle", "phase"):
        # angles at negligible-magnitude bins are numerically meaningless in
        # fp32; compare only where there is signal
        mag = np.abs(sps.spectrogram(x, fs=100.0, mode="complex")[2])
        keep = mag > 1e-4 * mag.max()
        a = np.array(s)[keep]
        d = s_ref[keep]
        # angles wrap at +-pi and unwrap offsets can differ by 2*pi steps at
        # near-zero bins; compare mod 2*pi
        diff = np.angle(np.exp(1j * (a - d)))
        np.testing.assert_allclose(diff, np.zeros_like(diff), atol=2e-3)
    else:
        assert_close(s, s_ref, rtol=2e-4)


def test_spectrogram_custom_params(rng):
    x = make_sig(rng, (4096,))
    kw = dict(nperseg=200, noverlap=50, nfft=256, scaling="spectrum", detrend="linear")
    f_ref, t_ref, s_ref = sps.spectrogram(x, **kw)
    f, t, s = msig.spectrogram(x, **kw)
    assert_close(s, s_ref)


STFT_CASES = [
    {},
    {"nperseg": 128},
    {"nperseg": 128, "noverlap": 100},
    {"nperseg": 128, "nfft": 256},
    {"boundary": "even"},
    {"boundary": "odd"},
    {"boundary": "constant"},
    {"boundary": None},
    {"padded": False},
    {"scaling": "psd"},
    {"detrend": "constant"},
    {"return_onesided": False},
]


@pytest.mark.parametrize("case", STFT_CASES)
def test_stft_matches_scipy(rng, case):
    x = make_sig(rng, (4000,))
    f_ref, t_ref, z_ref = sps.stft(x, fs=8000.0, **case)
    f, t, z = msig.stft(x, fs=8000.0, **case)
    assert z_ref.shape == tuple(np.array(z).shape)
    assert_close(f, f_ref)
    assert_close(t, t_ref)
    assert_close(z, z_ref)


def test_stft_multidim_axis0(rng):
    x = make_sig(rng, (3000, 4))
    f_ref, t_ref, z_ref = sps.stft(x, axis=0)
    f, t, z = msig.stft(x, axis=0)
    assert z_ref.shape == tuple(np.array(z).shape)
    assert_close(z, z_ref)


def test_stft_complex_input(rng):
    x = make_sig(rng, (2048,), complex_=True)
    with pytest.warns(UserWarning):
        f_ref, t_ref, z_ref = sps.stft(x)
    f, t, z = msig.stft(x)
    assert_close(z, z_ref)


ISTFT_CASES = [
    {},
    {"boundary": False},
    {"scaling": "psd"},
    {"nperseg": 256, "noverlap": 192},
    {"window": "hamming"},
]


@pytest.mark.parametrize("case", ISTFT_CASES)
def test_istft_matches_scipy(rng, case):
    x = make_sig(rng, (4000,))
    stft_kw = {k: case[k] for k in ("nperseg", "noverlap", "window") if k in case}
    if case.get("scaling") == "psd":
        stft_kw["scaling"] = "psd"
    if case.get("boundary") is False:
        stft_kw["boundary"] = None
    f, t, z = sps.stft(x, **stft_kw)
    if case.get("boundary") is False:
        # without boundary extension NOLA fails at the signal edges (scipy
        # warns); edge samples are divided by ~0 and are numerically
        # meaningless, so compare the interior only
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            t_ref, x_ref = sps.istft(z, **case)
            t2, x2 = msig.istft(z, **case)
        assert x_ref.shape == tuple(np.array(x2).shape)
        assert_close(np.array(x2)[128:-128], x_ref[128:-128])
    else:
        t_ref, x_ref = sps.istft(z, **case)
        t2, x2 = msig.istft(z, **case)
        assert x_ref.shape == tuple(np.array(x2).shape)
        assert_close(x2, x_ref)
    assert_close(t2, t_ref)


def test_istft_twosided(rng):
    x = make_sig(rng, (2048,))
    _, _, z = sps.stft(x, return_onesided=False)
    t_ref, x_ref = sps.istft(z, input_onesided=False)
    t2, x2 = msig.istft(z, input_onesided=False)
    assert_close(np.array(x2).real, x_ref.real)


def test_istft_batched(rng):
    x = make_sig(rng, (3, 4000))
    _, _, z = sps.stft(x)
    t_ref, x_ref = sps.istft(z)
    t2, x2 = msig.istft(z)
    assert x_ref.shape == tuple(np.array(x2).shape)
    assert_close(x2, x_ref)


def test_istft_memory_layouts_bit_identical(rng):
    """Equivalent NumPy/MLX layouts must produce identical ISTFT outputs."""
    import mlx.core as mx

    x = make_sig(rng, (2, 3, 2048))
    _, _, z_sp = sps.stft(x, nperseg=256)
    z_sp = z_sp.astype(np.complex64)
    _, x_ref = sps.istft(z_sp, nperseg=256)
    t_base, y_base = msig.istft(z_sp, nperseg=256)
    t_base_np, y_base_np = np.array(t_base), np.array(y_base)
    assert_close(y_base_np, x_ref)

    z_ro = z_sp.copy()
    z_ro.setflags(write=False)

    variants = [
        (np.ascontiguousarray(z_sp), {}),
        (np.ascontiguousarray(np.swapaxes(z_sp, -2, -1)), {"time_axis": -2, "freq_axis": -1}),
        (np.asfortranarray(z_sp), {}),
        (np.transpose(z_sp, (2, 0, 1, 3)), {"freq_axis": 0, "time_axis": -1}),
        (z_sp.astype(np.complex128), {}),
        (z_ro, {}),
        (mx.array(z_sp), {}),
        (mx.array(np.ascontiguousarray(z_sp)), {}),
    ]
    for arr, kw in variants:
        t_got, y_got = msig.istft(arr, nperseg=256, **kw)
        np.testing.assert_array_equal(np.array(t_got), t_base_np)
        np.testing.assert_array_equal(np.array(y_got), y_base_np)


def test_istft_strided_reversed_and_broadcast_views(rng):
    """Non-contiguous sliced, reversed, and broadcast NumPy views match scipy."""
    x = make_sig(rng, (3, 2048))
    _, _, z = sps.stft(x, nperseg=256)
    z = z.astype(np.complex64)

    for view in (
        z[:, :, ::2],
        z[:, :, ::-1],
        np.ascontiguousarray(z)[:, :, ::-1],
        np.broadcast_to(z[:1], (4,) + z.shape[1:]),
    ):
        _, x_ref = sps.istft(view, nperseg=256)
        _, x_got = msig.istft(view, nperseg=256)
        assert tuple(x_got.shape) == x_ref.shape
        assert_close(x_got, x_ref)


@pytest.mark.parametrize(
    "perm,freq_axis,time_axis",
    [
        ((0, 1, 2, 3), -2, -1),
        ((0, 1, 3, 2), -1, -2),
        ((2, 0, 1, 3), 0, -1),
        ((2, 3, 0, 1), 0, 1),
        ((3, 2, 0, 1), 1, 0),
        ((0, 3, 1, 2), 3, 1),
        ((0, 2, 3, 1), 1, 2),
        ((0, 3, 2, 1), 2, 1),
    ],
)
@pytest.mark.parametrize("nperseg", [256, 200])
def test_istft_arbitrary_axes(rng, perm, freq_axis, time_axis, nperseg):
    """Both Metal (pow2) and generic (non-pow2) paths handle all axis orders."""
    x = make_sig(rng, (2, 3, 1024))
    _, _, z = sps.stft(x, nperseg=nperseg)
    z = np.transpose(z.astype(np.complex64), perm)

    _, x_ref = sps.istft(z, nperseg=nperseg, freq_axis=freq_axis, time_axis=time_axis)
    t_got, x_got = msig.istft(z, nperseg=nperseg, freq_axis=freq_axis, time_axis=time_axis)
    assert tuple(x_got.shape) == x_ref.shape
    out_time_ax = (time_axis % z.ndim) - (1 if (freq_axis % z.ndim) < (time_axis % z.ndim) else 0)
    assert len(t_got) == x_got.shape[out_time_ax]
    assert_close(x_got, x_ref)


def test_istft_numpy_layout_zero_copy(rng):
    """SciPy STFT layouts transfer zero-copy directly in frame-major order."""
    from mlx_signal_processing.spectral import _istft_axes, _istft_numpy_layout

    x = make_sig(rng, (4, 1024))
    _, _, z_last = sps.stft(x, nperseg=128)
    z_last = z_last.astype(np.complex64)
    _, _, perm = _istft_axes(z_last.shape, -2, -1)
    z_in, gpu_perm = _istft_numpy_layout(z_last, perm)
    assert np.shares_memory(z_in, z_last)
    assert z_in.flags.c_contiguous
    assert gpu_perm is None

    _, _, z_ax0 = sps.stft(np.ascontiguousarray(x.T), nperseg=128, axis=0)
    z_ax0 = z_ax0.astype(np.complex64)
    _, _, perm0 = _istft_axes(z_ax0.shape, 0, -1)
    z_in0, gpu_perm0 = _istft_numpy_layout(z_ax0, perm0)
    assert np.shares_memory(z_in0, z_ax0)
    assert z_in0.flags.c_contiguous
    assert gpu_perm0 is None

    z_c = np.ascontiguousarray(z_last)
    z_in_c, gpu_perm_c = _istft_numpy_layout(z_c, perm)
    assert z_in_c is z_c
    assert gpu_perm_c == perm

    z_f = np.asfortranarray(z_last)
    z_in_f, gpu_perm_f = _istft_numpy_layout(z_f, perm)
    assert np.shares_memory(z_in_f, z_f)
    assert z_in_f.flags.c_contiguous
    assert gpu_perm_f is not None


def test_istft_axis_and_dtype_validation(rng):
    x = make_sig(rng, (2, 1024))
    _, _, z = sps.stft(x, nperseg=128)
    z = z.astype(np.complex64)

    with pytest.raises(ValueError, match="at least 2d"):
        msig.istft(np.ones(16, dtype=np.complex64))
    with pytest.raises(ValueError, match="differing"):
        msig.istft(z, freq_axis=-1, time_axis=-1)
    with pytest.raises(ValueError, match="differing"):
        msig.istft(z, freq_axis=1, time_axis=-2)
    with pytest.raises(IndexError):
        msig.istft(z, freq_axis=5, time_axis=-1)

    z128 = z.astype(np.complex128)
    with msig.config_context(float64="strict"):
        with pytest.raises(TypeError, match="strict"):
            msig.istft(z128, nperseg=128)
    with msig.config_context(float64="downcast", warn_on_downcast=True):
        with pytest.warns(msig.DowncastWarning):
            msig.istft(z128, nperseg=128)


def test_stft_istft_roundtrip_ours(rng):
    x = make_sig(rng, (4096,))
    f, t, z = msig.stft(x, nperseg=256)
    t2, x2 = msig.istft(z, nperseg=256)
    x2 = np.array(x2)[: len(x)]
    np.testing.assert_allclose(x2, x, rtol=1e-4, atol=1e-4)


def test_parseval_periodogram(rng):
    """Integral of the PSD equals mean power (Parseval), detrend off."""
    x = make_sig(rng, (4096,))
    fs = 10.0
    f, p = msig.periodogram(x, fs=fs, detrend=False)
    p = np.array(p)
    total = np.sum(p) * fs / len(x)
    np.testing.assert_allclose(total, np.mean(x.astype(np.float64) ** 2), rtol=1e-3)
