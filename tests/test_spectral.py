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
