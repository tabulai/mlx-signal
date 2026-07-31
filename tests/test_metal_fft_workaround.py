"""Regression tests for the MLX 0.32 Metal FFT size hole ((2^19, 2^21] and 2^22).

These sizes crash or fail to compile on the Metal FFT backend; mlx-signal must
route them through the CPU stream (direct transforms) or blocked overlap-add
(convolutions) transparently.
"""

import numpy as np
import pytest
import scipy.signal as sps

import mlx_signal as msig
from _utils import assert_close


def test_fftconvolve_padded_size_in_hole_uses_blocking(rng):
    # 2^20 signal x 4097 taps -> padded FFT would be 2^21 (broken); the blocked
    # path must engage and stay correct
    a = rng.standard_normal(1 << 20).astype(np.float32)
    b = rng.standard_normal(4097).astype(np.float32)
    ref = sps.fftconvolve(a, b)
    out = msig.fftconvolve(a, b)
    assert ref.shape == tuple(np.array(out).shape)
    assert_close(out, ref, rtol=2e-4, atol_frac=3e-5)


def test_fftconvolve_two_long_inputs_in_hole(rng):
    # both inputs long -> blocking not applicable; CPU-stream FFT must carry it
    a = rng.standard_normal(600_000).astype(np.float32)
    b = rng.standard_normal(600_000).astype(np.float32)
    ref = sps.fftconvolve(a, b)
    out = msig.fftconvolve(a, b)
    assert_close(out, ref, rtol=2e-4, atol_frac=3e-5)


@pytest.mark.parametrize("n", [600_000, 1 << 21])
def test_resample_in_hole(rng, n):
    x = rng.standard_normal(n).astype(np.float32)
    num = n // 2
    ref = sps.resample(x, num)
    out = msig.resample(x, num)
    assert_close(out, ref, rtol=2e-4, atol_frac=3e-5)


def test_hilbert_in_hole(rng):
    x = rng.standard_normal(600_000).astype(np.float32)
    ref = sps.hilbert(x)
    out = msig.hilbert(x)
    assert_close(out, ref, rtol=2e-4, atol_frac=3e-5)


def test_welch_large_nfft_in_hole(rng):
    x = rng.standard_normal(1 << 21).astype(np.float32)
    f_ref, p_ref = sps.welch(x, nperseg=1 << 20, nfft=1 << 21)
    f, p = msig.welch(x, nperseg=1 << 20, nfft=1 << 21)
    assert_close(p, p_ref, rtol=2e-4, atol_frac=3e-5)


def test_broken_predicate_boundaries():
    import mlx.core as mx

    from mlx_signal._fft_core import metal_fft_broken

    if not mx.metal.is_available():
        pytest.skip("CPU backend has no broken sizes")
    # trusted: <= 2^19 and exactly 2^20; everything else crashes (2^21, 2^22,
    # non-pow2 in (2^19, 2^21)) or silently returns wrong values (> 2^20)
    assert not metal_fft_broken(1 << 19)
    assert not metal_fft_broken(1 << 20)
    assert metal_fft_broken((1 << 19) + 1)
    assert metal_fft_broken((1 << 20) + 1)
    assert metal_fft_broken(1 << 21)
    assert metal_fft_broken(1 << 22)
    assert metal_fft_broken(1 << 23)
    assert metal_fft_broken(9_000_000)


def test_large_fft_values_correct_via_wrappers(rng):
    """Sizes where raw Metal FFT returns garbage must be correct through us."""
    import mlx.core as mx

    from mlx_signal import _fft_core

    x = rng.standard_normal(1 << 21).astype(np.float32)
    ref = np.fft.rfft(x.astype(np.float64))
    got = np.array(_fft_core.rfft(mx.array(x)))
    rel = np.max(np.abs(got - ref)) / np.max(np.abs(ref))
    assert rel < 1e-4

    x2 = rng.standard_normal(3_000_000).astype(np.float32)
    ref2 = np.fft.rfft(x2.astype(np.float64))
    got2 = np.array(_fft_core.rfft(mx.array(x2)))
    rel2 = np.max(np.abs(got2 - ref2)) / np.max(np.abs(ref2))
    assert rel2 < 1e-4


@pytest.mark.gpu
def test_fused_stft_kernel_matches_composed_path(rng):
    """The Stockham kernel and the as_strided+rfft path are independent."""
    import mlx.core as mx

    from mlx_signal import _stft_metal
    from mlx_signal.spectral import _detrend_and_window, _frame_view

    if not (mx.metal.is_available() and hasattr(mx, "view")):
        pytest.skip("no Metal GPU")

    for n, N, hop, detrend in [
        (5000, 256, 100, False),
        (5000, 256, 100, "constant"),
        (40000, 1024, 512, "constant"),
        (40000, 2048, 320, False),
        (3000, 64, 7, "constant"),
    ]:
        x = mx.array(rng.standard_normal((3, n)).astype(np.float32))
        win = mx.array(rng.standard_normal(N).astype(np.float32))
        nseg = (n - N) // hop + 1

        composed = mx.fft.rfft(
            _detrend_and_window(_frame_view(x, nseg, N, hop), detrend, win), axis=-1
        )
        fused = _stft_metal.rfft_frames(x, win, N, hop, detrend, power=False)
        np.testing.assert_allclose(
            np.array(fused), np.array(composed), rtol=1e-4,
            atol=1e-5 * float(mx.max(mx.abs(composed))),
        )
        fused_p = _stft_metal.rfft_frames(x, win, N, hop, detrend, power=True)
        ref_p = np.abs(np.array(composed)) ** 2
        np.testing.assert_allclose(
            np.array(fused_p), ref_p, rtol=1e-4, atol=1e-5 * ref_p.max()
        )


@pytest.mark.gpu
def test_istft_kernel_matches_composed_path(rng, monkeypatch):
    """The inverse-Stockham + gather-OLA pair vs the scatter-based mx path."""
    import mlx.core as mx

    import mlx_signal as msig
    from mlx_signal import _stft_metal

    if not _stft_metal.eligible_istft(1024):
        pytest.skip("no Metal GPU")

    for nperseg, noverlap, shape in [(1024, 512, (4000,)), (256, 192, (3, 8000)),
                                     (512, 256, (2, 3, 6000))]:
        x = rng.standard_normal(shape).astype(np.float32)
        _, _, z = sps.stft(x, nperseg=nperseg, noverlap=noverlap)
        t_k, x_k = msig.istft(z, nperseg=nperseg, noverlap=noverlap)
        with monkeypatch.context() as mp:
            mp.setattr(_stft_metal, "eligible_istft", lambda n: False)
            t_c, x_c = msig.istft(z, nperseg=nperseg, noverlap=noverlap)
        np.testing.assert_allclose(
            np.array(x_k), np.array(x_c), rtol=1e-4,
            atol=1e-5 * float(mx.max(mx.abs(x_c))),
        )
