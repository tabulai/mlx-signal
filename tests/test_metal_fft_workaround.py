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
