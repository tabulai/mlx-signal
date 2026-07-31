"""Goldens for the four-step FFT decomposition (lengths MLX Metal can't run)."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_signal import _fft_core, _fourstep

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="CPU backend needs no four-step path"
)

# broken-zone lengths with different factor structures
SIZES = [1 << 21, 1 << 22, (1 << 23), 3 * (1 << 21), 600_000, 1_536_000]


def _rel(got, ref):
    return np.max(np.abs(got - ref)) / max(np.max(np.abs(ref)), 1e-30)


@pytest.mark.parametrize("n", SIZES)
def test_fft_ifft_large(rng, n):
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    ref = np.fft.fft(x.astype(np.complex128))
    got = np.array(_fft_core.fft(mx.array(x)))
    assert _rel(got, ref) < 5e-6

    back = np.array(_fft_core.ifft(mx.array(got.astype(np.complex64))))
    assert _rel(back, x) < 5e-6


@pytest.mark.parametrize("n", SIZES)
def test_rfft_irfft_large(rng, n):
    x = rng.standard_normal(n).astype(np.float32)
    ref = np.fft.rfft(x.astype(np.float64))
    got = np.array(_fft_core.rfft(mx.array(x)))
    assert got.shape == ref.shape
    assert _rel(got, ref) < 5e-6

    back = np.array(_fft_core.irfft(mx.array(got.astype(np.complex64)), n=n))
    assert _rel(back, x) < 5e-6


def test_rfft_odd_length_in_hole(rng):
    n = 999_999  # odd, composite: full-complex path
    x = rng.standard_normal(n).astype(np.float32)
    ref = np.fft.rfft(x.astype(np.float64))
    got = np.array(_fft_core.rfft(mx.array(x)))
    assert _rel(got, ref) < 5e-6
    back = np.array(_fft_core.irfft(mx.array(got.astype(np.complex64)), n=n))
    assert _rel(back, x) < 5e-6


def test_batched_and_axis(rng):
    n = 1 << 21
    x = rng.standard_normal((3, n)).astype(np.float32)
    ref = np.fft.rfft(x.astype(np.float64), axis=-1)
    got = np.array(_fft_core.rfft(mx.array(x), axis=-1))
    assert _rel(got, ref) < 5e-6

    xt = np.ascontiguousarray(x.T)  # (n, 3), transform along axis 0
    ref0 = np.fft.rfft(xt.astype(np.float64), axis=0)
    got0 = np.array(_fft_core.rfft(mx.array(xt), axis=0))
    assert got0.shape == ref0.shape
    assert _rel(got0, ref0) < 5e-6


def test_padding_semantics(rng):
    x = rng.standard_normal(1_500_000).astype(np.float32)
    n = 1 << 21  # pad up
    ref = np.fft.rfft(x.astype(np.float64), n=n)
    got = np.array(_fft_core.rfft(mx.array(x), n=n))
    assert _rel(got, ref) < 5e-6


def test_prime_length_falls_back_to_cpu_stream(rng):
    n = 999_983  # prime, in the crash zone: no split exists
    assert _fourstep._choose_split(n) is None
    x = rng.standard_normal(n).astype(np.float32)
    ref = np.fft.rfft(x.astype(np.float64))
    got = np.array(_fft_core.rfft(mx.array(x)))  # must still be correct (CPU stream)
    assert _rel(got, ref) < 5e-6


def test_split_chooser_properties():
    for n in SIZES:
        n1, n2 = _fourstep._choose_split(n)
        assert n1 * n2 == n
        assert not _fft_core.metal_fft_broken(n1)
        assert not _fft_core.metal_fft_broken(n2)
