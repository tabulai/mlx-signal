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


# ---------------------------------------------------------------------------
# fused three-pass kernel pipeline (power-of-two broken lengths)
# ---------------------------------------------------------------------------

from mlx_signal import _fourstep_metal  # noqa: E402


@pytest.mark.gpu
@pytest.mark.parametrize("e", [21, 22, 23])
@pytest.mark.parametrize("inverse", [False, True])
def test_fft3_kernel_matches_numpy(rng, e, inverse):
    """The kernel route (dispatched for pow2 broken lengths) tracks float64
    numpy at the f32 noise floor, batched, both directions."""
    n = 1 << e
    assert _fourstep_metal.eligible(n)
    x = (rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n))).astype(
        np.complex64
    )
    fn = _fourstep._ifft_4step_last if inverse else _fourstep._fft_4step_last
    ref_fn = np.fft.ifft if inverse else np.fft.fft
    got = np.array(fn(mx.array(x)))
    for i in range(2):
        assert _rel(got[i].astype(np.complex128),
                    ref_fn(x[i].astype(np.complex128))) < 2e-6


@pytest.mark.gpu
def test_fft3_kernel_structured_signals(rng):
    """Tones and impulses expose phase errors that noise averages away."""
    n = 1 << 21
    t = np.arange(n)
    tone = np.exp(2j * np.pi * (98765 / n) * t).astype(np.complex64)[None]
    imp = np.zeros((1, n), np.complex64)
    imp[0, 4097] = 1.0
    for x in (tone, imp):
        got = np.array(_fourstep._fft_4step_last(mx.array(x)))[0]
        assert _rel(got.astype(np.complex128),
                    np.fft.fft(x[0].astype(np.complex128))) < 2e-6


@pytest.mark.gpu
def test_fft3_kernel_roundtrip_and_dispatch(rng, monkeypatch):
    """Eligible lengths actually take the kernel route (guards a gate stuck
    False), and forward-inverse roundtrips return the input."""
    calls = []
    real = _fourstep_metal.fft3_last

    def recording(a, inverse=False):
        calls.append(inverse)
        return real(a, inverse=inverse)

    monkeypatch.setattr(_fourstep_metal, "fft3_last", recording)
    n = 1 << 21
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    a = mx.array(x)
    rt = np.array(_fourstep._ifft_4step_last(_fourstep._fft_4step_last(a)))
    assert calls == [False, True], "eligible pow2 length did not dispatch to the kernel"
    assert _rel(rt, x) < 2e-6

    # non-pow2 broken lengths stay on the composed path
    calls.clear()
    x3 = (rng.standard_normal(3 << 20) + 1j * rng.standard_normal(3 << 20)).astype(
        np.complex64
    )
    got = np.array(_fourstep._fft_4step_last(mx.array(x3)))
    assert calls == []
    assert _rel(got.astype(np.complex128),
                np.fft.fft(x3.astype(np.complex128))) < 5e-6


@pytest.mark.gpu
def test_fft3_kernel_more_accurate_than_composed(rng):
    """The three table-free radix-2 passes must stay at least as accurate as
    the composed chain they replace (measured ~3x better)."""
    n = 1 << 22
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    ref = np.fft.fft(x.astype(np.complex128))
    kern = np.array(_fourstep_metal.fft3_last(mx.array(x)[None]))[0]
    n1, n2 = _fourstep._choose_split(n)
    A = mx.array(x).reshape(n2, n1)
    B = mx.fft.fft(mx.swapaxes(A, -2, -1), axis=-1)
    C = mx.fft.fft(mx.swapaxes(B * _fourstep._twiddle(n1, n2), -2, -1), axis=-1)
    comp = np.array(mx.swapaxes(C, -2, -1).reshape(n))
    assert _rel(kern.astype(np.complex128), ref) <= 1.5 * _rel(
        comp.astype(np.complex128), ref
    )
