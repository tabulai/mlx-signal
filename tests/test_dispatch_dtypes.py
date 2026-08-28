"""Dispatch routing and dtype-policy contract tests."""

import mlx.core as mx
import numpy as np
import pytest
import scipy.signal as sps

import mlx_signal_processing as msig
from _utils import assert_close


def test_all_dispatch_modes_agree(rng):
    x = rng.standard_normal(4096).astype(np.float32)
    results = {}
    for mode in ("auto", "mlx", "scipy"):
        with msig.config_context(dispatch=mode):
            f, p = msig.welch(x, nperseg=256)
            results[mode] = (np.array(f), np.array(p))
            assert isinstance(p, mx.array) and p.dtype == mx.float32, mode
    np.testing.assert_allclose(results["mlx"][1], results["scipy"][1], rtol=1e-4,
                               atol=1e-5 * results["scipy"][1].max())
    np.testing.assert_allclose(results["auto"][1], results["mlx"][1], rtol=1e-6)


def test_auto_small_input_routes_to_scipy(rng):
    """Below gpu_min_size the result must still be an f32 MLX array."""
    x = rng.standard_normal(512).astype(np.float32)
    with msig.config_context(dispatch="auto", gpu_min_size=1 << 20):
        f, p = msig.welch(x, nperseg=128)
    assert isinstance(p, mx.array) and p.dtype == mx.float32
    f_ref, p_ref = sps.welch(x, nperseg=128)
    assert_close(p, p_ref)


def test_float64_downcast_warns(rng):
    x = rng.standard_normal(65536)  # float64
    with msig.config_context(dispatch="mlx", warn_on_downcast=True):
        with pytest.warns(msig.DowncastWarning):
            f, p = msig.welch(x, nperseg=256)
    assert p.dtype == mx.float32
    f_ref, p_ref = sps.welch(x, nperseg=256)
    assert_close(p, p_ref)


def test_float64_strict_raises(rng):
    x = rng.standard_normal(65536)
    with msig.config_context(dispatch="mlx", float64="strict"):
        with pytest.raises(TypeError, match="float64"):
            msig.welch(x, nperseg=256)


def test_complex128_downcast(rng):
    x = (rng.standard_normal(65536) + 1j * rng.standard_normal(65536))
    with msig.config_context(dispatch="mlx", warn_on_downcast=True):
        with pytest.warns(msig.DowncastWarning):
            out = msig.hilbert(np.abs(x))  # float64 input
    assert out.dtype == mx.complex64


def test_int_input_accepted(rng):
    x = rng.integers(-100, 100, size=70000).astype(np.int32)
    out = msig.fftconvolve(x, x[:100])
    assert out.dtype == mx.float32
    assert_close(out, sps.fftconvolve(x.astype(np.float32), x[:100].astype(np.float32)),
                 rtol=1e-4)


def test_list_input_silent(rng):
    """Python lists convert without a downcast warning (MLX 32-bit defaults)."""
    import warnings

    x = [float(v) for v in rng.standard_normal(100)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", msig.DowncastWarning)
        msig.periodogram(x)


def test_mlx_array_passthrough(rng):
    x = mx.array(rng.standard_normal(65536).astype(np.float32))
    f, p = msig.welch(x, nperseg=1024)
    assert isinstance(p, mx.array)
    f_ref, p_ref = sps.welch(np.array(x), nperseg=1024)
    assert_close(p, p_ref)


def test_config_context_restores():
    base = msig.get_config().dispatch
    with msig.config_context(dispatch="scipy"):
        assert msig.get_config().dispatch == "scipy"
    assert msig.get_config().dispatch == base


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        msig.set_config(dispatch="cuda")
    with pytest.raises(TypeError):
        msig.set_config(nonsense=1)
