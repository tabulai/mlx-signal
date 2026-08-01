import os

import mlx.core as mx
import numpy as np
import pytest

import mlx_signal

# Set MLX_SIGNAL_SIMULATE_NO_METAL=1 to exercise the no-GPU code paths on a
# machine that has Metal (also used by CI).
if os.environ.get("MLX_SIGNAL_SIMULATE_NO_METAL"):
    mx.set_default_device(mx.cpu)
    mx.metal.is_available = lambda: False  # noqa: E731 - deliberate monkeypatch

HAS_GPU = mx.metal.is_available()
if not HAS_GPU:
    # CI runners have no Metal device; the MLX CPU backend still exercises
    # every op except custom metal kernels (marked with @pytest.mark.gpu).
    mx.set_default_device(mx.cpu)

requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="Metal GPU not available")


@pytest.fixture(autouse=True)
def _pin_mlx_path():
    """Golden tests must exercise the MLX path, not silently route to scipy.

    Without Metal, functions whose only MLX path is a custom kernel have no
    GPU implementation at all, so pinning dispatch="mlx" would raise; use
    auto with a zero threshold instead — MLX-CPU where possible, quiet scipy
    for kernel-only capabilities.
    """
    if HAS_GPU:
        ctx = mlx_signal.config_context(dispatch="mlx", warn_on_downcast=False)
    else:
        ctx = mlx_signal.config_context(
            dispatch="auto", gpu_min_size=0, warn_on_downcast=False,
            warn_on_fallback=False,
        )
    with ctx:
        yield


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
