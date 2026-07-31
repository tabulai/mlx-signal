import mlx.core as mx
import numpy as np
import pytest

import mlx_signal

HAS_GPU = mx.metal.is_available()
if not HAS_GPU:
    # CI runners have no Metal device; the MLX CPU backend still exercises
    # every op except custom metal kernels (marked with @pytest.mark.gpu).
    mx.set_default_device(mx.cpu)

requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="Metal GPU not available")


@pytest.fixture(autouse=True)
def _pin_mlx_path():
    """Golden tests must exercise the MLX path, not silently route to scipy."""
    with mlx_signal.config_context(dispatch="mlx", warn_on_downcast=False):
        yield


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
