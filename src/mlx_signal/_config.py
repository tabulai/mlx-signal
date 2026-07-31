"""Global configuration: dispatch policy, dtype policy, and fallback warnings.

Dispatch modes
--------------
- ``"auto"`` (default): route to the MLX/Metal path when the amount of work is
  above ``gpu_min_size`` elements, otherwise use scipy.signal (kernel-launch
  overhead makes the GPU a net loss on tiny inputs). Size-based routing is
  silent by design; both paths return MLX arrays with identical dtypes.
- ``"mlx"``: always use the MLX path. Calls the MLX path cannot handle
  (e.g. IIR filtering) raise ``NotImplementedError`` instead of falling back —
  useful for tests and for pinning the GPU path.
- ``"scipy"``: always use scipy.signal (still returns MLX arrays).
"""

from __future__ import annotations

import contextlib
import dataclasses
import warnings

__all__ = [
    "Config",
    "DowncastWarning",
    "FallbackWarning",
    "config_context",
    "get_config",
    "set_config",
]

_DISPATCH_MODES = ("auto", "mlx", "scipy")
_FLOAT64_MODES = ("downcast", "strict")


class FallbackWarning(UserWarning):
    """A call was routed to scipy.signal because the MLX path does not cover it."""


class DowncastWarning(UserWarning):
    """float64/complex128 input was downcast to float32/complex64 (Metal has no fp64)."""


@dataclasses.dataclass
class Config:
    dispatch: str = "auto"
    #: elements of work below which "auto" dispatch uses scipy
    gpu_min_size: int = 1 << 15
    float64: str = "downcast"  # or "strict" (raise on float64 input)
    warn_on_downcast: bool = True
    warn_on_fallback: bool = True


_config = Config()


def get_config() -> Config:
    """Return a copy of the current global configuration."""
    return dataclasses.replace(_config)


def set_config(**kwargs) -> None:
    """Update global configuration options.

    Examples
    --------
    >>> import mlx_signal
    >>> mlx_signal.set_config(dispatch="mlx", warn_on_downcast=False)
    """
    for key, value in kwargs.items():
        if not hasattr(_config, key):
            raise TypeError(f"unknown config option {key!r}")
        setattr(_config, key, value)
    if _config.dispatch not in _DISPATCH_MODES:
        raise ValueError(f"dispatch must be one of {_DISPATCH_MODES}")
    if _config.float64 not in _FLOAT64_MODES:
        raise ValueError(f"float64 must be one of {_FLOAT64_MODES}")


@contextlib.contextmanager
def config_context(**kwargs):
    """Temporarily override configuration options within a ``with`` block."""
    old = dataclasses.asdict(_config)
    try:
        set_config(**kwargs)
        yield
    finally:
        _config.__dict__.update(old)


def use_mlx(work_size: int) -> bool:
    """Decide whether the MLX path should run for ``work_size`` elements of work."""
    if _config.dispatch == "mlx":
        return True
    if _config.dispatch == "scipy":
        return False
    return work_size >= _config.gpu_min_size


def capability_fallback(func_name: str, reason: str) -> None:
    """Record a scipy fallback taken because the MLX path can't handle the call.

    Under ``dispatch="mlx"`` this raises ``NotImplementedError`` instead, so the
    GPU path can be pinned; otherwise it emits a :class:`FallbackWarning` (unlike
    size-based routing, capability fallbacks are loud so nobody ships a pipeline
    believing it runs on the GPU when it doesn't).
    """
    if _config.dispatch == "mlx":
        raise NotImplementedError(
            f"mlx_signal.{func_name}: {reason} has no MLX path; "
            "use dispatch='auto' to allow the scipy fallback"
        )
    if _config.warn_on_fallback:
        warnings.warn(
            f"mlx_signal.{func_name}: {reason}; falling back to scipy.signal",
            FallbackWarning,
            stacklevel=3,
        )
