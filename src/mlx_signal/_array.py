"""Array conversion and dtype policy: float32/complex64 in, MLX arrays out."""

from __future__ import annotations

import warnings

import mlx.core as mx
import numpy as np

from ._config import DowncastWarning, _config

_MX_FLOAT64 = getattr(mx, "float64", None)
_MX_COMPLEX128 = getattr(mx, "complex128", None)


def _downcast_notice(src: str) -> None:
    if _config.float64 == "strict":
        raise TypeError(
            f"{src} input is not supported with float64='strict' (Metal has no "
            "float64). Cast to float32/complex64, or set "
            "mlx_signal.set_config(float64='downcast')."
        )
    if _config.warn_on_downcast:
        warnings.warn(
            f"{src} input downcast to 32-bit for Metal (no float64 on the GPU); "
            "results are float32-accurate",
            DowncastWarning,
            stacklevel=4,
        )


def to_mlx(x) -> mx.array:
    """Convert input to an MLX array with dtype float32 or complex64.

    Python scalars/lists convert silently; explicit float64/complex128 arrays
    are downcast with a :class:`DowncastWarning` (or raise under the "strict"
    dtype policy).
    """
    if isinstance(x, mx.array):
        d = x.dtype
        if d in (mx.float32, mx.complex64):
            return x
        if _MX_FLOAT64 is not None and d == _MX_FLOAT64:
            _downcast_notice("float64")
            return x.astype(mx.float32)
        if _MX_COMPLEX128 is not None and d == _MX_COMPLEX128:
            _downcast_notice("complex128")
            return x.astype(mx.complex64)
        if d in (mx.float16, mx.bfloat16):
            return x.astype(mx.float32)
        return x.astype(mx.float32)  # integers / bool
    if not isinstance(x, np.ndarray):
        # lists / scalars: let MLX pick its native 32-bit defaults, then normalize
        return to_mlx(mx.array(x))
    if x.ndim == 0:
        # mx.array(np 0-d) yields shape (1,); go through a Python scalar
        return to_mlx(mx.array(x.item()))
    if x.dtype == np.float32 or x.dtype == np.complex64:
        return mx.array(np.ascontiguousarray(x))
    if x.dtype.kind == "f":
        if x.dtype.itemsize > 4:
            _downcast_notice(str(x.dtype))
        return mx.array(np.ascontiguousarray(x, dtype=np.float32))
    if x.dtype.kind == "c":
        _downcast_notice(str(x.dtype))
        return mx.array(np.ascontiguousarray(x, dtype=np.complex64))
    if x.dtype.kind in "iub":
        return mx.array(np.ascontiguousarray(x, dtype=np.float32))
    raise TypeError(f"unsupported input dtype: {x.dtype}")


def check_strict(x) -> None:
    """Raise under float64='strict' for 64-bit signal inputs on ANY dispatch path."""
    if _config.float64 != "strict":
        return
    bad = False
    if isinstance(x, mx.array):
        d = x.dtype
        bad = (_MX_FLOAT64 is not None and d == _MX_FLOAT64) or (
            _MX_COMPLEX128 is not None and d == _MX_COMPLEX128
        )
    elif isinstance(x, np.ndarray):
        bad = (x.dtype.kind == "f" and x.dtype.itemsize > 4) or (
            x.dtype.kind == "c" and x.dtype.itemsize > 8
        )
    if bad:
        raise TypeError(
            "float64/complex128 input is not supported with float64='strict' "
            "(results are float32 on every dispatch path). Cast to "
            "float32/complex64, or set mlx_signal.set_config(float64='downcast')."
        )


def signal_np(x) -> np.ndarray:
    """to_numpy for signal arguments of scipy fallbacks: enforces the strict policy."""
    check_strict(x)
    return to_numpy(x)


def to_numpy(x) -> np.ndarray:
    """Convert MLX or array-like input to a NumPy array (used by scipy fallbacks)."""
    if isinstance(x, mx.array):
        return np.array(x)
    return np.asarray(x)


def result_to_mlx(a) -> mx.array:
    """Convert a scipy-fallback result to MLX with the library's output dtypes.

    Silent by design: the caller chose (or was routed to) the scipy path, and the
    contract is that both paths return identical dtypes.
    """
    a = np.asarray(a)
    if a.dtype.kind == "f" and a.dtype.itemsize > 4:
        a = a.astype(np.float32)
    elif a.dtype.kind == "c" and a.dtype.itemsize > 8:
        a = a.astype(np.complex64)
    elif a.dtype.kind in "iub":
        a = a.astype(np.float32)
    return mx.array(np.ascontiguousarray(a))


def input_size(x) -> int:
    """Number of elements in an array-like, without forcing a conversion."""
    if isinstance(x, mx.array):
        return x.size
    return int(np.size(x))
