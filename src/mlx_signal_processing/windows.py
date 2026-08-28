"""Window functions: a cached wrapper over scipy.signal.get_window.

Windows are tiny and computed host-side; caching the float64 NumPy values keeps
scale factors (``win.sum()`` etc.) at full precision while the returned MLX
array is float32 like everything else in the library.
"""

from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np
from scipy.signal import get_window as _sp_get_window

__all__ = ["get_window"]


@functools.lru_cache(maxsize=256)
def _window_np(window, Nx: int, fftbins: bool = True) -> np.ndarray:
    """Cached float64 window values (hashable window spec only)."""
    w = _sp_get_window(window, Nx, fftbins=fftbins)
    return np.ascontiguousarray(w, dtype=np.float64)


def get_window(window, Nx: int, fftbins: bool = True) -> mx.array:
    """Return a window of a given length and type as a float32 MLX array.

    Accepts everything ``scipy.signal.get_window`` accepts: a window name
    (``"hann"``), a ``(name, param)`` tuple (``("kaiser", 8.6)``), or a float
    (kaiser beta).
    """
    key = tuple(window) if isinstance(window, list | tuple) else window
    w = _window_np(key, int(Nx), bool(fftbins))
    return mx.array(w.astype(np.float32))
