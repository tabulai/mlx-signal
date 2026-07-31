"""Signal-edge extensions along the last axis (mirrors scipy.signal._arraytools)."""

from __future__ import annotations

import mlx.core as mx


def _check_ext_len(x: mx.array, n: int) -> None:
    if n > x.shape[-1] - 1:
        raise ValueError(
            f"The extension length n ({n}) is too big. It must not exceed "
            f"x.shape[-1]-1, which is {x.shape[-1] - 1}."
        )


def even_ext(x: mx.array, n: int) -> mx.array:
    """Even (mirror, edge not repeated) extension: [x[n]..x[1], x, x[-2]..x[-n-1]]."""
    if n < 1:
        return x
    _check_ext_len(x, n)
    left = x[..., n:0:-1]
    right = x[..., -2 : -(n + 2) : -1]
    return mx.concatenate([left, x, right], axis=-1)


def odd_ext(x: mx.array, n: int) -> mx.array:
    """Odd (antisymmetric about the edge values) extension."""
    if n < 1:
        return x
    _check_ext_len(x, n)
    left = 2 * x[..., :1] - x[..., n:0:-1]
    right = 2 * x[..., -1:] - x[..., -2 : -(n + 2) : -1]
    return mx.concatenate([left, x, right], axis=-1)


def const_ext(x: mx.array, n: int) -> mx.array:
    """Constant (repeat edge value) extension."""
    if n < 1:
        return x
    left = mx.broadcast_to(x[..., :1], x.shape[:-1] + (n,))
    right = mx.broadcast_to(x[..., -1:], x.shape[:-1] + (n,))
    return mx.concatenate([left, x, right], axis=-1)


def zero_ext(x: mx.array, n: int) -> mx.array:
    """Zero-pad extension."""
    if n < 1:
        return x
    z = mx.zeros(x.shape[:-1] + (n,), dtype=x.dtype)
    return mx.concatenate([z, x, z], axis=-1)
