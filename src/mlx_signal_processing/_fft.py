"""FFT sizing helpers."""

from __future__ import annotations

__all__ = ["next_fast_len"]


def next_fast_len(target: int, real: bool = True) -> int:
    """Next FFT length >= ``target`` that is fast on Metal.

    MLX's Metal FFT is fastest at powers of two (other sizes route through
    slower mixed-radix/Bluestein paths), so unlike ``scipy.fft.next_fast_len``
    this always returns the next power of two. The ``real`` argument is
    accepted for scipy signature compatibility and ignored.
    """
    target = int(target)
    if target < 0:
        raise ValueError("Target length must be positive")
    if target <= 1:
        return target  # scipy: next_fast_len(0) == 0, (1) == 1
    return 1 << (target - 1).bit_length()
