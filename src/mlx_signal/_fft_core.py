"""Safe FFT wrappers working around broken Metal FFT lengths in MLX 0.32.

Empirically (M4 Max, macOS 26.2, mlx 0.32.0), the Metal FFT backend has two
distinct failure modes for large 1-D transform lengths:

- lengths in (2^19, 2^21] other than 2^20, and exactly 2^22, fail to load
  their four-step sub-kernels and raise
  ("Unable to load function four_step_mem_...");
- every other length above 2^20 runs but returns silently incorrect values
  (relative error ~1.0 versus a float64 reference).

Lengths at or below 2^19, and exactly 2^20, are verified accurate (~1e-6
relative, forward and inverse). These wrappers therefore trust the GPU only in
that verified region and route every other length through the MLX CPU stream:
same lazy graph, same unified memory, correct results — just slower for that
one op. Higher-level code additionally avoids big single FFTs where an
algorithm choice can (fftconvolve switches to blocked overlap-add), which
keeps long-signal filtering on the GPU.

If a future MLX release fixes the Metal FFT, relaxing ``metal_fft_broken`` is
the only change needed to reclaim full GPU performance.
"""

from __future__ import annotations

import mlx.core as mx

_SAFE_MAX = 1 << 19
_SAFE_EXACT = 1 << 20


def metal_fft_broken(n: int) -> bool:
    """True if MLX's Metal FFT cannot be *trusted* for a length-``n`` transform."""
    if not mx.metal.is_available():
        return False
    n = int(n)
    return not (n <= _SAFE_MAX or n == _SAFE_EXACT)


def _kw(n: int) -> dict:
    return {"stream": mx.cpu} if metal_fft_broken(int(n)) else {}


def rfft(a, n=None, axis=-1):
    length = int(a.shape[axis] if n is None else n)
    if metal_fft_broken(length):
        from . import _fourstep

        out = _fourstep.rfft_large(a, length, axis)
        if out is not None:
            return out
    return mx.fft.rfft(a, n=n, axis=axis, **_kw(length))


def irfft(a, n=None, axis=-1):
    length = int(2 * (a.shape[axis] - 1) if n is None else n)
    if metal_fft_broken(length):
        from . import _fourstep

        out = _fourstep.irfft_large(a, length, axis)
        if out is not None:
            return out
    return mx.fft.irfft(a, n=n, axis=axis, **_kw(length))


def fft(a, n=None, axis=-1):
    length = int(a.shape[axis] if n is None else n)
    if metal_fft_broken(length):
        from . import _fourstep

        out = _fourstep.fft_large(a, length, axis)
        if out is not None:
            return out
    return mx.fft.fft(a, n=n, axis=axis, **_kw(length))


def ifft(a, n=None, axis=-1):
    length = int(a.shape[axis] if n is None else n)
    if metal_fft_broken(length):
        from . import _fourstep

        out = _fourstep.ifft_large(a, length, axis)
        if out is not None:
            return out
    return mx.fft.ifft(a, n=n, axis=axis, **_kw(length))


def _any_broken(a, s, axes) -> bool:
    lens = s if s is not None else [a.shape[ax] for ax in (axes or range(a.ndim))]
    return any(metal_fft_broken(int(x)) for x in lens)


def rfftn(a, s=None, axes=None):
    if _any_broken(a, s, axes):
        return mx.fft.rfftn(a, s=s, axes=axes, stream=mx.cpu)
    return mx.fft.rfftn(a, s=s, axes=axes)


def irfftn(a, s=None, axes=None):
    if _any_broken(a, s, axes):
        return mx.fft.irfftn(a, s=s, axes=axes, stream=mx.cpu)
    return mx.fft.irfftn(a, s=s, axes=axes)


def fftn(a, s=None, axes=None):
    if _any_broken(a, s, axes):
        return mx.fft.fftn(a, s=s, axes=axes, stream=mx.cpu)
    return mx.fft.fftn(a, s=s, axes=axes)


def ifftn(a, s=None, axes=None):
    if _any_broken(a, s, axes):
        return mx.fft.ifftn(a, s=s, axes=axes, stream=mx.cpu)
    return mx.fft.ifftn(a, s=s, axes=axes)
