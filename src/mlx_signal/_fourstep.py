"""Four-step (Bailey) FFT decomposition for lengths MLX's Metal FFT can't run.

MLX 0.32's Metal FFT is broken above 2^20 (see ``_fft_core``). A length-N
transform with N = N1*N2 decomposes into batched sub-FFTs of lengths N1 and
N2 — both chosen inside the verified-safe zone — plus a twiddle multiply and
transposes, all standard MLX ops on the GPU:

    A  = x.reshape(N2, N1)                # n = n2*N1 + n1
    B  = fft(A.swap(-2, -1), axis=-1)     # DFT_N2 down the columns
    C  = fft((B * TW).swap(-2, -1), -1)   # twiddle, then DFT_N1
    X  = C.swap(-2, -1).reshape(N)        # k = k1*N2 + k2

Real transforms ride on top via the even/odd packing trick (identical math to
the Stockham kernel's untangle, vectorized), inverses via conjugation. The
twiddle phase is computed from an exact int64 outer product mod N, so accuracy
holds at any size. Lengths with no factorization into two safe factors (e.g.
large primes) return None and the caller keeps its CPU-stream fallback.
"""

from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np

from ._fft_core import metal_fft_broken


def _factorize(n: int) -> list[int]:
    factors = []
    m = n
    while m % 2 == 0:
        factors.append(2)
        m //= 2
    p = 3
    while p * p <= m:
        while m % p == 0:
            factors.append(p)
            m //= p
        p += 2
    if m > 1:
        factors.append(m)
    return factors


@functools.lru_cache(maxsize=128)
def _choose_split(n: int) -> tuple[int, int] | None:
    """Divisor pair (N1, N2), both Metal-safe, closest to a balanced split."""
    factors = _factorize(n)
    divisors = {1}
    for f in factors:
        divisors |= {d * f for d in divisors}
    best = None
    target = np.sqrt(n)
    for d in divisors:
        other = n // d
        if metal_fft_broken(d) or metal_fft_broken(other) or d == 1 or other == 1:
            continue
        score = abs(np.log(d / target))
        if best is None or score < best[0]:
            best = (score, d, other)
    if best is None:
        return None
    return best[1], best[2]


@functools.lru_cache(maxsize=4)
def _twiddle(n1: int, n2: int) -> mx.array:
    """TW[j1, k2] = exp(-2*pi*i * j1*k2 / (n1*n2)), phase exact via int64 mod."""
    n = n1 * n2
    prod = mx.arange(n1, dtype=mx.int64)[:, None] * mx.arange(n2, dtype=mx.int64)[None, :]
    theta = (prod % n).astype(mx.float32) * mx.array(-2.0 * np.pi / n, dtype=mx.float32)
    tw = mx.cos(theta).astype(mx.complex64) + mx.sin(theta).astype(mx.complex64) * mx.array(1j)
    mx.eval(tw)
    return tw


def _pad_last(a: mx.array, n: int) -> mx.array:
    """Zero-pad or truncate the last axis to length n (numpy fft semantics)."""
    cur = a.shape[-1]
    if cur == n:
        return a
    if cur > n:
        return a[..., :n]
    return mx.concatenate([a, mx.zeros(a.shape[:-1] + (n - cur,), dtype=a.dtype)], axis=-1)


def _fft_4step_last(a: mx.array) -> mx.array:
    """Forward complex FFT along the last axis via the four-step decomposition.

    ``a`` must be complex64 with a splittable last-axis length.
    """
    n = a.shape[-1]
    if not metal_fft_broken(n):
        return mx.fft.fft(a, axis=-1)
    split = _choose_split(n)
    assert split is not None, "caller must check splittability"
    n1, n2 = split
    batch = a.shape[:-1]
    A = a.reshape(batch + (n2, n1))
    B = mx.fft.fft(mx.swapaxes(A, -2, -1), axis=-1)  # (..., n1, n2)
    C = mx.fft.fft(mx.swapaxes(B * _twiddle(n1, n2), -2, -1), axis=-1)  # (..., n2, n1)
    return mx.swapaxes(C, -2, -1).reshape(batch + (n,))


def _ifft_4step_last(a: mx.array) -> mx.array:
    n = a.shape[-1]
    out = mx.conj(_fft_4step_last(mx.conj(a)))
    return out * mx.array(1.0 / n, dtype=mx.float32)


@functools.lru_cache(maxsize=8)
def _half_twiddle(m: int) -> mx.array:
    """w[k] = exp(-pi*i*k/m) for k in [0, m) — the rfft untangle twiddles."""
    w = np.exp(-1j * np.pi * np.arange(m) / m).astype(np.complex64)
    return mx.array(w)


def _splittable(n: int) -> bool:
    if not metal_fft_broken(n):
        return True
    return _choose_split(n) is not None


# ---------------------------------------------------------------------------
# public entry points (return None when the length can't be decomposed)
# ---------------------------------------------------------------------------


def fft_large(a: mx.array, n: int, axis: int = -1) -> mx.array | None:
    if not _splittable(n):
        return None
    a = a.astype(mx.complex64) if a.dtype != mx.complex64 else a
    a = mx.moveaxis(a, axis, -1) if axis not in (-1, a.ndim - 1) else a
    out = _fft_4step_last(_pad_last(a, n))
    return mx.moveaxis(out, -1, axis) if axis not in (-1, out.ndim - 1) else out


def ifft_large(a: mx.array, n: int, axis: int = -1) -> mx.array | None:
    if not _splittable(n):
        return None
    a = a.astype(mx.complex64) if a.dtype != mx.complex64 else a
    a = mx.moveaxis(a, axis, -1) if axis not in (-1, a.ndim - 1) else a
    out = _ifft_4step_last(_pad_last(a, n))
    return mx.moveaxis(out, -1, axis) if axis not in (-1, out.ndim - 1) else out


def rfft_large(a: mx.array, n: int, axis: int = -1) -> mx.array | None:
    """rfft of real ``a`` at length n. Even n uses the packed half-size trick."""
    if n % 2 or not _splittable(n // 2):
        # odd (or unsplittable half): full complex transform, then slice
        full = fft_large(a, n, axis)
        if full is None:
            return None
        sl = [slice(None)] * full.ndim
        sl[axis] = slice(n // 2 + 1)
        return full[tuple(sl)]
    m = n // 2
    a = mx.moveaxis(a, axis, -1) if axis not in (-1, a.ndim - 1) else a
    x = _pad_last(a.astype(mx.float32) if a.dtype != mx.float32 else a, n)
    z = mx.view(x, mx.complex64)  # z[t] = x[2t] + i*x[2t+1], shape (..., m)
    Z = _fft_4step_last(z)
    zmk = mx.concatenate([Z[..., :1], Z[..., 1:][..., ::-1]], axis=-1)  # Z[(m-k) % m]
    ze = 0.5 * (Z + mx.conj(zmk))
    zo = 0.5 * (Z - mx.conj(zmk))
    w = _half_twiddle(m)
    Xk = ze - mx.array(1j) * (w * zo)  # k = 0..m-1
    nyq = (mx.real(Z[..., :1]) - mx.imag(Z[..., :1])).astype(mx.complex64)
    out = mx.concatenate([Xk, nyq], axis=-1)
    return mx.moveaxis(out, -1, axis) if axis not in (-1, out.ndim - 1) else out


def irfft_large(a: mx.array, n: int, axis: int = -1) -> mx.array | None:
    """irfft of the one-sided spectrum ``a`` to a length-n real signal."""
    if n % 2 or not _splittable(n // 2):
        if not _splittable(n):
            return None
        a = a.astype(mx.complex64) if a.dtype != mx.complex64 else a
        a = mx.moveaxis(a, axis, -1) if axis not in (-1, a.ndim - 1) else a
        half = _pad_last(a, n // 2 + 1)
        # rebuild the full conjugate-symmetric spectrum: X[n-k] = conj(X[k])
        neg = mx.conj(half[..., 1 : n - n // 2][..., ::-1])
        out = mx.real(_ifft_4step_last(mx.concatenate([half, neg], axis=-1)))
        return mx.moveaxis(out, -1, axis) if axis not in (-1, out.ndim - 1) else out
    m = n // 2
    a = a.astype(mx.complex64) if a.dtype != mx.complex64 else a
    a = mx.moveaxis(a, axis, -1) if axis not in (-1, a.ndim - 1) else a
    X = _pad_last(a, m + 1)
    # irfft semantics: DC and Nyquist bins are taken as real
    X = mx.concatenate(
        [
            mx.real(X[..., :1]).astype(mx.complex64),
            X[..., 1:m],
            mx.real(X[..., m : m + 1]).astype(mx.complex64),
        ],
        axis=-1,
    )
    Xk = X[..., :m]
    xmk = mx.conj(X[..., 1:][..., ::-1])  # conj(X[m-k]) for k = 0..m-1
    E = 0.5 * (Xk + xmk)
    wO = 0.5 * (Xk - xmk)
    O = wO * mx.conj(_half_twiddle(m))
    z = _ifft_4step_last(E + mx.array(1j) * O)
    out = mx.stack([mx.real(z), mx.imag(z)], axis=-1).reshape(z.shape[:-1] + (n,))
    return mx.moveaxis(out, -1, axis) if axis not in (-1, out.ndim - 1) else out
