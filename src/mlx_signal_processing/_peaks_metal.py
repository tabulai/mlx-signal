"""Per-peak prominence-base Metal kernel: a two-level block-skip scan.

scipy's ``peak_prominences`` walks left and right from every peak until a
strictly higher sample, tracking the running minimum — a sequential Cython
loop that dominates ``find_peaks(prominence=...)`` (measured 75% of its
runtime at 2^23 samples / 2.8M peaks). Each peak's scan is independent, so
one GPU thread per peak parallelizes it; the catch is the heavy tail — a few
peaks scan millions of samples (0.07% of peaks carry half the visits) and
would serialize the whole grid. The kernel therefore skips whole blocks using
precomputed per-block maxima/minima (built with plain ``mx`` reductions):
a block whose max cannot terminate the scan and whose min cannot improve the
running minimum costs one comparison instead of ``BLOCK``.

Exactness contract (scipy-bit-identical for float32 sources):
- every GPU comparison operates on float32 values whose float64 embedding is
  exact, so each predicate decides identically to scipy's float64 scan;
- termination uses ``!(v <= x[peak])``, which matches scipy's loop condition
  including NaN (any comparison with NaN is false, ending the scan — and the
  block maxima propagate NaN, turning a NaN block into a terminator block the
  scan then enters elementwise);
- the running minimum updates on strict ``<``, so the surviving base is the
  occurrence of the minimum closest to the peak, exactly like scipy's
  outward walk; a block-level minimum is resolved to that occurrence by
  descending into the pending block from its peak-side edge;
- the caller computes the prominences themselves host-side in float64 from
  the returned base indices — bit-identical to scipy's arithmetic.
"""

from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np

#: samples per skip block; block aux totals 2 * N/BLOCK floats
BLOCK = 4096
#: every index the kernel forms (including the padded nb*BLOCK) must fit a
#: positive int32; beyond this the caller stays on scipy
MAX_N = (1 << 31) - 2 * BLOCK

_SRC = r"""
    uint tid = thread_position_in_grid.x;
    int N  = params[0];
    int B  = params[1];
    int nb = params[2];
    int P  = params[3];
    if (tid >= (uint)P) return;

    int peak = (int)peaks[tid];
    float xp = x[peak];

    // ---------------- left ----------------
    int lb = peak;
    {
        float lmin = xp;
        int bs = (peak / B) * B;
        bool term = false;
        for (int i = peak - 1; i >= bs; --i) {
            float v = x[i];
            if (!(v <= xp)) { term = true; break; }
            if (v < lmin) { lmin = v; lb = i; }
        }
        if (!term && bs > 0) {
            int pending = -1;
            int b = peak / B - 1;
            for (; b >= 0; --b) {
                if (!(bmax[b] <= xp)) break;              // terminator block
                if (bmin[b] < lmin) { lmin = bmin[b]; pending = b; }
            }
            if (pending >= 0) {
                // first occurrence of lmin scanning toward the peak side
                for (int j = (pending + 1) * B - 1; j >= pending * B; --j) {
                    if (x[j] == lmin) { lb = j; break; }
                }
            }
            if (b >= 0) {
                // elementwise scan of the terminator block (last in scan
                // order, so strict < correctly requires beating lmin)
                for (int j = (b + 1) * B - 1; j >= b * B; --j) {
                    float v = x[j];
                    if (!(v <= xp)) break;
                    if (v < lmin) { lmin = v; lb = j; }
                }
            }
        }
    }

    // ---------------- right ----------------
    int rb = peak;
    {
        float rmin = xp;
        int be = min(((peak / B) + 1) * B, N);
        bool term = false;
        for (int i = peak + 1; i < be; ++i) {
            float v = x[i];
            if (!(v <= xp)) { term = true; break; }
            if (v < rmin) { rmin = v; rb = i; }
        }
        if (!term && be < N) {
            int pending = -1;
            int b = peak / B + 1;
            for (; b < nb; ++b) {
                if (!(bmax[b] <= xp)) break;
                if (bmin[b] < rmin) { rmin = bmin[b]; pending = b; }
            }
            if (pending >= 0) {
                int hi = min((pending + 1) * B, N);
                for (int j = pending * B; j < hi; ++j) {
                    if (x[j] == rmin) { rb = j; break; }
                }
            }
            if (b < nb) {
                int hi = min((b + 1) * B, N);
                for (int j = b * B; j < hi; ++j) {
                    float v = x[j];
                    if (!(v <= xp)) break;
                    if (v < rmin) { rmin = v; rb = j; }
                }
            }
        }
    }

    left_bases[tid] = (uint)lb;
    right_bases[tid] = (uint)rb;
"""


@functools.lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="mlx_signal_processing_peak_prominence_bases",
        input_names=["x", "peaks", "bmax", "bmin", "params"],
        output_names=["left_bases", "right_bases"],
        source=_SRC,
        ensure_row_contiguous=True,
    )


def prominence_bases(x32: mx.array, peaks: np.ndarray):
    """Base indices for every peak. x32: (N,) float32; peaks: validated intp
    indices into x32. Returns (left_bases, right_bases) as np.intp arrays."""
    n = x32.size
    p = int(peaks.size)
    nb = -(-n // BLOCK)
    pad = nb * BLOCK - n
    if pad:
        # pad so the reductions tile evenly; -inf/+inf are identities for
        # max/min and the kernel never reads x past N
        xmax_src = mx.concatenate([x32, mx.full((pad,), -mx.inf, dtype=mx.float32)])
        xmin_src = mx.concatenate([x32, mx.full((pad,), mx.inf, dtype=mx.float32)])
    else:
        xmax_src = xmin_src = x32
    bmax = xmax_src.reshape(nb, BLOCK).max(axis=1)
    bmin = xmin_src.reshape(nb, BLOCK).min(axis=1)
    params = mx.array(np.array([n, BLOCK, nb, p], dtype=np.int32))
    lb, rb = _kernel()(
        inputs=[x32, mx.array(peaks.astype(np.uint32)), bmax, bmin, params],
        grid=(p, 1, 1),
        threadgroup=(min(256, max(32, p)), 1, 1),
        output_shapes=[(p,), (p,)],
        output_dtypes=[mx.uint32, mx.uint32],
    )
    return np.array(lb).astype(np.intp), np.array(rb).astype(np.intp)
