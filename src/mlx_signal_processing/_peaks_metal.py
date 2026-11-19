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
#: samples per threadgroup in the local-maxima compaction kernels (256 threads x 8)
_LMAX_BLOCK = 2048
#: every index the kernel forms (including the padded nb*BLOCK) must fit a
#: positive int32; beyond this the caller stays on scipy
MAX_N = (1 << 31) - 2 * BLOCK

_DENORM_MASK = mx.array(0x7FFFFFFF, dtype=mx.uint32)
_DENORM_ONE = mx.array(1, dtype=mx.uint32)
_DENORM_CAP = mx.array(0x007FFFFF, dtype=mx.uint32)
_has_denorm_compiled = mx.compile(
    lambda xu: mx.any((mx.bitwise_and(xu, _DENORM_MASK) - _DENORM_ONE) < _DENORM_CAP)
)


def has_f32_denormals_gpu(x32: mx.array) -> bool:
    """True if any float32 element of ``x32`` has an IEEE-754 subnormal bit pattern.

    Uses raw uint32 bit-inspection via ``mx.view``, which never flushes
    denormals on Metal.
    """
    if x32.size == 0:
        return False
    return bool(_has_denorm_compiled(mx.view(x32, mx.uint32)).item())


# ---------------------------------------------------------------------------
# two-pass GPU local-maxima compaction + piggybacked denormal detection
# ---------------------------------------------------------------------------
#
# Matches scipy.signal._peak_finding_utils._local_maxima_1d bit-for-bit for
# denormal-free float32 inputs: index `i` is the left edge of a local maximum
# iff x[i] > x[i-1] and either x[i] > x[i+1] (strict peak) or x[i..j] is a
# flat plateau followed by x[j+1] < x[i]. Unordered NaN comparisons return
# false everywhere, so NaNs never bridge or create peaks. Pass 1 counts peaks
# per 2048-sample block while inspecting raw uint32 bits of EVERY sample for
# float32 denormals; Pass 2 compacts the surviving peak indices (and optional
# plateau edges) directly on GPU.

_LMAX_HEADER = r"""
template <typename Ptr>
static inline int peak_right_edge(Ptr x, int i, int N) {
    if (i <= 0 || i >= N - 1) return -1;
    float xi = x[i];
    if (!(xi > x[i - 1])) return -1;
    float xn = x[i + 1];
    if (xi > xn) return i;
    if (xi == xn) {
        int j = i + 1;
        while (j + 1 < N && x[j + 1] == xi) {
            ++j;
        }
        if (j + 1 < N && x[j + 1] < xi) {
            return j;
        }
    }
    return -1;
}
"""

_LMAX_COUNT_SRC = r"""
    uint tid = thread_index_in_threadgroup;
    uint blk = threadgroup_position_in_grid.x;
    int N = params[0];
    int nb = params[1];
    if ((int)blk >= nb) return;

    int start = (int)blk * 2048 + (int)tid * 8;
    int end = min(start + 8, N);

    uint cnt = 0u;
    uint den = 0u;
    for (int i = start; i < end; ++i) {
        uint m = xu[i] & 0x7FFFFFFFu;
        den |= (uint)((m - 1u) < 0x007FFFFFu);
        if (peak_right_edge(x, i, N) >= 0) ++cnt;
    }

    threadgroup uint tg_cnt[256];
    threadgroup uint tg_den[256];
    tg_cnt[tid] = cnt;
    tg_den[tid] = den;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint off = 128u; off > 0u; off >>= 1) {
        if (tid < off) {
            tg_cnt[tid] += tg_cnt[tid + off];
            tg_den[tid] |= tg_den[tid + off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        block_counts[blk] = tg_cnt[0];
        block_denorm[blk] = tg_den[0];
    }
"""

_LMAX_EMIT_SRC = r"""
    uint tid = thread_index_in_threadgroup;
    uint blk = threadgroup_position_in_grid.x;
    int N = params[0];
    int nb = params[1];
    if ((int)blk >= nb) return;

    int start = (int)blk * 2048 + (int)tid * 8;
    int end = min(start + 8, N);

    // Adjacent local maxima are separated by >= 2 samples, so 8 samples hold <= 4
    int loc_l[4];
    int loc_r[4];
    uint cnt = 0u;
    for (int i = start; i < end; ++i) {
        int r = peak_right_edge(x, i, N);
        if (r >= 0) {
            loc_l[cnt] = i;
            loc_r[cnt] = r;
            ++cnt;
        }
    }

    threadgroup uint scan_a[256];
    threadgroup uint scan_b[256];
    scan_a[tid] = cnt;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup uint* src = scan_a;
    threadgroup uint* dst = scan_b;
    for (uint off = 1u; off < 256u; off <<= 1) {
        dst[tid] = src[tid] + ((tid >= off) ? src[tid - off] : 0u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup uint* tmp = src; src = dst; dst = tmp;
    }
    uint out_pos = block_offsets[blk] + (src[tid] - cnt);
    for (uint c = 0u; c < cnt; ++c) {
        uint l = (uint)loc_l[c];
        uint r = (uint)loc_r[c];
        peaks[out_pos + c] = l + ((r - l) >> 1);
{EDGE_STORE}
    }
"""


@functools.lru_cache(maxsize=1)
def _lmax_count_kernel():
    return mx.fast.metal_kernel(
        name="mlx_signal_processing_lmax_count",
        input_names=["x", "xu", "params"],
        output_names=["block_counts", "block_denorm"],
        source=_LMAX_COUNT_SRC,
        header=_LMAX_HEADER,
        ensure_row_contiguous=True,
    )


@functools.lru_cache(maxsize=2)
def _lmax_emit_kernel(with_edges: bool):
    edge_store = (
        "        left_edges[out_pos + c] = l;\n"
        "        right_edges[out_pos + c] = r;"
        if with_edges
        else ""
    )
    return mx.fast.metal_kernel(
        name=f"mlx_signal_processing_lmax_emit_{int(with_edges)}",
        input_names=["x", "block_offsets", "params"],
        output_names=["peaks", "left_edges", "right_edges"] if with_edges else ["peaks"],
        source=_LMAX_EMIT_SRC.replace("{EDGE_STORE}", edge_store),
        header=_LMAX_HEADER,
        ensure_row_contiguous=True,
    )


def local_maxima_gpu(x32: mx.array, need_edges: bool = False):
    """Detect local maxima on GPU while scanning ``x32`` for float32 denormals.

    Returns ``None`` if any float32 denormal is present (caller must fall back
    to exact CPU float64). Otherwise returns ``(peaks_u32, left_u32, right_u32)``
    where ``left_u32`` and ``right_u32`` are ``None`` unless ``need_edges`` is set.
    """
    n = x32.size
    nb = -(-n // _LMAX_BLOCK)
    params = mx.array([n, nb], dtype=mx.int32)
    xu = mx.view(x32, mx.uint32)
    bc, bd = _lmax_count_kernel()(
        inputs=[x32, xu, params],
        grid=(nb * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(nb,), (nb,)],
        output_dtypes=[mx.uint32, mx.uint32],
    )
    cs = mx.cumsum(bc)
    total = cs[-1]
    offsets = cs - bc
    any_den = mx.any(bd != 0)
    mx.eval(offsets, total, any_den)
    if bool(any_den.item()):
        return None
    p = int(total.item())
    if p == 0:
        empty = mx.zeros((0,), dtype=mx.uint32)
        return (empty, empty, empty) if need_edges else (empty, None, None)
    outs = _lmax_emit_kernel(need_edges)(
        inputs=[x32, offsets, params],
        grid=(nb * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(p,), (p,), (p,)] if need_edges else [(p,)],
        output_dtypes=[mx.uint32] * (3 if need_edges else 1),
    )
    if need_edges:
        return outs[0], outs[1], outs[2]
    return outs[0], None, None

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


def prominence_bases(x32: mx.array, peaks):
    """Base indices for every peak. x32: (N,) float32; peaks: validated intp
    NumPy indices or uint32 MLX indices into x32. Returns (left_bases, right_bases)
    as np.intp arrays."""
    n = x32.size
    if isinstance(peaks, mx.array):
        peaks_u32 = peaks if peaks.dtype == mx.uint32 else peaks.astype(mx.uint32)
        p = int(peaks_u32.size)
    else:
        p = int(peaks.size)
        peaks_u32 = mx.array(peaks.astype(np.uint32))
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
        inputs=[x32, peaks_u32, bmax, bmin, params],
        grid=(p, 1, 1),
        threadgroup=(min(256, max(32, p)), 1, 1),
        output_shapes=[(p,), (p,)],
        output_dtypes=[mx.uint32, mx.uint32],
    )
    return np.array(lb).astype(np.intp), np.array(rb).astype(np.intp)
