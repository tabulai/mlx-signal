# mlx-signal

**Metal/MLX-accelerated signal processing for Apple Silicon, mirroring `scipy.signal`.**

`scipy.signal` runs every `welch()`, `resample_poly()`, and `lfilter()` call on a
single performance core — NumPy/SciPy only link Apple's Accelerate for BLAS/LAPACK,
which signal processing never touches. On an M-series GPU those same workloads are
thousands of independent FFTs and dot products. mlx-signal keeps the scipy API and
moves the math to the GPU through [MLX](https://github.com/ml-explore/mlx):

```python
import numpy as np
import mlx_signal as sig   # drop-in: same signatures as scipy.signal

x = np.random.randn(64, 1 << 20).astype(np.float32)   # 64-channel recording
f, Pxx = sig.welch(x, fs=48_000, nperseg=1024)         # one batched GPU FFT
Pxx_np = np.array(Pxx)                                 # unified memory: cheap
```

Inputs can be NumPy or MLX arrays; outputs are MLX arrays in **unified memory**, so
`np.array(result)` costs a copy within the same RAM — and an audio/RF preprocessing
chain can feed an MLX model with **zero host-device transfers**, which neither scipy
nor torchaudio-on-MPS (with its silent CPU fallback bouncing) offers.

## Measured performance

Apple M4 Max (macOS 26.2), mlx 0.32.0, scipy 1.18.0, float32, median of 5 runs.
*e2e* = NumPy in / NumPy out (the drop-in experience). *device* = MLX arrays in and
out (steady-state pipelines). Reproduce with `python bench/bench.py`.

| function | shape | scipy | mlx-signal (e2e) | mlx-signal (device) | speedup (e2e / device) |
|---|---|---:|---:|---:|---:|
| welch | 64ch × 2^20, nperseg=1024 | 482.3 ms | 18.0 ms | 13.9 ms | **26.8x / 34.7x** |
| welch | 1ch × 2^22, nperseg=4096 | 47.6 ms | 1.8 ms | 0.8 ms | **26.7x / 57.2x** |
| welch | 256ch × 2^16, nperseg=256 | 111.9 ms | 5.0 ms | 3.6 ms | **22.2x / 31.4x** |
| spectrogram | 16ch × 2^20 | 67.3 ms | 7.3 ms | 3.7 ms | **9.2x / 18.1x** |
| stft | 16ch × 2^20, nperseg=1024 | 78.3 ms | 6.1 ms | 1.8 ms | **12.9x / 44.8x** |
| fftconvolve | 2^20 × 4097 | 11.2 ms | 1.8 ms | 1.5 ms | **6.4x / 7.5x** |
| fftconvolve | 2^22 × 257 | 47.8 ms | 2.9 ms | 1.7 ms | **16.3x / 27.7x** |
| oaconvolve | 2^23 × 513 | 27.4 ms | 4.0 ms | 2.7 ms | **6.8x / 10.4x** |
| correlate (batched) | 64ch × 2^18, 4096 taps | 65.2 ms | 8.4 ms | 6.1 ms | **7.8x / 10.7x** |
| resample_poly | 16ch, 48k→44.1k (147/160) | 120.3 ms | 6.4 ms | 4.2 ms | **18.8x / 28.3x** |
| upfirdn | 64ch × 2^18, up=2 down=3, 255 taps | 303.7 ms | 8.9 ms | 7.0 ms | **34.0x / 43.6x** |
| resample (FFT) | 2^20 → 2^18 | 4.3 ms | 0.4 ms | 0.3 ms | **11.0x / 14.4x** |
| hilbert | 2^20 | 9.9 ms | 0.8 ms | 0.6 ms | **12.9x / 17.9x** |
| lfilter (FIR) | 64ch × 2^20, 257 taps | 1634.7 ms | 17.7 ms | 9.8 ms | **92.1x / 167.7x** |
| filtfilt (FIR) | 64ch × 2^20, 257 taps | 3271.8 ms | 44.5 ms | 23.6 ms | **73.5x / 138.5x** |
| resample (FFT) >1M samples | 2^23 → ×0.75 | 67.1 ms | 65.7 ms | 64.2 ms | 1.0x¹ |
| hilbert >1M samples | 2^23 | 103.3 ms | 180.4 ms | 177.3 ms | 0.6x¹ |
| find_peaks | 2^23, prominence=1 | 218.0 ms | 217.2 ms | — | 1.0x² |

¹ MLX 0.32's Metal FFT is broken above 2^20 (see *Known limitations*); mlx-signal
routes those transform lengths through the CPU stream for correctness. Blocked
algorithms keep long-signal *filtering* on the GPU (see fftconvolve 2^22 above);
only functions needing one giant FFT (`resample`, `hilbert` on >1M samples) drop
to ~CPU speed until the upstream fix. For rational ratios, `resample_poly` is the
fast path at any length.
² `find_peaks` is bandwidth-bound index bookkeeping — the wrong shape for a GPU.
It is included for pipeline completeness (GPU sign-diff prefilter, host refinement)
and priced honestly at ~1x.

## How it compares beyond scipy

Other Mac-runnable implementations exist for parts of this API: torch/torchaudio
(CPU and the MPS GPU backend), `jax.scipy.signal` (XLA CPU), librosa, and soxr.
Same machine, conventions aligned, every output verified against scipy before
timing (full details: [bench/results/cross.md](bench/results/cross.md), reproduce
with `pip install -e ".[bench]" && python bench/bench_cross.py`). End-to-end
(NumPy in/out), best per row in bold:

| task | scipy | **mlx-signal** | torch/ta CPU | torch/ta MPS | jax (jit, CPU) | librosa | soxr |
|---|---:|---:|---:|---:|---:|---:|---:|
| welch, 64ch × 2^20 | 501 ms | **18 ms** | — | — | 137 ms | — | — |
| stft, 16ch × 2^20 | 46 ms | 5.7 ms | 25 ms | **4.9 ms** | 46 ms | 64 ms | — |
| fftconvolve, 2^20 × 4097 | 11 ms | **1.6 ms** | 39 ms | 4.6 ms | 16 ms | — | — |
| resample 48k→44.1k, 16ch × 2^20 | 122 ms | **6.6 ms**¹ | 8.7 ms¹ | 5.8 ms¹ | — | 56 ms | 53 ms |
| causal FIR, 64ch × 2^20, 257 taps | 1633 ms | **17 ms** | 2885 ms² | 70 ms² | — | — | — |

¹ Task-level: torchaudio's default anti-aliasing filter (`lowpass_filter_width=6`)
is far shorter than scipy's/ours (3201 taps here) — mlx-signal matches
torchaudio-MPS speed while doing ~20x the filter work at scipy-identical quality.
² torch has no FFT convolution for filtering, so the idiomatic path is direct
`conv1d` (O(n·k)); torchaudio's `lfilter` (its general IIR machinery) takes
**3.6 s on CPU and 21.5 s on MPS** for this FIR case — 1200x slower than
mlx-signal — which is exactly the patchy-MPS-coverage problem this library exists
to avoid.

Takeaways: nothing else offers GPU `welch`/`csd`/`coherence` (JAX mirrors scipy
on CPU only; torchaudio has no PSD estimation), and `upfirdn`/`find_peaks` are
scipy-only elsewhere. The one column that competes — `torch.stft` on MPS — edges
mlx-signal end-to-end (its NumPy↔GPU transfer is cheaper) but loses on-device,
where the actual transform runs 2x faster here (1.4 ms vs 3.1 ms); and it ships
inside a 2 GB torch dependency with the `lfilter` cliff above. The spectral core
frames segments as zero-copy strided views and folds all scaling into the window
before the FFT, so an stft is just two memory passes: windowed-gather, then FFT.

## Install

Requires Apple Silicon, macOS ≥ 13.5, Python ≥ 3.10.

```bash
git clone https://github.com/tabulai/mlx-signal && cd mlx-signal
pip install -e .            # or: uv pip install -e .
python -m pytest -q         # 274 golden tests against scipy
```

(PyPI release planned for 0.1.0.)

## What's implemented (v0.1)

| area | functions | notes |
|---|---|---|
| spectral | `periodogram` `welch` `csd` `coherence` `spectrogram` `stft` `istft` | one shared batched-FFT core; all windows, detrend, scaling, axis, median averaging |
| convolution | `convolve` `fftconvolve` `oaconvolve` `correlate` `correlation_lags` | N-d, all modes, complex; FFT lengths padded to powers of two |
| resampling | `upfirdn` `resample` `resample_poly` `decimate` | custom Metal kernel for `upfirdn` (one thread per output sample, taps staged in threadgroup memory) |
| filtering | `firwin` `firwin2` `lfilter` `filtfilt` `hilbert` | FIR paths on GPU with scipy-exact edge handling; design host-side |
| peaks | `find_peaks` `peak_prominences` `peak_widths` | exact scipy parity; host-side by design |
| utilities | `get_window` `next_fast_len` | cached windows; pow2 fast lengths |

**Deliberately deferred: general IIR.** `lfilter` with `len(a) > 1`, `filtfilt` for
IIR, `sosfilt`, and `decimate`'s default `ftype="iir"` are recursive — a GPU only
wins when batched across channels via an associative scan, which is on the roadmap.
Today those calls fall back to scipy and tell you so with a `FallbackWarning`
(pass `ftype="fir"` to `decimate` to stay on the GPU). You will not silently run
on one core believing you're on 40 GPU cores.

## Dispatch: when the GPU is used

Kernel launches cost more than tiny problems do, so smaller inputs would *lose* on
the GPU. Every function routes automatically:

- **`dispatch="auto"`** (default): inputs above `gpu_min_size` (default 2^15
  elements of work) run on MLX; smaller inputs run scipy. Both return identical
  MLX-array types/dtypes, so the routing is invisible.
- **`dispatch="mlx"`**: always MLX; calls with no MLX path raise
  `NotImplementedError` instead of falling back (pin the GPU in tests).
- **`dispatch="scipy"`**: always scipy (correctness reference; still returns MLX arrays).

```python
sig.set_config(dispatch="mlx")                  # global
with sig.config_context(gpu_min_size=1 << 18):  # scoped
    ...
```

Capability fallbacks (IIR, exotic padding modes, callable detrend) warn loudly;
size-based routing is silent by design.

## Dtype policy

Metal has no float64, period. Computation is **float32/complex64**. Explicit
float64/complex128 inputs downcast with a one-time `DowncastWarning`
(`set_config(float64="strict")` to raise instead; `warn_on_downcast=False` to
hush). Golden tests hold the fp32 pipeline to `rtol=1e-4` with a peak-relative
`atol≈1e-5` against float64 scipy references — honest fp32 accuracy, documented
rather than hidden.

## Known limitations

- **MLX 0.32 Metal FFT above 2^20 is broken upstream** — lengths in
  (2^19, 2^21] except 2^20 *crash* ("Unable to load function four_step_mem_…"),
  and, worse, other lengths above 2^20 *silently return wrong values*
  (rel. error ~1.0). mlx-signal verified the safe region empirically and routes
  every untrusted length through the MLX CPU stream (`_fft_core.py`), while
  `fftconvolve` switches to blocked overlap-add so long-signal filtering stays on
  the GPU. When MLX fixes this, relaxing one predicate reclaims full performance.
  (Worth reporting upstream to `ml-explore/mlx` if you can reproduce it.)
- Windows/filter design (`get_window`, `firwin*`) and `find_peaks` refinement run
  host-side — tiny work, and it keeps exact scipy parity.
- `lfilter`/`filtfilt` don't take `zi` on the GPU path yet (falls back, warns).
- `upfirdn` supports the default zero-padded `mode="constant"` on the GPU; other
  signal-extension modes fall back.
- Streaming/chunked APIs, `ShortTimeFFT`, and CWT are not yet implemented (below).

## Roadmap

- **IIR via associative scan** (batched `lfilter`/`sosfilt` — first-order linear
  recurrences parallelize; this is the most-requested gap)
- **CWT** — scipy *removed* `cwt` in 1.15, so this is a differentiator, not a clone
- Modern `ShortTimeFFT` class; `mx.compile` fusion of window/scale/magnitude chains
- Four-step FFT decomposition in-library to reclaim GPU speed for >2^20 transforms
  while upstream is broken
- fp16 mode; real-time streaming API; torchaudio benchmark column

## Examples

- [`examples/fm_demod.py`](examples/fm_demod.py) — the classic cuSignal FM
  demodulation chain (channel filter → polyphase decimate → discriminator →
  de-emphasis → audio resample), 12.1x end-to-end vs scipy on an M4 Max, with
  0.999 correlation to the true message.
- [`examples/eeg_bandpower.py`](examples/eeg_bandpower.py) — 64-channel × 10-minute
  EEG alpha-band power via one batched `welch`, 6x vs scipy including result
  readback.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
python -m pytest -q          # golden tests vs scipy (CPU-safe; GPU tests auto-skip)
ruff check src tests
python bench/bench.py        # GPU benchmark — the release gate runs on real hardware
```

CI runs lint + the full test suite on GitHub's arm64 macOS runners; the MLX CPU
backend covers every path except the custom Metal kernel (marked `gpu`).

## Acknowledgments

- **SciPy** — the API contract and the golden reference. Edge-case semantics were
  matched against scipy.signal (BSD-3-Clause) and are verified by 274 parity tests.
- **cuSignal** (RAPIDS) — validated the "keep the scipy API, swap the array
  library, add custom kernels where it counts" playbook this project follows.
- **MLX** — the lazy, unified-memory array framework that makes the zero-copy
  story possible.
