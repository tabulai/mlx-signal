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
| welch | 64ch × 2^20, nperseg=1024 | 495.0 ms | 8.5 ms | 4.3 ms | **58.5x / 115.0x** |
| welch | 1ch × 2^22, nperseg=4096 | 49.6 ms | 1.9 ms | 1.3 ms | **25.9x / 39.3x** |
| welch | 256ch × 2^16, nperseg=256 | 115.0 ms | 4.7 ms | 1.1 ms | **24.3x / 106.1x** |
| csd | 64ch × 2^20, nperseg=1024 | 936.5 ms | 16.3 ms | 8.2 ms | **57.6x / 114.4x** |
| coherence | 64ch × 2^20, nperseg=1024 | 1944.9 ms | 17.6 ms | 9.8 ms | **110.5x / 199.2x** |
| spectrogram | 16ch × 2^20 | 68.7 ms | 6.8 ms | 1.3 ms | **10.1x / 53.7x** |
| stft | 16ch × 2^20, nperseg=1024 | 81.4 ms | 4.7 ms | 1.3 ms | **17.3x / 63.8x** |
| istft | 16ch × 2^20, nperseg=1024 | 174.4 ms | 22.9 ms | 3.2 ms | **7.6x / 54.6x** |
| fftconvolve | 2^20 × 4097 | 11.46 ms | 1.50 ms | 1.27 ms | **7.7x / 9.0x** |
| fftconvolve | 2^22 × 257 | 48.86 ms | 2.63 ms | 0.86 ms | **18.6x / 56.6x** |
| oaconvolve | 2^23 × 513 | 27.19 ms | 4.34 ms | 2.34 ms | **6.3x / 11.6x** |
| correlate (batched) | 64ch × 2^18, 4096 taps | 64.49 ms | 6.80 ms | 4.73 ms | **9.5x / 13.6x** |
| resample_poly | 16ch, 48k→44.1k (147/160) | 120.7 ms | 6.1 ms | 3.9 ms | **19.7x / 31.0x** |
| upfirdn | 64ch × 2^18, up=2 down=3, 255 taps | 302.6 ms | 7.5 ms | 5.5 ms | **40.4x / 55.2x** |
| upfirdn (complex IQ) | 16ch × 2^20 c64, down=10, 201 taps | 182.7 ms | 3.8 ms | 1.3 ms | **47.6x / 145.4x** |
| resample (FFT) | 2^20 → 2^18 | 4.3 ms | 0.4 ms | 0.3 ms | **11.0x / 14.4x** |
| hilbert | 2^20 | 9.9 ms | 0.8 ms | 0.6 ms | **12.9x / 17.9x** |
| lfilter (FIR) | 64ch × 2^20, 257 taps | 1634.7 ms | 17.7 ms | 9.8 ms | **92.1x / 167.7x** |
| sosfilt (IIR) | 256ch × 2^20, butter-8 | 1306.4 ms | 42.5 ms | 12.0 ms | **30.8x / 109.2x** |
| sosfilt (IIR, single channel) | 1ch × 2^22, butter-8 | 20.6 ms | 2.0 ms | 1.3 ms | **10.4x / 16.1x** |
| sosfiltfilt (IIR) | 256ch × 2^20, butter-8 | 2680.6 ms | 125.2 ms | 41.0 ms | **21.4x / 65.4x** |
| filtfilt (FIR) | 64ch × 2^20, 257 taps | 3271.8 ms | 44.5 ms | 23.6 ms | **73.5x / 138.5x** |
| resample (FFT) >1M samples¹ | 2^23 → ×0.75 | 66.5 ms | 7.3 ms | 5.5 ms | **9.2x / 12.1x** |
| hilbert >1M samples¹ | 2^23 | 102.6 ms | 8.1 ms | 6.5 ms | **12.7x / 15.8x** |
| find_peaks | 2^23, prominence=1 | 218.0 ms | 217.2 ms | — | 1.0x² |

¹ MLX 0.32's Metal FFT is broken above 2^20 (see *Known limitations*); mlx-signal
runs those transform lengths through its own four-step (Bailey) decomposition —
two batched safe-size sub-FFTs plus a twiddle multiply — entirely on the GPU.
Only lengths with no safe factorization (e.g. large primes) fall back to a
CPU-stream FFT.
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
| stft, 16ch × 2^20 | 46 ms | 7.2 ms | 26 ms | **4.2 ms** | 43 ms | 65 ms | — |
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
on CPU only; torchaudio has no PSD estimation) — and here `csd`/`coherence` get
their own two-signal variant of the fused kernel that computes both spectra and
the cross spectrum in a single sweep (coherence: one pass instead of scipy's
five). `upfirdn`/`find_peaks` are scipy-only elsewhere. The one column that competes — `torch.stft` on MPS — still
edges mlx-signal end-to-end (its NumPy↔GPU transfer is cheaper) but loses badly
on-device, where the transform itself runs 3.5x faster here (0.96 ms vs 3.4 ms);
and it ships inside a 2 GB torch dependency with the `lfilter` cliff above. The
speed comes from a fused Metal kernel (`_stft_metal.py`): one threadgroup per
segment runs strided load → mean-detrend → window → a full radix-2 Stockham FFT
in threadgroup memory — welch's entire per-segment pipeline reads the signal
once and writes |X|² once, with no frames array ever materialized. `istft` runs
the mirror image: a per-segment inverse-Stockham kernel plus a gather-based
overlap-add (one thread per output sample sums its few overlapping segments —
no scatters, norm divide folded in). Non-pow2 or otherwise ineligible shapes
use the composed path: zero-copy `as_strided` framing, `mx.compile`-fused
detrend+window, scaling folded into the window.

## Install

Requires Apple Silicon, macOS ≥ 13.5, Python ≥ 3.10.

```bash
git clone https://github.com/tabulai/mlx-signal && cd mlx-signal
pip install -e .            # or: uv pip install -e .
python -m pytest -q         # 333 golden tests against scipy and NumPy
```

(PyPI release planned for 0.1.0.)

## What's implemented (v0.1)

| area | functions | notes |
|---|---|---|
| spectral | `periodogram` `welch` `csd` `coherence` `spectrogram` `stft` `istft` | one shared core; fused Stockham Metal kernels on the pow2 hot path (two-signal csd/coherence variant, inverse+gather-OLA for istft), batched FFT otherwise; all windows, detrend, scaling, axis, median averaging |
| convolution | `convolve` `fftconvolve` `oaconvolve` `correlate` `correlation_lags` | N-d, all modes, complex; pow2-padded FFTs, with long×short convolutions auto-blocked into small FFTs reassembled by a gather-OLA kernel |
| resampling | `upfirdn` `resample` `resample_poly` `decimate` | custom Metal kernel for `upfirdn`: one thread per output sample, taps tiled through threadgroup memory (or read direct at high `up`), complex-native — an IQ stream is one launch |
| filtering | `firwin` `firwin2` `lfilter` `filtfilt` `sosfilt` `sosfiltfilt` `hilbert` | FIR and SOS-IIR paths on GPU (sequential + block-parallel scan kernels, single channel up) with scipy-exact edge handling; design host-side |
| peaks | `find_peaks` `peak_prominences` `peak_widths` | exact scipy parity; host-side by design |
| utilities | `get_window` `next_fast_len` | cached windows; pow2 fast lengths |

**IIR runs on the GPU — even single-channel.** `sosfilt`/`sosfiltfilt` (and
`decimate`'s default `ftype="iir"`) run scipy's exact direct-form-II-transposed
cascade in custom Metal kernels with native `zi`/`zf` state, so streaming
chunk-by-chunk works. Long signals use a block-parallel associative-scan
kernel: the cascade is a linear system, so blocks compute their contributions
in parallel and entry states compose through the host-precomputed `A^L`
transition — parallel over time as well as channels, matching scipy's float32
results to ~1e-6 and beating it from one channel (16x) to 256 (109x). Short
signals use a per-channel-sequential kernel (bit-identical to float32 scipy)
when there are enough channels, else scipy. `lfilter` with `len(a) > 1`
(transfer-function form) still falls back to scipy with a `FallbackWarning`;
use the better-conditioned SOS form, as scipy itself recommends. You will not
silently run on one core believing you're on 40 GPU cores.

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

Capability fallbacks (transfer-function IIR, exotic padding modes, callable
detrend) warn loudly; size-based routing is silent by design.

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
  (rel. error ~1.0). mlx-signal verified the safe region empirically and works
  around it on the GPU: 1-D transforms use an in-library four-step (Bailey)
  decomposition into safe-size sub-FFTs (`_fourstep.py`), and `fftconvolve`
  switches to blocked overlap-add. Only unsplittable lengths (large primes) and
  the N-d FFT paths route through the MLX CPU stream (`_fft_core.py`). When MLX
  fixes this, relaxing one predicate retires the workaround. (Worth reporting
  upstream to `ml-explore/mlx` if you can reproduce it.)
- Windows/filter design (`get_window`, `firwin*`) and `find_peaks` refinement run
  host-side — tiny work, and it keeps exact scipy parity.
- `lfilter`/`filtfilt` don't take `zi` on the GPU path yet (falls back, warns);
  `sosfilt` takes and returns its `zi`/`zf` state natively on the GPU.
- `upfirdn` supports the default zero-padded `mode="constant"` on the GPU; other
  signal-extension modes fall back.
- Streaming/chunked APIs, `ShortTimeFFT`, and CWT are not yet implemented (below).

## Roadmap

- Transfer-function `lfilter`/`filtfilt` IIR (SOS form is GPU-native today;
  tf form falls back to scipy)
- **CWT** — scipy *removed* `cwt` in 1.15, so this is a differentiator, not a clone
- Modern `ShortTimeFFT` class
- fp16 mode; real-time streaming API; torchaudio benchmark column

## Examples

- [`examples/fm_demod.py`](examples/fm_demod.py) — the classic cuSignal FM
  demodulation chain (channel filter → polyphase decimate → discriminator →
  de-emphasis → audio resample), 16.5x end-to-end vs scipy on an M4 Max, with
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
  matched against scipy.signal (BSD-3-Clause) and are verified by 333 parity tests.
- **MLX** — the lazy, unified-memory array framework that makes the zero-copy
  story possible.
