# mlx-signal

**Metal/MLX-accelerated signal processing for Apple Silicon, mirroring `scipy.signal`.**

`scipy.signal` runs every `welch()`, `resample_poly()`, and `lfilter()` call on a
single performance core — NumPy/SciPy only link Apple's Accelerate for BLAS/LAPACK,
which signal processing never touches. On an M-series GPU those same workloads are
thousands of independent FFTs and dot products. mlx-signal keeps the scipy API and
moves the math to the GPU through [MLX](https://github.com/ml-explore/mlx):

```python
import numpy as np
import mlx_signal as sig   # scipy.signal signatures (implemented subset below)

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
| welch | 64ch × 2^20, nperseg=1024 | 460.40 ms | 10.99 ms | 4.81 ms | **41.9x / 95.8x** |
| welch | 1ch × 2^22, nperseg=4096 | 45.45 ms | 2.05 ms | 0.61 ms | **22.2x / 74.5x** |
| welch | 256ch × 2^16, nperseg=256 | 106.70 ms | 4.22 ms | 1.06 ms | **25.3x / 100.7x** |
| csd | 64ch × 2^20, nperseg=1024 | 899.76 ms | 15.92 ms | 8.03 ms | **56.5x / 112.1x** |
| coherence | 64ch × 2^20, nperseg=1024 | 1869.78 ms | 17.53 ms | 9.60 ms | **106.7x / 194.8x** |
| spectrogram | 16ch × 2^20 | 63.80 ms | 3.68 ms | 1.27 ms | **17.3x / 50.1x** |
| stft | 16ch × 2^20, nperseg=1024 | 76.32 ms | 4.73 ms | 1.22 ms | **16.1x / 62.3x** |
| istft | 16ch × 2^20, nperseg=1024 | 168.34 ms | 20.61 ms | 2.76 ms | **8.2x / 61.1x** |
| fftconvolve | 2^20 × 4097 | 10.61 ms | 1.55 ms | 1.38 ms | **6.8x / 7.7x** |
| fftconvolve | 2^22 × 257 | 42.48 ms | 2.65 ms | 0.73 ms | **16.0x / 58.4x** |
| fftconvolve (pair) | 2^20 × 2^20 | 20.49 ms | 2.67 ms | 0.92 ms | **7.7x / 22.3x** |
| oaconvolve | 2^23 × 513 | 25.48 ms | 4.50 ms | 1.25 ms | **5.7x / 20.3x** |
| correlate (batched) | 64ch × 2^18, 4096 taps | 61.70 ms | 7.17 ms | 4.60 ms | **8.6x / 13.4x** |
| correlate (auto) | 2^20 autocorrelation | 19.78 ms | 2.12 ms | 0.66 ms | **9.3x / 29.9x** |
| resample_poly | 16ch, 48k→44.1k (147/160) | 116.44 ms | 4.68 ms | 0.97 ms | **24.9x / 120.1x** |
| upfirdn | 64ch × 2^18, up=2 down=3, 255 taps | 292.68 ms | 4.35 ms | 2.32 ms | **67.3x / 126.2x** |
| upfirdn (complex IQ) | 16ch × 2^20 c64, down=10, 201 taps | 172.77 ms | 4.00 ms | 1.25 ms | **43.2x / 138.7x** |
| resample (FFT) | 2^20 → 2^18 | 4.09 ms | 0.46 ms | 0.78 ms | **9.0x / 5.3x** |
| hilbert | 2^20 | 8.62 ms | 1.40 ms | 1.26 ms | **6.1x / 6.8x** |
| lfilter (FIR) | 64ch × 2^20, 257 taps | 1560.86 ms | 12.84 ms | 4.87 ms | **121.6x / 320.6x** |
| lfilter (IIR) | 256ch × 2^20, butter-4 tf | 1473.67 ms | 42.16 ms | 11.74 ms | **35.0x / 125.5x** |
| lfilter (IIR, single channel) | 1ch × 2^22, butter-4 tf | 21.83 ms | 3.76 ms | 0.91 ms | **5.8x / 24.0x** |
| sosfilt (IIR) | 256ch × 2^20, butter-8 | 1275.24 ms | 41.87 ms | 11.52 ms | **30.5x / 110.7x** |
| sosfilt (IIR, single channel) | 1ch × 2^22, butter-8 | 19.99 ms | 2.48 ms | 1.35 ms | **8.1x / 14.9x** |
| sosfiltfilt (IIR) | 256ch × 2^20, butter-8 | 2617.35 ms | 128.90 ms | 39.74 ms | **20.3x / 65.9x** |
| filtfilt (IIR) | 256ch × 2^20, butter-4 tf | 2935.32 ms | 125.64 ms | 40.00 ms | **23.4x / 73.4x** |
| filtfilt (FIR) | 64ch × 2^20, 257 taps | 3162.31 ms | 33.67 ms | 13.49 ms | **93.9x / 234.4x** |
| resample (FFT) >1M samples¹ | 2^23 → ×0.75 | 63.00 ms | 3.96 ms | 3.15 ms | **15.9x / 20.0x** |
| hilbert >1M samples¹ | 2^23 | 96.53 ms | 5.95 ms | 4.61 ms | **16.2x / 20.9x** |
| find_peaks | 2^23, prominence=1 | 209.45 ms | 79.09 ms | — | **2.6x**² |
| peak_prominences | 2^23, 2.8M peaks | 150.43 ms | 19.28 ms | — | **7.8x**² |

¹ MLX 0.32's Metal FFT is broken above 2^20 (see *Known limitations*); mlx-signal
runs those transform lengths through its own four-step (Bailey) decomposition —
two batched safe-size sub-FFTs plus a twiddle multiply — entirely on the GPU.
Only lengths with no safe factorization (e.g. large primes) fall back to a
CPU-stream FFT.
² `find_peaks`' prominence stage — scipy's dominant cost, a sequential walk
from every peak — runs on the GPU (one thread per peak with block-skip aux,
bit-identical to scipy); the remaining index bookkeeping is host-side by
design, which caps the end-to-end win near ~2.7x.

## How it compares beyond scipy

Other Mac-runnable implementations exist for parts of this API: torch/torchaudio
(CPU and the MPS GPU backend), `jax.scipy.signal` (XLA CPU), librosa, and soxr.
Same machine, conventions aligned, every output verified against scipy before
publication (full details: [bench/results/cross.md](bench/results/cross.md), reproduce
with `pip install -e ".[bench]" && python bench/bench_cross.py`). End-to-end
(NumPy in/out), best per row in bold:

| task | scipy | **mlx-signal** | torch/ta CPU | torch/ta MPS | jax (jit, CPU) | librosa | soxr |
|---|---:|---:|---:|---:|---:|---:|---:|
| welch, 64ch × 2^20 | 490 ms | **8.6 ms** | — | — | 136 ms | — | — |
| stft, 16ch × 2^20 | 45 ms | **3.9 ms** | 25 ms | 4.1 ms | 42 ms | 64 ms | — |
| fftconvolve, 2^20 × 4097 | 12 ms | **1.6 ms** | 38 ms | 4.3 ms | 16 ms | — | — |
| resample 48k→44.1k, 16ch × 2^20 | 122 ms | **5.7 ms**¹ | 8.6 ms¹ | **5.7 ms**¹ | — | 56 ms | 52 ms |
| causal FIR, 64ch × 2^20, 257 taps | 1616 ms | **13 ms** | 2849 ms² | 70 ms² | — | — | — |

¹ Task-level: torchaudio's default anti-aliasing filter (`lowpass_filter_width=6`)
is far shorter than scipy's/ours (3201 taps here) — mlx-signal ties torchaudio-MPS
end-to-end while doing ~20x the filter work at scipy-identical quality, and once
arrays live on the GPU it runs 3x faster (1.05 ms vs 3.16 ms on-device).
² torch has no FFT convolution for filtering, so the idiomatic path is direct
`conv1d` (O(n·k)); torchaudio's `lfilter` (its general IIR machinery) takes
**3.5 s on CPU and 21.7 s on MPS** for this FIR case — over 1500x slower than
mlx-signal — which is exactly the patchy-MPS-coverage problem this library exists
to avoid.

Takeaways: nothing else offers GPU `welch`/`csd`/`coherence` (JAX mirrors scipy
on CPU only; torchaudio has no PSD estimation) — and here `csd`/`coherence` get
their own two-signal variant of the fused kernel that computes both spectra and
the cross spectrum in a single sweep (coherence: one pass instead of scipy's
five). `upfirdn`/`find_peaks` are scipy-only elsewhere. The closest competitor —
`torch.stft` on MPS — trades the end-to-end lead with mlx-signal run to run
(roughly 4–7 ms here; the NumPy↔GPU copy dominates and wobbles) but loses
decisively on-device, where the transform itself runs ~3x faster here
(0.93 ms vs 2.67 ms); and it ships inside a 2 GB torch dependency with the
`lfilter` cliff above. The
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
python -m pytest -q         # full golden suite against scipy and NumPy
```

(PyPI release planned for 0.1.0.)

## What's implemented (v0.1)

| area | functions | notes |
|---|---|---|
| spectral | `periodogram` `welch` `csd` `coherence` `spectrogram` `stft` `istft` | one shared core; fused Stockham Metal kernels on the pow2 hot path (two-signal csd/coherence variant, inverse+gather-OLA for istft), batched FFT otherwise; all windows, detrend, scaling, axis, median averaging |
| convolution | `convolve` `fftconvolve` `oaconvolve` `correlate` `correlation_lags` | N-d, all modes, complex; pow2-padded FFTs; long×short convolutions auto-block, and filters ≤1025 taps run a fused kernel pair (block FFT, spectrum multiply, inverse FFT in threadgroup memory) reassembled by gather-OLA; `fftconvolve(x, x)` and `correlate(x, x)` skip the second forward transform |
| resampling | `upfirdn` `resample` `resample_poly` `decimate` | custom Metal kernel for `upfirdn`: one thread per output sample, polyphase geometry in 32-bit arithmetic whenever indices fit (Apple GPUs emulate 64-bit divides — worth ~4x at high `up`), taps staged through threadgroup memory only where measured to pay (pure decimation with a wide gather stride), complex-native — an IQ stream is one launch; every scipy signal-extension mode via on-device boundary extension, and the statistical padtypes via on-device background subtraction |
| filtering | `firwin` `firwin2` `lfilter` `filtfilt` `sosfilt` `sosfiltfilt` `hilbert` | FIR, SOS-IIR, and transfer-function-IIR paths on GPU (sequential + block-parallel scan kernels, single channel up; tf form to order 16 with native `zi`/`zf`) with scipy-exact edge handling; design host-side |
| peaks | `find_peaks` `peak_prominences` `peak_widths` | exact scipy parity; prominence base-search on GPU for f32 sources (one thread per peak, two-level block-skip scan), index bookkeeping host-side |
| utilities | `get_window` `next_fast_len` | cached windows; pow2 fast lengths |

**IIR runs on the GPU — even single-channel.** `sosfilt`/`sosfiltfilt` (and
`decimate`'s default `ftype="iir"`) run scipy's exact direct-form-II-transposed
cascade in custom Metal kernels with native `zi`/`zf` state, so streaming
chunk-by-chunk works. `lfilter`/`filtfilt` with `len(a) > 1` (transfer-function
form) get the same treatment: an order-N DF2T kernel pair that spells out the
fused-multiply-add structure of scipy's compiled recurrence, bit-identical to
scipy-in-float32 up to order 16 (125x at 256 channels, 24x single-channel).
Long signals use a block-parallel associative-scan kernel: the filter is a
linear system, so blocks compute their contributions in parallel and entry
states compose through the host-precomputed `A^L` transition — parallel over
time as well as channels, matching scipy's float32 results to ~1e-6 for
typical wideband filters (~1e-5 worst-case for resonant tf filters near the
scan gate) and beating scipy from one channel up. Short signals use a
per-channel-sequential kernel when there are enough channels, else scipy.
High-order clustered-pole tf filters keep the exact sequential kernel (the
scan's companion-form transition is non-normal there); complex coefficients
and orders past 16 fall back to scipy with a `FallbackWarning` — use the
better-conditioned SOS form, as scipy itself recommends. You will not
silently run on one core believing you're on 40 GPU cores.

## Dispatch: when the GPU is used

Kernel launches cost more than tiny problems do, so smaller inputs would *lose* on
the GPU. Every function routes automatically:

- **`dispatch="auto"`** (default): inputs above `gpu_min_size` (default 2^15
  elements of work) run on MLX; smaller inputs run scipy. Both return identical
  MLX-array types/dtypes, so the routing is invisible.
- **`dispatch="mlx"`**: always MLX; calls with no MLX path raise
  `NotImplementedError` instead of falling back (pin the GPU in tests).
- **`dispatch="scipy"`**: scipy numerical kernels (the correctness reference for
  canonical float32/complex64 operands; still returns MLX arrays). It does not
  opt back into float64 SOS arithmetic.

```python
sig.set_config(dispatch="mlx")                  # global
with sig.config_context(gpu_min_size=1 << 18):  # scoped
    ...
```

Capability fallbacks (complex IIR coefficients, tf orders past 16, callable
detrend, boundary modes on signals shorter than their extension) warn loudly;
size-based routing is silent by design.

## Dtype policy

Metal has no float64, period. Computation is **float32/complex64**. Explicit
float64/complex128 signal and state arrays downcast with a one-time `DowncastWarning`
(`set_config(float64="strict")` to raise instead; `warn_on_downcast=False` to
hush). Golden tests hold the fp32 pipeline to `rtol=1e-4` with a peak-relative
`atol≈1e-5` against float64 scipy references — honest fp32 accuracy, documented
rather than hidden. SciPy filter-design routines naturally return tiny float64
SOS coefficient arrays; these canonicalize quietly to float32 by default (strict
mode rejects them), and a design that becomes unstable or loses a section's
numerator during quantization raises before filtering. Explicit complex128 or
extended-precision SOS arrays follow the downcast-warning policy. With scipy
older than 1.15, scan-unsafe `auto` SOS calls stay on scipy because that
backend's historical float32 recurrence order differs from the Metal kernel
(transfer-function `lfilter` probes the installed scipy's rounding directly —
it is a compiler property, not a version — and applies the same conservative
routing on a non-matching build). One hardware caveat: Apple GPUs flush
float32 denormals (magnitudes below ~1.2e-38) to zero where CPU scipy keeps
them, so bit-identity claims for the IIR kernels hold for normal-range data.

## Known limitations

- **MLX 0.32 Metal FFT above 2^20 is broken upstream** — lengths in
  (2^19, 2^21] except 2^20 *crash* ("Unable to load function four_step_mem_…"),
  and, worse, other lengths above 2^20 *silently return wrong values*
  (rel. error ~1.0). mlx-signal verified the safe region empirically and works
  around it on the GPU: 1-D transforms use an in-library four-step (Bailey)
  decomposition into safe-size sub-FFTs (`_fourstep.py`), and long×short
  `fftconvolve` switches to blocked overlap-add. Equal-size real pairs and
  autocorrelations at these lengths stay in the even/odd packed domain end to
  end — the product spectrum's inverse-transform input is computed directly
  from the packed forward transforms, so the untangle passes never run. Only unsplittable lengths
  (large primes) and the N-d FFT paths route through the MLX CPU stream
  (`_fft_core.py`). When MLX
  fixes this, relaxing one predicate retires the workaround. (Worth reporting
  upstream to `ml-explore/mlx` if you can reproduce it.)
- Windows/filter design (`get_window`, `firwin*`) and `find_peaks`' index
  refinement run host-side — tiny work, and it keeps exact scipy parity
  (the prominence base-search itself runs on the GPU for f32 sources).
- `lfilter`/`sosfilt` take and return `zi`/`zf` state natively on the GPU for
  IIR filters; FIR `lfilter` with `zi` follows scipy's convolution path on the
  CPU (falls back, warns).
- `upfirdn` serves every scipy signal-extension mode on the GPU by
  pre-extending the boundary on-device; only signals shorter than the
  required boundary extension (roughly the tap count over `up`, rounded up
  for downsample-phase alignment) fall back.
- Streaming/chunked APIs, `ShortTimeFFT`, and CWT are not yet implemented (below).

## Roadmap

- **CWT** — scipy *removed* `cwt` in 1.15, so this is a differentiator, not a clone
- Modern `ShortTimeFFT` class
- fp16 mode; real-time streaming API; torchaudio benchmark column

## Examples

- [`examples/fm_demod.py`](examples/fm_demod.py) — an SDR FM demodulation
  chain (channel filter → polyphase decimate → discriminator → de-emphasis →
  audio resample), 16.5x end-to-end vs scipy on an M4 Max, with 0.999
  correlation to the true message.
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
  matched against scipy.signal (BSD-3-Clause) and are verified by the parity suite.
- **MLX** — the lazy, unified-memory array framework that makes the zero-copy
  story possible.
