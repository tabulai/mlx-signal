# mlx-signal

**GPU-accelerated signal processing for Apple Silicon, with familiar
`scipy.signal` APIs.**

mlx-signal implements a practical subset of `scipy.signal` with
[MLX](https://github.com/ml-explore/mlx) and custom Metal kernels. It accepts
NumPy or MLX arrays and returns MLX arrays in Apple's unified memory.

```python
import numpy as np
import mlx_signal_processing as sig

x = np.random.randn(64, 1 << 20).astype(np.float32)
frequencies, power = sig.welch(x, fs=48_000, nperseg=1024)
power_np = np.array(power)
```

MLX pipelines can pass results straight into a model without a host-device
transfer. NumPy users get the same API and can convert the result with a normal
in-memory copy. Small jobs automatically stay on SciPy when a GPU launch would
cost more than it saves.

## Install

Requires Apple Silicon, macOS 14 or newer, and Python 3.10 or newer:

```bash
python -m pip install mlx-signal-processing
```

The distribution is named `mlx-signal-processing`; import it in Python as
`mlx_signal_processing`.

To work on mlx-signal from a checkout:

```bash
git clone https://github.com/tabulai/mlx-signal
cd mlx-signal
python -m pip install -e .   # or: uv pip install -e .
python -m pytest -q         # optional: run the SciPy parity suite
```

## Measured performance

Measured on an Apple M4 Max (macOS 26.2) with MLX 0.32.2, SciPy 1.18.1, and
float32 data. Every result passed a SciPy correctness check before timing.
Values are medians of 9 runs after 3 warmups:

- **e2e:** NumPy input and output, for drop-in use
- **device:** MLX input and output, for an on-device pipeline

See the
[full report](https://github.com/tabulai/mlx-signal/blob/main/bench/results/results.md),
or reproduce it with
`python bench/bench.py --warmup 3 --repeat 9`.

| function | shape | scipy | mlx-signal (e2e) | mlx-signal (device) | speedup (e2e / device) |
|---|---|---:|---:|---:|---:|
| welch | 64ch × 2^20, nperseg=1024 | 501.97 ms | 8.57 ms | 4.26 ms | **58.6x / 117.7x** |
| welch | 1ch × 2^22, nperseg=4096 | 49.52 ms | 1.83 ms | 0.58 ms | **27.1x / 85.7x** |
| welch | 256ch × 2^16, nperseg=256 | 117.02 ms | 2.23 ms | 1.09 ms | **52.5x / 107.7x** |
| csd | 64ch × 2^20, nperseg=1024 | 993.43 ms | 16.65 ms | 9.00 ms | **59.7x / 110.4x** |
| coherence | 64ch × 2^20, nperseg=1024 | 2040.15 ms | 19.19 ms | 10.67 ms | **106.3x / 191.2x** |
| spectrogram | 16ch × 2^20 | 69.37 ms | 3.59 ms | 1.29 ms | **19.3x / 53.6x** |
| stft | 16ch × 2^20, nperseg=1024 | 81.42 ms | 4.57 ms | 1.25 ms | **17.8x / 65.4x** |
| istft | 16ch × 2^20, nperseg=1024 | 189.38 ms | 18.92 ms | 1.46 ms | **10.0x / 129.3x** |
| fftconvolve | 2^20 × 4097 | 11.25 ms | 1.56 ms | 0.56 ms | **7.2x / 20.3x** |
| fftconvolve | 2^22 × 257 | 48.82 ms | 1.21 ms | 0.70 ms | **40.3x / 69.7x** |
| fftconvolve (pair) | 2^20 × 2^20 | 22.30 ms | 1.35 ms | 0.89 ms | **16.6x / 25.0x** |
| oaconvolve | 2^23 × 513 | 27.31 ms | 2.46 ms | 1.26 ms | **11.1x / 21.7x** |
| correlate (batched) | 64ch × 2^18, 4096 taps | 66.98 ms | 7.44 ms | 5.05 ms | **9.0x / 13.3x** |
| correlate (auto) | 2^20 autocorrelation | 22.78 ms | 1.69 ms | 0.56 ms | **13.5x / 40.8x** |
| resample_poly | 16ch, 48k→44.1k (147/160) | 124.10 ms | 3.55 ms | 1.07 ms | **35.0x / 116.4x** |
| upfirdn | 64ch × 2^18, up=2 down=3, 255 taps | 312.98 ms | 3.90 ms | 2.48 ms | **80.3x / 126.0x** |
| upfirdn (complex IQ) | 16ch × 2^20 c64, down=10, 201 taps | 186.36 ms | 3.79 ms | 1.41 ms | **49.2x / 132.4x** |
| resample (FFT) | 2^20 → 2^18 | 4.40 ms | 0.90 ms | 0.70 ms | **4.9x / 6.3x** |
| hilbert | 2^20 | 8.91 ms | 1.33 ms | 0.42 ms | **6.7x / 21.4x** |
| lfilter (FIR) | 64ch × 2^20, 257 taps | 1648.00 ms | 13.95 ms | 5.89 ms | **118.1x / 279.8x** |
| lfilter (IIR) | 256ch × 2^20, butter-4 tf | 1668.50 ms | 44.49 ms | 13.00 ms | **37.5x / 128.3x** |
| lfilter (IIR, single channel) | 1ch × 2^22, butter-4 tf | 24.59 ms | 1.54 ms | 0.93 ms | **16.0x / 26.6x** |
| sosfilt (IIR) | 256ch × 2^20, butter-8 | 1350.96 ms | 44.35 ms | 12.70 ms | **30.5x / 106.4x** |
| sosfilt (IIR, single channel) | 1ch × 2^22, butter-8 | 21.01 ms | 1.97 ms | 1.35 ms | **10.7x / 15.6x** |
| sosfiltfilt (IIR) | 256ch × 2^20, butter-8 | 2778.62 ms | 130.65 ms | 44.64 ms | **21.3x / 62.2x** |
| filtfilt (IIR) | 256ch × 2^20, butter-4 tf | 3304.42 ms | 131.38 ms | 45.64 ms | **25.2x / 72.4x** |
| filtfilt (FIR) | 64ch × 2^20, 257 taps | 3313.02 ms | 36.97 ms | 15.45 ms | **89.6x / 214.5x** |
| resample (FFT) >1M samples¹ | 2^23 → ×0.75 | 68.71 ms | 3.61 ms | 2.77 ms | **19.0x / 24.8x** |
| hilbert >1M samples¹ | 2^23 | 97.80 ms | 4.20 ms | 2.69 ms | **23.3x / 36.3x** |
| find_peaks | 2^23, prominence=1 | 227.18 ms | 92.43 ms | — | **2.5x**² |
| peak_prominences | 2^23, 2.8M peaks | 170.93 ms | 20.92 ms | — | **8.2x**² |

¹ MLX 0.32's Metal FFT fails at some lengths above 2^20. mlx-signal handles
them with its own four-step (Bailey) GPU decomposition. Power-of-two lengths
from 2^21 through 2^26 use three fused Metal passes: about 2x faster than the
composed path at 2^23 and 5x at 2^26, with better accuracy than MLX's large
FFT. Other factorable lengths use safe-size sub-FFTs; only lengths without a
safe factorization, such as large primes, use the CPU stream. See
[Known limitations](#known-limitations).

² The expensive prominence search in `find_peaks` runs on the GPU and matches
SciPy bit for bit. Index bookkeeping stays on the host, limiting the overall
speedup to about 2.5x.

## Comparison with other libraries

This end-to-end comparison uses the same machine and NumPy input/output. Shapes
and conventions are aligned, and every result is checked against SciPy. The
fastest result in each row is bold. See the
[full report](https://github.com/tabulai/mlx-signal/blob/main/bench/results/cross.md),
or reproduce it with
`uv sync --extra bench && uv run python bench/bench_cross.py`.

| task | scipy | **mlx-signal** | torch/ta CPU | torch/ta MPS | jax (jit, CPU) | librosa | soxr |
|---|---:|---:|---:|---:|---:|---:|---:|
| welch, 64ch × 2^20 | 518 ms | **8.4 ms** | — | — | 48 ms | — | — |
| stft, 16ch × 2^20 | 47 ms | **4.3 ms** | 26 ms | 4.4 ms | 23 ms | 66 ms | — |
| fftconvolve, 2^20 × 4097 | 12 ms | **1.3 ms** | 39 ms | 5.3 ms | 16 ms | — | — |
| resample 48k→44.1k, 16ch × 2^20 | 126 ms | **3.4 ms**¹ | 8.6 ms¹ | 6.3 ms¹ | — | 57 ms | 54 ms |
| causal FIR, 64ch × 2^20, 257 taps | 1662 ms | **14 ms** | 2815 ms² | 77 ms² | — | — | — |

¹ Torchaudio's default anti-aliasing filter is much shorter
(`lowpass_filter_width=6`, versus 3201 taps here). mlx-signal keeps SciPy's
default filter and matching output, yet still wins end to end. With arrays
already on the GPU, it is 3.4x faster (0.99 versus 3.34 ms).

² Torchaudio offers FFT convolution, but `lfilter` does not select it
automatically. The closest causal FIR operation is `conv1d` (O(n·k)), used in
the table. Torchaudio's general `lfilter` takes **3.4 s on CPU and 22.4 s on
MPS** for this case—more than 1600x slower than mlx-signal.

What the comparison shows:

- In this group, mlx-signal is the only GPU implementation of
  `welch`/`csd`/`coherence`. JAX runs those APIs on CPU, and torchaudio has no
  PSD estimator. The fused two-signal kernel computes coherence in one pass,
  compared with five in SciPy.
- `upfirdn` and `find_peaks` have no comparable implementation in the other
  libraries tested.
- `torch.stft` on MPS is close end to end (4.35 versus 4.28 ms) because data
  transfer dominates and the NumPy-to-GPU copy time varies. On-device,
  mlx-signal is 2.7x faster (0.92 versus 2.47 ms), without pulling in the 2 GB
  torch dependency.

The spectral hot path uses fused Metal kernels. STFT/Welch read each segment
once without materializing a frames array; ISTFT combines an inverse transform
with gather-based overlap-add. Shapes that do not fit the fused path use
zero-copy framing and compiled MLX operations.

## What's implemented (v0.1)

| area | functions | notes |
|---|---|---|
| spectral | `periodogram` `welch` `csd` `coherence` `spectrogram` `stft` `istft` | all SciPy windows, detrending, scaling, axes, and median averaging; fused power-of-two GPU paths, including two-signal CSD/coherence and inverse-plus-overlap-add ISTFT; batched FFT otherwise |
| convolution | `convolve` `fftconvolve` `oaconvolve` `correlate` `correlation_lags` | N-D, every mode, and complex data; long×short inputs block automatically; filters up to 1025 taps use fused FFT kernels; equal-input convolution and correlation skip a duplicate transform |
| resampling | `upfirdn` `resample` `resample_poly` `decimate` | complex-native GPU `upfirdn`; safe 32-bit indexing avoids emulated 64-bit divides (about 4x faster at high `up`); every SciPy signal-extension mode and statistical pad type handled on-device |
| filtering | `firwin` `firwin2` `lfilter` `filtfilt` `sosfilt` `sosfiltfilt` `hilbert` | GPU FIR, SOS-IIR, and transfer-function IIR, including single-channel data and native `zi`/`zf`; transfer-function order up to 16; SciPy-compatible edges; filter design stays on the host |
| peaks | `find_peaks` `peak_prominences` `peak_widths` | SciPy parity; prominence search on the GPU for float32 data, with index bookkeeping on the host |
| utilities | `get_window` `next_fast_len` | cached windows and power-of-two fast lengths |

### IIR filtering

`lfilter`, `filtfilt`, `sosfilt`, `sosfiltfilt`, and the default IIR path in
`decimate` run on the GPU, including single-channel inputs. SOS and
transfer-function filters support native `zi`/`zf`, so state can carry across
chunks. Transfer-function filters are supported through order 16.

Long signals use a block-parallel scan; shorter jobs use a sequential kernel
when worthwhile, then fall back to SciPy. In the table above, transfer-function
filtering reaches 128x at 256 channels and 27x for a single channel on-device.
The sequential path is bit-identical to matching SciPy float32 builds. The
parallel path typically matches to about 1e-6, with roughly 1e-5 worst-case
error for resonant filters near its routing threshold.

High-order filters with clustered poles stay on the sequential path. Complex
coefficients and transfer-function orders above 16 fall back to SciPy with a
`FallbackWarning`; use SOS form for those filters.

## Dispatch: when the GPU is used

GPU launches do not pay off for small inputs, so mlx-signal can choose the
backend for each call:

- **`dispatch="auto"`** (default) uses MLX above `gpu_min_size` (2^15 work
  elements by default) and SciPy below it.
- **`dispatch="mlx"`** always uses MLX and raises `NotImplementedError` when no
  MLX path exists.
- **`dispatch="scipy"`** uses SciPy's numerical kernels with canonical
  float32/complex64 inputs. It still returns MLX arrays and does not restore
  float64 SOS arithmetic.

All three modes return the same MLX types and dtypes.

```python
sig.set_config(dispatch="mlx")                  # global
with sig.config_context(gpu_min_size=1 << 18):  # scoped
    ...
```

Capability fallbacks issue a warning. Examples include complex IIR
coefficients, transfer-function orders above 16, callable detrending, signals
too short for a requested boundary extension, and exceptional non-finite
filter or padding cases. Size-based routing is silent.

## Dtype policy

- Computation uses **float32/complex64** because Metal does not support
  float64.
- Explicit float64/complex128 signal and state arrays downcast with a one-time
  `DowncastWarning`. Use `set_config(float64="strict")` to raise instead, or
  `warn_on_downcast=False` to disable the warning. Extended-precision SOS
  arrays follow the same rule.
- The parity suite compares the float32 pipeline with float64 SciPy references
  at `rtol=1e-4` and a peak-relative `atol≈1e-5`.
- SciPy's filter-design functions normally return small float64 SOS arrays.
  These convert quietly unless strict mode is enabled. Filtering stops with an
  error if conversion makes a design unstable or erases a section numerator.
- With SciPy older than 1.15, unsafe automatic SOS scans stay on SciPy because
  its historical float32 recurrence order differs from the Metal kernel.
  Transfer-function `lfilter` probes the installed SciPy build and takes the
  same conservative route when its compiler-dependent rounding differs.
- Apple GPUs flush float32 denormals below about 1.2e-38 to zero; CPU SciPy
  keeps them. IIR bit-identity therefore applies to normal-range data.

## Known limitations

- **MLX 0.32 has an upstream Metal FFT issue above 2^20.** Lengths in
  (2^19, 2^21], except 2^20, crash with
  `Unable to load function four_step_mem_…`; other lengths above 2^20 can
  return incorrect values with relative error around 1.0. mlx-signal uses its
  own four-step decomposition for 1-D transforms and blocked overlap-add for
  long×short convolution. Unsplittable lengths such as large primes, and N-D
  FFT paths, use MLX's CPU stream. The workaround can be removed when MLX fixes
  the affected range.
- Windows/filter design (`get_window`, `firwin*`) and `find_peaks`' index
  refinement run on the host. The prominence search itself runs on the GPU for
  float32 inputs.
- `lfilter`/`sosfilt` take and return `zi`/`zf` state natively on the GPU for
  IIR filters. FIR `lfilter` with `zi` uses SciPy's CPU convolution path and
  warns.
- `upfirdn` handles every SciPy extension mode on-device. It falls back for
  signals shorter than the required extension (roughly the tap count divided
  by `up`, adjusted for downsampling phase), and for non-finite taps with
  exceptional NaN/Inf edge semantics.
- Streaming APIs, `ShortTimeFFT`, and CWT are not yet implemented.

## Roadmap

- CWT (removed from SciPy in 1.15)
- Modern `ShortTimeFFT` class
- float16 mode
- Real-time streaming API
- Torchaudio benchmark coverage

## Examples

- [`examples/fm_demod.py`](https://github.com/tabulai/mlx-signal/blob/main/examples/fm_demod.py)
  — an SDR FM demodulation
  chain (channel filter → polyphase decimate → discriminator → de-emphasis →
  audio resample), 16.5x end-to-end versus SciPy on an M4 Max, with 0.999
  correlation to the true message.
- [`examples/eeg_bandpower.py`](https://github.com/tabulai/mlx-signal/blob/main/examples/eeg_bandpower.py)
  — 64-channel × 10-minute
  EEG alpha-band power via one batched `welch`, 6x versus SciPy including result
  readback.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
python -m pytest -q          # parity tests against SciPy
ruff check src tests
python bench/bench.py        # GPU benchmark
```

CI runs lint and the full test suite on GitHub's arm64 macOS runners. Tests
marked `gpu` require a Metal device; the rest also exercise MLX's CPU backend.

## Acknowledgments

- **SciPy** provides the API contract and numerical reference. Edge cases are
  matched against `scipy.signal` (BSD-3-Clause) in the parity suite.
- **MLX** provides the lazy, unified-memory array framework.
