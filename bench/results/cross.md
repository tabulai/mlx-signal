Cross-library benchmarks on Apple M4 Max (macOS 26.2), float32.
scipy 1.18.1, mlx 0.32.2, torch 2.13.0 (12 CPU threads), torchaudio 2.11.0, jax 0.11.1, librosa 1.0.0, soxr 1.1.0.

e2e = NumPy in / NumPy out; device = data resident on the accelerator. 'vs scipy' verifies each output against the scipy reference (max relative error, or aligned multi-channel quality for resamplers). Median of 5 runs after 2 warmups; rows marked (1 run) exceeded the 3 s guard.

### welch — 64ch x 2^20, nperseg=1024 (no torch/librosa equivalent exists)

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy.signal.welch | CPU (1 thread) | 518.05 ms | — | 1.0x | — |
| mlx-signal.welch | Metal GPU | 8.39 ms | 4.75 ms | 61.8x | 3e-07 |
| jax.scipy.signal.welch (jit) | CPU (XLA) | 48.01 ms | — | 10.8x | 2e-07 |

### stft — 16ch x 2^20, n_fft=1024, hop=512, center=False, periodic hann

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy.signal.stft | CPU (1 thread) | 47.29 ms | — | 1.0x | — |
| mlx-signal.stft | Metal GPU | 4.28 ms | 0.92 ms | 11.0x | 2e-07 |
| torch.stft | CPU (multithread) | 26.20 ms | — | 1.8x | 2e-07 |
| torch.stft | MPS GPU | 4.35 ms | 2.47 ms | 10.9x | 5e-07 |
| librosa.stft | CPU (numpy) | 66.35 ms | — | 0.7x | 2e-07 |
| jax.scipy.signal.stft (jit) | CPU (XLA) | 22.75 ms | — | 2.1x | 3e-07 |

### fftconvolve — 2^20 x 4097, mode=full

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy.signal.fftconvolve | CPU (1 thread) | 11.58 ms | — | 1.0x | — |
| mlx-signal.fftconvolve | Metal GPU | 1.33 ms | 0.52 ms | 8.7x | 8e-07 |
| torchaudio fftconvolve | CPU (multithread) | 39.38 ms | — | 0.3x | 5e-07 |
| torchaudio fftconvolve | MPS GPU | 5.29 ms | 3.96 ms | 2.2x | 3e-06 |
| jax fftconvolve (jit) | CPU (XLA) | 16.09 ms | — | 0.7x | 5e-07 |

### resample 48 kHz -> 44.1 kHz — 16ch x 2^20 (task-level: each library's default anti-aliasing filter; scipy/mlx use 3201 taps, torchaudio width=6)

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy resample_poly (147/160) | CPU (1 thread) | 126.09 ms | — | 1.0x | — |
| mlx-signal resample_poly | Metal GPU | 3.35 ms | 0.99 ms | 37.7x | 4e-07 |
| torchaudio resample (width=6) | CPU (multithread) | 8.61 ms | — | 14.6x | r=0.9975, nrmse=0.073 |
| torchaudio resample (width=6) | MPS GPU | 6.31 ms | 3.34 ms | 20.0x | r=0.9975, nrmse=0.073 |
| librosa.resample (soxr_hq) | CPU (C) | 57.09 ms | — | 2.2x | r=0.9890, nrmse=0.148 |
| soxr (HQ) | CPU (C) | 54.14 ms | — | 2.3x | r=0.9890, nrmse=0.148 |

### causal FIR filter — 64ch x 2^20, 257 taps

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy.signal.lfilter | CPU (1 thread) | 1661.99 ms | — | 1.0x | — |
| mlx-signal.lfilter (FIR) | Metal GPU | 13.80 ms | 5.86 ms | 120.5x | 4e-07 |
| torch conv1d (causal FIR) | CPU (multithread) | 2814.87 ms | — | 0.6x | 9e-07 |
| torch conv1d (causal FIR) | MPS GPU | 76.74 ms | 71.72 ms | 21.7x | 9e-07 |
| torchaudio lfilter (IIR machinery) | CPU (multithread) | 3435.53 ms (1 run) | — | 0.5x | 9e-07 |
| torchaudio lfilter (IIR machinery) | MPS GPU | 22166.62 ms (1 run) | — | 0.1x | 9e-07 |
