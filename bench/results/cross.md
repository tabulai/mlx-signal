Cross-library benchmarks on Apple M4 Max (macOS 26.2), float32.
scipy 1.18.0, mlx 0.32.0, torch 2.13.0 (12 CPU threads), torchaudio 2.11.0, jax 0.11.0, librosa 0.11.0, soxr 1.1.0.

e2e = NumPy in / NumPy out; device = data resident on the accelerator. 'vs scipy' verifies each output against the scipy reference (max relative error, or aligned multi-channel quality for resamplers). Rows marked (1 run) exceeded the 3 s guard.

### welch — 64ch x 2^20, nperseg=1024 (no torch/librosa equivalent exists)

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy.signal.welch | CPU (1 thread) | 505.87 ms | — | 1.0x | — |
| mlx-signal.welch | Metal GPU | 8.40 ms | 4.21 ms | 60.2x | 3e-07 |
| jax.scipy.signal.welch (jit) | CPU (XLA) | 136.80 ms | — | 3.7x | 2e-07 |

### stft — 16ch x 2^20, n_fft=1024, hop=512, center=False, periodic hann

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy.signal.stft | CPU (1 thread) | 46.21 ms | — | 1.0x | — |
| mlx-signal.stft | Metal GPU | 6.80 ms | 0.93 ms | 6.8x | 2e-07 |
| torch.stft | CPU (multithread) | 25.34 ms | — | 1.8x | 2e-07 |
| torch.stft | MPS GPU | 4.19 ms | 2.61 ms | 11.0x | 5e-07 |
| librosa.stft | CPU (numpy) | 64.14 ms | — | 0.7x | 2e-07 |
| jax.scipy.signal.stft (jit) | CPU (XLA) | 43.38 ms | — | 1.1x | 3e-07 |

### fftconvolve — 2^20 x 4097, mode=full

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy.signal.fftconvolve | CPU (1 thread) | 11.40 ms | — | 1.0x | — |
| mlx-signal.fftconvolve | Metal GPU | 1.68 ms | 1.45 ms | 6.8x | 8e-07 |
| torchaudio fftconvolve | CPU (multithread) | 39.38 ms | — | 0.3x | 5e-07 |
| torchaudio fftconvolve | MPS GPU | 4.27 ms | 3.88 ms | 2.7x | 3e-06 |
| jax fftconvolve (jit) | CPU (XLA) | 15.86 ms | — | 0.7x | 6e-07 |

### resample 48 kHz -> 44.1 kHz — 16ch x 2^20 (task-level: each library's default anti-aliasing filter; scipy/mlx use 3201 taps, torchaudio width=6)

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy resample_poly (147/160) | CPU (1 thread) | 122.98 ms | — | 1.0x | — |
| mlx-signal resample_poly | Metal GPU | 5.74 ms | 3.86 ms | 21.4x | 4e-07 |
| torchaudio resample (width=6) | CPU (multithread) | 8.34 ms | — | 14.7x | r=0.9975, nrmse=0.073 |
| torchaudio resample (width=6) | MPS GPU | 5.50 ms | 3.30 ms | 22.4x | r=0.9975, nrmse=0.073 |
| librosa.resample (soxr_hq) | CPU (C) | 57.10 ms | — | 2.2x | r=0.9890, nrmse=0.148 |
| soxr (HQ) | CPU (C) | 53.12 ms | — | 2.3x | r=0.9890, nrmse=0.148 |

### causal FIR filter — 64ch x 2^20, 257 taps

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy.signal.lfilter | CPU (1 thread) | 1642.70 ms | — | 1.0x | — |
| mlx-signal.lfilter (FIR) | Metal GPU | 14.00 ms | 5.12 ms | 117.4x | 4e-07 |
| torch conv1d (causal FIR) | CPU (multithread) | 2855.47 ms | — | 0.6x | 9e-07 |
| torch conv1d (causal FIR) | MPS GPU | 70.14 ms | 65.10 ms | 23.4x | 9e-07 |
| torchaudio lfilter (IIR machinery) | CPU (multithread) | 3612.76 ms (1 run) | — | 0.5x | 9e-07 |
| torchaudio lfilter (IIR machinery) | MPS GPU | 21980.46 ms (1 run) | — | 0.1x | 9e-07 |
