Cross-library benchmarks on Apple M4 Max (macOS 26.2), float32.
scipy 1.18.0, mlx 0.32.0, torch 2.13.0 (12 CPU threads), torchaudio 2.11.0, jax 0.11.0, librosa 0.11.0, soxr 1.1.0.

e2e = NumPy in / NumPy out; device = data resident on the accelerator. 'vs scipy' verifies each output against the scipy reference (max relative error, or aligned multi-channel quality for resamplers). Rows marked (1 run) exceeded the 3 s guard.

### welch — 64ch x 2^20, nperseg=1024 (no torch/librosa equivalent exists)

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy.signal.welch | CPU (1 thread) | 512.91 ms | — | 1.0x | — |
| mlx-signal.welch | Metal GPU | 8.34 ms | 4.25 ms | 61.5x | 3e-07 |
| jax.scipy.signal.welch (jit) | CPU (XLA) | 137.56 ms | — | 3.7x | 2e-07 |

### stft — 16ch x 2^20, n_fft=1024, hop=512, center=False, periodic hann

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy.signal.stft | CPU (1 thread) | 45.98 ms | — | 1.0x | — |
| mlx-signal.stft | Metal GPU | 4.36 ms | 1.50 ms | 10.5x | 2e-07 |
| torch.stft | CPU (multithread) | 25.50 ms | — | 1.8x | 2e-07 |
| torch.stft | MPS GPU | 4.77 ms | 2.51 ms | 9.6x | 5e-07 |
| librosa.stft | CPU (numpy) | 64.12 ms | — | 0.7x | 2e-07 |
| jax.scipy.signal.stft (jit) | CPU (XLA) | 43.30 ms | — | 1.1x | 3e-07 |

### fftconvolve — 2^20 x 4097, mode=full

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy.signal.fftconvolve | CPU (1 thread) | 11.34 ms | — | 1.0x | — |
| mlx-signal.fftconvolve | Metal GPU | 1.54 ms | 1.42 ms | 7.4x | 8e-07 |
| torchaudio fftconvolve | CPU (multithread) | 38.95 ms | — | 0.3x | 5e-07 |
| torchaudio fftconvolve | MPS GPU | 4.37 ms | 3.88 ms | 2.6x | 3e-06 |
| jax fftconvolve (jit) | CPU (XLA) | 15.97 ms | — | 0.7x | 6e-07 |

### resample 48 kHz -> 44.1 kHz — 16ch x 2^20 (task-level: each library's default anti-aliasing filter; scipy/mlx use 3201 taps, torchaudio width=6)

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy resample_poly (147/160) | CPU (1 thread) | 122.92 ms | — | 1.0x | — |
| mlx-signal resample_poly | Metal GPU | 6.14 ms | 3.84 ms | 20.0x | 4e-07 |
| torchaudio resample (width=6) | CPU (multithread) | 8.74 ms | — | 14.1x | r=0.9975, nrmse=0.073 |
| torchaudio resample (width=6) | MPS GPU | 5.59 ms | 3.29 ms | 22.0x | r=0.9975, nrmse=0.073 |
| librosa.resample (soxr_hq) | CPU (C) | 56.72 ms | — | 2.2x | r=0.9890, nrmse=0.148 |
| soxr (HQ) | CPU (C) | 53.33 ms | — | 2.3x | r=0.9890, nrmse=0.148 |

### causal FIR filter — 64ch x 2^20, 257 taps

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy.signal.lfilter | CPU (1 thread) | 1654.77 ms | — | 1.0x | — |
| mlx-signal.lfilter (FIR) | Metal GPU | 14.39 ms | 5.32 ms | 115.0x | 4e-07 |
| torch conv1d (causal FIR) | CPU (multithread) | 2949.11 ms | — | 0.6x | 9e-07 |
| torch conv1d (causal FIR) | MPS GPU | 68.61 ms | 64.30 ms | 24.1x | 9e-07 |
| torchaudio lfilter (IIR machinery) | CPU (multithread) | 3628.23 ms (1 run) | — | 0.5x | 9e-07 |
| torchaudio lfilter (IIR machinery) | MPS GPU | 21657.04 ms (1 run) | — | 0.1x | 9e-07 |
