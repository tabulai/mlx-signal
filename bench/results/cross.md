Cross-library benchmarks on Apple M4 Max (macOS 26.2), float32.
scipy 1.18.0, mlx 0.32.0, torch 2.13.0 (12 CPU threads), torchaudio 2.11.0, jax 0.11.0, librosa 0.11.0, soxr 1.1.0.

e2e = NumPy in / NumPy out; device = data resident on the accelerator. 'vs scipy' verifies each output against the scipy reference (max relative error, or best-lag correlation for resamplers). Rows marked (1 run) exceeded the 3 s guard.

### welch — 64ch x 2^20, nperseg=1024 (no torch/librosa equivalent exists)

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy.signal.welch | CPU (1 thread) | 490.68 ms | — | 1.0x | — |
| mlx-signal.welch | Metal GPU | 8.48 ms | 4.18 ms | 57.9x | 3e-07 |
| jax.scipy.signal.welch (jit) | CPU (XLA) | 134.68 ms | — | 3.6x | 2e-07 |

### stft — 16ch x 2^20, n_fft=1024, hop=512, center=False, periodic hann

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy.signal.stft | CPU (1 thread) | 45.02 ms | — | 1.0x | — |
| mlx-signal.stft | Metal GPU | 4.35 ms | 0.95 ms | 10.4x | 2e-07 |
| torch.stft | CPU (multithread) | 25.07 ms | — | 1.8x | 2e-07 |
| torch.stft | MPS GPU | 3.85 ms | 3.42 ms | 11.7x | 5e-07 |
| librosa.stft | CPU (numpy) | 62.65 ms | — | 0.7x | 2e-07 |
| jax.scipy.signal.stft (jit) | CPU (XLA) | 42.61 ms | — | 1.1x | 3e-07 |

### fftconvolve — 2^20 x 4097, mode=full

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy.signal.fftconvolve | CPU (1 thread) | 10.98 ms | — | 1.0x | — |
| mlx-signal.fftconvolve | Metal GPU | 1.56 ms | 0.47 ms | 7.0x | 8e-07 |
| torchaudio fftconvolve | CPU (multithread) | 37.62 ms | — | 0.3x | 5e-07 |
| torchaudio fftconvolve | MPS GPU | 4.29 ms | 3.88 ms | 2.6x | 3e-06 |
| jax fftconvolve (jit) | CPU (XLA) | 15.44 ms | — | 0.7x | 6e-07 |

### resample 48 kHz -> 44.1 kHz — 16ch x 2^20 (task-level: each library's default anti-aliasing filter; scipy/mlx use 3201 taps, torchaudio width=6)

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy resample_poly (147/160) | CPU (1 thread) | 120.75 ms | — | 1.0x | — |
| mlx-signal resample_poly | Metal GPU | 6.06 ms | 3.89 ms | 19.9x | 4e-07 |
| torchaudio resample (width=6) | CPU (multithread) | 8.24 ms | — | 14.6x | r=0.9976 |
| torchaudio resample (width=6) | MPS GPU | 5.42 ms | 3.07 ms | 22.3x | r=0.9976 |
| librosa.resample (soxr_hq) | CPU (C) | 54.84 ms | — | 2.2x | r=0.9891 |
| soxr (HQ) | CPU (C) | 51.44 ms | — | 2.3x | r=0.9891 |

### causal FIR filter — 64ch x 2^20, 257 taps

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy.signal.lfilter | CPU (1 thread) | 1607.95 ms | — | 1.0x | — |
| mlx-signal.lfilter (FIR) | Metal GPU | 12.95 ms | 4.84 ms | 124.2x | 4e-07 |
| torch conv1d (causal FIR) | CPU (multithread) | 2916.20 ms | — | 0.6x | 9e-07 |
| torch conv1d (causal FIR) | MPS GPU | 75.24 ms | 64.25 ms | 21.4x | 9e-07 |
| torchaudio lfilter (IIR machinery) | CPU (multithread) | 3603.08 ms (1 run) | — | 0.4x | 9e-07 |
| torchaudio lfilter (IIR machinery) | MPS GPU | 21210.31 ms (1 run) | — | 0.1x | 9e-07 |
