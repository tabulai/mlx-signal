Cross-library benchmarks on Apple M4 Max (macOS 26.2), float32.
scipy 1.18.0, mlx 0.32.0, torch 2.13.0 (12 CPU threads), torchaudio 2.11.0, jax 0.11.0, librosa 0.11.0, soxr 1.1.0.

e2e = NumPy in / NumPy out; device = data resident on the accelerator. 'vs scipy' verifies each output against the scipy reference (max relative error, or best-lag correlation for resamplers). Rows marked (1 run) exceeded the 3 s guard.

### welch — 64ch x 2^20, nperseg=1024 (no torch/librosa equivalent exists)

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy.signal.welch | CPU (1 thread) | 484.91 ms | — | 1.0x | — |
| mlx-signal.welch | Metal GPU | 9.39 ms | 4.32 ms | 51.6x | 3e-07 |
| jax.scipy.signal.welch (jit) | CPU (XLA) | 137.05 ms | — | 3.5x | 2e-07 |

### stft — 16ch x 2^20, n_fft=1024, hop=512, center=False, periodic hann

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy.signal.stft | CPU (1 thread) | 45.40 ms | — | 1.0x | — |
| mlx-signal.stft | Metal GPU | 4.24 ms | 0.91 ms | 10.7x | 2e-07 |
| torch.stft | CPU (multithread) | 24.63 ms | — | 1.8x | 2e-07 |
| torch.stft | MPS GPU | 4.43 ms | 3.60 ms | 10.2x | 5e-07 |
| librosa.stft | CPU (numpy) | 64.08 ms | — | 0.7x | 2e-07 |
| jax.scipy.signal.stft (jit) | CPU (XLA) | 43.97 ms | — | 1.0x | 3e-07 |

### fftconvolve — 2^20 x 4097, mode=full

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy.signal.fftconvolve | CPU (1 thread) | 11.37 ms | — | 1.0x | — |
| mlx-signal.fftconvolve | Metal GPU | 1.74 ms | 1.29 ms | 6.5x | 8e-07 |
| torchaudio fftconvolve | CPU (multithread) | 37.81 ms | — | 0.3x | 5e-07 |
| torchaudio fftconvolve | MPS GPU | 4.52 ms | 3.98 ms | 2.5x | 3e-06 |
| jax fftconvolve (jit) | CPU (XLA) | 15.63 ms | — | 0.7x | 6e-07 |

### resample 48 kHz -> 44.1 kHz — 16ch x 2^20 (task-level: each library's default anti-aliasing filter; scipy/mlx use 3201 taps, torchaudio width=6)

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy resample_poly (147/160) | CPU (1 thread) | 122.99 ms | — | 1.0x | — |
| mlx-signal resample_poly | Metal GPU | 5.97 ms | 4.05 ms | 20.6x | 4e-07 |
| torchaudio resample (width=6) | CPU (multithread) | 8.81 ms | — | 14.0x | r=0.9976 |
| torchaudio resample (width=6) | MPS GPU | 5.87 ms | 3.40 ms | 20.9x | r=0.9976 |
| librosa.resample (soxr_hq) | CPU (C) | 57.06 ms | — | 2.2x | r=0.9891 |
| soxr (HQ) | CPU (C) | 53.84 ms | — | 2.3x | r=0.9891 |

### causal FIR filter — 64ch x 2^20, 257 taps

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy.signal.lfilter | CPU (1 thread) | 1661.99 ms | — | 1.0x | — |
| mlx-signal.lfilter (FIR) | Metal GPU | 12.58 ms | 5.02 ms | 132.1x | 4e-07 |
| torch conv1d (causal FIR) | CPU (multithread) | 2836.59 ms | — | 0.6x | 9e-07 |
| torch conv1d (causal FIR) | MPS GPU | 72.24 ms | 67.72 ms | 23.0x | 9e-07 |
| torchaudio lfilter (IIR machinery) | CPU (multithread) | 3441.80 ms (1 run) | — | 0.5x | 9e-07 |
| torchaudio lfilter (IIR machinery) | MPS GPU | 21877.89 ms (1 run) | — | 0.1x | 9e-07 |
