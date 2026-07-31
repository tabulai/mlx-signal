Cross-library benchmarks on Apple M4 Max (macOS 26.2), float32.
scipy 1.18.0, mlx 0.32.0, torch 2.13.0 (12 CPU threads), torchaudio 2.11.0, jax 0.11.0, librosa 0.11.0, soxr 1.1.0.

e2e = NumPy in / NumPy out; device = data resident on the accelerator. 'vs scipy' verifies each output against the scipy reference (max relative error, or best-lag correlation for resamplers). Rows marked (1 run) exceeded the 3 s guard.

### welch — 64ch x 2^20, nperseg=1024 (no torch/librosa equivalent exists)

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy.signal.welch | CPU (1 thread) | 489.66 ms | — | 1.0x | — |
| mlx-signal.welch | Metal GPU | 28.45 ms | 23.62 ms | 17.2x | 9e-07 |
| jax.scipy.signal.welch (jit) | CPU (XLA) | 133.42 ms | — | 3.7x | 2e-07 |

### stft — 16ch x 2^20, n_fft=1024, hop=512, center=False, periodic hann

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy.signal.stft | CPU (1 thread) | 46.96 ms | — | 1.0x | — |
| mlx-signal.stft | Metal GPU | 6.27 ms | 3.02 ms | 7.5x | 5e-07 |
| torch.stft | CPU (multithread) | 25.34 ms | — | 1.9x | 2e-07 |
| torch.stft | MPS GPU | 4.04 ms | 2.45 ms | 11.6x | 5e-07 |
| librosa.stft | CPU (numpy) | 63.04 ms | — | 0.7x | 2e-07 |
| jax.scipy.signal.stft (jit) | CPU (XLA) | 43.54 ms | — | 1.1x | 3e-07 |

### fftconvolve — 2^20 x 4097, mode=full

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy.signal.fftconvolve | CPU (1 thread) | 10.97 ms | — | 1.0x | — |
| mlx-signal.fftconvolve | Metal GPU | 1.64 ms | 1.37 ms | 6.7x | 8e-07 |
| torchaudio fftconvolve | CPU (multithread) | 37.21 ms | — | 0.3x | 5e-07 |
| torchaudio fftconvolve | MPS GPU | 4.04 ms | 3.67 ms | 2.7x | 3e-06 |
| jax fftconvolve (jit) | CPU (XLA) | 15.09 ms | — | 0.7x | 6e-07 |

### resample 48 kHz -> 44.1 kHz — 16ch x 2^20 (task-level: each library's default anti-aliasing filter; scipy/mlx use 3201 taps, torchaudio width=6)

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy resample_poly (147/160) | CPU (1 thread) | 120.01 ms | — | 1.0x | — |
| mlx-signal resample_poly | Metal GPU | 6.29 ms | 4.12 ms | 19.1x | 4e-07 |
| torchaudio resample (width=6) | CPU (multithread) | 8.70 ms | — | 13.8x | r=0.9976 |
| torchaudio resample (width=6) | MPS GPU | 5.49 ms | 3.09 ms | 21.9x | r=0.9976 |
| librosa.resample (soxr_hq) | CPU (C) | 55.95 ms | — | 2.1x | r=0.9891 |
| soxr (HQ) | CPU (C) | 51.72 ms | — | 2.3x | r=0.9891 |

### causal FIR filter — 64ch x 2^20, 257 taps

| implementation | backend | e2e | device | speedup vs scipy | vs scipy |
|---|---|---:|---:|---:|---|
| scipy.signal.lfilter | CPU (1 thread) | 1597.42 ms | — | 1.0x | — |
| mlx-signal.lfilter (FIR) | Metal GPU | 17.55 ms | 9.49 ms | 91.0x | 8e-07 |
| torch conv1d (causal FIR) | CPU (multithread) | 3830.71 ms (1 run) | — | 0.4x | 9e-07 |
| torch conv1d (causal FIR) | MPS GPU | 69.37 ms | 64.45 ms | 23.0x | 9e-07 |
| torchaudio lfilter (IIR machinery) | CPU (multithread) | 3579.75 ms (1 run) | — | 0.4x | 9e-07 |
| torchaudio lfilter (IIR machinery) | MPS GPU | 21494.70 ms (1 run) | — | 0.1x | 9e-07 |
