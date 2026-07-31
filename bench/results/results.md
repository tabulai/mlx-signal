Benchmarks on Apple M4 Max (macOS 26.2), mlx 0.32.0, scipy 1.18.0, float32 inputs. e2e = NumPy in / NumPy out; device = MLX arrays in and out (steady state).

| function | shape | scipy | mlx-signal (e2e) | mlx-signal (device) | speedup (e2e / device) |
|---|---|---:|---:|---:|---:|
| welch | 64ch x 2^20, nperseg=1024 | 479.35 ms | 8.31 ms | 4.19 ms | 57.7x / 114.4x |
| welch | 1ch x 2^22, nperseg=4096 | 47.18 ms | 2.05 ms | 1.46 ms | 23.1x / 32.4x |
| welch | 256ch x 2^16, nperseg=256 | 110.44 ms | 2.39 ms | 1.28 ms | 46.2x / 86.2x |
| spectrogram | 16ch x 2^20 | 65.68 ms | 3.64 ms | 1.24 ms | 18.0x / 53.1x |
| stft | 16ch x 2^20, nperseg=1024 | 79.81 ms | 4.74 ms | 1.22 ms | 16.8x / 65.2x |
| fftconvolve | 2^20 x 4097 | 10.94 ms | 1.61 ms | 1.49 ms | 6.8x / 7.3x |
| fftconvolve | 2^22 x 257 | 44.10 ms | 2.76 ms | 0.94 ms | 16.0x / 46.8x |
| oaconvolve | 2^23 x 513 | 26.35 ms | 3.75 ms | 2.47 ms | 7.0x / 10.7x |
| correlate (batched) | 64ch x 2^18, 4096 taps | 63.55 ms | 8.41 ms | 5.89 ms | 7.6x / 10.8x |
| resample_poly | 16ch, 48k->44.1k (147/160) | 118.99 ms | 6.34 ms | 4.13 ms | 18.8x / 28.8x |
| upfirdn | 64ch x 2^18, up=2 down=3, 255 taps | 294.67 ms | 8.94 ms | 6.92 ms | 33.0x / 42.6x |
| resample (FFT) | 2^20 -> 2^18 | 4.17 ms | 0.36 ms | 0.29 ms | 11.6x / 14.4x |
| hilbert | 2^20 | 9.26 ms | 1.56 ms | 0.57 ms | 5.9x / 16.3x |
| resample (FFT, >1M) | 2^23 -> x0.75 | 65.71 ms | 6.12 ms | 5.33 ms | 10.7x / 12.3x |
| hilbert (>1M) | 2^23 | 100.81 ms | 8.14 ms | 6.08 ms | 12.4x / 16.6x |
| lfilter (FIR) | 64ch x 2^20, 257 taps | 1590.46 ms | 17.45 ms | 9.46 ms | 91.2x / 168.1x |
| filtfilt (FIR) | 64ch x 2^20, 257 taps | 3213.42 ms | 43.48 ms | 22.76 ms | 73.9x / 141.2x |
| istft | 16ch x 2^20, nperseg=1024 | 174.41 ms | 22.93 ms | 3.19 ms | 7.6x / 54.6x |
| csd | 64ch x 2^20, nperseg=1024 | 924.37 ms | 15.97 ms | 8.03 ms | 57.9x / 115.1x |
| coherence | 64ch x 2^20, nperseg=1024 | 1910.62 ms | 17.65 ms | 9.58 ms | 108.3x / 199.5x |
| sosfilt (IIR) | 256ch x 2^20, butter-8 | 1289.75 ms | 42.26 ms | 11.72 ms | 30.5x / 110.0x |
| sosfilt (IIR, 1ch scan) | 1ch x 2^22, butter-8 | 20.28 ms | 1.64 ms | 1.20 ms | 12.3x / 16.9x |
| sosfiltfilt (IIR) | 256ch x 2^20, butter-8 | 2651.99 ms | 125.99 ms | 39.84 ms | 21.0x / 66.6x |
| find_peaks | 2^23, prominence=1 | 217.28 ms | 215.50 ms | — | 1.0x / — |
