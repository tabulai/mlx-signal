Benchmarks on Apple M4 Max (macOS 26.2), mlx 0.32.0, scipy 1.18.0, float32 inputs. e2e = NumPy in / NumPy out; device = MLX arrays in and out (steady state).

| function | shape | scipy | mlx-signal (e2e) | mlx-signal (device) | speedup (e2e / device) |
|---|---|---:|---:|---:|---:|
| welch | 64ch x 2^20, nperseg=1024 | 489.75 ms | 27.86 ms | 24.05 ms | 17.6x / 20.4x |
| welch | 1ch x 2^22, nperseg=4096 | 48.16 ms | 2.12 ms | 1.28 ms | 22.7x / 37.5x |
| welch | 256ch x 2^16, nperseg=256 | 113.28 ms | 6.56 ms | 5.60 ms | 17.3x / 20.2x |
| spectrogram | 16ch x 2^20 | 68.24 ms | 8.11 ms | 5.69 ms | 8.4x / 12.0x |
| stft | 16ch x 2^20, nperseg=1024 | 78.97 ms | 6.84 ms | 3.31 ms | 11.6x / 23.8x |
| fftconvolve | 2^20 x 4097 | 11.25 ms | 0.66 ms | 0.54 ms | 17.0x / 20.7x |
| fftconvolve | 2^22 x 257 | 49.06 ms | 2.02 ms | 1.00 ms | 24.3x / 49.0x |
| oaconvolve | 2^23 x 513 | 27.04 ms | 3.64 ms | 2.61 ms | 7.4x / 10.4x |
| correlate (batched) | 64ch x 2^18, 4096 taps | 65.42 ms | 8.54 ms | 6.03 ms | 7.7x / 10.8x |
| resample_poly | 16ch, 48k->44.1k (147/160) | 120.66 ms | 6.37 ms | 4.17 ms | 18.9x / 28.9x |
| upfirdn | 64ch x 2^18, up=2 down=3, 255 taps | 302.81 ms | 8.95 ms | 6.92 ms | 33.8x / 43.7x |
| resample (FFT) | 2^20 -> 2^18 | 4.28 ms | 0.33 ms | 0.26 ms | 13.0x / 16.2x |
| hilbert | 2^20 | 9.84 ms | 0.69 ms | 0.52 ms | 14.2x / 19.0x |
| resample (FFT, >1M: CPU-FFT) | 2^23 -> x0.75 | 67.30 ms | 64.64 ms | 63.61 ms | 1.0x / 1.1x |
| hilbert (>1M: CPU-FFT) | 2^23 | 103.04 ms | 178.13 ms | 177.48 ms | 0.6x / 0.6x |
| lfilter (FIR) | 64ch x 2^20, 257 taps | 1627.23 ms | 17.46 ms | 9.85 ms | 93.2x / 165.2x |
| filtfilt (FIR) | 64ch x 2^20, 257 taps | 3257.67 ms | 44.92 ms | 23.77 ms | 72.5x / 137.1x |
| find_peaks | 2^23, prominence=1 | 219.65 ms | 216.21 ms | — | 1.0x / — |
