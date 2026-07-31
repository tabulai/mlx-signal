Benchmarks on Apple M4 Max (macOS 26.2), mlx 0.32.0, scipy 1.18.0, float32 inputs. e2e = NumPy in / NumPy out; device = MLX arrays in and out (steady state).

| function | shape | scipy | mlx-signal (e2e) | mlx-signal (device) | speedup (e2e / device) |
|---|---|---:|---:|---:|---:|
| welch | 64ch x 2^20, nperseg=1024 | 495.03 ms | 8.46 ms | 4.31 ms | 58.5x / 115.0x |
| welch | 1ch x 2^22, nperseg=4096 | 49.57 ms | 1.92 ms | 1.26 ms | 25.9x / 39.3x |
| welch | 256ch x 2^16, nperseg=256 | 114.95 ms | 4.74 ms | 1.08 ms | 24.3x / 106.1x |
| spectrogram | 16ch x 2^20 | 68.65 ms | 6.82 ms | 1.28 ms | 10.1x / 53.7x |
| stft | 16ch x 2^20, nperseg=1024 | 81.38 ms | 4.72 ms | 1.28 ms | 17.3x / 63.8x |
| fftconvolve | 2^20 x 4097 | 11.29 ms | 1.61 ms | 1.44 ms | 7.0x / 7.8x |
| fftconvolve | 2^22 x 257 | 48.22 ms | 2.50 ms | 0.97 ms | 19.3x / 49.9x |
| oaconvolve | 2^23 x 513 | 27.56 ms | 3.87 ms | 2.68 ms | 7.1x / 10.3x |
| correlate (batched) | 64ch x 2^18, 4096 taps | 66.01 ms | 8.46 ms | 6.14 ms | 7.8x / 10.8x |
| resample_poly | 16ch, 48k->44.1k (147/160) | 121.81 ms | 6.55 ms | 4.23 ms | 18.6x / 28.8x |
| upfirdn | 64ch x 2^18, up=2 down=3, 255 taps | 307.10 ms | 8.93 ms | 7.00 ms | 34.4x / 43.8x |
| resample (FFT) | 2^20 -> 2^18 | 4.39 ms | 0.35 ms | 0.29 ms | 12.6x / 15.2x |
| hilbert | 2^20 | 10.02 ms | 0.73 ms | 0.51 ms | 13.7x / 19.5x |
| resample (FFT, >1M: CPU-FFT) | 2^23 -> x0.75 | 68.49 ms | 65.60 ms | 64.99 ms | 1.0x / 1.1x |
| hilbert (>1M: CPU-FFT) | 2^23 | 104.62 ms | 182.14 ms | 183.11 ms | 0.6x / 0.6x |
| lfilter (FIR) | 64ch x 2^20, 257 taps | 1642.30 ms | 18.12 ms | 9.87 ms | 90.6x / 166.3x |
| filtfilt (FIR) | 64ch x 2^20, 257 taps | 3282.47 ms | 45.00 ms | 23.81 ms | 73.0x / 137.9x |
| find_peaks | 2^23, prominence=1 | 222.10 ms | 218.74 ms | — | 1.0x / — |
