Benchmarks on Apple M4 Max (macOS 26.2), mlx 0.32.0, scipy 1.18.0, float32 inputs. e2e = NumPy in / NumPy out; device = MLX arrays in and out (steady state).

| function | shape | scipy | mlx-signal (e2e) | mlx-signal (device) | speedup (e2e / device) |
|---|---|---:|---:|---:|---:|
| welch | 64ch x 2^20, nperseg=1024 | 482.33 ms | 18.03 ms | 13.91 ms | 26.8x / 34.7x |
| welch | 1ch x 2^22, nperseg=4096 | 47.64 ms | 1.79 ms | 0.83 ms | 26.7x / 57.2x |
| welch | 256ch x 2^16, nperseg=256 | 111.86 ms | 5.04 ms | 3.56 ms | 22.2x / 31.4x |
| spectrogram | 16ch x 2^20 | 67.32 ms | 7.32 ms | 3.72 ms | 9.2x / 18.1x |
| stft | 16ch x 2^20, nperseg=1024 | 78.33 ms | 6.07 ms | 1.75 ms | 12.9x / 44.8x |
| fftconvolve | 2^20 x 4097 | 11.19 ms | 1.75 ms | 1.50 ms | 6.4x / 7.5x |
| fftconvolve | 2^22 x 257 | 47.83 ms | 2.93 ms | 1.73 ms | 16.3x / 27.7x |
| oaconvolve | 2^23 x 513 | 27.40 ms | 4.00 ms | 2.65 ms | 6.8x / 10.4x |
| correlate (batched) | 64ch x 2^18, 4096 taps | 65.16 ms | 8.38 ms | 6.06 ms | 7.8x / 10.7x |
| resample_poly | 16ch, 48k->44.1k (147/160) | 120.31 ms | 6.40 ms | 4.24 ms | 18.8x / 28.3x |
| upfirdn | 64ch x 2^18, up=2 down=3, 255 taps | 303.73 ms | 8.94 ms | 6.97 ms | 34.0x / 43.6x |
| resample (FFT) | 2^20 -> 2^18 | 4.27 ms | 0.39 ms | 0.30 ms | 11.0x / 14.4x |
| hilbert | 2^20 | 9.85 ms | 0.76 ms | 0.55 ms | 12.9x / 17.9x |
| resample (FFT, >1M: CPU-FFT) | 2^23 -> x0.75 | 67.09 ms | 65.66 ms | 64.22 ms | 1.0x / 1.0x |
| hilbert (>1M: CPU-FFT) | 2^23 | 103.32 ms | 180.36 ms | 177.34 ms | 0.6x / 0.6x |
| lfilter (FIR) | 64ch x 2^20, 257 taps | 1634.73 ms | 17.74 ms | 9.75 ms | 92.1x / 167.7x |
| filtfilt (FIR) | 64ch x 2^20, 257 taps | 3271.80 ms | 44.53 ms | 23.62 ms | 73.5x / 138.5x |
| find_peaks | 2^23, prominence=1 | 218.02 ms | 217.17 ms | — | 1.0x / — |
