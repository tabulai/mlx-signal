Benchmarks on Apple M4 Max (macOS 26.2), mlx 0.32.0, scipy 1.18.0, float32 inputs. e2e = NumPy in / NumPy out; device = MLX arrays in and out (steady state).

| function | shape | scipy | mlx-signal (e2e) | mlx-signal (device) | speedup (e2e / device) |
|---|---|---:|---:|---:|---:|
| welch | 64ch x 2^20, nperseg=1024 | 480.88 ms | 8.38 ms | 4.28 ms | 57.4x / 112.3x |
| welch | 1ch x 2^22, nperseg=4096 | 48.06 ms | 1.69 ms | 1.45 ms | 28.4x / 33.2x |
| welch | 256ch x 2^16, nperseg=256 | 112.47 ms | 2.00 ms | 1.06 ms | 56.3x / 106.1x |
| spectrogram | 16ch x 2^20 | 68.74 ms | 6.61 ms | 1.27 ms | 10.4x / 53.9x |
| stft | 16ch x 2^20, nperseg=1024 | 78.12 ms | 4.42 ms | 1.29 ms | 17.7x / 60.7x |
| fftconvolve | 2^20 x 4097 | 11.36 ms | 1.62 ms | 0.53 ms | 7.0x / 21.6x |
| fftconvolve | 2^22 x 257 | 47.86 ms | 2.64 ms | 2.01 ms | 18.1x / 23.8x |
| oaconvolve | 2^23 x 513 | 27.07 ms | 3.74 ms | 2.63 ms | 7.2x / 10.3x |
| correlate (batched) | 64ch x 2^18, 4096 taps | 64.61 ms | 8.25 ms | 6.04 ms | 7.8x / 10.7x |
| resample_poly | 16ch, 48k->44.1k (147/160) | 120.57 ms | 6.32 ms | 4.25 ms | 19.1x / 28.4x |
| upfirdn | 64ch x 2^18, up=2 down=3, 255 taps | 302.58 ms | 8.86 ms | 6.96 ms | 34.2x / 43.4x |
| resample (FFT) | 2^20 -> 2^18 | 4.18 ms | 0.35 ms | 0.29 ms | 12.1x / 14.4x |
| hilbert | 2^20 | 9.66 ms | 0.68 ms | 0.49 ms | 14.2x / 19.7x |
| resample (FFT, >1M) | 2^23 -> x0.75 | 66.47 ms | 7.25 ms | 5.48 ms | 9.2x / 12.1x |
| hilbert (>1M) | 2^23 | 102.61 ms | 8.07 ms | 6.48 ms | 12.7x / 15.8x |
| lfilter (FIR) | 64ch x 2^20, 257 taps | 1626.61 ms | 17.48 ms | 9.77 ms | 93.0x / 166.5x |
| filtfilt (FIR) | 64ch x 2^20, 257 taps | 3256.47 ms | 44.21 ms | 23.51 ms | 73.7x / 138.5x |
| find_peaks | 2^23, prominence=1 | 215.94 ms | 214.65 ms | — | 1.0x / — |
