Benchmarks on Apple M4 Max (macOS 26.2), mlx 0.32.0, scipy 1.18.0, float32 inputs. e2e = NumPy in / NumPy out; device = MLX arrays in and out (steady state).

| function | shape | scipy | mlx-signal (e2e) | mlx-signal (device) | speedup (e2e / device) |
|---|---|---:|---:|---:|---:|
| welch | 64ch x 2^20, nperseg=1024 | 480.03 ms | 10.74 ms | 4.28 ms | 44.7x / 112.2x |
| welch | 1ch x 2^22, nperseg=4096 | 47.69 ms | 2.25 ms | 0.57 ms | 21.2x / 84.1x |
| welch | 256ch x 2^16, nperseg=256 | 110.30 ms | 5.09 ms | 1.04 ms | 21.7x / 105.6x |
| spectrogram | 16ch x 2^20 | 66.79 ms | 3.69 ms | 1.31 ms | 18.1x / 51.0x |
| stft | 16ch x 2^20, nperseg=1024 | 80.16 ms | 4.64 ms | 1.88 ms | 17.3x / 42.6x |
| fftconvolve | 2^20 x 4097 | 11.07 ms | 1.75 ms | 0.50 ms | 6.3x / 22.1x |
| fftconvolve | 2^22 x 257 | 43.99 ms | 2.67 ms | 0.71 ms | 16.5x / 61.7x |
| fftconvolve (pair) | 2^20 x 2^20 | 21.43 ms | 2.95 ms | 0.95 ms | 7.3x / 22.6x |
| correlate (auto) | 2^20 autocorrelation | 21.23 ms | 1.90 ms | 1.77 ms | 11.1x / 12.0x |
| oaconvolve | 2^23 x 513 | 26.80 ms | 2.21 ms | 1.25 ms | 12.1x / 21.4x |
| correlate (batched) | 64ch x 2^18, 4096 taps | 63.52 ms | 7.50 ms | 4.57 ms | 8.5x / 13.9x |
| resample_poly | 16ch, 48k->44.1k (147/160) | 119.35 ms | 6.04 ms | 3.81 ms | 19.8x / 31.4x |
| upfirdn | 64ch x 2^18, up=2 down=3, 255 taps | 296.87 ms | 7.47 ms | 5.50 ms | 39.7x / 54.0x |
| upfirdn (complex IQ) | 16ch x 2^20 c64, down=10, 201 taps | 180.70 ms | 3.79 ms | 1.28 ms | 47.7x / 141.7x |
| resample (FFT) | 2^20 -> 2^18 | 4.17 ms | 1.33 ms | 0.76 ms | 3.1x / 5.5x |
| hilbert | 2^20 | 9.30 ms | 1.88 ms | 1.37 ms | 5.0x / 6.8x |
| resample (FFT, >1M) | 2^23 -> x0.75 | 65.82 ms | 6.13 ms | 5.32 ms | 10.7x / 12.4x |
| hilbert (>1M) | 2^23 | 101.77 ms | 8.98 ms | 6.52 ms | 11.3x / 15.6x |
| lfilter (FIR) | 64ch x 2^20, 257 taps | 1598.32 ms | 12.96 ms | 4.84 ms | 123.4x / 330.3x |
| filtfilt (FIR) | 64ch x 2^20, 257 taps | 3217.71 ms | 34.36 ms | 13.52 ms | 93.6x / 237.9x |
| istft | 16ch x 2^20, nperseg=1024 | 173.53 ms | 21.45 ms | 2.31 ms | 8.1x / 75.1x |
| csd | 64ch x 2^20, nperseg=1024 | 926.92 ms | 15.95 ms | 8.13 ms | 58.1x / 114.1x |
| coherence | 64ch x 2^20, nperseg=1024 | 1914.06 ms | 18.00 ms | 9.63 ms | 106.3x / 198.7x |
| sosfilt (IIR) | 256ch x 2^20, butter-8 | 1294.55 ms | 42.12 ms | 11.74 ms | 30.7x / 110.3x |
| sosfilt (IIR, 1ch scan) | 1ch x 2^22, butter-8 | 20.84 ms | 1.67 ms | 1.16 ms | 12.5x / 17.9x |
| sosfiltfilt (IIR) | 256ch x 2^20, butter-8 | 2649.98 ms | 128.94 ms | 39.58 ms | 20.6x / 67.0x |
| find_peaks | 2^23, prominence=1 | 215.65 ms | 216.51 ms | — | 1.0x / — |
