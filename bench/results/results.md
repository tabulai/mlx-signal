Benchmarks on Apple M4 Max (macOS 26.2), mlx 0.32.0, scipy 1.18.0, float32 inputs. e2e = NumPy in / NumPy out; device = MLX arrays in and out (steady state).

| function | shape | scipy | mlx-signal (e2e) | mlx-signal (device) | speedup (e2e / device) |
|---|---|---:|---:|---:|---:|
| welch | 64ch x 2^20, nperseg=1024 | 493.73 ms | 9.69 ms | 4.36 ms | 50.9x / 113.3x |
| welch | 1ch x 2^22, nperseg=4096 | 48.08 ms | 2.01 ms | 0.61 ms | 23.9x / 79.4x |
| welch | 256ch x 2^16, nperseg=256 | 111.13 ms | 2.53 ms | 1.06 ms | 44.0x / 105.2x |
| spectrogram | 16ch x 2^20 | 66.81 ms | 3.57 ms | 1.25 ms | 18.7x / 53.6x |
| stft | 16ch x 2^20, nperseg=1024 | 79.78 ms | 4.75 ms | 1.22 ms | 16.8x / 65.2x |
| fftconvolve | 2^20 x 4097 | 10.93 ms | 1.61 ms | 1.40 ms | 6.8x / 7.8x |
| fftconvolve | 2^22 x 257 | 44.87 ms | 2.65 ms | 0.70 ms | 17.0x / 64.0x |
| fftconvolve (pair) | 2^20 x 2^20 | 20.13 ms | 2.63 ms | 0.94 ms | 7.7x / 21.5x |
| correlate (auto) | 2^20 autocorrelation | 20.26 ms | 1.96 ms | 0.60 ms | 10.3x / 33.8x |
| oaconvolve | 2^23 x 513 | 27.19 ms | 3.26 ms | 1.25 ms | 8.3x / 21.7x |
| correlate (batched) | 64ch x 2^18, 4096 taps | 63.66 ms | 6.99 ms | 4.59 ms | 9.1x / 13.9x |
| resample_poly | 16ch, 48k->44.1k (147/160) | 120.29 ms | 3.36 ms | 0.96 ms | 35.8x / 125.8x |
| upfirdn | 64ch x 2^18, up=2 down=3, 255 taps | 295.15 ms | 4.34 ms | 2.30 ms | 68.0x / 128.3x |
| upfirdn (complex IQ) | 16ch x 2^20 c64, down=10, 201 taps | 180.69 ms | 3.80 ms | 1.25 ms | 47.5x / 144.2x |
| resample (FFT) | 2^20 -> 2^18 | 4.25 ms | 0.55 ms | 0.35 ms | 7.7x / 12.1x |
| hilbert | 2^20 | 9.89 ms | 1.10 ms | 0.81 ms | 9.0x / 12.2x |
| resample (FFT, >1M) | 2^23 -> x0.75 | 66.45 ms | 6.44 ms | 5.61 ms | 10.3x / 11.8x |
| hilbert (>1M) | 2^23 | 102.26 ms | 8.57 ms | 6.43 ms | 11.9x / 15.9x |
| lfilter (FIR) | 64ch x 2^20, 257 taps | 1619.76 ms | 12.93 ms | 4.88 ms | 125.2x / 332.1x |
| filtfilt (FIR) | 64ch x 2^20, 257 taps | 3236.10 ms | 35.12 ms | 13.74 ms | 92.1x / 235.6x |
| istft | 16ch x 2^20, nperseg=1024 | 178.28 ms | 21.19 ms | 2.63 ms | 8.4x / 67.9x |
| csd | 64ch x 2^20, nperseg=1024 | 936.84 ms | 16.53 ms | 8.28 ms | 56.7x / 113.1x |
| coherence | 64ch x 2^20, nperseg=1024 | 1948.96 ms | 17.73 ms | 9.76 ms | 109.9x / 199.7x |
| sosfilt (IIR) | 256ch x 2^20, butter-8 | 1319.27 ms | 42.69 ms | 11.80 ms | 30.9x / 111.8x |
| sosfilt (IIR, 1ch scan) | 1ch x 2^22, butter-8 | 20.84 ms | 1.76 ms | 1.35 ms | 11.8x / 15.4x |
| sosfiltfilt (IIR) | 256ch x 2^20, butter-8 | 2698.31 ms | 127.03 ms | 41.67 ms | 21.2x / 64.7x |
| lfilter (IIR) | 256ch x 2^20, butter-4 tf | 1551.14 ms | 43.71 ms | 12.43 ms | 35.5x / 124.8x |
| lfilter (IIR, 1ch scan) | 1ch x 2^22, butter-4 tf | 23.31 ms | 1.39 ms | 0.95 ms | 16.7x / 24.5x |
| filtfilt (IIR) | 256ch x 2^20, butter-4 tf | 3090.99 ms | 128.11 ms | 41.22 ms | 24.1x / 75.0x |
| find_peaks | 2^23, prominence=1 | 224.22 ms | 223.08 ms | — | 1.0x / — |
