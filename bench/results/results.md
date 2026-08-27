Benchmarks on Apple M4 Max (macOS 26.2), mlx 0.32.2, scipy 1.18.1, float32 inputs. e2e = NumPy in / NumPy out; device = MLX arrays in and out (steady state).

| function | shape | scipy | mlx-signal (e2e) | mlx-signal (device) | speedup (e2e / device) |
|---|---|---:|---:|---:|---:|
| welch | 64ch x 2^20, nperseg=1024 | 460.40 ms | 10.99 ms | 4.81 ms | 41.9x / 95.8x |
| welch | 1ch x 2^22, nperseg=4096 | 45.45 ms | 2.05 ms | 0.61 ms | 22.2x / 74.5x |
| welch | 256ch x 2^16, nperseg=256 | 106.70 ms | 4.22 ms | 1.06 ms | 25.3x / 100.7x |
| spectrogram | 16ch x 2^20 | 63.80 ms | 3.68 ms | 1.27 ms | 17.3x / 50.1x |
| stft | 16ch x 2^20, nperseg=1024 | 76.32 ms | 4.73 ms | 1.22 ms | 16.1x / 62.3x |
| fftconvolve | 2^20 x 4097 | 10.61 ms | 1.55 ms | 1.38 ms | 6.8x / 7.7x |
| fftconvolve | 2^22 x 257 | 42.48 ms | 2.65 ms | 0.73 ms | 16.0x / 58.4x |
| fftconvolve (pair) | 2^20 x 2^20 | 20.49 ms | 2.67 ms | 0.92 ms | 7.7x / 22.3x |
| correlate (auto) | 2^20 autocorrelation | 19.78 ms | 2.12 ms | 0.66 ms | 9.3x / 29.9x |
| oaconvolve | 2^23 x 513 | 25.48 ms | 4.50 ms | 1.25 ms | 5.7x / 20.3x |
| correlate (batched) | 64ch x 2^18, 4096 taps | 61.70 ms | 7.17 ms | 4.60 ms | 8.6x / 13.4x |
| resample_poly | 16ch, 48k->44.1k (147/160) | 116.44 ms | 4.68 ms | 0.97 ms | 24.9x / 120.1x |
| upfirdn | 64ch x 2^18, up=2 down=3, 255 taps | 292.68 ms | 4.35 ms | 2.32 ms | 67.3x / 126.2x |
| upfirdn (complex IQ) | 16ch x 2^20 c64, down=10, 201 taps | 172.77 ms | 4.00 ms | 1.25 ms | 43.2x / 138.7x |
| resample (FFT) | 2^20 -> 2^18 | 4.09 ms | 0.46 ms | 0.78 ms | 9.0x / 5.3x |
| hilbert | 2^20 | 8.62 ms | 1.40 ms | 1.26 ms | 6.1x / 6.8x |
| resample (FFT, >1M) | 2^23 -> x0.75 | 63.00 ms | 3.96 ms | 3.15 ms | 15.9x / 20.0x |
| hilbert (>1M) | 2^23 | 96.53 ms | 5.95 ms | 4.61 ms | 16.2x / 20.9x |
| lfilter (FIR) | 64ch x 2^20, 257 taps | 1560.86 ms | 12.84 ms | 4.87 ms | 121.6x / 320.6x |
| filtfilt (FIR) | 64ch x 2^20, 257 taps | 3162.31 ms | 33.67 ms | 13.49 ms | 93.9x / 234.4x |
| istft | 16ch x 2^20, nperseg=1024 | 168.34 ms | 20.61 ms | 2.76 ms | 8.2x / 61.1x |
| csd | 64ch x 2^20, nperseg=1024 | 899.76 ms | 15.92 ms | 8.03 ms | 56.5x / 112.1x |
| coherence | 64ch x 2^20, nperseg=1024 | 1869.78 ms | 17.53 ms | 9.60 ms | 106.7x / 194.8x |
| sosfilt (IIR) | 256ch x 2^20, butter-8 | 1275.24 ms | 41.87 ms | 11.52 ms | 30.5x / 110.7x |
| sosfilt (IIR, 1ch scan) | 1ch x 2^22, butter-8 | 19.99 ms | 2.48 ms | 1.35 ms | 8.1x / 14.9x |
| sosfiltfilt (IIR) | 256ch x 2^20, butter-8 | 2617.35 ms | 128.90 ms | 39.74 ms | 20.3x / 65.9x |
| lfilter (IIR) | 256ch x 2^20, butter-4 tf | 1473.67 ms | 42.16 ms | 11.74 ms | 35.0x / 125.5x |
| lfilter (IIR, 1ch scan) | 1ch x 2^22, butter-4 tf | 21.83 ms | 3.76 ms | 0.91 ms | 5.8x / 24.0x |
| filtfilt (IIR) | 256ch x 2^20, butter-4 tf | 2935.32 ms | 125.64 ms | 40.00 ms | 23.4x / 73.4x |
| find_peaks | 2^23, prominence=1 | 209.45 ms | 79.09 ms | — | 2.6x / — |
| peak_prominences | 2^23, 2.8M peaks | 150.43 ms | 19.28 ms | — | 7.8x / — |
