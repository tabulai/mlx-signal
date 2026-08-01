Benchmarks on Apple M4 Max (macOS 26.2), mlx 0.32.0, scipy 1.18.0, float32 inputs. e2e = NumPy in / NumPy out; device = MLX arrays in and out (steady state).

| function | shape | scipy | mlx-signal (e2e) | mlx-signal (device) | speedup (e2e / device) |
|---|---|---:|---:|---:|---:|
| welch | 64ch x 2^20, nperseg=1024 | 485.87 ms | 8.31 ms | 4.19 ms | 58.5x / 116.0x |
| welch | 1ch x 2^22, nperseg=4096 | 48.35 ms | 1.91 ms | 0.55 ms | 25.3x / 88.2x |
| welch | 256ch x 2^16, nperseg=256 | 112.98 ms | 4.82 ms | 1.04 ms | 23.5x / 108.4x |
| spectrogram | 16ch x 2^20 | 66.59 ms | 3.75 ms | 1.26 ms | 17.7x / 53.0x |
| stft | 16ch x 2^20, nperseg=1024 | 81.09 ms | 5.85 ms | 1.24 ms | 13.9x / 65.6x |
| fftconvolve | 2^20 x 4097 | 11.20 ms | 1.65 ms | 1.37 ms | 6.8x / 8.2x |
| fftconvolve | 2^22 x 257 | 44.48 ms | 2.64 ms | 1.04 ms | 16.8x / 42.8x |
| oaconvolve | 2^23 x 513 | 27.42 ms | 3.56 ms | 2.57 ms | 7.7x / 10.7x |
| correlate (batched) | 64ch x 2^18, 4096 taps | 65.78 ms | 8.24 ms | 6.05 ms | 8.0x / 10.9x |
| resample_poly | 16ch, 48k->44.1k (147/160) | 120.72 ms | 6.13 ms | 3.90 ms | 19.7x / 31.0x |
| upfirdn | 64ch x 2^18, up=2 down=3, 255 taps | 302.61 ms | 7.49 ms | 5.48 ms | 40.4x / 55.2x |
| upfirdn (complex IQ) | 16ch x 2^20 c64, down=10, 201 taps | 182.68 ms | 3.83 ms | 1.26 ms | 47.6x / 145.4x |
| resample (FFT) | 2^20 -> 2^18 | 4.22 ms | 0.37 ms | 0.31 ms | 11.4x / 13.6x |
| hilbert | 2^20 | 9.45 ms | 1.55 ms | 1.33 ms | 6.1x / 7.1x |
| resample (FFT, >1M) | 2^23 -> x0.75 | 67.31 ms | 6.13 ms | 5.33 ms | 11.0x / 12.6x |
| hilbert (>1M) | 2^23 | 101.96 ms | 8.15 ms | 6.10 ms | 12.5x / 16.7x |
| lfilter (FIR) | 64ch x 2^20, 257 taps | 1649.50 ms | 17.57 ms | 9.48 ms | 93.9x / 173.9x |
| filtfilt (FIR) | 64ch x 2^20, 257 taps | 3233.16 ms | 43.89 ms | 24.00 ms | 73.7x / 134.7x |
| istft | 16ch x 2^20, nperseg=1024 | 180.80 ms | 16.67 ms | 1.96 ms | 10.8x / 92.4x |
| csd | 64ch x 2^20, nperseg=1024 | 937.58 ms | 16.11 ms | 8.07 ms | 58.2x / 116.1x |
| coherence | 64ch x 2^20, nperseg=1024 | 1941.17 ms | 17.24 ms | 9.70 ms | 112.6x / 200.1x |
| sosfilt (IIR) | 256ch x 2^20, butter-8 | 1315.36 ms | 42.92 ms | 12.00 ms | 30.6x / 109.6x |
| sosfilt (IIR, 1ch scan) | 1ch x 2^22, butter-8 | 25.73 ms | 1.98 ms | 1.20 ms | 13.0x / 21.5x |
| sosfiltfilt (IIR) | 256ch x 2^20, butter-8 | 2708.33 ms | 126.94 ms | 41.13 ms | 21.3x / 65.8x |
| find_peaks | 2^23, prominence=1 | 223.79 ms | 222.99 ms | — | 1.0x / — |
