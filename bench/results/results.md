Benchmarks on Apple M4 Max (macOS 26.2), mlx 0.32.0, scipy 1.18.0, float32 inputs. e2e = NumPy in / NumPy out; device = MLX arrays in and out (steady state).

| function | shape | scipy | mlx-signal (e2e) | mlx-signal (device) | speedup (e2e / device) |
|---|---|---:|---:|---:|---:|
| welch | 64ch x 2^20, nperseg=1024 | 485.33 ms | 10.26 ms | 4.24 ms | 47.3x / 114.4x |
| welch | 1ch x 2^22, nperseg=4096 | 47.53 ms | 1.49 ms | 1.25 ms | 31.9x / 38.1x |
| welch | 256ch x 2^16, nperseg=256 | 112.37 ms | 2.45 ms | 1.08 ms | 45.9x / 103.8x |
| spectrogram | 16ch x 2^20 | 67.62 ms | 3.61 ms | 1.29 ms | 18.7x / 52.5x |
| stft | 16ch x 2^20, nperseg=1024 | 79.91 ms | 7.38 ms | 1.23 ms | 10.8x / 64.9x |
| fftconvolve | 2^20 x 4097 | 10.98 ms | 1.50 ms | 1.31 ms | 7.3x / 8.4x |
| fftconvolve | 2^22 x 257 | 46.69 ms | 2.77 ms | 0.69 ms | 16.8x / 67.7x |
| fftconvolve (pair) | 2^20 x 2^20 | 21.14 ms | 2.25 ms | 1.77 ms | 9.4x / 11.9x |
| correlate (auto) | 2^20 autocorrelation | 21.46 ms | 3.48 ms | 1.34 ms | 6.2x / 16.1x |
| oaconvolve | 2^23 x 513 | 27.04 ms | 2.85 ms | 1.27 ms | 9.5x / 21.4x |
| correlate (batched) | 64ch x 2^18, 4096 taps | 65.26 ms | 6.68 ms | 4.72 ms | 9.8x / 13.8x |
| resample_poly | 16ch, 48k->44.1k (147/160) | 120.93 ms | 6.02 ms | 3.82 ms | 20.1x / 31.7x |
| upfirdn | 64ch x 2^18, up=2 down=3, 255 taps | 300.72 ms | 7.08 ms | 5.49 ms | 42.5x / 54.8x |
| upfirdn (complex IQ) | 16ch x 2^20 c64, down=10, 201 taps | 183.57 ms | 3.83 ms | 1.29 ms | 47.9x / 142.2x |
| resample (FFT) | 2^20 -> 2^18 | 4.20 ms | 0.89 ms | 0.82 ms | 4.7x / 5.2x |
| hilbert | 2^20 | 9.76 ms | 1.71 ms | 0.53 ms | 5.7x / 18.5x |
| resample (FFT, >1M) | 2^23 -> x0.75 | 67.19 ms | 6.42 ms | 5.57 ms | 10.5x / 12.1x |
| hilbert (>1M) | 2^23 | 102.98 ms | 8.24 ms | 6.35 ms | 12.5x / 16.2x |
| lfilter (FIR) | 64ch x 2^20, 257 taps | 1634.96 ms | 12.86 ms | 4.92 ms | 127.1x / 332.6x |
| filtfilt (FIR) | 64ch x 2^20, 257 taps | 3269.99 ms | 34.35 ms | 13.65 ms | 95.2x / 239.5x |
| istft | 16ch x 2^20, nperseg=1024 | 177.54 ms | 20.00 ms | 1.89 ms | 8.9x / 93.7x |
| csd | 64ch x 2^20, nperseg=1024 | 943.26 ms | 16.31 ms | 8.18 ms | 57.8x / 115.4x |
| coherence | 64ch x 2^20, nperseg=1024 | 1951.10 ms | 17.54 ms | 9.68 ms | 111.2x / 201.5x |
| sosfilt (IIR) | 256ch x 2^20, butter-8 | 1316.27 ms | 42.67 ms | 12.11 ms | 30.8x / 108.7x |
| sosfilt (IIR, 1ch scan) | 1ch x 2^22, butter-8 | 20.58 ms | 1.72 ms | 1.23 ms | 11.9x / 16.8x |
| sosfiltfilt (IIR) | 256ch x 2^20, butter-8 | 2700.31 ms | 124.19 ms | 40.82 ms | 21.7x / 66.2x |
| find_peaks | 2^23, prominence=1 | 218.89 ms | 216.44 ms | — | 1.0x / — |
