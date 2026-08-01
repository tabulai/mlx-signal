Benchmarks on Apple M4 Max (macOS 26.2), mlx 0.32.0, scipy 1.18.0, float32 inputs. e2e = NumPy in / NumPy out; device = MLX arrays in and out (steady state).

| function | shape | scipy | mlx-signal (e2e) | mlx-signal (device) | speedup (e2e / device) |
|---|---|---:|---:|---:|---:|
| welch | 64ch x 2^20, nperseg=1024 | 502.73 ms | 8.02 ms | 4.26 ms | 62.7x / 118.0x |
| welch | 1ch x 2^22, nperseg=4096 | 49.24 ms | 1.76 ms | 1.54 ms | 28.1x / 32.1x |
| welch | 256ch x 2^16, nperseg=256 | 115.26 ms | 5.44 ms | 1.08 ms | 21.2x / 106.5x |
| spectrogram | 16ch x 2^20 | 68.19 ms | 3.71 ms | 1.30 ms | 18.4x / 52.3x |
| stft | 16ch x 2^20, nperseg=1024 | 82.13 ms | 4.74 ms | 1.24 ms | 17.3x / 66.1x |
| fftconvolve | 2^20 x 4097 | 11.26 ms | 1.57 ms | 1.33 ms | 7.2x / 8.4x |
| fftconvolve | 2^22 x 257 | 49.01 ms | 2.91 ms | 0.73 ms | 16.9x / 67.3x |
| fftconvolve (pair) | 2^20 x 2^20 | 22.54 ms | 2.62 ms | 0.92 ms | 8.6x / 24.4x |
| correlate (auto) | 2^20 autocorrelation | 22.79 ms | 2.00 ms | 1.55 ms | 11.4x / 14.7x |
| oaconvolve | 2^23 x 513 | 27.03 ms | 2.31 ms | 1.27 ms | 11.7x / 21.2x |
| correlate (batched) | 64ch x 2^18, 4096 taps | 66.24 ms | 6.83 ms | 4.84 ms | 9.7x / 13.7x |
| resample_poly | 16ch, 48k->44.1k (147/160) | 121.85 ms | 6.20 ms | 3.92 ms | 19.6x / 31.1x |
| upfirdn | 64ch x 2^18, up=2 down=3, 255 taps | 308.15 ms | 7.14 ms | 5.53 ms | 43.2x / 55.7x |
| upfirdn (complex IQ) | 16ch x 2^20 c64, down=10, 201 taps | 185.45 ms | 4.19 ms | 1.29 ms | 44.2x / 144.1x |
| resample (FFT) | 2^20 -> 2^18 | 4.28 ms | 0.98 ms | 0.78 ms | 4.4x / 5.5x |
| hilbert | 2^20 | 10.02 ms | 1.63 ms | 1.35 ms | 6.1x / 7.4x |
| resample (FFT, >1M) | 2^23 -> x0.75 | 68.33 ms | 6.93 ms | 5.61 ms | 9.9x / 12.2x |
| hilbert (>1M) | 2^23 | 105.19 ms | 8.31 ms | 6.37 ms | 12.7x / 16.5x |
| lfilter (FIR) | 64ch x 2^20, 257 taps | 1644.54 ms | 13.25 ms | 4.96 ms | 124.1x / 331.4x |
| filtfilt (FIR) | 64ch x 2^20, 257 taps | 3287.88 ms | 35.10 ms | 13.86 ms | 93.7x / 237.2x |
| istft | 16ch x 2^20, nperseg=1024 | 182.77 ms | 18.58 ms | 3.43 ms | 9.8x / 53.3x |
| csd | 64ch x 2^20, nperseg=1024 | 961.18 ms | 16.42 ms | 8.17 ms | 58.5x / 117.7x |
| coherence | 64ch x 2^20, nperseg=1024 | 1981.51 ms | 17.85 ms | 9.78 ms | 111.0x / 202.7x |
| sosfilt (IIR) | 256ch x 2^20, butter-8 | 1325.68 ms | 43.20 ms | 12.25 ms | 30.7x / 108.2x |
| sosfilt (IIR, 1ch scan) | 1ch x 2^22, butter-8 | 20.76 ms | 1.85 ms | 1.34 ms | 11.2x / 15.4x |
| sosfiltfilt (IIR) | 256ch x 2^20, butter-8 | 2719.45 ms | 127.89 ms | 41.21 ms | 21.3x / 66.0x |
| find_peaks | 2^23, prominence=1 | 228.81 ms | 226.82 ms | — | 1.0x / — |
