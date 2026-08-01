Benchmarks on Apple M4 Max (macOS 26.2), mlx 0.32.0, scipy 1.18.0, float32 inputs. e2e = NumPy in / NumPy out; device = MLX arrays in and out (steady state).

| function | shape | scipy | mlx-signal (e2e) | mlx-signal (device) | speedup (e2e / device) |
|---|---|---:|---:|---:|---:|
| welch | 64ch x 2^20, nperseg=1024 | 476.70 ms | 10.37 ms | 4.35 ms | 46.0x / 109.6x |
| welch | 1ch x 2^22, nperseg=4096 | 46.94 ms | 1.19 ms | 0.89 ms | 39.3x / 52.9x |
| welch | 256ch x 2^16, nperseg=256 | 111.07 ms | 2.09 ms | 1.07 ms | 53.0x / 103.4x |
| spectrogram | 16ch x 2^20 | 66.04 ms | 3.54 ms | 1.31 ms | 18.6x / 50.5x |
| stft | 16ch x 2^20, nperseg=1024 | 78.07 ms | 4.44 ms | 1.24 ms | 17.6x / 62.9x |
| fftconvolve | 2^20 x 4097 | 11.17 ms | 1.54 ms | 1.39 ms | 7.3x / 8.1x |
| fftconvolve | 2^22 x 257 | 45.06 ms | 2.86 ms | 0.72 ms | 15.8x / 62.6x |
| fftconvolve (pair) | 2^20 x 2^20 | 21.57 ms | 2.47 ms | 0.92 ms | 8.7x / 23.5x |
| correlate (auto) | 2^20 autocorrelation | 21.14 ms | 1.92 ms | 0.60 ms | 11.0x / 35.5x |
| oaconvolve | 2^23 x 513 | 26.56 ms | 2.23 ms | 1.25 ms | 11.9x / 21.3x |
| correlate (batched) | 64ch x 2^18, 4096 taps | 63.68 ms | 7.31 ms | 4.81 ms | 8.7x / 13.2x |
| resample_poly | 16ch, 48k->44.1k (147/160) | 118.60 ms | 6.03 ms | 3.85 ms | 19.7x / 30.8x |
| upfirdn | 64ch x 2^18, up=2 down=3, 255 taps | 299.82 ms | 7.53 ms | 5.51 ms | 39.8x / 54.4x |
| upfirdn (complex IQ) | 16ch x 2^20 c64, down=10, 201 taps | 180.81 ms | 3.79 ms | 1.21 ms | 47.7x / 149.0x |
| resample (FFT) | 2^20 -> 2^18 | 4.11 ms | 1.24 ms | 0.75 ms | 3.3x / 5.5x |
| hilbert | 2^20 | 9.72 ms | 1.71 ms | 1.31 ms | 5.7x / 7.4x |
| resample (FFT, >1M) | 2^23 -> x0.75 | 66.10 ms | 6.39 ms | 5.57 ms | 10.3x / 11.9x |
| hilbert (>1M) | 2^23 | 101.68 ms | 7.72 ms | 6.35 ms | 13.2x / 16.0x |
| lfilter (FIR) | 64ch x 2^20, 257 taps | 1621.29 ms | 12.74 ms | 4.90 ms | 127.3x / 330.8x |
| filtfilt (FIR) | 64ch x 2^20, 257 taps | 3259.00 ms | 34.93 ms | 13.86 ms | 93.3x / 235.1x |
| istft | 16ch x 2^20, nperseg=1024 | 179.36 ms | 20.18 ms | 1.88 ms | 8.9x / 95.3x |
| csd | 64ch x 2^20, nperseg=1024 | 948.81 ms | 16.19 ms | 8.27 ms | 58.6x / 114.7x |
| coherence | 64ch x 2^20, nperseg=1024 | 1958.03 ms | 17.77 ms | 9.94 ms | 110.2x / 197.1x |
| sosfilt (IIR) | 256ch x 2^20, butter-8 | 1311.32 ms | 42.23 ms | 11.92 ms | 31.1x / 110.0x |
| sosfilt (IIR, 1ch scan) | 1ch x 2^22, butter-8 | 20.57 ms | 1.71 ms | 1.25 ms | 12.0x / 16.5x |
| sosfiltfilt (IIR) | 256ch x 2^20, butter-8 | 2691.73 ms | 124.06 ms | 40.79 ms | 21.7x / 66.0x |
| find_peaks | 2^23, prominence=1 | 220.60 ms | 219.55 ms | — | 1.0x / — |
