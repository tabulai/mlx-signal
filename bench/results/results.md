Benchmarks on Apple M4 Max (macOS 26.2), mlx 0.32.0, scipy 1.18.0, float32 inputs. e2e = NumPy in / NumPy out; device = MLX arrays in and out (steady state).

| function | shape | scipy | mlx-signal (e2e) | mlx-signal (device) | speedup (e2e / device) |
|---|---|---:|---:|---:|---:|
| welch | 64ch x 2^20, nperseg=1024 | 478.14 ms | 10.78 ms | 5.11 ms | 44.4x / 93.6x |
| welch | 1ch x 2^22, nperseg=4096 | 47.88 ms | 1.83 ms | 0.61 ms | 26.2x / 78.4x |
| welch | 256ch x 2^16, nperseg=256 | 110.97 ms | 5.01 ms | 1.04 ms | 22.2x / 106.7x |
| spectrogram | 16ch x 2^20 | 68.27 ms | 3.58 ms | 1.32 ms | 19.0x / 51.8x |
| stft | 16ch x 2^20, nperseg=1024 | 82.91 ms | 4.57 ms | 1.49 ms | 18.1x / 55.6x |
| fftconvolve | 2^20 x 4097 | 10.96 ms | 1.74 ms | 0.53 ms | 6.3x / 20.8x |
| fftconvolve | 2^22 x 257 | 46.44 ms | 3.04 ms | 0.72 ms | 15.3x / 64.1x |
| oaconvolve | 2^23 x 513 | 27.04 ms | 2.73 ms | 1.32 ms | 9.9x / 20.5x |
| correlate (batched) | 64ch x 2^18, 4096 taps | 65.31 ms | 7.08 ms | 4.63 ms | 9.2x / 14.1x |
| resample_poly | 16ch, 48k->44.1k (147/160) | 120.93 ms | 6.07 ms | 3.83 ms | 19.9x / 31.5x |
| upfirdn | 64ch x 2^18, up=2 down=3, 255 taps | 297.36 ms | 7.75 ms | 5.79 ms | 38.4x / 51.4x |
| upfirdn (complex IQ) | 16ch x 2^20 c64, down=10, 201 taps | 180.50 ms | 3.91 ms | 1.26 ms | 46.2x / 143.6x |
| resample (FFT) | 2^20 -> 2^18 | 4.18 ms | 0.91 ms | 0.78 ms | 4.6x / 5.4x |
| hilbert | 2^20 | 9.25 ms | 1.54 ms | 0.54 ms | 6.0x / 17.0x |
| resample (FFT, >1M) | 2^23 -> x0.75 | 65.23 ms | 6.10 ms | 5.30 ms | 10.7x / 12.3x |
| hilbert (>1M) | 2^23 | 100.21 ms | 8.03 ms | 6.09 ms | 12.5x / 16.5x |
| lfilter (FIR) | 64ch x 2^20, 257 taps | 1590.45 ms | 12.85 ms | 4.88 ms | 123.8x / 325.9x |
| filtfilt (FIR) | 64ch x 2^20, 257 taps | 3259.94 ms | 34.66 ms | 13.52 ms | 94.0x / 241.1x |
| istft | 16ch x 2^20, nperseg=1024 | 181.43 ms | 18.08 ms | 2.23 ms | 10.0x / 81.3x |
| csd | 64ch x 2^20, nperseg=1024 | 953.99 ms | 16.36 ms | 8.31 ms | 58.3x / 114.8x |
| coherence | 64ch x 2^20, nperseg=1024 | 1925.61 ms | 17.66 ms | 9.54 ms | 109.0x / 201.8x |
| sosfilt (IIR) | 256ch x 2^20, butter-8 | 1292.84 ms | 42.19 ms | 11.73 ms | 30.6x / 110.2x |
| sosfilt (IIR, 1ch scan) | 1ch x 2^22, butter-8 | 21.61 ms | 1.66 ms | 1.22 ms | 13.1x / 17.7x |
| sosfiltfilt (IIR) | 256ch x 2^20, butter-8 | 2662.26 ms | 129.11 ms | 39.68 ms | 20.6x / 67.1x |
| find_peaks | 2^23, prominence=1 | 216.14 ms | 217.25 ms | — | 1.0x / — |
