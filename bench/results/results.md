Benchmarks on Apple M4 Max (macOS 26.2), mlx 0.32.0, scipy 1.18.0, float32 inputs. e2e = NumPy in / NumPy out; device = MLX arrays in and out (steady state).

| function | shape | scipy | mlx-signal (e2e) | mlx-signal (device) | speedup (e2e / device) |
|---|---|---:|---:|---:|---:|
| welch | 64ch x 2^20, nperseg=1024 | 481.23 ms | 10.56 ms | 4.80 ms | 45.6x / 100.3x |
| welch | 1ch x 2^22, nperseg=4096 | 47.92 ms | 1.60 ms | 1.36 ms | 30.0x / 35.3x |
| welch | 256ch x 2^16, nperseg=256 | 115.69 ms | 2.41 ms | 1.49 ms | 48.0x / 77.8x |
| spectrogram | 16ch x 2^20 | 67.93 ms | 3.72 ms | 1.23 ms | 18.3x / 55.4x |
| stft | 16ch x 2^20, nperseg=1024 | 81.28 ms | 4.84 ms | 1.36 ms | 16.8x / 59.8x |
| fftconvolve | 2^20 x 4097 | 11.46 ms | 1.50 ms | 1.27 ms | 7.7x / 9.0x |
| fftconvolve | 2^22 x 257 | 48.86 ms | 2.63 ms | 0.86 ms | 18.6x / 56.6x |
| oaconvolve | 2^23 x 513 | 27.19 ms | 4.34 ms | 2.34 ms | 6.3x / 11.6x |
| correlate (batched) | 64ch x 2^18, 4096 taps | 64.49 ms | 6.80 ms | 4.73 ms | 9.5x / 13.6x |
| resample_poly | 16ch, 48k->44.1k (147/160) | 120.58 ms | 6.07 ms | 3.81 ms | 19.9x / 31.6x |
| upfirdn | 64ch x 2^18, up=2 down=3, 255 taps | 306.99 ms | 7.62 ms | 5.61 ms | 40.3x / 54.7x |
| upfirdn (complex IQ) | 16ch x 2^20 c64, down=10, 201 taps | 183.30 ms | 3.91 ms | 1.24 ms | 46.9x / 148.4x |
| resample (FFT) | 2^20 -> 2^18 | 4.37 ms | 0.77 ms | 0.68 ms | 5.7x / 6.4x |
| hilbert | 2^20 | 10.20 ms | 1.52 ms | 1.30 ms | 6.7x / 7.9x |
| resample (FFT, >1M) | 2^23 -> x0.75 | 67.37 ms | 6.59 ms | 5.84 ms | 10.2x / 11.5x |
| hilbert (>1M) | 2^23 | 103.84 ms | 8.35 ms | 6.73 ms | 12.4x / 15.4x |
| lfilter (FIR) | 64ch x 2^20, 257 taps | 1641.62 ms | 16.60 ms | 8.25 ms | 98.9x / 199.0x |
| filtfilt (FIR) | 64ch x 2^20, 257 taps | 3265.19 ms | 42.15 ms | 20.69 ms | 77.5x / 157.8x |
| istft | 16ch x 2^20, nperseg=1024 | 175.65 ms | 19.25 ms | 1.85 ms | 9.1x / 95.1x |
| csd | 64ch x 2^20, nperseg=1024 | 934.02 ms | 16.11 ms | 8.01 ms | 58.0x / 116.7x |
| coherence | 64ch x 2^20, nperseg=1024 | 1935.74 ms | 17.73 ms | 9.56 ms | 109.1x / 202.5x |
| sosfilt (IIR) | 256ch x 2^20, butter-8 | 1312.13 ms | 42.89 ms | 12.26 ms | 30.6x / 107.1x |
| sosfilt (IIR, 1ch scan) | 1ch x 2^22, butter-8 | 20.73 ms | 2.08 ms | 1.26 ms | 10.0x / 16.5x |
| sosfiltfilt (IIR) | 256ch x 2^20, butter-8 | 2701.17 ms | 125.44 ms | 41.08 ms | 21.5x / 65.7x |
| find_peaks | 2^23, prominence=1 | 219.41 ms | 222.73 ms | — | 1.0x / — |
