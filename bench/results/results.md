Benchmarks on Apple M4 Max (macOS 26.2), mlx 0.32.0, scipy 1.18.0, float32 inputs. e2e = NumPy in / NumPy out; device = MLX arrays in and out (steady state).

| function | shape | scipy | mlx-signal (e2e) | mlx-signal (device) | speedup (e2e / device) |
|---|---|---:|---:|---:|---:|
| welch | 64ch x 2^20, nperseg=1024 | 491.59 ms | 8.45 ms | 4.29 ms | 58.2x / 114.5x |
| welch | 1ch x 2^22, nperseg=4096 | 48.91 ms | 1.79 ms | 1.23 ms | 27.4x / 39.7x |
| welch | 256ch x 2^16, nperseg=256 | 113.80 ms | 2.09 ms | 1.10 ms | 54.5x / 103.1x |
| spectrogram | 16ch x 2^20 | 67.25 ms | 3.56 ms | 1.32 ms | 18.9x / 51.0x |
| stft | 16ch x 2^20, nperseg=1024 | 80.38 ms | 4.76 ms | 1.23 ms | 16.9x / 65.3x |
| fftconvolve | 2^20 x 4097 | 11.17 ms | 1.65 ms | 0.92 ms | 6.8x / 12.2x |
| fftconvolve | 2^22 x 257 | 48.09 ms | 2.74 ms | 0.73 ms | 17.6x / 65.8x |
| fftconvolve (pair) | 2^20 x 2^20 | 21.58 ms | 2.52 ms | 0.96 ms | 8.6x / 22.5x |
| correlate (auto) | 2^20 autocorrelation | 22.06 ms | 2.04 ms | 1.56 ms | 10.8x / 14.1x |
| oaconvolve | 2^23 x 513 | 27.10 ms | 2.69 ms | 1.24 ms | 10.1x / 21.9x |
| correlate (batched) | 64ch x 2^18, 4096 taps | 65.07 ms | 7.30 ms | 4.85 ms | 8.9x / 13.4x |
| resample_poly | 16ch, 48k->44.1k (147/160) | 122.08 ms | 4.33 ms | 0.99 ms | 28.2x / 122.7x |
| upfirdn | 64ch x 2^18, up=2 down=3, 255 taps | 300.57 ms | 4.34 ms | 2.29 ms | 69.2x / 131.0x |
| upfirdn (complex IQ) | 16ch x 2^20 c64, down=10, 201 taps | 183.14 ms | 4.48 ms | 1.26 ms | 40.8x / 145.8x |
| resample (FFT) | 2^20 -> 2^18 | 4.27 ms | 1.00 ms | 0.80 ms | 4.3x / 5.3x |
| hilbert | 2^20 | 9.33 ms | 2.01 ms | 1.31 ms | 4.7x / 7.1x |
| resample (FFT, >1M) | 2^23 -> x0.75 | 67.19 ms | 6.53 ms | 5.63 ms | 10.3x / 11.9x |
| hilbert (>1M) | 2^23 | 103.13 ms | 8.45 ms | 6.44 ms | 12.2x / 16.0x |
| lfilter (FIR) | 64ch x 2^20, 257 taps | 1648.79 ms | 13.06 ms | 5.08 ms | 126.2x / 324.9x |
| filtfilt (FIR) | 64ch x 2^20, 257 taps | 3253.45 ms | 34.83 ms | 13.73 ms | 93.4x / 237.0x |
| istft | 16ch x 2^20, nperseg=1024 | 178.74 ms | 19.48 ms | 1.96 ms | 9.2x / 91.2x |
| csd | 64ch x 2^20, nperseg=1024 | 937.92 ms | 15.98 ms | 8.16 ms | 58.7x / 115.0x |
| coherence | 64ch x 2^20, nperseg=1024 | 1939.59 ms | 17.44 ms | 9.74 ms | 111.2x / 199.0x |
| sosfilt (IIR) | 256ch x 2^20, butter-8 | 1307.77 ms | 42.82 ms | 12.08 ms | 30.5x / 108.2x |
| sosfilt (IIR, 1ch scan) | 1ch x 2^22, butter-8 | 20.57 ms | 1.81 ms | 1.35 ms | 11.3x / 15.3x |
| sosfiltfilt (IIR) | 256ch x 2^20, butter-8 | 2688.07 ms | 126.66 ms | 40.95 ms | 21.2x / 65.6x |
| find_peaks | 2^23, prominence=1 | 221.94 ms | 221.29 ms | — | 1.0x / — |
