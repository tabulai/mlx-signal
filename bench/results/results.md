Benchmarks on Apple M4 Max (macOS 26.2), mlx 0.32.0, scipy 1.18.0, float32 inputs. e2e = NumPy in / NumPy out; device = MLX arrays in and out (steady state).

| function | shape | scipy | mlx-signal (e2e) | mlx-signal (device) | speedup (e2e / device) |
|---|---|---:|---:|---:|---:|
| welch | 64ch x 2^20, nperseg=1024 | 498.47 ms | 8.14 ms | 4.31 ms | 61.3x / 115.7x |
| welch | 1ch x 2^22, nperseg=4096 | 49.41 ms | 2.24 ms | 1.35 ms | 22.0x / 36.6x |
| welch | 256ch x 2^16, nperseg=256 | 114.68 ms | 2.76 ms | 1.12 ms | 41.6x / 102.8x |
| spectrogram | 16ch x 2^20 | 69.36 ms | 7.01 ms | 1.29 ms | 9.9x / 53.7x |
| stft | 16ch x 2^20, nperseg=1024 | 81.33 ms | 4.77 ms | 1.26 ms | 17.1x / 64.7x |
| fftconvolve | 2^20 x 4097 | 11.29 ms | 1.60 ms | 1.34 ms | 7.0x / 8.4x |
| fftconvolve | 2^22 x 257 | 48.68 ms | 3.05 ms | 2.19 ms | 16.0x / 22.2x |
| fftconvolve (pair) | 2^20 x 2^20 | 22.07 ms | 2.52 ms | 0.93 ms | 8.8x / 23.7x |
| correlate (auto) | 2^20 autocorrelation | 22.59 ms | 1.95 ms | 1.58 ms | 11.6x / 14.3x |
| oaconvolve | 2^23 x 513 | 27.03 ms | 2.31 ms | 1.25 ms | 11.7x / 21.6x |
| correlate (batched) | 64ch x 2^18, 4096 taps | 66.00 ms | 7.23 ms | 4.82 ms | 9.1x / 13.7x |
| resample_poly | 16ch, 48k->44.1k (147/160) | 121.98 ms | 6.27 ms | 3.86 ms | 19.4x / 31.6x |
| upfirdn | 64ch x 2^18, up=2 down=3, 255 taps | 306.78 ms | 7.58 ms | 5.56 ms | 40.5x / 55.1x |
| upfirdn (complex IQ) | 16ch x 2^20 c64, down=10, 201 taps | 184.17 ms | 3.81 ms | 1.24 ms | 48.4x / 148.1x |
| resample (FFT) | 2^20 -> 2^18 | 4.26 ms | 0.86 ms | 0.47 ms | 5.0x / 9.1x |
| hilbert | 2^20 | 10.22 ms | 1.49 ms | 1.33 ms | 6.8x / 7.7x |
| resample (FFT, >1M) | 2^23 -> x0.75 | 69.25 ms | 6.61 ms | 5.68 ms | 10.5x / 12.2x |
| hilbert (>1M) | 2^23 | 104.93 ms | 8.24 ms | 6.52 ms | 12.7x / 16.1x |
| lfilter (FIR) | 64ch x 2^20, 257 taps | 1641.91 ms | 13.26 ms | 5.00 ms | 123.8x / 328.5x |
| filtfilt (FIR) | 64ch x 2^20, 257 taps | 3281.22 ms | 35.27 ms | 13.91 ms | 93.0x / 235.8x |
| istft | 16ch x 2^20, nperseg=1024 | 179.90 ms | 19.96 ms | 2.38 ms | 9.0x / 75.5x |
| csd | 64ch x 2^20, nperseg=1024 | 968.02 ms | 16.19 ms | 8.27 ms | 59.8x / 117.1x |
| coherence | 64ch x 2^20, nperseg=1024 | 1989.10 ms | 18.01 ms | 9.91 ms | 110.4x / 200.8x |
| sosfilt (IIR) | 256ch x 2^20, butter-8 | 1322.22 ms | 43.27 ms | 11.94 ms | 30.6x / 110.8x |
| sosfilt (IIR, 1ch scan) | 1ch x 2^22, butter-8 | 21.00 ms | 1.71 ms | 1.26 ms | 12.3x / 16.6x |
| sosfiltfilt (IIR) | 256ch x 2^20, butter-8 | 2715.50 ms | 127.60 ms | 40.80 ms | 21.3x / 66.6x |
| find_peaks | 2^23, prominence=1 | 224.21 ms | 227.34 ms | — | 1.0x / — |
