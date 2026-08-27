Benchmarks on Apple M4 Max (macOS 26.2), mlx 0.32.2, scipy 1.18.1, float32 inputs. e2e = NumPy in / NumPy out; device = MLX arrays in and out (steady state).

| function | shape | scipy | mlx-signal (e2e) | mlx-signal (device) | speedup (e2e / device) |
|---|---|---:|---:|---:|---:|
| welch | 64ch x 2^20, nperseg=1024 | 498.67 ms | 8.46 ms | 4.30 ms | 59.0x / 116.0x |
| welch | 1ch x 2^22, nperseg=4096 | 49.03 ms | 1.89 ms | 1.42 ms | 25.9x / 34.6x |
| welch | 256ch x 2^16, nperseg=256 | 114.46 ms | 2.59 ms | 1.09 ms | 44.2x / 105.2x |
| spectrogram | 16ch x 2^20 | 68.43 ms | 3.63 ms | 1.32 ms | 18.9x / 51.9x |
| stft | 16ch x 2^20, nperseg=1024 | 82.60 ms | 4.49 ms | 1.21 ms | 18.4x / 68.2x |
| fftconvolve | 2^20 x 4097 | 11.46 ms | 1.59 ms | 0.51 ms | 7.2x / 22.6x |
| fftconvolve | 2^22 x 257 | 48.08 ms | 2.64 ms | 2.27 ms | 18.2x / 21.2x |
| fftconvolve (pair) | 2^20 x 2^20 | 22.25 ms | 2.50 ms | 0.85 ms | 8.9x / 26.1x |
| correlate (auto) | 2^20 autocorrelation | 22.44 ms | 2.26 ms | 0.59 ms | 9.9x / 37.9x |
| oaconvolve | 2^23 x 513 | 27.12 ms | 5.34 ms | 1.25 ms | 5.1x / 21.7x |
| correlate (batched) | 64ch x 2^18, 4096 taps | 66.16 ms | 7.26 ms | 4.99 ms | 9.1x / 13.3x |
| resample_poly | 16ch, 48k->44.1k (147/160) | 123.40 ms | 5.57 ms | 0.99 ms | 22.2x / 124.1x |
| upfirdn | 64ch x 2^18, up=2 down=3, 255 taps | 303.33 ms | 4.22 ms | 2.29 ms | 71.9x / 132.2x |
| upfirdn (complex IQ) | 16ch x 2^20 c64, down=10, 201 taps | 184.35 ms | 3.95 ms | 1.92 ms | 46.6x / 96.0x |
| resample (FFT) | 2^20 -> 2^18 | 4.35 ms | 0.57 ms | 0.28 ms | 7.6x / 15.7x |
| hilbert | 2^20 | 9.42 ms | 0.90 ms | 0.43 ms | 10.4x / 21.9x |
| resample (FFT, >1M) | 2^23 -> x0.75 | 68.02 ms | 4.46 ms | 2.90 ms | 15.2x / 23.5x |
| hilbert (>1M) | 2^23 | 104.39 ms | 4.71 ms | 2.67 ms | 22.2x / 39.1x |
| lfilter (FIR) | 64ch x 2^20, 257 taps | 1618.20 ms | 13.18 ms | 4.93 ms | 122.8x / 328.3x |
| filtfilt (FIR) | 64ch x 2^20, 257 taps | 3268.56 ms | 34.64 ms | 13.51 ms | 94.4x / 241.9x |
| istft | 16ch x 2^20, nperseg=1024 | 176.91 ms | 22.39 ms | 2.79 ms | 7.9x / 63.4x |
| csd | 64ch x 2^20, nperseg=1024 | 946.63 ms | 16.25 ms | 8.06 ms | 58.3x / 117.4x |
| coherence | 64ch x 2^20, nperseg=1024 | 1964.71 ms | 17.74 ms | 9.60 ms | 110.8x / 204.6x |
| sosfilt (IIR) | 256ch x 2^20, butter-8 | 1322.49 ms | 42.88 ms | 11.81 ms | 30.8x / 112.0x |
| sosfilt (IIR, 1ch scan) | 1ch x 2^22, butter-8 | 20.50 ms | 2.36 ms | 1.75 ms | 8.7x / 11.7x |
| sosfiltfilt (IIR) | 256ch x 2^20, butter-8 | 2681.96 ms | 131.98 ms | 40.26 ms | 20.3x / 66.6x |
| lfilter (IIR) | 256ch x 2^20, butter-4 tf | 1549.76 ms | 42.51 ms | 11.78 ms | 36.5x / 131.6x |
| lfilter (IIR, 1ch scan) | 1ch x 2^22, butter-4 tf | 23.16 ms | 3.89 ms | 0.93 ms | 6.0x / 24.9x |
| filtfilt (IIR) | 256ch x 2^20, butter-4 tf | 3099.24 ms | 126.53 ms | 41.31 ms | 24.5x / 75.0x |
| find_peaks | 2^23, prominence=1 | 221.60 ms | 83.05 ms | — | 2.7x / — |
| peak_prominences | 2^23, 2.8M peaks | 158.44 ms | 20.99 ms | — | 7.5x / — |
