Benchmarks on Apple M4 Max (macOS 26.2), mlx 0.32.2, scipy 1.18.1, float32 inputs. e2e = NumPy in / NumPy out; device = MLX arrays in and out (steady state).
Every output passed the scipy correctness gate before timing; median of 9 runs after 3 warmups.

| function | shape | scipy | mlx-signal (e2e) | mlx-signal (device) | speedup (e2e / device) |
|---|---|---:|---:|---:|---:|
| welch | 64ch x 2^20, nperseg=1024 | 501.97 ms | 8.57 ms | 4.26 ms | 58.6x / 117.7x |
| welch | 1ch x 2^22, nperseg=4096 | 49.52 ms | 1.83 ms | 0.58 ms | 27.1x / 85.7x |
| welch | 256ch x 2^16, nperseg=256 | 117.02 ms | 2.23 ms | 1.09 ms | 52.5x / 107.7x |
| spectrogram | 16ch x 2^20 | 69.37 ms | 3.59 ms | 1.29 ms | 19.3x / 53.6x |
| stft | 16ch x 2^20, nperseg=1024 | 81.42 ms | 4.57 ms | 1.25 ms | 17.8x / 65.4x |
| fftconvolve | 2^20 x 4097 | 11.25 ms | 1.56 ms | 0.56 ms | 7.2x / 20.3x |
| fftconvolve | 2^22 x 257 | 48.82 ms | 1.21 ms | 0.70 ms | 40.3x / 69.7x |
| fftconvolve (pair) | 2^20 x 2^20 | 22.30 ms | 1.35 ms | 0.89 ms | 16.6x / 25.0x |
| correlate (auto) | 2^20 autocorrelation | 22.78 ms | 1.69 ms | 0.56 ms | 13.5x / 40.8x |
| oaconvolve | 2^23 x 513 | 27.31 ms | 2.46 ms | 1.26 ms | 11.1x / 21.7x |
| correlate (batched) | 64ch x 2^18, 4096 taps | 66.98 ms | 7.44 ms | 5.05 ms | 9.0x / 13.3x |
| resample_poly | 16ch, 48k->44.1k (147/160) | 124.10 ms | 3.55 ms | 1.07 ms | 35.0x / 116.4x |
| upfirdn | 64ch x 2^18, up=2 down=3, 255 taps | 312.98 ms | 3.90 ms | 2.48 ms | 80.3x / 126.0x |
| upfirdn (complex IQ) | 16ch x 2^20 c64, down=10, 201 taps | 186.36 ms | 3.79 ms | 1.41 ms | 49.2x / 132.4x |
| resample (FFT) | 2^20 -> 2^18 | 4.40 ms | 0.90 ms | 0.70 ms | 4.9x / 6.3x |
| hilbert | 2^20 | 8.91 ms | 1.33 ms | 0.42 ms | 6.7x / 21.4x |
| resample (FFT, >1M) | 2^23 -> x0.75 | 68.71 ms | 3.61 ms | 2.77 ms | 19.0x / 24.8x |
| hilbert (>1M) | 2^23 | 97.80 ms | 4.20 ms | 2.69 ms | 23.3x / 36.3x |
| lfilter (FIR) | 64ch x 2^20, 257 taps | 1648.00 ms | 13.95 ms | 5.89 ms | 118.1x / 279.8x |
| filtfilt (FIR) | 64ch x 2^20, 257 taps | 3313.02 ms | 36.97 ms | 15.45 ms | 89.6x / 214.5x |
| istft | 16ch x 2^20, nperseg=1024 | 189.38 ms | 18.92 ms | 1.46 ms | 10.0x / 129.3x |
| csd | 64ch x 2^20, nperseg=1024 | 993.43 ms | 16.65 ms | 9.00 ms | 59.7x / 110.4x |
| coherence | 64ch x 2^20, nperseg=1024 | 2040.15 ms | 19.19 ms | 10.67 ms | 106.3x / 191.2x |
| sosfilt (IIR) | 256ch x 2^20, butter-8 | 1350.96 ms | 44.35 ms | 12.70 ms | 30.5x / 106.4x |
| sosfilt (IIR, 1ch scan) | 1ch x 2^22, butter-8 | 21.01 ms | 1.97 ms | 1.35 ms | 10.7x / 15.6x |
| sosfiltfilt (IIR) | 256ch x 2^20, butter-8 | 2778.62 ms | 130.65 ms | 44.64 ms | 21.3x / 62.2x |
| lfilter (IIR) | 256ch x 2^20, butter-4 tf | 1668.50 ms | 44.49 ms | 13.00 ms | 37.5x / 128.3x |
| lfilter (IIR, 1ch scan) | 1ch x 2^22, butter-4 tf | 24.59 ms | 1.54 ms | 0.93 ms | 16.0x / 26.6x |
| filtfilt (IIR) | 256ch x 2^20, butter-4 tf | 3304.42 ms | 131.38 ms | 45.64 ms | 25.2x / 72.4x |
| find_peaks | 2^23, prominence=1 | 227.18 ms | 92.43 ms | — | 2.5x / — |
| peak_prominences | 2^23, 2.8M peaks | 170.93 ms | 20.92 ms | — | 8.2x / — |
