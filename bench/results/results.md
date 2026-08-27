Benchmarks on Apple M4 Max (macOS 26.2), mlx 0.32.0, scipy 1.18.0, float32 inputs. e2e = NumPy in / NumPy out; device = MLX arrays in and out (steady state).

| function | shape | scipy | mlx-signal (e2e) | mlx-signal (device) | speedup (e2e / device) |
|---|---|---:|---:|---:|---:|
| welch | 64ch x 2^20, nperseg=1024 | 494.84 ms | 8.11 ms | 4.22 ms | 61.0x / 117.2x |
| welch | 1ch x 2^22, nperseg=4096 | 48.85 ms | 1.77 ms | 1.42 ms | 27.5x / 34.4x |
| welch | 256ch x 2^16, nperseg=256 | 113.96 ms | 4.01 ms | 1.10 ms | 28.4x / 103.4x |
| spectrogram | 16ch x 2^20 | 68.16 ms | 3.73 ms | 1.30 ms | 18.3x / 52.4x |
| stft | 16ch x 2^20, nperseg=1024 | 82.94 ms | 7.00 ms | 1.27 ms | 11.8x / 65.3x |
| fftconvolve | 2^20 x 4097 | 11.22 ms | 1.64 ms | 1.35 ms | 6.8x / 8.3x |
| fftconvolve | 2^22 x 257 | 48.57 ms | 2.77 ms | 0.75 ms | 17.5x / 65.0x |
| fftconvolve (pair) | 2^20 x 2^20 | 21.89 ms | 2.63 ms | 1.09 ms | 8.3x / 20.1x |
| correlate (auto) | 2^20 autocorrelation | 22.12 ms | 2.27 ms | 1.55 ms | 9.8x / 14.3x |
| oaconvolve | 2^23 x 513 | 27.20 ms | 2.18 ms | 1.25 ms | 12.5x / 21.8x |
| correlate (batched) | 64ch x 2^18, 4096 taps | 64.81 ms | 7.24 ms | 4.64 ms | 9.0x / 14.0x |
| resample_poly | 16ch, 48k->44.1k (147/160) | 121.76 ms | 3.48 ms | 1.08 ms | 35.0x / 113.1x |
| upfirdn | 64ch x 2^18, up=2 down=3, 255 taps | 304.36 ms | 4.20 ms | 2.30 ms | 72.5x / 132.5x |
| upfirdn (complex IQ) | 16ch x 2^20 c64, down=10, 201 taps | 184.59 ms | 3.91 ms | 1.30 ms | 47.2x / 142.5x |
| resample (FFT) | 2^20 -> 2^18 | 4.32 ms | 0.87 ms | 0.67 ms | 5.0x / 6.5x |
| hilbert | 2^20 | 9.43 ms | 1.88 ms | 1.29 ms | 5.0x / 7.3x |
| resample (FFT, >1M) | 2^23 -> x0.75 | 67.75 ms | 4.05 ms | 3.20 ms | 16.7x / 21.2x |
| hilbert (>1M) | 2^23 | 102.41 ms | 8.12 ms | 6.33 ms | 12.6x / 16.2x |
| lfilter (FIR) | 64ch x 2^20, 257 taps | 1617.26 ms | 13.97 ms | 4.89 ms | 115.8x / 330.5x |
| filtfilt (FIR) | 64ch x 2^20, 257 taps | 3263.42 ms | 35.07 ms | 13.78 ms | 93.0x / 236.7x |
| istft | 16ch x 2^20, nperseg=1024 | 176.88 ms | 21.08 ms | 1.84 ms | 8.4x / 96.2x |
| csd | 64ch x 2^20, nperseg=1024 | 935.50 ms | 16.06 ms | 8.13 ms | 58.2x / 115.1x |
| coherence | 64ch x 2^20, nperseg=1024 | 1948.92 ms | 17.63 ms | 9.63 ms | 110.6x / 202.3x |
| sosfilt (IIR) | 256ch x 2^20, butter-8 | 1310.97 ms | 42.71 ms | 12.12 ms | 30.7x / 108.2x |
| sosfilt (IIR, 1ch scan) | 1ch x 2^22, butter-8 | 20.59 ms | 1.83 ms | 1.34 ms | 11.2x / 15.4x |
| sosfiltfilt (IIR) | 256ch x 2^20, butter-8 | 2689.10 ms | 124.59 ms | 40.46 ms | 21.6x / 66.5x |
| lfilter (IIR) | 256ch x 2^20, butter-4 tf | 1536.78 ms | 42.68 ms | 11.98 ms | 36.0x / 128.3x |
| lfilter (IIR, 1ch scan) | 1ch x 2^22, butter-4 tf | 22.98 ms | 1.78 ms | 0.95 ms | 12.9x / 24.1x |
| filtfilt (IIR) | 256ch x 2^20, butter-4 tf | 3072.40 ms | 125.49 ms | 41.22 ms | 24.5x / 74.5x |
| find_peaks | 2^23, prominence=1 | 219.25 ms | 85.80 ms | — | 2.6x / — |
| peak_prominences | 2^23, 2.8M peaks | 158.96 ms | 19.65 ms | — | 8.1x / — |
