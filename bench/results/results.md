Benchmarks on Apple M4 Max (macOS 26.2), mlx 0.32.0, scipy 1.18.0, float32 inputs. e2e = NumPy in / NumPy out; device = MLX arrays in and out (steady state).

| function | shape | scipy | mlx-signal (e2e) | mlx-signal (device) | speedup (e2e / device) |
|---|---|---:|---:|---:|---:|
| welch | 64ch x 2^20, nperseg=1024 | 481.01 ms | 8.09 ms | 4.28 ms | 59.5x / 112.4x |
| welch | 1ch x 2^22, nperseg=4096 | 48.11 ms | 1.31 ms | 0.86 ms | 36.6x / 55.6x |
| welch | 256ch x 2^16, nperseg=256 | 112.00 ms | 2.54 ms | 1.08 ms | 44.1x / 103.6x |
| spectrogram | 16ch x 2^20 | 66.25 ms | 6.56 ms | 1.31 ms | 10.1x / 50.7x |
| stft | 16ch x 2^20, nperseg=1024 | 78.85 ms | 7.30 ms | 1.25 ms | 10.8x / 63.1x |
| fftconvolve | 2^20 x 4097 | 10.95 ms | 1.58 ms | 1.49 ms | 6.9x / 7.4x |
| fftconvolve | 2^22 x 257 | 48.30 ms | 2.38 ms | 1.00 ms | 20.3x / 48.2x |
| oaconvolve | 2^23 x 513 | 27.01 ms | 5.81 ms | 2.62 ms | 4.6x / 10.3x |
| correlate (batched) | 64ch x 2^18, 4096 taps | 64.99 ms | 8.08 ms | 6.10 ms | 8.0x / 10.7x |
| resample_poly | 16ch, 48k->44.1k (147/160) | 119.57 ms | 6.15 ms | 4.16 ms | 19.4x / 28.7x |
| upfirdn | 64ch x 2^18, up=2 down=3, 255 taps | 302.69 ms | 8.60 ms | 6.96 ms | 35.2x / 43.5x |
| resample (FFT) | 2^20 -> 2^18 | 4.24 ms | 0.34 ms | 0.28 ms | 12.5x / 15.3x |
| hilbert | 2^20 | 9.85 ms | 0.70 ms | 0.50 ms | 14.2x / 19.9x |
| resample (FFT, >1M) | 2^23 -> x0.75 | 67.15 ms | 6.35 ms | 5.54 ms | 10.6x / 12.1x |
| hilbert (>1M) | 2^23 | 103.05 ms | 8.50 ms | 6.40 ms | 12.1x / 16.1x |
| lfilter (FIR) | 64ch x 2^20, 257 taps | 1605.51 ms | 17.46 ms | 9.82 ms | 92.0x / 163.5x |
| filtfilt (FIR) | 64ch x 2^20, 257 taps | 3246.45 ms | 44.65 ms | 23.61 ms | 72.7x / 137.5x |
| csd | 64ch x 2^20, nperseg=1024 | 936.53 ms | 16.27 ms | 8.19 ms | 57.6x / 114.4x |
| coherence | 64ch x 2^20, nperseg=1024 | 1944.90 ms | 17.60 ms | 9.76 ms | 110.5x / 199.2x |
| sosfilt (IIR) | 256ch x 2^20, butter-8 | 1306.35 ms | 42.46 ms | 11.96 ms | 30.8x / 109.2x |
| sosfilt (IIR, 1ch scan) | 1ch x 2^22, butter-8 | 20.62 ms | 1.98 ms | 1.28 ms | 10.4x / 16.1x |
| sosfiltfilt (IIR) | 256ch x 2^20, butter-8 | 2680.55 ms | 125.21 ms | 40.98 ms | 21.4x / 65.4x |
| find_peaks | 2^23, prominence=1 | 223.38 ms | 220.89 ms | — | 1.0x / — |
