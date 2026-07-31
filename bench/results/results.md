Benchmarks on Apple M4 Max (macOS 26.2), mlx 0.32.0, scipy 1.18.0, float32 inputs. e2e = NumPy in / NumPy out; device = MLX arrays in and out (steady state).

| function | shape | scipy | mlx-signal (e2e) | mlx-signal (device) | speedup (e2e / device) |
|---|---|---:|---:|---:|---:|
| welch | 64ch x 2^20, nperseg=1024 | 489.37 ms | 8.38 ms | 4.40 ms | 58.4x / 111.3x |
| welch | 1ch x 2^22, nperseg=4096 | 47.52 ms | 1.70 ms | 1.48 ms | 28.0x / 32.2x |
| welch | 256ch x 2^16, nperseg=256 | 112.95 ms | 2.56 ms | 1.09 ms | 44.2x / 103.2x |
| spectrogram | 16ch x 2^20 | 67.04 ms | 3.58 ms | 1.34 ms | 18.7x / 50.0x |
| stft | 16ch x 2^20, nperseg=1024 | 78.41 ms | 6.93 ms | 1.26 ms | 11.3x / 62.2x |
| fftconvolve | 2^20 x 4097 | 11.17 ms | 1.74 ms | 0.50 ms | 6.4x / 22.3x |
| fftconvolve | 2^22 x 257 | 47.87 ms | 2.60 ms | 1.02 ms | 18.4x / 47.1x |
| oaconvolve | 2^23 x 513 | 26.95 ms | 3.58 ms | 2.62 ms | 7.5x / 10.3x |
| correlate (batched) | 64ch x 2^18, 4096 taps | 65.15 ms | 8.46 ms | 6.26 ms | 7.7x / 10.4x |
| resample_poly | 16ch, 48k->44.1k (147/160) | 120.82 ms | 6.47 ms | 4.15 ms | 18.7x / 29.1x |
| upfirdn | 64ch x 2^18, up=2 down=3, 255 taps | 303.24 ms | 8.90 ms | 6.95 ms | 34.1x / 43.6x |
| resample (FFT) | 2^20 -> 2^18 | 4.36 ms | 0.35 ms | 0.28 ms | 12.3x / 15.4x |
| hilbert | 2^20 | 9.86 ms | 0.66 ms | 0.53 ms | 15.0x / 18.6x |
| resample (FFT, >1M) | 2^23 -> x0.75 | 67.41 ms | 6.38 ms | 5.52 ms | 10.6x / 12.2x |
| hilbert (>1M) | 2^23 | 103.98 ms | 8.21 ms | 6.43 ms | 12.7x / 16.2x |
| lfilter (FIR) | 64ch x 2^20, 257 taps | 1615.76 ms | 17.77 ms | 9.87 ms | 90.9x / 163.8x |
| filtfilt (FIR) | 64ch x 2^20, 257 taps | 3268.92 ms | 44.86 ms | 23.74 ms | 72.9x / 137.7x |
| sosfilt (IIR) | 256ch x 2^20, butter-8 | 1307.57 ms | 158.87 ms | 128.18 ms | 8.2x / 10.2x |
| sosfiltfilt (IIR) | 256ch x 2^20, butter-8 | 2693.86 ms | 602.42 ms | 518.40 ms | 4.5x / 5.2x |
| find_peaks | 2^23, prominence=1 | 223.61 ms | 221.54 ms | — | 1.0x / — |
