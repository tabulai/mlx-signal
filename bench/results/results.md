Benchmarks on Apple M4 Max (macOS 26.6.2), mlx 0.32.2, scipy 1.18.1, float32 inputs. e2e = NumPy in / NumPy out; device = MLX arrays in and out (steady state).
Every output passed the scipy correctness gate before timing; median of 9 runs after 3 warmups.

| function | shape | scipy | mlx-signal (e2e) | mlx-signal (device) | speedup (e2e / device) |
|---|---|---:|---:|---:|---:|
| welch | 64ch x 2^20, nperseg=1024 | 484.46 ms | 7.92 ms | 3.63 ms | 61.1x / 133.3x |
| welch | 1ch x 2^22, nperseg=4096 | 46.17 ms | 2.00 ms | 0.65 ms | 23.0x / 71.3x |
| welch | 256ch x 2^16, nperseg=256 | 112.58 ms | 2.60 ms | 0.98 ms | 43.4x / 114.5x |
| spectrogram | 16ch x 2^20 | 67.84 ms | 3.72 ms | 1.31 ms | 18.2x / 51.6x |
| stft | 16ch x 2^20, nperseg=1024 | 79.10 ms | 4.62 ms | 1.25 ms | 17.1x / 63.4x |
| fftconvolve | 2^20 x 4097 | 11.22 ms | 1.82 ms | 1.34 ms | 6.2x / 8.4x |
| fftconvolve | 2^22 x 257 | 49.18 ms | 2.92 ms | 0.74 ms | 16.8x / 66.3x |
| fftconvolve (pair) | 2^20 x 2^20 | 23.05 ms | 1.54 ms | 1.18 ms | 15.0x / 19.5x |
| correlate (auto) | 2^20 autocorrelation | 23.33 ms | 1.57 ms | 0.54 ms | 14.9x / 43.1x |
| oaconvolve | 2^23 x 513 | 27.36 ms | 2.93 ms | 1.30 ms | 9.3x / 21.0x |
| correlate (batched) | 64ch x 2^18, 4096 taps | 65.21 ms | 7.47 ms | 5.04 ms | 8.7x / 12.9x |
| resample_poly | 16ch, 48k->44.1k (147/160) | 120.68 ms | 3.21 ms | 0.92 ms | 37.6x / 130.7x |
| upfirdn | 64ch x 2^18, up=2 down=3, 255 taps | 302.12 ms | 4.28 ms | 2.65 ms | 70.6x / 114.2x |
| upfirdn (complex IQ) | 16ch x 2^20 c64, down=10, 201 taps | 181.73 ms | 3.64 ms | 1.57 ms | 49.9x / 116.0x |
| resample (FFT) | 2^20 -> 2^18 | 4.36 ms | 0.84 ms | 0.74 ms | 5.2x / 5.9x |
| hilbert | 2^20 | 8.46 ms | 1.32 ms | 0.44 ms | 6.4x / 19.4x |
| resample (FFT, >1M) | 2^23 -> x0.75 | 68.10 ms | 4.01 ms | 2.91 ms | 17.0x / 23.4x |
| hilbert (>1M) | 2^23 | 96.92 ms | 4.71 ms | 2.92 ms | 20.6x / 33.1x |
| lfilter (FIR) | 64ch x 2^20, 257 taps | 1636.53 ms | 13.27 ms | 4.97 ms | 123.4x / 329.3x |
| lfilter (FIR, f32 a) | 64ch x 2^20, 257 taps, f32 | 984.72 ms | 13.54 ms | 5.36 ms | 72.7x / 183.8x |
| filtfilt (FIR) | 64ch x 2^20, 257 taps | 3289.11 ms | 35.66 ms | 14.09 ms | 92.2x / 233.5x |
| filtfilt (FIR, f32 a) | 64ch x 2^20, 257 taps, f32 | 2004.95 ms | 36.22 ms | 13.95 ms | 55.4x / 143.7x |
| istft | 16ch x 2^20, nperseg=1024 | 180.41 ms | 5.31 ms | 1.50 ms | 34.0x / 120.5x |
| csd | 64ch x 2^20, nperseg=1024 | 952.09 ms | 16.17 ms | 7.68 ms | 58.9x / 124.0x |
| coherence | 64ch x 2^20, nperseg=1024 | 1972.20 ms | 16.78 ms | 8.40 ms | 117.5x / 234.9x |
| sosfilt (IIR) | 256ch x 2^20, butter-8 | 1378.86 ms | 45.83 ms | 12.75 ms | 30.1x / 108.2x |
| sosfilt (IIR, 1ch scan) | 1ch x 2^22, butter-8 | 21.53 ms | 2.34 ms | 1.96 ms | 9.2x / 11.0x |
| sosfiltfilt (IIR) | 256ch x 2^20, butter-8 | 2817.45 ms | 127.72 ms | 41.88 ms | 22.1x / 67.3x |
| lfilter (IIR) | 256ch x 2^20, butter-4 tf | 1672.30 ms | 45.98 ms | 12.31 ms | 36.4x / 135.9x |
| lfilter (IIR, 1ch scan) | 1ch x 2^22, butter-4 tf | 27.09 ms | 2.09 ms | 0.92 ms | 12.9x / 29.3x |
| filtfilt (IIR) | 256ch x 2^20, butter-4 tf | 3718.35 ms | 127.25 ms | 42.16 ms | 29.2x / 88.2x |
| find_peaks | 2^23, prominence=1 | 235.35 ms | 21.05 ms | — | 11.2x / — |
| peak_prominences | 2^23, 2.8M peaks | 162.48 ms | 16.26 ms | — | 10.0x / — |
