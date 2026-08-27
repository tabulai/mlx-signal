Benchmarks on Apple M4 Max (macOS 26.2), mlx 0.32.0, scipy 1.18.0, float32 inputs. e2e = NumPy in / NumPy out; device = MLX arrays in and out (steady state).

| function | shape | scipy | mlx-signal (e2e) | mlx-signal (device) | speedup (e2e / device) |
|---|---|---:|---:|---:|---:|
| welch | 64ch x 2^20, nperseg=1024 | 479.37 ms | 10.45 ms | 4.87 ms | 45.9x / 98.4x |
| welch | 1ch x 2^22, nperseg=4096 | 47.57 ms | 1.86 ms | 1.58 ms | 25.5x / 30.2x |
| welch | 256ch x 2^16, nperseg=256 | 110.27 ms | 5.20 ms | 1.06 ms | 21.2x / 104.4x |
| spectrogram | 16ch x 2^20 | 66.66 ms | 3.69 ms | 1.27 ms | 18.0x / 52.6x |
| stft | 16ch x 2^20, nperseg=1024 | 78.39 ms | 4.84 ms | 1.24 ms | 16.2x / 63.3x |
| fftconvolve | 2^20 x 4097 | 10.97 ms | 1.56 ms | 1.40 ms | 7.0x / 7.8x |
| fftconvolve | 2^22 x 257 | 44.16 ms | 2.70 ms | 2.17 ms | 16.3x / 20.3x |
| fftconvolve (pair) | 2^20 x 2^20 | 21.28 ms | 2.67 ms | 0.98 ms | 8.0x / 21.7x |
| correlate (auto) | 2^20 autocorrelation | 20.89 ms | 2.00 ms | 1.76 ms | 10.4x / 11.9x |
| oaconvolve | 2^23 x 513 | 26.56 ms | 4.61 ms | 1.26 ms | 5.8x / 21.1x |
| correlate (batched) | 64ch x 2^18, 4096 taps | 63.46 ms | 7.07 ms | 4.54 ms | 9.0x / 14.0x |
| resample_poly | 16ch, 48k->44.1k (147/160) | 119.99 ms | 5.56 ms | 0.97 ms | 21.6x / 124.0x |
| upfirdn | 64ch x 2^18, up=2 down=3, 255 taps | 294.46 ms | 4.36 ms | 2.31 ms | 67.6x / 127.7x |
| upfirdn (complex IQ) | 16ch x 2^20 c64, down=10, 201 taps | 180.26 ms | 3.88 ms | 1.27 ms | 46.4x / 142.5x |
| resample (FFT) | 2^20 -> 2^18 | 4.19 ms | 0.43 ms | 0.36 ms | 9.7x / 11.7x |
| hilbert | 2^20 | 9.07 ms | 1.60 ms | 1.31 ms | 5.7x / 6.9x |
| resample (FFT, >1M) | 2^23 -> x0.75 | 65.87 ms | 6.10 ms | 5.30 ms | 10.8x / 12.4x |
| hilbert (>1M) | 2^23 | 98.73 ms | 8.25 ms | 6.10 ms | 12.0x / 16.2x |
| lfilter (FIR) | 64ch x 2^20, 257 taps | 1591.15 ms | 12.92 ms | 4.86 ms | 123.2x / 327.7x |
| filtfilt (FIR) | 64ch x 2^20, 257 taps | 3210.78 ms | 34.32 ms | 13.54 ms | 93.6x / 237.1x |
| istft | 16ch x 2^20, nperseg=1024 | 174.01 ms | 21.28 ms | 2.92 ms | 8.2x / 59.5x |
| csd | 64ch x 2^20, nperseg=1024 | 921.74 ms | 15.98 ms | 8.05 ms | 57.7x / 114.4x |
| coherence | 64ch x 2^20, nperseg=1024 | 1896.76 ms | 17.55 ms | 9.64 ms | 108.1x / 196.8x |
| sosfilt (IIR) | 256ch x 2^20, butter-8 | 1288.13 ms | 42.15 ms | 11.76 ms | 30.6x / 109.6x |
| sosfilt (IIR, 1ch scan) | 1ch x 2^22, butter-8 | 20.28 ms | 2.40 ms | 1.86 ms | 8.4x / 10.9x |
| sosfiltfilt (IIR) | 256ch x 2^20, butter-8 | 2646.82 ms | 130.12 ms | 39.98 ms | 20.3x / 66.2x |
| lfilter (IIR) | 256ch x 2^20, butter-4 tf | 1504.03 ms | 42.34 ms | 11.86 ms | 35.5x / 126.8x |
| lfilter (IIR, 1ch scan) | 1ch x 2^22, butter-4 tf | 22.60 ms | 3.32 ms | 0.92 ms | 6.8x / 24.5x |
| filtfilt (IIR) | 256ch x 2^20, butter-4 tf | 2987.04 ms | 131.07 ms | 40.14 ms | 22.8x / 74.4x |
| find_peaks | 2^23, prominence=1 | 215.09 ms | 80.15 ms | — | 2.7x / — |
| peak_prominences | 2^23, 2.8M peaks | 155.08 ms | 19.46 ms | — | 8.0x / — |
