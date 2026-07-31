"""64-channel EEG alpha-band power on the GPU.

Synthesizes 10 minutes of 64-channel EEG at 500 Hz (1/f background plus
occipital-weighted 10 Hz alpha bursts), computes the Welch PSD of every channel
in one batched GPU call, and integrates 8-12 Hz band power — then does the same
with scipy.signal and compares wall time and results.

Run:  python examples/eeg_bandpower.py
"""

import time

import numpy as np
import scipy.signal as sps

import mlx_signal as msig

FS = 500.0
MINUTES = 10
N_CH = 64


def make_eeg() -> np.ndarray:
    rng = np.random.default_rng(42)
    n = int(FS * 60 * MINUTES)
    t = np.arange(n) / FS

    # 1/f-ish background: filtered white noise per channel
    white = rng.standard_normal((N_CH, n)).astype(np.float32)
    b = sps.firwin(65, 45 / (FS / 2))
    eeg = sps.lfilter(b, [1.0], white, axis=-1).astype(np.float32)

    # alpha bursts, strongest on the last ("occipital") channels
    burst_env = (np.sin(2 * np.pi * 0.05 * t) > 0.3).astype(np.float32)
    alpha = np.sin(2 * np.pi * 10.0 * t).astype(np.float32) * burst_env
    gain = np.linspace(0.05, 1.0, N_CH, dtype=np.float32)[:, None]
    return eeg + 8.0 * gain * alpha


def band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> np.ndarray:
    sel = (freqs >= lo) & (freqs <= hi)
    return np.trapezoid(psd[:, sel], freqs[sel], axis=-1)


def main():
    eeg = make_eeg()
    nperseg = 2048  # ~4 s segments; a power of two keeps the Metal FFT on its fast path
    print(f"{N_CH} channels x {eeg.shape[1]:,} samples "
          f"({MINUTES} min @ {FS:.0f} Hz), Welch nperseg={nperseg}\n")

    t0 = time.perf_counter()
    f_ref, p_ref = sps.welch(eeg, fs=FS, nperseg=nperseg)
    t_scipy = time.perf_counter() - t0
    alpha_ref = band_power(f_ref, p_ref, 8, 12)

    with msig.config_context(dispatch="mlx"):
        msig.welch(eeg, fs=FS, nperseg=nperseg)  # warmup
        t0 = time.perf_counter()
        f, p = msig.welch(eeg, fs=FS, nperseg=nperseg)
        f, p = np.array(f), np.array(p)
        t_mlx = time.perf_counter() - t0
    alpha = band_power(f, p, 8, 12)

    print(f"scipy.signal welch : {t_scipy * 1e3:8.1f} ms")
    print(f"mlx-signal welch   : {t_mlx * 1e3:8.1f} ms   ({t_scipy / t_mlx:.1f}x)")
    rel = np.max(np.abs(alpha - alpha_ref) / alpha_ref)
    print(f"max relative difference in alpha power: {rel:.2e}\n")

    print("alpha (8-12 Hz) band power by channel group:")
    for name, sl in [("frontal (0-15)", slice(0, 16)),
                     ("central (24-39)", slice(24, 40)),
                     ("occipital (48-63)", slice(48, 64))]:
        print(f"  {name:18s} {alpha[sl].mean():10.2f}")

    peaks, props = msig.find_peaks(p[-1], prominence=1.0)
    peak_freqs = f[peaks][np.argsort(props["prominences"])[::-1]][:3]
    print(f"\ntop spectral peaks, occipital channel: {np.round(peak_freqs, 2)} Hz")


if __name__ == "__main__":
    main()
