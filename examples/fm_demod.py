"""FM demodulation on the Apple Silicon GPU.

Synthesizes 5 seconds of an FM-modulated IQ stream at 2.4 MS/s (a 1 kHz tone
plus a slow chirp as the message), then runs the standard receive chain twice —
once with scipy.signal, once with mlx-signal — and compares wall time and
recovered audio.

    IQ @ 2.4 MS/s
      -> channel low-pass          fftconvolve        [GPU]
      -> decimate to 240 kS/s      resample_poly      [GPU polyphase kernel]
      -> quadrature discriminator  angle-diff         [GPU]
      -> de-emphasis               FIR lfilter        [GPU]
      -> audio at 48 kHz           resample_poly      [GPU]

Run:  python examples/fm_demod.py
"""

import time

import mlx.core as mx
import numpy as np
import scipy.signal as sps

import mlx_signal as msig

FS = 2_400_000
AUDIO_FS = 48_000
DURATION = 5.0
KF = 75_000.0  # frequency deviation, Hz


def make_iq() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    n = int(FS * DURATION)
    t = np.arange(n) / FS
    message = 0.7 * np.sin(2 * np.pi * 1000 * t) + 0.3 * np.sin(
        2 * np.pi * (200 + 40 * t) * t
    )
    phase = 2 * np.pi * KF * np.cumsum(message) / FS
    iq = np.exp(1j * phase) + 0.02 * (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    )
    return iq.astype(np.complex64), message.astype(np.float32)


# 75 us de-emphasis, approximated with a linear-phase FIR so the whole chain
# stays on the GPU (single-pole IIR lands with the associative-scan roadmap)
def deemphasis_taps(fs: float) -> np.ndarray:
    freqs = np.linspace(0, 1, 64)
    f_hz = freqs * fs / 2
    gain = 1.0 / np.sqrt(1.0 + (2 * np.pi * f_hz * 75e-6) ** 2)
    return np.asarray(sps.firwin2(129, freqs, gain), dtype=np.float32)


CHAN_TAPS = np.asarray(sps.firwin(257, 200_000 / (FS / 2)), dtype=np.float32)
DEEMPH_TAPS = deemphasis_taps(240_000)


def demod_scipy(iq: np.ndarray) -> np.ndarray:
    x = sps.fftconvolve(iq, CHAN_TAPS.astype(np.complex64), mode="same")
    x = sps.resample_poly(x, 1, 10)  # 2.4 MS/s -> 240 kS/s
    disc = np.angle(x[1:] * np.conj(x[:-1]))
    audio = sps.lfilter(DEEMPH_TAPS, [1.0], disc)
    return sps.resample_poly(audio, 1, 5)  # 240 kS/s -> 48 kS/s


def demod_mlx(iq) -> mx.array:
    x = msig.fftconvolve(iq, mx.array(CHAN_TAPS.astype(np.complex64)), mode="same")
    x = msig.resample_poly(x, 1, 10, axis=0)
    z = x[1:] * mx.conj(x[:-1])
    disc = mx.arctan2(mx.imag(z), mx.real(z))
    audio = msig.lfilter(DEEMPH_TAPS, [1.0], disc)
    return msig.resample_poly(audio, 1, 5, axis=0)


def main():
    iq, message = make_iq()
    print(f"{len(iq):,} IQ samples @ {FS / 1e6:.1f} MS/s "
          f"({DURATION:.0f} s of signal)\n")

    t0 = time.perf_counter()
    audio_ref = demod_scipy(iq)
    t_scipy = time.perf_counter() - t0

    with msig.config_context(dispatch="mlx"):
        demod_mlx(iq)  # warmup (kernel compilation)
        t0 = time.perf_counter()
        audio_mlx = np.array(demod_mlx(iq))
        t_mlx = time.perf_counter() - t0

    err = np.sqrt(np.mean((audio_mlx - audio_ref) ** 2)) / np.sqrt(
        np.mean(audio_ref**2)
    )
    print(f"scipy.signal chain : {t_scipy * 1e3:8.1f} ms")
    print(f"mlx-signal chain   : {t_mlx * 1e3:8.1f} ms   ({t_scipy / t_mlx:.1f}x)")
    print(f"relative RMS difference between the two chains: {err:.2e}")

    # recovered message quality: align for the chain's group delay first
    ref = sps.resample_poly(message, 1, 50)  # message at 48 kHz
    a = audio_mlx - audio_mlx.mean()
    r = ref - ref.mean()
    c = np.correlate(a, r, "full")
    lag = int(np.argmax(np.abs(c))) - (len(r) - 1)
    corr = np.abs(c).max() / (np.linalg.norm(a) * np.linalg.norm(r))
    print(f"correlation of demodulated audio with the true message: {corr:.4f} "
          f"(group delay {lag} samples)")


if __name__ == "__main__":
    main()
