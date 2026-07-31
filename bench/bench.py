"""Benchmark mlx-signal against scipy.signal (and torchaudio, if installed).

Timing discipline:
- MLX is lazy: every MLX timing closure ends in mx.eval(...) so the GPU work is
  actually done inside the timed region.
- Two MLX numbers are reported: "e2e" starts from NumPy arrays and ends with
  np.array(result) (the drop-in-replacement experience, including unified-memory
  transfer), "device" starts and ends with evaluated MLX arrays (steady-state
  pipelines that stay on-GPU, e.g. feeding an MLX model).
- Median of `--repeat` runs after `--warmup` warmups.

Usage:
    python bench/bench.py            # full matrix (~1-2 min)
    python bench/bench.py --quick    # smaller matrix
    python bench/bench.py --out bench/results/results.md
"""

from __future__ import annotations

import argparse
import platform
import statistics
import subprocess
import time

import mlx.core as mx
import numpy as np
import scipy.signal as sps

import mlx_signal as msig


def _median_time(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def _eval_all(out):
    if isinstance(out, tuple):
        for o in out:
            if isinstance(o, mx.array):
                mx.eval(o)
    else:
        mx.eval(out)


def _to_np_all(out):
    if isinstance(out, tuple):
        return tuple(np.array(o) if isinstance(o, mx.array) else o for o in out)
    return np.array(out)


class Case:
    def __init__(self, name, detail, scipy_fn, mlx_fn, np_inputs):
        self.name = name
        self.detail = detail
        self.scipy_fn = scipy_fn
        self.mlx_fn = mlx_fn
        self.np_inputs = np_inputs


def build_cases(quick: bool) -> list[Case]:
    rng = np.random.default_rng(0)
    f32 = np.float32
    cases = []

    def sig(shape):
        return rng.standard_normal(shape).astype(f32)

    # --- spectral -----------------------------------------------------------
    for ch, n, nper in ([(64, 1 << 20, 1024), (1, 1 << 22, 4096), (256, 1 << 16, 256)]
                        if not quick else [(64, 1 << 18, 1024)]):
        x = sig((ch, n)) if ch > 1 else sig(n)
        cases.append(Case(
            "welch", f"{ch}ch x 2^{int(np.log2(n))}, nperseg={nper}",
            lambda x=x, nper=nper: sps.welch(x, nperseg=nper),
            lambda x=x, nper=nper: msig.welch(x, nperseg=nper),
            (x,),
        ))

    x = sig((16, 1 << 20)) if not quick else sig((4, 1 << 18))
    cases.append(Case(
        "spectrogram", f"{x.shape[0]}ch x 2^{int(np.log2(x.shape[1]))}",
        lambda x=x: sps.spectrogram(x, nperseg=1024, noverlap=512),
        lambda x=x: msig.spectrogram(x, nperseg=1024, noverlap=512),
        (x,),
    ))

    x = sig((16, 1 << 20)) if not quick else sig((4, 1 << 18))
    cases.append(Case(
        "stft", f"{x.shape[0]}ch x 2^{int(np.log2(x.shape[1]))}, nperseg=1024",
        lambda x=x: sps.stft(x, nperseg=1024),
        lambda x=x: msig.stft(x, nperseg=1024),
        (x,),
    ))

    # --- convolution --------------------------------------------------------
    for n, m in ([(1 << 20, 4097), (1 << 22, 257)] if not quick else [(1 << 19, 1025)]):
        a, b = sig(n), sig(m)
        cases.append(Case(
            "fftconvolve", f"2^{int(np.log2(n))} x {m}",
            lambda a=a, b=b: sps.fftconvolve(a, b),
            lambda a=a, b=b: msig.fftconvolve(a, b),
            (a, b),
        ))

    a, b = sig(1 << 23), sig(513)
    if not quick:
        cases.append(Case(
            "oaconvolve", "2^23 x 513",
            lambda a=a, b=b: sps.oaconvolve(a, b),
            lambda a=a, b=b: msig.oaconvolve(a, b),
            (a, b),
        ))

    a = sig((64, 1 << 18))
    b = sig((1, 4096))
    cases.append(Case(
        "correlate (batched)", "64ch x 2^18, 4096 taps",
        lambda a=a, b=b: sps.fftconvolve(a, b[:, ::-1], mode="full"),
        lambda a=a, b=b: msig.correlate(a, b, mode="full"),
        (a, b),
    ))

    # --- resampling ---------------------------------------------------------
    x = sig((16, 1 << 20)) if not quick else sig((4, 1 << 18))
    cases.append(Case(
        "resample_poly", f"{x.shape[0]}ch, 48k->44.1k (147/160)",
        lambda x=x: sps.resample_poly(x, 147, 160, axis=-1),
        lambda x=x: msig.resample_poly(x, 147, 160, axis=-1),
        (x,),
    ))

    x = sig((64, 1 << 18))
    h = np.array(msig.firwin(255, 0.4))
    cases.append(Case(
        "upfirdn", "64ch x 2^18, up=2 down=3, 255 taps",
        lambda x=x, h=h: sps.upfirdn(h, x, 2, 3, axis=-1),
        lambda x=x, h=h: msig.upfirdn(h, x, 2, 3, axis=-1),
        (x, h),
    ))

    x = sig(1 << 20)
    cases.append(Case(
        "resample (FFT)", "2^20 -> 2^18",
        lambda x=x: sps.resample(x, x.size // 4),
        lambda x=x: msig.resample(x, x.size // 4),
        (x,),
    ))
    cases.append(Case(
        "hilbert", "2^20",
        lambda x=x: sps.hilbert(x),
        lambda x=x: msig.hilbert(x),
        (x,),
    ))
    if not quick:
        xl = sig(1 << 23)
        cases.append(Case(
            "resample (FFT, >1M)", "2^23 -> x0.75",
            lambda x=xl: sps.resample(x, 3 * x.size // 4),
            lambda x=xl: msig.resample(x, 3 * x.size // 4),
            (xl,),
        ))
        cases.append(Case(
            "hilbert (>1M)", "2^23",
            lambda x=xl: sps.hilbert(x),
            lambda x=xl: msig.hilbert(x),
            (xl,),
        ))

    # --- filtering ----------------------------------------------------------
    x = sig((64, 1 << 20)) if not quick else sig((8, 1 << 18))
    b = np.array(msig.firwin(257, 0.1))
    cases.append(Case(
        "lfilter (FIR)", f"{x.shape[0]}ch x 2^{int(np.log2(x.shape[1]))}, 257 taps",
        lambda x=x, b=b: sps.lfilter(b, [1.0], x, axis=-1),
        lambda x=x, b=b: msig.lfilter(b, [1.0], x, axis=-1),
        (x, b),
    ))

    cases.append(Case(
        "filtfilt (FIR)", f"{x.shape[0]}ch x 2^{int(np.log2(x.shape[1]))}, 257 taps",
        lambda x=x, b=b: sps.filtfilt(b, [1.0], x, axis=-1),
        lambda x=x, b=b: msig.filtfilt(b, [1.0], x, axis=-1),
        (x, b),
    ))

    # --- istft (inverse-Stockham + gather-OLA kernel pair) -------------------
    xst = sig((16, 1 << 20)) if not quick else sig((4, 1 << 18))
    _, _, zst = sps.stft(xst, nperseg=1024)
    zst = zst.astype(np.complex64)
    cases.append(Case(
        "istft", f"{xst.shape[0]}ch x 2^{int(np.log2(xst.shape[1]))}, nperseg=1024",
        lambda z=zst: sps.istft(z, nperseg=1024),
        lambda z=zst: msig.istft(z, nperseg=1024),
        (zst,),
    ))

    # --- csd / coherence (fused two-signal kernel) ---------------------------
    x = sig((64, 1 << 20)) if not quick else sig((8, 1 << 18))
    y2 = sig(x.shape)
    cases.append(Case(
        "csd", f"{x.shape[0]}ch x 2^{int(np.log2(x.shape[1]))}, nperseg=1024",
        lambda x=x, y2=y2: sps.csd(x, y2, nperseg=1024),
        lambda x=x, y2=y2: msig.csd(x, y2, nperseg=1024),
        (x, y2),
    ))
    cases.append(Case(
        "coherence", f"{x.shape[0]}ch x 2^{int(np.log2(x.shape[1]))}, nperseg=1024",
        lambda x=x, y2=y2: sps.coherence(x, y2, nperseg=1024),
        lambda x=x, y2=y2: msig.coherence(x, y2, nperseg=1024),
        (x, y2),
    ))

    # --- IIR (batched-channel kernel) ---------------------------------------
    x = sig((256, 1 << 20)) if not quick else sig((64, 1 << 18))
    sos_iir = np.asarray(sps.butter(8, 0.2, output="sos"), dtype=np.float64)
    cases.append(Case(
        "sosfilt (IIR)", f"{x.shape[0]}ch x 2^{int(np.log2(x.shape[1]))}, butter-8",
        lambda x=x, s_=sos_iir: sps.sosfilt(s_.astype(np.float32), x, axis=-1),
        lambda x=x, s_=sos_iir: msig.sosfilt(s_, x, axis=-1),
        (x,),
    ))
    x1 = sig((1, 1 << 22))
    cases.append(Case(
        "sosfilt (IIR, 1ch scan)", "1ch x 2^22, butter-8",
        lambda x=x1, s_=sos_iir: sps.sosfilt(s_.astype(np.float32), x, axis=-1),
        lambda x=x1, s_=sos_iir: msig.sosfilt(s_, x, axis=-1),
        (x1,),
    ))
    cases.append(Case(
        "sosfiltfilt (IIR)", f"{x.shape[0]}ch x 2^{int(np.log2(x.shape[1]))}, butter-8",
        lambda x=x, s_=sos_iir: sps.sosfiltfilt(s_.astype(np.float32), x, axis=-1),
        lambda x=x, s_=sos_iir: msig.sosfiltfilt(s_, x, axis=-1),
        (x,),
    ))

    # --- peaks (honesty check: expected ~1x) --------------------------------
    x = sig(1 << 23) if not quick else sig(1 << 20)
    cases.append(Case(
        "find_peaks", f"2^{int(np.log2(x.size))}, prominence=1",
        lambda x=x: sps.find_peaks(x, prominence=1.0),
        lambda x=x: msig.find_peaks(x, prominence=1.0),
        (x,),
    ))

    return cases


def run(quick: bool, warmup: int, repeat: int) -> list[dict]:
    rows = []
    with msig.config_context(dispatch="mlx", warn_on_downcast=False):
        for case in build_cases(quick):
            t_scipy = _median_time(case.scipy_fn, warmup, repeat)

            mlx_fn = case.mlx_fn
            t_e2e = _median_time(lambda fn=mlx_fn: _to_np_all(fn()), warmup, repeat)

            mx_inputs = tuple(mx.array(a) for a in case.np_inputs)
            mx.eval(*mx_inputs)

            # re-bind the public call to the pre-evaluated device arrays
            device_fn = _device_variant(case, mx_inputs)
            if device_fn is not None:
                t_dev = _median_time(lambda fn=device_fn: _eval_all(fn()), warmup, repeat)
            else:
                t_dev = float("nan")

            rows.append({
                "name": case.name,
                "detail": case.detail,
                "scipy_ms": t_scipy * 1e3,
                "e2e_ms": t_e2e * 1e3,
                "dev_ms": t_dev * 1e3,
            })
            print(f"{case.name:22s} {case.detail:34s} scipy {t_scipy*1e3:9.2f} ms   "
                  f"mlx e2e {t_e2e*1e3:8.2f} ms ({t_scipy/t_e2e:5.1f}x)   "
                  f"mlx dev {t_dev*1e3:8.2f} ms ({t_scipy/t_dev:5.1f}x)")
    return rows


def _device_variant(case: Case, mx_inputs):
    """Re-bind the case's public call to pre-evaluated MLX inputs."""
    n = case.name
    a = mx_inputs
    if n == "welch":
        nper = int(case.detail.split("nperseg=")[1])
        return lambda: msig.welch(a[0], nperseg=nper)
    if n == "spectrogram":
        return lambda: msig.spectrogram(a[0], nperseg=1024, noverlap=512)
    if n == "stft":
        return lambda: msig.stft(a[0], nperseg=1024)
    if n == "fftconvolve":
        return lambda: msig.fftconvolve(a[0], a[1])
    if n == "oaconvolve":
        return lambda: msig.oaconvolve(a[0], a[1])
    if n == "correlate (batched)":
        return lambda: msig.correlate(a[0], a[1], mode="full")
    if n == "resample_poly":
        return lambda: msig.resample_poly(a[0], 147, 160, axis=-1)
    if n == "upfirdn":
        return lambda: msig.upfirdn(a[1], a[0], 2, 3, axis=-1)
    if n == "resample (FFT)":
        return lambda: msig.resample(a[0], a[0].size // 4)
    if n.startswith("resample (FFT, >1M"):
        return lambda: msig.resample(a[0], 3 * a[0].size // 4)
    if n == "hilbert":
        return lambda: msig.hilbert(a[0])
    if n.startswith("hilbert (>1M"):
        return lambda: msig.hilbert(a[0])
    if n == "lfilter (FIR)":
        return lambda: msig.lfilter(a[1], [1.0], a[0], axis=-1)
    if n == "filtfilt (FIR)":
        return lambda: msig.filtfilt(a[1], [1.0], a[0], axis=-1)
    if n == "istft":
        return lambda: msig.istft(a[0], nperseg=1024)
    if n == "csd":
        return lambda: msig.csd(a[0], a[1], nperseg=1024)
    if n == "coherence":
        return lambda: msig.coherence(a[0], a[1], nperseg=1024)
    if n == "sosfilt (IIR)":
        import scipy.signal as _sps

        sos_iir = _sps.butter(8, 0.2, output="sos")
        return lambda: msig.sosfilt(sos_iir, a[0], axis=-1)
    if n == "sosfilt (IIR, 1ch scan)":
        import scipy.signal as _sps

        sos_iir = _sps.butter(8, 0.2, output="sos")
        return lambda: msig.sosfilt(sos_iir, a[0], axis=-1)
    if n == "sosfiltfilt (IIR)":
        import scipy.signal as _sps

        sos_iir = _sps.butter(8, 0.2, output="sos")
        return lambda: msig.sosfiltfilt(sos_iir, a[0], axis=-1)
    if n == "find_peaks":
        return None  # host-side by design
    return None


def to_markdown(rows: list[dict]) -> str:
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                          capture_output=True, text=True).stdout.strip()
    import scipy

    lines = [
        f"Benchmarks on {chip} (macOS {platform.mac_ver()[0]}), "
        f"mlx {mx.__version__}, scipy {scipy.__version__}, float32 inputs. "
        "e2e = NumPy in / NumPy out; device = MLX arrays in and out (steady state).",
        "",
        "| function | shape | scipy | mlx-signal (e2e) | mlx-signal (device) "
        "| speedup (e2e / device) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        se = r["scipy_ms"] / r["e2e_ms"]
        sd = r["scipy_ms"] / r["dev_ms"] if r["dev_ms"] == r["dev_ms"] else float("nan")
        dev = f"{r['dev_ms']:.2f} ms" if sd == sd else "—"
        sdtxt = f"{sd:.1f}x" if sd == sd else "—"
        lines.append(
            f"| {r['name']} | {r['detail']} | {r['scipy_ms']:.2f} ms | "
            f"{r['e2e_ms']:.2f} ms | {dev} | {se:.1f}x / {sdtxt} |"
        )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not mx.metal.is_available():
        print("WARNING: no Metal GPU; benchmarking the MLX CPU backend is not "
              "representative.")

    rows = run(args.quick, args.warmup, args.repeat)
    md = to_markdown(rows)
    print("\n" + md)
    if args.out:
        import pathlib

        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
