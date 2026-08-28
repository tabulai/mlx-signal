"""Cross-library benchmark: mlx-signal vs every other Mac implementation we know of.

Libraries (all optional; rows are skipped when not installed):
- scipy.signal          — the baseline (single-threaded C)
- mlx-signal            — this library (Metal GPU via MLX)
- torch / torchaudio    — CPU (multithreaded) and the MPS GPU backend
- jax.scipy.signal      — XLA-compiled multithreaded CPU (jitted)
- librosa               — stft (numpy) and resample (soxr backend)
- soxr                  — resampling only (C, the audio-world reference)

Fairness rules:
- Identical math where conventions allow: STFT uses periodic hann, n_fft=1024,
  hop=512, center=False everywhere; scipy-family scaling (1/win.sum()) is
  aligned before verification. Every implementation's output is checked against
  the scipy reference before results are published (column "vs scipy").
- resample is task-level: 48 kHz -> 44.1 kHz at each library's default quality
  (anti-aliasing filter lengths differ; noted, and outputs are checked by
  aligned multi-channel correlation, gain, bias, and RMS error instead of
  elementwise error).
- e2e timings start and end in NumPy (includes device transfer); device
  timings (GPU rows only) start and end with device-resident arrays.
- GPU work is synchronized inside the timed region (mx.eval /
  torch.mps.synchronize / block_until_ready). Median of 5 runs after warmup;
  anything slower than 3 s per call is reported from a single run.

Usage:  python bench/bench_cross.py [--out bench/results/cross.md]
"""

from __future__ import annotations

import argparse
import importlib
import platform
import statistics
import subprocess
import time
import warnings

import mlx.core as mx
import numpy as np
import scipy.signal as sps

import mlx_signal_processing as msig

try:
    from ._validation import rel_err, resample_quality
except ImportError:  # direct execution: python bench/bench_cross.py
    from _validation import rel_err, resample_quality

warnings.filterwarnings("ignore")

HAVE = {}
SKIP_REASONS = {}
for _mod in ("torch", "torchaudio", "jax", "librosa", "soxr"):
    try:
        HAVE[_mod] = importlib.import_module(_mod)
    except Exception as exc:
        HAVE[_mod] = None
        SKIP_REASONS[_mod] = f"{type(exc).__name__}: {exc}"


def _optional_attr(module_name, attr):
    """Resolve lazy optional APIs without letting one backend abort the suite."""
    module = HAVE[module_name]
    if module is None:
        return None
    try:
        return getattr(module, attr)
    except Exception as exc:
        SKIP_REASONS[f"{module_name}.{attr}"] = f"{type(exc).__name__}: {exc}"
        return None


LIBROSA_STFT = _optional_attr("librosa", "stft")
LIBROSA_RESAMPLE = _optional_attr("librosa", "resample")

torch = HAVE["torch"]
MPS = bool(torch and torch.backends.mps.is_available())
if HAVE["jax"]:
    import jax
    import jax.numpy as jnp
    import jax.scipy.signal as jss

GUARD_S = 3.0


def timed(fn, warmup=2, repeat=5):
    fn()  # warmup / compile
    t0 = time.perf_counter()
    fn()
    first = time.perf_counter() - t0
    if first > GUARD_S:
        return first, 1
    for _ in range(max(0, warmup - 1)):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), repeat


class Row:
    def __init__(self, impl, backend, e2e_s, nrep, device_s=None, check=""):
        self.impl = impl
        self.backend = backend
        self.e2e_ms = e2e_s * 1e3
        self.nrep = nrep
        self.device_ms = device_s * 1e3 if device_s is not None else None
        self.check = check


def _mx_dev(x):
    a = mx.array(x)
    mx.eval(a)
    return a


def _mps(x):
    t = torch.from_numpy(x).to("mps")
    torch.mps.synchronize()
    return t


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------


def task_welch(rng):
    x = rng.standard_normal((64, 1 << 20)).astype(np.float32)
    ref = sps.welch(x, nperseg=1024)[1]
    rows = [Row("scipy.signal.welch", "CPU (1 thread)",
                *timed(lambda: sps.welch(x, nperseg=1024)), check="—")]

    with msig.config_context(dispatch="mlx"):
        e2e, n1 = timed(lambda: np.array(msig.welch(x, nperseg=1024)[1]))
        xd = _mx_dev(x)
        dev, _ = timed(lambda: mx.eval(msig.welch(xd, nperseg=1024)[1]))
        chk = rel_err(np.array(msig.welch(x, nperseg=1024)[1]), ref)
    rows.append(Row("mlx-signal.welch", "Metal GPU", e2e, n1, dev, chk))

    if HAVE["jax"]:
        fn = jax.jit(lambda a: jss.welch(a, nperseg=1024)[1])
        e2e, n1 = timed(lambda: np.asarray(fn(jnp.asarray(x)).block_until_ready()))
        chk = rel_err(np.asarray(fn(jnp.asarray(x))), ref)
        rows.append(Row("jax.scipy.signal.welch (jit)", "CPU (XLA)", e2e, n1, check=chk))
    return "welch — 64ch x 2^20, nperseg=1024 (no torch/librosa equivalent exists)", rows


def task_stft(rng):
    x = rng.standard_normal((16, 1 << 20)).astype(np.float32)
    kw = dict(nperseg=1024, noverlap=512, boundary=None, padded=False)
    win_sum = sps.get_window("hann", 1024).sum()
    ref = sps.stft(x, **kw)[2] * win_sum  # align to unscaled-STFT convention
    rows = [Row("scipy.signal.stft", "CPU (1 thread)",
                *timed(lambda: sps.stft(x, **kw)), check="—")]

    with msig.config_context(dispatch="mlx"):
        e2e, n1 = timed(lambda: np.array(msig.stft(x, **kw)[2]))
        xd = _mx_dev(x)
        dev, _ = timed(lambda: mx.eval(msig.stft(xd, **kw)[2]))
        chk = rel_err(np.array(msig.stft(x, **kw)[2]) * win_sum, ref)
    rows.append(Row("mlx-signal.stft", "Metal GPU", e2e, n1, dev, chk))

    if torch:
        for devname in ["cpu"] + (["mps"] if MPS else []):
            w = torch.hann_window(1024, periodic=True, device=devname)

            def f(dev=devname, w=w):
                t = torch.from_numpy(x).to(dev)
                s = torch.stft(t, 1024, 512, window=w, center=False, return_complex=True)
                return s.cpu().numpy()

            e2e, n1 = timed(f)
            chk = rel_err(f(), ref)
            devt = None
            if devname == "mps":
                xt = _mps(x)

                def g(xt=xt, w=w):
                    s = torch.stft(xt, 1024, 512, window=w, center=False, return_complex=True)
                    torch.mps.synchronize()
                    return s

                devt, _ = timed(g)
            label = "CPU (multithread)" if devname == "cpu" else "MPS GPU"
            rows.append(Row("torch.stft", label, e2e, n1, devt, chk))

    if LIBROSA_STFT is not None:
        def f():
            return LIBROSA_STFT(x, n_fft=1024, hop_length=512, center=False)

        e2e, n1 = timed(f)
        rows.append(
            Row("librosa.stft", "CPU (numpy)", e2e, n1, check=rel_err(f(), ref))
        )

    if HAVE["jax"]:
        fn = jax.jit(lambda a: jss.stft(a, **kw)[2])
        e2e, n1 = timed(lambda: np.asarray(fn(jnp.asarray(x)).block_until_ready()))
        chk = rel_err(np.asarray(fn(jnp.asarray(x))) * win_sum, ref)
        rows.append(Row("jax.scipy.signal.stft (jit)", "CPU (XLA)", e2e, n1, check=chk))
    return "stft — 16ch x 2^20, n_fft=1024, hop=512, center=False, periodic hann", rows


def task_fftconvolve(rng):
    a = rng.standard_normal(1 << 20).astype(np.float32)
    b = rng.standard_normal(4097).astype(np.float32)
    ref = sps.fftconvolve(a, b)
    rows = [Row("scipy.signal.fftconvolve", "CPU (1 thread)",
                *timed(lambda: sps.fftconvolve(a, b)), check="—")]

    with msig.config_context(dispatch="mlx"):
        e2e, n1 = timed(lambda: np.array(msig.fftconvolve(a, b)))
        ad, bd = _mx_dev(a), _mx_dev(b)
        dev, _ = timed(lambda: mx.eval(msig.fftconvolve(ad, bd)))
        chk = rel_err(np.array(msig.fftconvolve(a, b)), ref)
    rows.append(Row("mlx-signal.fftconvolve", "Metal GPU", e2e, n1, dev, chk))

    if HAVE["torchaudio"]:
        taf = HAVE["torchaudio"].functional
        for devname in ["cpu"] + (["mps"] if MPS else []):

            def f(dev=devname):
                ta_ = torch.from_numpy(a).to(dev)
                tb_ = torch.from_numpy(b).to(dev)
                return taf.fftconvolve(ta_, tb_, mode="full").cpu().numpy()

            e2e, n1 = timed(f)
            chk = rel_err(f(), ref)
            devt = None
            if devname == "mps":
                at, bt = _mps(a), _mps(b)

                def g(at=at, bt=bt):
                    y = taf.fftconvolve(at, bt, mode="full")
                    torch.mps.synchronize()
                    return y

                devt, _ = timed(g)
            label = "CPU (multithread)" if devname == "cpu" else "MPS GPU"
            rows.append(Row("torchaudio fftconvolve", label, e2e, n1, devt, chk))

    if HAVE["jax"]:
        fn = jax.jit(lambda p, q: jss.fftconvolve(p, q, mode="full"))
        e2e, n1 = timed(lambda: np.asarray(fn(jnp.asarray(a), jnp.asarray(b)).block_until_ready()))
        chk = rel_err(np.asarray(fn(jnp.asarray(a), jnp.asarray(b))), ref)
        rows.append(Row("jax fftconvolve (jit)", "CPU (XLA)", e2e, n1, check=chk))
    return "fftconvolve — 2^20 x 4097, mode=full", rows


def task_resample(rng):
    x = rng.standard_normal((16, 1 << 20)).astype(np.float32)
    ref = sps.resample_poly(x, 147, 160, axis=-1)
    rows = [Row("scipy resample_poly (147/160)", "CPU (1 thread)",
                *timed(lambda: sps.resample_poly(x, 147, 160, axis=-1)), check="—")]

    with msig.config_context(dispatch="mlx"):
        e2e, n1 = timed(lambda: np.array(msig.resample_poly(x, 147, 160, axis=-1)))
        xd = _mx_dev(x)
        dev, _ = timed(lambda: mx.eval(msig.resample_poly(xd, 147, 160, axis=-1)))
        chk = rel_err(np.array(msig.resample_poly(x, 147, 160, axis=-1)), ref)
    rows.append(Row("mlx-signal resample_poly", "Metal GPU", e2e, n1, dev, chk))

    if HAVE["torchaudio"]:
        taf = HAVE["torchaudio"].functional
        for devname in ["cpu"] + (["mps"] if MPS else []):

            def f(dev=devname):
                t = torch.from_numpy(x).to(dev)
                return taf.resample(t, 160, 147).cpu().numpy()

            e2e, n1 = timed(f)
            chk = resample_quality(f(), ref)
            devt = None
            if devname == "mps":
                xt = _mps(x)

                def g(xt=xt):
                    y = taf.resample(xt, 160, 147)
                    torch.mps.synchronize()
                    return y

                devt, _ = timed(g)
            label = "CPU (multithread)" if devname == "cpu" else "MPS GPU"
            rows.append(Row("torchaudio resample (width=6)", label, e2e, n1, devt, chk))

    if LIBROSA_RESAMPLE is not None:
        def f():
            return LIBROSA_RESAMPLE(
                x, orig_sr=48000, target_sr=44100, res_type="soxr_hq", axis=-1
            )

        e2e, n1 = timed(f)
        rows.append(
            Row(
                "librosa.resample (soxr_hq)", "CPU (C)", e2e, n1,
                check=resample_quality(f(), ref),
            )
        )

    if HAVE["soxr"]:
        soxr = HAVE["soxr"]

        def f():
            return soxr.resample(x.T, 48000, 44100, quality="HQ").T

        e2e, n1 = timed(f)
        rows.append(Row("soxr (HQ)", "CPU (C)", e2e, n1, check=resample_quality(f(), ref)))
    return ("resample 48 kHz -> 44.1 kHz — 16ch x 2^20 (task-level: each library's "
            "default anti-aliasing filter; scipy/mlx use 3201 taps, torchaudio width=6)"), rows


def task_fir(rng):
    x = rng.standard_normal((64, 1 << 20)).astype(np.float32)
    b = np.asarray(sps.firwin(257, 0.1), dtype=np.float32)
    ref = sps.lfilter(b, [1.0], x, axis=-1)
    rows = [Row("scipy.signal.lfilter", "CPU (1 thread)",
                *timed(lambda: sps.lfilter(b, [1.0], x, axis=-1)), check="—")]

    with msig.config_context(dispatch="mlx"):
        e2e, n1 = timed(lambda: np.array(msig.lfilter(b, [1.0], x, axis=-1)))
        xd = _mx_dev(x)
        dev, _ = timed(lambda: mx.eval(msig.lfilter(b, [1.0], xd, axis=-1)))
        chk = rel_err(np.array(msig.lfilter(b, [1.0], x, axis=-1)), ref)
    rows.append(Row("mlx-signal.lfilter (FIR)", "Metal GPU", e2e, n1, dev, chk))

    if torch:
        kflip = np.ascontiguousarray(b[::-1])
        for devname in ["cpu"] + (["mps"] if MPS else []):
            kt = torch.from_numpy(kflip).to(devname)[None, None, :]

            def f(dev=devname, kt=kt):
                t = torch.from_numpy(x).to(dev)[:, None, :]
                y = torch.nn.functional.conv1d(torch.nn.functional.pad(t, (256, 0)), kt)
                return y[:, 0, :].cpu().numpy()

            e2e, n1 = timed(f)
            chk = rel_err(f(), ref)
            devt = None
            if devname == "mps":
                xt = _mps(x)[:, None, :]

                def g(xt=xt, kt=kt):
                    y = torch.nn.functional.conv1d(torch.nn.functional.pad(xt, (256, 0)), kt)
                    torch.mps.synchronize()
                    return y

                devt, _ = timed(g)
            label = "CPU (multithread)" if devname == "cpu" else "MPS GPU"
            rows.append(Row("torch conv1d (causal FIR)", label, e2e, n1, devt, chk))

    if HAVE["torchaudio"]:
        taf = HAVE["torchaudio"].functional
        a_t = np.zeros(257, dtype=np.float32)
        a_t[0] = 1.0
        for devname in ["cpu"] + (["mps"] if MPS else []):

            def f(dev=devname):
                t = torch.from_numpy(x).to(dev)
                y = taf.lfilter(t, torch.from_numpy(a_t).to(dev),
                                torch.from_numpy(b).to(dev), clamp=False)
                return y.cpu().numpy()

            e2e, n1 = timed(f)
            chk = rel_err(f(), ref)
            label = "CPU (multithread)" if devname == "cpu" else "MPS GPU"
            rows.append(Row("torchaudio lfilter (IIR machinery)", label, e2e, n1, check=chk))
    return "causal FIR filter — 64ch x 2^20, 257 taps", rows


# ---------------------------------------------------------------------------


def to_markdown(sections):
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                          capture_output=True, text=True).stdout.strip()
    import scipy

    hdr = [f"Cross-library benchmarks on {chip} (macOS {platform.mac_ver()[0]}), float32."]
    vers = [f"scipy {scipy.__version__}", f"mlx {mx.__version__}"]
    if torch:
        vers.append(f"torch {torch.__version__} ({torch.get_num_threads()} CPU threads)")
    if HAVE["torchaudio"]:
        vers.append(f"torchaudio {HAVE['torchaudio'].__version__}")
    if HAVE["jax"]:
        vers.append(f"jax {HAVE['jax'].__version__}")
    if HAVE["librosa"]:
        vers.append(f"librosa {HAVE['librosa'].__version__}")
    if HAVE["soxr"]:
        vers.append(f"soxr {HAVE['soxr'].__version__}")
    hdr.append(", ".join(vers) + ".")
    if SKIP_REASONS:
        skipped = "; ".join(f"{name}: {reason}" for name, reason in SKIP_REASONS.items())
        hdr.append(f"Skipped optional backends/capabilities: {skipped}.")
    hdr.append("")
    hdr.append("e2e = NumPy in / NumPy out; device = data resident on the accelerator. "
               "'vs scipy' verifies each output against the scipy reference "
               "(max relative error, or aligned multi-channel quality for resamplers). "
               "Median of 5 runs after 2 warmups; rows marked (1 run) exceeded "
               "the 3 s guard.")
    lines = hdr
    for title, rows in sections:
        base = rows[0].e2e_ms
        lines += ["", f"### {title}", "",
                  "| implementation | backend | e2e | device | speedup vs scipy | vs scipy |",
                  "|---|---|---:|---:|---:|---|"]
        for r in rows:
            dev = f"{r.device_ms:.2f} ms" if r.device_ms is not None else "—"
            note = " (1 run)" if r.nrep == 1 else ""
            lines.append(
                f"| {r.impl} | {r.backend} | {r.e2e_ms:.2f} ms{note} | {dev} "
                f"| {base / r.e2e_ms:.1f}x | {r.check} |"
            )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if torch:
        torch.set_grad_enabled(False)
    rng = np.random.default_rng(0)
    sections = []
    for task in (task_welch, task_stft, task_fftconvolve, task_resample, task_fir):
        title, rows = task(rng)
        sections.append((title, rows))
        print(f"\n{title}")
        for r in rows:
            dev = f"  dev {r.device_ms:8.2f} ms" if r.device_ms is not None else ""
            print(f"  {r.impl:38s} {r.backend:18s} e2e {r.e2e_ms:9.2f} ms{dev}   [{r.check}]")

    md = to_markdown(sections)
    if SKIP_REASONS:
        print("\nSkipped optional backends/capabilities:")
        for name, reason in SKIP_REASONS.items():
            print(f"  {name}: {reason}")
    if args.out:
        import pathlib

        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md)
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
