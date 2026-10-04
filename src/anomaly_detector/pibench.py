"""Raspberry Pi benchmark core. Needs only numpy and a TFLite runtime (no TensorFlow/pandas).

Each (variant, xnnpack-mode) runs in its OWN subprocess so peak RSS is per variant:
    parent  -> run_benchmark()        (hardware info, orchestration, report)
    worker  -> python -m anomaly_detector.pibench --worker ...   (one variant)

Measured from Python, so latencies include interpreter binding overhead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .baseline import predict
from .inference import TFLiteAutoencoder
from .metrics import compute_metrics

REPORT_SCHEMA = "pi_benchmark_report_v1"
BUNDLE_SCHEMA = "pi_bundle_v1"


@dataclass(frozen=True)
class Tolerances:
    """Desktop-vs-device agreement criteria, fixed BEFORE any Pi measurement."""

    min_decision_agreement: float = 0.999     # share of windows with identical anomaly decision
    max_metric_delta: float = 0.002           # |delta| of pooled F1, FAR, MAR vs desktop
    fp32_max_relative_score_diff: float = 1e-4  # FP32 only; INT8 kernels may differ by rounding


# ----------------------------------------------------------------------------- probes
THROTTLE_BITS = {
    0: "under_voltage_now", 1: "freq_capped_now", 2: "throttled_now", 3: "soft_temp_limit_now",
    16: "under_voltage_occurred", 17: "freq_capped_occurred", 18: "throttled_occurred",
    19: "soft_temp_limit_occurred",
}


def decode_throttled(value: int) -> dict[str, bool]:
    """Decode `vcgencmd get_throttled` bit flags."""
    return {name: bool(value & (1 << bit)) for bit, name in THROTTLE_BITS.items()}


def _read_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read().strip().replace("\x00", "")
    except OSError:
        return None


def _float_or_none(text: str | None, scale: float) -> float | None:
    try:
        return float(text) / scale if text else None
    except ValueError:
        return None


def read_thermal() -> dict:
    """CPU temperature, current frequency and firmware throttle flags (None if unavailable)."""
    thr = None
    if shutil.which("vcgencmd"):
        try:
            out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                                 text=True, timeout=2).stdout.strip()
            thr = int(out.split("=")[1], 16)
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            thr = None
    return {
        "temp_c": _float_or_none(_read_text("/sys/class/thermal/thermal_zone0/temp"), 1000.0),
        "freq_mhz": _float_or_none(
            _read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"), 1000.0),
        "throttled": None if thr is None else hex(thr),
        "throttle_flags": None if thr is None else decode_throttled(thr),
    }


def _pkg_version(*names: str) -> dict:
    from importlib import metadata
    found = {}
    for n in names:
        try:
            found[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            pass
    return found


def read_hardware() -> dict:
    """Identify the machine; `is_raspberry_pi` gates whether results count as Pi benchmarks."""
    model = _read_text("/proc/device-tree/model")
    cpuinfo = _read_text("/proc/cpuinfo") or ""
    meminfo = _read_text("/proc/meminfo") or ""
    mem_kb = next((int(l.split()[1]) for l in meminfo.splitlines() if l.startswith("MemTotal")), None)
    pick = lambda key: next((l.split(":", 1)[1].strip() for l in cpuinfo.splitlines()
                             if l.lower().startswith(key)), None)
    return {
        "device_model": model,
        "is_raspberry_pi": bool(model and model.startswith("Raspberry Pi")),
        "machine": platform.machine(), "platform": platform.platform(),
        "kernel": platform.release(), "cpu_count": os.cpu_count(),
        "cpu_model_name": pick("model name"), "cpu_revision": pick("revision"),
        "mem_total_mb": None if mem_kb is None else round(mem_kb / 1024, 1),
        "cpu_governor": _read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "tflite_packages": _pkg_version("ai-edge-litert", "tflite-runtime", "tensorflow",
                                        "tensorflow-cpu"),
    }


def current_rss_mb() -> float | None:
    txt = _read_text("/proc/self/status")
    if not txt:
        return None
    for line in txt.splitlines():
        if line.startswith("VmRSS"):
            return round(int(line.split()[1]) / 1024, 2)
    return None


def peak_rss_mb() -> float | None:
    """Process-lifetime peak resident set size (ru_maxrss)."""
    try:
        import resource
    except ImportError:  # Windows
        return None
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(v / (1024 * 1024) if sys.platform == "darwin" else v / 1024, 2)


# ------------------------------------------------------------------------- statistics
def latency_stats(ns: np.ndarray) -> dict:
    ms = np.asarray(ns, dtype=np.float64) / 1e6
    if ms.size == 0:
        raise ValueError("no latency samples")
    return {"n": int(ms.size), "mean_ms": float(ms.mean()), "median_ms": float(np.median(ms)),
            "p95_ms": float(np.percentile(ms, 95)), "p99_ms": float(np.percentile(ms, 99)),
            "min_ms": float(ms.min()), "max_ms": float(ms.max())}


def compare_to_reference(scores: np.ndarray, ref_scores: np.ndarray, y_true: np.ndarray,
                         threshold: float, tol: Tolerances, is_fp32: bool) -> dict:
    """Device scores vs desktop reference scores for the same windows and threshold."""
    scores, ref = np.asarray(scores, np.float64), np.asarray(ref_scores, np.float64)
    if scores.shape != ref.shape:
        raise ValueError(f"shape mismatch {scores.shape} vs {ref.shape}")
    rel = np.abs(scores - ref) / np.maximum(np.abs(ref), 1e-12)
    dev, dsk = predict(scores, threshold), predict(ref, threshold)
    m_dev, m_ref = compute_metrics(y_true, dev), compute_metrics(y_true, dsk)
    deltas = {k: (None if m_dev[k] is None or m_ref[k] is None else m_dev[k] - m_ref[k])
              for k in ("f1", "far", "mar")}
    agreement = float((dev == dsk).mean())
    checks = {
        "decision_agreement": agreement >= tol.min_decision_agreement,
        "metrics_within_tolerance": all(d is not None and abs(d) <= tol.max_metric_delta
                                        for d in deltas.values()),
    }
    if is_fp32:
        checks["fp32_relative_score_diff"] = float(rel.max()) <= tol.fp32_max_relative_score_diff
    return {"max_relative_score_diff": float(rel.max()), "median_relative_score_diff": float(np.median(rel)),
            "max_abs_score_diff": float(np.max(np.abs(scores - ref))),
            "decision_agreement": agreement, "metric_deltas_vs_desktop": deltas,
            "checks": checks, "passed": all(checks.values())}


def _summarize_thermal(samples: list[dict]) -> dict:
    temps = [s["temp_c"] for s in samples if s["temp_c"] is not None]
    freqs = [s["freq_mhz"] for s in samples if s["freq_mhz"] is not None]
    last = samples[-1]
    now = [s["throttle_flags"] for s in samples if s["throttle_flags"]]
    return {
        "samples": len(samples),
        "temp_start_c": samples[0]["temp_c"], "temp_end_c": last["temp_c"],
        "temp_max_c": max(temps) if temps else None,
        "freq_min_mhz": min(freqs) if freqs else None, "freq_max_mhz": max(freqs) if freqs else None,
        "throttled_end": last["throttled"],
        "throttled_now_in_any_sample": (any(f["throttled_now"] or f["freq_capped_now"] for f in now)
                                        if now else None),
        "throttled_occurred_since_boot": (last["throttle_flags"]["throttled_occurred"]
                                          or last["throttle_flags"]["freq_capped_occurred"]
                                          ) if last["throttle_flags"] else None,
    }


# ------------------------------------------------------------------------------ worker
def _read_chunk(path: Path, start: int, n: int, w: int, c: int) -> np.ndarray:
    row = w * c
    arr = np.fromfile(path, dtype=np.float32, count=n * row, offset=start * row * 4)
    return arr.reshape(n, w, c)


def run_variant(bundle_dir: str | Path, variant: str, *, num_threads: int = 1,
                xnnpack: bool = True, max_windows: int | None = None,
                sustained_seconds: float = 0.0, bucket_seconds: float = 10.0,
                warmup: int = 100, chunk: int = 512,
                tol: Tolerances | None = None) -> dict:
    """Benchmark one variant over the bundle's windows (this process only)."""
    b = Path(bundle_dir)
    man = json.loads((b / "manifest.json").read_text(encoding="utf-8"))
    tol = tol or Tolerances(**man["tolerances"])
    w, c = man["window_size"], man["n_channels"]
    n = man["n_windows"] if not max_windows else min(man["n_windows"], max_windows)
    v = man["variants"][variant]
    win_path = b / man["files"]["windows"]

    rss_start = current_rss_mb()
    t0 = time.perf_counter()
    det = TFLiteAutoencoder(model_path=b / v["file"], num_threads=num_threads, xnnpack=xnnpack)
    load_s = time.perf_counter() - t0
    rss_loaded = current_rss_mb()
    labels = np.fromfile(b / man["files"]["labels"], dtype=np.int8, count=n)
    ref = np.fromfile(b / v["ref_scores"], dtype=np.float64, count=n)

    det.reconstruction_errors(_read_chunk(win_path, 0, min(warmup, n), w, c))  # warm-up

    thermal = [read_thermal()]
    last_sample = time.monotonic()
    scores = np.empty(n, dtype=np.float64)
    inv, tot = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
    cpu0, wall0 = time.process_time(), time.perf_counter()
    for s in range(0, n, chunk):
        m = min(chunk, n - s)
        e, i_ns, t_ns = det.timed_errors(_read_chunk(win_path, s, m, w, c))
        scores[s : s + m], inv[s : s + m], tot[s : s + m] = e, i_ns, t_ns
        if time.monotonic() - last_sample >= 1.0:
            thermal.append(read_thermal())
            last_sample = time.monotonic()
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    thermal.append(read_thermal())
    peak_batch = peak_rss_mb()

    sustained = None
    if sustained_seconds > 0:
        pool = _read_chunk(win_path, 0, min(n, 256), w, c)
        buckets, t_end = [], time.monotonic() + sustained_seconds
        while time.monotonic() < t_end:
            b_start = time.monotonic()
            b_end = min(t_end, b_start + bucket_seconds)
            parts: list[np.ndarray] = []
            while time.monotonic() < b_end:
                parts.append(det.timed_errors(pool)[1])
            lat = np.concatenate(parts)
            th = read_thermal()
            buckets.append({
                "t_start_s": round(b_start - (t_end - sustained_seconds), 1),
                "median_invoke_ms": float(np.median(lat)) / 1e6,
                "p95_invoke_ms": float(np.percentile(lat, 95)) / 1e6,
                "windows_per_s": float(lat.size / (time.monotonic() - b_start)),
                "temp_c": th["temp_c"], "freq_mhz": th["freq_mhz"], "throttled": th["throttled"]})
            thermal.append(th)
        first, last = buckets[0]["median_invoke_ms"], buckets[-1]["median_invoke_ms"]
        sustained = {"seconds": sustained_seconds, "bucket_seconds": bucket_seconds,
                     "buckets": buckets,
                     "median_invoke_change_first_to_last_pct": 100.0 * (last - first) / first}

    thr = v["threshold"]
    is_fp32 = variant.startswith("fp32")
    m = compute_metrics(labels, predict(scores, thr))
    return {
        "variant": variant, "file": v["file"], "size_bytes": v["size_bytes"],
        "xnnpack": "default" if xnnpack else "off", "num_threads": num_threads,
        "backend": det.backend, "n_windows": int(n), "model_load_s": load_s,
        "io": det.describe(),
        "latency": {"invoke": latency_stats(inv), "end_to_end": latency_stats(tot),
                    "throughput_windows_per_s": float(n / (tot.sum() / 1e9))},
        "cpu_utilization_pct_of_one_core": float(100.0 * cpu / wall) if wall > 0 else None,
        "memory_mb": {"rss_start": rss_start, "rss_after_model_load": rss_loaded,
                      "peak_rss_after_batch": peak_batch, "peak_rss_final": peak_rss_mb()},
        "thermal": _summarize_thermal(thermal),
        "sustained": sustained,
        "accuracy": {"threshold": thr, "f1": m["f1"], "far": m["far"], "mar": m["mar"],
                     "n_points": m["n_points"]},
        "desktop_comparison": compare_to_reference(scores, ref, labels, thr, tol, is_fp32),
    }


def _worker_main(a: argparse.Namespace) -> None:
    try:
        res = run_variant(a.bundle, a.variant, num_threads=a.threads, xnnpack=(a.xnnpack == "default"),
                          max_windows=a.max_windows or None, sustained_seconds=a.sustained_seconds,
                          bucket_seconds=a.bucket_seconds)
    except RuntimeError as exc:  # e.g. XNNPACK toggle unsupported by this backend
        res = {"variant": a.variant, "xnnpack": a.xnnpack, "error": str(exc)}
    Path(a.out).write_text(json.dumps(res), encoding="utf-8")


# ---------------------------------------------------------------------------- parent
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def run_benchmark(bundle_dir: str | Path, *, variants: list[str] | None = None,
                  compare_xnnpack: bool = False, threads: int = 1,
                  sustained_seconds: float = 60.0, bucket_seconds: float = 10.0,
                  max_windows: int | None = None, cooldown_seconds: float = 20.0) -> dict:
    """Run every variant in its own subprocess and assemble the benchmark report."""
    b = Path(bundle_dir).resolve()
    man = json.loads((b / "manifest.json").read_text(encoding="utf-8"))
    if man.get("schema_version") != BUNDLE_SCHEMA:
        raise ValueError(f"unsupported bundle schema {man.get('schema_version')!r}")
    variants = variants or list(man["variants"])
    unknown = [x for x in variants if x not in man["variants"]]
    if unknown:
        raise ValueError(f"variants not in bundle: {unknown}")
    for name, v in man["variants"].items():  # integrity of what we are about to benchmark
        if _sha256(b / v["file"]) != v["sha256"]:
            raise ValueError(f"{v['file']}: sha256 differs from manifest (corrupted copy?)")

    pkg_root = Path(__file__).resolve().parents[1]  # dir containing anomaly_detector/
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(pkg_root)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))}
    modes = ["default"] + (["off"] if compare_xnnpack else [])
    hw = read_hardware()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        first = True
        for mode in modes:
            for var in variants:
                if not first and cooldown_seconds > 0:
                    time.sleep(cooldown_seconds)
                first = False
                out = Path(tmp) / f"{var}_{mode}.json"
                cmd = [sys.executable, "-m", "anomaly_detector.pibench", "--worker",
                       "--bundle", str(b), "--variant", var, "--xnnpack", mode,
                       "--threads", str(threads),
                       "--sustained-seconds", str(sustained_seconds if mode == "default" else 0),
                       "--bucket-seconds", str(bucket_seconds),
                       "--max-windows", str(max_windows or 0), "--out", str(out)]
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
                if proc.returncode != 0 or not out.exists():
                    raise RuntimeError(f"worker failed for {var}/{mode}:\n{proc.stderr[-2000:]}")
                res = json.loads(out.read_text(encoding="utf-8"))
                res["xnnpack_delegate_reported_in_log"] = "XNNPACK" in proc.stderr
                results.append(res)

    base = {(r["xnnpack"]): r for r in results if r.get("variant") == "fp32_tflite" and "error" not in r}
    for r in results:  # speed relative to FP32 under the same XNNPACK mode
        ref = base.get(r.get("xnnpack"))
        if ref and "error" not in r:
            r["speedup_vs_fp32_median_invoke"] = (
                ref["latency"]["invoke"]["median_ms"] / r["latency"]["invoke"]["median_ms"])

    return {
        "schema_version": REPORT_SCHEMA,
        "model_version": man["model_version"],
        "is_raspberry_pi": hw["is_raspberry_pi"],
        "hardware": hw,
        "run": {"started_utc": started, "threads": threads, "modes": modes,
                "sustained_seconds": sustained_seconds, "max_windows": max_windows,
                "cooldown_seconds": cooldown_seconds, "n_windows_in_bundle": man["n_windows"],
                "bundle_integrity": "sha256 verified",
                "desktop_reference": man["desktop"]},
        "tolerances": man["tolerances"],
        "results": results,
        "notes": [
            "Latency is measured from Python and includes interpreter binding overhead.",
            "peak_rss is the process-lifetime maximum (ru_maxrss); each variant ran in its own process.",
            "cpu_utilization is process CPU time / wall time during the batch pass, in % of ONE core.",
            "Temperature/frequency/throttle fields are null where sysfs or vcgencmd is unavailable.",
            "If is_raspberry_pi is false these are NOT Raspberry Pi benchmarks.",
        ],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--xnnpack", choices=["default", "off"], default="default")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--sustained-seconds", type=float, default=0.0)
    ap.add_argument("--bucket-seconds", type=float, default=10.0)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--out", required=True)
    _worker_main(ap.parse_args())
