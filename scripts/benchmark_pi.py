"""Benchmark the TFLite variants on this machine (intended: Raspberry Pi).

Run from inside the exported bundle:  python3 benchmark_pi.py
Needs only numpy and a TFLite runtime (ai-edge-litert, tflite-runtime or tensorflow).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from anomaly_detector.pibench import run_benchmark


def _f(v, d=3):
    return "n/a" if v is None else f"{v:.{d}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--sustained-seconds", type=float, default=60.0)
    ap.add_argument("--bucket-seconds", type=float, default=10.0)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--cooldown-seconds", type=float, default=20.0)
    ap.add_argument("--compare-xnnpack", action="store_true",
                    help="also run each variant with the default delegate disabled")
    ap.add_argument("--out", default="pi_benchmark_report.json")
    a = ap.parse_args()

    report = run_benchmark(
        a.bundle, variants=a.variants, compare_xnnpack=a.compare_xnnpack, threads=a.threads,
        sustained_seconds=a.sustained_seconds, bucket_seconds=a.bucket_seconds,
        max_windows=a.max_windows, cooldown_seconds=a.cooldown_seconds)
    Path(a.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    hw = report["hardware"]
    print(f"Hardware: {hw['device_model'] or hw['platform']}  ({hw['machine']}, {hw['cpu_count']} cores, "
          f"{hw['mem_total_mb']} MB)")
    print(f"Python {hw['python']}, numpy {hw['numpy']}, runtime packages {hw['tflite_packages']}")
    if not report["is_raspberry_pi"]:
        print("WARNING: this is NOT a Raspberry Pi; results are not Pi benchmarks.")
    print(f"\n{'variant':<18}{'xnnpack':>8}{'KB':>7}{'invoke ms':>11}{'p95':>8}{'e2e ms':>8}"
          f"{'win/s':>8}{'peakMB':>8}{'cpu%':>6}{'Tmax':>6}{'f1':>7}{'far':>7}{'mar':>7}  match")
    for r in report["results"]:
        if "error" in r:
            print(f"{r['variant']:<18}{r['xnnpack']:>8}  skipped: {r['error']}")
            continue
        L, M, T, A, D = r["latency"], r["memory_mb"], r["thermal"], r["accuracy"], r["desktop_comparison"]
        print(f"{r['variant']:<18}{r['xnnpack']:>8}{r['size_bytes'] / 1024:>7.1f}"
              f"{L['invoke']['median_ms']:>11.4f}{L['invoke']['p95_ms']:>8.3f}"
              f"{L['end_to_end']['median_ms']:>8.3f}{L['throughput_windows_per_s']:>8.0f}"
              f"{_f(M['peak_rss_final'], 1):>8}{_f(r['cpu_utilization_pct_of_one_core'], 0):>6}"
              f"{_f(T['temp_max_c'], 1):>6}{_f(A['f1']):>7}{_f(A['far']):>7}{_f(A['mar']):>7}"
              f"  {'PASS' if D['passed'] else 'FAIL'}")
        s = r.get("sustained")
        if s:
            print(f"    sustained {s['seconds']:.0f}s: median invoke first->last bucket "
                  f"{s['median_invoke_change_first_to_last_pct']:+.1f}%, throttled flags at end: {T['throttled_end']}")
    print(f"\nreport -> {a.out}")


if __name__ == "__main__":
    main()
