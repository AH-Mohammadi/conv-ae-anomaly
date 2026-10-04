"""Desktop side: export a self-contained benchmark bundle for the Raspberry Pi.

The bundle holds the .tflite files, the test windows/labels, DESKTOP reference scores
for each variant (computed from the .tflite files on disk), and the minimal source
modules (numpy-only) needed to run the benchmark with just a TFLite runtime.
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .baseline import predict
from .config import Config
from .evaluate import load_prepared, test_windows
from .inference import TFLiteAutoencoder
from .metrics import compute_metrics
from .pibench import BUNDLE_SCHEMA, Tolerances

SOURCE_MODULES = ("__init__.py", "inference.py", "metrics.py", "baseline.py", "pibench.py")

README = """Raspberry Pi benchmark bundle
=============================
Requirements
  * 64-bit Raspberry Pi OS: `uname -m` must print aarch64 (no TFLite wheels for 32-bit).
  * python3 and numpy, plus ONE TFLite runtime:
      pip install ai-edge-litert      # preferred; aarch64 wheels exist for CPython 3.10-3.13
      pip install tflite-runtime      # alternative; CPython 3.8-3.11, needs glibc >= 2.34
    (full tensorflow also works but inflates the measured memory)

Run
  python3 benchmark_pi.py                                    # all variants, 60 s sustained each
  python3 benchmark_pi.py --sustained-seconds 600 --compare-xnnpack

Then copy pi_benchmark_report.json back to the desktop (reports/metrics/).
For a fair thermal measurement use the official power supply, keep cooling and ambient
temperature consistent, note them, and run nothing else heavy.
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_bundle(cfg: Config, model_dir: str | Path, out_dir: str | Path,
                  max_windows: int | None = None) -> Path:
    model_dir, out = Path(model_dir), Path(out_dir)
    meta = json.loads((model_dir / "tflite_meta.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)

    prepared, excluded = load_prepared(cfg)
    pairs = [test_windows(pf, cfg) for pf in prepared]
    windows = np.concatenate([p[0] for p in pairs]).astype(np.float32)
    labels = np.concatenate([p[1] for p in pairs]).astype(np.int8)
    subsampled = False
    if max_windows and max_windows < windows.shape[0]:
        idx = np.linspace(0, windows.shape[0] - 1, max_windows).astype(np.int64)
        windows, labels, subsampled = windows[idx], labels[idx], True
    n, w, c = windows.shape

    windows.tofile(out / "windows_f32.bin")
    labels.tofile(out / "labels_i8.bin")

    variants, backend = {}, None
    for name, info in meta["variants"].items():
        src = model_dir / info["file"]
        shutil.copyfile(src, out / info["file"])
        det = TFLiteAutoencoder(model_path=out / info["file"], num_threads=1)
        backend = det.backend
        ref = det.reconstruction_errors(windows)
        ref_file = f"ref_scores_{name}.bin"
        ref.tofile(out / ref_file)
        m = compute_metrics(labels, predict(ref, info["threshold"]))
        variants[name] = {
            "file": info["file"], "size_bytes": (out / info["file"]).stat().st_size,
            "sha256": _sha256(out / info["file"]), "threshold": info["threshold"],
            "ref_scores": ref_file,
            "desktop_reference_metrics": {k: m[k] for k in ("f1", "far", "mar")},
        }

    pkg = Path(__file__).resolve().parent
    pkg_out = out / "anomaly_detector"
    pkg_out.mkdir(exist_ok=True)
    for f in SOURCE_MODULES:
        shutil.copyfile(pkg / f, pkg_out / f)
    script = pkg.parents[1] / "scripts" / "benchmark_pi.py"
    if not script.exists():
        raise FileNotFoundError(f"{script} not found; run from the repository checkout")
    shutil.copyfile(script, out / "benchmark_pi.py")
    (out / "README.txt").write_text(README, encoding="utf-8")

    manifest = {
        "schema_version": BUNDLE_SCHEMA,
        "model_version": meta["model_version"], "selected": meta["selected"],
        "window_size": w, "n_channels": c, "n_windows": int(n), "subsampled": subsampled,
        "files": {"windows": "windows_f32.bin", "labels": "labels_i8.bin"},
        "dtypes": {"windows": "float32", "labels": "int8", "ref_scores": "float64"},
        "variants": variants,
        "tolerances": asdict(Tolerances()),
        "protocol": {"excluded_files": excluded,
                     "windows": "all test-region windows, pooled across files, harness order",
                     "reference_scores": "desktop TFLite interpreter, 1 thread, from the .tflite files in this bundle"},
        "desktop": {"python": sys.version.split()[0], "numpy": np.__version__,
                    "platform": platform.platform(), "machine": platform.machine(),
                    "tflite_backend": backend},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out
