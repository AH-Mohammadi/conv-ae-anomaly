"""Export a self-contained benchmark bundle to copy onto the Raspberry Pi."""
from __future__ import annotations

import argparse

from anomaly_detector.config import load_config
from anomaly_detector.pibundle import export_bundle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/conv_ae.yaml")
    ap.add_argument("--out", default="pi_bundle")
    ap.add_argument("--max-windows", type=int, default=None,
                    help="evenly subsample the test windows (default: all)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = export_bundle(cfg, cfg.output.model_dir, a.out, a.max_windows)
    print(f"bundle -> {out}\nCopy this folder to the Pi, then run: python3 benchmark_pi.py")


if __name__ == "__main__":
    main()
