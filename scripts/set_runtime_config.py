"""Write a runtime-config command file: the local mechanism that simulates a future
Coordinator issuing a hot-swap command, with no networking involved.

  python scripts/set_runtime_config.py --threshold 1.2
  python scripts/replay_stream.py --runtime-config-file reports/metrics/runtime_config_command.json

The write is atomic (write to a temp file, then os.replace), so a concurrently running
replay process never observes a half-written command file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--calibration-scale", type=float, default=None)
    ap.add_argument("--out", default="reports/metrics/runtime_config_command.json")
    a = ap.parse_args()

    hparams = {}
    if a.threshold is not None:
        hparams["threshold"] = a.threshold
    if a.calibration_scale is not None:
        hparams["calibration_scale"] = a.calibration_scale
    if not hparams:
        raise SystemExit("give at least one of --threshold / --calibration-scale")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(hparams, indent=2), encoding="utf-8")
    tmp.replace(out)  # atomic
    print(f"wrote {hparams} -> {out}")


if __name__ == "__main__":
    main()
