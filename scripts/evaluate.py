"""Run the evaluation harness and write a JSON scorecard."""
from __future__ import annotations

import argparse

from anomaly_detector.config import load_config
from anomaly_detector.evaluate import run_evaluation, write_scorecard


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    card = run_evaluation(cfg)
    path = write_scorecard(card, cfg.output.metrics_dir)

    print(f"{'file':<16}{'points':>8}{'f1':>8}{'far':>8}{'mar':>8}")
    for row in card["per_file"]:
        print(
            f"{row['file']:<16}{row['n_points']:>8}"
            f"{_fmt(row['f1']):>8}{_fmt(row['far']):>8}{_fmt(row['mar']):>8}"
        )
    print("-" * 48)
    print(
        f"{'POOLED':<16}{card['counts']['n_points']:>8}"
        f"{_fmt(card['f1']):>8}{_fmt(card['far']):>8}{_fmt(card['mar']):>8}"
    )
    excl = card["protocol"]["excluded_files"]
    print(f"\nmodel={card['model']}  files={card['n_files']}  excluded={excl}")
    print(f"scorecard -> {path}")


if __name__ == "__main__":
    main()
