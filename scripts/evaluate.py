"""Run the z-score baseline through the evaluation harness and write a JSON scorecard."""
from __future__ import annotations

import argparse

from anomaly_detector.config import load_config
from anomaly_detector.evaluate import format_scorecard, run_evaluation, write_scorecard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    card = run_evaluation(cfg)
    path = write_scorecard(card, cfg.output.metrics_dir)
    print(format_scorecard(card))
    print(f"scorecard -> {path}")


if __name__ == "__main__":
    main()
