"""Train the Conv-AE, evaluate it with the shared harness, save model + scorecard."""
from __future__ import annotations

import argparse

from anomaly_detector.config import load_config
from anomaly_detector.evaluate import format_scorecard, write_scorecard
from anomaly_detector.train import run_conv_ae, save_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/conv_ae.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.detector.name != "conv_ae":
        raise SystemExit("configs passed to train.py must set detector.name: conv_ae")
    card, model, artifacts = run_conv_ae(cfg)
    model_dir = save_artifacts(card, model, artifacts, cfg, cfg.output.model_dir)
    path = write_scorecard(card, cfg.output.metrics_dir)

    tr = card["training"]
    print(f"trained on {tr['n_train_windows']} windows, validated on {tr['n_val_windows']}, "
          f"epochs={tr['epochs_run']}, params={tr['n_parameters']}, "
          f"best_val_loss={tr['best_val_loss']:.5f}\n")
    print(format_scorecard(card))
    print("\nthreshold sensitivity (informational):")
    for p, v in card["diagnostics"]["threshold_sensitivity"].items():
        print(f"  p{p:<5} thr={v['threshold']:.5f}  f1={v['f1']:.3f}  far={v['far']:.3f}  mar={v['mar']:.3f}")
    print(f"\nmodel -> {model_dir}\nscorecard -> {path}")


if __name__ == "__main__":
    main()
