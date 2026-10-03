"""Convert the trained Conv-AE to TFLite (FP32 + INT8 variants) and compare to FP32 Keras."""
from __future__ import annotations

import argparse

from anomaly_detector.compress import INT8_VARIANTS, run_compression, save_compression_outputs
from anomaly_detector.config import load_config


def _f(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _kb(n) -> str:
    return "n/a" if n is None else f"{n / 1024:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/conv_ae.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    report, blobs, cards = run_compression(cfg)
    paths = save_compression_outputs(report, blobs, cards, cfg, cfg.output.model_dir)

    print(f"{'variant':<18}{'size KB':>9}{'threshold':>11}{'f1':>8}{'far':>8}{'mar':>8}{'auc':>8}")
    for k, v in report["variants"].items():
        print(f"{k:<18}{_kb(v['size_bytes']):>9}{v['threshold']:>11.4f}"
              f"{_f(v['f1']):>8}{_f(v['far']):>8}{_f(v['mar']):>8}{_f(v['roc_auc']):>8}")
    t = report["fp32_tflite_vs_keras"]
    print(f"\nFP32 TFLite vs Keras score diff: max abs {t['max_abs_score_diff']:.2e}, "
          f"max relative {t['max_relative_score_diff']:.2e}")
    for v in INT8_VARIANTS:
        a = report["int8_analysis"][v]
        d, cl, ck = a["delta_vs_fp32_keras"], a["input_clip_fraction"], a["fp32_keras_with_clipped_inputs"]
        fx = a["with_fp32_keras_threshold"]
        print(f"\n[{v}]  {'PASSED' if a['passed'] else 'FAILED'}")
        print(f"  delta vs FP32 Keras: f1 {d['f1']:+.3f}  far {d['far']:+.3f}  mar {d['mar']:+.3f}  auc {d['roc_auc']:+.3f}")
        print(f"  score Pearson vs FP32: {a['score_agreement_vs_fp32_keras']['pearson']:.4f}   "
              f"median rel diff: {a['score_agreement_vs_fp32_keras']['median_relative_diff']:.3f}")
        print(f"  with FP32 threshold: f1 {_f(fx['f1'])}  far {_f(fx['far'])}  mar {_f(fx['mar'])}")
        print(f"  input range [{a['input_range'][0]:.1f}, {a['input_range'][1]:.1f}]  "
              f"clip fraction: validation {cl['validation']:.4f}  test {cl['test']:.4f}")
        print(f"  FP32 Keras with inputs clipped to this range: far {_f(ck['far'])}  auc {_f(ck['roc_auc'])}")
        if not a["passed"]:
            print(f"  failed checks: {[k for k, ok in a['checks'].items() if not ok]}")
    print(f"\nselected -> {report['acceptance']['selected']}  (copied to {paths['detector']})")
    print(f"ops (int8): {report['tflite']['int8_tflite']['ops']}")
    print(f"report -> {paths['report']}")


if __name__ == "__main__":
    main()
