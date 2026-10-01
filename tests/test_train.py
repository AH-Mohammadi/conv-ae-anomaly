import json

import numpy as np
import pytest

pytest.importorskip("keras")

from anomaly_detector.config import Config, DatasetConfig, DetectorConfig, ModelConfig, TrainingConfig
from anomaly_detector.train import run_conv_ae, save_artifacts, select_threshold


def _cfg(root, **kw):
    return Config(
        dataset=DatasetConfig(root=str(root), reference_rows=400, train_rows=300),
        model=ModelConfig(latent_dim=8, filter_count=8, kernel_size=5),
        training=TrainingConfig(epochs=15, batch_size=64, patience=5),
        detector=DetectorConfig(name="conv_ae", threshold=None),
        **kw,
    )


def test_select_threshold_percentile():
    assert select_threshold(np.arange(101, dtype=float), 90.0) == pytest.approx(90.0)
    with pytest.raises(ValueError):
        select_threshold(np.empty(0), 99.0)


def test_end_to_end_and_artifacts(synthetic_root, tmp_path):
    cfg = _cfg(synthetic_root)
    card, model, art = run_conv_ae(cfg, model_dir=tmp_path / "m_v1")
    assert card["model"] == "conv_ae" and card["n_files"] == 3
    assert card["threshold"] > 0
    assert "validation" in card["protocol"]["threshold_source"]
    assert card["roc_auc"] > 0.8  # injected anomaly is a large shift
    assert card["training"]["n_train_windows"] > 0 and card["training"]["n_val_windows"] > 0

    d = save_artifacts(card, model, art, cfg, tmp_path / "m_v1")
    meta = json.loads((d / "artifact.json").read_text())
    assert meta["threshold"] == card["threshold"] and meta["window_size"] == 32
    assert (d / "model.keras").exists() and (d / "scores.npz").exists()


def test_training_is_deterministic(synthetic_root):
    a, _, _ = run_conv_ae(_cfg(synthetic_root))
    b, _, _ = run_conv_ae(_cfg(synthetic_root))
    assert a["threshold"] == b["threshold"]
    assert (a["f1"], a["far"], a["mar"]) == (b["f1"], b["far"], b["mar"])
