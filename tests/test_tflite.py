import json

import numpy as np
import pytest

pytest.importorskip("keras")
pytest.importorskip("tensorflow")

from anomaly_detector.compress import (
    INT8_VARIANTS, convert_fp32, convert_int8, make_representative_windows,
    make_wide_calibration, run_compression, save_compression_outputs, tflite_op_names,
)
from anomaly_detector.config import (
    CompressionConfig, Config, DatasetConfig, DetectorConfig, ModelConfig, TrainingConfig,
)
from anomaly_detector.inference import TFLiteAutoencoder
from anomaly_detector.model import build_conv_ae
from anomaly_detector.train import run_conv_ae


@pytest.fixture(scope="module")
def small():
    """Untrained small model + random calibration data + converted blobs."""
    rng = np.random.default_rng(0)
    model = build_conv_ae(32, 8, latent_dim=8, filter_count=8)
    rep = rng.standard_normal((100, 32, 8)).astype("float32")
    return {"model": model, "rep": rep, "x": rng.standard_normal((20, 32, 8)).astype("float32"),
            "fp32": convert_fp32(model), "int8": convert_int8(model, rep)}


def test_fp32_matches_keras(small):
    det = TFLiteAutoencoder(model_content=small["fp32"])
    assert det.input_shape == (1, 32, 8) and not det.is_quantized
    keras_out = small["model"].predict(small["x"], verbose=0)
    assert np.allclose(det.reconstruct(small["x"]), keras_out, atol=1e-4)


def test_int8_io_types_and_size(small):
    det = TFLiteAutoencoder(model_content=small["int8"])
    d = det.describe()
    assert det.is_quantized and d["input"]["dtype"] == "int8" and d["output"]["dtype"] == "int8"
    assert d["input"]["scale"] > 0
    out = det.reconstruct(small["x"])
    assert out.shape == small["x"].shape and np.isfinite(out).all()
    assert len(small["int8"]) < len(small["fp32"])


def test_builtin_ops_only(small):
    ops = tflite_op_names(small["int8"])
    assert ops is not None and "CONV_2D" in ops
    assert not any("FLEX" in o.upper() or "CUSTOM" in o.upper() for o in ops)


def test_clip_fraction(small):
    det = TFLiteAutoencoder(model_content=small["int8"])
    assert det.input_clip_fraction(np.zeros((5, 32, 8), dtype="float32")) == 0.0
    assert det.input_clip_fraction(np.full((5, 32, 8), 1e4, dtype="float32")) == 1.0
    fp = TFLiteAutoencoder(model_content=small["fp32"])
    assert fp.input_clip_fraction(np.full((5, 32, 8), 1e4, dtype="float32")) == 0.0


def test_interpreter_input_validation(small):
    det = TFLiteAutoencoder(model_content=small["fp32"])
    with pytest.raises(ValueError):
        det.reconstruct(np.zeros((3, 16, 8), dtype="float32"))
    assert det.reconstruction_errors(np.zeros((0, 32, 8), dtype="float32")).shape == (0,)
    with pytest.raises(ValueError):
        TFLiteAutoencoder()


def test_calibration_helpers_deterministic():
    pool = np.random.default_rng(1).standard_normal((300, 32, 8))
    a = make_representative_windows(pool, 50, seed=0)
    b = make_representative_windows(pool, 50, seed=0)
    assert a.shape == (50, 32, 8) and a.dtype == np.float32 and np.array_equal(a, b)
    w = make_wide_calibration(a, 4.0, seed=0)
    assert w.shape == (100, 32, 8) and np.array_equal(w[:50], a)
    ratio = np.abs(w[50:]).max(axis=(1, 2)) / np.abs(a).max(axis=(1, 2))
    assert ratio.min() >= 1.0 - 1e-5 and ratio.max() <= 4.0 + 1e-5
    with pytest.raises(ValueError):
        make_representative_windows(np.empty((0, 32, 8)), 5, 0)


def test_run_compression_end_to_end(synthetic_root, tmp_path):
    cfg = Config(
        dataset=DatasetConfig(root=str(synthetic_root), reference_rows=400, train_rows=300),
        model=ModelConfig(latent_dim=8, filter_count=8, kernel_size=5),
        training=TrainingConfig(epochs=15, batch_size=64, patience=5),
        compression=CompressionConfig(calibration_samples=100),
        detector=DetectorConfig(name="conv_ae", threshold=None),
    )
    _, model, _ = run_conv_ae(cfg)
    report, blobs, cards = run_compression(cfg, model, model_dir=tmp_path / "m_v1")

    assert set(blobs) == {"fp32_tflite", "int8_tflite", "int8_wide_tflite"}
    assert report["fp32_tflite_vs_keras"]["max_relative_score_diff"] < 1e-3
    for v in ("fp32_keras", "fp32_tflite", *INT8_VARIANTS):
        assert report["variants"][v]["roc_auc"] > 0.8
        assert report["variants"][v]["threshold"] > 0
    for v in INT8_VARIANTS:
        assert report["int8_analysis"][v]["score_agreement_vs_fp32_keras"]["pearson"] > 0.8
        assert set(report["int8_analysis"][v]["checks"]) == {
            "f1_within_tolerance", "far_within_tolerance", "mar_within_tolerance",
            "auc_drop_within_tolerance"}
    assert report["acceptance"]["selected"] in (*INT8_VARIANTS, "fp32_tflite")

    cfg2 = Config(**{**cfg.__dict__, "output": type(cfg.output)(
        metrics_dir=str(tmp_path / "reports"), model_dir=str(tmp_path / "m_v1"))})
    paths = save_compression_outputs(report, blobs, cards, cfg2, tmp_path / "m_v1")
    for key in ("detector", "meta", "report", "fp32_tflite", "int8_tflite", "int8_wide_tflite"):
        assert paths[key].exists()
    meta = json.loads(paths["meta"].read_text())
    assert meta["selected"] == report["acceptance"]["selected"]
    assert meta["variants"][meta["selected"]]["threshold"] > 0
    chosen = tmp_path / "m_v1" / meta["variants"][meta["selected"]]["file"]
    assert paths["detector"].read_bytes() == chosen.read_bytes()
