from pathlib import Path

import pytest

from anomaly_detector.config import Config, DetectorConfig, PreprocessingConfig
from anomaly_detector.config import WindowingConfig, load_config

DEFAULT = Path(__file__).parents[1] / "configs" / "config.yaml"


def test_default_yaml_loads():
    cfg = load_config(DEFAULT)
    assert cfg.windowing.window_size == 32
    assert cfg.detector.threshold == 3.0


def test_invalid_values():
    with pytest.raises(ValueError):
        Config(windowing=WindowingConfig(window_size=0))
    with pytest.raises(ValueError):
        Config(detector=DetectorConfig(threshold=-1.0))
    with pytest.raises(ValueError):
        Config(preprocessing=PreprocessingConfig(normalization_method="minmax"))


def test_unknown_key(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("windowing:\n  window_sizee: 8\n")
    with pytest.raises(ValueError, match="windowing"):
        load_config(p)


from anomaly_detector.config import DatasetConfig, ModelConfig, TrainingConfig


def _ae(**kw):
    return Config(detector=DetectorConfig(name="conv_ae", threshold=None), **kw)


def test_conv_ae_valid_and_threshold_optional():
    cfg = _ae()
    assert cfg.detector.threshold is None
    with pytest.raises(ValueError):  # baseline must have a numeric threshold
        Config(detector=DetectorConfig(name="zscore_baseline", threshold=None))


def test_conv_ae_invalid_params():
    with pytest.raises(ValueError, match="divisible by 4"):
        _ae(windowing=WindowingConfig(window_size=30))
    with pytest.raises(ValueError, match="odd"):
        _ae(model=ModelConfig(kernel_size=4))
    with pytest.raises(ValueError, match="train_rows"):
        _ae(dataset=DatasetConfig(train_rows=400, reference_rows=400))
    with pytest.raises(ValueError, match="validation region"):
        _ae(dataset=DatasetConfig(train_rows=390, reference_rows=400))
    with pytest.raises(ValueError, match="percentile"):
        _ae(training=TrainingConfig(threshold_percentile=100.0))


def test_conv_ae_yaml_loads():
    cfg = load_config(Path(__file__).parents[1] / "configs" / "conv_ae.yaml")
    assert cfg.detector.name == "conv_ae" and cfg.detector.threshold is None
    assert cfg.model.latent_dim == 16


from anomaly_detector.config import CompressionConfig


def test_compression_config_validation():
    assert Config().compression.calibration_samples == 500
    with pytest.raises(ValueError):
        Config(compression=CompressionConfig(calibration_samples=0))
    with pytest.raises(ValueError):
        Config(compression=CompressionConfig(max_metric_change=-0.1))
    with pytest.raises(ValueError):
        Config(compression=CompressionConfig(wide_calibration_max_amplitude=0.5))
    cfg = load_config(Path(__file__).parents[1] / "configs" / "conv_ae.yaml")
    assert cfg.compression.wide_calibration_max_amplitude == 4.0
