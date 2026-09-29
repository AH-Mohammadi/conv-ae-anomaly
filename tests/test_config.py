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
