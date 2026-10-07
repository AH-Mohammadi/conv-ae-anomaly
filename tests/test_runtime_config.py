import pytest

from anomaly_detector.calibration import MIN_SCALE
from anomaly_detector.runtime_config import (
    ARTIFACT_DEPENDENT_EXAMPLES, HOT_SWAPPABLE_FIELDS, ConfigError, RuntimeConfig,
    apply_new_config, classify_param,
)


def _cfg(**kw):
    defaults = dict(threshold=0.9, calibration_scale=0.17, model_version="conv_ae_v1:fp32_tflite")
    defaults.update(kw)
    return RuntimeConfig(**defaults)


def test_classify_param():
    assert classify_param("threshold") == "hot_swappable"
    assert classify_param("calibration_scale") == "hot_swappable"
    assert classify_param("window_size") == "requires_new_model_artifact"
    assert classify_param("totally_unknown_name") == "requires_new_model_artifact"


def test_apply_valid_threshold_change_is_immutable():
    cur = _cfg()
    new = apply_new_config(cur, {"threshold": 1.2})
    assert new.threshold == 1.2
    assert cur.threshold == 0.9               # original untouched
    assert new.calibration_scale == cur.calibration_scale
    assert new.model_version == cur.model_version


def test_apply_valid_calibration_scale_change():
    cur = _cfg()
    new = apply_new_config(cur, {"calibration_scale": 0.5})
    assert new.calibration_scale == 0.5 and new.threshold == cur.threshold


def test_apply_both_fields_at_once():
    cur = _cfg()
    new = apply_new_config(cur, {"threshold": 1.1, "calibration_scale": 0.3})
    assert (new.threshold, new.calibration_scale) == (1.1, 0.3)


def test_unknown_field_rejected_with_artifact_examples():
    cur = _cfg()
    with pytest.raises(ConfigError) as exc:
        apply_new_config(cur, {"window_size": 64})
    msg = str(exc.value)
    assert "window_size" in msg
    for example in ARTIFACT_DEPENDENT_EXAMPLES:
        assert example in msg
    assert cur.threshold == 0.9  # rejected: nothing changed


def test_model_version_is_not_hot_swappable():
    with pytest.raises(ConfigError):
        apply_new_config(_cfg(), {"model_version": "some_other_model"})


def test_mixed_known_and_unknown_fields_rejected_atomically():
    cur = _cfg()
    with pytest.raises(ConfigError):
        apply_new_config(cur, {"threshold": 5.0, "window_size": 64})
    assert cur.threshold == 0.9  # the valid part was NOT partially applied


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_threshold_rejected(bad):
    with pytest.raises(ConfigError):
        apply_new_config(_cfg(), {"threshold": bad})


@pytest.mark.parametrize("bad", [0.0, -0.1, MIN_SCALE / 2, float("nan")])
def test_invalid_calibration_scale_rejected(bad):
    with pytest.raises(ConfigError):
        apply_new_config(_cfg(), {"calibration_scale": bad})


def test_config_version_changes_with_hot_swappable_fields_only():
    cur = _cfg()
    same = _cfg()
    assert cur.config_version == same.config_version
    assert apply_new_config(cur, {"threshold": 1.5}).config_version != cur.config_version
    assert apply_new_config(cur, {"calibration_scale": 0.4}).config_version != cur.config_version
    # changing model_version is impossible via apply_new_config, but constructing a
    # RuntimeConfig directly with a different model_version must NOT change config_version
    # (model identity is tracked separately from the hot-swappable fingerprint)
    other_model = RuntimeConfig(threshold=0.9, calibration_scale=0.17, model_version="different")
    assert other_model.config_version == cur.config_version


def test_calibrator_center_tracks_threshold_exactly():
    cfg = apply_new_config(_cfg(), {"threshold": 1.7})
    assert cfg.calibrator.transform_one(1.7) == pytest.approx(0.5, abs=1e-12)


def test_from_artifact_and_to_dict_round_trip():
    cfg = RuntimeConfig.from_artifact(0.88, {"method": "log_ratio_sigmoid", "center": 0.88,
                                             "scale": 0.17}, "conv_ae_v1:fp32_tflite")
    d = cfg.to_dict()
    assert d["threshold"] == 0.88 and d["calibration_scale"] == 0.17
    assert d["model_version"] == "conv_ae_v1:fp32_tflite" and "config_version" in d


def test_hot_swappable_fields_frozenset():
    assert HOT_SWAPPABLE_FIELDS == {"threshold", "calibration_scale"}
