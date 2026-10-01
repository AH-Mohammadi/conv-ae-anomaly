import json

import pytest

from anomaly_detector.config import Config, DatasetConfig
from anomaly_detector.evaluate import run_evaluation, write_scorecard


def _cfg(root):
    return Config(dataset=DatasetConfig(root=str(root), reference_rows=400))


def test_end_to_end_detects_injected_anomaly(synthetic_root):
    card = run_evaluation(_cfg(synthetic_root))
    assert card["n_files"] == 3
    assert card["f1"] > 0.7
    # Windows ending shortly after the anomaly still contain anomalous samples but
    # carry a 0 label (window-end assignment), so some "false alarms" are inherent
    # to the protocol. Expected FAR here is ~0.16; bound leaves margin.
    assert card["far"] < 0.25
    assert card["mar"] < 0.3


def test_deterministic(synthetic_root):
    a = run_evaluation(_cfg(synthetic_root))
    b = run_evaluation(_cfg(synthetic_root))
    assert (a["f1"], a["far"], a["mar"]) == (b["f1"], b["far"], b["mar"])
    assert a["per_file"] == b["per_file"]


def test_scorecard_is_json_with_stable_fields(synthetic_root, tmp_path):
    card = run_evaluation(_cfg(synthetic_root))
    path = write_scorecard(card, tmp_path / "out")
    loaded = json.loads(path.read_text())
    for key in ("schema_version", "dataset", "model", "f1", "far", "mar"):
        assert key in loaded
    assert loaded["dataset"] == "SKAB"


def test_contaminated_reference_aborts(tmp_path, make_skab_file):
    (tmp_path / "valve1").mkdir()
    make_skab_file(tmp_path / "valve1" / "0.csv", 0, label_start=5)
    with pytest.raises(ValueError, match="reference segment"):
        run_evaluation(_cfg(tmp_path))


def test_exclude_files(synthetic_root):
    cfg = Config(dataset=DatasetConfig(
        root=str(synthetic_root), reference_rows=400, exclude_files=["other/0.csv"]))
    card = run_evaluation(cfg)
    assert card["n_files"] == 2
    assert card["protocol"]["excluded_files"] == ["other/0.csv"]
    assert all(r["file"] != "other/0.csv" for r in card["per_file"])


def test_scorecard_has_auc_and_segment_diagnostics(synthetic_root):
    card = run_evaluation(_cfg(synthetic_root))
    assert 0.5 < card["roc_auc"] <= 1.0
    seg = card["diagnostics"]["far_by_segment"]
    # synthetic anomaly is rows 450-500 of 600, so both segments exist
    assert seg["pre_anomaly"]["normal_points"] > 0
    assert seg["post_anomaly"]["normal_points"] > 0
