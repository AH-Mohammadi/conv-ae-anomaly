"""Integration test for scripts/demo_runtime_reconfig.py and set_runtime_config.py:
runs the actual demo flow end-to-end against a real (small) trained+converted model."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("keras")
pytest.importorskip("tensorflow")

SCRIPTS = Path(__file__).parents[1] / "scripts"


def test_set_runtime_config_writes_atomic_json(tmp_path):
    out = tmp_path / "cmd.json"
    r = subprocess.run([sys.executable, str(SCRIPTS / "set_runtime_config.py"),
                        "--threshold", "1.5", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(out.read_text()) == {"threshold": 1.5}
    assert not out.with_suffix(".tmp").exists()  # temp file cleaned up by the rename


def test_set_runtime_config_requires_at_least_one_field(tmp_path):
    r = subprocess.run([sys.executable, str(SCRIPTS / "set_runtime_config.py"),
                        "--out", str(tmp_path / "cmd.json")], capture_output=True, text=True)
    assert r.returncode != 0


def test_demo_runtime_reconfig_end_to_end(small_model_dir, tmp_path):
    cfg, model_dir, root = small_model_dir["cfg"], small_model_dir["model_dir"], small_model_dir["root"]
    cfg_path = tmp_path / "conv_ae.yaml"
    import yaml

    raw = cfg.to_dict()
    raw["dataset"]["root"] = str(root)
    raw["output"] = {"metrics_dir": str(tmp_path / "reports"), "model_dir": str(model_dir)}
    cfg_path.write_text(yaml.safe_dump(raw))

    out = tmp_path / "demo.json"
    cmd_file = tmp_path / "command.json"
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "demo_runtime_reconfig.py"),
         "--config", str(cfg_path), "--variant", "fp32_tflite", "--file", "valve1/0.csv",
         "--new-threshold", "5.0", "--command-file", str(cmd_file), "--out", str(out)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert "Detector detected and applied the change" in r.stdout

    summary = json.loads(out.read_text())
    assert summary["actual_records"] == summary["expected_records"]
    assert summary["missed_windows"] == 0
    assert summary["config_applied"] is True
    assert summary["ring_buffer_identity_unchanged"] is True
    assert summary["window_id_contiguous_across_change"] is True
    assert summary["before_change"]["threshold"] != summary["after_change"]["threshold"]
    assert summary["after_change"]["threshold"] == 5.0
    assert summary["before_change"]["config_version"] != summary["after_change"]["config_version"]
    assert summary["before_change"]["n_records"] > 0 and summary["after_change"]["n_records"] > 0
    assert json.loads(cmd_file.read_text()) == {"threshold": 5.0}
