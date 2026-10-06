import json

import pytest

from anomaly_detector.records import SCHEMA_VERSION, build_output_record
from anomaly_detector.stream import StreamRecord


def _rec(**kw):
    defaults = dict(sample_index=42, score=1.23, threshold=0.9, is_anomaly=True,
                    processing_ns=5000, stage="scoring", calibrated_score=0.97)
    defaults.update(kw)
    return StreamRecord(**defaults)


def test_schema_fields_and_types():
    out = build_output_record(_rec(), model_version="conv_ae_v1:fp32_tflite",
                              config_version="abcd1234", file="valve1/0.csv")
    d = out.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION == "stream_output_record_v1"
    assert d["window_id"] == 42 and d["raw_error"] == 1.23 and d["score"] == 0.97
    assert d["threshold"] == 0.9 and d["is_anomaly"] is True
    assert d["model_version"] == "conv_ae_v1:fp32_tflite" and d["config_version"] == "abcd1234"
    assert d["file"] == "valve1/0.csv" and d["stage"] == "scoring"
    assert isinstance(d["processing_ns"], int)


def test_json_roundtrip_and_timestamp_format():
    out = build_output_record(_rec(), model_version="m", config_version="c")
    parsed = json.loads(out.to_json())
    assert parsed["window_id"] == 42
    from datetime import datetime
    datetime.fromisoformat(parsed["timestamp"])  # must parse as ISO-8601


def test_score_is_none_without_calibration():
    out = build_output_record(_rec(calibrated_score=None), model_version="m", config_version="c")
    assert out.to_dict()["score"] is None
    assert out.to_dict()["raw_error"] == 1.23  # raw error always present regardless


def test_file_defaults_to_none():
    out = build_output_record(_rec(), model_version="m", config_version="c")
    assert out.to_dict()["file"] is None
