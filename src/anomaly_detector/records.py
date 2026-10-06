"""Versioned structured output record: the interface a future Coordinator would
consume as the detector's "Knowledge" payload (MAPE-K). The Coordinator itself is
out of scope for this project; this module only defines and fills the schema.

Field meaning:
  model_version  identifies the deployed ARTIFACT: trained weights + which TFLite
                 variant (quantization changes the op set/weights, so it is
                 artifact-dependent, same classification as window_size/filters).
  config_version short fingerprint of the currently ACTIVE runtime-adjustable
                 parameters (today: just the threshold). It changes when a
                 hot-swappable parameter changes, WITHOUT a new model_version --
                 this is the hook Iteration 7's runtime reconfiguration uses to
                 mark a record as having been produced under a new configuration.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone

SCHEMA_VERSION = "stream_output_record_v1"


def config_fingerprint(active_params: dict) -> str:
    """Short, deterministic fingerprint of the currently active runtime parameters."""
    payload = json.dumps(active_params, sort_keys=True, default=float)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


@dataclass(frozen=True)
class OutputRecord:
    schema_version: str
    timestamp: str
    window_id: int
    score: float | None         # calibrated, in [0, 1]; None if no calibrator was configured
    raw_error: float            # raw reconstruction error (unbounded, model/variant-specific scale)
    threshold: float
    is_anomaly: bool
    model_version: str
    config_version: str
    processing_ns: int
    stage: str
    file: str | None = None     # source episode/file, optional metadata (not part of the core schema)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version, "timestamp": self.timestamp,
            "window_id": self.window_id, "score": self.score, "raw_error": self.raw_error,
            "threshold": self.threshold, "is_anomaly": self.is_anomaly,
            "model_version": self.model_version, "config_version": self.config_version,
            "processing_ns": self.processing_ns, "stage": self.stage, "file": self.file,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def build_output_record(rec, *, model_version: str, config_version: str,
                        file: str | None = None) -> OutputRecord:
    """rec: a stream.StreamRecord (raw score, calibrated_score, threshold, etc.)."""
    return OutputRecord(
        schema_version=SCHEMA_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        window_id=rec.sample_index, score=rec.calibrated_score, raw_error=rec.score,
        threshold=rec.threshold, is_anomaly=rec.is_anomaly, model_version=model_version,
        config_version=config_version, processing_ns=rec.processing_ns, stage=rec.stage, file=file,
    )
