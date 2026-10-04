import hashlib
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("keras")
pytest.importorskip("tensorflow")

from conftest import _write_file

from anomaly_detector.compress import run_compression, save_compression_outputs
from anomaly_detector.config import (
    CompressionConfig, Config, DatasetConfig, DetectorConfig, ModelConfig, OutputConfig,
    TrainingConfig,
)
from anomaly_detector.inference import TFLiteAutoencoder
from anomaly_detector.pibench import (
    Tolerances, compare_to_reference, decode_throttled, latency_stats, read_hardware,
    read_thermal, run_benchmark,
)
from anomaly_detector.pibundle import export_bundle
from anomaly_detector.train import run_conv_ae


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("skab")
    for sub, seeds in (("valve1", (1, 2)), ("other", (3,))):
        d = root / sub
        d.mkdir()
        for i, seed in enumerate(seeds):
            _write_file(d / f"{i}.csv", seed)
    work = tmp_path_factory.mktemp("work")
    cfg = Config(
        dataset=DatasetConfig(root=str(root), reference_rows=400, train_rows=300),
        model=ModelConfig(latent_dim=8, filter_count=8, kernel_size=5),
        training=TrainingConfig(epochs=15, batch_size=64, patience=5),
        compression=CompressionConfig(calibration_samples=100),
        detector=DetectorConfig(name="conv_ae", threshold=None),
        output=OutputConfig(metrics_dir=str(work / "reports"), model_dir=str(work / "m_v1")),
    )
    _, model, _ = run_conv_ae(cfg)
    report, blobs, cards = run_compression(cfg, model, model_dir=work / "m_v1")
    save_compression_outputs(report, blobs, cards, cfg, work / "m_v1")
    return export_bundle(cfg, work / "m_v1", work / "bundle")


# ---- pure helpers ------------------------------------------------------------------
def test_decode_throttled():
    assert not any(decode_throttled(0).values())
    f = decode_throttled(0x50005)
    assert f["under_voltage_now"] and f["throttled_now"] and f["throttled_occurred"]
    assert not f["freq_capped_now"]


def test_latency_stats():
    s = latency_stats(np.arange(1, 101) * 1_000_000)  # 1..100 ms
    assert s["n"] == 100 and s["median_ms"] == pytest.approx(50.5) and s["max_ms"] == 100.0
    with pytest.raises(ValueError):
        latency_stats(np.array([]))


def test_compare_to_reference():
    rng = np.random.default_rng(0)
    ref = rng.uniform(0.1, 2.0, 2000)
    y = (ref > 1.0).astype(np.int8)
    tol = Tolerances()
    ok = compare_to_reference(ref * (1 + 1e-7), ref, y, 1.0, tol, is_fp32=True)
    assert ok["passed"] and ok["decision_agreement"] == 1.0
    # large relative error, same decisions: only the FP32 numerical check fails
    bad = compare_to_reference(ref * 1.001, ref, y, 100.0, tol, is_fp32=True)
    assert not bad["passed"] and not bad["checks"]["fp32_relative_score_diff"]
    assert compare_to_reference(ref * 1.001, ref, y, 100.0, tol, is_fp32=False)["passed"]
    # flipped decisions fail agreement
    flipped = ref.copy()
    flipped[:50] = 5.0 - flipped[:50] * 0  # push 50 windows above threshold
    r = compare_to_reference(flipped, ref, y, 1.0, tol, is_fp32=False)
    assert not r["checks"]["decision_agreement"]
    with pytest.raises(ValueError):
        compare_to_reference(ref[:5], ref, y, 1.0, tol, True)


def test_probes_never_crash():
    hw = read_hardware()
    assert {"device_model", "is_raspberry_pi", "machine", "python", "numpy"} <= set(hw)
    assert isinstance(hw["is_raspberry_pi"], bool)
    assert set(read_thermal()) == {"temp_c", "freq_mhz", "throttled", "throttle_flags"}


# ---- bundle ------------------------------------------------------------------------
def test_bundle_contents(bundle):
    man = json.loads((bundle / "manifest.json").read_text())
    n, w, c = man["n_windows"], man["window_size"], man["n_channels"]
    assert (bundle / "windows_f32.bin").stat().st_size == n * w * c * 4
    assert (bundle / "labels_i8.bin").stat().st_size == n
    assert set(man["variants"]) == {"fp32_tflite", "int8_tflite", "int8_wide_tflite"}
    for v in man["variants"].values():
        assert hashlib.sha256((bundle / v["file"]).read_bytes()).hexdigest() == v["sha256"]
        assert (bundle / v["ref_scores"]).stat().st_size == n * 8
        assert v["threshold"] > 0
    assert man["tolerances"]["min_decision_agreement"] == 0.999
    assert (bundle / "benchmark_pi.py").exists()


def test_bundle_is_numpy_only(bundle):
    """The bundle's modules must import without pandas/yaml/tensorflow (Pi has none)."""
    code = ("import sys; sys.modules['pandas']=None; sys.modules['yaml']=None; "
            "sys.modules['tensorflow']=None; sys.modules['keras']=None; "
            "import anomaly_detector.pibench as p; print(p.__file__)")
    out = subprocess.run([sys.executable, "-c", code], cwd=bundle, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert str(bundle) in out.stdout


def test_xnnpack_toggle_same_outputs(bundle):
    x = np.random.default_rng(0).standard_normal((10, 32, 8)).astype("float32")
    a = TFLiteAutoencoder(model_path=bundle / "detector_fp32.tflite")
    try:
        b = TFLiteAutoencoder(model_path=bundle / "detector_fp32.tflite", xnnpack=False)
    except RuntimeError:
        pytest.skip("backend cannot disable the default delegate")
    assert np.allclose(a.reconstruct(x), b.reconstruct(x), atol=1e-4)


def test_timed_errors_match_untimed(bundle):
    x = np.random.default_rng(1).standard_normal((15, 32, 8)).astype("float32")
    det = TFLiteAutoencoder(model_path=bundle / "detector_int8_wide.tflite")
    e, inv, tot = det.timed_errors(x)
    assert np.array_equal(e, det.reconstruction_errors(x))
    assert (inv > 0).all() and (tot >= inv).all()


# ---- benchmark driver ------------------------------------------------------------------
def test_run_benchmark_end_to_end(bundle):
    rep = run_benchmark(bundle, variants=["fp32_tflite", "int8_wide_tflite"],
                        sustained_seconds=2, bucket_seconds=1, cooldown_seconds=0)
    man = json.loads((bundle / "manifest.json").read_text())
    assert rep["schema_version"] == "pi_benchmark_report_v1"
    assert rep["is_raspberry_pi"] == read_hardware()["is_raspberry_pi"]
    assert [r["variant"] for r in rep["results"]] == ["fp32_tflite", "int8_wide_tflite"]
    for r in rep["results"]:
        assert "error" not in r
        assert r["latency"]["invoke"]["n"] == man["n_windows"]
        assert r["latency"]["end_to_end"]["median_ms"] >= r["latency"]["invoke"]["median_ms"]
        assert r["desktop_comparison"]["passed"]  # same machine, same code -> must agree
        ref = man["variants"][r["variant"]]["desktop_reference_metrics"]
        assert r["accuracy"]["f1"] == pytest.approx(ref["f1"], abs=1e-9)
        assert len(r["sustained"]["buckets"]) >= 1
        assert {"temp_max_c", "throttled_end"} <= set(r["thermal"])
        assert r["memory_mb"]["peak_rss_final"] is None or r["memory_mb"]["peak_rss_final"] > 0
    assert "speedup_vs_fp32_median_invoke" in rep["results"][1]


def test_benchmark_rejects_corrupted_or_unknown(bundle, tmp_path):
    bad = tmp_path / "bad"
    shutil.copytree(bundle, bad)
    f = bad / "detector_fp32.tflite"
    f.write_bytes(f.read_bytes()[:-1] + b"\x00" if f.read_bytes()[-1] != 0 else f.read_bytes()[:-1] + b"\x01")
    with pytest.raises(ValueError, match="sha256"):
        run_benchmark(bad, sustained_seconds=0, cooldown_seconds=0)
    with pytest.raises(ValueError, match="not in bundle"):
        run_benchmark(bundle, variants=["nope"], sustained_seconds=0)
