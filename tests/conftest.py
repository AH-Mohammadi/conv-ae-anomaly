import numpy as np
import pandas as pd
import pytest

from anomaly_detector.data import SENSOR_COLUMNS


def _write_file(path, seed, n_rows=600, anomaly_rows=(450, 500), label_start=None):
    """Synthetic SKAB-like file: noise, then a big shift on 2 sensors."""
    rng = np.random.default_rng(seed)
    c = len(SENSOR_COLUMNS)
    means = np.linspace(1, 8, c)
    scales = np.linspace(0.5, 3, c)
    values = means + scales * rng.standard_normal((n_rows, c))
    labels = np.zeros(n_rows, dtype=int)
    a, b = anomaly_rows
    values[a:b, 0] += 10 * scales[0]
    values[a:b, 2] += 10 * scales[2]
    labels[a:b] = 1
    if label_start is not None:  # inject a label into the reference region
        labels[label_start] = 1
    df = pd.DataFrame(values, columns=list(SENSOR_COLUMNS))
    df.insert(0, "datetime", pd.date_range("2020-01-01", periods=n_rows, freq="s"))
    df["anomaly"] = labels
    df["changepoint"] = 0
    df.to_csv(path, sep=";", index=False)


@pytest.fixture
def make_skab_file():
    return _write_file


@pytest.fixture
def synthetic_root(tmp_path):
    for sub, seeds in (("valve1", (1, 2)), ("other", (3,))):
        d = tmp_path / sub
        d.mkdir()
        for i, seed in enumerate(seeds):
            _write_file(d / f"{i}.csv", seed)
    return tmp_path


@pytest.fixture(scope="session")
def small_model_dir(tmp_path_factory):
    """Train a tiny Conv-AE on synthetic SKAB-like data and convert it to TFLite.
    Shared, read-only, across test modules that need a real, small, deployable
    artifact -- keeps the suite fast by training/converting only once."""
    root = tmp_path_factory.mktemp("skab_shared")
    for sub, seeds in (("valve1", (1, 2)), ("other", (3,))):
        d = root / sub
        d.mkdir()
        for i, seed in enumerate(seeds):
            _write_file(d / f"{i}.csv", seed)
    work = tmp_path_factory.mktemp("work_shared")

    from anomaly_detector.compress import run_compression, save_compression_outputs
    from anomaly_detector.config import (
        Config, CompressionConfig, DatasetConfig, DetectorConfig, ModelConfig, OutputConfig,
        TrainingConfig,
    )
    from anomaly_detector.train import run_conv_ae

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
    return {"cfg": cfg, "model_dir": work / "m_v1", "root": root}
