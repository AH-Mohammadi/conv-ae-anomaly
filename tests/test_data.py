from pathlib import Path

import pandas as pd
import pytest

from anomaly_detector.data import (
    SENSOR_COLUMNS, check_reference_segment, discover_files, file_key, load_series,
)

REAL_ROOT = Path("data/raw/SKAB/data")


def test_load_synthetic(synthetic_root):
    files = discover_files(synthetic_root)
    assert [f.parent.name for f in files] == ["valve1", "valve1", "other"]
    s = load_series(files[0])
    assert s.values.shape == (600, len(SENSOR_COLUMNS))
    assert set(s.labels.tolist()) <= {0, 1}
    assert s.name == "valve1/0.csv"


def test_missing_column_raises(tmp_path, make_skab_file):
    p = tmp_path / "0.csv"
    make_skab_file(p, 0)
    df = pd.read_csv(p, sep=";").drop(columns=["Current"])
    df.to_csv(p, sep=";", index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_series(p)


def test_bad_labels_raise(tmp_path, make_skab_file):
    p = tmp_path / "0.csv"
    make_skab_file(p, 0)
    df = pd.read_csv(p, sep=";")
    df.loc[3, "anomaly"] = 2
    df.to_csv(p, sep=";", index=False)
    with pytest.raises(ValueError, match="0/1"):
        load_series(p)


def test_empty_file_raises(tmp_path):
    p = tmp_path / "0.csv"
    pd.DataFrame(columns=["datetime", *SENSOR_COLUMNS, "anomaly"]).to_csv(
        p, sep=";", index=False
    )
    with pytest.raises(ValueError, match="empty"):
        load_series(p)


def test_no_files_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_files(tmp_path)


def test_contaminated_reference_raises(tmp_path, make_skab_file):
    p = tmp_path / "0.csv"
    make_skab_file(p, 0, label_start=10)
    with pytest.raises(ValueError, match="reference segment"):
        check_reference_segment(load_series(p), 400)


@pytest.mark.skipif(not REAL_ROOT.exists(), reason="SKAB not downloaded")
def test_real_skab():
    files = discover_files(REAL_ROOT)
    assert len(files) == 34  # valve1: 16, valve2: 4, other: 14 (numbered 1-14)
    for f in files:
        s = load_series(f)
        assert s.labels.sum() > 0  # every labeled file contains anomalies
        if file_key(f) == "other/2.csv":  # known exception: first anomaly at row 104
            with pytest.raises(ValueError, match="reference segment"):
                check_reference_segment(s, 400)
        else:
            check_reference_segment(s, 400)
