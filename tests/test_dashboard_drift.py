"""Unit tests for dashboard live drift metrics loading and telemetry indicators."""

import json
from pathlib import Path

import pandas as pd

from dashboard.app import DRIFT_METRICS_PATH, load_drift_metrics


def test_load_drift_metrics_existing_file():
    """Verify load_drift_metrics successfully loads and parses JSON snapshot."""
    data = load_drift_metrics(DRIFT_METRICS_PATH)
    assert data is not None
    assert "reference_edition" in data
    assert "target_edition" in data
    assert "dataset_drift_detected" in data
    assert "features" in data
    assert isinstance(data["features"], dict)
    assert len(data["features"]) > 0


def test_load_drift_metrics_missing_file(tmp_path: Path):
    """Verify load_drift_metrics returns None gracefully for nonexistent path."""
    nonexistent = tmp_path / "nonexistent_metrics.json"
    result = load_drift_metrics(nonexistent)
    assert result is None


def test_load_drift_metrics_corrupt_file(tmp_path: Path):
    """Verify load_drift_metrics returns None gracefully for malformed JSON."""
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{invalid json content:", encoding="utf-8")
    result = load_drift_metrics(bad_file)
    assert result is None


def test_drift_metrics_table_structure(tmp_path: Path):
    """Verify drift metrics dictionary converts into expected display table format."""
    sample_data = {
        "reference_edition": 2021,
        "target_edition": 2022,
        "reference_sample_size": 10000,
        "target_sample_size": 10000,
        "total_features_evaluated": 2,
        "drifted_features_count": 1,
        "drifted_features_share": 0.5,
        "dataset_drift_detected": True,
        "features": {
            "overall": {
                "drift_detected": True,
                "ks_p_value": 0.0012,
                "ks_statistic": 0.045,
                "wasserstein_distance": 0.85,
                "psi": 0.12,
                "reference_mean": 70.5,
                "current_mean": 71.3,
            },
            "age": {
                "drift_detected": False,
                "ks_p_value": 0.85,
                "ks_statistic": 0.005,
                "wasserstein_distance": 0.05,
                "psi": 0.002,
                "reference_mean": 25.1,
                "current_mean": 25.2,
            },
        },
    }

    test_file = tmp_path / "sample_drift.json"
    test_file.write_text(json.dumps(sample_data), encoding="utf-8")

    loaded = load_drift_metrics(test_file)
    assert loaded is not None

    rows = []
    for feat_name, m in loaded["features"].items():
        rows.append(
            {
                "Attribute": feat_name,
                "Status": "DRIFTED" if m["drift_detected"] else "STABLE",
                "KS p-value": f"{m['ks_p_value']:.4f}",
                "PSI": f"{m['psi']:.4f}",
                "Mean Shift": f"{round(m['current_mean'] - m['reference_mean'], 2):+.2f}",
            }
        )

    df = pd.DataFrame(rows)
    assert len(df) == 2
    assert "Attribute" in df.columns
    assert "Status" in df.columns
    assert df.loc[df["Attribute"] == "overall", "Status"].values[0] == "DRIFTED"
    assert df.loc[df["Attribute"] == "age", "Status"].values[0] == "STABLE"
