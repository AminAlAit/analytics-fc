"""Unit and integration tests for multi-edition attribute drift analyzer pipeline."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.monitoring.config import DriftMonitoringConfig
from src.monitoring.drift_analyzer import (
    calculate_feature_drift,
    calculate_psi,
    export_drift_metrics_json,
    generate_evidently_html_report,
    load_edition_data,
    normalize_edition_year,
    run_attribute_drift_pipeline,
    run_drift_monitoring_suite,
)


def test_normalize_edition_year():
    """Verify 2-digit and 4-digit edition year normalization."""
    assert normalize_edition_year(21) == 2021
    assert normalize_edition_year(22) == 2022
    assert normalize_edition_year(2021) == 2021
    assert normalize_edition_year(2015) == 2015


def test_calculate_psi_identical_distribution():
    """Verify PSI approaches zero for identical distributions."""
    np.random.seed(42)
    sample_a = pd.Series(np.random.normal(70, 5, 2000))
    sample_b = pd.Series(np.random.normal(70, 5, 2000))

    psi = calculate_psi(sample_a, sample_b)
    assert psi < 0.05, f"Expected near-zero PSI for identical data, got {psi}"


def test_calculate_psi_shifted_distribution():
    """Verify PSI flags substantial distribution shifts."""
    np.random.seed(42)
    sample_ref = pd.Series(np.random.normal(60, 5, 2000))
    sample_drifted = pd.Series(np.random.normal(85, 5, 2000))

    psi = calculate_psi(sample_ref, sample_drifted)
    assert psi > 0.25, f"Expected high PSI for shifted data, got {psi}"


def test_calculate_feature_drift_metrics():
    """Verify feature drift metrics calculation returns expected structure and statistics."""
    np.random.seed(42)
    ref = pd.Series(np.random.normal(70, 10, 1000))
    curr = pd.Series(np.random.normal(75, 10, 1000))

    metrics = calculate_feature_drift(ref, curr, ks_threshold=0.05, psi_threshold=0.10)

    assert "wasserstein_distance" in metrics
    assert "ks_statistic" in metrics
    assert "ks_p_value" in metrics
    assert "psi" in metrics
    assert "drift_detected" in metrics
    assert metrics["wasserstein_distance"] > 3.0
    assert metrics["drift_detected"] is True
    assert metrics["reference_count"] == 1000
    assert metrics["current_count"] == 1000


def test_calculate_feature_drift_empty_series():
    """Verify robust handling of empty series with default zero values."""
    empty = pd.Series([], dtype=float)
    metrics = calculate_feature_drift(empty, empty)
    assert metrics["wasserstein_distance"] == 0.0
    assert metrics["ks_p_value"] == 1.0
    assert metrics["drift_detected"] is False


def test_load_edition_data_sampling_cap():
    """Verify load_edition_data respects sample size cap and extracts monitored columns."""
    sample_cap = 150
    df = load_edition_data(
        edition_year=21,
        sample_size=sample_cap,
        columns=["player_id", "positions", "overall", "potential", "pace"],
    )

    assert len(df) == sample_cap
    assert "player_id" in df.columns
    assert "overall" in df.columns
    assert "potential" in df.columns
    assert "pace" in df.columns


def test_run_attribute_drift_pipeline_execution():
    """Verify end-to-end drift pipeline execution produces complete summary and feature map."""
    cfg = DriftMonitoringConfig(
        reference_edition=21,
        target_edition=22,
        monitored_features=["overall", "potential", "value_eur", "wage_eur", "age"],
    )

    summary = run_attribute_drift_pipeline(config=cfg, sample_size=300)

    assert summary["reference_edition"] == 2021
    assert summary["target_edition"] == 2022
    assert summary["reference_sample_size"] == 300
    assert summary["target_sample_size"] == 300
    assert summary["total_features_evaluated"] == 5
    assert 0 <= summary["drifted_features_count"] <= 5
    assert 0.0 <= summary["drifted_features_share"] <= 1.0
    assert isinstance(summary["dataset_drift_detected"], bool)

    for feature in cfg.monitored_features:
        assert feature in summary["features"]
        f_metric = summary["features"][feature]
        assert "wasserstein_distance" in f_metric
        assert "ks_statistic" in f_metric
        assert "psi" in f_metric


def test_export_drift_metrics_json(tmp_path):
    """Verify export_drift_metrics_json writes deterministic, readable JSON output."""
    metrics = {
        "reference_edition": 2021,
        "target_edition": 2022,
        "drifted_features_count": 2,
        "features": {
            "overall": {"ks_p_value": 0.01, "drift_detected": True},
            "age": {"ks_p_value": 0.85, "drift_detected": False},
        },
    }

    out_file = tmp_path / "monitoring" / "drift_metrics.json"
    exported_path = export_drift_metrics_json(metrics, output_path=out_file)

    assert exported_path.exists()
    assert exported_path == out_file

    with open(exported_path, encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["reference_edition"] == 2021
    assert loaded["drifted_features_count"] == 2
    assert loaded["features"]["overall"]["drift_detected"] is True


def test_generate_evidently_html_report(tmp_path):
    """Verify generate_evidently_html_report renders a valid standalone HTML document."""
    df_ref = pd.DataFrame(
        {
            "overall": [70, 72, 75, 80, 85],
            "age": [20, 22, 25, 28, 31],
        }
    )
    df_curr = pd.DataFrame(
        {
            "overall": [71, 73, 76, 81, 86],
            "age": [21, 23, 26, 29, 32],
        }
    )

    out_html = tmp_path / "reports" / "test_report.html"
    result_path = generate_evidently_html_report(
        df_reference=df_ref,
        df_current=df_curr,
        output_path=out_html,
        columns=["overall", "age"],
    )

    assert result_path.exists()
    assert result_path.stat().st_size > 1000
    html_text = result_path.read_text(encoding="utf-8")
    assert "<html" in html_text.lower()


def test_run_drift_monitoring_suite_exports(tmp_path):
    """Verify run_drift_monitoring_suite creates both JSON metrics and HTML report."""
    json_path = tmp_path / "data" / "drift_metrics.json"
    reports_dir = tmp_path / "reports"

    cfg = DriftMonitoringConfig(
        reference_edition=21,
        target_edition=22,
        monitored_features=["overall", "potential", "age"],
        metrics_output_path=json_path,
        report_output_dir=reports_dir,
    )

    summary = run_drift_monitoring_suite(
        config=cfg,
        sample_size=150,
        export_json=True,
        export_html=True,
    )

    assert "metrics_json_path" in summary
    assert "html_report_path" in summary
    assert json_path.exists()
    assert Path(summary["html_report_path"]).exists()

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["reference_edition"] == 2021
    assert data["total_features_evaluated"] == 3
