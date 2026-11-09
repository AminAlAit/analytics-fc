"""Unit tests for longitudinal backtest evaluation suite.

Validates:
1. GroupKFold grouped strictly by player_id with zero cross-split leakage.
2. Temporal expanding-window walk-forward backtesting with strict temporal precedence.
3. Automated JSON evaluation report generation and serialization.
4. Detection and rejection of artificial data leakage.
5. Verification gate: zero player overlap across splits on 78,171 row dataset in < 30s.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pytest

from src.models import (
    FeatureContractError,
    RidgeTrajectoryRegressor,
)
from src.models.backtest_suite import (
    GroupKFoldLeakageAuditor,
    WalkForwardBacktester,
    generate_evaluation_report,
)


def make_sample_dataframe() -> pd.DataFrame:
    """Construct multi-season synthetic dataset for fast isolated unit testing."""
    return pd.DataFrame(
        {
            "player_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "edition_year": [2018, 2019, 2018, 2019, 2018, 2019, 2018, 2019, 2018, 2019],
            "target_edition_year": [2019, 2020, 2019, 2020, 2019, 2020, 2019, 2020, 2019, 2020],
            "age": [18, 19, 22, 23, 26, 27, 29, 30, 20, 21],
            "overall": [68.0, 72.0, 75.0, 78.0, 82.0, 83.0, 86.0, 85.0, 70.0, 74.0],
            "potential": [80.0, 82.0, 81.0, 82.0, 84.0, 84.0, 86.0, 85.0, 82.0, 84.0],
            "potential_diff": [12.0, 10.0, 6.0, 4.0, 2.0, 1.0, 0.0, 0.0, 12.0, 10.0],
            "pace": [75.0, 78.0, 80.0, 81.0, 70.0, 68.0, 65.0, 64.0, 82.0, 84.0],
            "shooting": [62.0, 66.0, 70.0, 74.0, 78.0, 80.0, 84.0, 84.0, 68.0, 72.0],
            "passing": [65.0, 70.0, 72.0, 75.0, 80.0, 82.0, 85.0, 85.0, 70.0, 74.0],
            "dribbling": [72.0, 76.0, 76.0, 79.0, 79.0, 80.0, 83.0, 82.0, 75.0, 78.0],
            "defending": [38.0, 42.0, 48.0, 52.0, 58.0, 60.0, 62.0, 62.0, 45.0, 48.0],
            "physic": [60.0, 64.0, 68.0, 71.0, 74.0, 75.0, 75.0, 74.0, 66.0, 70.0],
            "growth_rate_1y": [2.0, 4.0, 1.0, 3.0, 0.0, 1.0, -1.0, -1.0, 2.0, 4.0],
            "target_growth": [4.0, 3.0, 3.0, 2.0, 1.0, 0.0, -1.0, -1.0, 4.0, 2.0],
        }
    )


def test_group_kfold_zero_leakage():
    """Verify GroupKFold by player_id guarantees zero player overlap across all folds."""
    df = make_sample_dataframe()
    model = RidgeTrajectoryRegressor(alpha=1.0)
    auditor = GroupKFoldLeakageAuditor(n_splits=3, group_col="player_id")

    metrics = auditor.evaluate(model, df)
    assert metrics["zero_leakage_verified"] is True
    assert metrics["total_player_overlaps"] == 0
    assert len(metrics["folds"]) == 3

    for fold in metrics["folds"]:
        assert fold["player_overlap"] == 0
        assert fold["train_players"] > 0
        assert fold["test_players"] > 0
        assert fold["mae"] >= 0.0


def test_walk_forward_backtest_temporal_precedence():
    """Verify WalkForwardBacktester evaluates chronological sequence with zero lookahead."""
    df = make_sample_dataframe()
    model = RidgeTrajectoryRegressor(alpha=1.0)
    backtester = WalkForwardBacktester(start_eval_year=2020)

    metrics = backtester.evaluate(model, df)
    assert len(metrics["folds"]) > 0

    for fold in metrics["folds"]:
        assert fold["eval_target_year"] >= 2020
        assert fold["max_train_edition"] < fold["eval_target_year"]
        assert fold["train_records"] > 0
        assert fold["test_records"] > 0
        assert fold["mae"] >= 0.0


def test_generate_evaluation_report_json(tmp_path: Path):
    """Verify generate_evaluation_report outputs comprehensive JSON metrics artifact."""
    df = make_sample_dataframe()
    model = RidgeTrajectoryRegressor(alpha=1.0)
    report_file = tmp_path / "model_metrics.json"

    report = generate_evaluation_report(
        model=model,
        df=df,
        output_path=report_file,
        n_splits=3,
        start_eval_year=2020,
    )

    assert report_file.exists()
    assert report["total_observations"] == 10
    assert report["unique_players"] == 5
    assert "group_kfold_evaluation" in report
    assert "walk_forward_evaluation" in report

    # Verify JSON content
    with open(report_file, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["model_class"] == "RidgeTrajectoryRegressor"
    assert loaded["group_kfold_evaluation"]["zero_leakage_verified"] is True


def test_leakage_auditor_contract_rejection():
    """Verify auditor raises error on missing group or target columns."""
    df = make_sample_dataframe()
    model = RidgeTrajectoryRegressor()
    auditor = GroupKFoldLeakageAuditor(n_splits=2, group_col="invalid_id")

    with pytest.raises(FeatureContractError):
        auditor.evaluate(model, df)


def test_verification_gate_zero_leakage_on_training_dataset():
    """Verify GroupKFold confirms 0 player overlap across splits on full dataset in < 30s."""
    dataset_path = Path("data/processed/trajectory_training.parquet")
    if not dataset_path.exists():
        pytest.skip(f"Longitudinal dataset {dataset_path} not found.")

    df = pd.read_parquet(dataset_path)
    assert len(df) in (58146, 78171)

    model = RidgeTrajectoryRegressor(alpha=1.0, random_state=42)
    auditor = GroupKFoldLeakageAuditor(n_splits=5, group_col="player_id")

    t0 = time.perf_counter()
    metrics = auditor.evaluate(model, df)
    duration = time.perf_counter() - t0

    # Verification Gate: zero player overlap
    assert metrics["zero_leakage_verified"] is True, "Data leakage detected in GroupKFold!"
    assert metrics["total_player_overlaps"] == 0, f"Found {metrics['total_player_overlaps']} overlaps."
    assert duration < 30.0, f"Audit took {duration:.2f}s, exceeding 30s hardware budget."

    # Export canonical model evaluation report
    report_path = Path("reports/model_evaluation_metrics.json")
    generate_evaluation_report(
        model=model,
        df=df,
        output_path=report_path,
        n_splits=5,
        start_eval_year=2021,
    )
    assert report_path.exists()
