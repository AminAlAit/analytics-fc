"""Unit tests for baseline trajectory regressors (Ridge and HistGradientBoosting)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import (
    FeatureContractError,
    HistGradientBoostingTrajectoryRegressor,
    RidgeTrajectoryRegressor,
    TrajectoryFeatureContract,
    TrajectoryPrediction,
)


def make_sample_dataframe() -> pd.DataFrame:
    """Construct small contract-compliant DataFrame for fast isolated tests."""
    return pd.DataFrame(
        {
            "player_id": [101, 102, 103, 104],
            "edition_year": [2019, 2020, 2020, 2021],
            "target_edition_year": [2020, 2021, 2021, 2022],
            "age": [19, 23, 27, 31],
            "overall": [72.0, 77.0, 83.0, 88.0],
            "potential": [82.0, 81.0, 84.0, 88.0],
            "potential_diff": [10.0, 4.0, 1.0, 0.0],
            "pace": [78.0, 81.0, 70.0, 65.0],
            "shooting": [65.0, 72.0, 80.0, 85.0],
            "passing": [68.0, 74.0, 84.0, 86.0],
            "dribbling": [74.0, 78.0, 82.0, 84.0],
            "defending": [40.0, 50.0, 60.0, 55.0],
            "physic": [62.0, 70.0, 75.0, 72.0],
            "growth_rate_1y": [3.0, 1.0, 0.0, -1.0],
            "target_growth": [4.0, 2.0, 0.0, -1.0],
        }
    )


def test_ridge_regressor_lifecycle_and_fit():
    """Verify RidgeTrajectoryRegressor fit and predict lifecycle."""
    model = RidgeTrajectoryRegressor(alpha=1.0, random_state=42)
    assert model.is_fitted is False
    assert model.model_version == "0.1.0-ridge"

    df = make_sample_dataframe()
    X = TrajectoryFeatureContract.extract_feature_matrix(df)
    y = TrajectoryFeatureContract.extract_target_vector(df)

    # Cannot predict prior to fitting
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(X)

    model.fit(X, y)
    assert model.is_fitted is True
    assert model.training_duration_seconds > 0.0

    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 4
    assert np.all(np.isfinite(preds))

    trajectories = model.predict_trajectory(df)
    assert len(trajectories) == 4
    for item in trajectories:
        assert isinstance(item, TrajectoryPrediction)
        assert 35.0 <= item.predicted_overall <= 99.0
        assert item.model_version == "0.1.0-ridge"


def test_hgb_regressor_lifecycle_and_fit():
    """Verify HistGradientBoostingTrajectoryRegressor fit and predict lifecycle."""
    model = HistGradientBoostingTrajectoryRegressor(max_iter=50, random_state=42)
    assert model.is_fitted is False
    assert model.model_version == "0.1.0-hgb"

    df = make_sample_dataframe()
    X = TrajectoryFeatureContract.extract_feature_matrix(df)
    y = TrajectoryFeatureContract.extract_target_vector(df)

    # Cannot predict prior to fitting
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(X)

    model.fit(X, y)
    assert model.is_fitted is True
    assert model.training_duration_seconds > 0.0

    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 4
    assert np.all(np.isfinite(preds))

    trajectories = model.predict_trajectory(df)
    assert len(trajectories) == 4
    for item in trajectories:
        assert isinstance(item, TrajectoryPrediction)
        assert 35.0 <= item.predicted_overall <= 99.0
        assert item.model_version == "0.1.0-hgb"


def test_trajectory_regressors_contract_rejection():
    """Verify regressors reject malformed or invalid input DataFrames."""
    model = RidgeTrajectoryRegressor()
    df_missing = make_sample_dataframe().drop(columns=["potential_diff"])

    with pytest.raises(FeatureContractError):
        model.fit(df_missing, pd.Series([1.0, 2.0, 3.0, 4.0]))

    # Valid fit first
    df_valid = make_sample_dataframe()
    X_valid = TrajectoryFeatureContract.extract_feature_matrix(df_valid)
    y_valid = TrajectoryFeatureContract.extract_target_vector(df_valid)
    model.fit(X_valid, y_valid)

    # Predict rejects missing columns
    with pytest.raises(FeatureContractError):
        model.predict(df_missing)

    with pytest.raises(FeatureContractError):
        model.predict_trajectory(df_missing)


def test_trajectory_prediction_bounds_clamping():
    """Verify predict_trajectory strictly clamps output ratings to [35.0, 99.0]."""
    model = RidgeTrajectoryRegressor()
    df = make_sample_dataframe()
    X = TrajectoryFeatureContract.extract_feature_matrix(df)
    y = TrajectoryFeatureContract.extract_target_vector(df)
    model.fit(X, y)

    # Create synthetic extreme cases
    extreme_df = df.copy()
    extreme_df.loc[0, "overall"] = 98.0
    extreme_df.loc[0, "potential"] = 99.0
    extreme_df.loc[0, "potential_diff"] = 1.0

    extreme_df.loc[1, "overall"] = 36.0
    extreme_df.loc[1, "potential"] = 40.0
    extreme_df.loc[1, "potential_diff"] = 4.0

    trajectories = model.predict_trajectory(extreme_df)
    for traj in trajectories:
        assert 35.0 <= traj.predicted_overall <= 99.0


def test_trajectory_regressors_training_benchmark():
    """Verify models train on the 58,146 observation dataset in under 30 seconds."""
    dataset_path = Path("data/processed/trajectory_training.parquet")
    if not dataset_path.exists():
        pytest.skip(f"Longitudinal dataset {dataset_path} not found.")

    df = pd.read_parquet(dataset_path)
    assert len(df) == 58146

    X = TrajectoryFeatureContract.extract_feature_matrix(df)
    y = TrajectoryFeatureContract.extract_target_vector(df)

    # Benchmark Ridge
    ridge = RidgeTrajectoryRegressor(alpha=1.0, random_state=42)
    t0 = time.perf_counter()
    ridge.fit(X, y)
    t_ridge = time.perf_counter() - t0
    assert t_ridge < 30.0, f"Ridge fit took {t_ridge:.2f}s, exceeding 30s hardware budget."

    # Benchmark HistGradientBoosting
    hgb = HistGradientBoostingTrajectoryRegressor(max_iter=100, random_state=42)
    t0 = time.perf_counter()
    hgb.fit(X, y)
    t_hgb = time.perf_counter() - t0
    assert t_hgb < 30.0, f"HistGradientBoosting fit took {t_hgb:.2f}s, exceeding 30s budget."

    # Verify predictions on a sample slice
    sample_df = df.head(10).copy()
    ridge_preds = ridge.predict_trajectory(sample_df)
    hgb_preds = hgb.predict_trajectory(sample_df)

    assert len(ridge_preds) == 10
    assert len(hgb_preds) == 10
    for r_pred, h_pred in zip(ridge_preds, hgb_preds):
        assert 35.0 <= r_pred.predicted_overall <= 99.0
        assert 35.0 <= h_pred.predicted_overall <= 99.0
