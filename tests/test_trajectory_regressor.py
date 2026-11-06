"""Unit tests for baseline and quantile trajectory regressors.

Validates:
1. Ridge, HistGradientBoosting, and QuantileTrajectoryRegressor lifecycles.
2. Quantile prediction monotonicity (q10 <= median <= q90) and intervals.
3. Clamping to physical rating limits [35.0, 99.0].
4. Joblib artifact serialization and deserialization.
5. Single-threaded execution speed (< 15s) and median ceiling MAE < 2.5 OVR.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error

from src.models import (
    FeatureContractError,
    HistGradientBoostingTrajectoryRegressor,
    RidgeTrajectoryRegressor,
    TrajectoryFeatureContract,
    TrajectoryPrediction,
)
from src.models.trajectory_regressor import QuantileTrajectoryRegressor


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


def test_quantile_regressor_lifecycle_and_fit():
    """Verify QuantileTrajectoryRegressor fit, predict, and quantiles."""
    model = QuantileTrajectoryRegressor(max_iter=30, random_state=42)
    assert model.is_fitted is False
    assert model.model_version == "0.2.0-quantile-hgb"

    df = make_sample_dataframe()
    X = TrajectoryFeatureContract.extract_feature_matrix(df)
    y = TrajectoryFeatureContract.extract_target_vector(df)

    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(X)

    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_quantiles(X)

    model.fit(X, y)
    assert model.is_fitted is True
    assert model.training_duration_seconds > 0.0

    # Predict median
    median_preds = model.predict(X)
    assert isinstance(median_preds, np.ndarray)
    assert len(median_preds) == 4

    # Predict quantiles dict
    quantiles = model.predict_quantiles(X)
    assert "q10" in quantiles and "median" in quantiles and "q90" in quantiles
    assert len(quantiles["q10"]) == 4
    assert len(quantiles["median"]) == 4
    assert len(quantiles["q90"]) == 4

    # Check trajectory prediction and confidence interval
    trajectories = model.predict_trajectory(df)
    assert len(trajectories) == 4
    for item in trajectories:
        assert isinstance(item, TrajectoryPrediction)
        assert 35.0 <= item.predicted_overall <= 99.0
        assert item.confidence_interval is not None
        low, high = item.confidence_interval
        assert 35.0 <= low <= high <= 99.0
        assert low <= item.predicted_overall <= high


def test_trajectory_regressors_contract_rejection():
    """Verify regressors reject malformed or invalid input DataFrames."""
    model = QuantileTrajectoryRegressor(max_iter=20)
    df_missing = make_sample_dataframe().drop(columns=["potential_diff"])

    with pytest.raises(FeatureContractError):
        model.fit(df_missing, pd.Series([1.0, 2.0, 3.0, 4.0]))

    # Valid fit first
    df_valid = make_sample_dataframe()
    X_valid = TrajectoryFeatureContract.extract_feature_matrix(df_valid)
    y_valid = TrajectoryFeatureContract.extract_target_vector(df_valid)
    model.fit(X_valid, y_valid)

    with pytest.raises(FeatureContractError):
        model.predict(df_missing)

    with pytest.raises(FeatureContractError):
        model.predict_quantiles(df_missing)

    with pytest.raises(FeatureContractError):
        model.predict_trajectory(df_missing)


def test_trajectory_prediction_bounds_clamping():
    """Verify predict_trajectory strictly clamps output ratings to [35.0, 99.0]."""
    model = QuantileTrajectoryRegressor(max_iter=30)
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
        assert traj.confidence_interval is not None
        low, high = traj.confidence_interval
        assert 35.0 <= low <= high <= 99.0


def test_quantile_regressor_serialization(tmp_path: Path):
    """Verify QuantileTrajectoryRegressor serialization and deserialization via joblib."""
    df = make_sample_dataframe()
    X = TrajectoryFeatureContract.extract_feature_matrix(df)
    y = TrajectoryFeatureContract.extract_target_vector(df)

    model = QuantileTrajectoryRegressor(max_iter=20, random_state=42)
    model.fit(X, y)

    save_file = tmp_path / "test_model.joblib"
    model.save(save_file)
    assert save_file.exists()

    loaded_model = QuantileTrajectoryRegressor.load(save_file)
    assert loaded_model.is_fitted is True
    assert loaded_model.model_version == model.model_version

    orig_preds = model.predict(X)
    loaded_preds = loaded_model.predict(X)
    np.testing.assert_allclose(orig_preds, loaded_preds)


def test_quantile_regressor_verification_gate():
    """Verify single-threaded fit < 15s and median ceiling MAE < 2.5 OVR on 78,171 rows."""
    dataset_path = Path("data/processed/trajectory_training.parquet")
    if not dataset_path.exists():
        pytest.skip(f"Longitudinal dataset {dataset_path} not found.")

    df = pd.read_parquet(dataset_path)
    assert len(df) in (58146, 78171)

    # Train / validation split chronologically or on held-out cohort
    train_mask = df["edition_year"] <= 2021
    val_mask = df["edition_year"] > 2021

    df_train = df[train_mask]
    df_val = df[val_mask]

    X_train = TrajectoryFeatureContract.extract_feature_matrix(df_train)
    y_train = TrajectoryFeatureContract.extract_target_vector(df_train)

    X_val = TrajectoryFeatureContract.extract_feature_matrix(df_val)
    y_val = TrajectoryFeatureContract.extract_target_vector(df_val)

    model = QuantileTrajectoryRegressor(max_iter=60, random_state=42)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_duration = time.perf_counter() - t0

    # Gate 1: Fit time < 15s
    assert fit_duration < 15.0, f"Fit took {fit_duration:.2f}s, exceeding 15s hardware budget."

    # Gate 2: Validation MAE on median ceiling < 2.5 OVR
    val_preds = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_preds)
    assert val_mae < 2.5, f"Validation MAE {val_mae:.3f} exceeded 2.5 OVR threshold."

    # Save artifact to default models/trajectory_regressor.joblib
    artifact_path = model.save("models/trajectory_regressor.joblib")
    assert artifact_path.exists()
