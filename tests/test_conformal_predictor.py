"""Unit tests for conformal uncertainty bounds forecaster.

Validates:
1. ConformalTrajectoryPredictor lifecycle (fit -> calibrate -> predict).
2. Calculation of residual non-conformity scores and empirical quantile thresholds.
3. Clamping of predicted ratings and intervals to physical limits [35.0, 99.0].
4. Joblib round-trip serialization fidelity.
5. Verification gate: >= 90% empirical coverage on unseen test cohort.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import (
    FeatureContractError,
    TrajectoryFeatureContract,
    TrajectoryPrediction,
)
from src.models.conformal_predictor import ConformalTrajectoryPredictor


def make_sample_dataframe() -> pd.DataFrame:
    """Construct small contract-compliant DataFrame for isolated unit tests."""
    return pd.DataFrame(
        {
            "player_id": [101, 102, 103, 104, 105, 106],
            "edition_year": [2018, 2018, 2019, 2019, 2020, 2020],
            "target_edition_year": [2019, 2019, 2020, 2020, 2021, 2021],
            "age": [19, 21, 23, 25, 27, 29],
            "overall": [72.0, 75.0, 77.0, 80.0, 83.0, 86.0],
            "potential": [82.0, 83.0, 82.0, 84.0, 85.0, 86.0],
            "potential_diff": [10.0, 8.0, 5.0, 4.0, 2.0, 0.0],
            "pace": [78.0, 82.0, 80.0, 75.0, 70.0, 68.0],
            "shooting": [65.0, 70.0, 72.0, 78.0, 80.0, 84.0],
            "passing": [68.0, 72.0, 74.0, 80.0, 82.0, 86.0],
            "dribbling": [74.0, 78.0, 79.0, 81.0, 83.0, 85.0],
            "defending": [40.0, 45.0, 50.0, 55.0, 60.0, 62.0],
            "physic": [62.0, 68.0, 70.0, 72.0, 74.0, 75.0],
            "growth_rate_1y": [3.0, 2.0, 1.0, 1.0, 0.0, -1.0],
            "target_growth": [4.0, 3.0, 2.0, 1.0, 0.0, -1.0],
        }
    )


def test_conformal_predictor_lifecycle():
    """Verify fit, calibration, and prediction sequence."""
    df = make_sample_dataframe()
    X = TrajectoryFeatureContract.extract_feature_matrix(df)
    y = TrajectoryFeatureContract.extract_target_vector(df)

    predictor = ConformalTrajectoryPredictor(alpha=0.10, random_state=42)
    assert predictor.is_fitted is False
    assert predictor.is_calibrated is False

    # Cannot predict or calibrate prior to fitting
    with pytest.raises(RuntimeError, match="must be fitted"):
        predictor.calibrate(X, y)

    with pytest.raises(RuntimeError, match="not fitted"):
        predictor.predict(X)

    # Fit base model on first 4 records
    predictor.fit(X.iloc[:4], y.iloc[:4])
    assert predictor.is_fitted is True
    assert predictor.is_calibrated is False

    # Cannot predict intervals prior to calibration
    with pytest.raises(RuntimeError, match="not calibrated"):
        predictor.predict_interval(X)

    # Calibrate on last 2 records
    predictor.calibrate(X.iloc[4:], y.iloc[4:])
    assert predictor.is_calibrated is True
    assert predictor.q_val > 0.0
    assert len(predictor.calibration_scores) == 2

    # Predict intervals
    preds, lows, highs = predictor.predict_interval(X)
    assert len(preds) == 6
    assert np.all(lows <= preds)
    assert np.all(preds <= highs)


def test_conformal_bounds_clamping():
    """Verify physical rating bounds [35.0, 99.0] on extreme prospect cases."""
    df = make_sample_dataframe()
    X = TrajectoryFeatureContract.extract_feature_matrix(df)
    y = TrajectoryFeatureContract.extract_target_vector(df)

    predictor = ConformalTrajectoryPredictor(alpha=0.10)
    predictor.fit(X.iloc[:4], y.iloc[:4])
    predictor.calibrate(X.iloc[4:], y.iloc[4:])

    # Synthesize extreme edge cases
    extreme_df = df.copy()
    extreme_df.loc[0, "overall"] = 98.0
    extreme_df.loc[0, "potential"] = 99.0
    extreme_df.loc[0, "potential_diff"] = 1.0

    extreme_df.loc[1, "overall"] = 36.0
    extreme_df.loc[1, "potential"] = 40.0
    extreme_df.loc[1, "potential_diff"] = 4.0

    trajectories = predictor.predict_trajectory(extreme_df)
    assert len(trajectories) == 6
    for item in trajectories:
        assert isinstance(item, TrajectoryPrediction)
        assert 35.0 <= item.predicted_overall <= 99.0
        assert item.confidence_interval is not None
        low, high = item.confidence_interval
        assert 35.0 <= low <= high <= 99.0
        assert low <= item.predicted_overall <= high


def test_conformal_contract_rejection():
    """Verify malformed feature frames are rejected by fit and calibrate."""
    predictor = ConformalTrajectoryPredictor()
    df_missing = make_sample_dataframe().drop(columns=["growth_rate_1y"])

    with pytest.raises(FeatureContractError):
        predictor.fit(df_missing, pd.Series([1.0, 2.0]))

    df = make_sample_dataframe()
    X = TrajectoryFeatureContract.extract_feature_matrix(df)
    y = TrajectoryFeatureContract.extract_target_vector(df)
    predictor.fit(X.iloc[:4], y.iloc[:4])
    predictor.calibrate(X.iloc[4:], y.iloc[4:])

    with pytest.raises(FeatureContractError):
        predictor.calibrate(df_missing, pd.Series([1.0, 2.0]))

    with pytest.raises(FeatureContractError):
        predictor.predict(df_missing)

    with pytest.raises(FeatureContractError):
        predictor.predict_interval(df_missing)

    with pytest.raises(FeatureContractError):
        predictor.predict_trajectory(df_missing)


def test_conformal_predictor_serialization(tmp_path: Path):
    """Verify round-trip serialization fidelity via joblib."""
    df = make_sample_dataframe()
    X = TrajectoryFeatureContract.extract_feature_matrix(df)
    y = TrajectoryFeatureContract.extract_target_vector(df)

    predictor = ConformalTrajectoryPredictor(alpha=0.10, random_state=42)
    predictor.fit(X.iloc[:4], y.iloc[:4])
    predictor.calibrate(X.iloc[4:], y.iloc[4:])

    save_path = tmp_path / "conformal_model.joblib"
    predictor.save(save_path)
    assert save_path.exists()

    loaded = ConformalTrajectoryPredictor.load(save_path)
    assert loaded.is_calibrated is True
    assert loaded.q_val == predictor.q_val

    orig_p, orig_l, orig_h = predictor.predict_interval(X)
    load_p, load_l, load_h = loaded.predict_interval(X)
    np.testing.assert_allclose(orig_p, load_p)
    np.testing.assert_allclose(orig_l, load_l)
    np.testing.assert_allclose(orig_h, load_h)


def test_verification_gate_90_pct_coverage():
    """Verify empirical coverage >= 90% on unseen held-out test cohort."""
    dataset_path = Path("data/processed/trajectory_training.parquet")
    if not dataset_path.exists():
        pytest.skip(f"Longitudinal dataset {dataset_path} not found.")

    df = pd.read_parquet(dataset_path)
    assert len(df) in (58146, 78171)

    # Chronological partition
    train_df = df[df["edition_year"] <= 2020]
    calib_df = df[df["edition_year"] == 2021]
    test_df = df[df["edition_year"] > 2021]

    X_train = TrajectoryFeatureContract.extract_feature_matrix(train_df)
    y_train = TrajectoryFeatureContract.extract_target_vector(train_df)

    X_calib = TrajectoryFeatureContract.extract_feature_matrix(calib_df)
    y_calib = TrajectoryFeatureContract.extract_target_vector(calib_df)

    X_test = TrajectoryFeatureContract.extract_feature_matrix(test_df)
    y_test = TrajectoryFeatureContract.extract_target_vector(test_df)

    predictor = ConformalTrajectoryPredictor(alpha=0.10, random_state=42)

    t0 = time.perf_counter()
    predictor.fit(X_train, y_train)
    predictor.calibrate(X_calib, y_calib)
    fit_calib_duration = time.perf_counter() - t0

    # Ensure fast single-threaded execution
    assert fit_calib_duration < 15.0, (
        f"Fit and calibration took {fit_calib_duration:.2f}s, exceeding 15s budget."
    )

    metrics = predictor.evaluate_coverage(X_test, y_test)
    assert metrics["empirical_coverage_pct"] >= 90.0, (
        f"Empirical coverage {metrics['empirical_coverage_pct']}% below 90% threshold."
    )

    # Save default production artifact
    artifact_path = predictor.save("models/conformal_predictor.joblib")
    assert artifact_path.exists()
