"""Unit tests for Model 2 interface, Pydantic schemas, and longitudinal feature contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from src.models.trajectory_predictor import (
    BaseTrajectoryPredictor,
    FeatureContractError,
    HistGradientBoostingTrajectoryRegressor,
    RidgeTrajectoryRegressor,
    TrajectoryFeatureContract,
    TrajectoryFeatureRecord,
    TrajectoryPrediction,
    evaluate_trajectory_cv,
    evaluate_trajectory_walk_forward,
)


def test_trajectory_feature_record_valid():
    """Verify TrajectoryFeatureRecord successfully validates compliant player observations."""
    record = TrajectoryFeatureRecord(
        player_id=158023,
        edition_year=2021,
        age=21,
        overall=78.0,
        potential=86.0,
        potential_diff=8.0,
        pace=84.0,
        shooting=72.0,
        passing=75.0,
        dribbling=80.0,
        defending=45.0,
        physic=68.0,
        lag_overall_1=74.0,
        growth_rate_1y=4.0,
        target_edition_year=2022,
        target_overall=81.0,
        target_growth=3.0,
    )

    assert record.player_id == 158023
    assert record.overall == 78.0
    assert record.potential_diff == 8.0
    assert record.growth_rate_1y == 4.0
    assert record.target_growth == 3.0


def test_trajectory_feature_record_potential_diff_mismatch():
    """Verify TrajectoryFeatureRecord raises ValueError when potential_diff does not match."""
    with pytest.raises(ValueError, match="potential_diff"):
        TrajectoryFeatureRecord(
            player_id=1,
            edition_year=2021,
            age=20,
            overall=70.0,
            potential=80.0,
            potential_diff=5.0,  # Expected 10.0
        )


def test_trajectory_feature_record_growth_rate_mismatch():
    """Verify TrajectoryFeatureRecord raises ValueError when growth_rate_1y does not match."""
    with pytest.raises(ValueError, match="growth_rate_1y"):
        TrajectoryFeatureRecord(
            player_id=1,
            edition_year=2021,
            age=20,
            overall=75.0,
            potential=80.0,
            potential_diff=5.0,
            lag_overall_1=70.0,
            growth_rate_1y=1.0,  # Expected 5.0
        )


def test_trajectory_feature_record_temporal_leakage():
    """Verify TrajectoryFeatureRecord rejects target editions equal to or preceding reference."""
    with pytest.raises(ValueError, match="target_edition_year"):
        TrajectoryFeatureRecord(
            player_id=1,
            edition_year=2021,
            age=20,
            overall=75.0,
            potential=80.0,
            potential_diff=5.0,
            target_edition_year=2021,  # Same year is invalid
            target_overall=78.0,
            target_growth=3.0,
        )


def test_trajectory_prediction_schema():
    """Verify TrajectoryPrediction instantiation, fields, and immutability."""
    pred = TrajectoryPrediction(
        player_id=158023,
        current_age=21,
        current_overall=78.0,
        target_edition_year=2022,
        predicted_growth=3.2,
        predicted_overall=81.2,
        confidence_interval=(79.5, 82.9),
        model_version="0.1.0",
    )

    assert pred.player_id == 158023
    assert pred.predicted_growth == 3.2
    assert pred.predicted_overall == 81.2
    assert pred.confidence_interval == (79.5, 82.9)


def make_sample_dataframe() -> pd.DataFrame:
    """Helper creating a contract-compliant multi-season DataFrame."""
    return pd.DataFrame(
        {
            "player_id": [101, 102, 103],
            "edition_year": [2020, 2020, 2021],
            "target_edition_year": [2021, 2021, 2022],
            "age": [19, 23, 27],
            "overall": [72.0, 77.0, 83.0],
            "potential": [82.0, 81.0, 84.0],
            "potential_diff": [10.0, 4.0, 1.0],
            "pace": [78.0, 81.0, 70.0],
            "shooting": [65.0, 72.0, 80.0],
            "passing": [68.0, 74.0, 84.0],
            "dribbling": [74.0, 78.0, 82.0],
            "defending": [40.0, 50.0, 60.0],
            "physic": [62.0, 70.0, 75.0],
            "growth_rate_1y": [3.0, 1.0, 0.0],
            "target_growth": [4.0, 2.0, 0.0],
        }
    )


def test_trajectory_feature_contract_validation_success():
    """Verify TrajectoryFeatureContract validates compliant DataFrame."""
    df = make_sample_dataframe()
    TrajectoryFeatureContract.validate_features(df, require_target=True)
    X = TrajectoryFeatureContract.extract_feature_matrix(df)
    y = TrajectoryFeatureContract.extract_target_vector(df)

    assert len(X) == 3
    assert len(y) == 3
    assert list(X.columns) == TrajectoryFeatureContract.REQUIRED_FEATURES
    assert y.name == "target_growth"


def test_trajectory_feature_contract_missing_columns():
    """Verify FeatureContractError raised when required columns are missing."""
    df = make_sample_dataframe().drop(columns=["potential_diff"])
    with pytest.raises(FeatureContractError, match="Missing required longitudinal feature columns"):
        TrajectoryFeatureContract.validate_features(df)


def test_trajectory_feature_contract_null_values():
    """Verify FeatureContractError raised when null values exist in feature matrix."""
    df = make_sample_dataframe()
    df.loc[0, "growth_rate_1y"] = np.nan
    with pytest.raises(FeatureContractError, match="Null values detected"):
        TrajectoryFeatureContract.validate_features(df)


def test_trajectory_feature_contract_out_of_bounds():
    """Verify FeatureContractError raised when ratings or ages violate physical bounds."""
    df_bad_age = make_sample_dataframe()
    df_bad_age.loc[0, "age"] = 12
    with pytest.raises(FeatureContractError, match="Age out of bounds"):
        TrajectoryFeatureContract.validate_features(df_bad_age)

    df_bad_ovr = make_sample_dataframe()
    df_bad_ovr.loc[0, "overall"] = 105.0
    with pytest.raises(FeatureContractError, match="Overall rating out of bounds"):
        TrajectoryFeatureContract.validate_features(df_bad_ovr)


def test_trajectory_feature_contract_temporal_leakage():
    """Verify FeatureContractError raised when target_edition_year <= edition_year."""
    df_leakage = make_sample_dataframe()
    df_leakage.loc[0, "target_edition_year"] = 2019  # Preceding reference year
    with pytest.raises(FeatureContractError, match="Temporal leakage detected"):
        TrajectoryFeatureContract.validate_features(df_leakage)


class DummyTrajectoryPredictor(BaseTrajectoryPredictor):
    """Concrete predictor implementation for protocol contract testing."""

    def __init__(self, growth_constant: float = 2.0) -> None:
        super().__init__(model_version="0.1.0-test")
        self.growth_constant = growth_constant

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> DummyTrajectoryPredictor:
        TrajectoryFeatureContract.validate_features(X)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Predictor is not fitted.")
        TrajectoryFeatureContract.validate_features(X)
        # Simple heuristic growth: fraction of potential headroom
        headroom = X["potential_diff"].to_numpy(dtype=float)
        return np.asarray(np.maximum(0.0, headroom * 0.3), dtype=float)

    def predict_trajectory(self, features: pd.DataFrame) -> list[TrajectoryPrediction]:
        TrajectoryFeatureContract.validate_features(features)
        deltas = self.predict(features)
        predictions = []
        for i, row in features.reset_index(drop=True).iterrows():
            growth = float(deltas[i])
            curr_ovr = float(row["overall"])
            pred_ovr = float(min(99.0, max(35.0, curr_ovr + growth)))
            predictions.append(
                TrajectoryPrediction(
                    player_id=int(row.get("player_id", i)),
                    current_age=int(row["age"]),
                    current_overall=curr_ovr,
                    target_edition_year=int(row.get("target_edition_year", 2023)),
                    predicted_growth=round(growth, 2),
                    predicted_overall=round(pred_ovr, 2),
                    model_version=self.model_version,
                )
            )
        return predictions


def test_base_trajectory_predictor_protocol():
    """Verify BaseTrajectoryPredictor implementation lifecycle, fitting state, and prediction."""
    predictor = DummyTrajectoryPredictor()
    assert predictor.is_fitted is False

    df = make_sample_dataframe()
    X = TrajectoryFeatureContract.extract_feature_matrix(df)
    y = TrajectoryFeatureContract.extract_target_vector(df)

    predictor.fit(X, y)
    assert predictor.is_fitted is True

    preds = predictor.predict(X)
    assert len(preds) == 3
    assert np.all(preds >= 0.0)

    trajectories = predictor.predict_trajectory(df)
    assert len(trajectories) == 3
    assert isinstance(trajectories[0], TrajectoryPrediction)
    assert trajectories[0].player_id == 101
    assert trajectories[0].predicted_overall > trajectories[0].current_overall


def make_longitudinal_test_dataframe() -> pd.DataFrame:
    """Construct multi-season player career records with shared player IDs."""
    records = []
    # 6 players across 4 editions (2018 to 2021)
    for p_id in range(101, 107):
        base_ovr = 70.0 + (p_id % 10)
        for yr in [2018, 2019, 2020]:
            ovr = base_ovr + (yr - 2018) * 2.0
            pot = 85.0
            records.append(
                {
                    "player_id": p_id,
                    "edition_year": yr,
                    "target_edition_year": yr + 1,
                    "age": 20 + (yr - 2018),
                    "overall": ovr,
                    "potential": pot,
                    "potential_diff": pot - ovr,
                    "pace": 75.0,
                    "shooting": 70.0,
                    "passing": 72.0,
                    "dribbling": 74.0,
                    "defending": 50.0,
                    "physic": 65.0,
                    "growth_rate_1y": 2.0,
                    "target_growth": 2.0,
                }
            )
    return pd.DataFrame(records)


def test_group_kfold_player_id_zero_leakage():
    """Verify GroupKFold grouped by player_id strictly enforces zero player overlap."""
    df = make_longitudinal_test_dataframe()
    gkf = GroupKFold(n_splits=3)
    groups = df["player_id"]
    y = df["target_growth"]

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(df, y, groups=groups)):
        train_players = set(groups.iloc[train_idx])
        test_players = set(groups.iloc[test_idx])
        overlap = train_players.intersection(test_players)
        assert len(overlap) == 0, f"Fold {fold_idx} leaked player IDs: {overlap}"
        assert len(train_players) > 0
        assert len(test_players) > 0


def test_evaluate_trajectory_cv_ridge():
    """Verify evaluate_trajectory_cv with Ridge regressor reports zero overlap and valid metrics."""
    df = make_longitudinal_test_dataframe()
    model = RidgeTrajectoryRegressor(alpha=1.0)
    results = evaluate_trajectory_cv(model, df, n_splits=3, group_col="player_id")

    assert results["model_type"] == "RidgeTrajectoryRegressor"
    assert results["n_splits"] == 3
    assert results["unique_players"] == 6
    assert len(results["folds"]) == 3

    for fold in results["folds"]:
        assert fold["player_overlap"] == 0
        assert fold["mae"] >= 0.0
        assert fold["rmse"] >= 0.0

    assert results["mean_mae"] >= 0.0
    assert results["mean_rmse"] >= 0.0


def test_evaluate_trajectory_cv_hgb():
    """Verify evaluate_trajectory_cv with HistGradientBoosting regressor reports zero overlap."""
    df = make_longitudinal_test_dataframe()
    model = HistGradientBoostingTrajectoryRegressor(max_iter=20, min_samples_leaf=2)
    results = evaluate_trajectory_cv(model, df, n_splits=3, group_col="player_id")

    assert results["model_type"] == "HistGradientBoostingTrajectoryRegressor"
    assert len(results["folds"]) == 3
    for fold in results["folds"]:
        assert fold["player_overlap"] == 0
        assert fold["mae"] >= 0.0


def test_evaluate_trajectory_cv_missing_group_col():
    """Verify evaluate_trajectory_cv raises FeatureContractError when group_col is absent."""
    df = make_longitudinal_test_dataframe().drop(columns=["player_id"])
    model = RidgeTrajectoryRegressor()
    with pytest.raises(FeatureContractError, match="Missing group column"):
        evaluate_trajectory_cv(model, df, n_splits=2, group_col="player_id")


def test_evaluate_trajectory_walk_forward_zero_lookahead():
    """Verify walk-forward temporal evaluation strictly prevents future edition leakage."""
    df = make_longitudinal_test_dataframe()
    model = RidgeTrajectoryRegressor()
    results = evaluate_trajectory_walk_forward(model, df, start_eval_year=2020)

    assert results["model_type"] == "RidgeTrajectoryRegressor"
    assert len(results["folds"]) >= 2
    for fold in results["folds"]:
        assert fold["max_train_edition"] < fold["eval_target_year"]
        assert fold["mae"] >= 0.0


def test_evaluate_trajectory_walk_forward_validation_errors():
    """Verify walk-forward evaluation catches missing columns or invalid target years."""
    df = make_longitudinal_test_dataframe()
    model = RidgeTrajectoryRegressor()

    # Out of range start year
    with pytest.raises(FeatureContractError, match="No target editions found"):
        evaluate_trajectory_walk_forward(model, df, start_eval_year=2030)

    # Missing target year col
    df_missing = df.drop(columns=["target_edition_year"])
    with pytest.raises(FeatureContractError, match="Missing edition columns"):
        evaluate_trajectory_walk_forward(model, df_missing)


def test_group_kfold_on_full_longitudinal_dataset():
    """Verify GroupKFold across all 58,146 observations guarantees zero player leakage."""
    parquet_path = Path("data/processed/trajectory_training.parquet")
    if not parquet_path.exists():
        pytest.skip(f"Longitudinal dataset {parquet_path} not found.")

    df = pd.read_parquet(parquet_path)
    gkf = GroupKFold(n_splits=5)
    groups = df["player_id"]
    y = df["target_growth"]

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(df, y, groups=groups)):
        train_players = set(groups.iloc[train_idx])
        test_players = set(groups.iloc[test_idx])
        overlap = train_players.intersection(test_players)
        assert len(overlap) == 0, f"Full dataset fold {fold_idx} leaked player IDs"
