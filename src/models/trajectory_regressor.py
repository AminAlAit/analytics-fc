"""Quantile trajectory regressor using histogram-based gradient boosting.

Fits three quantile estimators (10th, 50th/median, 90th percentiles) to produce
developmental rating trajectories with calibrated confidence intervals under strict
hardware and memory safety constraints.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.models.trajectory_predictor import (
    BaseTrajectoryPredictor,
    TrajectoryFeatureContract,
    TrajectoryPrediction,
)

logger = logging.getLogger("analytics_fc.models.trajectory_regressor")


class QuantileTrajectoryRegressor(BaseTrajectoryPredictor):
    """Gradient boosted quantile regressor predicting median growth and uncertainty bounds.

    Estimates 10th, 50th (median), and 90th percentile developmental trajectories
    for multi-season player careers.
    """

    def __init__(
        self,
        max_iter: int = 60,
        learning_rate: float = 0.1,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        random_state: int = 42,
        model_version: str = "0.2.0-quantile-hgb",
    ) -> None:
        super().__init__(model_version=model_version)
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state

        self.estimator_q10 = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.10,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
        )
        self.estimator_median = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.50,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
        )
        self.estimator_q90 = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.90,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
        )
        self.training_duration_seconds: float = 0.0

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series[Any] | np.ndarray,
    ) -> QuantileTrajectoryRegressor:
        """Fit 10th, 50th, and 90th quantile estimators on validated feature matrix."""
        TrajectoryFeatureContract.validate_features(X)
        X_mat = TrajectoryFeatureContract.extract_feature_matrix(X)
        y_vec = np.asarray(y, dtype=float)

        t0 = time.perf_counter()
        logger.info("Fitting median quantile estimator (0.50)...")
        self.estimator_median.fit(X_mat, y_vec)

        logger.info("Fitting lower quantile estimator (0.10)...")
        self.estimator_q10.fit(X_mat, y_vec)

        logger.info("Fitting upper quantile estimator (0.90)...")
        self.estimator_q90.fit(X_mat, y_vec)

        self.training_duration_seconds = time.perf_counter() - t0
        self._is_fitted = True
        logger.info(
            "Quantile estimators fitted successfully in %.2fs.",
            self.training_duration_seconds,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict median expected future growth deltas (50th percentile)."""
        if not self.is_fitted:
            raise RuntimeError(
                "QuantileTrajectoryRegressor is not fitted. Call fit() before predict()."
            )
        TrajectoryFeatureContract.validate_features(X)
        X_mat = TrajectoryFeatureContract.extract_feature_matrix(X)
        preds = self.estimator_median.predict(X_mat)
        return np.asarray(preds, dtype=float)

    def predict_quantiles(
        self,
        X: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        """Predict 10th, 50th, and 90th percentile growth deltas."""
        if not self.is_fitted:
            raise RuntimeError(
                "QuantileTrajectoryRegressor is not fitted. Call fit() before predict()."
            )
        TrajectoryFeatureContract.validate_features(X)
        X_mat = TrajectoryFeatureContract.extract_feature_matrix(X)

        q10 = np.asarray(self.estimator_q10.predict(X_mat), dtype=float)
        median = np.asarray(self.estimator_median.predict(X_mat), dtype=float)
        q90 = np.asarray(self.estimator_q90.predict(X_mat), dtype=float)

        return {"q10": q10, "median": median, "q90": q90}

    def predict_trajectory(
        self,
        features: pd.DataFrame,
    ) -> list[TrajectoryPrediction]:
        """Generate structured trajectory predictions with 10th-90th confidence intervals."""
        TrajectoryFeatureContract.validate_features(features)
        quantiles = self.predict_quantiles(features)
        q10_deltas = quantiles["q10"]
        med_deltas = quantiles["median"]
        q90_deltas = quantiles["q90"]

        predictions: list[TrajectoryPrediction] = []
        for i, row in features.reset_index(drop=True).iterrows():
            curr_ovr = float(row["overall"])
            median_growth = float(med_deltas[i])
            q10_growth = float(q10_deltas[i])
            q90_growth = float(q90_deltas[i])

            pred_ovr = float(np.clip(curr_ovr + median_growth, 35.0, 99.0))
            lower_bound = float(np.clip(curr_ovr + q10_growth, 35.0, 99.0))
            upper_bound = float(np.clip(curr_ovr + q90_growth, 35.0, 99.0))

            # Guarantee strict monotonicity for the interval
            ci_low = min(lower_bound, pred_ovr, upper_bound)
            ci_high = max(lower_bound, pred_ovr, upper_bound)

            predictions.append(
                TrajectoryPrediction(
                    player_id=int(row.get("player_id", i)),
                    current_age=int(row["age"]),
                    current_overall=curr_ovr,
                    target_edition_year=int(
                        row.get("target_edition_year", int(row.get("edition_year", 2021)) + 1)
                    ),
                    predicted_growth=round(median_growth, 2),
                    predicted_overall=round(pred_ovr, 2),
                    confidence_interval=(round(ci_low, 2), round(ci_high, 2)),
                    model_version=self.model_version,
                )
            )
        return predictions

    def save(self, path: str | Path | None = None) -> Path:
        """Serialize model artifact to disk using joblib."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted QuantileTrajectoryRegressor.")

        save_path = Path(path) if path is not None else Path("models/trajectory_regressor.joblib")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)
        logger.info("QuantileTrajectoryRegressor serialized to %s", save_path)
        return save_path

    @classmethod
    def load(cls, path: str | Path) -> QuantileTrajectoryRegressor:
        """Load serialized QuantileTrajectoryRegressor from disk."""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found at {load_path}")
        model = joblib.load(load_path)
        if not isinstance(model, QuantileTrajectoryRegressor):
            raise TypeError(
                f"Expected QuantileTrajectoryRegressor instance, got {type(model).__name__}"
            )
        return model
