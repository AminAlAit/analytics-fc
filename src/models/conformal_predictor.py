"""Conformal uncertainty bounds forecaster for prospect developmental trajectories.

Implements split conformal prediction (Inductive Conformal Prediction) providing
distribution-free, non-parametric prediction intervals with guaranteed finite-sample
coverage for player rating projections under physical boundary constraints.
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

logger = logging.getLogger("analytics_fc.models.conformal_predictor")


class ConformalTrajectoryPredictor(BaseTrajectoryPredictor):
    """Split conformal predictor computing finite-sample coverage intervals for rating growth.

    Calibrates absolute residual non-conformity scores on an independent holdout
    split to construct mathematically guaranteed prediction intervals.
    """

    def __init__(
        self,
        base_estimator: Any | None = None,
        alpha: float = 0.10,
        random_state: int = 42,
        model_version: str = "0.2.0-conformal-hgb",
    ) -> None:
        super().__init__(model_version=model_version)
        self.alpha = alpha
        self.random_state = random_state

        if base_estimator is not None:
            self.estimator = base_estimator
        else:
            self.estimator = HistGradientBoostingRegressor(
                loss="squared_error",
                max_iter=60,
                learning_rate=0.1,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                random_state=self.random_state,
            )

        self._is_calibrated: bool = False
        self.calibration_scores: np.ndarray = np.array([], dtype=float)
        self.q_val: float = 0.0
        self.training_duration_seconds: float = 0.0
        self.calibration_duration_seconds: float = 0.0

    @property
    def is_calibrated(self) -> bool:
        """Return True if non-conformity threshold has been calibrated."""
        return self._is_calibrated

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series[Any] | np.ndarray,
    ) -> ConformalTrajectoryPredictor:
        """Fit underlying regression estimator on training feature matrix."""
        TrajectoryFeatureContract.validate_features(X)
        X_mat = TrajectoryFeatureContract.extract_feature_matrix(X)
        y_vec = np.asarray(y, dtype=float)

        t0 = time.perf_counter()
        logger.info("Fitting base estimator for conformal prediction...")
        self.estimator.fit(X_mat, y_vec)
        self.training_duration_seconds = time.perf_counter() - t0
        self._is_fitted = True
        logger.info("Base estimator fitted in %.2fs.", self.training_duration_seconds)
        return self

    def calibrate(
        self,
        X_calib: pd.DataFrame,
        y_calib: pd.Series[Any] | np.ndarray,
        alpha: float | None = None,
    ) -> ConformalTrajectoryPredictor:
        """Compute non-conformity scores on calibration split and estimate threshold."""
        if not self.is_fitted:
            raise RuntimeError("Base estimator must be fitted before calibration.")

        TrajectoryFeatureContract.validate_features(X_calib)
        X_mat = TrajectoryFeatureContract.extract_feature_matrix(X_calib)
        y_vec = np.asarray(y_calib, dtype=float)

        t0 = time.perf_counter()
        if alpha is not None:
            self.alpha = alpha

        calib_preds = np.asarray(self.estimator.predict(X_mat), dtype=float)
        self.calibration_scores = np.abs(y_vec - calib_preds)

        n = len(self.calibration_scores)
        if n == 0:
            raise ValueError("Calibration set cannot be empty.")

        # Finite-sample adjusted quantile level: ceil((n + 1) * (1 - alpha)) / n
        q_level = min(1.0, float(np.ceil((n + 1) * (1.0 - self.alpha)) / n))
        self.q_val = float(np.quantile(self.calibration_scores, q_level, method="higher"))

        self.calibration_duration_seconds = time.perf_counter() - t0
        self._is_calibrated = True
        logger.info(
            "Conformal predictor calibrated: n=%d, alpha=%.2f, threshold q_val=%.3f in %.2fs.",
            n,
            self.alpha,
            self.q_val,
            self.calibration_duration_seconds,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict expected point growth deltas."""
        if not self.is_fitted:
            raise RuntimeError("Estimator is not fitted. Call fit() before predict().")
        TrajectoryFeatureContract.validate_features(X)
        X_mat = TrajectoryFeatureContract.extract_feature_matrix(X)
        preds = self.estimator.predict(X_mat)
        return np.asarray(preds, dtype=float)

    def predict_interval(
        self,
        X: pd.DataFrame,
        alpha: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute point prediction, lower conformal bound, and upper conformal bound.

        Returns:
            Tuple of (point_predictions, lower_bounds, upper_bounds).
        """
        if not self.is_calibrated:
            raise RuntimeError("Predictor is not calibrated. Call calibrate() first.")

        point_preds = self.predict(X)

        if alpha is not None and alpha != self.alpha:
            n = len(self.calibration_scores)
            q_level = min(1.0, float(np.ceil((n + 1) * (1.0 - alpha)) / n))
            q_thresh = float(np.quantile(self.calibration_scores, q_level, method="higher"))
        else:
            q_thresh = self.q_val

        lower_bounds = point_preds - q_thresh
        upper_bounds = point_preds + q_thresh
        return point_preds, lower_bounds, upper_bounds

    def predict_trajectory(
        self,
        features: pd.DataFrame,
    ) -> list[TrajectoryPrediction]:
        """Generate structured trajectory predictions with calibrated confidence intervals."""
        TrajectoryFeatureContract.validate_features(features)
        point_preds, lower_deltas, upper_deltas = self.predict_interval(features)

        predictions: list[TrajectoryPrediction] = []
        for i, row in features.reset_index(drop=True).iterrows():
            curr_ovr = float(row["overall"])
            growth = float(point_preds[i])
            low_delta = float(lower_deltas[i])
            high_delta = float(upper_deltas[i])

            pred_ovr = float(np.clip(curr_ovr + growth, 35.0, 99.0))
            lower_bound = float(np.clip(curr_ovr + low_delta, 35.0, 99.0))
            upper_bound = float(np.clip(curr_ovr + high_delta, 35.0, 99.0))

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
                    predicted_growth=round(growth, 2),
                    predicted_overall=round(pred_ovr, 2),
                    confidence_interval=(round(ci_low, 2), round(ci_high, 2)),
                    model_version=self.model_version,
                )
            )
        return predictions

    def evaluate_coverage(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series[Any] | np.ndarray,
    ) -> dict[str, float]:
        """Evaluate empirical coverage rate and mean interval width on unseen test data."""
        y_vec = np.asarray(y_test, dtype=float)
        _, lower_bounds, upper_bounds = self.predict_interval(X_test)

        is_covered = (y_vec >= lower_bounds) & (y_vec <= upper_bounds)
        coverage_pct = float(np.mean(is_covered) * 100.0)
        mean_width = float(np.mean(upper_bounds - lower_bounds))

        return {
            "target_coverage_pct": round((1.0 - self.alpha) * 100.0, 2),
            "empirical_coverage_pct": round(coverage_pct, 2),
            "mean_interval_width": round(mean_width, 3),
            "calibrated_threshold": round(self.q_val, 3),
            "test_sample_size": float(len(y_vec)),
        }

    def save(self, path: str | Path | None = None) -> Path:
        """Serialize calibrated conformal predictor artifact to disk."""
        if not self.is_calibrated:
            raise RuntimeError("Cannot save an uncalibrated ConformalTrajectoryPredictor.")

        save_path = Path(path) if path is not None else Path("models/conformal_predictor.joblib")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)
        logger.info("ConformalTrajectoryPredictor serialized to %s", save_path)
        return save_path

    @classmethod
    def load(cls, path: str | Path) -> ConformalTrajectoryPredictor:
        """Load serialized ConformalTrajectoryPredictor from disk."""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found at {load_path}")
        model = joblib.load(load_path)
        if not isinstance(model, ConformalTrajectoryPredictor):
            raise TypeError(
                f"Expected ConformalTrajectoryPredictor instance, got {type(model).__name__}"
            )
        return model
