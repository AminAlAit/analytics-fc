"""Model 2: Career trajectory predictor interface and longitudinal feature contracts.

Establishes:
1. Pydantic feature schemas validating individual player observation vectors.
2. TrajectoryFeatureContract validating dataframe matrices, historical lag invariants,
   and zero-leakage temporal constraints.
3. BaseTrajectoryPredictor abstract base class for cross-season growth forecasting.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

logger = logging.getLogger("analytics_fc.models.trajectory_predictor")


class FeatureContractError(ValueError):
    """Raised when longitudinal career vector schemas violate structural invariants."""


class TrajectoryFeatureRecord(BaseModel):
    """Pydantic contract validating an individual multi-season player feature observation."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    player_id: int = Field(gt=0, description="Canonical player identification ID")
    edition_year: int = Field(ge=2000, le=2050, description="Observation FIFA edition year")
    age: int = Field(ge=15, le=50, description="Player age at reference edition")
    overall: float = Field(ge=35.0, le=99.0, description="Current overall rating")
    potential: float = Field(ge=35.0, le=99.0, description="Current potential ceiling")
    potential_diff: float = Field(description="Ceiling headroom (potential minus overall rating)")
    pace: float = Field(default=50.0, ge=1.0, le=99.0)
    shooting: float = Field(default=50.0, ge=1.0, le=99.0)
    passing: float = Field(default=50.0, ge=1.0, le=99.0)
    dribbling: float = Field(default=50.0, ge=1.0, le=99.0)
    defending: float = Field(default=50.0, ge=1.0, le=99.0)
    physic: float = Field(default=50.0, ge=1.0, le=99.0)
    growth_rate_1y: float | None = Field(
        default=None,
        description="1-year historical rating growth velocity (overall_t minus overall_t-1)",
    )
    lag_overall_1: float | None = Field(
        default=None,
        ge=35.0,
        le=99.0,
        description="Overall rating in previous edition (t-1)",
    )
    target_edition_year: int | None = Field(
        default=None,
        ge=2000,
        le=2050,
        description="Forecast target edition year",
    )
    target_overall: float | None = Field(
        default=None,
        ge=35.0,
        le=99.0,
        description="Ground truth overall rating in target edition",
    )
    target_growth: float | None = Field(
        default=None,
        description="Actual rating delta (target_overall minus overall)",
    )

    @model_validator(mode="after")
    def validate_invariants(self) -> TrajectoryFeatureRecord:
        """Validate relational and temporal constraints."""
        # Validate potential headroom consistency
        expected_diff = round(self.potential - self.overall, 2)
        if abs(self.potential_diff - expected_diff) > 0.05:
            raise ValueError(
                f"potential_diff ({self.potential_diff}) must equal potential ({self.potential}) "
                f"minus overall ({self.overall}) = {expected_diff}"
            )

        # Validate 1-year lag consistency if both are present
        if self.lag_overall_1 is not None and self.growth_rate_1y is not None:
            expected_velocity = round(self.overall - self.lag_overall_1, 2)
            if abs(self.growth_rate_1y - expected_velocity) > 0.05:
                raise ValueError(
                    f"growth_rate_1y ({self.growth_rate_1y}) must equal overall ({self.overall}) "
                    f"minus lag_overall_1 ({self.lag_overall_1}) = {expected_velocity}"
                )

        # Validate target temporal ordering (leakage prevention)
        if self.target_edition_year is not None:
            if self.target_edition_year <= self.edition_year:
                raise ValueError(
                    f"target_edition_year ({self.target_edition_year}) must strictly succeed "
                    f"reference edition_year ({self.edition_year})"
                )

        # Validate target growth consistency if both are present
        if self.target_overall is not None and self.target_growth is not None:
            expected_growth = round(self.target_overall - self.overall, 2)
            if abs(self.target_growth - expected_growth) > 0.05:
                raise ValueError(
                    f"target_growth ({self.target_growth}) must equal target_overall "
                    f"({self.target_overall}) minus overall ({self.overall}) = {expected_growth}"
                )

        return self


class TrajectoryPrediction(BaseModel):
    """Structured result of a career trajectory growth forecast."""

    model_config = ConfigDict(frozen=True)

    player_id: int
    current_age: int
    current_overall: float
    target_edition_year: int
    predicted_growth: float
    predicted_overall: float = Field(ge=35.0, le=99.0)
    confidence_interval: tuple[float, float] | None = None
    model_version: str = "0.1.0"


class TrajectoryFeatureContract:
    """Longitudinal feature contract and tabular matrix validator for Model 2."""

    REQUIRED_FEATURES: list[str] = [
        "age",
        "overall",
        "potential",
        "potential_diff",
        "pace",
        "shooting",
        "passing",
        "dribbling",
        "defending",
        "physic",
        "growth_rate_1y",
    ]

    CORE_COLUMNS: list[str] = [
        "player_id",
        "edition_year",
        *REQUIRED_FEATURES,
    ]

    TARGET_COLUMN: str = "target_growth"

    @classmethod
    def validate_features(cls, df: pd.DataFrame, require_target: bool = False) -> None:
        """Validate input DataFrame conforms to the Model 2 longitudinal feature contract."""
        if not isinstance(df, pd.DataFrame):
            raise FeatureContractError("Expected pandas DataFrame as input.")

        if len(df) == 0:
            raise FeatureContractError("Input DataFrame cannot be empty.")

        # Check required feature columns
        missing = [col for col in cls.REQUIRED_FEATURES if col not in df.columns]
        if missing:
            raise FeatureContractError(f"Missing required longitudinal feature columns: {missing}")

        # Check target column if required
        if require_target and cls.TARGET_COLUMN not in df.columns:
            raise FeatureContractError(f"Missing required target column: '{cls.TARGET_COLUMN}'")

        # Check missing values in required features
        null_counts = df[cls.REQUIRED_FEATURES].isna().sum()
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        if cols_with_nulls:
            raise FeatureContractError(
                f"Null values detected in required feature columns: {cols_with_nulls}"
            )

        # Range bounds invariants
        if (df["age"] < 15).any() or (df["age"] > 50).any():
            invalid_ages = df.loc[(df["age"] < 15) | (df["age"] > 50), "age"].tolist()[:5]
            raise FeatureContractError(f"Age out of bounds [15, 50]: {invalid_ages}")

        if (df["overall"] < 35).any() or (df["overall"] > 99).any():
            invalid_ovr = df.loc[(df["overall"] < 35) | (df["overall"] > 99), "overall"].tolist()[
                :5
            ]
            raise FeatureContractError(f"Overall rating out of bounds [35, 99]: {invalid_ovr}")

        if (df["potential"] < 35).any() or (df["potential"] > 99).any():
            invalid_pot = df.loc[
                (df["potential"] < 35) | (df["potential"] > 99), "potential"
            ].tolist()[:5]
            raise FeatureContractError(f"Potential rating out of bounds [35, 99]: {invalid_pot}")

        # Check temporal order if target_edition_year and edition_year are present
        if "target_edition_year" in df.columns and "edition_year" in df.columns:
            invalid_temporal = df[df["target_edition_year"] <= df["edition_year"]]
            if len(invalid_temporal) > 0:
                raise FeatureContractError(
                    f"Temporal leakage detected: target_edition_year must succeed edition_year. "
                    f"Found {len(invalid_temporal)} violating rows."
                )

    @classmethod
    def extract_feature_matrix(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Extract clean, contract-compliant feature matrix X."""
        cls.validate_features(df, require_target=False)
        return df[cls.REQUIRED_FEATURES].copy()

    @classmethod
    def extract_target_vector(cls, df: pd.DataFrame) -> pd.Series[Any]:
        """Extract clean, contract-compliant target vector y."""
        cls.validate_features(df, require_target=True)
        return df[cls.TARGET_COLUMN].copy()


class BaseTrajectoryPredictor(ABC):
    """Abstract base class establishing the contract for Model 2 trajectory regressors."""

    def __init__(self, model_version: str = "0.1.0") -> None:
        self.model_version = model_version
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Return True if regressor parameters have been estimated from data."""
        return self._is_fitted

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series[Any] | np.ndarray) -> BaseTrajectoryPredictor:
        """Fit model weights on validated longitudinal feature matrix X and target y."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict expected future growth deltas for input feature matrix X."""

    def predict_trajectory(
        self,
        features: pd.DataFrame,
    ) -> list[TrajectoryPrediction]:
        """Generate structured trajectory predictions for input records."""
        TrajectoryFeatureContract.validate_features(features)
        deltas = self.predict(features)
        predictions: list[TrajectoryPrediction] = []
        for i, row in features.reset_index(drop=True).iterrows():
            growth = float(deltas[i])
            curr_ovr = float(row["overall"])
            pred_ovr = float(np.clip(curr_ovr + growth, 35.0, 99.0))
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
                    model_version=self.model_version,
                )
            )
        return predictions


class RidgeTrajectoryRegressor(BaseTrajectoryPredictor):
    """L2 regularized linear regression baseline for rating growth trajectory."""

    def __init__(
        self,
        alpha: float = 1.0,
        random_state: int = 42,
        model_version: str = "0.1.0-ridge",
    ) -> None:
        super().__init__(model_version=model_version)
        self.alpha = alpha
        self.random_state = random_state
        self.estimator = Ridge(alpha=self.alpha, random_state=self.random_state)
        self.training_duration_seconds: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series[Any] | np.ndarray) -> RidgeTrajectoryRegressor:
        """Fit Ridge regression on validated longitudinal feature matrix."""
        TrajectoryFeatureContract.validate_features(X)
        X_mat = TrajectoryFeatureContract.extract_feature_matrix(X)
        t0 = time.perf_counter()
        self.estimator.fit(X_mat, y)
        self.training_duration_seconds = time.perf_counter() - t0
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict expected future growth deltas using fitted Ridge weights."""
        if not self.is_fitted:
            raise RuntimeError(
                "RidgeTrajectoryRegressor is not fitted. Call fit() before predict()."
            )
        TrajectoryFeatureContract.validate_features(X)
        X_mat = TrajectoryFeatureContract.extract_feature_matrix(X)
        preds = self.estimator.predict(X_mat)
        return np.asarray(preds, dtype=float)


class HistGradientBoostingTrajectoryRegressor(BaseTrajectoryPredictor):
    """Histogram-based gradient boosting baseline for non-linear growth curves."""

    def __init__(
        self,
        max_iter: int = 100,
        learning_rate: float = 0.1,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        random_state: int = 42,
        model_version: str = "0.1.0-hgb",
    ) -> None:
        super().__init__(model_version=model_version)
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state
        self.estimator = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
        )
        self.training_duration_seconds: float = 0.0

    def fit(
        self, X: pd.DataFrame, y: pd.Series[Any] | np.ndarray
    ) -> HistGradientBoostingTrajectoryRegressor:
        """Fit histogram GBDT regression on validated longitudinal feature matrix."""
        TrajectoryFeatureContract.validate_features(X)
        X_mat = TrajectoryFeatureContract.extract_feature_matrix(X)
        t0 = time.perf_counter()
        self.estimator.fit(X_mat, y)
        self.training_duration_seconds = time.perf_counter() - t0
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict expected future growth deltas using fitted GBDT trees."""
        if not self.is_fitted:
            raise RuntimeError(
                "HistGradientBoostingTrajectoryRegressor is not fitted. Call fit() before predict()."
            )
        TrajectoryFeatureContract.validate_features(X)
        X_mat = TrajectoryFeatureContract.extract_feature_matrix(X)
        preds = self.estimator.predict(X_mat)
        return np.asarray(preds, dtype=float)
