"""Model 2 package for career trajectory and growth forecasting."""

from src.models.trajectory_predictor import (
    BaseTrajectoryPredictor,
    FeatureContractError,
    TrajectoryFeatureContract,
    TrajectoryFeatureRecord,
    TrajectoryPrediction,
)

__all__ = [
    "BaseTrajectoryPredictor",
    "FeatureContractError",
    "TrajectoryFeatureContract",
    "TrajectoryFeatureRecord",
    "TrajectoryPrediction",
]
