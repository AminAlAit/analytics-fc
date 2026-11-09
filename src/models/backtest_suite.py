"""Longitudinal model evaluation suite and zero-leakage walk-forward harness.

Provides:
1. Leak-free cross-validation strictly grouped by player_id using GroupKFold.
2. Temporal expanding-window walk-forward backtesting across sequential FIFA editions.
3. Automated generation and export of model evaluation metrics report JSON.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold

from src.models.trajectory_predictor import (
    BaseTrajectoryPredictor,
    FeatureContractError,
    TrajectoryFeatureContract,
)

logger = logging.getLogger("analytics_fc.models.backtest_suite")


class GroupKFoldLeakageAuditor:
    """GroupKFold evaluation harness verifying zero player overlap across splits."""

    def __init__(
        self,
        n_splits: int = 5,
        group_col: str = "player_id",
    ) -> None:
        self.n_splits = n_splits
        self.group_col = group_col

    def evaluate(
        self,
        model: BaseTrajectoryPredictor,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Execute GroupKFold cross-validation, strictly isolating player careers."""
        TrajectoryFeatureContract.validate_features(df, require_target=True)
        if self.group_col not in df.columns:
            raise FeatureContractError(
                f"Missing group column '{self.group_col}' required for GroupKFold."
            )

        y = TrajectoryFeatureContract.extract_target_vector(df)
        groups = df[self.group_col]

        gkf = GroupKFold(n_splits=self.n_splits)
        folds: list[dict[str, Any]] = []

        total_overlaps = 0
        t0 = time.perf_counter()

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(df, y, groups=groups)):
            train_players = set(groups.iloc[train_idx])
            test_players = set(groups.iloc[test_idx])
            overlap = len(train_players.intersection(test_players))
            total_overlaps += overlap

            if overlap > 0:
                raise RuntimeError(
                    f"Data leakage detected in fold {fold_idx}: "
                    f"{overlap} players present in both train and test splits."
                )

            fold_model = copy.deepcopy(model)
            fold_model._is_fitted = False

            fold_model.fit(df.iloc[train_idx], y.iloc[train_idx])
            preds = fold_model.predict(df.iloc[test_idx])
            y_test = np.asarray(y.iloc[test_idx], dtype=float)

            mae = float(mean_absolute_error(y_test, preds))
            rmse = float(root_mean_squared_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))

            folds.append(
                {
                    "fold": fold_idx,
                    "train_records": int(len(train_idx)),
                    "test_records": int(len(test_idx)),
                    "train_players": int(len(train_players)),
                    "test_players": int(len(test_players)),
                    "player_overlap": int(overlap),
                    "mae": round(mae, 4),
                    "rmse": round(rmse, 4),
                    "r2": round(r2, 4),
                }
            )

        duration = time.perf_counter() - t0
        mae_list = [f["mae"] for f in folds]
        rmse_list = [f["rmse"] for f in folds]
        r2_list = [f["r2"] for f in folds]

        return {
            "evaluation_type": "GroupKFold_Player_Isolated",
            "n_splits": self.n_splits,
            "total_records": int(len(df)),
            "unique_players": int(groups.nunique()),
            "total_player_overlaps": total_overlaps,
            "zero_leakage_verified": total_overlaps == 0,
            "duration_seconds": round(duration, 3),
            "folds": folds,
            "mean_mae": round(float(np.mean(mae_list)), 4),
            "std_mae": round(float(np.std(mae_list)), 4),
            "mean_rmse": round(float(np.mean(rmse_list)), 4),
            "std_rmse": round(float(np.std(rmse_list)), 4),
            "mean_r2": round(float(np.mean(r2_list)), 4),
            "std_r2": round(float(np.std(r2_list)), 4),
        }


class WalkForwardBacktester:
    """Chronological expanding-window walk-forward evaluator for longitudinal models."""

    def __init__(
        self,
        start_eval_year: int = 2020,
        reference_year_col: str = "edition_year",
        target_year_col: str = "target_edition_year",
    ) -> None:
        self.start_eval_year = start_eval_year
        self.reference_year_col = reference_year_col
        self.target_year_col = target_year_col

    def evaluate(
        self,
        model: BaseTrajectoryPredictor,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Execute expanding window backtesting across sequential editions."""
        TrajectoryFeatureContract.validate_features(df, require_target=True)
        if self.reference_year_col not in df.columns or self.target_year_col not in df.columns:
            raise FeatureContractError(
                "Missing reference or target edition columns for walk-forward evaluation."
            )

        y = TrajectoryFeatureContract.extract_target_vector(df)
        unique_targets = sorted(
            [int(yr) for yr in df[self.target_year_col].unique() if yr >= self.start_eval_year]
        )

        if not unique_targets:
            raise FeatureContractError(
                f"No target editions found >= start_eval_year ({self.start_eval_year})."
            )

        folds: list[dict[str, Any]] = []
        t0 = time.perf_counter()

        for eval_year in unique_targets:
            train_mask = df[self.target_year_col] < eval_year
            test_mask = df[self.target_year_col] == eval_year

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            if len(train_indices) == 0 or len(test_indices) == 0:
                continue

            overlap = len(set(train_indices).intersection(set(test_indices)))
            if overlap > 0:
                raise RuntimeError(
                    f"Data leakage detected in walk-forward evaluation: "
                    f"{overlap} records present in both train and test splits."
                )

            max_train_target = int(df[self.target_year_col].iloc[train_indices].max())
            min_test_target = int(df[self.target_year_col].iloc[test_indices].min())

            if max_train_target >= min_test_target:
                raise RuntimeError(
                    f"Temporal leakage detected: max train target edition ({max_train_target}) "
                    f"must strictly precede test edition ({min_test_target})."
                )

            fold_model = copy.deepcopy(model)
            fold_model._is_fitted = False

            fold_model.fit(df.iloc[train_indices], y.iloc[train_indices])
            preds = fold_model.predict(df.iloc[test_indices])
            y_test = np.asarray(y.iloc[test_indices], dtype=float)

            mae = float(mean_absolute_error(y_test, preds))
            rmse = float(root_mean_squared_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))

            folds.append(
                {
                    "eval_target_year": int(eval_year),
                    "train_records": int(len(train_indices)),
                    "test_records": int(len(test_indices)),
                    "max_train_edition": max_train_target,
                    "max_train_target_edition": max_train_target,
                    "mae": round(mae, 4),
                    "rmse": round(rmse, 4),
                    "r2": round(r2, 4),
                }
            )

        duration = time.perf_counter() - t0
        mae_list = [f["mae"] for f in folds]
        rmse_list = [f["rmse"] for f in folds]
        r2_list = [f["r2"] for f in folds]

        return {
            "evaluation_type": "Temporal_Walk_Forward",
            "start_eval_year": self.start_eval_year,
            "evaluation_years": [f["eval_target_year"] for f in folds],
            "duration_seconds": round(duration, 3),
            "folds": folds,
            "mean_mae": round(float(np.mean(mae_list)), 4) if mae_list else 0.0,
            "std_mae": round(float(np.std(mae_list)), 4) if mae_list else 0.0,
            "mean_rmse": round(float(np.mean(rmse_list)), 4) if rmse_list else 0.0,
            "std_rmse": round(float(np.std(rmse_list)), 4) if rmse_list else 0.0,
            "mean_r2": round(float(np.mean(r2_list)), 4) if r2_list else 0.0,
            "std_r2": round(float(np.std(r2_list)), 4) if r2_list else 0.0,
        }


def generate_evaluation_report(
    model: BaseTrajectoryPredictor,
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    n_splits: int = 5,
    start_eval_year: int = 2020,
) -> dict[str, Any]:
    """Execute complete backtest suite and serialize evaluation metrics report to JSON."""
    logger.info("Executing GroupKFold leakage audit...")
    auditor = GroupKFoldLeakageAuditor(n_splits=n_splits)
    gkf_metrics = auditor.evaluate(model, df)

    logger.info("Executing temporal walk-forward backtest...")
    backtester = WalkForwardBacktester(start_eval_year=start_eval_year)
    wf_metrics = backtester.evaluate(model, df)

    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model_class": model.__class__.__name__,
        "model_version": getattr(model, "model_version", "unknown"),
        "total_observations": int(len(df)),
        "unique_players": int(df["player_id"].nunique()) if "player_id" in df.columns else None,
        "group_kfold_evaluation": gkf_metrics,
        "walk_forward_evaluation": wf_metrics,
    }

    if output_path is not None:
        save_file = Path(output_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Evaluation report saved to %s", save_file)

    return report
