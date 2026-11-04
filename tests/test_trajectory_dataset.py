"""Unit and integration tests for longitudinal trajectory dataset extraction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.build_trajectory_dataset import (
    DEFAULT_OUTPUT_PATH,
    build_longitudinal_trajectory_dataset,
)
from src.models.trajectory_predictor import TrajectoryFeatureContract


def test_build_longitudinal_trajectory_dataset_execution(tmp_path: Path):
    """Verify trajectory extraction builds contract-compliant parquet dataset."""
    test_parquet = tmp_path / "test_trajectories.parquet"

    df = build_longitudinal_trajectory_dataset(
        output_path=test_parquet,
        sample_limit=500,
    )

    assert len(df) == 500
    assert test_parquet.exists()
    assert test_parquet.stat().st_size > 1000

    # Verify reload from parquet
    reloaded = pd.read_parquet(test_parquet)
    assert len(reloaded) == 500
    assert list(reloaded.columns) == list(df.columns)


def test_trajectory_dataset_cardinality_and_player_count():
    """Verify production parquet contains over 3,000 multi-year player careers."""
    if not DEFAULT_OUTPUT_PATH.exists():
        build_longitudinal_trajectory_dataset()

    df = pd.read_parquet(DEFAULT_OUTPUT_PATH)

    # Acceptance criteria: extract over 3,000 multi-year player careers
    assert len(df) > 3000, f"Expected >3,000 observations, got {len(df)}"
    assert df["player_id"].nunique() > 3000, (
        f"Expected >3,000 unique players, got {df['player_id'].nunique()}"
    )


def test_trajectory_dataset_zero_null_targets():
    """Verify dataset contains zero null target variables or features."""
    if not DEFAULT_OUTPUT_PATH.exists():
        build_longitudinal_trajectory_dataset()

    df = pd.read_parquet(DEFAULT_OUTPUT_PATH)

    # Definition of Done: zero null target variables
    assert df["target_growth"].isna().sum() == 0
    assert df["target_overall"].isna().sum() == 0

    # Ensure zero nulls in required features
    null_counts = df[TrajectoryFeatureContract.REQUIRED_FEATURES].isna().sum()
    assert null_counts.sum() == 0, f"Found nulls in features: {null_counts[null_counts > 0]}"


def test_trajectory_dataset_invariants_and_bounds():
    """Verify physical rating bounds and temporal ordering invariants."""
    if not DEFAULT_OUTPUT_PATH.exists():
        build_longitudinal_trajectory_dataset()

    df = pd.read_parquet(DEFAULT_OUTPUT_PATH)

    # Physical bounds
    assert (df["age"] >= 15).all() and (df["age"] <= 50).all()
    assert (df["overall"] >= 35.0).all() and (df["overall"] <= 99.0).all()
    assert (df["potential"] >= 35.0).all() and (df["potential"] <= 99.0).all()
    assert (df["target_overall"] >= 35.0).all() and (df["target_overall"] <= 99.0).all()

    # Relational consistency
    np_diff = (df["potential"] - df["overall"]).round(2)
    assert (df["potential_diff"].round(2) == np_diff).all()

    np_growth = (df["target_overall"] - df["overall"]).round(2)
    assert (df["target_growth"].round(2) == np_growth).all()

    # Temporal sequencing: target year must succeed edition year
    assert (df["target_edition_year"] > df["edition_year"]).all()


def test_trajectory_dataset_missing_db_error(tmp_path: Path):
    """Verify FileNotFoundError raised when DuckDB path does not exist."""
    fake_db = tmp_path / "nonexistent.duckdb"
    with pytest.raises(FileNotFoundError, match="DuckDB database not found"):
        build_longitudinal_trajectory_dataset(db_path=fake_db)
