"""Tests for FeatureStore extraction, validation, and serialization.

Verifies exact row cardinality (58,146 legacy and 78,171 modern rows),
zero lookahead leakage invariants, Pydantic schema validation, and Parquet persistence.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.models.feature_store import FeatureStore


@pytest.fixture(scope="module")
def feature_store() -> FeatureStore:
    """Instantiate standard FeatureStore instance."""
    return FeatureStore()


def test_feature_store_verification_gate_cardinality(feature_store: FeatureStore) -> None:
    """Verify FeatureStore extracts exactly 58,146 trajectory career rows from baseline staging."""
    df_legacy = feature_store.extract_trajectory_features(source_table="stg_fifa_granular")
    assert len(df_legacy) == 58146
    assert df_legacy["player_id"].nunique() > 10000


def test_feature_store_modern_fact_cardinality(feature_store: FeatureStore) -> None:
    """Verify FeatureStore extracts 78,171 multi-season trajectories from gold fct_player_seasons."""
    df_modern = feature_store.extract_trajectory_features(source_table="fct_player_seasons")
    assert len(df_modern) == 78171
    assert df_modern["player_id"].nunique() > 12000


def test_feature_store_zero_lookahead_leakage(feature_store: FeatureStore) -> None:
    """Verify zero lookahead leakage: target_edition_year strictly succeeds edition_year."""
    df = feature_store.extract_trajectory_features(
        source_table="fct_player_seasons", sample_limit=1000
    )

    # Invariants
    assert (df["target_edition_year"] == df["edition_year"] + 1).all()
    assert (df["target_overall"].notna()).all()
    assert (df["lag_overall_1"].notna()).all()

    # Exact arithmetic
    expected_growth_1y = (df["overall"] - df["lag_overall_1"]).round(2)
    assert (df["growth_rate_1y"].round(2) == expected_growth_1y).all()

    expected_target_growth = (df["target_overall"] - df["overall"]).round(2)
    assert (df["target_growth"].round(2) == expected_target_growth).all()


def test_feature_store_contract_validation(feature_store: FeatureStore) -> None:
    """Verify extracted feature matrix satisfies TrajectoryFeatureContract."""
    sample_df = feature_store.extract_trajectory_features(sample_limit=500)
    # Should complete with zero errors
    feature_store.validate_contract(sample_df, require_target=True)


def test_feature_store_export_parquet(feature_store: FeatureStore, tmp_path: Path) -> None:
    """Verify Parquet export writes valid, reloadable compressed file."""
    sample_df = feature_store.extract_trajectory_features(sample_limit=200)
    out_file = tmp_path / "test_features.parquet"

    saved_path = feature_store.export_parquet(sample_df, destination_path=out_file)
    assert saved_path.exists()
    assert saved_path.stat().st_size > 1000

    reloaded = pd.read_parquet(saved_path)
    assert len(reloaded) == 200
    assert list(reloaded.columns) == list(sample_df.columns)


def test_feature_store_temporal_splits(feature_store: FeatureStore) -> None:
    """Verify temporal train/val/test splits have zero chronological overlap."""
    sample_df = feature_store.extract_trajectory_features(sample_limit=1000)
    train_df, val_df, test_df = feature_store.get_temporal_splits(
        sample_df,
        train_end_year=2020,
        val_year=2021,
    )

    if not train_df.empty:
        assert (train_df["edition_year"] <= 2020).all()
    if not val_df.empty:
        assert (val_df["edition_year"] == 2021).all()
    if not test_df.empty:
        assert (test_df["edition_year"] > 2021).all()
