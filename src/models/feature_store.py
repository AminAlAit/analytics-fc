"""Feature Store module for extracting, validating, and serializing longitudinal player features.

Queries DuckDB dimensional and fact models, enforces strict causal temporal
contracts (zero lookahead leakage), validates row vectors with Pydantic,
and exports compressed Parquet datasets for machine learning model training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from pipelines.config import config
from src.models.trajectory_predictor import TrajectoryFeatureContract

logger = logging.getLogger("analytics_fc.models.feature_store")


@dataclass(frozen=True)
class FeatureStoreConfig:
    """Configuration settings for DuckDB query and feature extraction."""

    db_path: Path = config.dbt_project_dir / "analytics_fc.duckdb"
    output_path: Path = config.base_dir / "data" / "processed" / "trajectory_training.parquet"
    duckdb_threads: int = 1
    duckdb_memory_limit: str = "1GB"


class FeatureStore:
    """Manages extraction and serialization of longitudinal trajectory matrices."""

    def __init__(self, cfg: FeatureStoreConfig | None = None) -> None:
        self.cfg = cfg or FeatureStoreConfig()

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """Create a configured read-only connection to the warehouse database."""
        if not self.cfg.db_path.exists():
            raise FileNotFoundError(f"Database not found at {self.cfg.db_path}")

        con = duckdb.connect(str(self.cfg.db_path), read_only=True)
        con.execute(f"SET threads = {self.cfg.duckdb_threads};")
        con.execute(f"SET memory_limit = '{self.cfg.duckdb_memory_limit}';")
        return con

    def extract_trajectory_features(
        self,
        source_table: str = "fct_player_seasons",
        sample_limit: int | None = None,
    ) -> pd.DataFrame:
        """Query DuckDB for multi-season trajectories with verified 1-year lag and target lead.

        Supports extracting from gold fact table fct_player_seasons (78,171 rows)
        or historical baseline stg_fifa_granular (58,146 rows).
        """
        con = self.get_connection()
        try:
            limit_clause = f"LIMIT {int(sample_limit)}" if sample_limit and sample_limit > 0 else ""

            if source_table == "stg_fifa_granular":
                # Historical verification query across FIFA 15-22
                query = f"""
                    WITH player_seasons AS (
                        SELECT
                            player_id,
                            short_name,
                            fifa_edition_year AS edition_year,
                            age,
                            overall,
                            potential,
                            potential - overall AS potential_diff,
                            COALESCE(pace, 50.0) AS pace,
                            COALESCE(shooting, 50.0) AS shooting,
                            COALESCE(passing, 50.0) AS passing,
                            COALESCE(dribbling, 50.0) AS dribbling,
                            COALESCE(defending, 50.0) AS defending,
                            COALESCE(physic, 50.0) AS physic,
                            LAG(overall) OVER (PARTITION BY player_id ORDER BY fifa_edition_year ASC) AS lag_overall_1,
                            LAG(fifa_edition_year) OVER (PARTITION BY player_id ORDER BY fifa_edition_year ASC) AS lag_year_1,
                            LEAD(overall) OVER (PARTITION BY player_id ORDER BY fifa_edition_year ASC) AS target_overall,
                            LEAD(fifa_edition_year) OVER (PARTITION BY player_id ORDER BY fifa_edition_year ASC) AS target_edition_year
                        FROM main.stg_fifa_granular
                    )
                    SELECT
                        player_id,
                        edition_year,
                        short_name,
                        age,
                        overall,
                        potential,
                        potential_diff,
                        pace,
                        shooting,
                        passing,
                        dribbling,
                        defending,
                        physic,
                        lag_overall_1,
                        overall - lag_overall_1 AS growth_rate_1y,
                        target_edition_year,
                        target_overall,
                        target_overall - overall AS target_growth
                    FROM player_seasons
                    WHERE lag_year_1 = edition_year - 1
                      AND target_edition_year = edition_year + 1
                      AND target_overall IS NOT NULL
                      AND lag_overall_1 IS NOT NULL
                      AND age BETWEEN 15 AND 50
                      AND overall BETWEEN 35.0 AND 99.0
                      AND potential BETWEEN 35.0 AND 99.0
                    ORDER BY player_id, edition_year
                    {limit_clause};
                """
            else:
                # Production Gold fact table extraction across FIFA 15 to FC 24
                query = f"""
                    WITH player_seasons AS (
                        SELECT
                            player_id,
                            short_name,
                            fifa_edition_year AS edition_year,
                            age,
                            overall,
                            potential,
                            potential_gap AS potential_diff,
                            COALESCE(pace, 50.0) AS pace,
                            COALESCE(shooting, 50.0) AS shooting,
                            COALESCE(passing, 50.0) AS passing,
                            COALESCE(dribbling, 50.0) AS dribbling,
                            COALESCE(defending, 50.0) AS defending,
                            COALESCE(physic, 50.0) AS physic,
                            prev_overall AS lag_overall_1,
                            prev_fifa_edition_year AS lag_year_1,
                            LEAD(overall) OVER (PARTITION BY player_id ORDER BY fifa_edition_year ASC) AS target_overall,
                            LEAD(fifa_edition_year) OVER (PARTITION BY player_id ORDER BY fifa_edition_year ASC) AS target_edition_year
                        FROM main.fct_player_seasons
                    )
                    SELECT
                        player_id,
                        edition_year,
                        short_name,
                        age,
                        overall,
                        potential,
                        potential_diff,
                        pace,
                        shooting,
                        passing,
                        dribbling,
                        defending,
                        physic,
                        lag_overall_1,
                        overall - lag_overall_1 AS growth_rate_1y,
                        target_edition_year,
                        target_overall,
                        target_overall - overall AS target_growth
                    FROM player_seasons
                    WHERE lag_year_1 = edition_year - 1
                      AND target_edition_year = edition_year + 1
                      AND target_overall IS NOT NULL
                      AND lag_overall_1 IS NOT NULL
                      AND age BETWEEN 15 AND 50
                      AND overall BETWEEN 35.0 AND 99.0
                      AND potential BETWEEN 35.0 AND 99.0
                    ORDER BY player_id, edition_year
                    {limit_clause};
                """

            df = con.execute(query).df()
            logger.info("Extracted %d trajectory career rows from %s", len(df), source_table)
            return df
        finally:
            con.close()

    def validate_contract(self, df: pd.DataFrame, require_target: bool = True) -> None:
        """Validate dataframe matrix against Pydantic schema and zero-leakage contract."""
        TrajectoryFeatureContract.validate_features(df, require_target=require_target)

    def export_parquet(
        self,
        df: pd.DataFrame,
        destination_path: Path | str | None = None,
    ) -> Path:
        """Serialize feature matrix to compressed Parquet file."""
        target = Path(destination_path) if destination_path else self.cfg.output_path
        target.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(target, index=False, compression="snappy")
        logger.info("Exported feature matrix (%d rows) to %s", len(df), target)
        return target

    def get_temporal_splits(
        self,
        df: pd.DataFrame,
        train_end_year: int = 2021,
        val_year: int = 2022,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split feature matrix chronologically without forward-looking data leakage."""
        train_df = df[df["edition_year"] <= train_end_year].copy()
        val_df = df[df["edition_year"] == val_year].copy()
        test_df = df[df["edition_year"] > val_year].copy()
        return train_df, val_df, test_df
