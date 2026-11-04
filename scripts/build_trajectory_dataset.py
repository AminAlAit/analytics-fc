"""Extract multi-season longitudinal career trajectory dataset from DuckDB.

Generates paired observations (X_t, y_{t+1}) tracking player development from youth
to prime across FIFA editions 15 through 22. Validates output rows against
TrajectoryFeatureContract and exports to Parquet.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.models.trajectory_predictor import (
    FeatureContractError,
    TrajectoryFeatureContract,
)

logger = logging.getLogger("analytics_fc.scripts.build_trajectory_dataset")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DUCKDB_PATH = BASE_DIR / "dbt_project" / "analytics_fc.duckdb"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "trajectory_training.parquet"


def build_longitudinal_trajectory_dataset(
    db_path: Path | str | None = None,
    output_path: Path | str | None = None,
    sample_limit: int | None = None,
) -> pd.DataFrame:
    """Query DuckDB for multi-season trajectories, validate contract, and export parquet."""
    database_path = Path(db_path) if db_path else DEFAULT_DUCKDB_PATH
    target_path = Path(output_path) if output_path else DEFAULT_OUTPUT_PATH

    if not database_path.exists():
        raise FileNotFoundError(f"DuckDB database not found at {database_path}")

    logger.info("Connecting to DuckDB at %s", database_path)
    con = duckdb.connect(str(database_path), read_only=True)
    try:
        con.execute("SET threads = 1;")
        con.execute("SET memory_limit = '1GB';")

        limit_clause = f"LIMIT {int(sample_limit)}" if sample_limit and sample_limit > 0 else ""

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
                    LAG(overall) OVER (PARTITION BY player_id ORDER BY fifa_edition_year) AS lag_overall_1,
                    LAG(fifa_edition_year) OVER (PARTITION BY player_id ORDER BY fifa_edition_year) AS lag_year_1,
                    LEAD(overall) OVER (PARTITION BY player_id ORDER BY fifa_edition_year) AS target_overall,
                    LEAD(fifa_edition_year) OVER (PARTITION BY player_id ORDER BY fifa_edition_year) AS target_edition_year
                FROM stg_fifa_granular
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
    finally:
        con.close()

    logger.info("Extracted %d raw candidate rows from DuckDB", len(df))

    # Validate against Model 2 Longitudinal Feature Contract
    try:
        TrajectoryFeatureContract.validate_features(df, require_target=True)
    except FeatureContractError as err:
        logger.error("Trajectory contract validation failed: %s", err)
        raise

    # Ensure zero null values in target and features
    null_count = int(df[TrajectoryFeatureContract.TARGET_COLUMN].isna().sum())
    if null_count > 0:
        raise ValueError(f"Found {null_count} null target values in dataset.")

    # Export to Parquet
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target_path, index=False, engine="pyarrow")
    logger.info(
        "Successfully exported %d longitudinal career paths (%d unique players) to %s",
        len(df),
        df["player_id"].nunique(),
        target_path,
    )

    return df


def main() -> None:
    """CLI entrypoint for building the longitudinal trajectory training parquet."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Build Model 2 longitudinal trajectory dataset.")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DUCKDB_PATH))
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--limit", type=int, default=None, help="Optional row extraction limit")
    args = parser.parse_args()

    df = build_longitudinal_trajectory_dataset(
        db_path=args.db_path,
        output_path=args.output_path,
        sample_limit=args.limit,
    )

    print("\nLongitudinal Career Trajectory Dataset Summary:")
    print(f"- Total Multi-Year Observations: {len(df):,}")
    print(f"- Unique Players: {df['player_id'].nunique():,}")
    print(f"- Age Span: {df['age'].min()} to {df['age'].max()} years")
    print(f"- Target Growth Mean: {df['target_growth'].mean():+.2f} points")
    print(f"- Target Growth Std: {df['target_growth'].std():.2f} points")
    print(f"- Parquet Artifact: {args.output_path}\n")


if __name__ == "__main__":
    main()
