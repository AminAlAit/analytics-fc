"""Seed minimal DuckDB test fixtures for CI pipelines and dbt schema validation.

Creates raw schema tables (fifatable, fifa_granular, tm_stats, tm_trophies)
with deterministic sample records so dbt compile, run, and test can execute
in isolated CI environments without requiring the 2GB Kaggle raw datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BASE_DIR / "dbt_project" / "analytics_fc.duckdb"


def seed_ci_fixtures(db_path: Path | None = None) -> Path:
    target_path = db_path or DEFAULT_DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(target_path))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS raw;")

        # 1. raw.fifatable
        con.execute(
            """
            CREATE OR REPLACE TABLE raw.fifatable (
                player_id INTEGER,
                Fifa_year INTEGER,
                age INTEGER,
                short_name VARCHAR,
                long_name VARCHAR,
                nationality VARCHAR,
                club VARCHAR,
                position VARCHAR,
                overall INTEGER,
                max_potential INTEGER,
                minutes INTEGER,
                goals INTEGER
            );
            """
        )

        # Multi-season trajectory records for trajectory calculations
        for pid, name in [(1, "Player One"), (2, "Player Two")]:
            for yr, age, ovr, pot, mins, gls in [
                (2018, 19, 70, 85, 1000, 5),
                (2019, 20, 75, 86, 1500, 10),
                (2020, 21, 80, 88, 2000, 15),
                (2021, 22, 85, 90, 2500, 20),
            ]:
                con.execute(
                    """
                    INSERT INTO raw.fifatable VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    [
                        pid,
                        yr,
                        age,
                        name,
                        f"{name} Long",
                        "England",
                        "Arsenal",
                        "ST",
                        ovr,
                        pot,
                        mins,
                        gls,
                    ],
                )

        # 2. raw.fifa_granular
        con.execute(
            """
            CREATE OR REPLACE TABLE raw.fifa_granular (
                player_id INTEGER,
                fifa_year INTEGER,
                short_name VARCHAR,
                long_name VARCHAR,
                age INTEGER,
                dob DATE,
                nationality_name VARCHAR,
                club_name VARCHAR,
                league_name VARCHAR,
                player_positions VARCHAR,
                preferred_foot VARCHAR,
                height_cm INTEGER,
                weight_kg INTEGER,
                overall INTEGER,
                potential INTEGER,
                value_eur DOUBLE,
                wage_eur DOUBLE,
                international_reputation INTEGER,
                weak_foot INTEGER,
                skill_moves INTEGER,
                pace INTEGER,
                shooting INTEGER,
                passing INTEGER,
                dribbling INTEGER,
                defending INTEGER,
                physic INTEGER,
                attacking_crossing INTEGER,
                attacking_finishing INTEGER,
                attacking_heading_accuracy INTEGER,
                attacking_short_passing INTEGER,
                attacking_volleys INTEGER,
                skill_dribbling INTEGER,
                skill_curve INTEGER,
                skill_fk_accuracy INTEGER,
                skill_long_passing INTEGER,
                skill_ball_control INTEGER,
                movement_acceleration INTEGER,
                movement_sprint_speed INTEGER,
                movement_agility INTEGER,
                movement_reactions INTEGER,
                movement_balance INTEGER,
                power_shot_power INTEGER,
                power_jumping INTEGER,
                power_stamina INTEGER,
                power_strength INTEGER,
                power_long_shots INTEGER,
                mentality_aggression INTEGER,
                mentality_interceptions INTEGER,
                mentality_positioning INTEGER,
                mentality_vision INTEGER,
                mentality_penalties INTEGER,
                mentality_composure INTEGER,
                defending_marking_awareness INTEGER,
                defending_standing_tackle INTEGER,
                defending_sliding_tackle INTEGER,
                goalkeeping_diving INTEGER,
                goalkeeping_handling INTEGER,
                goalkeeping_kicking INTEGER,
                goalkeeping_positioning INTEGER,
                goalkeeping_reflexes INTEGER
            );
            """
        )

        con.execute(
            """
            INSERT INTO raw.fifa_granular VALUES (
                1, 2021, 'Player One', 'Player One Long', 22, '1999-01-01', 'England', 'Arsenal',
                'Premier League', 'ST, CF', 'Right', 180, 75,
                85, 90, 50000000.0, 100000.0, 3, 4, 4,
                85, 85, 80, 85, 40, 75,
                75, 85, 80, 80, 75, 85, 75, 70, 75, 85,
                85, 85, 80, 85, 75, 85, 75, 80, 75, 80,
                65, 40, 85, 80, 75, 80, 40, 40, 35,
                10, 10, 10, 10, 10
            );
            """
        )

        # 3. raw.tm_stats
        con.execute(
            """
            CREATE OR REPLACE TABLE raw.tm_stats (
                FifaIndex INTEGER,
                short_name VARCHAR,
                competition VARCHAR,
                club VARCHAR,
                games_played INTEGER,
                goals INTEGER,
                assists INTEGER,
                minutes INTEGER
            );
            """
        )
        con.execute(
            """
            INSERT INTO raw.tm_stats VALUES (1, 'Player One', 'Premier League', 'Arsenal', 38, 20, 10, 2500);
            """
        )

        # 4. raw.tm_trophies
        con.execute(
            """
            CREATE OR REPLACE TABLE raw.tm_trophies (
                short_name VARCHAR,
                club VARCHAR,
                team_trophy INTEGER,
                individual_trophy INTEGER,
                trophy VARCHAR,
                year INTEGER
            );
            """
        )
        con.execute(
            """
            INSERT INTO raw.tm_trophies VALUES ('Player One', 'Arsenal', 1, 0, 'FA Cup', 2020);
            """
        )

    finally:
        con.close()

    print(f"CI test fixtures successfully seeded to {target_path}")
    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed minimal DuckDB test fixtures for CI.")
    parser.add_argument(
        "--db-path", type=Path, default=DEFAULT_DB_PATH, help="Destination DuckDB path."
    )
    args = parser.parse_args()
    seed_ci_fixtures(db_path=args.db_path)


if __name__ == "__main__":
    main()
