"""Seed minimal DuckDB test fixtures for CI pipelines and dbt schema validation.

Creates raw schema tables (fifatable, fifa_granular, granular_15_24, tm_stats,
tm_trophies) with deterministic multi-edition sample records so dbt compile,
run, test, drift checks, and trajectory suites execute cleanly in isolated CI
without requiring the 2GB raw datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BASE_DIR / "dbt_project" / "analytics_fc.duckdb"
CI_PLAYER_COUNT = 600


def seed_ci_fixtures(db_path: Path | None = None) -> Path:
    """Create and populate deterministic raw schema tables for CI verification."""
    target_path = db_path or DEFAULT_DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(target_path))
    try:
        con.execute("SET threads = 1;")
        con.execute("SET memory_limit = '1GB';")
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

        for pid, name in [(1, "Player One"), (2, "Player Two")]:
            for yr, age, ovr, pot, mins, gls in [
                (2018, 19, 70, 85, 1000, 5),
                (2019, 20, 75, 86, 1500, 10),
                (2020, 21, 80, 88, 2000, 15),
                (2021, 22, 85, 90, 2500, 20),
            ]:
                con.execute(
                    "INSERT INTO raw.fifatable VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
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

        # 2. raw.fifa_granular (600 players x 3 consecutive editions = 1,800 rows)
        con.execute(
            f"""
            CREATE OR REPLACE TABLE raw.fifa_granular AS
            SELECT
                p.pid::INTEGER AS player_id,
                yr.fifa_year::INTEGER AS fifa_year,
                ('Player ' || p.pid)::VARCHAR AS short_name,
                ('Player ' || p.pid || ' Long')::VARCHAR AS long_name,
                (19 + (p.pid % 12) + (yr.fifa_year - 2020))::INTEGER AS age,
                DATE '1999-01-01' AS dob,
                CASE WHEN p.pid % 3 = 0 THEN 'France' WHEN p.pid % 3 = 1 THEN 'England' ELSE 'Spain' END AS nationality_name,
                'Arsenal'::VARCHAR AS club_name,
                'Premier League'::VARCHAR AS league_name,
                CASE
                    WHEN p.pid % 10 = 0 THEN 'GK'
                    WHEN p.pid % 3 = 0 THEN 'ST, CF'
                    WHEN p.pid % 3 = 1 THEN 'CM, CDM'
                    ELSE 'CB'
                END AS player_positions,
                CASE WHEN p.pid % 2 = 0 THEN 'Right' ELSE 'Left' END AS preferred_foot,
                (175 + (p.pid % 18))::INTEGER AS height_cm,
                (70 + (p.pid % 16))::INTEGER AS weight_kg,
                (67 + (p.pid % 20) + (yr.fifa_year - 2020))::INTEGER AS overall,
                (78 + (p.pid % 17))::INTEGER AS potential,
                (1000000.0 * (1 + (p.pid % 40)) * (1 + 0.1 * (yr.fifa_year - 2020)))::DOUBLE AS value_eur,
                (15000.0 + 1000.0 * (p.pid % 60))::DOUBLE AS wage_eur,
                (1 + (p.pid % 4))::INTEGER AS international_reputation,
                (2 + (p.pid % 3))::INTEGER AS weak_foot,
                (2 + (p.pid % 3))::INTEGER AS skill_moves,
                (54 + ((p.pid * 3) % 35) + (yr.fifa_year - 2020))::INTEGER AS pace,
                (49 + ((p.pid * 5) % 40) + (yr.fifa_year - 2020))::INTEGER AS shooting,
                (54 + ((p.pid * 7) % 35) + (yr.fifa_year - 2020))::INTEGER AS passing,
                (55 + ((p.pid * 11) % 34) + (yr.fifa_year - 2020))::INTEGER AS dribbling,
                (39 + ((p.pid * 13) % 45) + (yr.fifa_year - 2020))::INTEGER AS defending,
                (57 + ((p.pid * 17) % 32) + (yr.fifa_year - 2020))::INTEGER AS physic,
                (55 + (p.pid % 30))::INTEGER AS attacking_crossing,
                (55 + (p.pid % 32))::INTEGER AS attacking_finishing,
                (52 + (p.pid % 30))::INTEGER AS attacking_heading_accuracy,
                (60 + (p.pid % 28))::INTEGER AS attacking_short_passing,
                (50 + (p.pid % 30))::INTEGER AS attacking_volleys,
                (58 + (p.pid % 30))::INTEGER AS skill_dribbling,
                (54 + (p.pid % 30))::INTEGER AS skill_curve,
                (50 + (p.pid % 30))::INTEGER AS skill_fk_accuracy,
                (56 + (p.pid % 28))::INTEGER AS skill_long_passing,
                (62 + (p.pid % 28))::INTEGER AS skill_ball_control,
                (58 + (p.pid % 32))::INTEGER AS movement_acceleration,
                (58 + (p.pid % 32))::INTEGER AS movement_sprint_speed,
                (58 + (p.pid % 30))::INTEGER AS movement_agility,
                (60 + (p.pid % 28))::INTEGER AS movement_reactions,
                (58 + (p.pid % 30))::INTEGER AS movement_balance,
                (60 + (p.pid % 30))::INTEGER AS power_shot_power,
                (56 + (p.pid % 30))::INTEGER AS power_jumping,
                (60 + (p.pid % 30))::INTEGER AS power_stamina,
                (58 + (p.pid % 30))::INTEGER AS power_strength,
                (54 + (p.pid % 32))::INTEGER AS power_long_shots,
                (55 + (p.pid % 30))::INTEGER AS mentality_aggression,
                (48 + (p.pid % 38))::INTEGER AS mentality_interceptions,
                (52 + (p.pid % 35))::INTEGER AS mentality_positioning,
                (56 + (p.pid % 32))::INTEGER AS mentality_vision,
                (52 + (p.pid % 30))::INTEGER AS mentality_penalties,
                (60 + (p.pid % 28))::INTEGER AS mentality_composure,
                (45 + (p.pid % 40))::INTEGER AS defending_marking_awareness,
                (46 + (p.pid % 40))::INTEGER AS defending_standing_tackle,
                (44 + (p.pid % 40))::INTEGER AS defending_sliding_tackle,
                CASE WHEN p.pid % 10 = 0 THEN 82 ELSE 10 END::INTEGER AS goalkeeping_diving,
                CASE WHEN p.pid % 10 = 0 THEN 80 ELSE 10 END::INTEGER AS goalkeeping_handling,
                CASE WHEN p.pid % 10 = 0 THEN 78 ELSE 10 END::INTEGER AS goalkeeping_kicking,
                CASE WHEN p.pid % 10 = 0 THEN 81 ELSE 10 END::INTEGER AS goalkeeping_positioning,
                CASE WHEN p.pid % 10 = 0 THEN 84 ELSE 10 END::INTEGER AS goalkeeping_reflexes
            FROM range(1, {CI_PLAYER_COUNT + 1}) AS p(pid)
            CROSS JOIN (VALUES (2020), (2021), (2022)) AS yr(fifa_year);
            """
        )

        # 3. raw.granular_15_24 (backing stg_modern_players -> dim_players & fct_player_seasons)
        con.execute(
            """
            CREATE OR REPLACE TABLE raw.granular_15_24 AS
            SELECT
                player_id,
                (fifa_year - 2000)::INTEGER AS fifa_version,
                1::INTEGER AS fifa_update,
                short_name,
                long_name,
                age,
                dob,
                nationality_name,
                14::INTEGER AS nationality_id,
                club_name,
                1::INTEGER AS club_team_id,
                league_name,
                13::INTEGER AS league_id,
                1::INTEGER AS league_level,
                player_positions,
                preferred_foot,
                height_cm,
                weight_kg,
                overall,
                potential,
                value_eur,
                wage_eur,
                (value_eur * 1.5)::DOUBLE AS release_clause_eur,
                international_reputation,
                weak_foot,
                skill_moves,
                pace,
                shooting,
                passing,
                dribbling,
                defending,
                physic,
                attacking_crossing,
                attacking_finishing,
                attacking_heading_accuracy,
                attacking_short_passing,
                attacking_volleys,
                skill_dribbling,
                skill_curve,
                skill_fk_accuracy,
                skill_long_passing,
                skill_ball_control,
                movement_acceleration,
                movement_sprint_speed,
                movement_agility,
                movement_reactions,
                movement_balance,
                power_shot_power,
                power_jumping,
                power_stamina,
                power_strength,
                power_long_shots,
                mentality_aggression,
                mentality_interceptions,
                mentality_positioning,
                mentality_vision,
                mentality_penalties,
                mentality_composure,
                defending_marking_awareness,
                defending_standing_tackle,
                defending_sliding_tackle,
                goalkeeping_diving,
                goalkeeping_handling,
                goalkeeping_kicking,
                goalkeeping_positioning,
                goalkeeping_reflexes,
                CASE WHEN player_id % 10 = 0 THEN 50 ELSE 10 END::INTEGER AS goalkeeping_speed
            FROM raw.fifa_granular;
            """
        )

        # 4. raw.tm_stats
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
            INSERT INTO raw.tm_stats VALUES (1, 'Player 1', 'Premier League', 'Arsenal', 38, 20, 10, 2500);
            """
        )

        # 5. raw.tm_trophies
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
            INSERT INTO raw.tm_trophies VALUES ('Player One', 'Arsenal', 1, 0, 'FA Cup', 2020);
            """
        )

    finally:
        con.close()

    print(f"CI test fixtures successfully seeded to {target_path}")
    return target_path


def main() -> None:
    """Parse CLI args and seed CI DuckDB fixtures."""
    parser = argparse.ArgumentParser(description="Seed minimal DuckDB test fixtures for CI.")
    parser.add_argument(
        "--db-path", type=Path, default=DEFAULT_DB_PATH, help="Destination DuckDB path."
    )
    args = parser.parse_args()
    seed_ci_fixtures(db_path=args.db_path)


if __name__ == "__main__":
    main()
