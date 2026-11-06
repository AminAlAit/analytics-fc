"""Tests for dbt singular data quality and integrity tests.

Verifies career timeline monotonicity, non-decreasing age trajectories,
and zero orphaned seasonal fact records across the 180,021 record lakehouse.
"""

from __future__ import annotations

import duckdb
import pytest

from pipelines.config import config


@pytest.fixture(scope="module")
def duckdb_con():
    """Create read-only connection to project DuckDB instance."""
    db_path = config.dbt_project_dir / "analytics_fc.duckdb"
    con = duckdb.connect(str(db_path), read_only=True)
    yield con
    con.close()


def test_singular_test_files_exist() -> None:
    """Verify singular dbt test SQL files exist on disk."""
    tests_dir = config.dbt_project_dir / "tests"
    assert (tests_dir / "assert_career_timeline_monotonic.sql").exists()
    assert (tests_dir / "assert_no_orphan_seasons.sql").exists()


def test_assert_career_timeline_monotonic_passes(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Verify assert_career_timeline_monotonic returns 0 violations on warehouse data."""
    sql_path = config.dbt_project_dir / "tests" / "assert_career_timeline_monotonic.sql"
    sql_content = sql_path.read_text(encoding="utf-8")
    # Replace dbt ref with DuckDB relation name
    executable_sql = sql_content.replace(
        "{{ ref('fct_player_seasons') }}", "main.fct_player_seasons"
    )

    violations = duckdb_con.execute(executable_sql).fetchall()
    assert len(violations) == 0


def test_assert_no_orphan_seasons_passes(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Verify assert_no_orphan_seasons returns 0 violations on warehouse data."""
    sql_path = config.dbt_project_dir / "tests" / "assert_no_orphan_seasons.sql"
    sql_content = sql_path.read_text(encoding="utf-8")
    # Replace dbt refs with DuckDB relation names
    executable_sql = sql_content.replace(
        "{{ ref('fct_player_seasons') }}", "main.fct_player_seasons"
    ).replace("{{ ref('dim_players') }}", "main.dim_players")

    orphans = duckdb_con.execute(executable_sql).fetchall()
    assert len(orphans) == 0


def test_timeline_monotonicity_logic_catches_synthetic_violation() -> None:
    """Verify the monotonicity logic detects an artificial backward jump in age."""
    mem_con = duckdb.connect(":memory:")
    mem_con.execute("""
        CREATE TABLE test_seasons (
            player_id INTEGER,
            fifa_edition_year INTEGER,
            prev_fifa_edition_year INTEGER,
            career_season_index INTEGER,
            age INTEGER,
            date_of_birth DATE
        );
        INSERT INTO test_seasons VALUES
            (1, 2021, NULL, 1, 20, '2001-01-01'),
            (1, 2022, 2021, 2, 19, '2001-01-01'); -- Violation: age went from 20 down to 19 with same DOB
    """)

    detection_query = """
        WITH season_lagged AS (
            SELECT
                player_id,
                fifa_edition_year,
                prev_fifa_edition_year,
                career_season_index,
                LAG(career_season_index) OVER (PARTITION BY player_id ORDER BY fifa_edition_year ASC) AS prev_career_season_index,
                age,
                LAG(age) OVER (PARTITION BY player_id ORDER BY fifa_edition_year ASC) AS prev_age,
                date_of_birth,
                LAG(date_of_birth) OVER (PARTITION BY player_id ORDER BY fifa_edition_year ASC) AS prev_date_of_birth
            FROM test_seasons
        )
        SELECT player_id
        FROM season_lagged
        WHERE prev_fifa_edition_year IS NOT NULL
          AND (
              fifa_edition_year <= prev_fifa_edition_year
              OR career_season_index <= prev_career_season_index
              OR (date_of_birth = prev_date_of_birth AND age < prev_age)
          )
    """

    caught = mem_con.execute(detection_query).fetchall()
    assert len(caught) == 1
    assert caught[0][0] == 1
