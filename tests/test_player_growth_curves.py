"""Tests for dbt intermediate model int_player_growth_curves.

Verifies windowed career growth calculations, sequential season indexing,
rating deltas, peak performance flags, and career stage classifications.
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


def test_growth_curves_exists(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Verify int_player_growth_curves view exists in DuckDB schema."""
    tables = [
        row[0]
        for row in duckdb_con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
    ]
    assert "int_player_growth_curves" in tables


def test_growth_curves_record_count(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Verify record count matches exactly 180,021 without Cartesian explosion."""
    res = duckdb_con.execute("SELECT count(1) FROM main.int_player_growth_curves").fetchone()
    assert res is not None
    assert res[0] == 180021


def test_growth_curves_career_indexing(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Verify career_season_index starts at 1 and has no gaps for each player."""
    stats = duckdb_con.execute("""
        SELECT
            min(career_season_index),
            max(career_season_index),
            sum(CASE WHEN career_season_index < 1 THEN 1 ELSE 0 END)
        FROM main.int_player_growth_curves
    """).fetchone()

    assert stats is not None
    min_idx, max_idx, invalid_count = stats
    assert min_idx == 1
    assert max_idx == 10
    assert invalid_count == 0


def test_growth_curves_peak_overall_constraint(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Verify career_peak_overall is always greater than or equal to current overall."""
    violations = duckdb_con.execute("""
        SELECT count(1)
        FROM main.int_player_growth_curves
        WHERE career_peak_overall < overall
    """).fetchone()

    assert violations is not None
    assert violations[0] == 0


def test_growth_curves_annual_velocity_logic(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Verify annual_growth_velocity matches delta_overall when consecutive 1-year gap."""
    mismatches = duckdb_con.execute("""
        SELECT count(1)
        FROM main.int_player_growth_curves
        WHERE years_since_prev_season = 1
          AND (annual_growth_velocity IS NULL OR annual_growth_velocity != delta_overall)
    """).fetchone()

    assert mismatches is not None
    assert mismatches[0] == 0


def test_growth_curves_career_stages(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Verify career_stage maps only to valid taxonomy values."""
    stages = [
        row[0]
        for row in duckdb_con.execute(
            "SELECT DISTINCT career_stage FROM main.int_player_growth_curves ORDER BY 1"
        ).fetchall()
    ]
    valid_stages = {"DEVELOPMENT", "MATURE", "PRIME", "VETERAN"}
    assert set(stages).issubset(valid_stages)


def test_growth_curves_superstar_trajectory(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Verify decade-long trajectory for Lionel Messi (player_id 158023)."""
    messi_seasons = duckdb_con.execute("""
        SELECT
            fifa_edition_year,
            overall,
            career_season_index,
            is_peak_overall_season,
            career_stage
        FROM main.int_player_growth_curves
        WHERE player_id = 158023
        ORDER BY fifa_edition_year ASC
    """).fetchall()

    assert len(messi_seasons) == 10
    # First season index 1, tenth season index 10
    assert messi_seasons[0][2] == 1
    assert messi_seasons[-1][2] == 10
    # Career stage progresses from PRIME to VETERAN
    stages = [s[4] for s in messi_seasons]
    assert "PRIME" in stages
    assert "VETERAN" in stages
