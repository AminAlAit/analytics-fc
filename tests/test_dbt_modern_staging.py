"""Tests for dbt modern staging model stg_modern_players.

Verifies DuckDB view creation, 180,021 record count integrity,
seasonal span (2015 to 2024), column casting, and schema contract adherence.
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


def test_stg_modern_players_exists(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Verify stg_modern_players view exists in DuckDB schema."""
    tables = [
        row[0]
        for row in duckdb_con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
    ]
    assert "stg_modern_players" in tables


def test_stg_modern_players_record_count(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Verify stg_modern_players returns exactly 180,021 records."""
    res = duckdb_con.execute("SELECT count(1) FROM main.stg_modern_players").fetchone()
    assert res is not None
    assert res[0] == 180021


def test_stg_modern_players_editions(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Verify seasons cover FIFA 15 (2015) through FC 24 (2024)."""
    editions = [
        row[0]
        for row in duckdb_con.execute(
            "SELECT DISTINCT fifa_edition_year FROM main.stg_modern_players ORDER BY 1"
        ).fetchall()
    ]
    expected_years = list(range(2015, 2025))
    assert editions == expected_years


def test_stg_modern_players_null_constraints(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Verify key primary attributes have zero nulls."""
    null_counts = duckdb_con.execute("""
        SELECT
            sum(CASE WHEN player_id IS NULL THEN 1 ELSE 0 END),
            sum(CASE WHEN fifa_edition_year IS NULL THEN 1 ELSE 0 END),
            sum(CASE WHEN overall IS NULL THEN 1 ELSE 0 END),
            sum(CASE WHEN potential IS NULL THEN 1 ELSE 0 END)
        FROM main.stg_modern_players
    """).fetchone()

    assert null_counts == (0, 0, 0, 0)


def test_stg_modern_players_rating_ranges(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Verify overall and potential ratings adhere to FIFA domain limits."""
    stats = duckdb_con.execute("""
        SELECT
            min(overall), max(overall),
            min(potential), max(potential)
        FROM main.stg_modern_players
    """).fetchone()

    assert stats is not None
    min_ovr, max_ovr, min_pot, max_pot = stats
    assert min_ovr >= 35
    assert max_ovr <= 99
    assert min_pot >= 35
    assert max_pot <= 99
