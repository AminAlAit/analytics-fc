"""Tests for dbt gold dimensional marts dim_players and fct_player_seasons.

Verifies table creation, primary key uniqueness on dim_players (53,111 rows),
exact fact record volume on fct_player_seasons (180,021 rows),
foreign key referential integrity, and lifetime career metrics.
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


def test_marts_tables_exist(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Verify dim_players and fct_player_seasons tables exist in DuckDB schema."""
    tables = [
        row[0]
        for row in duckdb_con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
    ]
    assert "dim_players" in tables
    assert "fct_player_seasons" in tables


def test_dim_players_volume_and_uniqueness(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Verify dim_players has exactly 53,111 unique player records."""
    stats = duckdb_con.execute("""
        SELECT
            count(1),
            count(DISTINCT player_id)
        FROM main.dim_players
    """).fetchone()

    assert stats is not None
    total_count, distinct_count = stats
    assert total_count == 53111
    assert distinct_count == 53111


def test_fct_player_seasons_record_volume(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Verify fct_player_seasons preserves exact verified volume of 180,021 records."""
    count = duckdb_con.execute("SELECT count(1) FROM main.fct_player_seasons").fetchone()

    assert count is not None
    assert count[0] == 180021


def test_foreign_key_referential_integrity(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Verify zero orphan records in fct_player_seasons without matching dim_players parent."""
    orphans = duckdb_con.execute("""
        SELECT count(1)
        FROM main.fct_player_seasons f
        LEFT JOIN main.dim_players d
            ON f.player_id = d.player_id
        WHERE d.player_id IS NULL
    """).fetchone()

    assert orphans is not None
    assert orphans[0] == 0


def test_dim_players_lifetime_growth_calculation(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Verify lifetime_growth equals career_peak_overall minus debut_overall."""
    mismatches = duckdb_con.execute("""
        SELECT count(1)
        FROM main.dim_players
        WHERE lifetime_growth != (career_peak_overall - debut_overall)
    """).fetchone()

    assert mismatches is not None
    assert mismatches[0] == 0


def test_superstar_grain_and_linkage(duckdb_con: duckdb.DuckDBPyConnection) -> None:
    """Verify star player entity linkage for Lionel Messi (158023)."""
    dim_row = duckdb_con.execute("""
        SELECT player_id, short_name, total_seasons_recorded, career_peak_overall
        FROM main.dim_players
        WHERE player_id = 158023
    """).fetchone()

    assert dim_row is not None
    assert dim_row[0] == 158023
    assert dim_row[2] == 10
    assert dim_row[3] >= 93

    fact_count = duckdb_con.execute("""
        SELECT count(1)
        FROM main.fct_player_seasons
        WHERE player_id = 158023
    """).fetchone()

    assert fact_count is not None
    assert fact_count[0] == 10
