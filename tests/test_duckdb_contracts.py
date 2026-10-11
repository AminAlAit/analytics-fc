import time
from pathlib import Path
import duckdb
import pytest
from scripts.ceiling_matcher import SKILL_COLS

DB_PATH = Path(__file__).resolve().parent.parent / "dbt_project" / "analytics_fc.duckdb"


@pytest.fixture(scope="module")
def db_connection():
    assert DB_PATH.exists(), f"Database file not found at {DB_PATH}"
    con = duckdb.connect(str(DB_PATH), read_only=True)
    con.execute("SET threads = 1;")
    con.execute("SET memory_limit = '1GB';")
    yield con
    con.close()


def test_duckdb_read_only_contract(db_connection):
    # Ensure mutating commands fail in read_only mode
    with pytest.raises(Exception):
        db_connection.execute("CREATE TABLE test_table_fail AS SELECT 1;")


def test_stg_fifa_granular_cardinality(db_connection):
    row_count = db_connection.execute("SELECT count(*) FROM stg_fifa_granular;").fetchone()[0]
    # Verified facts: stg_fifa_granular contains over 140,000 player seasons
    assert row_count >= 140000, f"Expected at least 140,000 rows in stg_fifa_granular, found {row_count}"


def test_stg_fifa_granular_schema_columns(db_connection):
    cols_df = db_connection.execute("DESCRIBE stg_fifa_granular;").df()
    existing_cols = set(cols_df["column_name"].str.lower())

    mandatory_fields = [
        "player_id", "short_name", "long_name", "age",
        "fifa_edition_year", "overall", "potential", "positions"
    ]
    for field in mandatory_fields:
        assert field in existing_cols, f"Mandatory column '{field}' missing from stg_fifa_granular"

    for skill in SKILL_COLS:
        assert skill in existing_cols, f"Skill column '{skill}' missing from stg_fifa_granular"


def test_duckdb_query_latency_budget(db_connection):
    t0 = time.perf_counter()
    query = """
    SELECT player_id, short_name, overall, potential, positions
    FROM stg_fifa_granular
    WHERE age <= 21 AND overall >= 78
    ORDER BY overall DESC
    LIMIT 10;
    """
    res = db_connection.execute(query).df()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert len(res) == 10
    assert elapsed_ms < 150.0, f"DuckDB query took {elapsed_ms:.2f}ms, exceeding 150ms latency budget"
