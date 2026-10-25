"""DuckDB catalog search engine for Analytics FC.

Provides parameterized querying, player discovery, and profile retrieval across
the complete 180,021 historical player dataset with hardware safety guards.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import duckdb

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "dbt_project" / "analytics_fc.duckdb"

CORE_SEARCH_COLUMNS = [
    "player_id",
    "fifa_edition_year",
    "short_name",
    "long_name",
    "age",
    "nationality",
    "club",
    "league_name",
    "positions",
    "preferred_foot",
    "height_cm",
    "weight_kg",
    "overall",
    "potential",
    "value_eur",
    "wage_eur",
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",
]

ALL_SKILL_COLUMNS = [
    "attacking_crossing",
    "attacking_finishing",
    "attacking_heading_accuracy",
    "attacking_short_passing",
    "attacking_volleys",
    "skill_dribbling",
    "skill_curve",
    "skill_fk_accuracy",
    "skill_long_passing",
    "skill_ball_control",
    "movement_acceleration",
    "movement_sprint_speed",
    "movement_agility",
    "movement_reactions",
    "movement_balance",
    "power_shot_power",
    "power_jumping",
    "power_stamina",
    "power_strength",
    "power_long_shots",
    "mentality_aggression",
    "mentality_interceptions",
    "mentality_positioning",
    "mentality_vision",
    "mentality_penalties",
    "mentality_composure",
    "defending_marking_awareness",
    "defending_standing_tackle",
    "defending_sliding_tackle",
    "goalkeeping_diving",
    "goalkeeping_handling",
    "goalkeeping_kicking",
    "goalkeeping_positioning",
    "goalkeeping_reflexes",
]


def resolve_db_path(db_path: Path | str | None = None) -> Path:
    """Resolve and validate the DuckDB database file path."""
    resolved = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not resolved.exists():
        raise FileNotFoundError(f"DuckDB database not found at {resolved}")
    return resolved


def get_connection(db_path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """Open a read-only DuckDB connection with strict hardware resource limits."""
    path = resolve_db_path(db_path)
    con = duckdb.connect(str(path), read_only=True)
    con.execute("SET threads = 1;")
    con.execute("SET memory_limit = '1GB';")
    return con


def get_catalog_stats(
    db_path: Path | str | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> dict[str, int]:
    """Retrieve verified total row count and distinct player count from the catalog."""
    should_close = con is None
    active_con = con if con is not None else get_connection(db_path)
    try:
        row = active_con.execute(
            """
            SELECT 
                COUNT(*) AS total_rows,
                COUNT(DISTINCT player_id) AS distinct_players,
                MIN(fifa_edition_year) AS min_year,
                MAX(fifa_edition_year) AS max_year
            FROM stg_fifa_granular;
            """
        ).fetchone()
        return {
            "total_rows": int(row[0]),
            "distinct_players": int(row[1]),
            "min_year": int(row[2]),
            "max_year": int(row[3]),
        }
    finally:
        if should_close:
            active_con.close()


def search_players(
    query: str,
    edition_year: int | None = None,
    min_overall: int | None = None,
    max_overall: int | None = None,
    limit: int = 50,
    offset: int = 0,
    db_path: Path | str | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> list[dict[str, Any]]:
    """Execute a parameterized search for players matching a name query and optional filters.

    Args:
        query: Substring to search across short_name and long_name.
        edition_year: Optional FIFA edition year filter (e.g. 2024).
        min_overall: Optional minimum overall rating threshold.
        max_overall: Optional maximum overall rating threshold.
        limit: Maximum number of rows to return (default 50).
        offset: Number of initial matching rows to skip (default 0).
        db_path: Optional path override for DuckDB database file.
        con: Optional active DuckDB connection to reuse across calls.

    Returns:
        List of player record dictionaries sorted by overall DESC, potential DESC.
    """
    clean_query = query.strip()
    where_clauses: list[str] = []
    params: list[Any] = []

    if clean_query:
        pattern = f"%{clean_query}%"
        where_clauses.append("(short_name ILIKE ? OR long_name ILIKE ?)")
        params.extend([pattern, pattern])

    if edition_year is not None:
        where_clauses.append("fifa_edition_year = ?")
        params.append(edition_year)

    if min_overall is not None:
        where_clauses.append("overall >= ?")
        params.append(min_overall)

    if max_overall is not None:
        where_clauses.append("overall <= ?")
        params.append(max_overall)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    select_cols = ", ".join(CORE_SEARCH_COLUMNS)

    sql = f"""
        SELECT {select_cols}
        FROM stg_fifa_granular
        {where_sql}
        ORDER BY overall DESC, potential DESC, short_name ASC
        LIMIT ? OFFSET ?;
    """
    params.extend([max(1, limit), max(0, offset)])

    should_close = con is None
    active_con = con if con is not None else get_connection(db_path)
    try:
        cursor = active_con.execute(sql, params)
        col_names = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(col_names, row)) for row in rows]
    finally:
        if should_close:
            active_con.close()


def get_player_by_id_and_year(
    player_id: int,
    fifa_edition_year: int,
    db_path: Path | str | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> dict[str, Any] | None:
    """Retrieve complete player profile including all 34 tactical skill attributes.

    Args:
        player_id: Canonical player identifier.
        fifa_edition_year: Specific FIFA edition year.
        db_path: Optional path override for DuckDB database file.
        con: Optional active DuckDB connection to reuse across calls.

    Returns:
        Dictionary of complete player attributes, or None if not found.
    """
    all_columns = CORE_SEARCH_COLUMNS + ALL_SKILL_COLUMNS
    select_cols = ", ".join(all_columns)

    sql = f"""
        SELECT {select_cols}
        FROM stg_fifa_granular
        WHERE player_id = ? AND fifa_edition_year = ?
        LIMIT 1;
    """

    should_close = con is None
    active_con = con if con is not None else get_connection(db_path)
    try:
        cursor = active_con.execute(sql, [player_id, fifa_edition_year])
        col_names = [desc[0] for desc in cursor.description]
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(zip(col_names, row))
    finally:
        if should_close:
            active_con.close()
