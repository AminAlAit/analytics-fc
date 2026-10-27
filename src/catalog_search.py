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

SECTOR_MAP: dict[str, str] = {
    "GK": "Goalkeepers",
    "CB": "Defenders",
    "LB": "Defenders",
    "RB": "Defenders",
    "LWB": "Defenders",
    "RWB": "Defenders",
    "CDM": "Defensive & Central Midfielders",
    "CM": "Defensive & Central Midfielders",
    "CAM": "Attacking Midfielders & Wingers",
    "LM": "Attacking Midfielders & Wingers",
    "RM": "Attacking Midfielders & Wingers",
    "LW": "Attacking Midfielders & Wingers",
    "RW": "Attacking Midfielders & Wingers",
    "CF": "Center Forwards & Strikers",
    "ST": "Center Forwards & Strikers",
}

SECTOR_ORDER: list[str] = [
    "Goalkeepers",
    "Defenders",
    "Defensive & Central Midfielders",
    "Attacking Midfielders & Wingers",
    "Center Forwards & Strikers",
]


def get_primary_position(positions: str | None) -> str:
    """Extract canonical first position from a comma-separated positions string."""
    if not positions:
        return "ST"
    return str(positions).split(",")[0].strip().upper()


def get_sector_for_position(position: str | None) -> str:
    """Map a football pitch position to its canonical tactical sector."""
    primary = get_primary_position(position)
    return SECTOR_MAP.get(primary, "Attacking Midfielders & Wingers")

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
    match_long_name: bool | None = None,
    db_path: Path | str | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> list[dict[str, Any]]:
    """Execute a parameterized search for players matching a name query and optional filters.

    Args:
        query: Substring to search across player names.
        edition_year: Optional FIFA edition year filter (e.g. 2024).
        min_overall: Optional minimum overall rating threshold.
        max_overall: Optional maximum overall rating threshold.
        limit: Maximum number of rows to return (default 50).
        offset: Number of initial matching rows to skip (default 0).
        match_long_name: Whether to include long_name in search (defaults to True if query contains spaces).
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
        should_match_long = match_long_name if match_long_name is not None else (" " in clean_query)
        if should_match_long:
            where_clauses.append("(short_name ILIKE ? OR long_name ILIKE ?)")
            params.extend([pattern, pattern])
        else:
            where_clauses.append("short_name ILIKE ?")
            params.append(pattern)

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
        ORDER BY overall DESC, potential DESC, fifa_edition_year DESC, player_id ASC
        LIMIT ? OFFSET ?;
    """
    params.extend([max(1, limit), max(0, offset)])

    should_close = con is None
    active_con = con if con is not None else get_connection(db_path)
    try:
        cursor = active_con.execute(sql, params)
        col_names = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            record = dict(zip(col_names, row))
            record["primary_pos"] = get_primary_position(record.get("positions"))
            record["tactical_sector"] = get_sector_for_position(record.get("positions"))
            result.append(record)
        return result
    finally:
        if should_close:
            active_con.close()


def search_players_paginated(
    query: str = "",
    edition_year: int | None = None,
    min_overall: int | None = None,
    max_overall: int | None = None,
    sector: str | None = None,
    limit: int = 50,
    offset: int = 0,
    include_sector_counts: bool = True,
    match_long_name: bool | None = None,
    db_path: Path | str | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> dict[str, Any]:
    """Execute a faceted, paginated player search across the catalog.

    Args:
        query: Substring to search across player names.
        edition_year: Optional FIFA edition year filter (e.g. 2022).
        min_overall: Optional minimum overall rating threshold.
        max_overall: Optional maximum overall rating threshold.
        sector: Optional tactical sector filter (e.g. 'Defenders').
        limit: Maximum number of rows to return per page (default 50).
        offset: Number of initial matching rows to skip (default 0).
        include_sector_counts: Whether to compute sector breakdown counts.
        match_long_name: Whether to include long_name in search (defaults to True if query contains spaces).
        db_path: Optional path override for DuckDB database file.
        con: Optional active DuckDB connection to reuse across calls.

    Returns:
        Dictionary containing:
            - items: List of player record dictionaries with primary_pos and tactical_sector
            - total_matching: Total count of matching records across all pages
            - limit: Current page size limit
            - offset: Current page offset
            - has_more: Boolean indicating if additional matching records exist
            - sector_counts: Dictionary mapping each tactical sector to count of matches
    """
    clean_query = query.strip()
    base_clauses: list[str] = []
    base_params: list[Any] = []

    if clean_query:
        pattern = f"%{clean_query}%"
        should_match_long = match_long_name if match_long_name is not None else (" " in clean_query)
        if should_match_long:
            base_clauses.append("(short_name ILIKE ? OR long_name ILIKE ?)")
            base_params.extend([pattern, pattern])
        else:
            base_clauses.append("short_name ILIKE ?")
            base_params.append(pattern)

    if edition_year is not None:
        base_clauses.append("fifa_edition_year = ?")
        base_params.append(edition_year)

    if min_overall is not None:
        base_clauses.append("overall >= ?")
        base_params.append(min_overall)

    if max_overall is not None:
        base_clauses.append("overall <= ?")
        base_params.append(max_overall)

    item_clauses = list(base_clauses)
    item_params = list(base_params)

    if sector is not None:
        sector_positions = [
            pos for pos, sec in SECTOR_MAP.items() if sec.lower() == sector.strip().lower()
        ]
        if sector_positions:
            placeholders = ", ".join(["?" for _ in sector_positions])
            item_clauses.append(f"TRIM(SPLIT_PART(positions, ',', 1)) IN ({placeholders})")
            item_params.extend(sector_positions)

    where_sql = f"WHERE {' AND '.join(item_clauses)}" if item_clauses else ""
    select_cols = ", ".join(CORE_SEARCH_COLUMNS)

    sql = f"""
        SELECT {select_cols}, COUNT(*) OVER() AS total_matching
        FROM stg_fifa_granular
        {where_sql}
        ORDER BY overall DESC, potential DESC, fifa_edition_year DESC, player_id ASC
        LIMIT ? OFFSET ?;
    """
    safe_limit = max(1, limit)
    safe_offset = max(0, offset)
    item_params.extend([safe_limit, safe_offset])

    should_close = con is None
    active_con = con if con is not None else get_connection(db_path)
    try:
        cursor = active_con.execute(sql, item_params)
        col_names = [desc[0] for desc in cursor.description]
        raw_rows = cursor.fetchall()

        items: list[dict[str, Any]] = []
        total_matching = 0

        for row in raw_rows:
            record = dict(zip(col_names, row))
            total_matching = int(record.pop("total_matching", 0))
            record["primary_pos"] = get_primary_position(record.get("positions"))
            record["tactical_sector"] = get_sector_for_position(record.get("positions"))
            items.append(record)

        if not items:
            if safe_offset == 0:
                total_matching = 0
            else:
                count_where = f"WHERE {' AND '.join(item_clauses)}" if item_clauses else ""
                count_params = item_params[:-2]
                count_res = active_con.execute(
                    f"SELECT COUNT(*) FROM stg_fifa_granular {count_where};",
                    count_params,
                ).fetchone()
                total_matching = int(count_res[0]) if count_res else 0

        has_more = (safe_offset + len(items)) < total_matching

        sector_counts: dict[str, int] = {sec: 0 for sec in SECTOR_ORDER}
        if include_sector_counts:
            base_where = f"WHERE {' AND '.join(base_clauses)}" if base_clauses else ""
            sector_sql = f"""
                SELECT 
                    CASE
                        WHEN TRIM(SPLIT_PART(positions, ',', 1)) = 'GK' THEN 'Goalkeepers'
                        WHEN TRIM(SPLIT_PART(positions, ',', 1)) IN ('CB', 'LB', 'RB', 'LWB', 'RWB') THEN 'Defenders'
                        WHEN TRIM(SPLIT_PART(positions, ',', 1)) IN ('CDM', 'CM') THEN 'Defensive & Central Midfielders'
                        WHEN TRIM(SPLIT_PART(positions, ',', 1)) IN ('CAM', 'LM', 'RM', 'LW', 'RW') THEN 'Attacking Midfielders & Wingers'
                        WHEN TRIM(SPLIT_PART(positions, ',', 1)) IN ('CF', 'ST') THEN 'Center Forwards & Strikers'
                        ELSE 'Other'
                    END AS sector,
                    COUNT(*) AS cnt
                FROM stg_fifa_granular
                {base_where}
                GROUP BY sector;
            """
            for sec, cnt in active_con.execute(sector_sql, base_params).fetchall():
                if sec in sector_counts:
                    sector_counts[sec] = int(cnt)

        return {
            "items": items,
            "total_matching": total_matching,
            "limit": safe_limit,
            "offset": safe_offset,
            "has_more": has_more,
            "sector_counts": sector_counts,
        }
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
