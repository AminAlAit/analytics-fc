"""Pitch-dark institutional layout and live catalog telemetry HUD.

Counts prefer main.fct_player_seasons plus main.dim_players when that dimension
table exists. Slim serving bundles omit dim_players, so the HUD falls back to
COUNT(*) and COUNT(DISTINCT player_id) on main.fct_player_seasons.
"""

from __future__ import annotations

import html
from pathlib import Path

import duckdb

STYLESHEET_PATH = Path(__file__).resolve().parent.parent / "styles" / "institutional.css"

_EMPTY_COUNTS = {
    "season_rows": 0,
    "distinct_players": 0,
    "source": "unavailable",
}


def load_institutional_css() -> str:
    """Return the institutional stylesheet, wrapped for Streamlit markdown."""
    css = STYLESHEET_PATH.read_text(encoding="utf-8")
    return f"<style>\n{css}\n</style>"


def _relation_exists(con: duckdb.DuckDBPyConnection, schema_name: str, table_name: str) -> bool:
    """Return True when information_schema lists the relation."""
    row = con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = ?
          AND table_name = ?;
        """,
        [schema_name, table_name],
    ).fetchone()
    return bool(row is not None and int(row[0]) > 0)


def query_hud_counts(db_path: Path | str) -> dict[str, int | str]:
    """Read season and player counts with a dim_players fallback.

    Full local warehouses use fct_player_seasons row count and dim_players row
    count. Bundles that only ship fct_player_seasons use distinct player_id.
    If that fact table is absent, stg_fifa_granular is the last fallback.
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"DuckDB database not found at {path}")

    con = duckdb.connect(str(path), read_only=True)
    try:
        con.execute("SET threads = 1;")
        con.execute("SET memory_limit = '1GB';")

        has_fct = _relation_exists(con, "main", "fct_player_seasons")
        has_dim = _relation_exists(con, "main", "dim_players")
        has_stg = _relation_exists(con, "main", "stg_fifa_granular")

        if has_fct and has_dim:
            row = con.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM main.fct_player_seasons),
                    (SELECT COUNT(*) FROM main.dim_players);
                """
            ).fetchone()
            source = "fct_player_seasons+dim_players"
        elif has_fct:
            row = con.execute(
                """
                SELECT COUNT(*), COUNT(DISTINCT player_id)
                FROM main.fct_player_seasons;
                """
            ).fetchone()
            source = "fct_player_seasons"
        elif has_stg:
            row = con.execute(
                """
                SELECT COUNT(*), COUNT(DISTINCT player_id)
                FROM main.stg_fifa_granular;
                """
            ).fetchone()
            source = "stg_fifa_granular"
        else:
            return dict(_EMPTY_COUNTS)

        if row is None:
            return dict(_EMPTY_COUNTS)
        return {
            "season_rows": int(row[0] or 0),
            "distinct_players": int(row[1] or 0),
            "source": source,
        }
    finally:
        con.close()


def render_terminal_bar(
    *,
    status_label: str,
    season_rows: int,
    distinct_players: int,
    catalog_rows: int,
) -> str:
    """Render the executive telemetry HUD markup."""
    safe_status = html.escape(status_label)
    return f"""
    <div class="institutional-frame">
        <div class="terminal-bar">
            <div class="brand-title">
                <span>⚽ ANALYTICS FC</span>
                <span class="brand-accent">// SCOUTING INTELLIGENCE ENGINE</span>
            </div>
            <div class="telemetry-group">
                <span class="telemetry-chip live">● {safe_status}</span>
                <span class="telemetry-chip">{season_rows:,} SEASONS / {distinct_players:,} PLAYERS</span>
                <span class="telemetry-chip live">CATALOG SEARCH: {catalog_rows:,} RECORDS ACTIVE</span>
                <span class="telemetry-chip live">KERNEL: GAUSSIAN RBF (&lt;5ms)</span>
            </div>
        </div>
    </div>
    """
