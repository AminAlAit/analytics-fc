"""Contract tests for the institutional HUD and slim-bundle count fallback."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from dashboard.components.layout import (
    STYLESHEET_PATH,
    load_institutional_css,
    query_hud_counts,
    render_terminal_bar,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_DB = REPO_ROOT / "dbt_project" / "analytics_fc.duckdb"


def _write_table(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    rows: list[tuple[int | None, int]],
) -> None:
    con.execute(f"CREATE TABLE {table_name} (player_id INTEGER, overall INTEGER);")
    con.executemany(f"INSERT INTO {table_name} VALUES (?, ?);", rows)


def test_stylesheet_exposes_pitch_dark_tokens() -> None:
    """Verify the design system file is what the layout injects."""
    css = load_institutional_css()
    assert css.startswith("<style>")
    assert css.rstrip().endswith("</style>")
    assert "--afc-bg: #0A0E17;" in css
    assert ".terminal-bar" in css
    assert ".kpi-grid" in css
    assert STYLESHEET_PATH.name == "institutional.css"


def test_hud_uses_dim_players_when_present(tmp_path: Path) -> None:
    """Full warehouse shape: season rows from fct, players from dim_players."""
    db_path = tmp_path / "full.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        _write_table(con, "fct_player_seasons", [(1, 80), (1, 84), (2, 77)])
        _write_table(con, "dim_players", [(1, 84), (2, 77), (3, 90)])
    finally:
        con.close()

    counts = query_hud_counts(db_path)
    assert counts == {
        "season_rows": 3,
        "distinct_players": 3,
        "source": "fct_player_seasons+dim_players",
    }
    query_hud_counts(db_path)


def test_hud_falls_back_when_dim_players_missing(tmp_path: Path) -> None:
    """Slim serving bundle: distinct player_id from fct_player_seasons only."""
    db_path = tmp_path / "slim.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        _write_table(
            con,
            "fct_player_seasons",
            [(1, 80), (1, 84), (2, 77), (None, 60)],
        )
        _write_table(con, "stg_fifa_granular", [(9, 50)])
    finally:
        con.close()

    counts = query_hud_counts(db_path)
    assert counts["source"] == "fct_player_seasons"
    assert counts["season_rows"] == 4
    assert counts["distinct_players"] == 2


def test_hud_uses_staging_when_marts_are_absent(tmp_path: Path) -> None:
    """Last fallback is stg_fifa_granular, including when dim_players exists alone."""
    db_path = tmp_path / "staging.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        _write_table(con, "dim_players", [(1, 88)])
        _write_table(con, "stg_fifa_granular", [(4, 70), (4, 71), (5, 66)])
    finally:
        con.close()

    counts = query_hud_counts(db_path)
    assert counts == {
        "season_rows": 3,
        "distinct_players": 2,
        "source": "stg_fifa_granular",
    }


def test_hud_empty_catalog_and_missing_file(tmp_path: Path) -> None:
    """Empty databases return zeros. A missing file raises before connect."""
    empty_path = tmp_path / "empty.duckdb"
    con = duckdb.connect(str(empty_path))
    con.close()
    assert query_hud_counts(empty_path)["source"] == "unavailable"

    missing = tmp_path / "missing.duckdb"
    with pytest.raises(FileNotFoundError):
        query_hud_counts(missing)


def test_terminal_bar_formats_counts_and_escapes_status() -> None:
    """HUD markup keeps search catalog rows separate from mart player counts."""
    markup = render_terminal_bar(
        status_label="EMBEDDED <DUCKDB>",
        season_rows=180021,
        distinct_players=53111,
        catalog_rows=142079,
    )
    assert "180,021 SEASONS / 53,111 PLAYERS" in markup
    assert "CATALOG SEARCH: 142,079 RECORDS ACTIVE" in markup
    assert "EMBEDDED &lt;DUCKDB&gt;" in markup
    assert "<script>" not in markup
    assert "institutional-frame" in markup


def test_app_preserves_serving_bootstrap_and_forecaster_tab() -> None:
    """Cloud bootstrap and Model 2 tab stay in the dashboard entrypoint."""
    source = (REPO_ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
    assert "def ensure_serving_assets" not in source
    assert "ensure_serving_assets()" in source
    assert "def _render_trajectory_forecaster_tab" in source
    assert "Model 2: Trajectory Forecaster" in source
    assert "DARK_CSS" not in source


def test_live_hud_counts_on_canonical_duckdb() -> None:
    """Verified local mart counts when dim_players is present."""
    if not CANONICAL_DB.exists():
        pytest.skip("Canonical DuckDB is not available.")
    counts = query_hud_counts(CANONICAL_DB)
    is_full_warehouse = CANONICAL_DB.stat().st_size > 50_000_000
    if is_full_warehouse and counts["source"] == "fct_player_seasons+dim_players":
        assert counts["season_rows"] == 180021
        assert counts["distinct_players"] == 53111
    elif counts["source"] in ("fct_player_seasons+dim_players", "fct_player_seasons"):
        assert int(counts["season_rows"]) > 0
        assert int(counts["distinct_players"]) > 0
    else:
        raise AssertionError(f"Unexpected HUD source: {counts['source']}")
