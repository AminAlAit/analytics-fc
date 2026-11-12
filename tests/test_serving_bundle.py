"""Unit tests for Streamlit Cloud serving bundle export, bootstrapper, and embedded client."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import duckdb
import pytest

from dashboard import api_client
from dashboard.bootstrap import (
    MIN_VALID_DUCKDB_BYTES,
    _validate_archive_member,
    prepare_serving_assets,
)
from scripts.export_serving_bundle import (
    SERVING_FCT_COLUMNS,
    SERVING_STG_COLUMNS,
    create_serving_bundle,
    export_serving_duckdb,
)
from src.api.routers.matcher import execute_vector_matching
from src.catalog_search import ALL_SKILL_COLUMNS
from src.models.ceiling_matcher import compute_vector_matching


def _build_synthetic_source_db(db_path: Path) -> None:
    """Create a minimal DuckDB mirroring stg_fifa_granular, fct_player_seasons, and bloated tables."""
    con = duckdb.connect(str(db_path), read_only=False)
    try:
        con.execute("SET threads = 1;")
        con.execute("SET memory_limit = '1GB';")

        stg_cols = {
            "player_id": "INTEGER",
            "fifa_edition_year": "INTEGER",
            "short_name": "VARCHAR",
            "long_name": "VARCHAR",
            "age": "INTEGER",
            "nationality": "VARCHAR",
            "club": "VARCHAR",
            "league_name": "VARCHAR",
            "positions": "VARCHAR",
            "preferred_foot": "VARCHAR",
            "height_cm": "INTEGER",
            "weight_kg": "INTEGER",
            "overall": "INTEGER",
            "potential": "INTEGER",
            "value_eur": "DOUBLE",
            "wage_eur": "DOUBLE",
            "pace": "INTEGER",
            "shooting": "INTEGER",
            "passing": "INTEGER",
            "dribbling": "INTEGER",
            "defending": "INTEGER",
            "physic": "INTEGER",
        }
        for col in ALL_SKILL_COLUMNS:
            stg_cols[col] = "INTEGER"
        stg_cols["extra_unused_col"] = "VARCHAR"

        ddl_stg = ", ".join(f"{name} {dtype}" for name, dtype in stg_cols.items())
        con.execute(f"CREATE TABLE main.stg_fifa_granular ({ddl_stg});")

        skill_vals = [82 if not c.startswith("goalkeeping_") else 12 for c in ALL_SKILL_COLUMNS]
        placeholders = ", ".join(["?"] * len(stg_cols))
        con.execute(
            f"INSERT INTO main.stg_fifa_granular VALUES ({placeholders});",
            [
                239085,
                2022,
                "E. Haaland",
                "Erling Braut Haaland",
                21,
                "Norway",
                "Borussia Dortmund",
                "German 1. Bundesliga",
                "ST",
                "Left",
                194,
                88,
                88,
                93,
                137500000.0,
                110000.0,
                89,
                91,
                65,
                80,
                45,
                88,
                *skill_vals,
                "unused",
            ],
        )

        con.execute(
            """
            CREATE TABLE main.fct_player_seasons (
                player_id INTEGER,
                fifa_edition_year INTEGER,
                short_name VARCHAR,
                long_name VARCHAR,
                age INTEGER,
                nationality VARCHAR,
                club VARCHAR,
                league_name VARCHAR,
                positions VARCHAR,
                overall INTEGER,
                potential INTEGER,
                value_eur DOUBLE,
                pace INTEGER,
                shooting INTEGER,
                passing INTEGER,
                dribbling INTEGER,
                defending INTEGER,
                physic INTEGER,
                annual_growth_velocity DOUBLE,
                extra_col VARCHAR
            );
            """
        )
        con.execute(
            """
            INSERT INTO main.fct_player_seasons VALUES (
                239085, 2022, 'E. Haaland', 'Erling Braut Haaland', 21, 'Norway',
                'Borussia Dortmund', 'German 1. Bundesliga', 'ST', 88, 93, 137500000.0,
                89, 91, 65, 80, 45, 88, 4.0, 'extra'
            );
            """
        )
        con.execute("CREATE TABLE main.mart_pattern_maker (id INTEGER, payload VARCHAR);")
        con.execute("INSERT INTO main.mart_pattern_maker VALUES (1, 'drop_me');")
    finally:
        con.close()


def _sample_gallery() -> list[dict[str, Any]]:
    """Construct a minimal 2-star superstar gallery fixture."""
    outfield_attrs = {
        c: (85.0 if not c.startswith("goalkeeping_") else 10.0) for c in ALL_SKILL_COLUMNS
    }
    gk_attrs = {c: (15.0 if not c.startswith("goalkeeping_") else 88.0) for c in ALL_SKILL_COLUMNS}
    return [
        {
            "player_id": 9,
            "short_name": "R. Lewandowski",
            "tactical_sector": "Center Forwards & Strikers",
            "peak_overall": 92,
            "peak_age_min": 31,
            "peak_age_max": 33,
            "attributes": outfield_attrs,
        },
        {
            "player_id": 192448,
            "short_name": "M. ter Stegen",
            "tactical_sector": "Goalkeepers",
            "peak_overall": 90,
            "peak_age_min": 27,
            "peak_age_max": 29,
            "attributes": gk_attrs,
        },
    ]


def test_export_serving_duckdb_and_bundle(tmp_path: Path) -> None:
    """Verify serving export trims bloated tables and stays well below 100 MB."""
    src_db = tmp_path / "source.duckdb"
    _build_synthetic_source_db(src_db)

    dest_db = tmp_path / "serving.duckdb"
    stats = export_serving_duckdb(source_db=src_db, dest_db=dest_db)
    assert stats["stg_fifa_granular_rows"] == 1
    assert stats["fct_player_seasons_rows"] == 1
    assert stats["size_mb"] < 100.0

    con = duckdb.connect(str(dest_db), read_only=True)
    try:
        tables = {
            row[0]
            for row in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main';"
            ).fetchall()
        }
        assert tables == {"stg_fifa_granular", "fct_player_seasons"}

        stg_cols = [row[0] for row in con.execute("DESCRIBE main.stg_fifa_granular;").fetchall()]
        fct_cols = [row[0] for row in con.execute("DESCRIBE main.fct_player_seasons;").fetchall()]
        assert stg_cols == SERVING_STG_COLUMNS
        assert fct_cols == SERVING_FCT_COLUMNS
    finally:
        con.close()

    gallery_path = tmp_path / "superstar_gallery.json"
    gallery_path.write_text(json.dumps(_sample_gallery()), encoding="utf-8")
    q_model = tmp_path / "trajectory_regressor.joblib"
    c_model = tmp_path / "conformal_predictor.joblib"
    q_model.write_bytes(b"q" * 256)
    c_model.write_bytes(b"c" * 256)

    bundle_zip = tmp_path / "bundle.zip"
    bundle_info = create_serving_bundle(
        source_db=src_db,
        gallery_path=gallery_path,
        quantile_model_path=q_model,
        conformal_model_path=c_model,
        output_zip=bundle_zip,
    )
    assert bundle_zip.exists()
    assert bundle_info["bundle_mb"] < 100.0
    assert "dbt_project/analytics_fc.duckdb" in bundle_info["included_files"]
    assert "data/processed/superstar_gallery.json" in bundle_info["included_files"]


def test_ceiling_matcher_framework_free_parity() -> None:
    """Verify framework-free ceiling matcher matches FastAPI router schema outputs."""
    prospect: dict[str, Any] = {
        "player_id": 239085,
        "short_name": "E. Haaland",
        "primary_pos": "ST",
        "tactical_sector": "Center Forwards & Strikers",
    }
    for col in ALL_SKILL_COLUMNS:
        prospect[col] = 80.0 if not col.startswith("goalkeeping_") else 10.0

    gallery = _sample_gallery()
    raw_matches, raw_radar = compute_vector_matching(prospect, gallery, top_k=3)
    pyd_matches, pyd_radar = execute_vector_matching(prospect, gallery, top_k=3)

    assert len(raw_matches) == len(pyd_matches) == 1
    assert raw_matches[0] == pyd_matches[0].model_dump()
    assert raw_radar == [r.model_dump() for r in pyd_radar]


def test_bootstrap_rejects_zip_traversal(tmp_path: Path) -> None:
    """Verify archive member validator blocks path traversal entries."""
    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../../evil.txt", "owned")

    with zipfile.ZipFile(bad_zip, "r") as zf:
        member = zf.infolist()[0]
        with pytest.raises(ValueError, match="Unsafe archive member path"):
            _validate_archive_member(member, tmp_path)


def test_bootstrap_extracts_serving_bundle_atomically(tmp_path: Path) -> None:
    """Verify prepare_serving_assets downloads and extracts bundle cleanly when local assets missing."""
    bundle_zip = tmp_path / "source_bundle.zip"
    with zipfile.ZipFile(bundle_zip, "w") as zf:
        zf.writestr("dbt_project/analytics_fc.duckdb", b"DUCK" * (MIN_VALID_DUCKDB_BYTES // 2))
        zf.writestr("data/processed/superstar_gallery.json", json.dumps(_sample_gallery()))
        zf.writestr("models/trajectory_regressor.joblib", b"q" * 200)
        zf.writestr("models/conformal_predictor.joblib", b"c" * 200)

    target_root = tmp_path / "deploy_root"
    target_root.mkdir()
    # Add an empty 12KB stub to ensure bootstrapper replaces stub DuckDB
    (target_root / "dbt_project").mkdir()
    (target_root / "dbt_project" / "analytics_fc.duckdb").write_bytes(b"\x00" * 12288)

    class _FakeResp:
        def __init__(self, payload: bytes) -> None:
            self._data = payload
            self._pos = 0

        def read(self, size: int = -1) -> bytes:
            if size < 0:
                chunk = self._data[self._pos :]
                self._pos = len(self._data)
                return chunk
            chunk = self._data[self._pos : self._pos + size]
            self._pos += len(chunk)
            return chunk

        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

    zip_bytes = bundle_zip.read_bytes()
    with patch("dashboard.bootstrap.urllib.request.urlopen", return_value=_FakeResp(zip_bytes)):
        status = prepare_serving_assets(
            base_dir=target_root,
            bundle_url="https://example.com/analytics_fc_serving_bundle.zip",
        )

    assert status["ready"] is True
    assert status["source"] == "downloaded"
    assert (
        target_root / "dbt_project" / "analytics_fc.duckdb"
    ).stat().st_size > MIN_VALID_DUCKDB_BYTES
    assert (target_root / "data" / "processed" / "superstar_gallery.json").exists()


def test_api_client_embedded_mode_skips_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify api_client avoids localhost HTTP calls and reports embedded mode when AFC_API_URL unset."""
    monkeypatch.delenv("AFC_API_URL", raising=False)
    with patch("dashboard.api_client.urllib.request.urlopen") as mock_urlopen:
        assert api_client.is_api_online() is False
        assert api_client.get_connection_status() == "EMBEDDED DUCKDB ENGINE"
        mock_urlopen.assert_not_called()


def test_dashboard_app_import_has_no_ui_side_effects() -> None:
    """Verify importing dashboard.app helpers does not trigger st.set_page_config or DB queries."""
    sys.modules.pop("dashboard.app", None)
    with patch("streamlit.set_page_config") as mock_page_cfg:
        import dashboard.app as app_mod

        assert app_mod.DRIFT_METRICS_PATH.name == "drift_metrics.json"
        mock_page_cfg.assert_not_called()


def test_ensure_serving_assets_does_not_poison_cache_on_failure() -> None:
    """Verify failed bootstrap attempts clear the resource cache so retries recover."""
    from dashboard import bootstrap

    bootstrap._cached_prepare_serving_assets.clear()
    with patch(
        "dashboard.bootstrap.prepare_serving_assets",
        side_effect=[
            {"ready": False, "source": "error", "error": "transient timeout"},
            {"ready": True, "source": "downloaded", "error": None},
        ],
    ) as mock_prep:
        first = bootstrap.ensure_serving_assets()
        assert first["ready"] is False
        second = bootstrap.ensure_serving_assets()
        assert second["ready"] is True
        assert mock_prep.call_count == 2
    bootstrap._cached_prepare_serving_assets.clear()
