"""Master end-to-end integration suite validating all Analytics FC subsystems."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import yaml
from fastapi.testclient import TestClient

from dashboard import api_client
from dashboard.app import build_tactical_radar_altair, build_trajectory_fan_chart
from scripts.verify_git_hygiene import (
    EXPECTED_AUTHOR_EMAIL,
    EXPECTED_AUTHOR_NAME,
    scan_text_for_violations,
    validate_author_identity,
    validate_commit_message,
)
from src.api.cache_manager import (
    AtomicArtifactSlot,
    DuckDBConnectionPool,
    PlayerProfileCacheManager,
)
from src.api.main import app
from src.catalog_search import get_catalog_stats, search_players_paginated
from src.models.archetype_indexer import ArchetypeVectorIndex
from src.models.archetype_promoter import ALL_SKILL_COLS, run_archetype_promotion_pipeline
from src.models.ceiling_matcher import compute_vector_matching
from src.models.conformal_predictor import ConformalTrajectoryPredictor
from src.models.trajectory_predictor import TrajectoryFeatureContract
from src.models.trajectory_regressor import QuantileTrajectoryRegressor
from src.monitoring.observability import compute_batch_psi_alerts, get_telemetry_snapshot
from src.utils import HardwareWatchdog, set_process_safety

BASE_DIR = Path(__file__).resolve().parent.parent


def _seed_synthetic_integration_assets(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Create minimal synthetic DB, gallery, and models when full weights are absent in CI."""
    db_path = tmp_path / "integration.duckdb"
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
        for col in ALL_SKILL_COLS:
            stg_cols[col] = "INTEGER"
        ddl = ", ".join(f"{name} {dtype}" for name, dtype in stg_cols.items())
        con.execute(f"CREATE TABLE main.stg_fifa_granular ({ddl});")
        skills = [85 if not c.startswith("goalkeeping_") else 12 for c in ALL_SKILL_COLS]
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
                *skills,
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
                height_cm DOUBLE,
                weight_kg DOUBLE,
                overall INTEGER,
                potential INTEGER,
                value_eur DOUBLE,
                pace DOUBLE,
                shooting DOUBLE,
                passing DOUBLE,
                dribbling DOUBLE,
                defending DOUBLE,
                physic DOUBLE,
                annual_growth_velocity DOUBLE
            );
            INSERT INTO main.fct_player_seasons VALUES
                (239085, 2022, 'E. Haaland', 'Erling Braut Haaland', 21, 'Norway',
                 'Borussia Dortmund', 'German 1. Bundesliga', 'ST', 194.0, 88.0, 88, 93, 137500000.0,
                 89.0, 91.0, 65.0, 80.0, 45.0, 88.0, 4.0),
                (9001, 2022, 'A. Prime', 'Alex Prime', 25, 'Spain',
                 'Real Club', 'La Liga', 'CM', 180.0, 75.0, 89, 90, 80000000.0,
                 82.0, 81.0, 88.0, 86.0, 76.0, 80.0, 1.0),
                (9001, 2023, 'A. Prime', 'Alex Prime', 26, 'Spain',
                 'Real Club', 'La Liga', 'CM', 180.0, 75.0, 90, 90, 85000000.0,
                 83.0, 82.0, 89.0, 87.0, 77.0, 81.0, 1.0);
            """
        )
    finally:
        con.close()

    gallery = [
        {
            "player_id": 9,
            "short_name": "R. Lewandowski",
            "long_name": "Robert Lewandowski",
            "nationality": "Poland",
            "primary_position": "ST",
            "positions": "ST",
            "tactical_sector": "Center Forwards & Strikers",
            "peak_overall": 92,
            "peak_seasons_count": 2,
            "peak_age_min": 31,
            "peak_age_max": 32,
            "peak_editions_observed": "2020-2021",
            "height_cm": 185.0,
            "weight_kg": 81.0,
            "attributes": {
                c: (84.0 if not c.startswith("goalkeeping_") else 11.0) for c in ALL_SKILL_COLS
            },
        }
    ]
    gallery_path = tmp_path / "superstar_gallery.json"
    gallery_path.write_text(json.dumps(gallery), encoding="utf-8")

    rng = np.random.default_rng(7)
    n_rows = 40
    overall = rng.integers(65, 88, size=n_rows).astype(float)
    potential = np.clip(overall + rng.integers(1, 10, size=n_rows), 65.0, 95.0)
    X_df = pd.DataFrame(
        {
            "age": rng.integers(18, 32, size=n_rows).astype(float),
            "overall": overall,
            "potential": potential,
            "potential_diff": potential - overall,
            "pace": rng.uniform(55.0, 90.0, size=n_rows),
            "shooting": rng.uniform(50.0, 90.0, size=n_rows),
            "passing": rng.uniform(55.0, 88.0, size=n_rows),
            "dribbling": rng.uniform(55.0, 90.0, size=n_rows),
            "defending": rng.uniform(35.0, 85.0, size=n_rows),
            "physic": rng.uniform(55.0, 88.0, size=n_rows),
            "growth_rate_1y": rng.uniform(-1.0, 4.0, size=n_rows),
        }
    )[TrajectoryFeatureContract.REQUIRED_FEATURES]
    y_vec = np.clip(0.3 * X_df["potential_diff"].to_numpy(), -2.0, 5.0)

    q_path = tmp_path / "trajectory_regressor.joblib"
    c_path = tmp_path / "conformal_predictor.joblib"
    QuantileTrajectoryRegressor(max_iter=10, min_samples_leaf=4, random_state=7).fit(
        X_df.iloc[:25], y_vec[:25]
    ).save(q_path)
    ConformalTrajectoryPredictor(alpha=0.10, random_state=7).fit(
        X_df.iloc[:25], y_vec[:25]
    ).calibrate(X_df.iloc[25:], y_vec[25:]).save(c_path)
    return db_path, gallery_path, q_path, c_path


def _run_core_integration_flow(
    tmp_path: Path,
    db_path: Path,
    gallery_path: Path,
    q_path: Path,
    c_path: Path,
) -> dict[str, Any]:
    """Exercise DuckDB pool, Models 1/2, Altair charts, and telemetry against given assets."""
    pool = DuckDBConnectionPool(db_path=db_path, threads=1, memory_limit="1GB")
    cache_mgr = PlayerProfileCacheManager(pool=pool)
    try:
        with pool.acquire() as con:
            stats = get_catalog_stats(con=con)
            assert stats["total_rows"] >= 1
            search_res = search_players_paginated(con=con, query="Haaland", limit=5)
            assert len(search_res["items"]) >= 1
            target: dict[str, Any] = search_res["items"][0]

        prof_1, hit_1, _ = cache_mgr.get_player_profile(
            player_id=int(target["player_id"]),
            edition_year=int(target["fifa_edition_year"]),
        )
        prof_2, hit_2, _ = cache_mgr.get_player_profile(
            player_id=int(target["player_id"]),
            edition_year=int(target["fifa_edition_year"]),
        )
        assert prof_1 is not None
        assert hit_1 is False
        assert hit_2 is True
    finally:
        cache_mgr.close()

    # Model 1 Ceiling Matcher & Automated Archetype Promoter
    gallery_data: list[dict[str, Any]] = json.loads(gallery_path.read_text(encoding="utf-8"))
    matches, radar_dims = compute_vector_matching(prof_1, gallery_data, top_k=3)
    assert len(matches) >= 1
    assert len(radar_dims) == 6

    slot: AtomicArtifactSlot[ArchetypeVectorIndex] = AtomicArtifactSlot(
        ArchetypeVectorIndex(gallery_path=gallery_path)
    )
    promo = run_archetype_promotion_pipeline(
        db_path=db_path,
        base_gallery_path=gallery_path,
        output_dir=tmp_path / "promo_artifacts",
        version_tag="master_test",
        min_edition_year=2022,
        max_promotions=3,
        write_parquet=True,
        slot=slot,
    )
    assert promo.execution_time_ms < 1000.0
    assert Path(promo.json_artifact_path).exists()

    # Model 2 Trajectory Forecaster & Altair visualization builders
    q_model, c_model = api_client.load_trajectory_models(q_path, c_path)
    forecast_resp = api_client.forecast_trajectory(
        player_id=int(target["player_id"]),
        edition_year=int(target["fifa_edition_year"]),
        horizon_years=3,
        include_conformal=True,
        quantile_model=q_model,
        conformal_model=c_model,
        db_path=db_path,
    )
    assert forecast_resp is not None
    assert len(forecast_resp["forecast"]) == 3
    fan_chart = build_trajectory_fan_chart(forecast_resp)
    assert "layer" in fan_chart.to_dict()

    radar_chart = build_tactical_radar_altair(
        ["Pace", "Shooting", "Passing", "Dribbling", "Defending", "Physical"],
        [85.0, 88.0, 72.0, 80.0, 45.0, 84.0],
        [88.0, 92.0, 79.0, 86.0, 49.0, 85.0],
    )
    assert "layer" in radar_chart.to_dict()

    snap = get_telemetry_snapshot()
    assert snap.memory.rss_mb <= 1024.0
    return target


def test_master_synthetic_ci_fallback(tmp_path: Path) -> None:
    """Guarantee CI fallback assets execute the full integration flow cleanly."""
    db_path, gallery_path, q_path, c_path = _seed_synthetic_integration_assets(tmp_path)
    target = _run_core_integration_flow(tmp_path, db_path, gallery_path, q_path, c_path)
    assert int(target["player_id"]) == 239085


def test_master_platform_end_to_end_sub_60s(tmp_path: Path) -> None:
    """Exercise hardware guard, DuckDB, Models 1/2, API v2, charts, telemetry, and hygiene."""
    t_start = time.perf_counter()

    # 1. Hardware guard & process safety
    safety = set_process_safety(threads=1, n_jobs=1, max_memory_mb=1024.0)
    assert safety["rss_mb"] <= 1024.0

    with HardwareWatchdog(max_memory_mb=1024.0, poll_interval_seconds=0.25) as wd:
        assert wd.check_now() <= 1024.0

        real_db = BASE_DIR / "dbt_project" / "analytics_fc.duckdb"
        real_gallery = BASE_DIR / "data" / "processed" / "superstar_gallery.json"
        real_q = BASE_DIR / "models" / "trajectory_regressor.joblib"
        real_c = BASE_DIR / "models" / "conformal_predictor.joblib"

        use_real = (
            real_db.exists()
            and real_db.stat().st_size > 1_000_000
            and real_gallery.exists()
            and real_q.exists()
            and real_c.exists()
        )
        if use_real:
            db_path, gallery_path, q_path, c_path = real_db, real_gallery, real_q, real_c
        else:
            db_path, gallery_path, q_path, c_path = _seed_synthetic_integration_assets(tmp_path)

        target = _run_core_integration_flow(tmp_path, db_path, gallery_path, q_path, c_path)

        # 5. Observability & FastAPI v2 endpoints
        metrics_file = BASE_DIR / "data" / "monitoring" / "drift_metrics.json"
        if metrics_file.exists():
            drift_snap = compute_batch_psi_alerts(metrics_path=metrics_file)
            assert drift_snap.evaluated_features >= 40

        client = TestClient(app)
        telem_res = client.get("/api/v2/telemetry")
        assert telem_res.status_code == 200

        if use_real:
            assert client.get("/api/v2/health").status_code == 200
            assert client.get("/api/v2/players/search?query=Haaland&limit=3").status_code == 200
            pid = int(target["player_id"])
            year = int(target["fifa_edition_year"])
            assert client.get(f"/api/v2/scout/{pid}?edition_year={year}").status_code == 200

    # 6. Git hygiene & Docker Compose stack validation
    ok_msg, errs_msg = validate_commit_message("Expand master integration test suite")
    assert ok_msg is True and errs_msg == []
    assert validate_author_identity(EXPECTED_AUTHOR_NAME, EXPECTED_AUTHOR_EMAIL)[0] is True
    assert scan_text_for_violations("Clean Python integration payload") == []

    compose = yaml.safe_load((BASE_DIR / "docker-compose.yml").read_text(encoding="utf-8"))
    assert set(compose["services"].keys()) == {"api", "dashboard"}

    total_sec = time.perf_counter() - t_start
    assert total_sec < 60.0, f"Master integration suite took {total_sec:.2f}s (> 60s budget)"
