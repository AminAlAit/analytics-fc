"""Integration and verification gate tests for Analytics FC FastAPI v2.

Validates:
1. Asynchronous /api/v2 router endpoints (health, search, forecast, match, scout).
2. Pydantic v2 request/response schema enforcement (422 validation checks, 404 misses).
3. Dependency injection lifecycle for single-threaded read-only DuckDB sessions.
4. Quantile + conformal trajectory projection across 1 to 3 year horizons.
5. Verification gate: response inference latency < 30ms.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

from src.api.main import app
from src.api.v2.dependencies import (
    DuckDBSessionManager,
    get_db_session,
    get_session_manager,
)

HAALAND_ID = 239085
HAALAND_YEAR = 2022
COURTOIS_ID = 192119
COURTOIS_YEAR = 2022


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Instantiate reusable TestClient and warm up ML registry once."""
    tc = TestClient(app)
    # Warm up DuckDB metadata and in-memory ML model registry
    tc.get("/api/v2/health")
    return tc


def test_v2_health_diagnostics(client: TestClient) -> None:
    """Assert /api/v2/health returns verified dataset telemetry and loaded models."""
    resp = client.get("/api/v2/health")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "healthy"
    assert data["api_version"] == "2.0.0"
    assert data["database_connected"] is True
    assert data["catalog_rows"] == 180021
    assert data["distinct_players"] == 53111
    assert data["min_year"] == 2015
    assert data["max_year"] == 2024

    models = data["models"]
    assert models["quantile_regressor_loaded"] is True
    assert models["conformal_predictor_loaded"] is True
    assert models["archetype_index_loaded"] is True
    assert models["archetype_count"] == 100

    hw = data["hardware_limits"]
    assert hw["threads"] == 1
    assert hw["memory_limit"] == "1GB"


def test_v2_duckdb_dependency_teardown() -> None:
    """Verify get_db_session dependency cleanly opens and closes DuckDB connections."""
    gen = get_db_session()
    con = next(gen)
    count = con.execute("SELECT COUNT(*) FROM main.fct_player_seasons;").fetchone()
    assert count is not None and count[0] == 180021

    # Exhaust generator to trigger finally block teardown
    with pytest.raises(StopIteration):
        next(gen)


def test_v2_faceted_player_search_and_latency(client: TestClient) -> None:
    """Verify /api/v2/players/search returns modern 2024 records in < 30ms."""
    params = {
        "edition_year": 2024,
        "min_overall": 88,
        "sector": "Center Forwards & Strikers",
        "limit": 10,
    }
    # Warm query cache
    client.get("/api/v2/players/search", params=params)

    resp = client.get("/api/v2/players/search", params=params)
    assert resp.status_code == 200
    data = resp.json()

    assert data["total_matching"] > 0
    assert len(data["items"]) > 0
    assert data["inference_time_ms"] < 30.0, (
        f"Search latency {data['inference_time_ms']}ms exceeded 30ms verification gate."
    )

    for item in data["items"]:
        assert item["fifa_edition_year"] == 2024
        assert item["overall"] >= 88
        assert item["tactical_sector"] == "Center Forwards & Strikers"


def test_v2_search_validation_rejects_invalid_bounds(client: TestClient) -> None:
    """Verify HTTP 422 returned on out-of-range query parameters or invalid sector."""
    resp = client.get("/api/v2/players/search?edition_year=2030")
    assert resp.status_code == 422

    resp2 = client.get("/api/v2/players/search?min_overall=10")
    assert resp2.status_code == 422

    resp3 = client.get("/api/v2/players/search?sector=InvalidSector")
    assert resp3.status_code == 422


def test_v2_trajectory_forecast_multi_horizon(client: TestClient) -> None:
    """Verify 3-year quantile and conformal trajectory projection under 30ms."""
    payload = {
        "player_id": HAALAND_ID,
        "edition_year": HAALAND_YEAR,
        "horizon_years": 3,
        "include_conformal": True,
    }
    # Warmup JIT paths
    client.post("/api/v2/trajectories/forecast", json=payload)

    resp = client.post("/api/v2/trajectories/forecast", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["player_id"] == HAALAND_ID
    assert "Haaland" in data["short_name"]
    assert data["reference_edition_year"] == HAALAND_YEAR
    assert data["historical_growth_1y"] == 4.0  # 88 - 84 from 2021 to 2022
    assert data["inference_time_ms"] < 30.0, (
        f"Trajectory forecast took {data['inference_time_ms']}ms, exceeding 30ms gate."
    )

    forecast = data["forecast"]
    assert len(forecast) == 3
    for idx, pt in enumerate(forecast, start=1):
        assert pt["target_edition_year"] == HAALAND_YEAR + idx
        assert 35.0 <= pt["q10_overall"] <= pt["q50_overall"] <= pt["q90_overall"] <= 99.0
        assert pt["conformal_lower"] is not None
        assert pt["conformal_upper"] is not None
        assert 35.0 <= pt["conformal_lower"] <= pt["conformal_upper"] <= 99.0


def test_v2_trajectory_forecast_goalkeeper_null_imputation(client: TestClient) -> None:
    """Verify goalkeepers with null outfield stats project cleanly without errors."""
    payload = {
        "player_id": COURTOIS_ID,
        "edition_year": COURTOIS_YEAR,
        "horizon_years": 2,
        "include_conformal": True,
    }
    resp = client.post("/api/v2/trajectories/forecast", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "Courtois" in data["short_name"]
    assert len(data["forecast"]) == 2
    assert data["inference_time_ms"] < 30.0


def test_v2_archetype_matching_and_latency(client: TestClient) -> None:
    """Verify /api/v2/archetypes/match completes in < 30ms with tactical isolation."""
    payload = {
        "player_id": HAALAND_ID,
        "edition_year": HAALAND_YEAR,
        "top_k": 3,
    }
    client.post("/api/v2/archetypes/match", json=payload)

    resp = client.post("/api/v2/archetypes/match", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["prospect_id"] == HAALAND_ID
    assert data["primary_position"] == "ST"
    assert len(data["matches"]) == 3
    assert data["inference_time_ms"] < 30.0, (
        f"Archetype match took {data['inference_time_ms']}ms, exceeding 30ms gate."
    )

    for match in data["matches"]:
        assert match["player_id"] != HAALAND_ID
        assert 0.0 <= match["similarity_score_pct"] <= 100.0

    # Test goalkeeper isolation
    gk_resp = client.post(
        "/api/v2/archetypes/match",
        json={"player_id": COURTOIS_ID, "edition_year": COURTOIS_YEAR, "top_k": 3},
    )
    assert gk_resp.status_code == 200
    gk_data = gk_resp.json()
    for match in gk_data["matches"]:
        assert match["tactical_sector"] == "Goalkeepers"


def test_v2_comprehensive_scout_dossier(client: TestClient) -> None:
    """Verify unified /api/v2/scout/{player_id} endpoint under 30ms."""
    url = f"/api/v2/scout/{HAALAND_ID}?edition_year=2024&horizon_years=3&top_k=3"
    client.get(url)

    resp = client.get(url)
    assert resp.status_code == 200
    data = resp.json()

    assert data["profile"]["player_id"] == HAALAND_ID
    assert data["profile"]["fifa_edition_year"] == 2024
    assert data["profile"]["overall"] == 91
    assert len(data["trajectory"]["forecast"]) == 3
    assert len(data["archetypes"]["matches"]) == 3
    assert data["db_query_time_ms"] >= 0.0
    assert data["total_inference_time_ms"] < 30.0, (
        f"Comprehensive scout inference took {data['total_inference_time_ms']}ms, exceeding 30ms gate."
    )
    total_compute = data["db_query_time_ms"] + data["total_inference_time_ms"]
    assert total_compute < 30.0, (
        f"Comprehensive scout total compute took {total_compute:.2f}ms, exceeding 30ms gate."
    )


def test_v2_not_found_and_validation_errors(client: TestClient) -> None:
    """Verify 404 on unknown player IDs and 422 on invalid schemas."""
    resp_404 = client.post(
        "/api/v2/trajectories/forecast",
        json={"player_id": 99999999, "edition_year": 2022, "horizon_years": 1},
    )
    assert resp_404.status_code == 404

    resp_404_scout = client.get("/api/v2/scout/99999999?edition_year=2024")
    assert resp_404_scout.status_code == 404

    resp_422 = client.post(
        "/api/v2/trajectories/forecast",
        json={"player_id": HAALAND_ID, "edition_year": 2022, "horizon_years": 10},
    )
    assert resp_422.status_code == 422


def test_v2_health_returns_503_when_db_missing(tmp_path: Path) -> None:
    """Verify /api/v2/health catches FileNotFoundError and returns HTTP 503."""
    missing_db = tmp_path / "nonexistent.duckdb"
    app.dependency_overrides[get_session_manager] = lambda: DuckDBSessionManager(db_path=missing_db)
    try:
        local_client = TestClient(app)
        resp = local_client.get("/api/v2/health")
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"].lower()
    finally:
        app.dependency_overrides.clear()


def test_v1_and_v2_concurrent_duckdb_config_parity() -> None:
    """Verify concurrent v1 and v2 ASGI requests share identical DuckDB configuration."""

    async def _run_concurrent() -> list[int]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
            tasks = [
                ac.get("/api/v1/players/search?limit=5"),
                ac.get("/api/v2/players/search?edition_year=2024&min_overall=88&limit=5"),
                ac.get(f"/api/v1/players/{HAALAND_ID}?edition_year={HAALAND_YEAR}"),
                ac.get(f"/api/v2/scout/{HAALAND_ID}?edition_year=2024&horizon_years=1&top_k=3"),
            ]
            responses = await asyncio.gather(*tasks)
            return [r.status_code for r in responses]

    statuses = asyncio.run(_run_concurrent())
    assert statuses == [200, 200, 200, 200], f"Concurrent v1+v2 failure: {statuses}"
