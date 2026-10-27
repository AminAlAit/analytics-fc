"""Integration and contract test suite for Analytics FC FastAPI microservice.

Verifies route availability, query parameter validation, schema compliance,
Pydantic model integrity, error handling, and vector inference performance.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from src.api.main import app

HAALAND_ID = 239085
HAALAND_YEAR = 2022
COURTOIS_ID = 192119
COURTOIS_YEAR = 2022


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Instantiate reusable Starlette TestClient for FastAPI application."""
    return TestClient(app)


# ==============================================================================
# 1. HEALTH AND DIAGNOSTIC PROBES
# ==============================================================================


def test_health_probe_status_and_metadata(client: TestClient) -> None:
    """Assert /health returns 200 with service metadata and hardware constraints."""
    response = client.get("/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["service"] == "analytics-fc-api"
    assert payload["version"] == "1.0.0"

    # Verify database telemetry
    db_meta = payload["database"]
    assert db_meta["status"] == "connected"
    assert db_meta["catalog_records"] > 140000

    # Verify superstar gallery artifact telemetry
    gallery_meta = payload["artifacts"]
    assert gallery_meta["gallery_status"] == "loaded"
    assert gallery_meta["superstar_count"] == 100

    # Verify laptop-friendly hardware limits
    hw = payload["hardware_limits"]
    assert hw["threads"] == 1
    assert hw["memory_limit"] == "1GB"


# ==============================================================================
# 2. CATALOG SEARCH AND PAGINATION ENDPOINTS
# ==============================================================================


def test_player_search_default_parameters(client: TestClient) -> None:
    """Assert default search returns paginated records and sector distributions."""
    response = client.get("/api/v1/players/search?limit=25")
    assert response.status_code == 200

    payload = response.json()
    assert "items" in payload
    assert "total_matching" in payload
    assert "sector_counts" in payload

    assert len(payload["items"]) == 25
    assert payload["total_matching"] > 140000
    assert payload["limit"] == 25
    assert payload["offset"] == 0
    assert payload["has_more"] is True

    # Validate mandatory summary fields
    first = payload["items"][0]
    assert "player_id" in first
    assert "short_name" in first
    assert "fifa_edition_year" in first
    assert "overall" in first
    assert "primary_pos" in first
    assert "tactical_sector" in first


def test_player_search_by_name_filter(client: TestClient) -> None:
    """Assert search filters correctly by player surname."""
    response = client.get("/api/v1/players/search?query=Haaland")
    assert response.status_code == 200

    payload = response.json()
    assert payload["total_matching"] > 0
    found_haaland = any("haaland" in p["short_name"].lower() for p in payload["items"])
    assert found_haaland is True


def test_player_search_faceted_filters(client: TestClient) -> None:
    """Assert combining edition year, rating floor, and tactical sector filters."""
    response = client.get(
        "/api/v1/players/search",
        params={
            "edition_year": 2021,
            "min_overall": 86,
            "sector": "Defenders",
            "limit": 10,
        },
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["total_matching"] > 0
    for p in payload["items"]:
        assert p["fifa_edition_year"] == 2021
        assert p["overall"] >= 86
        assert p["tactical_sector"] == "Defenders"


def test_player_search_offset_pagination(client: TestClient) -> None:
    """Assert offset pagination returns sequential distinct player-season slices."""
    res_page1 = client.get("/api/v1/players/search?limit=5&offset=0")
    res_page2 = client.get("/api/v1/players/search?limit=5&offset=5")

    assert res_page1.status_code == 200
    assert res_page2.status_code == 200

    items_page1 = [(p["player_id"], p["fifa_edition_year"]) for p in res_page1.json()["items"]]
    items_page2 = [(p["player_id"], p["fifa_edition_year"]) for p in res_page2.json()["items"]]

    assert len(items_page1) == 5
    assert len(items_page2) == 5
    assert set(items_page1).isdisjoint(set(items_page2))


def test_player_search_validation_errors(client: TestClient) -> None:
    """Assert query validation boundaries reject invalid inputs with HTTP 422."""
    # Out of range overall rating
    res_invalid_rating = client.get("/api/v1/players/search?min_overall=30")
    assert res_invalid_rating.status_code == 422

    # Out of range FIFA year
    res_invalid_year = client.get("/api/v1/players/search?edition_year=1998")
    assert res_invalid_year.status_code == 422

    # Exceeding page size limit
    res_invalid_limit = client.get("/api/v1/players/search?limit=500")
    assert res_invalid_limit.status_code == 422


# ==============================================================================
# 3. PLAYER PROFILE RETRIEVAL ENDPOINTS
# ==============================================================================


def test_get_player_profile_success(client: TestClient) -> None:
    """Assert complete 34-skill profile returned for valid player and year."""
    url = f"/api/v1/players/{HAALAND_ID}?edition_year={HAALAND_YEAR}"
    response = client.get(url)
    assert response.status_code == 200

    profile = response.json()
    assert profile["player_id"] == HAALAND_ID
    assert profile["fifa_edition_year"] == HAALAND_YEAR
    assert "E. Haaland" in profile["short_name"]
    assert profile["primary_pos"] == "ST"
    assert profile["tactical_sector"] == "Center Forwards & Strikers"

    # Check granular traits are populated
    assert profile["attacking_finishing"] is not None
    assert profile["movement_sprint_speed"] is not None
    assert profile["power_shot_power"] is not None


def test_get_player_profile_not_found(client: TestClient) -> None:
    """Assert HTTP 404 returned when player does not exist."""
    response = client.get("/api/v1/players/99999999?edition_year=2022")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# ==============================================================================
# 4. CATALOG METRICS ENDPOINT
# ==============================================================================


def test_get_catalog_statistics(client: TestClient) -> None:
    """Assert catalog statistics match verified historical bounds."""
    response = client.get("/api/v1/players/stats/catalog")
    assert response.status_code == 200

    stats = response.json()
    assert stats["total_rows"] >= 140000
    assert stats["distinct_players"] >= 45000
    assert stats["min_year"] <= 2015
    assert stats["max_year"] >= 2022


# ==============================================================================
# 5. CEILING SIMILARITY MATCHER ENDPOINT
# ==============================================================================


def test_ceiling_matcher_outfield_player(client: TestClient) -> None:
    """Assert vector matching computes similarity and radar in under 20ms."""
    payload = {
        "player_id": HAALAND_ID,
        "fifa_edition_year": HAALAND_YEAR,
        "top_k": 3,
    }
    # Initial request to ensure cold start cache initialization
    client.post("/api/v1/match/ceiling", json=payload)

    response = client.post("/api/v1/match/ceiling", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["kernel"] == "gaussian_rbf"
    assert data["inference_time_ms"] < 20.0

    # Prospect verification
    assert data["prospect"]["player_id"] == HAALAND_ID

    # Matches verification
    matches = data["matches"]
    assert len(matches) == 3
    assert matches[0]["rank"] == 1
    assert matches[0]["similarity_pct"] > 0
    assert len(matches[0]["strengths"]) > 0
    assert "attributes" in matches[0]

    # Macro radar verification (6 outfield dimensions)
    radar = data["macro_radar"]
    assert len(radar) == 6
    dims = [r["dimension"] for r in radar]
    assert dims == ["Pace", "Shooting", "Passing", "Dribbling", "Defending", "Physical"]


def test_ceiling_matcher_goalkeeper(client: TestClient) -> None:
    """Assert goalkeeper matching selects keeper archetypes and goalkeeper radar."""
    payload = {
        "player_id": COURTOIS_ID,
        "fifa_edition_year": COURTOIS_YEAR,
        "top_k": 3,
    }
    response = client.post("/api/v1/match/ceiling", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["prospect"]["primary_pos"] == "GK"

    matches = data["matches"]
    assert len(matches) == 3
    for m in matches:
        assert m["sector"] == "Goalkeepers"

    # Macro radar verification (6 goalkeeper dimensions)
    radar = data["macro_radar"]
    assert len(radar) == 6
    dims = [r["dimension"] for r in radar]
    assert dims == ["Diving", "Handling", "Kicking", "Reflexes", "Positioning", "Reactions"]


def test_ceiling_matcher_nonexistent_player_returns_404(client: TestClient) -> None:
    """Assert matcher returns 404 when requested prospect does not exist."""
    payload = {
        "player_id": 99999999,
        "fifa_edition_year": 2022,
        "top_k": 3,
    }
    response = client.post("/api/v1/match/ceiling", json=payload)
    assert response.status_code == 404


def test_ceiling_matcher_invalid_parameters_returns_422(client: TestClient) -> None:
    """Assert invalid top_k boundary triggers HTTP 422 Unprocessable Entity."""
    payload = {
        "player_id": HAALAND_ID,
        "fifa_edition_year": HAALAND_YEAR,
        "top_k": 50,  # Valid range is 1 to 10
    }
    response = client.post("/api/v1/match/ceiling", json=payload)
    assert response.status_code == 422
