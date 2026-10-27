"""Resilient REST API client for Analytics FC scouting dashboard.

Communicates with the FastAPI microservice over HTTP with automatic,
graceful fallback to local DuckDB and in-memory vector engines if offline.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from src.api.routers.matcher import execute_vector_matching, get_gallery
from src.catalog_search import (
    get_player_by_id_and_year,
    search_players_paginated,
)

logger = logging.getLogger("analytics_fc.dashboard.api_client")

API_BASE_URL = os.environ.get("AFC_API_URL", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_TIMEOUT_SEC = 1.0


def is_api_online(timeout: float = 0.5) -> bool:
    """Check if the FastAPI backend service is reachable and healthy."""
    url = f"{API_BASE_URL}/health"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return bool(resp.status == 200)
    except Exception:
        return False


def get_connection_status() -> str:
    """Return human-readable badge status for backend telemetry bar."""
    if is_api_online():
        return "FASTAPI REST ONLINE"
    return "LOCAL DUCKDB FALLBACK"


def search_players(
    query: str = "",
    edition_year: int | None = None,
    min_overall: int | None = None,
    max_overall: int | None = None,
    sector: str | None = None,
    limit: int = 50,
    offset: int = 0,
    include_sector_counts: bool = True,
) -> dict[str, Any]:
    """Execute player search via FastAPI endpoint or fallback to local DuckDB engine."""
    params: dict[str, Any] = {"limit": limit, "offset": offset, "include_sector_counts": str(include_sector_counts).lower()}
    if query:
        params["query"] = query
    if edition_year is not None:
        params["edition_year"] = edition_year
    if min_overall is not None:
        params["min_overall"] = min_overall
    if max_overall is not None:
        params["max_overall"] = max_overall
    if sector is not None:
        params["sector"] = sector

    url = f"{API_BASE_URL}/api/v1/players/search?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_SEC) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.debug(f"FastAPI search unavailable ({exc}), invoking local DuckDB fallback.")

    # Graceful local fallback
    return search_players_paginated(
        query=query,
        edition_year=edition_year,
        min_overall=min_overall,
        max_overall=max_overall,
        sector=sector,
        limit=limit,
        offset=offset,
        include_sector_counts=include_sector_counts,
    )


def get_player_profile(
    player_id: int,
    edition_year: int,
) -> dict[str, Any] | None:
    """Retrieve complete 34-skill player profile via FastAPI or fallback to local DuckDB."""
    params = urllib.parse.urlencode({"edition_year": edition_year})
    url = f"{API_BASE_URL}/api/v1/players/{player_id}?{params}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_SEC) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as http_err:
        if http_err.code == 404:
            return None
        logger.debug(f"HTTP error {http_err.code} on player profile, using local fallback.")
    except Exception as exc:
        logger.debug(f"FastAPI profile lookup unavailable ({exc}), invoking local fallback.")

    # Graceful local fallback
    return get_player_by_id_and_year(
        player_id=player_id,
        fifa_edition_year=edition_year,
    )


def match_ceiling(
    player_id: int,
    edition_year: int,
    top_k: int = 3,
) -> dict[str, Any] | None:
    """Compute superstar ceiling similarity via FastAPI or fallback to in-memory math."""
    url = f"{API_BASE_URL}/api/v1/match/ceiling"
    payload = json.dumps({
        "player_id": player_id,
        "fifa_edition_year": edition_year,
        "top_k": top_k,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_SEC) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as http_err:
        if http_err.code == 404:
            return None
        logger.debug(f"HTTP error {http_err.code} on ceiling match, using local fallback.")
    except Exception as exc:
        logger.debug(f"FastAPI matcher unavailable ({exc}), invoking local in-memory fallback.")

    # Graceful local fallback
    prospect = get_player_by_id_and_year(player_id, edition_year)
    if prospect is None:
        return None

    gallery = get_gallery()
    matches, radar_dims = execute_vector_matching(prospect, gallery, top_k=top_k)

    return {
        "prospect": prospect,
        "matches": [m.model_dump() for m in matches],
        "macro_radar": [r.model_dump() for r in radar_dims],
        "kernel": "gaussian_rbf",
        "inference_time_ms": 1.0,
    }
