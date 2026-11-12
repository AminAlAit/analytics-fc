"""Resilient REST API client for Analytics FC scouting dashboard.

Communicates with the FastAPI microservice over HTTP when AFC_API_URL is configured,
with automatic embedded DuckDB and in-memory vector engine execution otherwise.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from src.catalog_search import (
    get_catalog_stats,
    get_player_by_id_and_year,
    search_players_paginated,
)
from src.models.ceiling_matcher import compute_vector_matching, load_gallery_cached

logger = logging.getLogger("analytics_fc.dashboard.api_client")

DEFAULT_TIMEOUT_SEC = 1.0


def _get_api_base_url() -> str | None:
    """Return configured external FastAPI base URL, or None for embedded mode."""
    raw = os.environ.get("AFC_API_URL", "").strip().rstrip("/")
    return raw if raw else None


def is_api_online(timeout: float = 0.5) -> bool:
    """Check if a configured FastAPI backend service is reachable and healthy."""
    base_url = _get_api_base_url()
    if not base_url:
        return False

    url = f"{base_url}/health"
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
    return "EMBEDDED DUCKDB ENGINE"


def get_catalog_stats_summary() -> dict[str, int]:
    """Retrieve verified catalog row counts, distinct players, and edition range."""
    base_url = _get_api_base_url()
    if base_url:
        url = f"{base_url}/api/v1/players/stats/catalog"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_SEC) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.debug(f"FastAPI stats unavailable ({exc}), using embedded DuckDB.")

    return get_catalog_stats()


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
    """Execute player search via FastAPI endpoint or embedded DuckDB engine."""
    base_url = _get_api_base_url()
    if base_url:
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "include_sector_counts": str(include_sector_counts).lower(),
        }
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

        url = f"{base_url}/api/v1/players/search?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_SEC) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.debug(f"FastAPI search unavailable ({exc}), invoking embedded DuckDB.")

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
    """Retrieve complete 34-skill player profile via FastAPI or embedded DuckDB."""
    base_url = _get_api_base_url()
    if base_url:
        params = urllib.parse.urlencode({"edition_year": edition_year})
        url = f"{base_url}/api/v1/players/{player_id}?{params}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_SEC) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as http_err:
            if http_err.code == 404:
                return None
            logger.debug(f"HTTP error {http_err.code} on player profile, using embedded DuckDB.")
        except Exception as exc:
            logger.debug(f"FastAPI profile lookup unavailable ({exc}), invoking embedded DuckDB.")

    return get_player_by_id_and_year(
        player_id=player_id,
        fifa_edition_year=edition_year,
    )


def match_ceiling(
    player_id: int,
    edition_year: int,
    top_k: int = 3,
) -> dict[str, Any] | None:
    """Compute superstar ceiling similarity via FastAPI or embedded vector math."""
    base_url = _get_api_base_url()
    if base_url:
        url = f"{base_url}/api/v1/match/ceiling"
        payload = json.dumps(
            {
                "player_id": player_id,
                "fifa_edition_year": edition_year,
                "top_k": top_k,
            }
        ).encode("utf-8")

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
            logger.debug(f"HTTP error {http_err.code} on ceiling match, using embedded matcher.")
        except Exception as exc:
            logger.debug(f"FastAPI matcher unavailable ({exc}), invoking embedded matcher.")

    prospect = get_player_by_id_and_year(player_id, edition_year)
    if prospect is None:
        return None

    gallery = load_gallery_cached()
    matches, radar_dims = compute_vector_matching(prospect, gallery, top_k=top_k)

    return {
        "prospect": prospect,
        "matches": matches,
        "macro_radar": radar_dims,
        "kernel": "gaussian_rbf",
        "inference_time_ms": 1.0,
    }
