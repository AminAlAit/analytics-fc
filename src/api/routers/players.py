"""Player catalog search and detail retrieval endpoints.

Provides high-performance paginated search across 180,021 historical records
and granular profile retrieval with all 34 skill attributes.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Path as PathParam, Query, status

from src.api.config import settings
from src.api.schemas import (
    CatalogStatsResponse,
    PaginatedPlayerResponse,
    PlayerDetail,
)
from src.catalog_search import (
    get_catalog_stats,
    get_player_by_id_and_year,
    search_players_paginated,
)

logger = logging.getLogger("analytics_fc.api.players")

router = APIRouter(prefix="/api/v1/players", tags=["Players"])


@router.get(
    "/search",
    response_model=PaginatedPlayerResponse,
    summary="Search Player Catalog",
    description="Search players by name substring, edition year, overall rating range, and tactical sector with pagination.",
)
def search_catalog_players(
    query: Annotated[
        str, Query(max_length=100, description="Player search substring.")
    ] = "",
    edition_year: Annotated[
        int | None, Query(ge=2015, le=2024, description="FIFA edition year filter.")
    ] = None,
    min_overall: Annotated[
        int | None, Query(ge=40, le=99, description="Minimum overall rating threshold.")
    ] = None,
    max_overall: Annotated[
        int | None, Query(ge=40, le=99, description="Maximum overall rating threshold.")
    ] = None,
    sector: Annotated[
        str | None, Query(description="Tactical sector filter (e.g. Defenders).")
    ] = None,
    limit: Annotated[
        int, Query(ge=1, le=200, description="Maximum page size limit.")
    ] = 50,
    offset: Annotated[
        int, Query(ge=0, description="Initial record offset.")
    ] = 0,
    include_sector_counts: Annotated[
        bool, Query(description="Include tactical sector distribution counts.")
    ] = True,
) -> PaginatedPlayerResponse:
    """Execute a validated paginated search across the historical player database."""
    try:
        result = search_players_paginated(
            query=query,
            edition_year=edition_year,
            min_overall=min_overall,
            max_overall=max_overall,
            sector=sector,
            limit=limit,
            offset=offset,
            include_sector_counts=include_sector_counts,
            db_path=settings.db_path,
        )
        return PaginatedPlayerResponse.model_validate(result)
    except Exception as exc:
        logger.error(f"Catalog search failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database search execution error: {exc}",
        ) from exc


@router.get(
    "/stats/catalog",
    response_model=CatalogStatsResponse,
    summary="Catalog Global Statistics",
    description="Retrieve verified total row count, distinct player entities, and edition boundaries.",
)
def get_catalog_statistics() -> CatalogStatsResponse:
    """Retrieve verified catalog volume metrics."""
    try:
        stats = get_catalog_stats(db_path=settings.db_path)
        return CatalogStatsResponse.model_validate(stats)
    except Exception as exc:
        logger.error(f"Catalog statistics lookup failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve catalog stats: {exc}",
        ) from exc


@router.get(
    "/{player_id}",
    response_model=PlayerDetail,
    summary="Get Player Detail Profile",
    description="Retrieve complete player profile including all 34 granular skill attributes for a specific edition.",
)
def get_player_profile(
    player_id: Annotated[
        int, PathParam(description="Canonical unique player identifier.")
    ],
    edition_year: Annotated[
        int, Query(ge=2015, le=2024, description="FIFA edition year.")
    ] = 2022,
) -> PlayerDetail:
    """Retrieve full player profile or return 404 if not found."""
    try:
        profile = get_player_by_id_and_year(
            player_id=player_id,
            fifa_edition_year=edition_year,
            db_path=settings.db_path,
        )
    except Exception as exc:
        logger.error(f"Error fetching player profile for {player_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database read error: {exc}",
        ) from exc

    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Player ID {player_id} not found in FIFA {edition_year} edition.",
        )

    return PlayerDetail.model_validate(profile)
