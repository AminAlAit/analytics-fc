"""Superstar ceiling matching vector inference router.

Executes high-dimensional Gaussian RBF similarity projections,
priority development gap computations, and hexagonal macro radar formatting.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, status

from src.api.config import settings
from src.api.schemas import (
    CeilingMatchRequest,
    CeilingMatchResponse,
    MacroRadarDimension,
    MatchedSuperstar,
    PlayerDetail,
)
from src.catalog_search import get_player_by_id_and_year
from src.models.ceiling_matcher import (
    GOALKEEPER_MACRO_DIMENSIONS,
    OUTFIELD_MACRO_DIMENSIONS,
    OUTFIELD_SKILL_COLS,
    compute_vector_matching,
    load_gallery_cached,
)

logger = logging.getLogger("analytics_fc.api.matcher")

router = APIRouter(prefix="/api/v1/match", tags=["Matcher"])

__all__ = [
    "GOALKEEPER_MACRO_DIMENSIONS",
    "OUTFIELD_MACRO_DIMENSIONS",
    "OUTFIELD_SKILL_COLS",
    "execute_vector_matching",
    "get_gallery",
    "match_player_ceiling",
    "router",
]


def get_gallery() -> list[dict[str, Any]]:
    """Retrieve superstar gallery artifact with in-memory caching."""
    return load_gallery_cached(settings.gallery_path)


def execute_vector_matching(
    prospect: dict[str, Any],
    gallery: list[dict[str, Any]],
    top_k: int = 3,
) -> tuple[list[MatchedSuperstar], list[MacroRadarDimension]]:
    """Execute high-dimensional unit vector projection and Gaussian RBF kernel similarity."""
    raw_matches, raw_radar = compute_vector_matching(
        prospect=prospect, gallery=gallery, top_k=top_k
    )
    matches = [MatchedSuperstar.model_validate(item) for item in raw_matches]
    radar_dims = [MacroRadarDimension.model_validate(dim) for dim in raw_radar]
    return matches, radar_dims


@router.post(
    "/ceiling",
    response_model=CeilingMatchResponse,
    summary="Superstar Ceiling Inference",
    description="Calculate directional playstyle similarity and priority trait gaps against 100 historical superstars.",
)
def match_player_ceiling(request: CeilingMatchRequest) -> CeilingMatchResponse:
    """Project a player profile against the superstar archetype database."""
    try:
        prospect = get_player_by_id_and_year(
            player_id=request.player_id,
            fifa_edition_year=request.fifa_edition_year,
            db_path=settings.db_path,
        )
    except Exception as exc:
        logger.error(f"Failed to lookup prospect {request.player_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database lookup error: {exc}",
        ) from exc

    if prospect is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Player ID {request.player_id} not found in FIFA {request.fifa_edition_year} edition.",
        )

    try:
        gallery = get_gallery()
        t_start = time.perf_counter()
        matches, radar_dims = execute_vector_matching(
            prospect=prospect,
            gallery=gallery,
            top_k=request.top_k,
        )
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    except Exception as exc:
        logger.error(f"Vector inference failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution error: {exc}",
        ) from exc

    return CeilingMatchResponse(
        prospect=PlayerDetail.model_validate(prospect),
        matches=matches,
        macro_radar=radar_dims,
        kernel="gaussian_rbf",
        inference_time_ms=round(elapsed_ms, 2),
    )
