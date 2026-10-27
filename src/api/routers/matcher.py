"""Superstar ceiling matching vector inference router.

Executes high-dimensional Gaussian RBF similarity projections,
priority development gap computations, and hexagonal macro radar formatting.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, status
import numpy as np

from src.api.config import settings
from src.api.schemas import (
    CeilingMatchRequest,
    CeilingMatchResponse,
    MacroRadarDimension,
    MatchedSuperstar,
    PlayerDetail,
    TraitGap,
)
from src.catalog_search import (
    ALL_SKILL_COLUMNS,
    get_player_by_id_and_year,
)

logger = logging.getLogger("analytics_fc.api.matcher")

router = APIRouter(prefix="/api/v1/match", tags=["Matcher"])

# Outfield skill attributes (excluding 5 specialized goalkeeping traits)
OUTFIELD_SKILL_COLS = [c for c in ALL_SKILL_COLUMNS if not c.startswith("goalkeeping_")]

# Outfield 6-axis macro tactical radar dimensions
OUTFIELD_MACRO_DIMENSIONS: dict[str, list[str]] = {
    "Pace": ["movement_acceleration", "movement_sprint_speed"],
    "Shooting": ["attacking_finishing", "power_shot_power", "power_long_shots"],
    "Passing": ["attacking_short_passing", "skill_long_passing", "mentality_vision"],
    "Dribbling": ["skill_dribbling", "skill_ball_control", "movement_agility"],
    "Defending": ["defending_standing_tackle", "defending_marking_awareness", "mentality_interceptions"],
    "Physical": ["power_stamina", "power_strength", "power_jumping"],
}

# Goalkeeper 6-axis macro tactical radar dimensions
GOALKEEPER_MACRO_DIMENSIONS: dict[str, list[str]] = {
    "Diving": ["goalkeeping_diving"],
    "Handling": ["goalkeeping_handling"],
    "Kicking": ["goalkeeping_kicking"],
    "Reflexes": ["goalkeeping_reflexes"],
    "Positioning": ["goalkeeping_positioning"],
    "Reactions": ["movement_reactions"],
}

_GALLERY_CACHE: list[dict[str, Any]] | None = None


def get_gallery() -> list[dict[str, Any]]:
    """Retrieve superstar gallery artifact with in-memory caching."""
    global _GALLERY_CACHE
    if _GALLERY_CACHE is None:
        if not settings.gallery_path.exists():
            raise FileNotFoundError(f"Superstar gallery artifact not found at {settings.gallery_path}")
        with open(settings.gallery_path, "r", encoding="utf-8") as f:
            _GALLERY_CACHE = json.load(f)
    return _GALLERY_CACHE


def execute_vector_matching(
    prospect: dict[str, Any],
    gallery: list[dict[str, Any]],
    top_k: int = 3,
) -> tuple[list[MatchedSuperstar], list[MacroRadarDimension]]:
    """Execute high-dimensional unit vector projection and Gaussian RBF kernel similarity."""
    primary_pos = str(prospect.get("primary_pos", "ST")).upper()
    sector = prospect.get("tactical_sector", "Center Forwards & Strikers")
    is_gk = (primary_pos == "GK")

    active_cols = [c for c in ALL_SKILL_COLUMNS if c.startswith("goalkeeping_")] if is_gk else OUTFIELD_SKILL_COLS

    # 1. Construct and L2-normalize prospect directional unit vector
    p_vec = np.array([
        float(prospect[c]) if prospect.get(c) is not None else 50.0
        for c in active_cols
    ])
    p_norm = np.linalg.norm(p_vec)
    p_unit = p_vec / p_norm if p_norm > 0 else p_vec

    # 2. Project across compatible superstar gallery archetypes
    candidates: list[dict[str, Any]] = []

    for star in gallery:
        if star["player_id"] == int(prospect["player_id"]):
            continue

        star_sector = star["tactical_sector"]
        sector_match = (star_sector == sector) or (
            sector in ["Attacking Midfielders & Wingers", "Center Forwards & Strikers"]
            and star_sector in ["Attacking Midfielders & Wingers", "Center Forwards & Strikers"]
        )

        if is_gk and star_sector != "Goalkeepers":
            continue
        if not is_gk and star_sector == "Goalkeepers":
            continue
        if not sector_match:
            continue

        s_vec = np.array([
            float(star["attributes"].get(c, 50.0)) if star["attributes"].get(c) is not None else 50.0
            for c in active_cols
        ])
        s_norm = np.linalg.norm(s_vec)
        s_unit = s_vec / s_norm if s_norm > 0 else s_vec

        # Normalized Euclidean distance & Gaussian RBF calibrated percentage
        norm_dist = float(np.linalg.norm(p_unit - s_unit))
        calibrated_pct = round(float(np.exp(-(norm_dist ** 2) / (2 * (0.28 ** 2))) * 100), 1)

        # Attribute difference vectors
        trait_deltas: dict[str, float] = {}
        for col in active_cols:
            s_val = star["attributes"].get(col)
            p_val = prospect.get(col)
            if s_val is not None and p_val is not None:
                trait_deltas[col] = round(float(s_val) - float(p_val), 1)

        top_growth_needed = sorted(
            [TraitGap(attribute=k, gap=v) for k, v in trait_deltas.items() if v > 0],
            key=lambda x: x.gap,
            reverse=True,
        )[:3]

        strengths_aligned = sorted(
            [{"attribute": k, "difference": abs(v)} for k, v in trait_deltas.items()],
            key=lambda x: x["difference"],
        )[:3]

        peak_window = f"{star.get('peak_age_min', 26)} to {star.get('peak_age_max', 29)}"

        candidates.append({
            "player_id": int(star["player_id"]),
            "name": star["short_name"],
            "sector": star["tactical_sector"],
            "peak_overall": int(star["peak_overall"]),
            "peak_age_window": peak_window,
            "similarity_pct": calibrated_pct,
            "strengths": strengths_aligned,
            "growth_needed": top_growth_needed,
            "attributes": {k: float(v) for k, v in star["attributes"].items()},
        })

    candidates.sort(key=lambda x: x["similarity_pct"], reverse=True)
    top_candidates = candidates[:top_k]

    matches: list[MatchedSuperstar] = [
        MatchedSuperstar(rank=idx + 1, **cand)
        for idx, cand in enumerate(top_candidates)
    ]

    # 3. Compute 6-axis macro tactical radar comparison against top archetype
    radar_dims: list[MacroRadarDimension] = []
    if matches:
        best_star = top_candidates[0]
        macro_dict = GOALKEEPER_MACRO_DIMENSIONS if is_gk else OUTFIELD_MACRO_DIMENSIONS

        for dim, subcols in macro_dict.items():
            valid_p = [float(prospect[c]) for c in subcols if prospect.get(c) is not None]
            p_mean = float(np.mean(valid_p)) if valid_p else 50.0

            valid_s = [
                float(best_star["attributes"].get(c, 50.0))
                for c in subcols
                if best_star["attributes"].get(c) is not None
            ]
            s_mean = float(np.mean(valid_s)) if valid_s else 50.0

            radar_dims.append(
                MacroRadarDimension(
                    dimension=dim,
                    prospect_value=round(p_mean, 1),
                    benchmark_value=round(s_mean, 1),
                    gap=round(s_mean - p_mean, 1),
                )
            )

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
