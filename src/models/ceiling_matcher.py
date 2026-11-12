"""Framework-free Gaussian RBF ceiling matching and macro-radar computation.

Provides pure NumPy and standard-library vector matching over superstar_gallery.json
so both FastAPI routers and standalone Streamlit deployments share identical math
without coupling dashboard clients to web framework dependencies.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from src.catalog_search import ALL_SKILL_COLUMNS

DEFAULT_GALLERY_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "superstar_gallery.json"
)

# Outfield skill attributes (excluding 5 specialized goalkeeping traits)
OUTFIELD_SKILL_COLS: list[str] = [c for c in ALL_SKILL_COLUMNS if not c.startswith("goalkeeping_")]
GK_SKILL_COLS: list[str] = [c for c in ALL_SKILL_COLUMNS if c.startswith("goalkeeping_")]

# Outfield 6-axis macro tactical radar dimensions
OUTFIELD_MACRO_DIMENSIONS: dict[str, list[str]] = {
    "Pace": ["movement_acceleration", "movement_sprint_speed"],
    "Shooting": ["attacking_finishing", "power_shot_power", "power_long_shots"],
    "Passing": ["attacking_short_passing", "skill_long_passing", "mentality_vision"],
    "Dribbling": ["skill_dribbling", "skill_ball_control", "movement_agility"],
    "Defending": [
        "defending_standing_tackle",
        "defending_marking_awareness",
        "mentality_interceptions",
    ],
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

_GALLERY_LOCK = threading.Lock()
_GALLERY_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}


def load_gallery_cached(gallery_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load and cache superstar_gallery.json safely across threads."""
    resolved = Path(gallery_path) if gallery_path is not None else DEFAULT_GALLERY_PATH
    if not resolved.exists():
        raise FileNotFoundError(f"Superstar gallery artifact not found at {resolved}")

    mtime = resolved.stat().st_mtime
    cache_key = str(resolved.resolve())
    with _GALLERY_LOCK:
        cached = _GALLERY_CACHE.get(cache_key)
        if cached is not None and cached[0] == mtime:
            return cached[1]

        with open(resolved, encoding="utf-8") as f:
            data: list[dict[str, Any]] = json.load(f)
        _GALLERY_CACHE[cache_key] = (mtime, data)
        return data


def compute_vector_matching(
    prospect: dict[str, Any],
    gallery: list[dict[str, Any]],
    top_k: int = 3,
    sigma: float = 0.28,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute directional unit-vector projection and Gaussian RBF similarity."""
    primary_pos = str(prospect.get("primary_pos", "ST")).upper()
    sector = prospect.get("tactical_sector", "Center Forwards & Strikers")
    is_gk = primary_pos == "GK"

    active_cols = GK_SKILL_COLS if is_gk else OUTFIELD_SKILL_COLS

    # 1. Construct and L2-normalize prospect directional unit vector
    p_vec = np.array(
        [float(prospect[c]) if prospect.get(c) is not None else 50.0 for c in active_cols],
        dtype=float,
    )
    p_norm = float(np.linalg.norm(p_vec))
    p_unit = p_vec / p_norm if p_norm > 0.0 else p_vec

    # 2. Project across compatible superstar gallery archetypes
    candidates: list[dict[str, Any]] = []
    offensive_sectors = {"Attacking Midfielders & Wingers", "Center Forwards & Strikers"}

    for star in gallery:
        if int(star["player_id"]) == int(prospect["player_id"]):
            continue

        star_sector = star["tactical_sector"]
        sector_match = (star_sector == sector) or (
            sector in offensive_sectors and star_sector in offensive_sectors
        )

        if is_gk and star_sector != "Goalkeepers":
            continue
        if not is_gk and star_sector == "Goalkeepers":
            continue
        if not sector_match:
            continue

        star_attrs = star.get("attributes", {})
        s_vec = np.array(
            [
                float(star_attrs.get(c, 50.0)) if star_attrs.get(c) is not None else 50.0
                for c in active_cols
            ],
            dtype=float,
        )
        s_norm = float(np.linalg.norm(s_vec))
        s_unit = s_vec / s_norm if s_norm > 0.0 else s_vec

        norm_dist = float(np.linalg.norm(p_unit - s_unit))
        calibrated_pct = round(float(np.exp(-(norm_dist**2) / (2.0 * (sigma**2))) * 100.0), 1)

        trait_deltas: dict[str, float] = {}
        for col in active_cols:
            s_val = star_attrs.get(col)
            p_val = prospect.get(col)
            if s_val is not None and p_val is not None:
                trait_deltas[col] = round(float(s_val) - float(p_val), 1)

        growth_items: list[dict[str, Any]] = [
            {"attribute": k, "gap": v} for k, v in trait_deltas.items() if v > 0
        ]
        top_growth_needed = sorted(
            growth_items,
            key=lambda x: float(x["gap"]),
            reverse=True,
        )[:3]

        strength_items: list[dict[str, Any]] = [
            {"attribute": k, "difference": abs(v)} for k, v in trait_deltas.items()
        ]
        strengths_aligned = sorted(
            strength_items,
            key=lambda x: float(x["difference"]),
        )[:3]

        peak_window = f"{star.get('peak_age_min', 26)} to {star.get('peak_age_max', 29)}"

        candidates.append(
            {
                "player_id": int(star["player_id"]),
                "name": star["short_name"],
                "sector": star["tactical_sector"],
                "peak_overall": int(star["peak_overall"]),
                "peak_age_window": peak_window,
                "similarity_pct": calibrated_pct,
                "strengths": strengths_aligned,
                "growth_needed": top_growth_needed,
                "attributes": {k: float(v) for k, v in star_attrs.items()},
            }
        )

    candidates.sort(key=lambda x: float(x["similarity_pct"]), reverse=True)
    top_candidates = candidates[:top_k]
    matches = [{"rank": idx + 1, **cand} for idx, cand in enumerate(top_candidates)]

    # 3. Compute 6-axis macro tactical radar comparison against top archetype
    radar_dims: list[dict[str, Any]] = []
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
                {
                    "dimension": dim,
                    "prospect_value": round(p_mean, 1),
                    "benchmark_value": round(s_mean, 1),
                    "gap": round(s_mean - p_mean, 1),
                }
            )

    return matches, radar_dims
