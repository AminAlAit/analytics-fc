"""High-dimensional archetype vector index for prospect ceiling matching.

Builds an in-memory normalized vector index using Gaussian RBF similarity to match
youth prospects against prime superstar archetypes with tactical role penalties.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("analytics_fc.models.archetype_indexer")

# Positional sector taxonomy
SECTOR_MAP: dict[str, str] = {
    "GK": "Goalkeepers",
    "CB": "Defenders",
    "LB": "Defenders",
    "RB": "Defenders",
    "LWB": "Defenders",
    "RWB": "Defenders",
    "CDM": "Defensive & Central Midfielders",
    "CM": "Defensive & Central Midfielders",
    "CAM": "Attacking Midfielders & Wingers",
    "LM": "Attacking Midfielders & Wingers",
    "RM": "Attacking Midfielders & Wingers",
    "LW": "Attacking Midfielders & Wingers",
    "RW": "Attacking Midfielders & Wingers",
    "CF": "Center Forwards & Strikers",
    "ST": "Center Forwards & Strikers",
}

# Granular outfield skill attributes (29 features)
OUTFIELD_SKILL_COLS: list[str] = [
    "attacking_crossing",
    "attacking_finishing",
    "attacking_heading_accuracy",
    "attacking_short_passing",
    "attacking_volleys",
    "skill_dribbling",
    "skill_curve",
    "skill_fk_accuracy",
    "skill_long_passing",
    "skill_ball_control",
    "movement_acceleration",
    "movement_sprint_speed",
    "movement_agility",
    "movement_reactions",
    "movement_balance",
    "power_shot_power",
    "power_jumping",
    "power_stamina",
    "power_strength",
    "power_long_shots",
    "mentality_aggression",
    "mentality_interceptions",
    "mentality_positioning",
    "mentality_vision",
    "mentality_penalties",
    "mentality_composure",
    "defending_marking_awareness",
    "defending_standing_tackle",
    "defending_sliding_tackle",
]

# Granular goalkeeping skill attributes (5 features)
GK_SKILL_COLS: list[str] = [
    "goalkeeping_diving",
    "goalkeeping_handling",
    "goalkeeping_kicking",
    "goalkeeping_positioning",
    "goalkeeping_reflexes",
]

# Inter-sector tactical incompatibility distance penalty matrix
# Values scale distance: dist_penalized = dist * (1.0 + penalty)
SECTOR_DISTANCE_PENALTIES: dict[tuple[str, str], float] = {
    ("Defenders", "Defenders"): 0.0,
    ("Defensive & Central Midfielders", "Defensive & Central Midfielders"): 0.0,
    ("Attacking Midfielders & Wingers", "Attacking Midfielders & Wingers"): 0.0,
    ("Center Forwards & Strikers", "Center Forwards & Strikers"): 0.0,
    ("Goalkeepers", "Goalkeepers"): 0.0,
    # Fluid offensive coupling
    ("Attacking Midfielders & Wingers", "Center Forwards & Strikers"): 0.05,
    ("Center Forwards & Strikers", "Attacking Midfielders & Wingers"): 0.05,
    # Midfield to adjacent lines
    ("Defensive & Central Midfielders", "Defenders"): 0.20,
    ("Defenders", "Defensive & Central Midfielders"): 0.20,
    ("Defensive & Central Midfielders", "Attacking Midfielders & Wingers"): 0.15,
    ("Attacking Midfielders & Wingers", "Defensive & Central Midfielders"): 0.15,
    # Distant outfield transitions
    ("Defenders", "Center Forwards & Strikers"): 0.60,
    ("Center Forwards & Strikers", "Defenders"): 0.60,
    ("Defenders", "Attacking Midfielders & Wingers"): 0.40,
    ("Attacking Midfielders & Wingers", "Defenders"): 0.40,
    ("Defensive & Central Midfielders", "Center Forwards & Strikers"): 0.35,
    ("Center Forwards & Strikers", "Defensive & Central Midfielders"): 0.35,
}


class ArchetypeMatchResult(BaseModel):
    """Structured result of an archetype similarity match."""

    model_config = ConfigDict(frozen=True)

    player_id: int
    superstar_name: str
    tactical_sector: str
    primary_position: str
    peak_overall: int
    similarity_score_pct: float = Field(ge=0.0, le=100.0)
    raw_distance: float
    penalized_distance: float
    top_growth_priorities: list[dict[str, Any]]
    closest_matching_traits: list[dict[str, Any]]


class ArchetypeVectorIndex:
    """In-memory normalized vector index using Gaussian RBF similarity."""

    def __init__(
        self,
        gallery_path: str | Path | None = None,
        sigma: float = 0.28,
    ) -> None:
        self.sigma = sigma
        self.gallery: list[dict[str, Any]] = []
        self.outfield_metadata: list[dict[str, Any]] = []
        self.gk_metadata: list[dict[str, Any]] = []
        self.outfield_matrix: np.ndarray = np.empty((0, len(OUTFIELD_SKILL_COLS)), dtype=float)
        self.gk_matrix: np.ndarray = np.empty((0, len(GK_SKILL_COLS)), dtype=float)
        self._is_loaded: bool = False

        if gallery_path is not None:
            self.load_gallery(gallery_path)
        else:
            default_path = Path("data/processed/superstar_gallery.json")
            if default_path.exists():
                self.load_gallery(default_path)

    @property
    def is_loaded(self) -> bool:
        """Return True if index matrices have been loaded and normalized."""
        return self._is_loaded

    def load_gallery(self, path: str | Path) -> None:
        """Load superstar gallery JSON and build normalized numpy index matrices."""
        gallery_file = Path(path)
        if not gallery_file.exists():
            raise FileNotFoundError(f"Superstar gallery file not found at {gallery_file}")

        with open(gallery_file, encoding="utf-8") as f:
            self.gallery = json.load(f)

        if not self.gallery:
            raise ValueError(f"Empty gallery loaded from {gallery_file}")

        outfield_rows: list[list[float]] = []
        gk_rows: list[list[float]] = []
        self.outfield_metadata = []
        self.gk_metadata = []

        for star in self.gallery:
            pos = str(star.get("primary_position", "")).strip().upper()
            sector = star.get("tactical_sector", SECTOR_MAP.get(pos, "Defenders"))
            attrs = star.get("attributes", {})

            if sector == "Goalkeepers" or pos == "GK":
                vec = [float(attrs.get(c, 50.0)) for c in GK_SKILL_COLS]
                gk_rows.append(vec)
                self.gk_metadata.append(star)
            else:
                vec = [float(attrs.get(c, 50.0)) for c in OUTFIELD_SKILL_COLS]
                outfield_rows.append(vec)
                self.outfield_metadata.append(star)

        # Normalize outfield matrix to unit vectors
        if outfield_rows:
            raw_outfield = np.array(outfield_rows, dtype=float)
            norms = np.linalg.norm(raw_outfield, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            self.outfield_matrix = raw_outfield / norms

        # Normalize GK matrix to unit vectors
        if gk_rows:
            raw_gk = np.array(gk_rows, dtype=float)
            norms = np.linalg.norm(raw_gk, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            self.gk_matrix = raw_gk / norms

        self._is_loaded = True
        logger.info(
            "ArchetypeVectorIndex loaded %d superstars (%d outfield, %d GK).",
            len(self.gallery),
            len(self.outfield_metadata),
            len(self.gk_metadata),
        )

    def calculate_rbf_similarity(
        self,
        norm_distances: np.ndarray,
        sigma: float | None = None,
    ) -> np.ndarray:
        """Compute calibrated similarity percentage via Gaussian RBF kernel.

        K(d) = exp(- (d^2) / (2 * sigma^2)) * 100.0
        """
        s = sigma if sigma is not None else self.sigma
        return np.exp(-(norm_distances**2) / (2.0 * (s**2))) * 100.0

    def query(
        self,
        attributes: dict[str, float] | np.ndarray,
        primary_position: str,
        top_k: int = 3,
        exclude_player_id: int | None = None,
        apply_penalties: bool = True,
    ) -> list[ArchetypeMatchResult]:
        """Query nearest prime superstar archetypes using Gaussian RBF similarity."""
        if not self.is_loaded:
            raise RuntimeError("ArchetypeVectorIndex is not loaded. Call load_gallery() first.")

        clean_pos = primary_position.strip().upper()
        is_gk = clean_pos == "GK"
        query_sector = SECTOR_MAP.get(clean_pos, "Defenders" if not is_gk else "Goalkeepers")

        active_cols = GK_SKILL_COLS if is_gk else OUTFIELD_SKILL_COLS
        target_matrix = self.gk_matrix if is_gk else self.outfield_matrix
        target_meta = self.gk_metadata if is_gk else self.outfield_metadata

        if target_matrix.shape[0] == 0:
            return []

        # Vectorize input query
        if isinstance(attributes, dict):
            raw_q = np.array([float(attributes.get(c, 50.0)) for c in active_cols], dtype=float)
        else:
            raw_q = np.asarray(attributes, dtype=float)
            if len(raw_q) != len(active_cols):
                raise ValueError(
                    f"Attribute vector dimension {len(raw_q)} does not match {len(active_cols)}"
                )

        # L2 Directional Normalization
        q_norm = np.linalg.norm(raw_q)
        q_unit = raw_q / q_norm if q_norm > 0.0 else raw_q

        # Vectorized Euclidean distance between unit vectors: ||u_star - u_query||
        diffs = target_matrix - q_unit
        raw_distances = np.linalg.norm(diffs, axis=1)

        # Apply tactical role incompatibility distance penalties
        penalized_distances = raw_distances.copy()
        if apply_penalties and not is_gk:
            for idx, star in enumerate(target_meta):
                star_sector = star.get("tactical_sector", query_sector)
                penalty = SECTOR_DISTANCE_PENALTIES.get((query_sector, star_sector), 0.30)
                if penalty > 0.0:
                    penalized_distances[idx] = float(
                        np.sqrt(raw_distances[idx] ** 2 + (penalty * 0.20) ** 2)
                    )

        # Compute calibrated Gaussian RBF percentage
        similarities = self.calculate_rbf_similarity(penalized_distances)

        candidates: list[ArchetypeMatchResult] = []
        for idx, star in enumerate(target_meta):
            if exclude_player_id is not None and star["player_id"] == exclude_player_id:
                continue

            star_attrs = star.get("attributes", {})
            star_vec = [float(star_attrs.get(c, 50.0)) for c in active_cols]

            trait_deltas = {
                col: round(float(star_vec[i]) - float(raw_q[i]), 1)
                for i, col in enumerate(active_cols)
            }

            top_growth: list[dict[str, Any]] = [
                {"attribute": k, "gap": v}
                for k, v in sorted(trait_deltas.items(), key=lambda item: item[1], reverse=True)
                if v > 0
            ][:3]

            closest_traits: list[dict[str, Any]] = [
                {"attribute": k, "difference": round(abs(v), 1)}
                for k, v in sorted(trait_deltas.items(), key=lambda item: abs(item[1]))
            ][:3]

            candidates.append(
                ArchetypeMatchResult(
                    player_id=star["player_id"],
                    superstar_name=star["short_name"],
                    tactical_sector=star.get("tactical_sector", query_sector),
                    primary_position=star.get("primary_position", clean_pos),
                    peak_overall=star.get("peak_overall", 85),
                    similarity_score_pct=round(float(similarities[idx]), 1),
                    raw_distance=round(float(raw_distances[idx]), 4),
                    penalized_distance=round(float(penalized_distances[idx]), 4),
                    top_growth_priorities=top_growth,
                    closest_matching_traits=closest_traits,
                )
            )

        # Sort descending by calibrated similarity percentage
        candidates.sort(key=lambda x: x.similarity_score_pct, reverse=True)
        return candidates[:top_k]
