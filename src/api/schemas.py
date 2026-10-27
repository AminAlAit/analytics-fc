"""Pydantic data schemas and validation contracts for Analytics FC REST API.

Provides strictly typed models for search parameters, player entities,
paginated collections, and ceiling similarity inference payloads.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PlayerSearchParams(BaseModel):
    """Query parameters for searching player catalog with validation boundaries."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(default="", max_length=100, description="Player search substring.")
    edition_year: int | None = Field(
        default=None, ge=2015, le=2024, description="FIFA edition year filter."
    )
    min_overall: int | None = Field(
        default=None, ge=40, le=99, description="Minimum overall rating threshold."
    )
    max_overall: int | None = Field(
        default=None, ge=40, le=99, description="Maximum overall rating threshold."
    )
    sector: str | None = Field(
        default=None, description="Tactical sector filter (e.g. Defenders)."
    )
    limit: int = Field(default=50, ge=1, le=200, description="Page size limit.")
    offset: int = Field(default=0, ge=0, description="Page offset starting index.")


class PlayerSummary(BaseModel):
    """Compact player representation returned in catalog search results."""

    model_config = ConfigDict(from_attributes=True)

    player_id: int = Field(description="Canonical unique player identifier.")
    fifa_edition_year: int = Field(description="FIFA edition year.")
    short_name: str = Field(description="Display short name.")
    long_name: str | None = Field(default=None, description="Full player legal name.")
    age: int = Field(ge=14, le=55, description="Player age at edition release.")
    nationality: str | None = Field(default=None, description="Country of nationality.")
    club: str | None = Field(default=None, description="Club team name.")
    league_name: str | None = Field(default=None, description="Domestic league name.")
    positions: str = Field(description="Comma-separated playable positions.")
    primary_pos: str = Field(default="ST", description="Primary tactical pitch position.")
    tactical_sector: str = Field(default="Center Forwards & Strikers", description="Canonical tactical sector classification.")
    preferred_foot: str | None = Field(default=None, description="Preferred kicking foot.")
    height_cm: int | None = Field(default=None, description="Player height in centimeters.")
    weight_kg: int | None = Field(default=None, description="Player weight in kilograms.")
    overall: int = Field(ge=40, le=99, description="Current overall rating.")
    potential: int = Field(ge=40, le=99, description="Stated potential rating ceiling.")
    value_eur: float | None = Field(default=None, description="Market valuation in Euros.")
    wage_eur: float | None = Field(default=None, description="Weekly wage in Euros.")
    pace: int | None = Field(default=None, ge=0, le=99, description="Pace macro rating.")
    shooting: int | None = Field(default=None, ge=0, le=99, description="Shooting macro rating.")
    passing: int | None = Field(default=None, ge=0, le=99, description="Passing macro rating.")
    dribbling: int | None = Field(default=None, ge=0, le=99, description="Dribbling macro rating.")
    defending: int | None = Field(default=None, ge=0, le=99, description="Defending macro rating.")
    physic: int | None = Field(default=None, ge=0, le=99, description="Physicality macro rating.")

    @model_validator(mode="before")
    @classmethod
    def populate_position_and_sector(cls, data: Any) -> Any:
        """Derive primary position and tactical sector if omitted from raw record."""
        if isinstance(data, dict):
            positions = data.get("positions")
            if not data.get("primary_pos") and positions:
                data["primary_pos"] = str(positions).split(",")[0].strip().upper()
            if not data.get("tactical_sector") and data.get("primary_pos"):
                from src.catalog_search import SECTOR_MAP
                data["tactical_sector"] = SECTOR_MAP.get(
                    data["primary_pos"], "Attacking Midfielders & Wingers"
                )
        return data


class PlayerDetail(PlayerSummary):
    """Complete player profile with all 34 granular technical and physical attributes."""

    attacking_crossing: int | None = Field(default=None, ge=0, le=99)
    attacking_finishing: int | None = Field(default=None, ge=0, le=99)
    attacking_heading_accuracy: int | None = Field(default=None, ge=0, le=99)
    attacking_short_passing: int | None = Field(default=None, ge=0, le=99)
    attacking_volleys: int | None = Field(default=None, ge=0, le=99)

    skill_dribbling: int | None = Field(default=None, ge=0, le=99)
    skill_curve: int | None = Field(default=None, ge=0, le=99)
    skill_fk_accuracy: int | None = Field(default=None, ge=0, le=99)
    skill_long_passing: int | None = Field(default=None, ge=0, le=99)
    skill_ball_control: int | None = Field(default=None, ge=0, le=99)

    movement_acceleration: int | None = Field(default=None, ge=0, le=99)
    movement_sprint_speed: int | None = Field(default=None, ge=0, le=99)
    movement_agility: int | None = Field(default=None, ge=0, le=99)
    movement_reactions: int | None = Field(default=None, ge=0, le=99)
    movement_balance: int | None = Field(default=None, ge=0, le=99)

    power_shot_power: int | None = Field(default=None, ge=0, le=99)
    power_jumping: int | None = Field(default=None, ge=0, le=99)
    power_stamina: int | None = Field(default=None, ge=0, le=99)
    power_strength: int | None = Field(default=None, ge=0, le=99)
    power_long_shots: int | None = Field(default=None, ge=0, le=99)

    mentality_aggression: int | None = Field(default=None, ge=0, le=99)
    mentality_interceptions: int | None = Field(default=None, ge=0, le=99)
    mentality_positioning: int | None = Field(default=None, ge=0, le=99)
    mentality_vision: int | None = Field(default=None, ge=0, le=99)
    mentality_penalties: int | None = Field(default=None, ge=0, le=99)
    mentality_composure: int | None = Field(default=None, ge=0, le=99)

    defending_marking_awareness: int | None = Field(default=None, ge=0, le=99)
    defending_standing_tackle: int | None = Field(default=None, ge=0, le=99)
    defending_sliding_tackle: int | None = Field(default=None, ge=0, le=99)

    goalkeeping_diving: int | None = Field(default=None, ge=0, le=99)
    goalkeeping_handling: int | None = Field(default=None, ge=0, le=99)
    goalkeeping_kicking: int | None = Field(default=None, ge=0, le=99)
    goalkeeping_positioning: int | None = Field(default=None, ge=0, le=99)
    goalkeeping_reflexes: int | None = Field(default=None, ge=0, le=99)


class PaginatedPlayerResponse(BaseModel):
    """Paginated collection response for catalog player queries."""

    items: list[PlayerSummary] = Field(description="List of matching player summaries.")
    total_matching: int = Field(ge=0, description="Total matching players in database.")
    limit: int = Field(ge=1, description="Requested page size.")
    offset: int = Field(ge=0, description="Requested page offset.")
    has_more: bool = Field(description="Whether additional matching pages exist.")
    sector_counts: dict[str, int] = Field(
        default_factory=dict, description="Distribution of matches across tactical sectors."
    )


class CeilingMatchRequest(BaseModel):
    """Request payload for superstar ceiling matching inference."""

    model_config = ConfigDict(extra="forbid")

    player_id: int = Field(description="Prospect canonical player identifier.")
    fifa_edition_year: int = Field(ge=2015, le=2024, description="Prospect FIFA edition year.")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of ceiling matches to return.")


class TraitGap(BaseModel):
    """Trait development gap comparing prospect against superstar prime benchmark."""

    attribute: str = Field(description="Skill attribute identifier.")
    gap: float = Field(description="Development gap difference.")


class MatchedSuperstar(BaseModel):
    """Single superstar archetype match with similarity score and development vectors."""

    rank: int = Field(ge=1, description="Similarity target rank.")
    player_id: int = Field(description="Superstar identifier.")
    name: str = Field(description="Superstar display name.")
    sector: str = Field(description="Superstar primary tactical sector.")
    peak_overall: int = Field(ge=40, le=99, description="Superstar career peak overall rating.")
    peak_age_window: str = Field(description="Career peak age window.")
    similarity_pct: float = Field(ge=0.0, le=100.0, description="Calibrated Gaussian RBF match percentage.")
    strengths: list[dict[str, Any]] = Field(default_factory=list, description="Shared core attributes.")
    growth_needed: list[TraitGap] = Field(default_factory=list, description="Priority growth attributes.")
    attributes: dict[str, float] = Field(default_factory=dict, description="Benchmark normalized skill values.")


class MacroRadarDimension(BaseModel):
    """Single dimension pair for hexagonal radar chart visualization."""

    dimension: str = Field(description="Tactical dimension label.")
    prospect_value: float = Field(ge=0.0, le=100.0, description="Prospect score.")
    benchmark_value: float = Field(ge=0.0, le=100.0, description="Superstar benchmark score.")
    gap: float = Field(description="Score difference.")


class CeilingMatchResponse(BaseModel):
    """Full projection response for superstar ceiling matching."""

    prospect: PlayerDetail = Field(description="Detailed prospect profile.")
    matches: list[MatchedSuperstar] = Field(description="Top matched superstar archetypes.")
    macro_radar: list[MacroRadarDimension] = Field(
        default_factory=list, description="Hexagonal radar comparison points."
    )
    kernel: str = Field(default="gaussian_rbf", description="Similarity kernel mathematical formulation.")
    inference_time_ms: float = Field(ge=0.0, description="Execution duration in milliseconds.")


class CatalogStatsResponse(BaseModel):
    """Diagnostic response providing catalog scale and year coverage."""

    total_rows: int = Field(ge=0, description="Total historical player records.")
    distinct_players: int = Field(ge=0, description="Distinct player entity count.")
    min_year: int = Field(ge=2015, description="Earliest FIFA edition year.")
    max_year: int = Field(le=2024, description="Latest FIFA edition year.")
