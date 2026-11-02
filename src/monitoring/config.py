"""Configuration settings, feature groupings, and comparison presets for data drift monitoring.

Defines:
1. Feature groupings: headline ratings, physical traits, financial metrics, and 34 tactical skills.
2. Edition comparison presets: reference edition (FIFA 21) versus target edition (FIFA 22).
3. Statistical drift test thresholds: Kolmogorov-Smirnov p-value and Population Stability Index (PSI).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Canonical repository paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
DEFAULT_DUCKDB_PATH: Path = BASE_DIR / "dbt_project" / "analytics_fc.duckdb"
DEFAULT_REPORT_DIR: Path = BASE_DIR / "reports" / "drift"

# Feature groupings for stg_fifa_granular
HEADLINE_NUMERICAL_FEATURES: list[str] = [
    "overall",
    "potential",
    "value_eur",
    "wage_eur",
    "age",
    "height_cm",
    "weight_kg",
]

FACE_STAT_FEATURES: list[str] = [
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",
]

TACTICAL_SKILL_FEATURES: list[str] = [
    # Attacking
    "attacking_crossing",
    "attacking_finishing",
    "attacking_heading_accuracy",
    "attacking_short_passing",
    "attacking_volleys",
    # Skill
    "skill_dribbling",
    "skill_curve",
    "skill_fk_accuracy",
    "skill_long_passing",
    "skill_ball_control",
    # Movement
    "movement_acceleration",
    "movement_sprint_speed",
    "movement_agility",
    "movement_reactions",
    "movement_balance",
    # Power
    "power_shot_power",
    "power_jumping",
    "power_stamina",
    "power_strength",
    "power_long_shots",
    # Mentality
    "mentality_aggression",
    "mentality_interceptions",
    "mentality_positioning",
    "mentality_vision",
    "mentality_penalties",
    "mentality_composure",
    # Defending
    "defending_marking_awareness",
    "defending_standing_tackle",
    "defending_sliding_tackle",
    # Goalkeeping
    "goalkeeping_diving",
    "goalkeeping_handling",
    "goalkeeping_kicking",
    "goalkeeping_positioning",
    "goalkeeping_reflexes",
]

CATEGORICAL_FEATURES: list[str] = [
    "nationality",
    "club",
    "league_name",
    "preferred_foot",
]

# Combined numerical feature list monitored for statistical distribution drift
ALL_MONITORED_NUMERICAL_FEATURES: list[str] = (
    HEADLINE_NUMERICAL_FEATURES + FACE_STAT_FEATURES + TACTICAL_SKILL_FEATURES
)


@dataclass(frozen=True)
class DriftMonitoringConfig:
    """Drift engine parameter presets, edition targets, and statistical thresholds."""

    reference_edition: int = 21
    target_edition: int = 22
    ks_p_value_threshold: float = 0.05
    psi_threshold: float = 0.10
    monitored_features: list[str] = field(
        default_factory=lambda: list(ALL_MONITORED_NUMERICAL_FEATURES)
    )
    categorical_features: list[str] = field(default_factory=lambda: list(CATEGORICAL_FEATURES))
    duckdb_path: Path = DEFAULT_DUCKDB_PATH
    report_output_dir: Path = DEFAULT_REPORT_DIR
