import json
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.api_client import (
    forecast_trajectory,
    get_catalog_stats_summary,
    get_connection_status,
    get_player_profile,
    load_trajectory_models,
    match_ceiling,
    search_players,
)
from dashboard.bootstrap import ensure_serving_assets
from src.catalog_search import (
    SECTOR_MAP,
    SECTOR_ORDER,
)

# Injected Custom CSS: Tactical Command Dark Design System
DARK_CSS = """
<style>
    /* Global Base and Pitch Dark Canvas */
    .stApp {
        background-color: #0A0E17;
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Streamlit Chrome Overrides */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Executive Hero Status Bar */
    .terminal-bar {
        background: linear-gradient(90deg, #111827 0%, #161F30 100%);
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 0.85rem 1.25rem;
        margin-bottom: 1.25rem;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        gap: 0.75rem;
    }
    .brand-title {
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #F8FAFC;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .brand-accent {
        color: #10B981;
        font-family: 'JetBrains Mono', 'Roboto Mono', monospace;
    }
    .telemetry-group {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        align-items: center;
    }
    .telemetry-chip {
        font-family: 'JetBrains Mono', 'Roboto Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.25rem 0.65rem;
        border-radius: 4px;
        background-color: #0F172A;
        border: 1px solid #1E293B;
        color: #94A3B8;
        letter-spacing: 0.04em;
    }
    .telemetry-chip.live {
        border-color: rgba(16, 185, 129, 0.4);
        color: #10B981;
    }

    /* Tactical Dossier Card */
    .dossier-card {
        background-color: #111827;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
    }
    .dossier-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 1rem;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 0.85rem;
    }
    .player-name-display {
        font-size: 1.75rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.2;
    }
    .player-sub-meta {
        color: #94A3B8;
        font-size: 0.85rem;
        margin-top: 0.25rem;
        display: flex;
        gap: 0.75rem;
        align-items: center;
    }
    .meta-tag {
        background: #1E293B;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        color: #CBD5E1;
        font-size: 0.75rem;
    }

    /* KPI Pods */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.85rem;
        margin-bottom: 0.5rem;
    }
    .kpi-pod {
        background-color: #161F30;
        border: 1px solid #23324A;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        text-align: left;
    }
    .kpi-label {
        font-size: 0.7rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }
    .kpi-val {
        font-family: 'JetBrains Mono', 'Roboto Mono', monospace;
        font-size: 1.85rem;
        font-weight: 800;
        line-height: 1;
        font-variant-numeric: tabular-nums;
    }
    .kpi-val.emerald {
        color: #10B981;
    }
    .kpi-val.cyan {
        color: #06B6D4;
    }
    .kpi-val.amber {
        color: #F59E0B;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: #94A3B8;
        margin-top: 0.35rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Section Subheadings */
    .section-header-hud {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        font-size: 1.05rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 1.5rem;
        margin-bottom: 0.85rem;
        letter-spacing: -0.01em;
    }
    .section-prefix {
        color: #10B981;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
    }

    /* Archetype Pods */
    .archetype-card-dark {
        background-color: #111827;
        border: 1px solid #1E293B;
        border-top: 3px solid #10B981;
        border-radius: 8px;
        padding: 1.15rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .archetype-card-dark.rank-2 {
        border-top: 3px solid #06B6D4;
    }
    .archetype-card-dark.rank-3 {
        border-top: 3px solid #64748B;
    }
    .pod-topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.65rem;
    }
    .pod-tier-chip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        background: #1E293B;
        color: #94A3B8;
    }
    .pod-pct-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.4rem;
        font-weight: 800;
        color: #10B981;
        font-variant-numeric: tabular-nums;
    }
    .pod-pct-badge.cyan {
        color: #06B6D4;
    }
    .confidence-meter-track {
        width: 100%;
        height: 5px;
        background-color: #1E293B;
        border-radius: 999px;
        margin-bottom: 0.85rem;
        overflow: hidden;
    }
    .confidence-meter-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #10B981, #06B6D4);
    }
    .star-name {
        font-size: 1.25rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 0 0 0.25rem 0;
    }
    .star-meta {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-bottom: 0.85rem;
    }
    .trait-badge-group {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin-top: 0.35rem;
        margin-bottom: 0.75rem;
    }
    .trait-pill-aligned {
        font-size: 0.72rem;
        font-weight: 600;
        background-color: rgba(16, 185, 129, 0.12);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
    }
    .trait-pill-growth {
        font-size: 0.72rem;
        font-weight: 600;
        background-color: rgba(245, 158, 11, 0.12);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
    }

    /* Tactical Gap Telemetry Table */
    .gap-metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.45rem 0.75rem;
        background-color: #161F30;
        border: 1px solid #1E293B;
        border-radius: 4px;
        margin-bottom: 0.4rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
    }
    .gap-val-p {
        color: #10B981;
        font-weight: 700;
    }
    .gap-val-s {
        color: #06B6D4;
        font-weight: 700;
    }
    .gap-delta {
        font-weight: 700;
    }
    .gap-delta.pos {
        color: #10B981;
    }
    .gap-delta.neg {
        color: #F59E0B;
    }

    /* Audit Terminal Box */
    .audit-terminal-box {
        background-color: #0B111D;
        border: 1px solid #1E293B;
        border-left: 3px solid #F59E0B;
        border-radius: 6px;
        padding: 1.25rem;
        font-family: 'JetBrains Mono', 'Roboto Mono', monospace;
        font-size: 0.85rem;
        line-height: 1.6;
        color: #CBD5E1;
    }

    /* Tab Custom Styling */
    button[data-baseweb="tab"] {
        color: #94A3B8 !important;
        font-weight: 600 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #10B981 !important;
        border-bottom-color: #10B981 !important;
    }
</style>
"""


def _configure_page() -> None:
    """Configure Streamlit page metadata and inject pitch-dark styling."""
    st.set_page_config(
        page_title="Analytics FC | Tactical Intelligence Terminal",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(DARK_CSS, unsafe_allow_html=True)


# ==============================================================================
# PATHS AND TAXONOMY CONSTANTS
# ==============================================================================
# Path setup and imports moved to top of file

DUCKDB_PATH = BASE_DIR / "dbt_project" / "analytics_fc.duckdb"
GALLERY_PATH = BASE_DIR / "data" / "processed" / "superstar_gallery.json"
DRIFT_METRICS_PATH = BASE_DIR / "data" / "monitoring" / "drift_metrics.json"

SECTOR_MAP_LEGACY = {
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

SKILL_COLS = [
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
    "goalkeeping_diving",
    "goalkeeping_handling",
    "goalkeeping_kicking",
    "goalkeeping_positioning",
    "goalkeeping_reflexes",
]
OUTFIELD_COLS = [col for col in SKILL_COLS if not col.startswith("goalkeeping_")]


# ==============================================================================
# HARDWARE SAFE RESOURCE LOADING
# ==============================================================================
@st.cache_data
def load_gallery():
    if not GALLERY_PATH.exists():
        return []
    with open(GALLERY_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_drift_metrics(path: Path | str | None = None) -> dict[str, Any] | None:
    """Load exported drift metrics JSON snapshot for dashboard display."""
    target = Path(path) if path else DRIFT_METRICS_PATH
    if not target.exists():
        return None
    try:
        with open(target, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data
def cached_search_players(
    query: str = "",
    edition_year: int | None = None,
    min_overall: int | None = None,
    sector: str | None = None,
    limit: int = 50,
):
    return search_players(
        query=query,
        edition_year=edition_year,
        min_overall=min_overall,
        sector=sector,
        limit=limit,
        include_sector_counts=True,
    )


@st.cache_data
def cached_get_player_profile(player_id: int, fifa_edition_year: int):
    return get_player_profile(player_id, fifa_edition_year)


@st.cache_data
def cached_match_ceiling(player_id: int, fifa_edition_year: int, top_k: int = 3):
    return match_ceiling(player_id, fifa_edition_year, top_k)


@st.cache_data
def cached_catalog_stats() -> dict[str, int]:
    """Retrieve cached catalog telemetry counts."""
    return get_catalog_stats_summary()


@st.cache_resource
def _cached_load_trajectory_models() -> tuple[Any, Any] | None:
    """Load and cache Model 2 quantile and conformal artifacts."""
    try:
        return load_trajectory_models()
    except FileNotFoundError:
        return None


def get_cached_trajectory_models() -> tuple[Any, Any] | None:
    """Return cached Model 2 artifacts, clearing cache if missing."""
    loaded = _cached_load_trajectory_models()
    if loaded is None:
        _cached_load_trajectory_models.clear()
    return loaded


def build_trajectory_fan_data(forecast_resp: dict[str, Any]) -> pd.DataFrame:
    """Convert Model 2 forecast payload into step-0 anchored DataFrame for Altair."""
    ref_year = int(forecast_resp["reference_edition_year"])
    curr_age = int(forecast_resp["current_age"])
    curr_ovr = float(forecast_resp["current_overall"])
    potential = float(forecast_resp["potential"])
    points = forecast_resp.get("forecast", [])
    has_conformal = any(
        pt.get("conformal_lower") is not None and pt.get("conformal_upper") is not None
        for pt in points
    )

    rows: list[dict[str, Any]] = [
        {
            "year": ref_year,
            "age": curr_age,
            "step": 0,
            "q10": curr_ovr,
            "q50": curr_ovr,
            "q90": curr_ovr,
            "potential": potential,
            "conformal_lower": curr_ovr if has_conformal else np.nan,
            "conformal_upper": curr_ovr if has_conformal else np.nan,
        }
    ]
    for idx, pt in enumerate(points, start=1):
        c_low = pt.get("conformal_lower")
        c_high = pt.get("conformal_upper")
        rows.append(
            {
                "year": int(pt["target_edition_year"]),
                "age": int(pt["projected_age"]),
                "step": idx,
                "q10": float(pt["q10_overall"]),
                "q50": float(pt["q50_overall"]),
                "q90": float(pt["q90_overall"]),
                "potential": potential,
                "conformal_lower": float(c_low) if c_low is not None else np.nan,
                "conformal_upper": float(c_high) if c_high is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_trajectory_fan_chart(forecast_resp: dict[str, Any]) -> alt.LayerChart:
    """Build Pitch-Dark Altair quantile + conformal trajectory fan chart."""
    df = build_trajectory_fan_data(forecast_resp)
    all_vals: list[float] = [
        *df["q10"].dropna().tolist(),
        *df["q90"].dropna().tolist(),
        *df["potential"].dropna().tolist(),
        *df["conformal_lower"].dropna().tolist(),
        *df["conformal_upper"].dropna().tolist(),
    ]
    y_min = max(35.0, float(np.floor(min(all_vals) - 3.0)))
    y_max = min(99.0, float(np.ceil(max(all_vals) + 3.0)))

    x_enc = alt.X("year:O", title="FIFA Edition Year", axis=alt.Axis(labelAngle=0))
    y_scale = alt.Scale(domain=[y_min, y_max])

    layers: list[alt.Chart] = []

    df_conf = df.dropna(subset=["conformal_lower", "conformal_upper"])
    if not df_conf.empty:
        conformal_band = (
            alt.Chart(df_conf)
            .mark_area(color="#06B6D4", opacity=0.16)
            .encode(
                x=x_enc,
                y=alt.Y("conformal_lower:Q", title="Overall Rating (OVR)", scale=y_scale),
                y2="conformal_upper:Q",
                tooltip=[
                    alt.Tooltip("year:O", title="Year"),
                    alt.Tooltip("conformal_lower:Q", title="Conformal Low", format=".1f"),
                    alt.Tooltip("conformal_upper:Q", title="Conformal High", format=".1f"),
                ],
            )
        )
        layers.append(conformal_band)

    quantile_band = (
        alt.Chart(df)
        .mark_area(color="#10B981", opacity=0.25)
        .encode(
            x=x_enc,
            y=alt.Y("q10:Q", title="Overall Rating (OVR)", scale=y_scale),
            y2="q90:Q",
            tooltip=[
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip("q10:Q", title="P10 Floor", format=".1f"),
                alt.Tooltip("q90:Q", title="P90 Ceiling", format=".1f"),
            ],
        )
    )
    layers.append(quantile_band)

    potential_line = (
        alt.Chart(df)
        .mark_line(color="#F59E0B", strokeDash=[5, 4], strokeWidth=2)
        .encode(
            x=x_enc,
            y=alt.Y("potential:Q", scale=y_scale),
            tooltip=[alt.Tooltip("potential:Q", title="Stated Potential", format=".0f")],
        )
    )
    layers.append(potential_line)

    median_line = (
        alt.Chart(df)
        .mark_line(color="#10B981", strokeWidth=3)
        .encode(
            x=x_enc,
            y=alt.Y("q50:Q", scale=y_scale),
        )
    )
    median_points = (
        alt.Chart(df)
        .mark_circle(color="#10B981", size=85)
        .encode(
            x=x_enc,
            y=alt.Y("q50:Q", scale=y_scale),
            tooltip=[
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip("age:Q", title="Age"),
                alt.Tooltip("q50:Q", title="Median (P50)", format=".1f"),
                alt.Tooltip("q10:Q", title="P10", format=".1f"),
                alt.Tooltip("q90:Q", title="P90", format=".1f"),
            ],
        )
    )
    layers.extend([median_line, median_points])

    return (
        alt.layer(*layers)
        .properties(height=360)
        .configure_view(strokeWidth=0)
        .configure(background="#0A0E17")
        .configure_axis(
            gridColor="#1E293B",
            domainColor="#334155",
            labelColor="#94A3B8",
            titleColor="#94A3B8",
        )
    )


def format_currency_eur(val):
    if pd.isna(val) or val <= 0:
        return "N/A"
    if val >= 1_000_000:
        return f"€{val / 1_000_000:.1f}M"
    if val >= 1_000:
        return f"€{val / 1_000:.0f}K"
    return f"€{int(val)}"


# ==============================================================================
# ALTAIR HEXAGONAL RADAR BUILDER
# ==============================================================================
def build_tactical_radar_altair(
    categories, p_vals, s_vals, p_name="Prospect", s_name="Prime Benchmark"
):
    n = len(categories)
    angles = [i * 2 * np.pi / n - np.pi / 2 for i in range(n)]

    # 1. Concentric Hexagonal Grid Lines (20, 40, 60, 80, 100)
    grid_rows = []
    for level in [20, 40, 60, 80, 100]:
        for ang in angles:
            grid_rows.append(
                {"x": level * np.cos(ang), "y": level * np.sin(ang), "level": str(level)}
            )
        grid_rows.append(
            {"x": level * np.cos(angles[0]), "y": level * np.sin(angles[0]), "level": str(level)}
        )
    df_grid = pd.DataFrame(grid_rows)

    grid_chart = (
        alt.Chart(df_grid)
        .mark_line(color="#223147", strokeWidth=1, strokeDash=[3, 3])
        .encode(
            x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-150, 150])),
            y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-150, 150])),
            detail="level:N",
        )
    )

    # 2. Axis Spokes
    spoke_rows = []
    for i, ang in enumerate(angles):
        spoke_rows.append({"x": 0, "y": 0, "spoke": str(i)})
        spoke_rows.append({"x": 100 * np.cos(ang), "y": 100 * np.sin(ang), "spoke": str(i)})
    df_spokes = pd.DataFrame(spoke_rows)

    spoke_chart = (
        alt.Chart(df_spokes)
        .mark_line(color="#1E293B", strokeWidth=1)
        .encode(x="x:Q", y="y:Q", detail="spoke:N")
    )

    # 3. Prospect Polygon (Emerald #10B981)
    p_rows = [
        {
            "x": v * np.cos(ang),
            "y": v * np.sin(ang),
            "entity": p_name,
            "order": i,
            "val": v,
            "dimension": c,
        }
        for i, (c, v, ang) in enumerate(zip(categories, p_vals, angles))
    ]
    p_rows.append(
        {
            "x": p_vals[0] * np.cos(angles[0]),
            "y": p_vals[0] * np.sin(angles[0]),
            "entity": p_name,
            "order": n,
            "val": p_vals[0],
            "dimension": categories[0],
        }
    )
    df_p = pd.DataFrame(p_rows)

    p_line = (
        alt.Chart(df_p)
        .mark_line(color="#10B981", strokeWidth=3)
        .encode(x="x:Q", y="y:Q", order="order:Q")
    )
    p_points = (
        alt.Chart(df_p)
        .mark_circle(color="#10B981", size=60)
        .encode(x="x:Q", y="y:Q", tooltip=["dimension:N", "val:Q"])
    )

    # 4. Prime Superstar Archetype Polygon (Cyan #06B6D4)
    s_rows = [
        {
            "x": v * np.cos(ang),
            "y": v * np.sin(ang),
            "entity": s_name,
            "order": i,
            "val": v,
            "dimension": c,
        }
        for i, (c, v, ang) in enumerate(zip(categories, s_vals, angles))
    ]
    s_rows.append(
        {
            "x": s_vals[0] * np.cos(angles[0]),
            "y": s_vals[0] * np.sin(angles[0]),
            "entity": s_name,
            "order": n,
            "val": s_vals[0],
            "dimension": categories[0],
        }
    )
    df_s = pd.DataFrame(s_rows)

    s_line = (
        alt.Chart(df_s)
        .mark_line(color="#06B6D4", strokeWidth=2.5, strokeDash=[4, 4])
        .encode(x="x:Q", y="y:Q", order="order:Q")
    )
    s_points = (
        alt.Chart(df_s)
        .mark_circle(color="#06B6D4", size=50)
        .encode(x="x:Q", y="y:Q", tooltip=["dimension:N", "val:Q"])
    )

    # 5. Dimension Perimeter Labels
    label_rows = []
    for cat, p, s, ang in zip(categories, p_vals, s_vals, angles):
        p_clean = 50.0 if (pd.isna(p) or np.isnan(p)) else p
        s_clean = 50.0 if (pd.isna(s) or np.isnan(s)) else s
        label_rows.append(
            {
                "x": 122 * np.cos(ang),
                "y": 122 * np.sin(ang),
                "label": f"{cat}: {int(round(p_clean))} | {int(round(s_clean))}",
            }
        )
    df_labels = pd.DataFrame(label_rows)

    labels_chart = (
        alt.Chart(df_labels)
        .mark_text(color="#94A3B8", fontSize=11, fontWeight=700)
        .encode(x="x:Q", y="y:Q", text="label:N")
    )

    combined_chart = (
        (grid_chart + spoke_chart + p_line + p_points + s_line + s_points + labels_chart)
        .properties(height=420)
        .configure_view(strokeWidth=0)
        .configure(background="#0A0E17")
    )
    return combined_chart


def _render_prospect_dossier_and_matcher(cand_selected: dict[str, Any]) -> dict[str, Any]:
    """Render the selected player dossier, top-3 archetypes, radar, and trait audit."""
    prospect_full = cached_get_player_profile(
        cand_selected["player_id"], cand_selected["fifa_edition_year"]
    )
    prospect = prospect_full if prospect_full is not None else cand_selected

    primary_pos = str(prospect["positions"]).split(",")[0].strip().upper()
    sector = SECTOR_MAP.get(primary_pos, "Attacking Midfielders & Wingers")
    growth_delta = int(prospect["potential"]) - int(prospect["overall"])
    growth_disp = f"+{growth_delta}" if growth_delta >= 0 else str(growth_delta)

    formatted_val = format_currency_eur(prospect["value_eur"])
    height_disp = (
        f"{int(prospect['height_cm'])} cm" if pd.notna(prospect.get("height_cm")) else "N/A"
    )
    weight_disp = (
        f"{int(prospect['weight_kg'])} kg" if pd.notna(prospect.get("weight_kg")) else "N/A"
    )
    st.markdown(
        f"""
    <div class="dossier-card">
        <div class="dossier-header">
            <div>
                <h2 class="player-name-display">{prospect["short_name"]}</h2>
                <div class="player-sub-meta">
                    <span class="meta-tag">FIFA {prospect["fifa_edition_year"]}</span>
                    <span><b>Club:</b> {prospect["club"]}</span>
                    <span>•</span>
                    <span><b>League:</b> {prospect["league_name"]}</span>
                    <span>•</span>
                    <span><b>Nationality:</b> {prospect["nationality"]}</span>
                </div>
            </div>
            <div style="text-align: right;">
                <span class="meta-tag" style="background:#0F172A; border:1px solid #23324A; font-family:monospace; font-size:0.85rem; color:#10B981;">
                    MARKET VALUATION: {formatted_val}
                </span>
                <div style="color:#64748B; font-size:0.75rem; margin-top:0.35rem; font-family:monospace;">
                    Foot: {prospect["preferred_foot"]} | {height_disp} / {weight_disp}
                </div>
            </div>
        </div>
        <div class="kpi-grid">
            <div class="kpi-pod">
                <div class="kpi-label">Current Overall</div>
                <div class="kpi-val emerald">{prospect["overall"]}</div>
                <div class="kpi-sub">Baseline Performance</div>
            </div>
            <div class="kpi-pod">
                <div class="kpi-label">Stated Potential</div>
                <div class="kpi-val cyan">{prospect["potential"]}</div>
                <div class="kpi-sub">{growth_disp} Trajectory Vector</div>
            </div>
            <div class="kpi-pod">
                <div class="kpi-label">Evaluation Age</div>
                <div class="kpi-val amber">{prospect["age"]}</div>
                <div class="kpi-sub">U21 Scouting Window</div>
            </div>
            <div class="kpi-pod">
                <div class="kpi-label">Tactical Sector</div>
                <div class="kpi-val" style="font-size:1.15rem; padding-top:0.35rem; color:#F8FAFC;">{sector}</div>
                <div class="kpi-sub">Role: {prospect["positions"]}</div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    is_gk = primary_pos == "GK"
    active_cols = (
        [c for c in SKILL_COLS if c.startswith("goalkeeping_")] if is_gk else OUTFIELD_COLS
    )
    ceiling_result = cached_match_ceiling(
        int(prospect["player_id"]), int(prospect["fifa_edition_year"]), top_k=3
    )
    top_3 = ceiling_result.get("matches", []) if ceiling_result else []
    macro_radar_dims = ceiling_result.get("macro_radar", []) if ceiling_result else []

    st.markdown(
        """
    <div class="section-header-hud">
        <span class="section-prefix">// 01</span>
        <span>PROJECTED SUPERSTAR CEILING ARCHETYPES</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if not top_3:
        st.info("No matching superstar archetypes found for this tactical sector.")
        return prospect

    card_cols = st.columns(len(top_3))
    tier_labels = ["#1 PRIMARY TARGET", "#2 SECONDARY MATCH", "#3 CONTINGENT MATCH"]
    tier_classes = ["", "rank-2", "rank-3"]
    badge_colors = ["", "cyan", ""]

    for idx, cand in enumerate(top_3):
        with card_cols[idx]:
            aligned_pills = "".join(
                [
                    f"<span class='trait-pill-aligned'>{t['attribute'].replace('skill_', '').replace('movement_', '').replace('mentality_', '')}</span>"
                    for t in cand["strengths"]
                ]
            )
            growth_pills = "".join(
                [
                    f"<span class='trait-pill-growth'>{g['attribute'].replace('attacking_', '').replace('mentality_', '')} (+{g['gap']})</span>"
                    for g in cand["growth_needed"][:2]
                ]
            )

            st.markdown(
                f"""
            <div class="archetype-card-dark {tier_classes[idx]}">
                <div>
                    <div class="pod-topbar">
                        <span class="pod-tier-chip">{tier_labels[idx]}</span>
                        <span class="pod-pct-badge {badge_colors[idx]}">{cand["similarity_pct"]}%</span>
                    </div>
                    <div class="confidence-meter-track">
                        <div class="confidence-meter-fill" style="width: {cand["similarity_pct"]}%;"></div>
                    </div>
                    <h3 class="star-name">{cand["name"]}</h3>
                    <div class="star-meta">
                        <b>Peak: {cand["peak_overall"]} OVR</b> (Ages {cand["peak_age_window"]})<br>
                        <span>Sector: {cand["sector"]}</span>
                    </div>
                    <div style="font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.25rem;">
                        Core Matching Traits:
                    </div>
                    <div class="trait-badge-group">
                        {aligned_pills}
                    </div>
                </div>
                <div>
                    <div style="font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.25rem;">
                        Priority Growth Focus:
                    </div>
                    <div class="trait-badge-group">
                        {growth_pills}
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    best_star = top_3[0]
    st.markdown(
        f"""
    <div class="section-header-hud" style="margin-top:2rem;">
        <span class="section-prefix">// 02</span>
        <span>TACTICAL FINGERPRINT: {prospect["short_name"].upper()} VS PRIME {best_star["name"].upper()}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if not is_gk:
        macro_dimensions = {
            "Pace": ["movement_acceleration", "movement_sprint_speed"],
            "Shooting": ["attacking_finishing", "power_shot_power", "power_long_shots"],
            "Passing": [
                "attacking_short_passing",
                "skill_long_passing",
                "mentality_vision",
            ],
            "Dribbling": ["skill_dribbling", "skill_ball_control", "movement_agility"],
            "Defending": [
                "defending_standing_tackle",
                "defending_marking_awareness",
                "mentality_interceptions",
            ],
            "Physical": ["power_stamina", "power_strength", "power_jumping"],
        }
    else:
        macro_dimensions = {
            "Diving": ["goalkeeping_diving"],
            "Handling": ["goalkeeping_handling"],
            "Kicking": ["goalkeeping_kicking"],
            "Reflexes": ["goalkeeping_reflexes"],
            "Positioning": ["goalkeeping_positioning"],
            "Reactions": ["movement_reactions"],
        }

    if macro_radar_dims:
        radar_cats = [r["dimension"] for r in macro_radar_dims]
        p_vals = [float(r["prospect_value"]) for r in macro_radar_dims]
        s_vals = [float(r["benchmark_value"]) for r in macro_radar_dims]
        comparison_rows = [
            {
                "dimension": r["dimension"],
                "p_val": round(r["prospect_value"], 1),
                "s_val": round(r["benchmark_value"], 1),
                "delta": round(r["gap"], 1),
            }
            for r in macro_radar_dims
        ]
    else:
        radar_cats = list(macro_dimensions.keys())
        p_vals = []
        s_vals = []
        comparison_rows = []

        for dim, subcols in macro_dimensions.items():
            valid_p = [float(prospect[c]) for c in subcols if pd.notna(prospect[c])]
            p_mean = float(np.mean(valid_p)) if valid_p else 50.0
            valid_s = [
                float(best_star["attributes"].get(c, 50.0))
                for c in subcols
                if pd.notna(best_star["attributes"].get(c))
            ]
            s_mean = float(np.mean(valid_s)) if valid_s else 50.0
            p_vals.append(p_mean)
            s_vals.append(s_mean)
            delta = round(s_mean - p_mean, 1)
            comparison_rows.append(
                {
                    "dimension": dim,
                    "p_val": round(p_mean, 1),
                    "s_val": round(s_mean, 1),
                    "delta": delta,
                }
            )

    col_radar, col_telemetry = st.columns([3, 2])

    with col_radar:
        radar_chart = build_tactical_radar_altair(
            radar_cats,
            p_vals,
            s_vals,
            p_name=str(prospect["short_name"]),
            s_name=str(best_star["name"]),
        )
        st.altair_chart(radar_chart, use_container_width=True)

    with col_telemetry:
        rows_html = []
        for cr in comparison_rows:
            delta_sign = f"+{cr['delta']}" if cr["delta"] > 0 else f"{cr['delta']}"
            delta_class = "neg" if cr["delta"] > 0 else "pos"
            rows_html.append(f"""
            <div class="gap-metric-row">
                <span style="font-weight:700; color:#F8FAFC;">{cr["dimension"]}</span>
                <div>
                    <span class="gap-val-p">{cr["p_val"]}</span>
                    <span style="color:#64748B;"> vs </span>
                    <span class="gap-val-s">{cr["s_val"]}</span>
                    <span class="gap-delta {delta_class}" style="margin-left:0.75rem;">({delta_sign})</span>
                </div>
            </div>
            """)

        telemetry_card_html = f"""
        <div style="background:#111827; border:1px solid #1E293B; border-radius:8px; padding:1.25rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem; border-bottom:1px solid #1E293B; padding-bottom:0.5rem;">
                <span style="font-size:0.8rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.05em;">Macro Dimension</span>
                <span style="font-size:0.75rem; font-family:monospace; color:#94A3B8;">
                    <span style="color:#10B981;">● PROSPECT</span> | <span style="color:#06B6D4;">● ARCHETYPE</span>
                </span>
            </div>
            {"".join(rows_html)}
        </div>
        """
        st.markdown(telemetry_card_html, unsafe_allow_html=True)

    with st.expander("🔬 Deep Vector Audit: Granular Trait-by-Trait Comparison (All Skills)"):
        detailed_data = []
        for col in active_cols:
            p_val = prospect[col]
            s_val = best_star["attributes"].get(col)
            if pd.notna(p_val) and s_val is not None:
                detailed_data.append(
                    {
                        "Skill Trait": col.replace("_", " ").title(),
                        f"Prospect ({prospect['short_name']})": round(float(p_val), 1),
                        f"Archetype ({best_star['name']})": round(float(s_val), 1),
                        "Net Gap": round(float(s_val) - float(p_val), 1),
                    }
                )
        st.dataframe(pd.DataFrame(detailed_data), use_container_width=True, hide_index=True)

    return prospect


def _render_trajectory_forecaster_tab(selected_prospect: dict[str, Any] | None) -> None:
    """Render the Model 2 autoregressive trajectory forecaster tab."""
    st.markdown(
        """
    <div class="section-header-hud" style="margin-top:0.5rem; margin-bottom:0.75rem;">
        <span class="section-prefix">// MODEL 2</span>
        <span>QUANTILE &amp; CONFORMAL TRAJECTORY FORECASTER</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if selected_prospect is None:
        st.info(
            "Select a player in 'The Superstar Ceiling Matcher' tab to generate a multi-year development trajectory forecast."
        )
        return

    player_id = int(selected_prospect["player_id"])
    edition_year = int(selected_prospect["fifa_edition_year"])
    short_name = str(selected_prospect["short_name"])
    club_name = selected_prospect.get("club") or "Free Agent"

    st.caption(
        f"Active Target: **{short_name}** ({club_name} | FIFA {edition_year}). "
        "Change the active player in Tab 1 Global Player Discovery."
    )

    col_horizon, col_conf = st.columns([2, 2])
    with col_horizon:
        horizon_years = int(
            st.slider("Forecast Horizon (Seasons Ahead):", min_value=1, max_value=3, value=3)
        )
    with col_conf:
        include_conformal = bool(
            st.checkbox(
                "Overlay 90% Conformal Prediction Band",
                value=True,
                help="Adds finite-sample calibrated interval bounds from ConformalTrajectoryPredictor.",
            )
        )

    models = get_cached_trajectory_models()
    quantile_model, conformal_model = models if models is not None else (None, None)

    forecast_resp = forecast_trajectory(
        player_id=player_id,
        edition_year=edition_year,
        horizon_years=horizon_years,
        include_conformal=include_conformal,
        quantile_model=quantile_model,
        conformal_model=conformal_model,
    )

    if not forecast_resp or not forecast_resp.get("forecast"):
        if models is None:
            st.info(
                "Model 2 trajectory artifacts (`models/trajectory_regressor.joblib`) are not available "
                "and no live API server responded."
            )
        else:
            st.warning(
                f"Trajectory features are unavailable for {short_name} (FIFA {edition_year}) "
                "in the current database."
            )
        return

    final_pt = forecast_resp["forecast"][-1]
    curr_ovr = float(forecast_resp["current_overall"])
    proj_q50 = float(final_pt["q50_overall"])
    delta_q50 = proj_q50 - curr_ovr
    delta_sign = f"+{delta_q50:.1f}" if delta_q50 >= 0 else f"{delta_q50:.1f}"
    hist_growth = float(forecast_resp.get("historical_growth_1y", 0.0))
    hist_sign = f"+{hist_growth:.1f}" if hist_growth >= 0 else f"{hist_growth:.1f}"

    conf_low = final_pt.get("conformal_lower")
    conf_high = final_pt.get("conformal_upper")
    if include_conformal and conf_low is not None and conf_high is not None:
        band_val = f"{float(conf_low):.1f} to {float(conf_high):.1f}"
        band_sub = f"Year +{horizon_years} 90% Conformal Interval"
    else:
        band_val = f"{float(final_pt['q10_overall']):.1f} to {float(final_pt['q90_overall']):.1f}"
        band_sub = f"Year +{horizon_years} Q10 to Q90 Quantile Span"

    latency_ms = float(forecast_resp.get("inference_time_ms", 0.0))
    q_ver = str(forecast_resp.get("quantile_model_version", "v1"))

    st.markdown(
        f"""
    <div class="kpi-grid" style="margin-bottom:1.25rem;">
        <div class="kpi-pod">
            <div class="kpi-label">Baseline Overall</div>
            <div class="kpi-val emerald">{curr_ovr:.0f}</div>
            <div class="kpi-sub">FIFA {edition_year} | Prior 1Y: {hist_sign}</div>
        </div>
        <div class="kpi-pod">
            <div class="kpi-label">Year +{horizon_years} Median (Q50)</div>
            <div class="kpi-val cyan">{proj_q50:.1f}</div>
            <div class="kpi-sub">{delta_sign} vs baseline | Age {int(final_pt["projected_age"])}</div>
        </div>
        <div class="kpi-pod">
            <div class="kpi-label">Uncertainty Envelope</div>
            <div class="kpi-val amber" style="font-size:1.35rem; padding-top:0.25rem;">{band_val}</div>
            <div class="kpi-sub">{band_sub}</div>
        </div>
        <div class="kpi-pod">
            <div class="kpi-label">Inference Latency</div>
            <div class="kpi-val" style="color:#F8FAFC;">{latency_ms:.1f} ms</div>
            <div class="kpi-sub">Model {q_ver}</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.altair_chart(build_trajectory_fan_chart(forecast_resp), use_container_width=True)

    table_rows: list[dict[str, Any]] = []
    for pt in forecast_resp["forecast"]:
        row: dict[str, Any] = {
            "Target Season": f"FIFA {int(pt['target_edition_year'])}",
            "Projected Age": int(pt["projected_age"]),
            "Pred. Growth (1Y)": round(float(pt["predicted_growth"]), 2),
            "Floor (Q10)": round(float(pt["q10_overall"]), 1),
            "Median (Q50)": round(float(pt["q50_overall"]), 1),
            "Ceiling (Q90)": round(float(pt["q90_overall"]), 1),
        }
        if pt.get("conformal_lower") is not None and pt.get("conformal_upper") is not None:
            row["90% Conf. Low"] = round(float(pt["conformal_lower"]), 1)
            row["90% Conf. High"] = round(float(pt["conformal_upper"]), 1)
        table_rows.append(row)

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    c_ver = forecast_resp.get("conformal_model_version")
    if not include_conformal:
        conformal_note = "Conformal overlay disabled; displaying Q10 to Q90 quantile spread."
    elif c_ver:
        conformal_note = f"Conformal calibration: {c_ver}."
    else:
        conformal_note = (
            "Conformal calibration artifact unavailable; displaying Q10 to Q90 quantile spread."
        )
    st.caption(f"Quantile model: {q_ver}. {conformal_note}")


# ==============================================================================
# MAIN STREAMLIT ENTRYPOINT
# ==============================================================================
def main() -> None:
    """Render the Analytics FC Streamlit intelligence terminal."""
    _configure_page()
    bootstrap_status = ensure_serving_assets()
    if not bootstrap_status.get("ready", False):
        st.error(
            "Serving artifacts are missing and could not be downloaded automatically. "
            f"Missing: {bootstrap_status.get('missing_core', [])}. "
            f"Details: {bootstrap_status.get('error') or 'Set AFC_BUNDLE_URL or run local build.'}"
        )
        st.stop()

    backend_status = get_connection_status()
    status_chip_class = "telemetry-chip live"
    try:
        catalog_stats = cached_catalog_stats()
    except Exception:
        catalog_stats = {"total_rows": 0, "distinct_players": 0, "min_year": 2015, "max_year": 2022}

    total_rows = int(catalog_stats.get("total_rows", 0))
    distinct_players = int(catalog_stats.get("distinct_players", 0))

    st.markdown(
        f"""
    <div class="terminal-bar">
        <div class="brand-title">
            <span>⚽ ANALYTICS FC</span>
            <span class="brand-accent">// SCOUTING INTELLIGENCE ENGINE</span>
        </div>
        <div class="telemetry-group">
            <span class="{status_chip_class}">● {backend_status}</span>
            <span class="telemetry-chip">{total_rows:,} SEASONS / {distinct_players:,} PLAYERS</span>
            <span class="telemetry-chip live">CATALOG SEARCH: {total_rows:,} RECORDS ACTIVE</span>
            <span class="telemetry-chip live">KERNEL: GAUSSIAN RBF (&lt;5ms)</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Navigation Tabs
    tab_matcher, tab_trajectory, tab_audit = st.tabs(
        [
            "🌟 The Superstar Ceiling Matcher",
            "📈 Model 2: Trajectory Forecaster",
            "🏛️ Methodology & Historical Model Audit",
        ]
    )

    gallery = load_gallery()
    if not gallery:
        st.warning(
            "Superstar gallery artifact (superstar_gallery.json) not found or empty. Please run scripts/build_superstar_gallery.py to compile the archetype database."
        )
        st.stop()

    # ==============================================================================
    # TAB 1: THE SUPERSTAR CEILING MATCHER
    # ==============================================================================
    selected_prospect: dict[str, Any] | None = None
    with tab_matcher:
        st.markdown(
            """
        <div class="section-header-hud" style="margin-top:0.5rem; margin-bottom:0.75rem;">
            <span class="section-prefix">// 00</span>
            <span>GLOBAL PLAYER DISCOVERY & SEARCH ENGINE</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        col_search, col_edition, col_min_ovr, col_sector = st.columns([2.5, 1, 1, 1.5])

        with col_search:
            search_query = st.text_input(
                "Search Player Name (min 3 chars):",
                value="",
                placeholder=f"Search all {total_rows:,} records (e.g. Haaland, Mbappe, Saka)...",
                help="Debounced search scans across the complete historical DuckDB catalog.",
            )

        with col_edition:
            edition_options = ["All Editions"] + [str(y) for y in range(2022, 2014, -1)]
            selected_edition_str = st.selectbox("FIFA Edition:", edition_options)
            selected_edition = (
                int(selected_edition_str) if selected_edition_str != "All Editions" else None
            )

        with col_min_ovr:
            min_overall = st.slider("Min Overall:", min_value=50, max_value=90, value=75, step=1)

        with col_sector:
            sector_options = ["All Tactical Sectors"] + SECTOR_ORDER
            selected_sector_str = st.selectbox("Tactical Sector:", sector_options)
            selected_sector = (
                selected_sector_str if selected_sector_str != "All Tactical Sectors" else None
            )

        # Execute search query
        query_str = search_query.strip()
        is_active_search = len(query_str) >= 3

        if is_active_search:
            search_result = cached_search_players(
                query=query_str,
                edition_year=selected_edition,
                min_overall=min_overall,
                sector=selected_sector,
                limit=50,
            )
            candidates = search_result["items"]
            total_found = search_result["total_matching"]
        else:
            # Default discovery fallback: Top elite prospects matching filters
            search_result = cached_search_players(
                query="",
                edition_year=selected_edition,
                min_overall=max(78, min_overall),
                sector=selected_sector,
                limit=50,
            )
            candidates = search_result["items"]
            total_found = search_result["total_matching"]

        if not candidates:
            st.warning(
                "No players matched the selected criteria. Try adjusting the search term or lowering the overall rating threshold."
            )
        else:
            # Player Selection Dropdown
            player_keys = [f"{c['player_id']}_{c['fifa_edition_year']}" for c in candidates]
            player_map = {f"{c['player_id']}_{c['fifa_edition_year']}": c for c in candidates}

            status_note = (
                f"{total_found} MATCHES FOUND IN CATALOG"
                if is_active_search
                else f"SHOWING TOP PROSPECTS ({total_found} IN POOL - TYPE 3+ CHARS TO SEARCH COMPLETE CATALOG)"
            )
            selected_key = st.selectbox(
                f"Select Player for Projection ({status_note}):",
                player_keys,
                format_func=lambda k: (
                    f"{player_map[k]['short_name']} ({player_map[k]['club'] or 'Free Agent'} | "
                    f"FIFA {player_map[k]['fifa_edition_year']} | "
                    f"{player_map[k]['overall']} OVR / {player_map[k]['potential']} POT | "
                    f"{player_map[k]['tactical_sector']})"
                ),
            )
            selected_prospect = _render_prospect_dossier_and_matcher(player_map[selected_key])

    # ==============================================================================
    # TAB 2: MODEL 2 TRAJECTORY FORECASTER
    # ==============================================================================
    with tab_trajectory:
        _render_trajectory_forecaster_tab(selected_prospect)

    # ==============================================================================
    # TAB 3: METHODOLOGY & HISTORICAL MODEL AUDIT
    # ==============================================================================
    with tab_audit:
        st.markdown(
            """
        <div class="section-header-hud">
            <span class="section-prefix">// 03</span>
            <span>PHASE 3 EXPERIMENTAL AUDIT AND REPRODUCIBILITY RECORD</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="audit-terminal-box">
            <b>TECHNICAL STATUS BRIEFING: PHASE 3 EXPERIMENT ARCHIVE</b><br><br>
            <b>1. Silverware Classification Experiment:</b><br>
            The initial Phase 3 proof of concept evaluated three Random Forest classifiers trained on pairwise dyad records (2,786,960 pair rows, 4,313 distinct players) to predict whether a player would capture major silverware.<br><br>
            <b>2. Critical Audit Findings (Recorded 2026-10-10):</b><br>
            • Severe Label Sparsity: The trophy label join between Transfermarkt and FIFA occurred on short_name only. The reference table held 10 distinct trophy records, resulting in only 4 out of 4,313 distinct players being labelled positive (0.022% positive label frequency).<br>
            • Pair-Level Data Leakage: Random train/test splits across row dyads placed the exact same player in both train and test partitions, inflating baseline ROC-AUC to 99.8%.<br><br>
            <b>3. Architectural Resolution:</b><br>
            • As documented in our technical charter, binary silverware prediction has been permanently separated from the operational scouting engine.<br>
            • The production platform now relies exclusively on The Superstar Ceiling Matcher: an unsupervised continuous vector projection using L2 normalization and a Gaussian RBF kernel (bandwidth 0.28).<br>
            • All Phase 3 model runs remain archived in local MLflow (mlruns) for scientific verification and historical reference.
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="section-header-hud" style="margin-top:2rem;">
            <span class="section-prefix">// 04</span>
            <span>PRODUCTION DATA DRIFT & ATTRIBUTE STABILITY MONITORING</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        drift_data = load_drift_metrics()

        if not drift_data:
            st.info(
                "Drift telemetry snapshot (data/monitoring/drift_metrics.json) not found. "
                "Execute the monitoring pipeline to generate live drift indicators."
            )
        else:
            ref_year = drift_data.get("reference_edition", 2021)
            target_year = drift_data.get("target_edition", 2022)
            ref_sample = drift_data.get("reference_sample_size", 0)
            target_sample = drift_data.get("target_sample_size", 0)
            total_feats = drift_data.get("total_features_evaluated", 0)
            drifted_count = drift_data.get("drifted_features_count", 0)
            drifted_share = drift_data.get("drifted_features_share", 0.0)
            is_drifted = drift_data.get("dataset_drift_detected", False)

            status_text = "DRIFT DETECTED" if is_drifted else "HEALTHY (NO DATASET DRIFT)"
            status_color = "#EF4444" if is_drifted else "#10B981"
            status_bg = "rgba(239, 68, 68, 0.12)" if is_drifted else "rgba(16, 185, 129, 0.12)"
            status_border = "#EF4444" if is_drifted else "#10B981"

            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

            with kpi_col1:
                st.markdown(
                    f"""
                    <div style="background:{status_bg}; border:1px solid {status_border}; border-radius:6px; padding:0.9rem; text-align:center;">
                        <div style="font-size:0.75rem; color:#94A3B8; font-family:'JetBrains Mono', monospace; letter-spacing:0.05em;">DATASET DRIFT STATUS</div>
                        <div style="font-size:1.05rem; font-weight:800; color:{status_color}; margin-top:0.3rem; font-family:'JetBrains Mono', monospace;">● {status_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with kpi_col2:
                st.markdown(
                    f"""
                    <div style="background:#0F172A; border:1px solid #1E293B; border-radius:6px; padding:0.9rem; text-align:center;">
                        <div style="font-size:0.75rem; color:#94A3B8; font-family:'JetBrains Mono', monospace; letter-spacing:0.05em;">EDITIONS COMPARED</div>
                        <div style="font-size:1.1rem; font-weight:700; color:#F8FAFC; margin-top:0.3rem; font-family:'JetBrains Mono', monospace;">FIFA {ref_year} → FIFA {target_year}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with kpi_col3:
                st.markdown(
                    f"""
                    <div style="background:#0F172A; border:1px solid #1E293B; border-radius:6px; padding:0.9rem; text-align:center;">
                        <div style="font-size:0.75rem; color:#94A3B8; font-family:'JetBrains Mono', monospace; letter-spacing:0.05em;">DRIFTED ATTRIBUTES</div>
                        <div style="font-size:1.1rem; font-weight:700; color:{status_color}; margin-top:0.3rem; font-family:'JetBrains Mono', monospace;">{drifted_count} / {total_feats} ({drifted_share * 100:.1f}%)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with kpi_col4:
                st.markdown(
                    f"""
                    <div style="background:#0F172A; border:1px solid #1E293B; border-radius:6px; padding:0.9rem; text-align:center;">
                        <div style="font-size:0.75rem; color:#94A3B8; font-family:'JetBrains Mono', monospace; letter-spacing:0.05em;">SAMPLE STRATIFICATION</div>
                        <div style="font-size:1.1rem; font-weight:700; color:#F8FAFC; margin-top:0.3rem; font-family:'JetBrains Mono', monospace;">{ref_sample:,} vs {target_sample:,}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                """
                <div style="background:#090D16; border:1px solid #1E293B; border-left:3px solid #3B82F6; border-radius:4px; padding:0.6rem 0.9rem; margin-top:1rem; margin-bottom:1rem; font-family:'JetBrains Mono', monospace; font-size:0.78rem; color:#94A3B8;">
                    <span style="color:#60A5FA; font-weight:600;">STATISTICAL TEST SPECIFICATIONS:</span> Two-sample Kolmogorov-Smirnov test (significance threshold: p &lt; 0.05) &amp; Population Stability Index (drift threshold: PSI &gt; 0.10). Dataset-level drift triggers when &ge; 25.0% of monitored features drift.
                </div>
                """,
                unsafe_allow_html=True,
            )

            features_dict = drift_data.get("features", {})
            if features_dict:
                filter_col1, _ = st.columns([1.5, 3.5])
                with filter_col1:
                    show_drifted_only = st.checkbox("Show Drifted Attributes Only", value=False)

                table_rows = []
                for feat_name, m in features_dict.items():
                    is_feat_drifted = m.get("drift_detected", False)
                    if show_drifted_only and not is_feat_drifted:
                        continue
                    ref_mean = m.get("reference_mean", 0.0)
                    curr_mean = m.get("current_mean", 0.0)
                    net_shift = round(curr_mean - ref_mean, 2)
                    table_rows.append(
                        {
                            "Attribute": feat_name,
                            "Status": "DRIFTED" if is_feat_drifted else "STABLE",
                            "KS p-value": f"{m.get('ks_p_value', 1.0):.4f}",
                            "KS Statistic": f"{m.get('ks_statistic', 0.0):.4f}",
                            "Wasserstein Distance": f"{m.get('wasserstein_distance', 0.0):.4f}",
                            "PSI": f"{m.get('psi', 0.0):.4f}",
                            f"Ref (FIFA {ref_year}) Mean": f"{ref_mean:.2f}",
                            f"Target (FIFA {target_year}) Mean": f"{curr_mean:.2f}",
                            "Mean Shift": f"{net_shift:+.2f}",
                        }
                    )

                if table_rows:
                    df_drift_display = pd.DataFrame(table_rows)
                    st.dataframe(df_drift_display, use_container_width=True, hide_index=True)
                else:
                    st.info("No attributes matching current filter.")


if __name__ == "__main__":
    main()
