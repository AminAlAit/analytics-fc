import json
from pathlib import Path
import altair as alt
import duckdb
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# CONFIGURATION AND PAGE SETUP
# ==============================================================================
st.set_page_config(
    page_title="Analytics FC | Tactical Intelligence Terminal",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Injected Custom CSS: Tactical Command Dark Design System
st.markdown("""
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
""", unsafe_allow_html=True)

# ==============================================================================
# PATHS AND TAXONOMY CONSTANTS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DUCKDB_PATH = BASE_DIR / "dbt_project" / "analytics_fc.duckdb"
GALLERY_PATH = BASE_DIR / "data" / "processed" / "superstar_gallery.json"

SECTOR_MAP = {
    "GK": "Goalkeepers",
    "CB": "Defenders", "LB": "Defenders", "RB": "Defenders",
    "LWB": "Defenders", "RWB": "Defenders",
    "CDM": "Defensive & Central Midfielders", "CM": "Defensive & Central Midfielders",
    "CAM": "Attacking Midfielders & Wingers", "LM": "Attacking Midfielders & Wingers",
    "RM": "Attacking Midfielders & Wingers", "LW": "Attacking Midfielders & Wingers",
    "RW": "Attacking Midfielders & Wingers",
    "CF": "Center Forwards & Strikers", "ST": "Center Forwards & Strikers",
}

SKILL_COLS = [
    "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
    "attacking_short_passing", "attacking_volleys",
    "skill_dribbling", "skill_curve", "skill_fk_accuracy",
    "skill_long_passing", "skill_ball_control",
    "movement_acceleration", "movement_sprint_speed", "movement_agility",
    "movement_reactions", "movement_balance",
    "power_shot_power", "power_jumping", "power_stamina",
    "power_strength", "power_long_shots",
    "mentality_aggression", "mentality_interceptions", "mentality_positioning",
    "mentality_vision", "mentality_penalties", "mentality_composure",
    "defending_marking_awareness", "defending_standing_tackle", "defending_sliding_tackle",
    "goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking",
    "goalkeeping_positioning", "goalkeeping_reflexes",
]
OUTFIELD_COLS = [col for col in SKILL_COLS if not col.startswith("goalkeeping_")]

# ==============================================================================
# HARDWARE SAFE RESOURCE LOADING
# ==============================================================================
@st.cache_data
def load_gallery():
    if not GALLERY_PATH.exists():
        return []
    with open(GALLERY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def get_popular_wonderkids():
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        con.execute("SET threads = 1;")
        con.execute("SET memory_limit = '1GB';")
        query = """
        SELECT 
            player_id, short_name, long_name, age, fifa_edition_year, overall, potential, positions,
            nationality, club, league_name, preferred_foot, height_cm, weight_kg, value_eur, wage_eur,
            pace, shooting, passing, dribbling, defending, physic,
            """ + ", ".join(SKILL_COLS) + """
        FROM stg_fifa_granular
        WHERE age <= 21 AND overall >= 78
        ORDER BY overall DESC, potential DESC
        LIMIT 150
        """
        df = con.execute(query).df()
    finally:
        con.close()

    # Precompute primary position and tactical sector
    df["primary_pos"] = df["positions"].apply(
        lambda p: str(p).split(",")[0].strip().upper() if pd.notna(p) else "ST"
    )
    df["tactical_sector"] = df["primary_pos"].apply(
        lambda p: SECTOR_MAP.get(p, "Attacking Midfielders & Wingers")
    )
    return df

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
def build_tactical_radar_altair(categories, p_vals, s_vals, p_name="Prospect", s_name="Prime Benchmark"):
    n = len(categories)
    angles = [i * 2 * np.pi / n - np.pi / 2 for i in range(n)]

    # 1. Concentric Hexagonal Grid Lines (20, 40, 60, 80, 100)
    grid_rows = []
    for level in [20, 40, 60, 80, 100]:
        for ang in angles:
            grid_rows.append({"x": level * np.cos(ang), "y": level * np.sin(ang), "level": str(level)})
        grid_rows.append({"x": level * np.cos(angles[0]), "y": level * np.sin(angles[0]), "level": str(level)})
    df_grid = pd.DataFrame(grid_rows)

    grid_chart = alt.Chart(df_grid).mark_line(
        color="#223147", strokeWidth=1, strokeDash=[3, 3]
    ).encode(
        x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-150, 150])),
        y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-150, 150])),
        detail="level:N"
    )

    # 2. Axis Spokes
    spoke_rows = []
    for i, ang in enumerate(angles):
        spoke_rows.append({"x": 0, "y": 0, "spoke": str(i)})
        spoke_rows.append({"x": 100 * np.cos(ang), "y": 100 * np.sin(ang), "spoke": str(i)})
    df_spokes = pd.DataFrame(spoke_rows)

    spoke_chart = alt.Chart(df_spokes).mark_line(
        color="#1E293B", strokeWidth=1
    ).encode(
        x="x:Q", y="y:Q", detail="spoke:N"
    )

    # 3. Prospect Polygon (Emerald #10B981)
    p_rows = [
        {"x": v * np.cos(ang), "y": v * np.sin(ang), "entity": p_name, "order": i, "val": v, "dimension": c}
        for i, (c, v, ang) in enumerate(zip(categories, p_vals, angles))
    ]
    p_rows.append({
        "x": p_vals[0] * np.cos(angles[0]), "y": p_vals[0] * np.sin(angles[0]),
        "entity": p_name, "order": n, "val": p_vals[0], "dimension": categories[0]
    })
    df_p = pd.DataFrame(p_rows)

    p_line = alt.Chart(df_p).mark_line(
        color="#10B981", strokeWidth=3
    ).encode(
        x="x:Q", y="y:Q", order="order:Q"
    )
    p_points = alt.Chart(df_p).mark_circle(
        color="#10B981", size=60
    ).encode(
        x="x:Q", y="y:Q", tooltip=["dimension:N", "val:Q"]
    )

    # 4. Prime Superstar Archetype Polygon (Cyan #06B6D4)
    s_rows = [
        {"x": v * np.cos(ang), "y": v * np.sin(ang), "entity": s_name, "order": i, "val": v, "dimension": c}
        for i, (c, v, ang) in enumerate(zip(categories, s_vals, angles))
    ]
    s_rows.append({
        "x": s_vals[0] * np.cos(angles[0]), "y": s_vals[0] * np.sin(angles[0]),
        "entity": s_name, "order": n, "val": s_vals[0], "dimension": categories[0]
    })
    df_s = pd.DataFrame(s_rows)

    s_line = alt.Chart(df_s).mark_line(
        color="#06B6D4", strokeWidth=2.5, strokeDash=[4, 4]
    ).encode(
        x="x:Q", y="y:Q", order="order:Q"
    )
    s_points = alt.Chart(df_s).mark_circle(
        color="#06B6D4", size=50
    ).encode(
        x="x:Q", y="y:Q", tooltip=["dimension:N", "val:Q"]
    )

    # 5. Dimension Perimeter Labels
    label_rows = []
    for cat, p, s, ang in zip(categories, p_vals, s_vals, angles):
        p_clean = 50.0 if (pd.isna(p) or np.isnan(p)) else p
        s_clean = 50.0 if (pd.isna(s) or np.isnan(s)) else s
        label_rows.append({
            "x": 122 * np.cos(ang),
            "y": 122 * np.sin(ang),
            "label": f"{cat}: {int(round(p_clean))} | {int(round(s_clean))}"
        })
    df_labels = pd.DataFrame(label_rows)

    labels_chart = alt.Chart(df_labels).mark_text(
        color="#94A3B8", fontSize=11, fontWeight=700
    ).encode(
        x="x:Q", y="y:Q", text="label:N"
    )

    combined_chart = (grid_chart + spoke_chart + p_line + p_points + s_line + s_points + labels_chart).properties(
        height=420
    ).configure_view(
        strokeWidth=0
    ).configure(
        background="#0A0E17"
    )
    return combined_chart

# ==============================================================================
# UI HEADER: EXECUTIVE COMMAND STATUS BAR
# ==============================================================================
st.markdown("""
<div class="terminal-bar">
    <div class="brand-title">
        <span>⚽ ANALYTICS FC</span>
        <span class="brand-accent">// SCOUTING INTELLIGENCE ENGINE</span>
    </div>
    <div class="telemetry-group">
        <span class="telemetry-chip live">● DUCKDB READ ONLY</span>
        <span class="telemetry-chip">180,021 SEASONS / 45,629 PLAYERS</span>
        <span class="telemetry-chip">CANDIDATE POOL: 150 ELITE PROSPECTS</span>
        <span class="telemetry-chip live">KERNEL: GAUSSIAN RBF (&lt;5ms)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation Tabs
tab_matcher, tab_audit = st.tabs([
    "🌟 The Superstar Ceiling Matcher",
    "🏛️ Methodology & Historical Model Audit"
])

gallery = load_gallery()
if not gallery:
    st.warning("Superstar gallery artifact (superstar_gallery.json) not found or empty. Please run scripts/build_superstar_gallery.py to compile the archetype database.")
    st.stop()

# ==============================================================================
# TAB 1: THE SUPERSTAR CEILING MATCHER
# ==============================================================================
with tab_matcher:
    wonderkids_df = get_popular_wonderkids()

    col_filter_sector, col_filter_player = st.columns([1, 2])

    with col_filter_sector:
        sector_options = [
            "All Tactical Sectors",
            "Attacking Midfielders & Wingers",
            "Center Forwards & Strikers",
            "Defensive & Central Midfielders",
            "Defenders",
            "Goalkeepers",
        ]
        selected_sector = st.selectbox("Filter by Tactical Sector:", sector_options)

    if selected_sector != "All Tactical Sectors":
        active_prospects_df = wonderkids_df[wonderkids_df["tactical_sector"] == selected_sector]
        if active_prospects_df.empty:
            active_prospects_df = wonderkids_df
    else:
        active_prospects_df = wonderkids_df

    with col_filter_player:
        player_map = {
            f"{row['player_id']}_{row['fifa_edition_year']}": row
            for _, row in active_prospects_df.iterrows()
        }
        player_keys = list(player_map.keys())
        selected_key = st.selectbox(
            "Select Under-21 Prospect for Projection:",
            player_keys,
            format_func=lambda k: (
                f"{player_map[k]['short_name']} (Age {player_map[k]['age']} in FIFA {player_map[k]['fifa_edition_year']} : "
                f"{player_map[k]['overall']} OVR / {player_map[k]['potential']} POT)"
            )
        )
        prospect = player_map[selected_key]

    primary_pos = str(prospect["positions"]).split(",")[0].strip().upper()
    sector = SECTOR_MAP.get(primary_pos, "Attacking Midfielders & Wingers")
    growth_delta = int(prospect["potential"]) - int(prospect["overall"])

    # Executive Player Dossier Card
    formatted_val = format_currency_eur(prospect["value_eur"])
    height_disp = f"{int(prospect['height_cm'])} cm" if pd.notna(prospect.get("height_cm")) else "N/A"
    weight_disp = f"{int(prospect['weight_kg'])} kg" if pd.notna(prospect.get("weight_kg")) else "N/A"
    st.markdown(f"""
    <div class="dossier-card">
        <div class="dossier-header">
            <div>
                <h2 class="player-name-display">{prospect['short_name']}</h2>
                <div class="player-sub-meta">
                    <span class="meta-tag">FIFA {prospect['fifa_edition_year']}</span>
                    <span><b>Club:</b> {prospect['club']}</span>
                    <span>•</span>
                    <span><b>League:</b> {prospect['league_name']}</span>
                    <span>•</span>
                    <span><b>Nationality:</b> {prospect['nationality']}</span>
                </div>
            </div>
            <div style="text-align: right;">
                <span class="meta-tag" style="background:#0F172A; border:1px solid #23324A; font-family:monospace; font-size:0.85rem; color:#10B981;">
                    MARKET VALUATION: {formatted_val}
                </span>
                <div style="color:#64748B; font-size:0.75rem; margin-top:0.35rem; font-family:monospace;">
                    Foot: {prospect['preferred_foot']} | {height_disp} / {weight_disp}
                </div>
            </div>
        </div>
        <div class="kpi-grid">
            <div class="kpi-pod">
                <div class="kpi-label">Current Overall</div>
                <div class="kpi-val emerald">{prospect['overall']}</div>
                <div class="kpi-sub">Baseline Performance</div>
            </div>
            <div class="kpi-pod">
                <div class="kpi-label">Stated Potential</div>
                <div class="kpi-val cyan">{prospect['potential']}</div>
                <div class="kpi-sub">+{growth_delta} Trajectory Vector</div>
            </div>
            <div class="kpi-pod">
                <div class="kpi-label">Evaluation Age</div>
                <div class="kpi-val amber">{prospect['age']}</div>
                <div class="kpi-sub">U21 Scouting Window</div>
            </div>
            <div class="kpi-pod">
                <div class="kpi-label">Tactical Sector</div>
                <div class="kpi-val" style="font-size:1.15rem; padding-top:0.35rem; color:#F8FAFC;">{sector}</div>
                <div class="kpi-sub">Role: {prospect['positions']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Vector Matching Engine (NumPy in-memory vector projection <5ms)
    is_gk = (primary_pos == "GK")
    active_cols = [c for c in SKILL_COLS if c.startswith("goalkeeping_")] if is_gk else OUTFIELD_COLS

    p_vec = np.array([float(prospect[c]) if pd.notna(prospect[c]) else 50.0 for c in active_cols])
    p_norm = np.linalg.norm(p_vec)
    p_unit = p_vec / p_norm if p_norm > 0 else p_vec

    candidates = []
    for star in gallery:
        if star["player_id"] == int(prospect["player_id"]):
            continue
        star_sector = star["tactical_sector"]
        sector_match = (star_sector == sector) or (
            sector in ["Attacking Midfielders & Wingers", "Center Forwards & Strikers"] and
            star_sector in ["Attacking Midfielders & Wingers", "Center Forwards & Strikers"]
        )
        if is_gk and star_sector != "Goalkeepers":
            continue
        if not is_gk and star_sector == "Goalkeepers":
            continue
        if not sector_match:
            continue

        s_vec = np.array([float(star["attributes"].get(c, 50.0)) for c in active_cols])
        s_norm = np.linalg.norm(s_vec)
        s_unit = s_vec / s_norm if s_norm > 0 else s_vec

        norm_dist = float(np.linalg.norm(p_unit - s_unit))
        calibrated_pct = round(float(np.exp(- (norm_dist ** 2) / (2 * (0.28 ** 2))) * 100), 1)

        trait_deltas = {}
        for col in active_cols:
            s_val = star["attributes"].get(col)
            p_val = prospect[col]
            if s_val is not None and pd.notna(p_val):
                trait_deltas[col] = round(float(s_val) - float(p_val), 1)

        top_growth_needed = sorted(
            [{"attribute": k, "gap": v} for k, v in trait_deltas.items() if v > 0],
            key=lambda x: x["gap"], reverse=True
        )[:3]

        strengths_aligned = sorted(
            [{"attribute": k, "difference": abs(v)} for k, v in trait_deltas.items()],
            key=lambda x: x["difference"]
        )[:3]

        peak_window = f"{star.get('peak_age_min', 26)} to {star.get('peak_age_max', 29)}"
        candidates.append({
            "name": star["short_name"],
            "sector": star["tactical_sector"],
            "peak_overall": star["peak_overall"],
            "peak_age_window": peak_window,
            "similarity_pct": calibrated_pct,
            "growth_needed": top_growth_needed,
            "strengths": strengths_aligned,
            "attributes": star["attributes"],
        })

    candidates.sort(key=lambda x: x["similarity_pct"], reverse=True)
    top_3 = candidates[:3]

    st.markdown("""
    <div class="section-header-hud">
        <span class="section-prefix">// 01</span>
        <span>PROJECTED SUPERSTAR CEILING ARCHETYPES</span>
    </div>
    """, unsafe_allow_html=True)

    if not top_3:
        st.info("No matching superstar archetypes found for this tactical sector.")
    else:
        card_cols = st.columns(len(top_3))
        tier_labels = ["#1 PRIMARY TARGET", "#2 SECONDARY MATCH", "#3 CONTINGENT MATCH"]
        tier_classes = ["", "rank-2", "rank-3"]
        badge_colors = ["", "cyan", ""]

        for idx, cand in enumerate(top_3):
            with card_cols[idx]:
            aligned_pills = "".join([
                f"<span class='trait-pill-aligned'>{t['attribute'].replace('skill_', '').replace('movement_', '').replace('mentality_', '')}</span>"
                for t in cand["strengths"]
            ])
            growth_pills = "".join([
                f"<span class='trait-pill-growth'>{g['attribute'].replace('attacking_', '').replace('mentality_', '')} (+{g['gap']})</span>"
                for g in cand["growth_needed"][:2]
            ])

            st.markdown(f"""
            <div class="archetype-card-dark {tier_classes[idx]}">
                <div>
                    <div class="pod-topbar">
                        <span class="pod-tier-chip">{tier_labels[idx]}</span>
                        <span class="pod-pct-badge {badge_colors[idx]}">{cand['similarity_pct']}%</span>
                    </div>
                    <div class="confidence-meter-track">
                        <div class="confidence-meter-fill" style="width: {cand['similarity_pct']}%;"></div>
                    </div>
                    <h3 class="star-name">{cand['name']}</h3>
                    <div class="star-meta">
                        <b>Peak: {cand['peak_overall']} OVR</b> (Ages {cand['peak_age_window']})<br>
                        <span>Sector: {cand['sector']}</span>
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
            """, unsafe_allow_html=True)

    # Tactical Fingerprint: Dual-Polygon Radar and Gap Telemetry
    if top_3:
        best_star = top_3[0]
        st.markdown(f"""
        <div class="section-header-hud" style="margin-top:2rem;">
            <span class="section-prefix">// 02</span>
            <span>TACTICAL FINGERPRINT: {prospect['short_name'].upper()} VS PRIME {best_star['name'].upper()}</span>
        </div>
        """, unsafe_allow_html=True)

        if not is_gk:
            macro_dimensions = {
                "Pace": ["movement_acceleration", "movement_sprint_speed"],
                "Shooting": ["attacking_finishing", "power_shot_power", "power_long_shots"],
                "Passing": ["attacking_short_passing", "skill_long_passing", "mentality_vision"],
                "Dribbling": ["skill_dribbling", "skill_ball_control", "movement_agility"],
                "Defending": ["defending_standing_tackle", "defending_marking_awareness", "mentality_interceptions"],
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

        radar_cats = list(macro_dimensions.keys())
        p_vals = []
        s_vals = []
        comparison_rows = []

        for dim, subcols in macro_dimensions.items():
            valid_p = [float(prospect[c]) for c in subcols if pd.notna(prospect[c])]
            p_mean = float(np.mean(valid_p)) if valid_p else 50.0
            valid_s = [float(best_star["attributes"].get(c, 50.0)) for c in subcols if pd.notna(best_star["attributes"].get(c))]
            s_mean = float(np.mean(valid_s)) if valid_s else 50.0
            p_vals.append(p_mean)
            s_vals.append(s_mean)
            delta = round(s_mean - p_mean, 1)
            comparison_rows.append({
                "dimension": dim,
                "p_val": round(p_mean, 1),
                "s_val": round(s_mean, 1),
                "delta": delta,
            })

        col_radar, col_telemetry = st.columns([3, 2])

        with col_radar:
            radar_chart = build_tactical_radar_altair(
                radar_cats, p_vals, s_vals,
                p_name=str(prospect["short_name"]),
                s_name=str(best_star["name"])
            )
            st.altair_chart(radar_chart, use_container_width=True)

        with col_telemetry:
            rows_html = []
            for cr in comparison_rows:
                delta_sign = f"+{cr['delta']}" if cr["delta"] > 0 else f"{cr['delta']}"
                delta_class = "neg" if cr["delta"] > 0 else "pos"
                rows_html.append(f"""
                <div class="gap-metric-row">
                    <span style="font-weight:700; color:#F8FAFC;">{cr['dimension']}</span>
                    <div>
                        <span class="gap-val-p">{cr['p_val']}</span>
                        <span style="color:#64748B;"> vs </span>
                        <span class="gap-val-s">{cr['s_val']}</span>
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
                    detailed_data.append({
                        "Skill Trait": col.replace("_", " ").title(),
                        f"Prospect ({prospect['short_name']})": round(float(p_val), 1),
                        f"Archetype ({best_star['name']})": round(float(s_val), 1),
                        "Net Gap": round(float(s_val) - float(p_val), 1),
                    })
            st.dataframe(pd.DataFrame(detailed_data), use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 2: METHODOLOGY & HISTORICAL MODEL AUDIT
# ==============================================================================
with tab_audit:
    st.markdown("""
    <div class="section-header-hud">
        <span class="section-prefix">// 03</span>
        <span>PHASE 3 EXPERIMENTAL AUDIT AND REPRODUCIBILITY RECORD</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
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
    """, unsafe_allow_html=True)
