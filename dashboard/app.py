import json
import streamlit as st
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

# --- Configuration ---
st.set_page_config(
    page_title="Analytics FC | Control Room",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for elegance and readability
st.markdown("""
<style>
    h1, h2, h3 {
        color: #111827;
        font-weight: 700;
    }
    .metric-card {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        padding: 1.25rem;
        border-radius: 0.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-val {
        font-size: 1.75rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .archetype-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-top: 4px solid #2563EB;
        padding: 1.25rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        height: 100%;
    }
    .match-badge {
        font-size: 1.5rem;
        font-weight: 800;
        color: #10B981;
    }
</style>
""", unsafe_allow_html=True)

# --- Paths ---
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
    "CF": "Center Forwards & Strikers", "ST": "Center Forwards & Strikers"
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
    "goalkeeping_positioning", "goalkeeping_reflexes"
]
OUTFIELD_COLS = [col for col in SKILL_COLS if not col.startswith("goalkeeping_")]

# --- Cached Resource Loading ---
@st.cache_resource
def load_gallery():
    if not GALLERY_PATH.exists():
        return []
    with open(GALLERY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def get_popular_wonderkids():
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    con.execute("SET threads = 1;")
    query = """
    SELECT 
        player_id, short_name, age, fifa_edition_year, overall, potential, positions,
        pace, shooting, passing, dribbling, defending, physic,
        height_cm, weight_kg,
        """ + ", ".join(SKILL_COLS) + """
    FROM stg_fifa_granular
    WHERE age <= 21 AND overall >= 78
    ORDER BY overall DESC, potential DESC
    LIMIT 150
    """
    df = con.execute(query).df()
    con.close()
    return df

# --- UI Setup ---
st.title("⚽ Analytics FC | Scouting & Intelligence Engine")
tab_matcher, tab_legacy = st.tabs(["🌟 The Superstar Ceiling Matcher", "🏛️ Legacy Trophy Predictor (PoC)"])

gallery = load_gallery()

# ==============================================================================
# TAB 1: THE SUPERSTAR CEILING MATCHER
# ==============================================================================
with tab_matcher:
    st.markdown("""
    **Continuous Multi-Dimensional Archetype Projection**  
    Instead of betting on scalar wonderkid potential, this engine projects an under-21 prospect's 
    granular technical and physical skill vector onto the **Top 100 historical superstar personal peaks**.
    """)
    st.divider()

    wonderkids_df = get_popular_wonderkids()
    
    # Selection Controls
    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        player_options = [
            f"{row['short_name']} (Age {row['age']} in FIFA {row['fifa_edition_year']} - {row['overall']} OVR)" 
            for _, row in wonderkids_df.iterrows()
        ]
        selected_option = st.selectbox("Select an Under-21 Prospect:", player_options)
        selected_idx = player_options.index(selected_option)
        prospect = wonderkids_df.iloc[selected_idx]

    with col_sel2:
        st.markdown(f"**Club / Nation Data:** Verified in DuckDB")
        st.markdown(f"**Positions:** `{prospect['positions']}`")

    # Prospect Snapshot Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='metric-card'><div class='metric-val'>{prospect['overall']}</div><div class='metric-lbl'>Current Overall</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><div class='metric-val'>{prospect['potential']}</div><div class='metric-lbl'>Stated Potential</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><div class='metric-val'>{prospect['age']}</div><div class='metric-lbl'>Evaluation Age</div></div>", unsafe_allow_html=True)
    with c4:
        primary_pos = str(prospect["positions"]).split(",")[0].strip().upper()
        sector = SECTOR_MAP.get(primary_pos, "Attacking Midfielders & Wingers")
        st.markdown(f"<div class='metric-card'><div class='metric-val' style='font-size:1.1rem; padding-top:0.5rem;'>{sector}</div><div class='metric-lbl'>Tactical Sector</div></div>", unsafe_allow_html=True)

    st.markdown("### 🏆 Projected Superstar Ceiling Archetypes")
    
    # Matching Engine (NumPy in-memory vector similarity <5ms)
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

        # Delta analysis
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

        peak_window = f"{star.get('peak_age_min', 26)}-{star.get('peak_age_max', 29)}"
        candidates.append({
            "name": star["short_name"],
            "sector": star["tactical_sector"],
            "peak_overall": star["peak_overall"],
            "peak_age_window": peak_window,
            "similarity_pct": calibrated_pct,
            "growth_needed": top_growth_needed,
            "strengths": strengths_aligned,
            "attributes": star["attributes"]
        })

    candidates.sort(key=lambda x: x["similarity_pct"], reverse=True)
    top_3 = candidates[:3]

    # Render Top 3 Cards
    card_cols = st.columns(3)
    for idx, cand in enumerate(top_3):
        with card_cols[idx]:
            st.markdown(f"""
            <div class='archetype-card'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='font-size:0.85rem; color:#6B7280; font-weight:600;'>ARCHETYPE #{idx+1}</span>
                    <span class='match-badge'>{cand['similarity_pct']}%</span>
                </div>
                <h3 style='margin:0.25rem 0;'>{cand['name']}</h3>
                <p style='color:#4B5563; font-size:0.9rem; margin-bottom:0.75rem;'>
                    <b>Peak: {cand['peak_overall']} OVR</b> (Ages {cand['peak_age_window']})<br>
                    <span style='color:#6B7280;'>{cand['sector']}</span>
                </p>
                <hr style='margin:0.5rem 0; border:none; border-top:1px solid #E5E7EB;'>
                <p style='font-size:0.85rem; margin-bottom:0.35rem;'><b>Closest Core Traits:</b></p>
                <p style='font-size:0.8rem; color:#059669; margin-bottom:0.5rem;'>
                    {', '.join([t['attribute'].replace('skill_', '').replace('movement_', '').replace('mentality_', '') for t in cand['strengths']])}
                </p>
                <p style='font-size:0.85rem; margin-bottom:0.35rem;'><b>Key Growth Priorities:</b></p>
                <p style='font-size:0.8rem; color:#D97706; margin-bottom:0;'>
                    {', '.join([f"{g['attribute'].replace('attacking_', '').replace('mentality_', '')} (+{g['gap']})" for g in cand['growth_needed'][:2]])}
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Visual Comparison with #1 Archetype
    if top_3:
        best_star = top_3[0]
        st.subheader(f"📊 Tactical Fingerprint: {prospect['short_name']} vs Prime {best_star['name']}")
        
        # Macro trait aggregation
        macro_traits = {
            "Pace": ["movement_acceleration", "movement_sprint_speed"],
            "Shooting": ["attacking_finishing", "power_shot_power", "power_long_shots"],
            "Passing": ["attacking_short_passing", "skill_long_passing", "mentality_vision"],
            "Dribbling": ["skill_dribbling", "skill_ball_control", "movement_agility"],
            "Defending": ["defending_standing_tackle", "defending_marking_awareness", "mentality_interceptions"],
            "Physical": ["power_stamina", "power_strength", "power_jumping"]
        }
        
        chart_data = []
        for macro, subcols in macro_traits.items():
            p_macro_val = np.mean([float(prospect[c]) for c in subcols if pd.notna(prospect[c])])
            s_macro_val = np.mean([float(best_star["attributes"].get(c, 50.0)) for c in subcols])
            chart_data.append({
                "Skill Dimension": macro,
                f"Prospect ({prospect['short_name']})": round(p_macro_val, 1),
                f"Peak Archetype ({best_star['name']})": round(s_macro_val, 1)
            })
            
        df_chart = pd.DataFrame(chart_data).set_index("Skill Dimension")
        st.bar_chart(df_chart, height=320)

# ==============================================================================
# TAB 2: LEGACY TROPHY PREDICTOR (POC ARCHIVE)
# ==============================================================================
with tab_legacy:
    st.info("""
    **Methodology Notice:** This model represents the Phase 3 proof of concept. 
    As documented in our technical audit, binary silverware predictions were parked due to 
    extreme Transfermarkt label sparsity (0.02% positive labels) and pair-level data leakage. 
    It is retained here for historical and audit reference.
    """)
    st.caption("Phase 3 Experiment Archive: Random Forest trained on pairwise dyads.")
