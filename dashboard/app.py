import streamlit as st
import duckdb
import mlflow
import pandas as pd
from pathlib import Path

# --- Configuration ---
st.set_page_config(
    page_title="Analytics FC | Control Room",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for elegance
st.markdown("""
<style>
    h1 {
        color: #1F2937;
        font-weight: 700;
    }
    .metric-box {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2563EB;
    }
    .metric-label {
        font-size: 1rem;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# --- Paths ---
BASE_DIR = Path(__file__).parent.parent
DUCKDB_PATH = BASE_DIR / "dbt_project" / "analytics_fc.duckdb"
MLFLOW_URI = f"sqlite:///{BASE_DIR / 'mlflow.db'}"

# --- Data & Model Loading (Cached) ---
@st.cache_resource
def get_duckdb_conn():
    return duckdb.connect(str(DUCKDB_PATH), read_only=True)

@st.cache_resource
def load_ml_model():
    mlflow.set_tracking_uri(MLFLOW_URI)
    exp = mlflow.get_experiment_by_name('Ultimate_Comparison_Test')
    if not exp:
        return None
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], filter_string="tags.mlflow.runName = '3_Combined_Model'")
    if runs.empty:
        return None
    
    run_id = runs.iloc[0].run_id
    model_uri = f"runs:/{run_id}/3_Combined_Model_model"
    model = mlflow.sklearn.load_model(model_uri)
    return model

@st.cache_data
def get_sample_players():
    conn = get_duckdb_conn()
    query = """
    SELECT 
        p.player_b_name as name,
        f.overall,
        f.max_potential,
        p.scdi_overall,
        p.scdi_potential,
        CASE WHEN t.short_name IS NOT NULL THEN 1 ELSE 0 END AS actual_trophy
    FROM mart_pattern_maker p
    INNER JOIN stg_fifatable f ON p.player_b_id = f.player_id AND f.age = 22
    LEFT JOIN (SELECT DISTINCT short_name FROM stg_tm_trophies) t 
      ON p.player_b_name = t.short_name
    LIMIT 100
    """
    df = conn.execute(query).df().dropna()
    # Clean up duplicates to make the list neat
    return df.drop_duplicates(subset=['name']).head(20).sort_values(by='name')

# --- Main UI ---
st.title("⚽ Analytics FC | Prospect Predictor")
st.markdown("Select a prospect from the sidebar to view their age-22 stats and our **Combined Model** (Raw Talent + SCDI Shape) prediction for future silverware.")

# Load data and model
df_players = get_sample_players()
model = load_ml_model()

if model is None:
    st.error("Could not load the Combined Model from MLflow.")
    st.stop()

# --- Sidebar ---
st.sidebar.header("Player Selection")
selected_name = st.sidebar.selectbox("Choose a Prospect", df_players['name'].tolist())

# --- Player Dashboard ---
player_data = df_players[df_players['name'] == selected_name].iloc[0]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""<div class="metric-box"><div class="metric-value">{player_data['overall']}</div><div class="metric-label">Overall (Age 22)</div></div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-box"><div class="metric-value">{player_data['max_potential']}</div><div class="metric-label">Max Potential</div></div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-box"><div class="metric-value">{player_data['scdi_overall']:.2f}</div><div class="metric-label">SCDI Overall Match</div></div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-box"><div class="metric-value">{player_data['scdi_potential']:.2f}</div><div class="metric-label">SCDI Potential Match</div></div>""", unsafe_allow_html=True)

st.divider()

# --- Prediction Section ---
st.subheader("🤖 AI Prediction")

# Prepare features exactly as the model expects them
features = pd.DataFrame([{
    'overall': player_data['overall'],
    'max_potential': player_data['max_potential'],
    'scdi_overall': player_data['scdi_overall'],
    'scdi_potential': player_data['scdi_potential']
}])

# Predict
prob = model.predict_proba(features)[0][1] * 100
prediction = model.predict(features)[0]

pred_col, act_col = st.columns(2)

with pred_col:
    st.markdown("#### Model Forecast")
    if prediction == 1:
        st.success(f"🏆 **Highly likely to win a trophy** ({prob:.1f}% confidence)")
    else:
        st.warning(f"❌ **Unlikely to win a trophy** ({prob:.1f}% confidence)")

with act_col:
    st.markdown("#### Historical Reality")
    if player_data['actual_trophy'] == 1:
        st.info("Verified: This player actually won silverware.")
    else:
        st.info("Verified: No silverware found for this player.")

st.divider()

# --- Feature Importance ---
st.subheader("📊 What Drives the AI's Decision?")
st.markdown("This chart extracts the internal **Feature Importances** directly from the Random Forest model, showing how heavily it weighs each stat when making a prediction.")

importances = model.feature_importances_
feature_names = ['Overall Rating', 'Max Potential', 'SCDI Overall Match', 'SCDI Potential Match']
df_importance = pd.DataFrame({
    "Importance Weight": importances
}, index=feature_names)

st.bar_chart(df_importance, horizontal=True)

st.divider()
st.caption("Phase 4/5: Local Dashboarding - Powered by DuckDB, MLflow & Streamlit")
