"""Interactive Model 2 trajectory fan chart.

Forecasts come from dashboard.api_client.forecast_trajectory. This module turns
that payload into a median line and a 10th to 90th percentile cone.
"""

from __future__ import annotations

from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.api_client import forecast_trajectory


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


def build_forecast_fan(
    player_id: int,
    edition_year: int,
    *,
    horizon_years: int = 3,
    include_conformal: bool = True,
    quantile_model: Any = None,
    conformal_model: Any = None,
    db_path: Any = None,
) -> tuple[dict[str, Any] | None, alt.LayerChart | None]:
    """Forecast through the dashboard client and draw the fan chart."""
    forecast_resp = forecast_trajectory(
        player_id=player_id,
        edition_year=edition_year,
        horizon_years=horizon_years,
        include_conformal=include_conformal,
        quantile_model=quantile_model,
        conformal_model=conformal_model,
        db_path=db_path,
    )
    if not forecast_resp or not forecast_resp.get("forecast"):
        return forecast_resp, None
    return forecast_resp, build_trajectory_fan_chart(forecast_resp)


def render_trajectory_forecaster(
    selected_prospect: dict[str, Any] | None,
    *,
    quantile_model: Any = None,
    conformal_model: Any = None,
) -> None:
    """Render horizon controls, KPI strip, and the trajectory fan chart."""
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

    forecast_resp = forecast_trajectory(
        player_id=player_id,
        edition_year=edition_year,
        horizon_years=horizon_years,
        include_conformal=include_conformal,
        quantile_model=quantile_model,
        conformal_model=conformal_model,
    )

    if not forecast_resp or not forecast_resp.get("forecast"):
        if quantile_model is None:
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
