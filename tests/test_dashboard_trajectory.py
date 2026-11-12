"""Unit tests for Model 2 trajectory dashboard client and visualization helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import altair as alt
import duckdb
import numpy as np
import pandas as pd

from dashboard import api_client
from dashboard.app import build_trajectory_fan_chart, build_trajectory_fan_data
from src.catalog_search import ALL_SKILL_COLUMNS
from src.models.conformal_predictor import ConformalTrajectoryPredictor
from src.models.trajectory_predictor import TrajectoryFeatureContract
from src.models.trajectory_regressor import QuantileTrajectoryRegressor


def _build_training_frame(n_rows: int = 40) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate a tiny deterministic feature matrix matching TrajectoryFeatureContract."""
    rng = np.random.default_rng(42)
    age = rng.integers(17, 34, size=n_rows).astype(float)
    overall = rng.integers(60, 88, size=n_rows).astype(float)
    potential = np.clip(overall + rng.integers(0, 12, size=n_rows), 60.0, 96.0)
    df = pd.DataFrame(
        {
            "age": age,
            "overall": overall,
            "potential": potential,
            "potential_diff": potential - overall,
            "pace": rng.uniform(50.0, 92.0, size=n_rows),
            "shooting": rng.uniform(45.0, 90.0, size=n_rows),
            "passing": rng.uniform(48.0, 88.0, size=n_rows),
            "dribbling": rng.uniform(50.0, 91.0, size=n_rows),
            "defending": rng.uniform(30.0, 85.0, size=n_rows),
            "physic": rng.uniform(52.0, 89.0, size=n_rows),
            "growth_rate_1y": rng.uniform(-2.0, 5.0, size=n_rows),
        }
    )[TrajectoryFeatureContract.REQUIRED_FEATURES]
    y = np.clip(0.35 * df["potential_diff"].to_numpy() - 0.1 * (age - 21.0), -3.0, 6.0)
    return df, y


def _fit_mini_trajectory_models() -> tuple[
    QuantileTrajectoryRegressor, ConformalTrajectoryPredictor
]:
    """Fit lightweight quantile and conformal trajectory models for fast tests."""
    X_all, y_all = _build_training_frame(48)
    X_train, y_train = X_all.iloc[:32].reset_index(drop=True), y_all[:32]
    X_calib, y_calib = X_all.iloc[32:].reset_index(drop=True), y_all[32:]

    q_model = QuantileTrajectoryRegressor(
        max_iter=12,
        min_samples_leaf=4,
        random_state=42,
        model_version="0.2.0-test-q",
    ).fit(X_train, y_train)

    c_model = (
        ConformalTrajectoryPredictor(alpha=0.10, random_state=42, model_version="0.2.0-test-c")
        .fit(X_train, y_train)
        .calibrate(X_calib, y_calib)
    )
    return q_model, c_model


def _create_fct_db(db_path: Path) -> None:
    """Create a synthetic DuckDB with main.fct_player_seasons (outfield + GK null row)."""
    con = duckdb.connect(str(db_path), read_only=False)
    try:
        con.execute("SET threads = 1;")
        con.execute("SET memory_limit = '1GB';")
        con.execute(
            """
            CREATE TABLE main.fct_player_seasons (
                player_id INTEGER,
                fifa_edition_year INTEGER,
                short_name VARCHAR,
                age INTEGER,
                overall INTEGER,
                potential INTEGER,
                pace DOUBLE,
                shooting DOUBLE,
                passing DOUBLE,
                dribbling DOUBLE,
                defending DOUBLE,
                physic DOUBLE,
                annual_growth_velocity DOUBLE
            );
            """
        )
        con.execute(
            """
            INSERT INTO main.fct_player_seasons VALUES
                (239085, 2022, 'E. Haaland', 21, 88, 93, 89.0, 91.0, 65.0, 80.0, 45.0, 88.0, 4.0),
                (192448, 2022, 'M. ter Stegen', 29, 90, 90, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
            """
        )
    finally:
        con.close()


def _create_stg_only_db(db_path: Path) -> None:
    """Create a synthetic DuckDB with only main.stg_fifa_granular to verify LAG fallback."""
    con = duckdb.connect(str(db_path), read_only=False)
    try:
        con.execute("SET threads = 1;")
        con.execute("SET memory_limit = '1GB';")
        stg_cols = {
            "player_id": "INTEGER",
            "fifa_edition_year": "INTEGER",
            "short_name": "VARCHAR",
            "age": "INTEGER",
            "overall": "INTEGER",
            "potential": "INTEGER",
            "pace": "DOUBLE",
            "shooting": "DOUBLE",
            "passing": "DOUBLE",
            "dribbling": "DOUBLE",
            "defending": "DOUBLE",
            "physic": "DOUBLE",
        }
        for col in ALL_SKILL_COLUMNS:
            stg_cols[col] = "INTEGER"
        ddl = ", ".join(f"{name} {dtype}" for name, dtype in stg_cols.items())
        con.execute(f"CREATE TABLE main.stg_fifa_granular ({ddl});")

        skill_vals = [78 if not c.startswith("goalkeeping_") else 11 for c in ALL_SKILL_COLUMNS]
        placeholders = ", ".join(["?"] * len(stg_cols))
        con.execute(
            f"INSERT INTO main.stg_fifa_granular VALUES ({placeholders});",
            [
                239085,
                2021,
                "E. Haaland",
                20,
                84,
                92,
                87.0,
                89.0,
                63.0,
                78.0,
                44.0,
                86.0,
                *skill_vals,
            ],
        )
        con.execute(
            f"INSERT INTO main.stg_fifa_granular VALUES ({placeholders});",
            [
                239085,
                2022,
                "E. Haaland",
                21,
                88,
                93,
                89.0,
                91.0,
                65.0,
                80.0,
                45.0,
                88.0,
                *skill_vals,
            ],
        )
    finally:
        con.close()


def test_forecast_trajectory_multi_horizon_and_conformal_toggle(tmp_path: Path) -> None:
    """Embedded forecast computes monotone quantiles, conformal bounds, and horizon lengths."""
    db_path = tmp_path / "analytics_fc.duckdb"
    _create_fct_db(db_path)
    q_model, c_model = _fit_mini_trajectory_models()

    resp_3y = api_client.forecast_trajectory(
        player_id=239085,
        edition_year=2022,
        horizon_years=3,
        include_conformal=True,
        quantile_model=q_model,
        conformal_model=c_model,
        db_path=db_path,
    )
    assert resp_3y is not None
    assert resp_3y["player_id"] == 239085
    assert resp_3y["short_name"] == "E. Haaland"
    assert resp_3y["historical_growth_1y"] == 4.0
    assert len(resp_3y["forecast"]) == 3

    for idx, pt in enumerate(resp_3y["forecast"], start=1):
        assert pt["target_edition_year"] == 2022 + idx
        assert pt["projected_age"] == 21 + idx
        assert 35.0 <= pt["q10_overall"] <= pt["q50_overall"] <= pt["q90_overall"] <= 99.0
        assert pt["conformal_lower"] is not None
        assert pt["conformal_upper"] is not None
        assert 35.0 <= pt["conformal_lower"] <= pt["conformal_upper"] <= 99.0

    resp_1y_no_conf = api_client.forecast_trajectory(
        player_id=239085,
        edition_year=2022,
        horizon_years=1,
        include_conformal=False,
        quantile_model=q_model,
        conformal_model=c_model,
        db_path=db_path,
    )
    assert resp_1y_no_conf is not None
    assert len(resp_1y_no_conf["forecast"]) == 1
    assert resp_1y_no_conf["conformal_model_version"] is None
    assert resp_1y_no_conf["forecast"][0]["conformal_lower"] is None
    assert resp_1y_no_conf["forecast"][0]["conformal_upper"] is None


def test_forecast_trajectory_stg_lag_fallback_and_goalkeeper_nulls(tmp_path: Path) -> None:
    """Verify LAG growth fallback without fct_player_seasons and 50.0 GK null imputation."""
    stg_db = tmp_path / "stg_only.duckdb"
    _create_stg_only_db(stg_db)
    q_model, c_model = _fit_mini_trajectory_models()

    stg_resp = api_client.forecast_trajectory(
        player_id=239085,
        edition_year=2022,
        horizon_years=2,
        include_conformal=True,
        quantile_model=q_model,
        conformal_model=c_model,
        db_path=stg_db,
    )
    assert stg_resp is not None
    assert stg_resp["historical_growth_1y"] == 4.0
    assert len(stg_resp["forecast"]) == 2

    fct_db = tmp_path / "gk_nulls.duckdb"
    _create_fct_db(fct_db)
    gk_resp = api_client.forecast_trajectory(
        player_id=192448,
        edition_year=2022,
        horizon_years=2,
        include_conformal=True,
        quantile_model=q_model,
        conformal_model=c_model,
        db_path=fct_db,
    )
    assert gk_resp is not None
    assert gk_resp["historical_growth_1y"] == 0.0
    for pt in gk_resp["forecast"]:
        assert 35.0 <= pt["q10_overall"] <= pt["q50_overall"] <= pt["q90_overall"] <= 99.0


def test_forecast_trajectory_missing_player_and_load_models(tmp_path: Path) -> None:
    """Verify None on missing player and joblib round-trip loading."""
    db_path = tmp_path / "analytics_fc.duckdb"
    _create_fct_db(db_path)
    q_model, c_model = _fit_mini_trajectory_models()

    q_path = tmp_path / "trajectory_regressor.joblib"
    c_path = tmp_path / "conformal_predictor.joblib"
    q_model.save(q_path)
    c_model.save(c_path)

    loaded_q, loaded_c = api_client.load_trajectory_models(q_path, c_path)
    assert loaded_q.model_version == "0.2.0-test-q"
    assert loaded_c is not None
    assert loaded_c.model_version == "0.2.0-test-c"

    loaded_q_only, missing_c = api_client.load_trajectory_models(
        q_path, tmp_path / "nonexistent_conformal.joblib"
    )
    assert loaded_q_only.is_fitted
    assert missing_c is None

    missing_resp = api_client.forecast_trajectory(
        player_id=99999999,
        edition_year=2022,
        horizon_years=3,
        quantile_model=loaded_q,
        conformal_model=loaded_c,
        db_path=db_path,
    )
    assert missing_resp is None

    with (
        patch("dashboard.api_client._get_api_base_url", return_value=None),
        patch(
            "dashboard.api_client.load_trajectory_models",
            side_effect=FileNotFoundError("models/trajectory_regressor.joblib missing"),
        ),
    ):
        no_model_resp = api_client.forecast_trajectory(
            player_id=239085,
            edition_year=2022,
            horizon_years=3,
            db_path=db_path,
        )
    assert no_model_resp is None


def test_forecast_trajectory_remote_api_mode() -> None:
    """When AFC_API_URL is reachable, forecast_trajectory delegates to v2 endpoint."""
    fake_payload: dict[str, Any] = {
        "player_id": 239085,
        "short_name": "E. Haaland",
        "reference_edition_year": 2022,
        "current_age": 21,
        "current_overall": 88.0,
        "potential": 93.0,
        "historical_growth_1y": 4.0,
        "forecast": [
            {
                "target_edition_year": 2023,
                "projected_age": 22,
                "predicted_growth": 1.5,
                "predicted_overall": 89.5,
                "q10_overall": 87.2,
                "q50_overall": 89.5,
                "q90_overall": 91.8,
                "conformal_lower": 86.0,
                "conformal_upper": 93.0,
            }
        ],
        "quantile_model_version": "0.2.0-api",
        "conformal_model_version": "0.2.0-api-conf",
        "inference_time_ms": 3.4,
    }

    class _FakeHttpResp:
        status = 200

        def read(self) -> bytes:
            return json.dumps(fake_payload).encode("utf-8")

        def __enter__(self) -> _FakeHttpResp:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

    with (
        patch("dashboard.api_client.is_api_online", return_value=True),
        patch("dashboard.api_client._get_api_base_url", return_value="http://api.local:8000"),
        patch(
            "dashboard.api_client.urllib.request.urlopen", return_value=_FakeHttpResp()
        ) as urlopen_mock,
    ):
        resp = api_client.forecast_trajectory(239085, 2022, horizon_years=1, include_conformal=True)

    assert resp == fake_payload
    urlopen_mock.assert_called_once()
    req_arg = urlopen_mock.call_args.args[0]
    req_url = getattr(req_arg, "full_url", str(req_arg))
    assert req_url.endswith("/api/v2/trajectories/forecast")


def test_build_trajectory_fan_data_and_chart_spec(tmp_path: Path) -> None:
    """Fan DataFrame anchors at baseline season and compiles an Altair LayerChart fast."""
    db_path = tmp_path / "analytics_fc.duckdb"
    _create_fct_db(db_path)
    q_model, c_model = _fit_mini_trajectory_models()

    forecast_resp = api_client.forecast_trajectory(
        player_id=239085,
        edition_year=2022,
        horizon_years=3,
        include_conformal=True,
        quantile_model=q_model,
        conformal_model=c_model,
        db_path=db_path,
    )
    assert forecast_resp is not None

    fan_df = build_trajectory_fan_data(forecast_resp)
    assert len(fan_df) == 4
    assert int(fan_df.iloc[0]["year"]) == 2022
    assert float(fan_df.iloc[0]["q50"]) == 88.0
    assert float(fan_df.iloc[0]["q10"]) == 88.0
    assert float(fan_df.iloc[0]["q90"]) == 88.0
    assert (
        not fan_df[["year", "q10", "q50", "q90", "conformal_lower", "conformal_upper"]]
        .isna()
        .any()
        .any()
    )

    t0 = time.perf_counter()
    chart = build_trajectory_fan_chart(forecast_resp)
    chart_dict = chart.to_dict()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert isinstance(chart, alt.LayerChart)
    assert "layer" in chart_dict
    assert len(chart_dict["layer"]) >= 3
    assert elapsed_ms < 500.0
