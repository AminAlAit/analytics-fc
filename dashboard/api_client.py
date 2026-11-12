"""Resilient REST API client for Analytics FC scouting dashboard.

Communicates with the FastAPI microservice over HTTP when AFC_API_URL is configured,
with automatic embedded DuckDB and in-memory vector engine execution otherwise.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.catalog_search import (
    get_catalog_stats,
    get_connection,
    get_player_by_id_and_year,
    search_players_paginated,
)
from src.models.ceiling_matcher import compute_vector_matching, load_gallery_cached
from src.models.conformal_predictor import ConformalTrajectoryPredictor
from src.models.trajectory_predictor import TrajectoryFeatureContract
from src.models.trajectory_regressor import QuantileTrajectoryRegressor

logger = logging.getLogger("analytics_fc.dashboard.api_client")

DEFAULT_TIMEOUT_SEC = 1.0
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_QUANTILE_MODEL_PATH = BASE_DIR / "models" / "trajectory_regressor.joblib"
DEFAULT_CONFORMAL_MODEL_PATH = BASE_DIR / "models" / "conformal_predictor.joblib"


def _get_api_base_url() -> str | None:
    """Return configured external FastAPI base URL, or None for embedded mode."""
    raw = os.environ.get("AFC_API_URL", "").strip().rstrip("/")
    return raw if raw else None


def is_api_online(timeout: float = 0.5) -> bool:
    """Check if a configured FastAPI backend service is reachable and healthy."""
    base_url = _get_api_base_url()
    if not base_url:
        return False

    url = f"{base_url}/health"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return bool(resp.status == 200)
    except Exception:
        return False


def get_connection_status() -> str:
    """Return human-readable badge status for backend telemetry bar."""
    if is_api_online():
        return "FASTAPI REST ONLINE"
    return "EMBEDDED DUCKDB ENGINE"


def get_catalog_stats_summary() -> dict[str, int]:
    """Retrieve verified catalog row counts, distinct players, and edition range."""
    base_url = _get_api_base_url()
    if base_url:
        url = f"{base_url}/api/v1/players/stats/catalog"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_SEC) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.debug(f"FastAPI stats unavailable ({exc}), using embedded DuckDB.")

    return get_catalog_stats()


def search_players(
    query: str = "",
    edition_year: int | None = None,
    min_overall: int | None = None,
    max_overall: int | None = None,
    sector: str | None = None,
    limit: int = 50,
    offset: int = 0,
    include_sector_counts: bool = True,
) -> dict[str, Any]:
    """Execute player search via FastAPI endpoint or embedded DuckDB engine."""
    base_url = _get_api_base_url()
    if base_url:
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "include_sector_counts": str(include_sector_counts).lower(),
        }
        if query:
            params["query"] = query
        if edition_year is not None:
            params["edition_year"] = edition_year
        if min_overall is not None:
            params["min_overall"] = min_overall
        if max_overall is not None:
            params["max_overall"] = max_overall
        if sector is not None:
            params["sector"] = sector

        url = f"{base_url}/api/v1/players/search?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_SEC) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.debug(f"FastAPI search unavailable ({exc}), invoking embedded DuckDB.")

    return search_players_paginated(
        query=query,
        edition_year=edition_year,
        min_overall=min_overall,
        max_overall=max_overall,
        sector=sector,
        limit=limit,
        offset=offset,
        include_sector_counts=include_sector_counts,
    )


def get_player_profile(
    player_id: int,
    edition_year: int,
) -> dict[str, Any] | None:
    """Retrieve complete 34-skill player profile via FastAPI or embedded DuckDB."""
    base_url = _get_api_base_url()
    if base_url:
        params = urllib.parse.urlencode({"edition_year": edition_year})
        url = f"{base_url}/api/v1/players/{player_id}?{params}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_SEC) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as http_err:
            if http_err.code == 404:
                return None
            logger.debug(f"HTTP error {http_err.code} on player profile, using embedded DuckDB.")
        except Exception as exc:
            logger.debug(f"FastAPI profile lookup unavailable ({exc}), invoking embedded DuckDB.")

    return get_player_by_id_and_year(
        player_id=player_id,
        fifa_edition_year=edition_year,
    )


def match_ceiling(
    player_id: int,
    edition_year: int,
    top_k: int = 3,
) -> dict[str, Any] | None:
    """Compute superstar ceiling similarity via FastAPI or embedded vector math."""
    base_url = _get_api_base_url()
    if base_url:
        url = f"{base_url}/api/v1/match/ceiling"
        payload = json.dumps(
            {
                "player_id": player_id,
                "fifa_edition_year": edition_year,
                "top_k": top_k,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_SEC) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as http_err:
            if http_err.code == 404:
                return None
            logger.debug(f"HTTP error {http_err.code} on ceiling match, using embedded matcher.")
        except Exception as exc:
            logger.debug(f"FastAPI matcher unavailable ({exc}), invoking embedded matcher.")

    prospect = get_player_by_id_and_year(player_id, edition_year)
    if prospect is None:
        return None

    gallery = load_gallery_cached()
    matches, radar_dims = compute_vector_matching(prospect, gallery, top_k=top_k)

    return {
        "prospect": prospect,
        "matches": matches,
        "macro_radar": radar_dims,
        "kernel": "gaussian_rbf",
        "inference_time_ms": 1.0,
    }


def _safe_numeric(value: Any, default: float = 50.0) -> float:
    """Convert nullable or NaN attribute values safely to float."""
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(numeric) else numeric


def _predict_single_hgb(estimator: Any, x_arr: np.ndarray) -> float:
    """Run single-row prediction against fitted HistGradientBoosting estimator."""
    if hasattr(estimator, "_predict_iterations") and hasattr(estimator, "_baseline_prediction"):
        out = np.full((1, 1), estimator._baseline_prediction, dtype=np.float64, order="F")
        estimator._predict_iterations(x_arr, estimator._predictors, out, False, 1)
        return float(out[0, 0])
    x_df = pd.DataFrame(x_arr, columns=TrajectoryFeatureContract.REQUIRED_FEATURES)
    return float(estimator.predict(x_df)[0])


def load_trajectory_models(
    quantile_path: Path | str | None = None,
    conformal_path: Path | str | None = None,
) -> tuple[QuantileTrajectoryRegressor, ConformalTrajectoryPredictor | None]:
    """Load serialized Model 2 quantile and optional conformal predictor artifacts."""
    q_path = Path(quantile_path) if quantile_path is not None else DEFAULT_QUANTILE_MODEL_PATH
    c_path = Path(conformal_path) if conformal_path is not None else DEFAULT_CONFORMAL_MODEL_PATH

    quantile_model = QuantileTrajectoryRegressor.load(q_path)
    conformal_model: ConformalTrajectoryPredictor | None = None
    if c_path.exists():
        conformal_model = ConformalTrajectoryPredictor.load(c_path)
    return quantile_model, conformal_model


def _fetch_trajectory_player_record(
    player_id: int,
    edition_year: int,
    db_path: Path | str | None = None,
) -> tuple[dict[str, Any] | None, float]:
    """Fetch player attributes and 1-year growth velocity from DuckDB."""
    con = get_connection(db_path)
    try:
        has_fct = con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = 'fct_player_seasons';
            """
        ).fetchone()
        if has_fct and has_fct[0] > 0:
            sql_fct = """
                SELECT
                    player_id,
                    fifa_edition_year,
                    short_name,
                    age,
                    overall,
                    potential,
                    pace,
                    shooting,
                    passing,
                    dribbling,
                    defending,
                    physic,
                    COALESCE(annual_growth_velocity, 0.0) AS growth_1y
                FROM main.fct_player_seasons
                WHERE player_id = ? AND fifa_edition_year = ?
                LIMIT 1;
            """
            cursor = con.execute(sql_fct, [player_id, edition_year])
            cols = [desc[0] for desc in cursor.description]
            row = cursor.fetchone()
            if row is not None:
                rec = dict(zip(cols, row))
                growth_1y = round(float(rec.pop("growth_1y", 0.0)), 2)
                return rec, growth_1y

        has_stg = con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = 'stg_fifa_granular';
            """
        ).fetchone()
        if has_stg and has_stg[0] > 0:
            sql_stg = """
                WITH player_hist AS (
                    SELECT
                        player_id,
                        fifa_edition_year,
                        short_name,
                        age,
                        overall,
                        potential,
                        pace,
                        shooting,
                        passing,
                        dribbling,
                        defending,
                        physic,
                        LAG(fifa_edition_year) OVER (ORDER BY fifa_edition_year) AS prev_year,
                        LAG(overall) OVER (ORDER BY fifa_edition_year) AS prev_overall
                    FROM stg_fifa_granular
                    WHERE player_id = ?
                )
                SELECT
                    player_id,
                    fifa_edition_year,
                    short_name,
                    age,
                    overall,
                    potential,
                    pace,
                    shooting,
                    passing,
                    dribbling,
                    defending,
                    physic,
                    CASE
                        WHEN prev_year IS NOT NULL AND (fifa_edition_year - prev_year) = 1
                        THEN CAST(overall - prev_overall AS DOUBLE)
                        ELSE 0.0
                    END AS growth_1y
                FROM player_hist
                WHERE fifa_edition_year = ?
                LIMIT 1;
            """
            cursor = con.execute(sql_stg, [player_id, edition_year])
            cols = [desc[0] for desc in cursor.description]
            row = cursor.fetchone()
            if row is not None:
                rec = dict(zip(cols, row))
                growth_1y = round(float(rec.pop("growth_1y", 0.0)), 2)
                return rec, growth_1y

        return None, 0.0
    finally:
        con.close()


def forecast_trajectory(
    player_id: int,
    edition_year: int,
    horizon_years: int = 3,
    include_conformal: bool = True,
    quantile_model: QuantileTrajectoryRegressor | None = None,
    conformal_model: ConformalTrajectoryPredictor | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Compute 1 to 3 year quantile + conformal trajectory forecast via API v2 or embedded models."""
    safe_horizon = max(1, min(3, int(horizon_years)))
    base_url = _get_api_base_url()
    if base_url:
        url = f"{base_url}/api/v2/trajectories/forecast"
        payload = json.dumps(
            {
                "player_id": player_id,
                "edition_year": edition_year,
                "horizon_years": safe_horizon,
                "include_conformal": include_conformal,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_SEC) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as http_err:
            if http_err.code == 404:
                return None
            logger.debug(
                f"HTTP error {http_err.code} on trajectory forecast, using embedded models."
            )
        except Exception as exc:
            logger.debug(f"FastAPI v2 forecast unavailable ({exc}), using embedded models.")

    t0 = time.perf_counter()
    record, growth_1y = _fetch_trajectory_player_record(
        player_id=player_id,
        edition_year=edition_year,
        db_path=db_path,
    )
    if record is None:
        return None

    active_q = quantile_model
    active_c = conformal_model
    if active_q is None:
        try:
            active_q, loaded_c = load_trajectory_models()
        except FileNotFoundError:
            logger.debug("Trajectory quantile artifact not found locally; returning None.")
            return None
        if active_c is None:
            active_c = loaded_c

    conformal_enabled = bool(include_conformal and active_c is not None and active_c.is_calibrated)

    curr_age = int(record["age"])
    curr_ovr = float(record["overall"])
    pot = float(record["potential"])
    ref_year = int(record["fifa_edition_year"])

    pace = _safe_numeric(record.get("pace"), 50.0)
    shooting = _safe_numeric(record.get("shooting"), 50.0)
    passing = _safe_numeric(record.get("passing"), 50.0)
    dribbling = _safe_numeric(record.get("dribbling"), 50.0)
    defending = _safe_numeric(record.get("defending"), 50.0)
    physic = _safe_numeric(record.get("physic"), 50.0)

    state_age = float(curr_age)
    state_ovr = curr_ovr
    state_vel = growth_1y
    x_arr = np.empty((1, 11), dtype=np.float64)
    forecast_points: list[dict[str, Any]] = []

    for step in range(1, safe_horizon + 1):
        x_arr[0, 0] = state_age
        x_arr[0, 1] = state_ovr
        x_arr[0, 2] = pot
        x_arr[0, 3] = pot - state_ovr
        x_arr[0, 4] = pace
        x_arr[0, 5] = shooting
        x_arr[0, 6] = passing
        x_arr[0, 7] = dribbling
        x_arr[0, 8] = defending
        x_arr[0, 9] = physic
        x_arr[0, 10] = state_vel

        q10_delta = _predict_single_hgb(active_q.estimator_q10, x_arr)
        med_delta = _predict_single_hgb(active_q.estimator_median, x_arr)
        q90_delta = _predict_single_hgb(active_q.estimator_q90, x_arr)

        raw_q10 = float(np.clip(state_ovr + q10_delta, 35.0, 99.0))
        raw_med = float(np.clip(state_ovr + med_delta, 35.0, 99.0))
        raw_q90 = float(np.clip(state_ovr + q90_delta, 35.0, 99.0))

        q10_ovr = round(min(raw_q10, raw_med, raw_q90), 2)
        q50_ovr = round(raw_med, 2)
        q90_ovr = round(max(raw_q10, raw_med, raw_q90), 2)

        conf_low: float | None = None
        conf_high: float | None = None
        if conformal_enabled and active_c is not None:
            c_center = _predict_single_hgb(active_c.estimator, x_arr)
            c_margin = active_c.q_val * float(np.sqrt(step))
            conf_low = round(float(np.clip(state_ovr + c_center - c_margin, 35.0, 99.0)), 2)
            conf_high = round(float(np.clip(state_ovr + c_center + c_margin, 35.0, 99.0)), 2)

        forecast_points.append(
            {
                "target_edition_year": ref_year + step,
                "projected_age": curr_age + step,
                "predicted_growth": round(med_delta, 2),
                "predicted_overall": q50_ovr,
                "q10_overall": q10_ovr,
                "q50_overall": q50_ovr,
                "q90_overall": q90_ovr,
                "conformal_lower": conf_low,
                "conformal_upper": conf_high,
            }
        )

        state_age += 1.0
        state_vel = round(q50_ovr - state_ovr, 2)
        state_ovr = q50_ovr

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "player_id": int(record["player_id"]),
        "short_name": str(record["short_name"]),
        "reference_edition_year": ref_year,
        "current_age": curr_age,
        "current_overall": curr_ovr,
        "potential": pot,
        "historical_growth_1y": growth_1y,
        "forecast": forecast_points,
        "quantile_model_version": active_q.model_version,
        "conformal_model_version": (
            active_c.model_version if (conformal_enabled and active_c is not None) else None
        ),
        "inference_time_ms": elapsed_ms,
    }
