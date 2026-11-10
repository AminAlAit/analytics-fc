"""Asynchronous FastAPI v2 router for Analytics FC.

Exposes high-performance endpoints for:
1. Faceted player search across 180,021 modern records (2015 to 2024).
2. Multi-horizon quantile and conformal career trajectory forecasting.
3. Gaussian RBF superstar archetype vector similarity matching.
4. Single-pass comprehensive prospect scouting dossiers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Annotated, Any

import duckdb
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi import Path as PathParam
from pydantic import BaseModel, ConfigDict, Field

from src.api.v2.dependencies import (
    DuckDBSessionManager,
    ModelRegistry,
    get_archetype_indexer,
    get_conformal_predictor,
    get_db_session,
    get_model_registry,
    get_quantile_regressor,
    get_session_manager,
)
from src.catalog_search import (
    ALL_SKILL_COLUMNS,
    SECTOR_ORDER,
    get_primary_position,
    get_sector_for_position,
)
from src.models.archetype_indexer import ArchetypeVectorIndex
from src.models.conformal_predictor import ConformalTrajectoryPredictor
from src.models.trajectory_predictor import TrajectoryFeatureContract
from src.models.trajectory_regressor import QuantileTrajectoryRegressor

logger = logging.getLogger("analytics_fc.api.v2.router")

router = APIRouter(prefix="/api/v2", tags=["API v2"])

SEARCH_COLUMNS_V2 = [
    "player_id",
    "fifa_edition_year",
    "short_name",
    "long_name",
    "age",
    "nationality",
    "club",
    "league_name",
    "positions",
    "overall",
    "potential",
    "value_eur",
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",
]


# ==============================================================================
# PYDANTIC V2 REQUEST AND RESPONSE SCHEMAS
# ==============================================================================


class HealthStatusV2(BaseModel):
    """Diagnostic status for API v2 service, DuckDB connection, and ML artifacts."""

    status: str
    api_version: str = "2.0.0"
    database_connected: bool
    catalog_rows: int = Field(ge=0)
    distinct_players: int = Field(ge=0)
    min_year: int
    max_year: int
    models: dict[str, Any]
    hardware_limits: dict[str, Any]
    response_time_ms: float = Field(ge=0.0)


class PlayerSearchItemV2(BaseModel):
    """Player summary item returned in faceted v2 search results."""

    model_config = ConfigDict(from_attributes=True)

    player_id: int
    fifa_edition_year: int
    short_name: str
    long_name: str | None = None
    age: int = Field(ge=14, le=55)
    nationality: str | None = None
    club: str | None = None
    league_name: str | None = None
    positions: str
    primary_pos: str
    tactical_sector: str
    overall: int = Field(ge=35, le=99)
    potential: int = Field(ge=35, le=99)
    value_eur: float | None = None
    pace: int | None = None
    shooting: int | None = None
    passing: int | None = None
    dribbling: int | None = None
    defending: int | None = None
    physic: int | None = None


class SearchResponseV2(BaseModel):
    """Paginated faceted search response."""

    items: list[PlayerSearchItemV2]
    total_matching: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
    has_more: bool
    sector_counts: dict[str, int]
    inference_time_ms: float = Field(ge=0.0)


class TrajectoryForecastRequestV2(BaseModel):
    """Request schema for trajectory forecasting."""

    model_config = ConfigDict(extra="forbid")

    player_id: int = Field(gt=0)
    edition_year: int = Field(ge=2015, le=2024)
    horizon_years: int = Field(default=1, ge=1, le=3)
    include_conformal: bool = True


class TrajectoryPointV2(BaseModel):
    """Single-year projected trajectory milestone with quantile and conformal bounds."""

    target_edition_year: int
    projected_age: int = Field(ge=15, le=60)
    predicted_growth: float
    predicted_overall: float = Field(ge=35.0, le=99.0)
    q10_overall: float = Field(ge=35.0, le=99.0)
    q50_overall: float = Field(ge=35.0, le=99.0)
    q90_overall: float = Field(ge=35.0, le=99.0)
    conformal_lower: float | None = Field(default=None, ge=35.0, le=99.0)
    conformal_upper: float | None = Field(default=None, ge=35.0, le=99.0)


class TrajectoryForecastResponseV2(BaseModel):
    """Multi-horizon developmental forecast response."""

    player_id: int
    short_name: str
    reference_edition_year: int
    current_age: int
    current_overall: float
    potential: float
    historical_growth_1y: float
    forecast: list[TrajectoryPointV2]
    quantile_model_version: str
    conformal_model_version: str | None = None
    inference_time_ms: float = Field(ge=0.0)


class ArchetypeMatchRequestV2(BaseModel):
    """Request schema for Gaussian RBF superstar archetype matching."""

    model_config = ConfigDict(extra="forbid")

    player_id: int = Field(gt=0)
    edition_year: int = Field(ge=2015, le=2024)
    top_k: int = Field(default=3, ge=1, le=10)


class ArchetypeMatchItemV2(BaseModel):
    """Ranked superstar archetype similarity entry."""

    rank: int = Field(ge=1)
    player_id: int
    superstar_name: str
    tactical_sector: str
    primary_position: str
    peak_overall: int = Field(ge=35, le=99)
    similarity_score_pct: float = Field(ge=0.0, le=100.0)
    raw_distance: float
    penalized_distance: float
    top_growth_priorities: list[dict[str, Any]]
    closest_matching_traits: list[dict[str, Any]]


class ArchetypeMatchResponseV2(BaseModel):
    """Vector similarity response containing nearest prime superstar archetypes."""

    prospect_id: int
    prospect_name: str
    edition_year: int
    primary_position: str
    tactical_sector: str
    matches: list[ArchetypeMatchItemV2]
    inference_time_ms: float = Field(ge=0.0)


class ComprehensiveScoutResponseV2(BaseModel):
    """Unified scouting dossier combining profile, trajectory, and archetypes."""

    profile: PlayerSearchItemV2
    trajectory: TrajectoryForecastResponseV2
    archetypes: ArchetypeMatchResponseV2
    db_query_time_ms: float = Field(ge=0.0)
    total_inference_time_ms: float = Field(ge=0.0)


# ==============================================================================
# INTERNAL SYNCHRONOUS DATA AND INFERENCE HELPERS
# ==============================================================================


def _fetch_player_with_velocity(
    con: duckdb.DuckDBPyConnection,
    player_id: int,
    edition_year: int,
    include_skills: bool = False,
) -> tuple[dict[str, Any] | None, float]:
    """Retrieve target edition profile and 1-year growth velocity from materialized base tables."""
    cols = [*SEARCH_COLUMNS_V2, "COALESCE(annual_growth_velocity, 0.0) AS growth_1y"]
    select_sql = ", ".join(cols)
    sql = f"""
        SELECT {select_sql}
        FROM main.fct_player_seasons
        WHERE player_id = ? AND fifa_edition_year = ?
        LIMIT 1;
    """
    cursor = con.execute(sql, [player_id, edition_year])
    col_names = [desc[0] for desc in cursor.description]
    row = cursor.fetchone()

    if row is None:
        return None, 0.0

    rec = dict(zip(col_names, row))
    growth_1y = round(float(rec.pop("growth_1y", 0.0)), 2)
    rec["primary_pos"] = get_primary_position(rec.get("positions"))
    rec["tactical_sector"] = get_sector_for_position(rec.get("positions"))

    if include_skills:
        skill_sql = f"""
            SELECT {", ".join(ALL_SKILL_COLUMNS)}
            FROM raw.granular_15_24
            WHERE player_id = ? AND fifa_version = ?
            LIMIT 1;
        """
        skill_cursor = con.execute(skill_sql, [player_id, edition_year - 2000])
        skill_cols = [desc[0] for desc in skill_cursor.description]
        skill_row = skill_cursor.fetchone()
        if skill_row is not None:
            rec.update(dict(zip(skill_cols, skill_row)))

    return rec, growth_1y


def _predict_single_hgb(estimator: Any, X_arr: np.ndarray) -> float:
    """Fast single-threaded prediction for a single 1x11 float64 feature vector."""
    if hasattr(estimator, "_predict_iterations") and hasattr(estimator, "_baseline_prediction"):
        out = np.full((1, 1), estimator._baseline_prediction, dtype=np.float64, order="F")
        estimator._predict_iterations(X_arr, estimator._predictors, out, False, 1)
        return float(out[0, 0])
    X_df = pd.DataFrame(X_arr, columns=TrajectoryFeatureContract.REQUIRED_FEATURES)
    return float(estimator.predict(X_df)[0])


def _run_trajectory_inference(
    record: dict[str, Any],
    growth_1y: float,
    horizon_years: int,
    include_conformal: bool,
    quantile_model: QuantileTrajectoryRegressor,
    conformal_model: ConformalTrajectoryPredictor | None,
) -> TrajectoryForecastResponseV2:
    """Execute iterative multi-horizon trajectory inference in memory."""
    t0 = time.perf_counter()

    curr_age = int(record["age"])
    curr_ovr = float(record["overall"])
    pot = float(record["potential"])
    ref_year = int(record["fifa_edition_year"])

    # Safe fallback imputation for goalkeepers or sparse records
    pace = float(record.get("pace") or 50.0)
    shooting = float(record.get("shooting") or 50.0)
    passing = float(record.get("passing") or 50.0)
    dribbling = float(record.get("dribbling") or 50.0)
    defending = float(record.get("defending") or 50.0)
    physic = float(record.get("physic") or 50.0)

    state_age = float(curr_age)
    state_ovr = curr_ovr
    state_vel = growth_1y

    forecast_points: list[TrajectoryPointV2] = []
    X_arr = np.empty((1, 11), dtype=np.float64)

    for step in range(1, horizon_years + 1):
        X_arr[0, 0] = state_age
        X_arr[0, 1] = state_ovr
        X_arr[0, 2] = pot
        X_arr[0, 3] = pot - state_ovr
        X_arr[0, 4] = pace
        X_arr[0, 5] = shooting
        X_arr[0, 6] = passing
        X_arr[0, 7] = dribbling
        X_arr[0, 8] = defending
        X_arr[0, 9] = physic
        X_arr[0, 10] = state_vel

        q10_delta = _predict_single_hgb(quantile_model.estimator_q10, X_arr)
        med_delta = _predict_single_hgb(quantile_model.estimator_median, X_arr)
        q90_delta = _predict_single_hgb(quantile_model.estimator_q90, X_arr)

        raw_q10 = float(np.clip(state_ovr + q10_delta, 35.0, 99.0))
        raw_med = float(np.clip(state_ovr + med_delta, 35.0, 99.0))
        raw_q90 = float(np.clip(state_ovr + q90_delta, 35.0, 99.0))

        q10_ovr = round(min(raw_q10, raw_med, raw_q90), 2)
        q50_ovr = round(raw_med, 2)
        q90_ovr = round(max(raw_q10, raw_med, raw_q90), 2)

        conf_low: float | None = None
        conf_high: float | None = None
        if include_conformal and conformal_model is not None and conformal_model.is_calibrated:
            c_center = _predict_single_hgb(conformal_model.estimator, X_arr)
            c_margin = conformal_model.q_val * float(np.sqrt(step))
            conf_low = round(float(np.clip(state_ovr + c_center - c_margin, 35.0, 99.0)), 2)
            conf_high = round(float(np.clip(state_ovr + c_center + c_margin, 35.0, 99.0)), 2)

        forecast_points.append(
            TrajectoryPointV2(
                target_edition_year=ref_year + step,
                projected_age=curr_age + step,
                predicted_growth=round(med_delta, 2),
                predicted_overall=q50_ovr,
                q10_overall=q10_ovr,
                q50_overall=q50_ovr,
                q90_overall=q90_ovr,
                conformal_lower=conf_low,
                conformal_upper=conf_high,
            )
        )

        # Advance state for next forecast step
        state_age += 1.0
        state_vel = round(q50_ovr - state_ovr, 2)
        state_ovr = q50_ovr

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return TrajectoryForecastResponseV2(
        player_id=int(record["player_id"]),
        short_name=str(record["short_name"]),
        reference_edition_year=ref_year,
        current_age=curr_age,
        current_overall=curr_ovr,
        potential=pot,
        historical_growth_1y=growth_1y,
        forecast=forecast_points,
        quantile_model_version=quantile_model.model_version,
        conformal_model_version=(conformal_model.model_version if conformal_model else None),
        inference_time_ms=elapsed_ms,
    )


def _run_archetype_inference(
    record: dict[str, Any],
    top_k: int,
    indexer: ArchetypeVectorIndex,
) -> ArchetypeMatchResponseV2:
    """Execute in-memory Gaussian RBF archetype search."""
    t0 = time.perf_counter()
    primary_pos = str(record["primary_pos"])
    player_id = int(record["player_id"])

    attrs = {col: float(record[col]) for col in ALL_SKILL_COLUMNS if record.get(col) is not None}
    matches = indexer.query(
        attributes=attrs,
        primary_position=primary_pos,
        top_k=top_k,
        exclude_player_id=player_id,
    )

    items = [
        ArchetypeMatchItemV2(
            rank=idx + 1,
            player_id=m.player_id,
            superstar_name=m.superstar_name,
            tactical_sector=m.tactical_sector,
            primary_position=m.primary_position,
            peak_overall=m.peak_overall,
            similarity_score_pct=m.similarity_score_pct,
            raw_distance=m.raw_distance,
            penalized_distance=m.penalized_distance,
            top_growth_priorities=m.top_growth_priorities,
            closest_matching_traits=m.closest_matching_traits,
        )
        for idx, m in enumerate(matches)
    ]

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return ArchetypeMatchResponseV2(
        prospect_id=player_id,
        prospect_name=str(record["short_name"]),
        edition_year=int(record["fifa_edition_year"]),
        primary_position=primary_pos,
        tactical_sector=str(record["tactical_sector"]),
        matches=items,
        inference_time_ms=elapsed_ms,
    )


def _execute_modern_search(
    con: duckdb.DuckDBPyConnection,
    query: str,
    edition_year: int | None,
    min_overall: int | None,
    max_overall: int | None,
    sector: str | None,
    limit: int,
    offset: int,
    include_sector_counts: bool,
) -> dict[str, Any]:
    """Execute faceted catalog search over materialized main.fct_player_seasons (2015 to 2024)."""
    clean_query = query.strip()
    base_clauses: list[str] = []
    base_params: list[Any] = []

    if clean_query:
        pattern = f"%{clean_query}%"
        if " " in clean_query:
            base_clauses.append("(short_name ILIKE ? OR long_name ILIKE ?)")
            base_params.extend([pattern, pattern])
        else:
            base_clauses.append("short_name ILIKE ?")
            base_params.append(pattern)

    if edition_year is not None:
        base_clauses.append("fifa_edition_year = ?")
        base_params.append(edition_year)

    if min_overall is not None:
        base_clauses.append("overall >= ?")
        base_params.append(min_overall)

    if max_overall is not None:
        base_clauses.append("overall <= ?")
        base_params.append(max_overall)

    base_where = f"WHERE {' AND '.join(base_clauses)}" if base_clauses else ""
    select_cols = ", ".join(SEARCH_COLUMNS_V2)

    sector_filter_sql = ""
    item_params = list(base_params)
    if sector is not None:
        canonical_sector = next(
            (s for s in SECTOR_ORDER if s.lower() == sector.strip().lower()), None
        )
        if canonical_sector is not None:
            sector_filter_sql = "WHERE tactical_sector = ?"
            item_params.append(canonical_sector)

    safe_limit = max(1, limit)
    safe_offset = max(0, offset)
    item_params.extend([safe_limit, safe_offset])

    # Single-scan CTE computes sector totals across base_clauses prior to sector filter
    sql = f"""
        WITH base AS (
            SELECT {select_cols},
                   TRIM(SPLIT_PART(positions, ',', 1)) AS primary_pos,
                   CASE
                       WHEN TRIM(SPLIT_PART(positions, ',', 1)) = 'GK' THEN 'Goalkeepers'
                       WHEN TRIM(SPLIT_PART(positions, ',', 1)) IN ('CB', 'LB', 'RB', 'LWB', 'RWB') THEN 'Defenders'
                       WHEN TRIM(SPLIT_PART(positions, ',', 1)) IN ('CDM', 'CM') THEN 'Defensive & Central Midfielders'
                       WHEN TRIM(SPLIT_PART(positions, ',', 1)) IN ('CAM', 'LM', 'RM', 'LW', 'RW') THEN 'Attacking Midfielders & Wingers'
                       WHEN TRIM(SPLIT_PART(positions, ',', 1)) IN ('CF', 'ST') THEN 'Center Forwards & Strikers'
                       ELSE 'Other'
                   END AS tactical_sector
            FROM main.fct_player_seasons
            {base_where}
        ),
        with_sector_counts AS (
            SELECT *,
                   SUM(CASE WHEN tactical_sector = 'Goalkeepers' THEN 1 ELSE 0 END) OVER() AS cnt_gk,
                   SUM(CASE WHEN tactical_sector = 'Defenders' THEN 1 ELSE 0 END) OVER() AS cnt_def,
                   SUM(CASE WHEN tactical_sector = 'Defensive & Central Midfielders' THEN 1 ELSE 0 END) OVER() AS cnt_mid,
                   SUM(CASE WHEN tactical_sector = 'Attacking Midfielders & Wingers' THEN 1 ELSE 0 END) OVER() AS cnt_am,
                   SUM(CASE WHEN tactical_sector = 'Center Forwards & Strikers' THEN 1 ELSE 0 END) OVER() AS cnt_st
            FROM base
        )
        SELECT *, COUNT(*) OVER() AS total_matching
        FROM with_sector_counts
        {sector_filter_sql}
        ORDER BY overall DESC, potential DESC, fifa_edition_year DESC, player_id ASC
        LIMIT ? OFFSET ?;
    """

    cursor = con.execute(sql, item_params)
    col_names = [desc[0] for desc in cursor.description]
    raw_rows = cursor.fetchall()

    items: list[dict[str, Any]] = []
    total_matching = 0
    sector_counts: dict[str, int] = {sec: 0 for sec in SECTOR_ORDER}

    for row in raw_rows:
        rec = dict(zip(col_names, row))
        total_matching = int(rec.pop("total_matching", 0))
        cnt_gk = int(rec.pop("cnt_gk", 0))
        cnt_def = int(rec.pop("cnt_def", 0))
        cnt_mid = int(rec.pop("cnt_mid", 0))
        cnt_am = int(rec.pop("cnt_am", 0))
        cnt_st = int(rec.pop("cnt_st", 0))
        if include_sector_counts and not items:
            sector_counts = {
                "Goalkeepers": cnt_gk,
                "Defenders": cnt_def,
                "Defensive & Central Midfielders": cnt_mid,
                "Attacking Midfielders & Wingers": cnt_am,
                "Center Forwards & Strikers": cnt_st,
            }
        items.append(rec)

    if not items and (safe_offset > 0 or include_sector_counts):
        if include_sector_counts:
            sector_sql = f"""
                SELECT
                    CASE
                        WHEN TRIM(SPLIT_PART(positions, ',', 1)) = 'GK' THEN 'Goalkeepers'
                        WHEN TRIM(SPLIT_PART(positions, ',', 1)) IN ('CB', 'LB', 'RB', 'LWB', 'RWB') THEN 'Defenders'
                        WHEN TRIM(SPLIT_PART(positions, ',', 1)) IN ('CDM', 'CM') THEN 'Defensive & Central Midfielders'
                        WHEN TRIM(SPLIT_PART(positions, ',', 1)) IN ('CAM', 'LM', 'RM', 'LW', 'RW') THEN 'Attacking Midfielders & Wingers'
                        WHEN TRIM(SPLIT_PART(positions, ',', 1)) IN ('CF', 'ST') THEN 'Center Forwards & Strikers'
                        ELSE 'Other'
                    END AS sec,
                    COUNT(*) AS cnt
                FROM main.fct_player_seasons
                {base_where}
                GROUP BY sec;
            """
            for sec, cnt in con.execute(sector_sql, base_params).fetchall():
                if sec in sector_counts:
                    sector_counts[sec] = int(cnt)
        if safe_offset > 0:
            if sector is not None:
                canonical_sector = next(
                    (s for s in SECTOR_ORDER if s.lower() == sector.strip().lower()), None
                )
                total_matching = sector_counts.get(canonical_sector or "", 0)
            else:
                total_matching = sum(sector_counts.values())

    return {
        "items": items,
        "total_matching": total_matching,
        "limit": safe_limit,
        "offset": safe_offset,
        "has_more": (safe_offset + len(items)) < total_matching,
        "sector_counts": sector_counts,
    }


# ==============================================================================
# ASYNCHRONOUS FASTAPI V2 ENDPOINTS
# ==============================================================================


@router.get(
    "/health",
    response_model=HealthStatusV2,
    summary="API v2 Diagnostics Probe",
)
async def health_check_v2(
    session_manager: Annotated[DuckDBSessionManager, Depends(get_session_manager)],
    registry: Annotated[ModelRegistry, Depends(get_model_registry)],
) -> HealthStatusV2:
    """Verify DuckDB connectivity, model registry warmup, and hardware limits."""
    t0 = time.perf_counter()

    def _run_diag() -> tuple[int, int, int, int, dict[str, Any]]:
        con = session_manager.get_connection()
        try:
            row = con.execute(
                """
                SELECT
                    COUNT(*) AS total_rows,
                    COUNT(DISTINCT player_id) AS distinct_players,
                    MIN(fifa_edition_year) AS min_year,
                    MAX(fifa_edition_year) AS max_year
                FROM main.fct_player_seasons;
                """
            ).fetchone()
            if row is None:
                raise RuntimeError("fct_player_seasons table returned empty telemetry.")
        finally:
            con.close()

        registry.warmup()
        q_reg = registry.get_quantile_regressor()
        c_pred = registry.get_conformal_predictor()
        a_idx = registry.get_archetype_indexer()

        models_meta = {
            "quantile_regressor_loaded": q_reg.is_fitted,
            "quantile_version": q_reg.model_version,
            "conformal_predictor_loaded": c_pred.is_calibrated,
            "conformal_version": c_pred.model_version,
            "archetype_index_loaded": a_idx.is_loaded,
            "archetype_count": len(a_idx.gallery),
        }
        return int(row[0]), int(row[1]), int(row[2]), int(row[3]), models_meta

    try:
        total, distinct, min_yr, max_yr, models_meta = await asyncio.to_thread(_run_diag)
    except Exception as exc:
        logger.error("API v2 health probe failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service dependency unavailable: {exc}",
        ) from exc

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return HealthStatusV2(
        status="healthy",
        database_connected=True,
        catalog_rows=total,
        distinct_players=distinct,
        min_year=min_yr,
        max_year=max_yr,
        models=models_meta,
        hardware_limits={"threads": 1, "memory_limit": "1GB"},
        response_time_ms=elapsed_ms,
    )


@router.get(
    "/players/search",
    response_model=SearchResponseV2,
    summary="Faceted Player Search (v2)",
)
async def search_players_v2(
    con: Annotated[duckdb.DuckDBPyConnection, Depends(get_db_session)],
    query: Annotated[str, Query(max_length=100)] = "",
    edition_year: Annotated[int | None, Query(ge=2015, le=2024)] = None,
    min_overall: Annotated[int | None, Query(ge=35, le=99)] = None,
    max_overall: Annotated[int | None, Query(ge=35, le=99)] = None,
    sector: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 25,
    offset: Annotated[int, Query(ge=0)] = 0,
    include_sector_counts: Annotated[bool, Query()] = True,
) -> SearchResponseV2:
    """Execute asynchronous faceted search across 180,021 modern seasonal records."""
    if sector is not None:
        valid_sectors = {s.lower() for s in SECTOR_ORDER}
        if sector.strip().lower() not in valid_sectors:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid tactical sector '{sector}'. Valid options: {SECTOR_ORDER}",
            )

    t0 = time.perf_counter()
    try:
        result = await asyncio.to_thread(
            _execute_modern_search,
            con,
            query,
            edition_year,
            min_overall,
            max_overall,
            sector,
            limit,
            offset,
            include_sector_counts,
        )
    except Exception as exc:
        logger.error("API v2 search error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query error: {exc}",
        ) from exc

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    result["inference_time_ms"] = elapsed_ms
    return SearchResponseV2.model_validate(result)


@router.post(
    "/trajectories/forecast",
    response_model=TrajectoryForecastResponseV2,
    summary="Quantile and Conformal Trajectory Forecast",
)
async def forecast_trajectory_v2(
    payload: TrajectoryForecastRequestV2,
    con: Annotated[duckdb.DuckDBPyConnection, Depends(get_db_session)],
    quantile_model: Annotated[QuantileTrajectoryRegressor, Depends(get_quantile_regressor)],
    conformal_model: Annotated[ConformalTrajectoryPredictor, Depends(get_conformal_predictor)],
) -> TrajectoryForecastResponseV2:
    """Forecast 1 to 3 year player rating trajectories with uncertainty bounds."""

    def _compute() -> TrajectoryForecastResponseV2 | None:
        record, growth_1y = _fetch_player_with_velocity(
            con,
            payload.player_id,
            payload.edition_year,
            include_skills=False,
        )
        if record is None:
            return None
        return _run_trajectory_inference(
            record=record,
            growth_1y=growth_1y,
            horizon_years=payload.horizon_years,
            include_conformal=payload.include_conformal,
            quantile_model=quantile_model,
            conformal_model=conformal_model if payload.include_conformal else None,
        )

    try:
        result = await asyncio.to_thread(_compute)
    except Exception as exc:
        logger.error("Trajectory forecast failed for player %s: %s", payload.player_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Trajectory inference error: {exc}",
        ) from exc

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Player ID {payload.player_id} not found for FIFA {payload.edition_year}.",
        )
    return result


@router.post(
    "/archetypes/match",
    response_model=ArchetypeMatchResponseV2,
    summary="High-Dimensional Archetype Vector Match",
)
async def match_archetype_v2(
    payload: ArchetypeMatchRequestV2,
    con: Annotated[duckdb.DuckDBPyConnection, Depends(get_db_session)],
    indexer: Annotated[ArchetypeVectorIndex, Depends(get_archetype_indexer)],
) -> ArchetypeMatchResponseV2:
    """Match a prospect against prime superstar archetypes via Gaussian RBF index."""

    def _compute() -> ArchetypeMatchResponseV2 | None:
        record, _ = _fetch_player_with_velocity(
            con,
            payload.player_id,
            payload.edition_year,
            include_skills=True,
        )
        if record is None:
            return None
        return _run_archetype_inference(
            record=record,
            top_k=payload.top_k,
            indexer=indexer,
        )

    try:
        result = await asyncio.to_thread(_compute)
    except Exception as exc:
        logger.error("Archetype matching failed for player %s: %s", payload.player_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Archetype matching error: {exc}",
        ) from exc

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Player ID {payload.player_id} not found for FIFA {payload.edition_year}.",
        )
    return result


@router.get(
    "/scout/{player_id}",
    response_model=ComprehensiveScoutResponseV2,
    summary="Comprehensive Scouting Dossier",
)
async def comprehensive_scout_v2(
    player_id: Annotated[int, PathParam(gt=0)],
    con: Annotated[duckdb.DuckDBPyConnection, Depends(get_db_session)],
    registry: Annotated[ModelRegistry, Depends(get_model_registry)],
    edition_year: Annotated[int, Query(ge=2015, le=2024)] = 2024,
    horizon_years: Annotated[int, Query(ge=1, le=3)] = 3,
    top_k: Annotated[int, Query(ge=1, le=10)] = 3,
) -> ComprehensiveScoutResponseV2:
    """Retrieve complete player profile, 3-year trajectory, and superstar matches."""

    def _compute() -> ComprehensiveScoutResponseV2 | None:
        t_db = time.perf_counter()
        record, growth_1y = _fetch_player_with_velocity(
            con,
            player_id,
            edition_year,
            include_skills=True,
        )
        if record is None:
            return None
        db_ms = round((time.perf_counter() - t_db) * 1000.0, 2)

        quantile_model = registry.get_quantile_regressor()
        conformal_model = registry.get_conformal_predictor()
        indexer = registry.get_archetype_indexer()

        profile_item = PlayerSearchItemV2.model_validate(record)
        traj_resp = _run_trajectory_inference(
            record=record,
            growth_1y=growth_1y,
            horizon_years=horizon_years,
            include_conformal=True,
            quantile_model=quantile_model,
            conformal_model=conformal_model,
        )
        arch_resp = _run_archetype_inference(
            record=record,
            top_k=top_k,
            indexer=indexer,
        )

        total_inf_ms = round(traj_resp.inference_time_ms + arch_resp.inference_time_ms, 2)
        return ComprehensiveScoutResponseV2(
            profile=profile_item,
            trajectory=traj_resp,
            archetypes=arch_resp,
            db_query_time_ms=db_ms,
            total_inference_time_ms=total_inf_ms,
        )

    try:
        result = await asyncio.to_thread(_compute)
    except Exception as exc:
        logger.error("Comprehensive scout failed for player %s: %s", player_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scouting dossier error: {exc}",
        ) from exc

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Player ID {player_id} not found for FIFA {edition_year}.",
        )
    return result
