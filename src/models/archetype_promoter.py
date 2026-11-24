"""Automated Archetype Discovery & Promotion Pipeline (Model 1 lifecycle).

Queries historical and modern player season marts under strict single-threaded,
1GB RAM hardware limits to qualify emerging prime superstars, evaluate
tactical vector novelty via Gaussian RBF distance, export versioned catalog
artifacts, and execute atomic zero-downtime swaps on ModelRegistry and
AtomicArtifactSlot holders.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import numpy as np
import pandas as pd
import psutil
from pydantic import BaseModel, ConfigDict, Field

from src.models.archetype_indexer import (
    GK_SKILL_COLS,
    OUTFIELD_SKILL_COLS,
    SECTOR_MAP,
    ArchetypeVectorIndex,
)

if TYPE_CHECKING:
    from src.api.cache_manager import AtomicArtifactSlot
    from src.api.v2.dependencies import ModelRegistry

logger = logging.getLogger("analytics_fc.models.archetype_promoter")

ALL_SKILL_COLS: list[str] = [*OUTFIELD_SKILL_COLS, *GK_SKILL_COLS]

MACRO_TO_GRANULAR_GROUPS: dict[str, tuple[str, ...]] = {
    "pace": ("movement_acceleration", "movement_sprint_speed"),
    "shooting": (
        "attacking_finishing",
        "attacking_volleys",
        "power_shot_power",
        "power_long_shots",
        "mentality_penalties",
        "mentality_positioning",
    ),
    "passing": (
        "attacking_crossing",
        "attacking_short_passing",
        "skill_curve",
        "skill_fk_accuracy",
        "skill_long_passing",
        "mentality_vision",
    ),
    "dribbling": (
        "skill_dribbling",
        "skill_ball_control",
        "movement_agility",
        "movement_reactions",
        "movement_balance",
        "mentality_composure",
    ),
    "defending": (
        "defending_marking_awareness",
        "defending_standing_tackle",
        "defending_sliding_tackle",
        "mentality_interceptions",
        "attacking_heading_accuracy",
    ),
    "physic": (
        "power_jumping",
        "power_stamina",
        "power_strength",
        "mentality_aggression",
    ),
}

_GRANULAR_TO_MACRO: dict[str, str] = {
    skill: macro for macro, skills in MACRO_TO_GRANULAR_GROUPS.items() for skill in skills
}


class PromotedArchetype(BaseModel):
    """Canonical schema matching data/processed/superstar_gallery.json entries."""

    model_config = ConfigDict(extra="ignore")

    player_id: int = Field(gt=0)
    short_name: str
    long_name: str
    nationality: str
    primary_position: str
    positions: str
    tactical_sector: str
    peak_overall: int = Field(ge=35, le=99)
    peak_seasons_count: int = Field(ge=1)
    peak_age_min: int = Field(ge=14, le=55)
    peak_age_max: int = Field(ge=14, le=55)
    peak_editions_observed: str
    height_cm: float
    weight_kg: float
    attributes: dict[str, float]


class NoveltyEvaluation(BaseModel):
    """Vector novelty evaluation result against existing ArchetypeVectorIndex."""

    model_config = ConfigDict(frozen=True)

    player_id: int
    short_name: str
    tactical_sector: str
    is_novel: bool
    reason: str
    min_rbf_distance: float
    max_similarity_pct: float


class PromotionRunResult(BaseModel):
    """Structured telemetry summary of an end-to-end promotion run."""

    model_config = ConfigDict(frozen=True)

    candidates_evaluated: int
    newly_promoted_count: int
    skipped_existing_count: int
    skipped_redundant_count: int
    total_gallery_size: int
    promoted_player_ids: list[int]
    promoted_names: list[str]
    json_artifact_path: str
    parquet_artifact_path: str | None = None
    index_version_after_swap: int | None = None
    execution_time_ms: float


def apply_promoter_hardware_limits(max_memory_mb: float = 1024.0) -> dict[str, Any]:
    """Enforce below-normal OS priority, single-thread env vars, and check RSS cap."""
    for env_key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "DUCKDB_THREADS",
    ):
        os.environ[env_key] = "1"

    proc = psutil.Process()
    priority_applied = "unchanged"
    try:
        below_normal = getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", None)
        if os.name == "nt" and below_normal is not None:
            proc.nice(below_normal)
            priority_applied = "BELOW_NORMAL_PRIORITY_CLASS"
        else:
            proc.nice(10)
            priority_applied = "nice_10"
    except Exception as exc:
        logger.debug("Skipping OS priority adjustment: %s", exc)

    rss_mb = round(proc.memory_info().rss / (1024.0 * 1024.0), 2)
    if rss_mb > max_memory_mb:
        raise MemoryError(
            f"Process RSS {rss_mb:.1f} MB exceeds promoter 1GB cap ({max_memory_mb:.1f} MB)."
        )

    return {
        "pid": proc.pid,
        "priority_applied": priority_applied,
        "rss_mb": rss_mb,
        "max_memory_mb": max_memory_mb,
    }


def _clean_str(val: Any) -> str | None:
    """Return stripped non-empty string or None when value is null or NaN."""
    if val is None:
        return None
    try:
        if bool(pd.isna(val)):
            return None
    except TypeError:
        pass
    cleaned = str(val).strip()
    if not cleaned or cleaned.lower() in {"nan", "none"}:
        return None
    return cleaned


def _first_valid_str(*values: Any, fallback: str) -> str:
    """Return the first non-null non-empty string value, else fallback."""
    for candidate in values:
        cleaned = _clean_str(candidate)
        if cleaned is not None:
            return cleaned
    return fallback


def _safe_attr_val(attrs: dict[str, Any] | None, col: str, default: float = 50.0) -> float:
    """Safely convert a skill attribute dict entry into a finite float."""
    if not isinstance(attrs, dict):
        return default
    raw = attrs.get(col)
    if raw is None:
        return default
    try:
        val = float(raw)
        if np.isnan(val) or np.isinf(val):
            return default
        return val
    except (TypeError, ValueError):
        return default


def _build_empty_index(sigma: float = 0.28) -> ArchetypeVectorIndex:
    """Construct an empty ArchetypeVectorIndex without touching disk defaults."""
    empty_idx = ArchetypeVectorIndex.__new__(ArchetypeVectorIndex)
    empty_idx.sigma = sigma
    empty_idx.gallery = []
    empty_idx.outfield_metadata = []
    empty_idx.gk_metadata = []
    empty_idx.outfield_matrix = np.empty((0, len(OUTFIELD_SKILL_COLS)), dtype=float)
    empty_idx.gk_matrix = np.empty((0, len(GK_SKILL_COLS)), dtype=float)
    empty_idx._is_loaded = False
    return empty_idx


def _extract_primary_position(positions_val: Any) -> str:
    """Return normalized primary tactical position token."""
    raw = _clean_str(positions_val)
    if not raw:
        return "CM"
    first_token = raw.split(",")[0].strip().upper()
    return first_token if first_token else "CM"


def _longest_consecutive_run(years: list[int]) -> list[int]:
    """Return the longest sorted consecutive integer year sequence."""
    unique_years = sorted(set(years))
    if not unique_years:
        return []
    best_run: list[int] = [unique_years[0]]
    current_run: list[int] = [unique_years[0]]
    for yr in unique_years[1:]:
        if yr == current_run[-1] + 1:
            current_run.append(yr)
        else:
            current_run = [yr]
        if len(current_run) >= len(best_run):
            best_run = list(current_run)
    return best_run


def _synthesize_attributes(
    peak_macro_rows: list[dict[str, Any]],
    granular_rows_desc: list[dict[str, Any]],
    peak_years: set[int],
    primary_pos: str,
    overall_val: float,
) -> dict[str, float]:
    """Build complete 34-skill dictionary from prime rows and historical fallback."""
    is_gk = primary_pos == "GK"
    attrs: dict[str, float] = {}

    macro_means: dict[str, float] = {}
    for macro_col in MACRO_TO_GRANULAR_GROUPS:
        vals = [
            float(r[macro_col])
            for r in peak_macro_rows
            if r.get(macro_col) is not None and not pd.isna(r[macro_col])
        ]
        if vals:
            macro_means[macro_col] = float(np.mean(vals))

    exact_gran = [
        r
        for r in granular_rows_desc
        if r.get("fifa_edition_year") is not None
        and not pd.isna(r.get("fifa_edition_year"))
        and int(r["fifa_edition_year"]) in peak_years
    ]
    source_gran = exact_gran if exact_gran else granular_rows_desc[:1]

    for skill in ALL_SKILL_COLS:
        gran_vals = [
            float(r[skill])
            for r in source_gran
            if r.get(skill) is not None and not pd.isna(r[skill])
        ]
        if not gran_vals and granular_rows_desc:
            gran_vals = [
                float(r[skill])
                for r in granular_rows_desc[:1]
                if r.get(skill) is not None and not pd.isna(r[skill])
            ]

        if gran_vals:
            val = float(np.mean(gran_vals))
        elif skill.startswith("goalkeeping_"):
            val = float(overall_val) if is_gk else 11.0
        else:
            macro_key = _GRANULAR_TO_MACRO.get(skill)
            if macro_key and macro_key in macro_means:
                val = macro_means[macro_key]
            else:
                val = 45.0 if is_gk else round(min(95.0, max(50.0, overall_val - 8.0)), 2)

        attrs[skill] = round(float(np.clip(val, 1.0, 99.0)), 2)
    return attrs


def _table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    """Return True if main.<table_name> exists in the DuckDB catalog."""
    row = con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = ?;
        """,
        [table_name],
    ).fetchone()
    return bool(row is not None and row[0] > 0)


def _fetch_granular_by_player(
    con: duckdb.DuckDBPyConnection,
    player_ids: list[int],
    has_granular: bool,
) -> dict[int, list[dict[str, Any]]]:
    """Fetch granular skill history only for qualified candidate IDs."""
    if not has_granular or not player_ids:
        return {}
    placeholders = ", ".join(["?"] * len(player_ids))
    skill_select = ", ".join(ALL_SKILL_COLS)
    sql = f"""
        SELECT player_id, fifa_edition_year, {skill_select}
        FROM main.stg_fifa_granular
        WHERE player_id IN ({placeholders})
          AND player_id IS NOT NULL
          AND fifa_edition_year IS NOT NULL
        ORDER BY player_id, fifa_edition_year DESC;
    """
    cursor = con.execute(sql, player_ids)
    cols = [d[0] for d in cursor.description]
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in cursor.fetchall():
        rec = dict(zip(cols, row, strict=True))
        grouped.setdefault(int(rec["player_id"]), []).append(rec)
    return grouped


def _fetch_dim_metadata(
    con: duckdb.DuckDBPyConnection,
    player_ids: list[int],
    has_dim: bool,
) -> dict[int, dict[str, Any]]:
    """Fetch player demographic metadata only for qualified candidate IDs."""
    if not has_dim or not player_ids:
        return {}
    placeholders = ", ".join(["?"] * len(player_ids))
    sql = f"""
        SELECT player_id, short_name, long_name, nationality, height_cm, weight_kg
        FROM main.dim_players
        WHERE player_id IN ({placeholders});
    """
    cursor = con.execute(sql, player_ids)
    cols = [d[0] for d in cursor.description]
    return {int(row[0]): dict(zip(cols, row, strict=True)) for row in cursor.fetchall()}


def discover_prime_candidates(
    db_path: Path | str = Path("dbt_project/analytics_fc.duckdb"),
    min_age: int = 24,
    max_age: int = 29,
    min_overall: int = 88,
    min_consecutive_seasons: int = 2,
    sector_percentile: float = 0.98,
    min_edition_year: int | None = None,
) -> list[dict[str, Any]]:
    """Query fct_player_seasons and dim_players for qualified prime archetypes."""
    database_path = Path(db_path)
    if not database_path.exists():
        raise FileNotFoundError(f"DuckDB database not found at {database_path}")

    con = duckdb.connect(str(database_path), read_only=True)
    try:
        con.execute("SET threads = 1;")
        con.execute("SET memory_limit = '1GB';")

        has_fct = _table_exists(con, "fct_player_seasons")
        has_dim = _table_exists(con, "dim_players")
        has_granular = _table_exists(con, "stg_fifa_granular")
        if not has_fct:
            raise RuntimeError("Required table main.fct_player_seasons not found in DuckDB.")

        year_clause = "AND fifa_edition_year >= ?" if min_edition_year is not None else ""
        sql = f"""
            SELECT
                player_id,
                fifa_edition_year,
                short_name,
                long_name,
                nationality,
                height_cm,
                weight_kg,
                age,
                positions,
                overall,
                pace,
                shooting,
                passing,
                dribbling,
                defending,
                physic
            FROM main.fct_player_seasons
            WHERE age BETWEEN ? AND ?
              AND player_id IS NOT NULL
              AND fifa_edition_year IS NOT NULL
              AND overall IS NOT NULL
            {year_clause}
            ORDER BY player_id, fifa_edition_year;
        """
        params: list[Any] = [int(min_age), int(max_age)]
        if min_edition_year is not None:
            params.append(int(min_edition_year))

        prime_df = con.execute(sql, params).df()
        if prime_df.empty:
            return []

        prime_df["primary_position"] = prime_df["positions"].map(_extract_primary_position)
        prime_df["tactical_sector"] = (
            prime_df["primary_position"]
            .map(lambda p: SECTOR_MAP.get(p, "Attacking Midfielders & Wingers"))
            .astype(str)
        )

        sector_cutoffs = (
            prime_df.groupby("tactical_sector")["overall"].quantile(sector_percentile).to_dict()
        )
        sector_thresh = prime_df["tactical_sector"].map(sector_cutoffs).fillna(float(min_overall))
        qualified_df = prime_df[
            (prime_df["overall"] >= float(min_overall)) | (prime_df["overall"] >= sector_thresh)
        ]
        if qualified_df.empty:
            return []

        # Filter consecutive career seasons before reading granular skills or dim_players
        rows_by_pid: dict[int, list[dict[str, Any]]] = {}
        for rec in qualified_df.to_dict("records"):
            rows_by_pid.setdefault(int(rec["player_id"]), []).append(rec)

        qualified_runs: dict[int, tuple[list[int], list[dict[str, Any]]]] = {}
        for pid_int, recs in rows_by_pid.items():
            years_run = _longest_consecutive_run([int(r["fifa_edition_year"]) for r in recs])
            if len(years_run) >= int(min_consecutive_seasons):
                run_set = set(years_run)
                run_recs = [r for r in recs if int(r["fifa_edition_year"]) in run_set]
                qualified_runs[pid_int] = (years_run, run_recs)

        if not qualified_runs:
            return []

        candidate_ids = sorted(qualified_runs.keys())
        gran_by_pid = _fetch_granular_by_player(con, candidate_ids, has_granular)
        dim_by_pid = _fetch_dim_metadata(con, candidate_ids, has_dim)
    finally:
        con.close()

    candidates: list[dict[str, Any]] = []
    for pid_int, (years_run, run_recs) in qualified_runs.items():
        peak_ovr = max(int(r["overall"]) for r in run_recs)
        peak_rows = [r for r in run_recs if int(r["overall"]) == peak_ovr]
        latest_peak = peak_rows[-1]
        dim_meta = dim_by_pid.get(pid_int, {})

        primary_pos = str(latest_peak["primary_position"])
        sector = str(latest_peak["tactical_sector"])
        peak_years = {int(r["fifa_edition_year"]) for r in peak_rows}

        attrs = _synthesize_attributes(
            peak_macro_rows=peak_rows,
            granular_rows_desc=gran_by_pid.get(pid_int, []),
            peak_years=peak_years,
            primary_pos=primary_pos,
            overall_val=float(peak_ovr),
        )

        heights = [
            float(r["height_cm"])
            for r in run_recs
            if r.get("height_cm") is not None and not pd.isna(r["height_cm"])
        ]
        weights = [
            float(r["weight_kg"])
            for r in run_recs
            if r.get("weight_kg") is not None and not pd.isna(r["weight_kg"])
        ]
        default_h = dim_meta.get("height_cm")
        default_w = dim_meta.get("weight_kg")
        height_cm = (
            float(np.mean(heights))
            if heights
            else (float(default_h) if default_h is not None and not pd.isna(default_h) else 180.0)
        )
        weight_kg = (
            float(np.mean(weights))
            if weights
            else (float(default_w) if default_w is not None and not pd.isna(default_w) else 75.0)
        )

        short_name = _first_valid_str(
            latest_peak.get("short_name"),
            dim_meta.get("short_name"),
            fallback=str(pid_int),
        )
        long_name = _first_valid_str(
            latest_peak.get("long_name"),
            dim_meta.get("long_name"),
            fallback=short_name,
        )
        nationality = _first_valid_str(
            latest_peak.get("nationality"),
            dim_meta.get("nationality"),
            fallback="Unknown",
        )
        positions_str = _first_valid_str(
            latest_peak.get("positions"),
            fallback=primary_pos,
        )

        entry = PromotedArchetype(
            player_id=pid_int,
            short_name=short_name,
            long_name=long_name,
            nationality=nationality,
            primary_position=primary_pos,
            positions=positions_str,
            tactical_sector=sector,
            peak_overall=peak_ovr,
            peak_seasons_count=len(run_recs),
            peak_age_min=min(int(r["age"]) for r in run_recs),
            peak_age_max=max(int(r["age"]) for r in run_recs),
            peak_editions_observed=f"{min(years_run)}-{max(years_run)}",
            height_cm=round(height_cm, 2),
            weight_kg=round(weight_kg, 2),
            attributes=attrs,
        )
        candidates.append(entry.model_dump())

    candidates.sort(key=lambda item: (-item["peak_overall"], item["player_id"]))
    return candidates


def evaluate_vector_novelty(
    candidate: dict[str, Any],
    index: ArchetypeVectorIndex,
    min_rbf_distance: float = 0.03,
    max_similarity_pct: float = 98.5,
) -> NoveltyEvaluation:
    """Validate whether a candidate adds novel tactical variety to the index."""
    pid = int(candidate["player_id"])
    short_name = str(candidate["short_name"])
    primary_pos = str(candidate.get("primary_position", "CM")).strip().upper()
    sector = str(
        candidate.get(
            "tactical_sector",
            SECTOR_MAP.get(primary_pos, "Attacking Midfielders & Wingers"),
        )
    )

    existing_ids = {int(star["player_id"]) for star in index.gallery}
    if pid in existing_ids:
        return NoveltyEvaluation(
            player_id=pid,
            short_name=short_name,
            tactical_sector=sector,
            is_novel=False,
            reason="already_in_gallery",
            min_rbf_distance=0.0,
            max_similarity_pct=100.0,
        )

    is_gk = sector == "Goalkeepers" or primary_pos == "GK"
    active_cols = GK_SKILL_COLS if is_gk else OUTFIELD_SKILL_COLS
    target_matrix = index.gk_matrix if is_gk else index.outfield_matrix

    if target_matrix.shape[0] == 0:
        return NoveltyEvaluation(
            player_id=pid,
            short_name=short_name,
            tactical_sector=sector,
            is_novel=True,
            reason="first_in_domain",
            min_rbf_distance=1.0,
            max_similarity_pct=0.0,
        )

    attrs = candidate.get("attributes") if isinstance(candidate.get("attributes"), dict) else {}
    raw_vec = np.array([_safe_attr_val(attrs, c, 50.0) for c in active_cols], dtype=float)
    norm = float(np.linalg.norm(raw_vec))
    unit_vec = raw_vec / norm if norm > 0.0 else raw_vec

    distances = np.linalg.norm(target_matrix - unit_vec, axis=1)
    min_dist = float(np.min(distances))
    similarities = index.calculate_rbf_similarity(np.array([min_dist], dtype=float))
    max_sim = round(float(similarities[0]), 2)

    is_novel = bool(min_dist >= min_rbf_distance and max_sim <= max_similarity_pct)
    reason = "novel_archetype" if is_novel else "redundant_vector"

    return NoveltyEvaluation(
        player_id=pid,
        short_name=short_name,
        tactical_sector=sector,
        is_novel=is_novel,
        reason=reason,
        min_rbf_distance=round(min_dist, 6),
        max_similarity_pct=max_sim,
    )


def export_versioned_gallery(
    gallery: list[dict[str, Any]],
    output_dir: Path | str = Path("data/processed"),
    version_tag: str = "v2",
    write_parquet: bool = True,
) -> dict[str, Path]:
    """Atomically write versioned superstar gallery JSON and Parquet catalog artifacts."""
    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    json_path = dest_dir / f"superstar_gallery_{version_tag}.json"
    pq_path = dest_dir / f"superstar_gallery_{version_tag}.parquet"

    fd, tmp_json_str = tempfile.mkstemp(dir=str(dest_dir), suffix=".json.tmp")
    os.close(fd)
    tmp_json_path = Path(tmp_json_str)
    tmp_pq_path: Path | None = None

    try:
        tmp_json_path.write_text(json.dumps(gallery, indent=2), encoding="utf-8")

        if write_parquet:
            base_cols = [
                "player_id",
                "short_name",
                "long_name",
                "nationality",
                "primary_position",
                "positions",
                "tactical_sector",
                "peak_overall",
                "peak_seasons_count",
                "peak_age_min",
                "peak_age_max",
                "peak_editions_observed",
                "height_cm",
                "weight_kg",
            ]
            flat_rows: list[dict[str, Any]] = []
            for star in gallery:
                row = {k: v for k, v in star.items() if k != "attributes"}
                attrs = star.get("attributes") if isinstance(star.get("attributes"), dict) else {}
                for col in ALL_SKILL_COLS:
                    row[col] = _safe_attr_val(attrs, col, 50.0)
                flat_rows.append(row)

            df_flat = (
                pd.DataFrame(flat_rows)
                if flat_rows
                else pd.DataFrame(columns=[*base_cols, *ALL_SKILL_COLS])
            )
            fd_pq, tmp_pq_str = tempfile.mkstemp(dir=str(dest_dir), suffix=".parquet.tmp")
            os.close(fd_pq)
            tmp_pq_path = Path(tmp_pq_str)

            con = duckdb.connect(database=":memory:")
            try:
                con.execute("SET threads = 1;")
                con.execute("SET memory_limit = '1GB';")
                con.register("gallery_df", df_flat)
                safe_tmp = str(tmp_pq_path).replace("'", "''")
                con.execute(f"COPY gallery_df TO '{safe_tmp}' (FORMAT PARQUET);")
            finally:
                con.close()

        os.replace(tmp_json_path, json_path)
        artifacts: dict[str, Path] = {"json": json_path}
        if write_parquet and tmp_pq_path is not None:
            os.replace(tmp_pq_path, pq_path)
            artifacts["parquet"] = pq_path
        return artifacts
    except Exception:
        if tmp_json_path.exists():
            tmp_json_path.unlink()
        if tmp_pq_path is not None and tmp_pq_path.exists():
            tmp_pq_path.unlink()
        raise


def atomic_reload_archetype_index(
    gallery_path: Path | str,
    slot: AtomicArtifactSlot[ArchetypeVectorIndex] | None = None,
    registry: ModelRegistry | None = None,
    sigma: float = 0.28,
) -> tuple[ArchetypeVectorIndex, int | None]:
    """Build a fresh ArchetypeVectorIndex and atomically swap API holders."""
    resolved_path = Path(gallery_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Superstar gallery file not found at {resolved_path}")

    raw_gallery = json.loads(resolved_path.read_text(encoding="utf-8"))
    if raw_gallery:
        fresh_index = ArchetypeVectorIndex(gallery_path=resolved_path, sigma=sigma)
    else:
        fresh_index = _build_empty_index(sigma=sigma)

    swapped_version: int | None = None
    if slot is not None:
        swapped_version = slot.swap(fresh_index)

    if registry is not None:
        lock = getattr(registry, "_lock", None)
        if lock is not None:
            with lock:
                registry._archetype_indexer = fresh_index
                registry.gallery_path = resolved_path
        else:
            registry._archetype_indexer = fresh_index
            registry.gallery_path = resolved_path

    return fresh_index, swapped_version


def run_archetype_promotion_pipeline(
    db_path: Path | str = Path("dbt_project/analytics_fc.duckdb"),
    base_gallery_path: Path | str = Path("data/processed/superstar_gallery.json"),
    output_dir: Path | str = Path("data/processed"),
    version_tag: str = "v2",
    min_age: int = 24,
    max_age: int = 29,
    min_overall: int = 88,
    min_consecutive_seasons: int = 2,
    sector_percentile: float = 0.98,
    min_edition_year: int | None = None,
    min_rbf_distance: float = 0.03,
    max_similarity_pct: float = 98.5,
    max_promotions: int | None = None,
    write_parquet: bool = True,
    slot: AtomicArtifactSlot[ArchetypeVectorIndex] | None = None,
    registry: ModelRegistry | None = None,
) -> PromotionRunResult:
    """Execute full archetype discovery, RBF novelty gating, export, and atomic reload."""
    t0 = time.perf_counter()
    apply_promoter_hardware_limits(max_memory_mb=1024.0)

    gallery_file = Path(base_gallery_path)
    base_gallery: list[dict[str, Any]] = (
        json.loads(gallery_file.read_text(encoding="utf-8")) if gallery_file.exists() else []
    )
    if base_gallery:
        working_index = ArchetypeVectorIndex(gallery_path=gallery_file)
    else:
        working_index = _build_empty_index(sigma=0.28)

    candidates = discover_prime_candidates(
        db_path=db_path,
        min_age=min_age,
        max_age=max_age,
        min_overall=min_overall,
        min_consecutive_seasons=min_consecutive_seasons,
        sector_percentile=sector_percentile,
        min_edition_year=min_edition_year,
    )

    merged_gallery = list(base_gallery)
    promoted_ids: list[int] = []
    promoted_names: list[str] = []
    skipped_existing = 0
    skipped_redundant = 0

    for cand in candidates:
        if max_promotions is not None and len(promoted_ids) >= max_promotions:
            break

        eval_res = evaluate_vector_novelty(
            candidate=cand,
            index=working_index,
            min_rbf_distance=min_rbf_distance,
            max_similarity_pct=max_similarity_pct,
        )
        if eval_res.is_novel:
            merged_gallery.append(cand)
            promoted_ids.append(int(cand["player_id"]))
            promoted_names.append(str(cand["short_name"]))
            # Register promoted vector in working_index to prevent intra-batch near-duplicates
            working_index.gallery.append(cand)
            p_pos = str(cand.get("primary_position", "CM")).strip().upper()
            is_gk = cand.get("tactical_sector") == "Goalkeepers" or p_pos == "GK"
            active_cols = GK_SKILL_COLS if is_gk else OUTFIELD_SKILL_COLS
            cand_attrs = cand.get("attributes") if isinstance(cand.get("attributes"), dict) else {}
            raw_v = np.array(
                [_safe_attr_val(cand_attrs, c, 50.0) for c in active_cols], dtype=float
            )
            norm_v = float(np.linalg.norm(raw_v))
            unit_v = (raw_v / norm_v if norm_v > 0.0 else raw_v).reshape(1, -1)
            if is_gk:
                working_index.gk_matrix = (
                    np.vstack([working_index.gk_matrix, unit_v])
                    if working_index.gk_matrix.shape[0] > 0
                    else unit_v
                )
            else:
                working_index.outfield_matrix = (
                    np.vstack([working_index.outfield_matrix, unit_v])
                    if working_index.outfield_matrix.shape[0] > 0
                    else unit_v
                )
        elif eval_res.reason == "already_in_gallery":
            skipped_existing += 1
        else:
            skipped_redundant += 1

    artifacts = export_versioned_gallery(
        gallery=merged_gallery,
        output_dir=output_dir,
        version_tag=version_tag,
        write_parquet=write_parquet,
    )

    swapped_version: int | None = None
    if slot is not None or registry is not None:
        _, swapped_version = atomic_reload_archetype_index(
            gallery_path=artifacts["json"],
            slot=slot,
            registry=registry,
        )

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return PromotionRunResult(
        candidates_evaluated=len(candidates),
        newly_promoted_count=len(promoted_ids),
        skipped_existing_count=skipped_existing,
        skipped_redundant_count=skipped_redundant,
        total_gallery_size=len(merged_gallery),
        promoted_player_ids=promoted_ids,
        promoted_names=promoted_names,
        json_artifact_path=str(artifacts["json"]),
        parquet_artifact_path=str(artifacts["parquet"]) if "parquet" in artifacts else None,
        index_version_after_swap=swapped_version,
        execution_time_ms=elapsed_ms,
    )
