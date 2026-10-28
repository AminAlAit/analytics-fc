"""Superstar archetype gallery generation task for Prefect orchestration pipelines.

Extracts deterministic personal peak seasons for top 100 historical players,
imputes composure values, and exports the verified gallery JSON artifact.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from prefect import task

from pipelines.config import config
from src.catalog_search import ALL_SKILL_COLUMNS, SECTOR_MAP

logger = logging.getLogger("analytics_fc.pipelines.tasks.gallery")

GALLERY_TARGET_COUNT = 100


def query_peak_player_seasons(
    db_path: Path | str,
    target_count: int = GALLERY_TARGET_COUNT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract top superstar career peak seasons and composure fallbacks from DuckDB."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        con.execute(f"SET threads = {config.duckdb_threads};")
        con.execute(f"SET memory_limit = '{config.duckdb_memory_limit}';")

        skill_selects = ", ".join([f"s.{col}" for col in ALL_SKILL_COLUMNS])

        query = f"""
        WITH player_career_summary AS (
            SELECT 
                player_id,
                MAX(overall) AS peak_overall,
                MAX(international_reputation) AS max_rep,
                COUNT(DISTINCT fifa_edition_year) AS total_career_seasons
            FROM stg_fifa_granular
            GROUP BY player_id
        ),
        
        peak_counts AS (
            SELECT 
                s.player_id,
                COUNT(*) AS seasons_at_peak
            FROM stg_fifa_granular s
            INNER JOIN player_career_summary p 
                ON s.player_id = p.player_id 
               AND s.overall = p.peak_overall
            GROUP BY s.player_id
        ),
        
        ranked_stars AS (
            SELECT 
                c.player_id, 
                c.peak_overall
            FROM player_career_summary c
            INNER JOIN peak_counts pc ON c.player_id = pc.player_id
            ORDER BY 
                c.peak_overall DESC, 
                pc.seasons_at_peak DESC, 
                c.max_rep DESC, 
                c.player_id ASC
            LIMIT {target_count}
        )
        
        SELECT 
            s.player_id,
            s.short_name,
            s.long_name,
            s.nationality,
            s.positions,
            s.age,
            s.fifa_edition_year,
            s.overall,
            s.height_cm,
            s.weight_kg,
            r.peak_overall,
            {skill_selects}
        FROM stg_fifa_granular s
        INNER JOIN ranked_stars r 
            ON s.player_id = r.player_id 
           AND s.overall = r.peak_overall
        ORDER BY r.peak_overall DESC, s.player_id, s.fifa_edition_year;
        """
        df = con.execute(query).df()

        composure_fallback_query = """
        SELECT player_id, AVG(mentality_composure) AS career_avg_composure
        FROM stg_fifa_granular
        WHERE mentality_composure IS NOT NULL
        GROUP BY player_id;
        """
        composure_df = con.execute(composure_fallback_query).df().set_index("player_id")
        return df, composure_df
    finally:
        con.close()


def generate_superstar_gallery(
    target_count: int = GALLERY_TARGET_COUNT,
    output_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Compile and export superstar archetypes with 34 granular tactical attributes."""
    dest_path = output_path or config.gallery_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    df, composure_df = query_peak_player_seasons(config.duckdb_path, target_count=target_count)

    gallery: list[dict[str, Any]] = []
    for player_id, group in df.groupby("player_id"):
        pos_list = [p.strip().upper() for p in str(group["positions"].iloc[0]).split(",")]
        primary_pos = pos_list[0]
        sector = SECTOR_MAP.get(primary_pos, "Attacking Midfielders & Wingers")

        attributes: dict[str, float] = {}
        for col in ALL_SKILL_COLUMNS:
            val_series = group[col].dropna()
            if not val_series.empty:
                attributes[col] = round(float(val_series.mean()), 2)
            elif col == "mentality_composure":
                if player_id in composure_df.index and pd.notna(composure_df.loc[player_id, "career_avg_composure"]):
                    attributes[col] = round(float(composure_df.loc[player_id, "career_avg_composure"]), 2)
                else:
                    reactions_val = group["movement_reactions"].dropna().mean()
                    attributes[col] = round(float(reactions_val if pd.notna(reactions_val) else 80.0), 2)
            else:
                attributes[col] = 50.0

        centroid = {
            "player_id": int(player_id),
            "short_name": group["short_name"].iloc[0],
            "long_name": group["long_name"].iloc[0],
            "nationality": group["nationality"].iloc[0],
            "primary_position": primary_pos,
            "positions": group["positions"].iloc[0],
            "tactical_sector": sector,
            "peak_overall": int(group["peak_overall"].iloc[0]),
            "peak_seasons_count": len(group),
            "peak_age_min": int(group["age"].min()),
            "peak_age_max": int(group["age"].max()),
            "peak_editions_observed": f"{group['fifa_edition_year'].min()}-{group['fifa_edition_year'].max()}",
            "height_cm": float(group["height_cm"].mean()),
            "weight_kg": float(group["weight_kg"].mean()),
            "attributes": attributes,
        }
        gallery.append(centroid)

    gallery.sort(key=lambda x: x["peak_overall"], reverse=True)

    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(gallery, f, indent=2)

    logger.info(f"Generated {len(gallery)} superstar archetypes saved to {dest_path}")
    return gallery


def verify_gallery_artifact(gallery_path: Path | None = None) -> dict[str, Any]:
    """Validate superstar gallery integrity contracts."""
    target_path = gallery_path or config.gallery_path
    if not target_path.exists():
        raise FileNotFoundError(f"Superstar gallery artifact missing at {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        gallery = json.load(f)

    if len(gallery) != GALLERY_TARGET_COUNT:
        raise AssertionError(
            f"Expected {GALLERY_TARGET_COUNT} superstars, found {len(gallery)}"
        )

    for star in gallery:
        assert "player_id" in star
        assert "short_name" in star
        assert "tactical_sector" in star
        assert "peak_overall" in star
        assert "attributes" in star

        attrs = star["attributes"]
        for col in ALL_SKILL_COLUMNS:
            assert col in attrs, f"Missing attribute {col} for {star['short_name']}"
            assert attrs[col] is not None, f"Null attribute {col} for {star['short_name']}"

    return {
        "status": "VALID",
        "superstar_count": len(gallery),
        "path": str(target_path),
    }


@task(name="generate-superstar-gallery", retries=1, retry_delay_seconds=2)
def superstar_gallery_task() -> dict[str, Any]:
    """Prefect task compiling and verifying superstar gallery archetypes."""
    t0 = time.perf_counter()
    logger.info("Starting superstar gallery generation task...")

    generate_superstar_gallery()
    verification = verify_gallery_artifact()
    duration = round(time.perf_counter() - t0, 2)

    return {
        "status": "COMPLETED",
        "archetype_count": verification["superstar_count"],
        "gallery_path": verification["path"],
        "duration_seconds": duration,
    }
