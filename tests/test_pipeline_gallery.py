"""Test suite for Prefect superstar archetype gallery generation tasks.

Verifies personal peak career query logic, composure imputation,
artifact schema contracts, and integration into the master pipeline flow.
"""

from __future__ import annotations

import json
from pathlib import Path

from pipelines.config import config
from pipelines.flows import master_pipeline_flow
from pipelines.tasks.gallery import (
    GALLERY_TARGET_COUNT,
    generate_superstar_gallery,
    query_peak_player_seasons,
    superstar_gallery_task,
    verify_gallery_artifact,
)
from src.catalog_search import ALL_SKILL_COLUMNS


def test_query_peak_player_seasons() -> None:
    """Assert SQL extraction retrieves exactly target count of unique superstars."""
    df, composure_df = query_peak_player_seasons(config.duckdb_path, target_count=GALLERY_TARGET_COUNT)

    assert df["player_id"].nunique() == GALLERY_TARGET_COUNT
    assert not composure_df.empty
    assert "player_id" in df.columns
    assert "peak_overall" in df.columns


def test_generate_and_verify_superstar_gallery(tmp_path: Path) -> None:
    """Assert gallery generation compiles 100 archetypes with 34 complete traits."""
    test_output = tmp_path / "test_superstar_gallery.json"
    gallery = generate_superstar_gallery(target_count=GALLERY_TARGET_COUNT, output_path=test_output)

    assert len(gallery) == GALLERY_TARGET_COUNT
    assert test_output.exists()

    verification = verify_gallery_artifact(test_output)
    assert verification["status"] == "VALID"
    assert verification["superstar_count"] == GALLERY_TARGET_COUNT

    # Validate first entry attributes
    top_star = gallery[0]
    assert top_star["peak_overall"] >= 90
    assert len(top_star["attributes"]) == len(ALL_SKILL_COLUMNS)
    for col in ALL_SKILL_COLUMNS:
        assert col in top_star["attributes"]
        assert top_star["attributes"][col] is not None


def test_superstar_gallery_task_execution() -> None:
    """Assert Prefect task runs and returns verified archetype metadata."""
    result = superstar_gallery_task.fn()

    assert result["status"] == "COMPLETED"
    assert result["archetype_count"] == GALLERY_TARGET_COUNT
    assert result["duration_seconds"] > 0


def test_master_pipeline_flow_with_gallery() -> None:
    """Assert master pipeline flow runs all stages through gallery compilation."""
    summary = master_pipeline_flow(dry_run=False)

    assert summary["pipeline"] == "analytics-fc-master-pipeline"
    assert summary["status"] == "COMPLETED"
    assert summary["dry_run"] is False

    gallery_summary = summary["gallery"]
    assert gallery_summary is not None
    assert gallery_summary["status"] == "COMPLETED"
    assert gallery_summary["archetype_count"] == GALLERY_TARGET_COUNT
