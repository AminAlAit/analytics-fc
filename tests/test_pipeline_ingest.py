"""Test suite for Prefect raw data ingestion and DVC validation tasks.

Verifies DVC pointer file integrity, physical data assets, DuckDB raw table
row count contracts, and end-to-end task integration within the master flow.
"""

from __future__ import annotations

from pipelines.flows import master_pipeline_flow
from pipelines.tasks.ingest import (
    EXPECTED_FIFA_GRANULAR_ROWS,
    EXPECTED_FIFATABLE_ROWS,
    ingest_raw_data_task,
    validate_dvc_raw_artifacts,
    verify_and_ingest_raw_tables,
)


def test_validate_dvc_raw_artifacts() -> None:
    """Assert DVC pointer file exists and references tracked raw directory."""
    dvc_info = validate_dvc_raw_artifacts()

    assert dvc_info["archive_file_exists"] is True
    assert dvc_info["md5_hash"].endswith(".dir")
    assert dvc_info["tracked_files"] >= 80
    assert dvc_info["total_size_bytes"] > 2_000_000_000


def test_verify_and_ingest_raw_tables() -> None:
    """Assert DuckDB raw schema contains verified row counts."""
    counts = verify_and_ingest_raw_tables()

    assert counts["fifatable"] == EXPECTED_FIFATABLE_ROWS
    assert counts["fifa_granular"] == EXPECTED_FIFA_GRANULAR_ROWS


def test_ingest_raw_data_task_execution() -> None:
    """Assert Prefect ingestion task completes with contract metadata."""
    result = ingest_raw_data_task.fn()

    assert result["status"] == "COMPLETED"
    assert result["dvc_hash"].endswith(".dir")
    assert result["tables"]["fifatable"] == EXPECTED_FIFATABLE_ROWS
    assert result["tables"]["fifa_granular"] == EXPECTED_FIFA_GRANULAR_ROWS
    assert result["duration_ms"] > 0


def test_master_pipeline_flow_with_ingestion() -> None:
    """Assert master pipeline flow runs environment check and ingestion stage."""
    summary = master_pipeline_flow(dry_run=False)

    assert summary["pipeline"] == "analytics-fc-master-pipeline"
    assert summary["status"] == "COMPLETED"
    assert summary["dry_run"] is False

    ingestion = summary["ingestion"]
    assert ingestion is not None
    assert ingestion["status"] == "COMPLETED"
    assert ingestion["tables"]["fifatable"] == EXPECTED_FIFATABLE_ROWS
    assert ingestion["tables"]["fifa_granular"] == EXPECTED_FIFA_GRANULAR_ROWS
