"""Contract and execution test suite for Prefect orchestration flows.

Verifies configuration boundaries, hardware resource constraints, environment
topology checks, and ephemeral flow execution without persistent background daemons.
"""

from __future__ import annotations

from pipelines.config import config
from pipelines.flows import master_pipeline_flow, verify_pipeline_environment


def test_pipeline_config_hardware_safety_boundaries() -> None:
    """Assert hardware constraints and paths adhere to laptop safety standards."""
    assert config.duckdb_threads == 1
    assert config.duckdb_memory_limit == "1GB"
    assert config.base_dir.exists()
    assert config.data_raw_dir.exists()
    assert config.data_processed_dir.exists()
    assert config.dbt_project_dir.exists()
    assert config.duckdb_path.exists()


def test_verify_pipeline_environment_task_execution() -> None:
    """Assert environment verification task validates DuckDB and directory topology."""
    results = verify_pipeline_environment.fn()

    assert results["base_dir_exists"] is True
    assert results["data_raw_dir_exists"] is True
    assert results["duckdb_connected"] is True
    assert results["catalog_row_count"] >= 140000
    assert results["duckdb_threads_limit"] == 1
    assert results["duckdb_memory_limit"] == "1GB"


def test_master_pipeline_flow_ephemeral_run() -> None:
    """Assert master pipeline flow completes successfully with structured summary."""
    summary = master_pipeline_flow(dry_run=True)

    assert summary["pipeline"] == "analytics-fc-master-pipeline"
    assert summary["status"] == "COMPLETED"
    assert summary["dry_run"] is True
    assert summary["duration_seconds"] >= 0.0
    assert summary["environment"]["duckdb_connected"] is True
