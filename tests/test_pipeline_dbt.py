"""Test suite for Prefect dbt transformation and test execution tasks.

Verifies programmatic dbt compilation, model execution, contract data tests,
output parsing, and end-to-end task integration within the master pipeline flow.
"""

from __future__ import annotations

from pipelines.flows import master_pipeline_flow
from pipelines.tasks.dbt_runner import (
    dbt_runner_task,
    execute_dbt_command,
    parse_dbt_summary,
)


def test_dbt_compile_command() -> None:
    """Assert dbt compile executes cleanly with single thread constraint."""
    code, stdout, stderr = execute_dbt_command(["compile"])

    assert code == 0, f"dbt compile failed: {stderr or stdout}"
    assert "models" in stdout
    assert "Concurrency: 1 threads" in stdout


def test_dbt_summary_parser() -> None:
    """Assert regex output parser correctly extracts model and test counts."""
    sample_output = "Done. PASS=6 WARN=0 ERROR=0 SKIP=0 NO-OP=0 REUSED=0 TOTAL=6"
    parsed = parse_dbt_summary(sample_output)

    assert parsed["pass"] == 6
    assert parsed["warn"] == 0
    assert parsed["error"] == 0
    assert parsed["total"] == 6


def test_dbt_runner_task_execution() -> None:
    """Assert dbt task compiles models and runs data tests under hardware caps."""
    result = dbt_runner_task.fn()

    assert result["status"] == "COMPLETED"
    assert result["models_passed"] >= 6
    assert result["tests_passed"] >= 4
    assert result["total_duration_seconds"] > 0


def test_master_pipeline_flow_with_dbt() -> None:
    """Assert master pipeline flow runs environment check, ingest, and dbt stages."""
    summary = master_pipeline_flow(dry_run=False)

    assert summary["pipeline"] == "analytics-fc-master-pipeline"
    assert summary["status"] == "COMPLETED"
    assert summary["dry_run"] is False

    dbt_summary = summary["dbt"]
    assert dbt_summary is not None
    assert dbt_summary["status"] == "COMPLETED"
    assert dbt_summary["models_passed"] >= 6
    assert dbt_summary["tests_passed"] >= 4
