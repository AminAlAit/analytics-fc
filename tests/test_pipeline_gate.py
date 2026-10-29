"""Unit and integration tests for the Automated Quality Test Gate task and flow."""

import pytest

from pipelines.flows import master_pipeline_flow
from pipelines.tasks.test_runner import (
    execute_test_suite,
    parse_pytest_summary,
    quality_gate_task,
)


def test_parse_pytest_summary_output():
    """Verify regex summary parser correctly extracts test counts."""
    sample_stdout = "60 passed, 1 warning in 12.34s"
    metrics = parse_pytest_summary(sample_stdout)
    assert metrics["passed"] == 60
    assert metrics["failed"] == 0
    assert metrics["warnings"] == 1
    assert metrics["total"] == 60

    sample_failure = "58 passed, 2 failed, 3 warnings in 8.50s"
    metrics_fail = parse_pytest_summary(sample_failure)
    assert metrics_fail["passed"] == 58
    assert metrics_fail["failed"] == 2
    assert metrics_fail["warnings"] == 3
    assert metrics_fail["total"] == 60


def test_execute_test_suite_runner():
    """Verify direct subprocess test suite execution returns exit code and timing."""
    returncode, stdout, stderr, duration = execute_test_suite(
        test_args=["tests/test_duckdb_contracts.py"]
    )
    assert returncode == 0
    assert duration > 0.0
    assert "passed" in stdout


def test_quality_gate_task_execution():
    """Verify Prefect task execution passes against data contract suite."""
    result = quality_gate_task.fn(test_args=["tests/test_duckdb_contracts.py"])
    assert result["status"] == "PASSED"
    assert result["exit_code"] == 0
    assert result["tests_passed"] >= 4
    assert result["tests_failed"] == 0
    assert result["duration_seconds"] > 0.0


def test_quality_gate_task_failure_behavior():
    """Verify quality gate raises RuntimeError when test suite fails."""
    with pytest.raises(RuntimeError) as exc_info:
        quality_gate_task.fn(test_args=["tests/nonexistent_test_target_failure.py"])
    assert "Quality test gate failed" in str(exc_info.value)


def test_master_pipeline_flow_dry_run():
    """Verify master pipeline flow supports dry_run mode with quality gate telemetry."""
    summary = master_pipeline_flow(dry_run=True)
    assert summary["pipeline"] == "analytics-fc-master-pipeline"
    assert summary["status"] == "COMPLETED"
    assert summary["dry_run"] is True
    assert summary["environment"]["duckdb_connected"] is True
    assert summary["quality_gate"] is None
