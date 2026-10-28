"""Programmatic dbt transformation and test execution tasks for Prefect flows.

Executes dbt models and schema data tests against the local DuckDB database
under single-thread hardware constraints, capturing structured telemetry.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from prefect import task

from pipelines.config import config

logger = logging.getLogger("analytics_fc.pipelines.tasks.dbt_runner")

DBT_PROJECT_DIR = config.dbt_project_dir
VENV_SCRIPTS_DIR = config.base_dir / ".venv" / "Scripts"
DBT_EXE = VENV_SCRIPTS_DIR / "dbt.exe"


def execute_dbt_command(
    args: list[str],
    project_dir: Path | None = None,
) -> tuple[int, str, str]:
    """Execute dbt CLI invocation using virtual environment executable."""
    target_dir = project_dir or DBT_PROJECT_DIR
    cmd = [
        str(DBT_EXE if DBT_EXE.exists() else "dbt"),
        *args,
        "--profiles-dir",
        ".",
        "--threads",
        str(config.duckdb_threads),
    ]

    logger.info(f"Executing dbt command: {' '.join(cmd)} in {target_dir}")
    proc = subprocess.run(
        cmd,
        cwd=str(target_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def parse_dbt_summary(output: str) -> dict[str, int]:
    """Extract pass, warn, and error counts from dbt output text."""
    counts = {"pass": 0, "warn": 0, "error": 0, "total": 0}
    match = re.search(r"PASS=(\d+)\s+WARN=(\d+)\s+ERROR=(\d+).+TOTAL=(\d+)", output)
    if match:
        counts["pass"] = int(match.group(1))
        counts["warn"] = int(match.group(2))
        counts["error"] = int(match.group(3))
        counts["total"] = int(match.group(4))
    return counts


def run_dbt_pipeline() -> dict[str, Any]:
    """Run full dbt transformation suite and data contract tests sequentially."""
    t0 = time.perf_counter()

    # Step 1: Execute dbt run
    run_t0 = time.perf_counter()
    run_code, run_stdout, run_stderr = execute_dbt_command(["run"])
    run_duration = round(time.perf_counter() - run_t0, 2)

    if run_code != 0:
        logger.error(f"dbt run failed with exit code {run_code}:\n{run_stderr or run_stdout}")
        raise RuntimeError(f"dbt run failed with exit code {run_code}")

    run_counts = parse_dbt_summary(run_stdout)

    # Step 2: Execute dbt test
    test_t0 = time.perf_counter()
    test_code, test_stdout, test_stderr = execute_dbt_command(["test"])
    test_duration = round(time.perf_counter() - test_t0, 2)

    if test_code != 0:
        logger.error(f"dbt test failed with exit code {test_code}:\n{test_stderr or test_stdout}")
        raise RuntimeError(f"dbt test failed with exit code {test_code}")

    test_counts = parse_dbt_summary(test_stdout)
    total_duration = round(time.perf_counter() - t0, 2)

    logger.info(
        f"dbt pipeline completed in {total_duration}s: "
        f"{run_counts['pass']} models built, {test_counts['pass']} tests passed."
    )

    return {
        "status": "COMPLETED",
        "models_passed": run_counts["pass"],
        "tests_passed": test_counts["pass"],
        "run_duration_seconds": run_duration,
        "test_duration_seconds": test_duration,
        "total_duration_seconds": total_duration,
    }


@task(name="run-dbt-transformations", retries=1, retry_delay_seconds=2)
def dbt_runner_task() -> dict[str, Any]:
    """Prefect task triggering dbt model builds and contract tests."""
    logger.info("Starting programmatic dbt transformation task...")
    return run_dbt_pipeline()
