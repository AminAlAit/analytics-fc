"""Automated quality test gate task for Prefect orchestration pipelines.

Executes pytest regression and contract test suites under single-process
hardware constraints, validating system integrity before production handoff.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from prefect import task

from pipelines.config import config

logger = logging.getLogger("analytics_fc.pipelines.tasks.test_runner")

BASE_DIR = config.base_dir
DEFAULT_TEST_TARGETS = [
    "tests/test_duckdb_contracts.py",
    "tests/test_dvc_contracts.py",
    "tests/test_gallery_integrity.py",
    "tests/test_catalog_search.py",
    "tests/test_ceiling_matcher.py",
]


def parse_pytest_summary(stdout: str) -> dict[str, int]:
    """Parse pytest output to extract passed, failed, and warning counts."""
    metrics = {"passed": 0, "failed": 0, "warnings": 0, "total": 0}

    match_passed = re.search(r"(\d+)\s+passed", stdout)
    if match_passed:
        metrics["passed"] = int(match_passed.group(1))

    match_failed = re.search(r"(\d+)\s+failed", stdout)
    if match_failed:
        metrics["failed"] = int(match_failed.group(1))

    match_warnings = re.search(r"(\d+)\s+warning", stdout)
    if match_warnings:
        metrics["warnings"] = int(match_warnings.group(1))

    metrics["total"] = metrics["passed"] + metrics["failed"]
    return metrics


def execute_test_suite(
    test_args: list[str] | None = None,
    cwd: Path | None = None,
) -> tuple[int, str, str, float]:
    """Execute pytest via virtual environment subprocess."""
    work_dir = cwd or BASE_DIR
    args = test_args if test_args is not None else DEFAULT_TEST_TARGETS
    cmd = [sys.executable, "-m", "pytest", "-q", *args]

    logger.info(f"Executing automated test gate: {' '.join(cmd)} in {work_dir}")
    t0 = time.perf_counter()

    proc = subprocess.run(
        cmd,
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    duration = round(time.perf_counter() - t0, 2)
    return proc.returncode, proc.stdout, proc.stderr, duration


@task(name="automated-quality-test-gate", retries=0)
def quality_gate_task(
    test_args: list[str] | None = None,
) -> dict[str, Any]:
    """Prefect task executing the automated quality test gate."""
    logger.info("Initiating Automated Quality Test Gate...")
    returncode, stdout, stderr, duration = execute_test_suite(test_args=test_args)
    parsed_metrics = parse_pytest_summary(stdout)

    result = {
        "gate": "automated-quality-test-gate",
        "exit_code": returncode,
        "status": "PASSED" if returncode == 0 else "FAILED",
        "duration_seconds": duration,
        "tests_passed": parsed_metrics["passed"],
        "tests_failed": parsed_metrics["failed"],
        "warnings": parsed_metrics["warnings"],
        "total_tests": parsed_metrics["total"],
    }

    if returncode != 0:
        logger.error(
            f"Quality test gate FAILED with exit code {returncode} in {duration}s: {parsed_metrics['failed']} failed."
        )
        raise RuntimeError(
            f"Quality test gate failed (exit code {returncode}): {parsed_metrics['failed']} failed tests."
        )

    logger.info(
        f"Quality test gate PASSED in {duration}s: {parsed_metrics['passed']} tests verified."
    )
    return result
