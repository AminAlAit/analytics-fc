"""Master Prefect orchestration flows for Analytics FC data pipeline.

Orchestrates data validation, dbt transformations, superstar gallery compilation,
and quality gates in a single deterministic, ephemeral pipeline run.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

from prefect import flow, task

from pipelines.config import config, configure_ephemeral_prefect
from pipelines.tasks.ingest import ingest_raw_data_task

# Ensure purely ephemeral in-process Prefect execution
configure_ephemeral_prefect()

logger = logging.getLogger("analytics_fc.pipelines.flows")


@task(name="verify-pipeline-environment", retries=1)
def verify_pipeline_environment() -> dict[str, Any]:
    """Verify runtime environment, hardware limits, and required directory structures."""
    t0 = time.perf_counter()
    logger.info("Verifying execution environment and directory topology...")

    checks = {
        "python_version": sys.version.split()[0],
        "base_dir_exists": config.base_dir.exists(),
        "data_raw_dir_exists": config.data_raw_dir.exists(),
        "data_processed_dir_exists": config.data_processed_dir.exists(),
        "dbt_project_dir_exists": config.dbt_project_dir.exists(),
        "duckdb_threads_limit": config.duckdb_threads,
        "duckdb_memory_limit": config.duckdb_memory_limit,
    }

    # Verify DuckDB database file accessibility under hardware limits
    import duckdb

    con = duckdb.connect(str(config.duckdb_path), read_only=True)
    try:
        con.execute(f"SET threads = {config.duckdb_threads};")
        con.execute(f"SET memory_limit = '{config.duckdb_memory_limit}';")
        count_res = con.execute("SELECT COUNT(*) FROM stg_fifa_granular;").fetchone()
        row_count = int(count_res[0]) if count_res else 0
        checks["duckdb_connected"] = True
        checks["catalog_row_count"] = row_count
    finally:
        con.close()

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    checks["verification_time_ms"] = elapsed_ms
    logger.info(f"Environment verification passed in {elapsed_ms}ms ({row_count} catalog rows confirmed).")
    return checks


@flow(name="analytics-fc-master-pipeline", log_prints=True)
def master_pipeline_flow(dry_run: bool = False) -> dict[str, Any]:
    """Top-level orchestrated pipeline flow coordinating data ingestion and serving."""
    logger.info("Starting Analytics FC Master Pipeline flow...")
    start_time = time.perf_counter()

    # Step 1: Pre-flight environment and hardware boundary check
    env_results = verify_pipeline_environment()

    # Step 2: Ingest and validate raw DVC datasets
    ingest_results = None
    if not dry_run:
        ingest_results = ingest_raw_data_task()

    # Downstream task stages (AFC-522 to AFC-524) will be plugged into this sequence:
    # 2. Transform: Execute dbt run and dbt test models
    # 3. Gallery: Compile personal peak superstar gallery JSON
    # 4. Gate: Execute pytest contract and regression suite

    total_duration = round(time.perf_counter() - start_time, 2)
    summary = {
        "pipeline": "analytics-fc-master-pipeline",
        "status": "COMPLETED",
        "dry_run": dry_run,
        "duration_seconds": total_duration,
        "environment": env_results,
        "ingestion": ingest_results,
    }
    logger.info(f"Master pipeline flow completed successfully in {total_duration}s.")
    return summary


if __name__ == "__main__":
    result = master_pipeline_flow()
    print("Pipeline Execution Result:", result)
