"""Data ingestion and DVC integrity verification tasks for Analytics FC pipelines.

Validates DVC pointer hashes, confirms physical presence of tracked datasets,
and verifies DuckDB bronze raw table records under single thread constraints.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import duckdb
import yaml
from prefect import task

from pipelines.config import config

logger = logging.getLogger("analytics_fc.pipelines.tasks.ingest")

EXPECTED_FIFATABLE_ROWS = 43199
EXPECTED_FIFA_GRANULAR_ROWS = 142079
RAW_DVC_PATH = config.base_dir / "data" / "raw.dvc"


def validate_dvc_raw_artifacts(dvc_path: Path | None = None) -> dict[str, Any]:
    """Validate DVC pointer file integrity and underlying filesystem assets."""
    target_dvc = dvc_path or RAW_DVC_PATH
    if not target_dvc.exists():
        raise FileNotFoundError(f"Missing raw DVC pointer file at {target_dvc}")

    with open(target_dvc, encoding="utf-8") as f:
        dvc_meta = yaml.safe_load(f)

    if "outs" not in dvc_meta or not dvc_meta["outs"]:
        raise ValueError("Invalid DVC pointer structure: missing 'outs' section")

    out = dvc_meta["outs"][0]
    md5_hash = out.get("md5", "")
    nfiles = out.get("nfiles", 0)
    size_bytes = out.get("size", 0)

    if not md5_hash or not md5_hash.endswith(".dir"):
        raise ValueError(f"Unexpected DVC directory hash format: {md5_hash}")

    if not config.data_raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist at {config.data_raw_dir}")

    archive_csv = config.data_raw_dir / "fifaaccuracy_archive" / "fifatable.csv"
    if not archive_csv.exists():
        raise FileNotFoundError(f"Historical raw archive missing at {archive_csv}")

    return {
        "dvc_pointer": str(target_dvc),
        "md5_hash": md5_hash,
        "tracked_files": nfiles,
        "total_size_bytes": size_bytes,
        "archive_file_exists": True,
    }


def verify_and_ingest_raw_tables(db_path: Path | str | None = None) -> dict[str, int]:
    """Verify raw tables in DuckDB under strict hardware constraints."""
    target_db = str(db_path or config.duckdb_path)
    con = duckdb.connect(target_db, read_only=False)

    try:
        # Enforce hardware caps
        con.execute(f"SET threads = {config.duckdb_threads};")
        con.execute(f"SET memory_limit = '{config.duckdb_memory_limit}';")
        con.execute("CREATE SCHEMA IF NOT EXISTS raw;")

        # Check raw.fifatable
        fifatable_res = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='raw' AND table_name='fifatable';"
        ).fetchone()
        has_fifatable = bool(fifatable_res and fifatable_res[0] > 0)

        if not has_fifatable:
            archive_csv = config.data_raw_dir / "fifaaccuracy_archive" / "fifatable.csv"
            logger.info(f"Ingesting raw.fifatable from {archive_csv}...")
            con.execute(
                f"CREATE OR REPLACE TABLE raw.fifatable AS SELECT * FROM read_csv_auto('{archive_csv.as_posix()}', ignore_errors=true);"
            )

        # Check raw.fifa_granular
        granular_res = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='raw' AND table_name='fifa_granular';"
        ).fetchone()
        has_granular = bool(granular_res and granular_res[0] > 0)

        if not has_granular:
            raise RuntimeError(
                "raw.fifa_granular table missing and requires initial staging build."
            )

        # Query and verify verified record counts
        fifatable_count = int(con.execute("SELECT COUNT(*) FROM raw.fifatable;").fetchone()[0])
        granular_count = int(con.execute("SELECT COUNT(*) FROM raw.fifa_granular;").fetchone()[0])

        if fifatable_count != EXPECTED_FIFATABLE_ROWS:
            raise AssertionError(
                f"raw.fifatable row count mismatch: expected {EXPECTED_FIFATABLE_ROWS}, found {fifatable_count}"
            )

        if granular_count != EXPECTED_FIFA_GRANULAR_ROWS:
            raise AssertionError(
                f"raw.fifa_granular row count mismatch: expected {EXPECTED_FIFA_GRANULAR_ROWS}, found {granular_count}"
            )

        return {
            "fifatable": fifatable_count,
            "fifa_granular": granular_count,
        }
    finally:
        con.close()


@task(name="ingest-and-validate-raw-data", retries=2, retry_delay_seconds=1)
def ingest_raw_data_task() -> dict[str, Any]:
    """Prefect task validating DVC hashes and enforcing DuckDB raw table contracts."""
    t0 = time.perf_counter()
    logger.info("Executing DVC pointer validation and raw data ingestion...")

    dvc_info = validate_dvc_raw_artifacts()
    table_counts = verify_and_ingest_raw_tables()

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    logger.info(
        f"Raw data ingestion and validation completed in {elapsed_ms}ms "
        f"({table_counts['fifa_granular']} granular, {table_counts['fifatable']} historical rows)."
    )

    return {
        "status": "COMPLETED",
        "dvc_hash": dvc_info["md5_hash"],
        "tracked_files": dvc_info["tracked_files"],
        "tables": table_counts,
        "duration_ms": elapsed_ms,
    }
