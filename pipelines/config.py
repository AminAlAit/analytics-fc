"""Configuration settings and hardware constraints for Prefect orchestration flows.

Defines canonical repository paths, DuckDB resource limits (single thread,
1GB memory ceiling), and runtime configurations for ephemeral pipeline execution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Centralized pipeline path references and hardware safety boundaries."""

    base_dir: Path = Path(__file__).resolve().parent.parent
    data_raw_dir: Path = base_dir / "data" / "raw"
    data_processed_dir: Path = base_dir / "data" / "processed"
    dbt_project_dir: Path = base_dir / "dbt_project"
    duckdb_path: Path = dbt_project_dir / "analytics_fc.duckdb"
    gallery_path: Path = data_processed_dir / "superstar_gallery.json"

    # Hardware resource constraints for laptop execution safety
    duckdb_threads: int = 1
    duckdb_memory_limit: str = "1GB"


def configure_ephemeral_prefect() -> None:
    """Enforce ephemeral in-process Prefect execution with zero external daemons."""
    os.environ["PREFECT_API_URL"] = ""
    os.environ["PREFECT_LOGGING_LEVEL"] = "INFO"
    os.environ["PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS"] = "10"


config = PipelineConfig()
