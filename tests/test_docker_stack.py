"""Configuration and resource-cap tests for multi-service Docker Compose stack."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
DOCKERFILE_PATH = BASE_DIR / "Dockerfile"
COMPOSE_PATH = BASE_DIR / "docker-compose.yml"


def _parse_memory_mb(mem_str: str) -> float:
    """Convert Docker Compose memory string (e.g. '768M', '1.5G') to MiB."""
    clean = str(mem_str).strip().upper()
    if clean.endswith("G"):
        return float(clean[:-1]) * 1024.0
    if clean.endswith("M"):
        return float(clean[:-1])
    raise ValueError(f"Unsupported memory unit: {mem_str}")


def test_dockerfile_multistage_and_hardware_caps() -> None:
    """Verify multi-stage Dockerfile copies src and runtime modules safely."""
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert "AS builder" in content
    assert "AS runtime" in content
    assert "COPY --from=builder /opt/venv /opt/venv" in content
    assert "pydantic-settings" in content
    assert "COPY --chown=appuser:appuser src/ ./src/" in content
    assert "COPY --chown=appuser:appuser dashboard/ ./dashboard/" in content
    assert "COPY --chown=appuser:appuser data/monitoring/ ./data/monitoring/" in content
    assert "OMP_NUM_THREADS=1" in content
    assert "DUCKDB_THREADS=1" in content
    assert "USER appuser" in content
    assert "EXPOSE 8000 8501" in content


def test_compose_services_and_resource_ceilings() -> None:
    """Verify Compose stack defines api and dashboard capped at 1.0 CPU and 1.5GB total."""
    compose = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    services = compose.get("services", {})
    assert set(services.keys()) == {"api", "dashboard"}

    api = services["api"]
    dashboard = services["dashboard"]
    assert "uvicorn" in " ".join(api["command"])
    assert "--workers" in api["command"]
    assert "1" in api["command"]
    assert dashboard["environment"]["AFC_API_URL"] == "http://api:8000"
    assert dashboard["depends_on"]["api"]["condition"] == "service_healthy"

    total_cpus = 0.0
    total_mem_mb = 0.0
    for svc_name, svc in services.items():
        limits = svc["deploy"]["resources"]["limits"]
        svc_cpus = float(limits["cpus"])
        svc_mem = _parse_memory_mb(limits["memory"])
        assert svc_cpus > 0.0, f"{svc_name} must define positive CPU cap"
        assert svc_mem > 0.0, f"{svc_name} must define positive memory cap"
        total_cpus += svc_cpus
        total_mem_mb += svc_mem

        ro_mounts = [v for v in svc.get("volumes", []) if v.endswith(":ro")]
        assert len(ro_mounts) >= 2, f"{svc_name} must mount data artifacts read-only"

    assert total_cpus <= 1.0
    assert total_mem_mb <= 1536.0

    stack_limits = compose.get("x-stack-limits", {})
    assert float(stack_limits["cpus"]) == 1.0
    assert _parse_memory_mb(stack_limits["memory"]) == 1536.0


def test_docker_compose_config_validation() -> None:
    """Execute `docker compose config` when Docker CLI is installed."""
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI not available on PATH.")

    proc = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_PATH), "config"],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"docker compose config failed: {proc.stderr}"
    resolved = yaml.safe_load(proc.stdout)
    assert "api" in resolved["services"]
    assert "dashboard" in resolved["services"]
