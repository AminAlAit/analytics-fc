"""Unit and endpoint tests for runtime observability and PSI drift telemetry."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import app
from src.monitoring.observability import (
    TelemetryCollector,
    collect_memory_telemetry,
    compute_batch_psi_alerts,
    get_telemetry_collector,
    get_telemetry_snapshot,
)


def test_telemetry_collector_latency_percentiles() -> None:
    """Verify LatencyPercentiles calculation and endpoint request counting."""
    collector = TelemetryCollector(max_samples=16)
    empty_stats = collector.percentile_stats()
    assert empty_stats.sample_count == 0
    assert empty_stats.p95_ms == 0.0

    for ms in [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0]:
        collector.record_latency("/api/v2/search", ms)
    collector.record_latency("/api/v2/telemetry", 0.5)

    stats = collector.percentile_stats()
    assert stats.sample_count == 8
    assert stats.p50_ms <= stats.p95_ms <= stats.p99_ms <= stats.max_ms
    assert stats.endpoints["/api/v2/search"] == 7
    assert stats.endpoints["/api/v2/telemetry"] == 1

    collector.clear()
    assert collector.percentile_stats().sample_count == 0


def test_memory_telemetry_respects_1gb_ceiling() -> None:
    """Verify RSS memory sampling flags both healthy and over-cap states."""
    healthy = collect_memory_telemetry(memory_ceiling_mb=1024.0)
    assert healthy.rss_mb > 0.0
    assert healthy.within_memory_ceiling is True

    breached = collect_memory_telemetry(memory_ceiling_mb=0.01)
    assert breached.within_memory_ceiling is False
    assert breached.memory_utilization_pct > 100.0


def test_compute_batch_psi_alerts_and_cached_snapshot(tmp_path: Path) -> None:
    """Detect PSI warnings/critical shifts on live batches and cached drift metrics."""
    rng = np.random.default_rng(7)
    ref_df = pd.DataFrame(
        {
            "overall": rng.normal(70.0, 5.0, size=1000),
            "pace": rng.normal(68.0, 6.0, size=1000),
        }
    )
    shifted_df = pd.DataFrame(
        {
            "overall": rng.normal(70.0, 5.0, size=1000),
            "pace": rng.normal(86.0, 5.0, size=1000),
        }
    )

    batch_summary = compute_batch_psi_alerts(
        reference_df=ref_df,
        current_df=shifted_df,
        features=["overall", "pace"],
        psi_threshold=0.10,
        critical_threshold=0.25,
    )
    assert batch_summary.source == "live_batch"
    assert batch_summary.evaluated_features == 2
    assert batch_summary.alert_count >= 1
    assert any(a.feature == "pace" and a.severity == "critical" for a in batch_summary.alerts)

    # Cached snapshot against canonical drift_metrics.json (if present)
    metrics_file = Path("data/monitoring/drift_metrics.json")
    if metrics_file.exists():
        snap_summary = compute_batch_psi_alerts(metrics_path=metrics_file, psi_threshold=0.10)
        assert snap_summary.source == "cached_snapshot"
        assert snap_summary.evaluated_features >= 40

    # Missing or corrupt metrics file must degrade cleanly
    missing_summary = compute_batch_psi_alerts(metrics_path=tmp_path / "missing.json")
    assert missing_summary.source == "unavailable"
    assert missing_summary.alerts == []

    bad_json = tmp_path / "corrupt.json"
    bad_json.write_text("{invalid json", encoding="utf-8")
    corrupt_summary = compute_batch_psi_alerts(metrics_path=bad_json)
    assert corrupt_summary.source == "unavailable"


def test_v2_telemetry_endpoint_sub_10ms_with_psi_alerts() -> None:
    """GET /api/v2/telemetry returns memory, latency percentiles, and PSI alerts in < 10ms."""
    get_telemetry_collector().clear()
    client = TestClient(app)

    # Warmup primes the mtime cache and records first observation
    warmup = client.get("/api/v2/telemetry")
    assert warmup.status_code == 200

    t0 = time.perf_counter()
    resp = client.get("/api/v2/telemetry")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert resp.status_code == 200
    assert elapsed_ms < 10.0, f"GET /api/v2/telemetry took {elapsed_ms:.2f}ms"

    payload = resp.json()
    assert payload["status"] in {"healthy", "memory_pressure"}
    assert payload["response_time_ms"] < 10.0
    assert payload["memory"]["rss_mb"] > 0.0
    assert payload["latency_percentiles"]["sample_count"] >= 2
    assert "drift_telemetry" in payload
    assert "alerts" in payload["drift_telemetry"]

    direct = get_telemetry_snapshot()
    assert direct.response_time_ms < 10.0


def test_non_finite_latency_and_malformed_metrics_sanitization(tmp_path: Path) -> None:
    """Reject NaN/Inf latency poison and sanitize corrupt PSI/edition fields cleanly."""
    collector = TelemetryCollector(max_samples=8)
    collector.record_latency("/api/v2/telemetry", float("nan"))
    collector.record_latency("/api/v2/telemetry", float("inf"))
    collector.record_latency("/api/v2/telemetry", float("-inf"))
    collector.record_latency("/api/v2/telemetry", -3.0)
    collector.record_latency("/api/v2/telemetry", 4.0)

    stats = collector.percentile_stats()
    assert stats.sample_count == 2
    assert stats.p50_ms == 2.0
    assert stats.max_ms == 4.0

    malformed_path = tmp_path / "malformed_metrics.json"
    malformed_path.write_text(
        """
        {
          "reference_edition": "invalid",
          "target_edition": 2022,
          "dataset_drift_detected": true,
          "features": {
            "overall": {"psi": null},
            "pace": {"psi": "bad-string"},
            "passing": {"psi": -0.5},
            "shooting": {"psi": 0.31}
          }
        }
        """,
        encoding="utf-8",
    )
    summary = compute_batch_psi_alerts(metrics_path=malformed_path)
    assert summary.source == "cached_snapshot"
    assert summary.reference_edition is None
    assert summary.target_edition == 2022
    assert summary.evaluated_features == 4
    assert summary.alert_count == 1
    assert summary.alerts[0].feature == "shooting"
    assert summary.alerts[0].severity == "critical"
