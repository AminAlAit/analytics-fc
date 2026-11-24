"""Runtime observability telemetry and Population Stability Index (PSI) alerts.

Provides thread-safe latency percentile tracking, process RSS memory ceiling
diagnostics, and batch or cached PSI drift alerts for FastAPI v2 monitoring.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
from pydantic import BaseModel, ConfigDict, Field

from src.monitoring.config import ALL_MONITORED_NUMERICAL_FEATURES, DEFAULT_METRICS_PATH

logger = logging.getLogger("analytics_fc.monitoring.observability")


def _safe_non_negative_float(value: Any, default: float = 0.0) -> float:
    """Convert raw numeric value into a finite non-negative float."""
    if value is None or isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return max(0.0, parsed)


def _safe_optional_int(value: Any) -> int | None:
    """Convert raw edition metadata into int, returning None when invalid."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class LatencyPercentiles(BaseModel):
    """Summary percentiles computed from recorded runtime endpoint latencies."""

    model_config = ConfigDict(frozen=True)

    sample_count: int = Field(ge=0)
    p50_ms: float = Field(ge=0.0)
    p95_ms: float = Field(ge=0.0)
    p99_ms: float = Field(ge=0.0)
    max_ms: float = Field(ge=0.0)
    endpoints: dict[str, int] = Field(default_factory=dict)


class PSIAlert(BaseModel):
    """Single feature Population Stability Index drift alert."""

    model_config = ConfigDict(frozen=True)

    feature: str
    psi: float = Field(ge=0.0)
    threshold: float = Field(ge=0.0)
    severity: str


class MemoryTelemetry(BaseModel):
    """Process RSS memory telemetry evaluated against the 1GB hardware ceiling."""

    model_config = ConfigDict(frozen=True)

    pid: int
    rss_mb: float = Field(ge=0.0)
    memory_ceiling_mb: float = Field(gt=0.0)
    within_memory_ceiling: bool
    memory_utilization_pct: float = Field(ge=0.0)


class DriftAlertSummary(BaseModel):
    """Aggregated Population Stability Index drift status and active alerts."""

    model_config = ConfigDict(frozen=True)

    source: str
    reference_edition: int | None = None
    target_edition: int | None = None
    dataset_drift_detected: bool
    evaluated_features: int = Field(ge=0)
    psi_threshold: float = Field(ge=0.0)
    critical_threshold: float = Field(ge=0.0)
    alert_count: int = Field(ge=0)
    max_psi: float = Field(ge=0.0)
    alerts: list[PSIAlert]


class TelemetrySnapshot(BaseModel):
    """Unified observability payload returned by GET /api/v2/telemetry."""

    model_config = ConfigDict(frozen=True)

    status: str
    memory: MemoryTelemetry
    latency_percentiles: LatencyPercentiles
    drift_telemetry: DriftAlertSummary
    response_time_ms: float = Field(ge=0.0)


class TelemetryCollector:
    """Thread-safe bounded ring buffer capturing runtime API latency samples."""

    def __init__(self, max_samples: int = 512) -> None:
        if max_samples < 1:
            raise ValueError("max_samples must be at least 1.")
        self._max_samples = max_samples
        self._lock = threading.RLock()
        self._samples: deque[tuple[str, float]] = deque(maxlen=max_samples)
        self._endpoint_counts: dict[str, int] = {}

    def record_latency(self, endpoint: str, latency_ms: float) -> None:
        """Record a finite, non-negative latency observation for a specific route."""
        try:
            parsed = float(latency_ms)
        except (TypeError, ValueError):
            return
        if not math.isfinite(parsed):
            return
        safe_latency = max(0.0, parsed)
        with self._lock:
            self._samples.append((endpoint, safe_latency))
            self._endpoint_counts[endpoint] = self._endpoint_counts.get(endpoint, 0) + 1

    def percentile_stats(self) -> LatencyPercentiles:
        """Compute p50, p95, p99, and max latencies in milliseconds."""
        with self._lock:
            if not self._samples:
                return LatencyPercentiles(
                    sample_count=0,
                    p50_ms=0.0,
                    p95_ms=0.0,
                    p99_ms=0.0,
                    max_ms=0.0,
                    endpoints={},
                )
            values = np.fromiter((val for _, val in self._samples), dtype=float)
            counts = dict(self._endpoint_counts)

        p50, p95, p99 = np.percentile(values, [50.0, 95.0, 99.0])
        return LatencyPercentiles(
            sample_count=int(values.size),
            p50_ms=round(float(p50), 3),
            p95_ms=round(float(p95), 3),
            p99_ms=round(float(p99), 3),
            max_ms=round(float(np.max(values)), 3),
            endpoints=counts,
        )

    def clear(self) -> None:
        """Reset stored latency observations and endpoint counts."""
        with self._lock:
            self._samples.clear()
            self._endpoint_counts.clear()


_DEFAULT_COLLECTOR = TelemetryCollector()
_METRICS_CACHE_LOCK = threading.Lock()
_METRICS_CACHE: dict[str, Any] = {"path": None, "mtime_ns": None, "payload": None}


def get_telemetry_collector() -> TelemetryCollector:
    """Return the global singleton TelemetryCollector instance."""
    return _DEFAULT_COLLECTOR


def collect_memory_telemetry(memory_ceiling_mb: float = 1024.0) -> MemoryTelemetry:
    """Measure process RSS memory against the hardware ceiling."""
    proc = psutil.Process()
    rss_mb = round(proc.memory_info().rss / (1024.0 * 1024.0), 2)
    util_pct = round((rss_mb / memory_ceiling_mb) * 100.0, 2)
    return MemoryTelemetry(
        pid=proc.pid,
        rss_mb=rss_mb,
        memory_ceiling_mb=memory_ceiling_mb,
        within_memory_ceiling=rss_mb <= memory_ceiling_mb,
        memory_utilization_pct=util_pct,
    )


def _calculate_psi_array(
    reference: pd.Series,
    current: pd.Series,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Compute Population Stability Index across quantile bins cleanly."""
    try:
        from src.monitoring.drift_analyzer import calculate_psi

        return calculate_psi(
            reference=reference, current=current, num_bins=num_bins, epsilon=epsilon
        )
    except ImportError:
        ref_clean = reference.dropna().to_numpy(dtype=float)
        curr_clean = current.dropna().to_numpy(dtype=float)
        if ref_clean.size == 0 or curr_clean.size == 0:
            return 0.0
        edges = np.unique(np.percentile(ref_clean, np.linspace(0.0, 100.0, num_bins + 1)))
        if edges.size < 2:
            return 0.0
        ref_counts, _ = np.histogram(ref_clean, bins=edges)
        curr_counts, _ = np.histogram(curr_clean, bins=edges)
        ref_pct = (ref_counts + epsilon) / (ref_clean.size + epsilon * ref_counts.size)
        curr_pct = (curr_counts + epsilon) / (curr_clean.size + epsilon * curr_counts.size)
        return float(max(0.0, np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))))


def _load_cached_drift_metrics(metrics_path: Path) -> dict[str, Any] | None:
    """Load drift_metrics.json with mtime-based in-memory caching for sub-millisecond reads."""
    if not metrics_path.exists():
        return None
    stat = metrics_path.stat()
    key_path = str(metrics_path.resolve())
    with _METRICS_CACHE_LOCK:
        if (
            _METRICS_CACHE["path"] == key_path
            and _METRICS_CACHE["mtime_ns"] == stat.st_mtime_ns
            and _METRICS_CACHE["payload"] is not None
        ):
            return dict(_METRICS_CACHE["payload"])

        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to parse drift metrics JSON at %s: %s", metrics_path, exc)
            return None

        if not isinstance(payload, dict):
            return None

        _METRICS_CACHE["path"] = key_path
        _METRICS_CACHE["mtime_ns"] = stat.st_mtime_ns
        _METRICS_CACHE["payload"] = payload
        return dict(payload)


def compute_batch_psi_alerts(
    reference_df: pd.DataFrame | None = None,
    current_df: pd.DataFrame | None = None,
    features: list[str] | None = None,
    psi_threshold: float = 0.10,
    critical_threshold: float = 0.25,
    metrics_path: Path | str | None = None,
) -> DriftAlertSummary:
    """Compute PSI drift alerts across live DataFrame batches or cached snapshot metrics."""
    alerts: list[PSIAlert] = []

    if reference_df is not None and current_df is not None:
        candidate_features = features or list(ALL_MONITORED_NUMERICAL_FEATURES)
        evaluated = 0
        max_psi = 0.0
        for feat in candidate_features:
            if feat not in reference_df.columns or feat not in current_df.columns:
                continue
            evaluated += 1
            psi_val = round(
                _calculate_psi_array(reference_df[feat], current_df[feat]),
                4,
            )
            max_psi = max(max_psi, psi_val)
            if psi_val >= psi_threshold:
                severity = "critical" if psi_val >= critical_threshold else "warning"
                alerts.append(
                    PSIAlert(
                        feature=feat,
                        psi=psi_val,
                        threshold=psi_threshold,
                        severity=severity,
                    )
                )

        alerts.sort(key=lambda item: (-item.psi, item.feature))
        return DriftAlertSummary(
            source="live_batch",
            reference_edition=None,
            target_edition=None,
            dataset_drift_detected=len(alerts) > 0,
            evaluated_features=evaluated,
            psi_threshold=psi_threshold,
            critical_threshold=critical_threshold,
            alert_count=len(alerts),
            max_psi=round(max_psi, 4),
            alerts=alerts,
        )

    target_metrics = Path(metrics_path) if metrics_path is not None else DEFAULT_METRICS_PATH
    cached = _load_cached_drift_metrics(target_metrics)
    if cached is None:
        return DriftAlertSummary(
            source="unavailable",
            reference_edition=None,
            target_edition=None,
            dataset_drift_detected=False,
            evaluated_features=0,
            psi_threshold=psi_threshold,
            critical_threshold=critical_threshold,
            alert_count=0,
            max_psi=0.0,
            alerts=[],
        )

    feat_map = cached.get("features", {})
    max_psi = 0.0
    evaluated = 0
    if isinstance(feat_map, dict):
        for feat_name, stats in feat_map.items():
            if not isinstance(stats, dict):
                continue
            evaluated += 1
            psi_val = round(_safe_non_negative_float(stats.get("psi"), default=0.0), 4)
            max_psi = max(max_psi, psi_val)
            if psi_val >= psi_threshold:
                severity = "critical" if psi_val >= critical_threshold else "warning"
                alerts.append(
                    PSIAlert(
                        feature=str(feat_name),
                        psi=psi_val,
                        threshold=psi_threshold,
                        severity=severity,
                    )
                )

    alerts.sort(key=lambda item: (-item.psi, item.feature))
    return DriftAlertSummary(
        source="cached_snapshot",
        reference_edition=_safe_optional_int(cached.get("reference_edition")),
        target_edition=_safe_optional_int(cached.get("target_edition")),
        dataset_drift_detected=bool(cached.get("dataset_drift_detected", len(alerts) > 0)),
        evaluated_features=evaluated,
        psi_threshold=psi_threshold,
        critical_threshold=critical_threshold,
        alert_count=len(alerts),
        max_psi=round(max_psi, 4),
        alerts=alerts,
    )


def get_telemetry_snapshot(
    collector: TelemetryCollector | None = None,
    metrics_path: Path | str | None = None,
    psi_threshold: float = 0.10,
    critical_threshold: float = 0.25,
    memory_ceiling_mb: float = 1024.0,
) -> TelemetrySnapshot:
    """Assemble full runtime observability health and PSI alert snapshot in < 10ms."""
    t0 = time.perf_counter()
    active_collector = collector or get_telemetry_collector()

    memory = collect_memory_telemetry(memory_ceiling_mb=memory_ceiling_mb)
    drift_telemetry = compute_batch_psi_alerts(
        psi_threshold=psi_threshold,
        critical_threshold=critical_threshold,
        metrics_path=metrics_path,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    active_collector.record_latency("/api/v2/telemetry", elapsed_ms)
    percentiles = active_collector.percentile_stats()

    status_label = "healthy" if memory.within_memory_ceiling else "memory_pressure"
    return TelemetrySnapshot(
        status=status_label,
        memory=memory,
        latency_percentiles=percentiles,
        drift_telemetry=drift_telemetry,
        response_time_ms=elapsed_ms,
    )
