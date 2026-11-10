"""Dependency injection providers for Analytics FC API v2.

Provides:
1. Hardware-governed DuckDB read-only session manager with deterministic teardown.
2. In-memory ModelRegistry singleton caching trained quantile, conformal, and
   archetype vector indexing engines for sub-30ms inference latency.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Generator
from pathlib import Path

import duckdb

from src.api.config import BASE_DIR, settings
from src.models.archetype_indexer import ArchetypeVectorIndex
from src.models.conformal_predictor import ConformalTrajectoryPredictor
from src.models.trajectory_regressor import QuantileTrajectoryRegressor

logger = logging.getLogger("analytics_fc.api.v2.dependencies")


class DuckDBSessionManager:
    """Thread-safe, hardware-capped DuckDB read-only session manager."""

    def __init__(
        self,
        db_path: Path | None = None,
        threads: int = 1,
        memory_limit: str = "1GB",
    ) -> None:
        self.db_path = db_path or settings.db_path
        self.threads = threads
        self.memory_limit = memory_limit

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """Open a single-threaded read-only DuckDB connection matching v1 config."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"DuckDB database not found at {self.db_path}")

        con = duckdb.connect(str(self.db_path), read_only=True)
        con.execute(f"SET threads = {int(self.threads)};")
        con.execute(f"SET memory_limit = '{self.memory_limit}';")
        return con

    def session(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Yield a managed DuckDB connection and ensure deterministic teardown."""
        con = self.get_connection()
        try:
            yield con
        finally:
            con.close()


class ModelRegistry:
    """Thread-safe in-memory cache holding ML inference artifacts."""

    def __init__(
        self,
        quantile_path: Path | None = None,
        conformal_path: Path | None = None,
        gallery_path: Path | None = None,
    ) -> None:
        self.quantile_path = quantile_path or (BASE_DIR / "models" / "trajectory_regressor.joblib")
        self.conformal_path = conformal_path or (BASE_DIR / "models" / "conformal_predictor.joblib")
        self.gallery_path = gallery_path or settings.gallery_path

        self._lock = threading.Lock()
        self._quantile_regressor: QuantileTrajectoryRegressor | None = None
        self._conformal_predictor: ConformalTrajectoryPredictor | None = None
        self._archetype_indexer: ArchetypeVectorIndex | None = None

    def get_quantile_regressor(self) -> QuantileTrajectoryRegressor:
        """Retrieve or load cached QuantileTrajectoryRegressor."""
        if self._quantile_regressor is None:
            with self._lock:
                if self._quantile_regressor is None:
                    logger.info("Loading QuantileTrajectoryRegressor from %s", self.quantile_path)
                    self._quantile_regressor = QuantileTrajectoryRegressor.load(self.quantile_path)
        return self._quantile_regressor

    def get_conformal_predictor(self) -> ConformalTrajectoryPredictor:
        """Retrieve or load cached ConformalTrajectoryPredictor."""
        if self._conformal_predictor is None:
            with self._lock:
                if self._conformal_predictor is None:
                    logger.info("Loading ConformalTrajectoryPredictor from %s", self.conformal_path)
                    self._conformal_predictor = ConformalTrajectoryPredictor.load(
                        self.conformal_path
                    )
        return self._conformal_predictor

    def get_archetype_indexer(self) -> ArchetypeVectorIndex:
        """Retrieve or load cached ArchetypeVectorIndex."""
        if self._archetype_indexer is None:
            with self._lock:
                if self._archetype_indexer is None:
                    logger.info("Loading ArchetypeVectorIndex from %s", self.gallery_path)
                    self._archetype_indexer = ArchetypeVectorIndex(gallery_path=self.gallery_path)
        return self._archetype_indexer

    def warmup(self) -> None:
        """Eagerly load all ML artifacts into memory during startup."""
        self.get_quantile_regressor()
        self.get_conformal_predictor()
        self.get_archetype_indexer()


_INIT_LOCK = threading.Lock()
_SESSION_MANAGER: DuckDBSessionManager | None = None
_MODEL_REGISTRY: ModelRegistry | None = None


def get_session_manager() -> DuckDBSessionManager:
    """Return global singleton DuckDBSessionManager."""
    global _SESSION_MANAGER
    if _SESSION_MANAGER is None:
        with _INIT_LOCK:
            if _SESSION_MANAGER is None:
                _SESSION_MANAGER = DuckDBSessionManager(db_path=settings.db_path)
    return _SESSION_MANAGER


def get_db_session() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """FastAPI dependency yielding a read-only DuckDB connection."""
    manager = get_session_manager()
    yield from manager.session()


def get_model_registry() -> ModelRegistry:
    """FastAPI dependency returning the cached ModelRegistry singleton."""
    global _MODEL_REGISTRY
    if _MODEL_REGISTRY is None:
        with _INIT_LOCK:
            if _MODEL_REGISTRY is None:
                _MODEL_REGISTRY = ModelRegistry()
    return _MODEL_REGISTRY


def get_quantile_regressor() -> QuantileTrajectoryRegressor:
    """FastAPI dependency returning cached QuantileTrajectoryRegressor."""
    return get_model_registry().get_quantile_regressor()


def get_conformal_predictor() -> ConformalTrajectoryPredictor:
    """FastAPI dependency returning cached ConformalTrajectoryPredictor."""
    return get_model_registry().get_conformal_predictor()


def get_archetype_indexer() -> ArchetypeVectorIndex:
    """FastAPI dependency returning cached ArchetypeVectorIndex."""
    return get_model_registry().get_archetype_indexer()
