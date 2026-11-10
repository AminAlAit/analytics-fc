"""High-performance LRU profile cache and hardware-capped DuckDB connection manager.

Provides:
1. Thread-safe in-memory LRU + TTL cache delivering sub-2ms hits for player dossiers.
2. Thread-safe DuckDB read-only connection pool enforcing threads=1, memory_limit=1GB,
   deterministic try/finally teardown, and zero Windows file lock conflicts (EBUSY).
3. Atomic reference holder for zero-downtime in-memory artifact swaps.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import duckdb

from src.api.config import settings
from src.catalog_search import get_primary_position, get_sector_for_position

logger = logging.getLogger("analytics_fc.api.cache_manager")

T = TypeVar("T")


@dataclass
class _CacheEntry(Generic[T]):
    """Internal timestamped cache item."""

    value: T
    expires_at: float | None


class LRUProfileCache(Generic[T]):
    """Thread-safe in-memory LRU cache with optional TTL expiration."""

    def __init__(
        self,
        max_size: int = 1024,
        ttl_seconds: float | None = 300.0,
    ) -> None:
        if max_size < 1:
            raise ValueError("max_size must be at least 1.")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive when provided.")

        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._store: OrderedDict[str, _CacheEntry[T]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    def get(self, key: str) -> T | None:
        """Retrieve item by key, returning None on miss or expiration."""
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None

            if entry.expires_at is not None and now >= entry.expires_at:
                del self._store[key]
                self._expirations += 1
                self._misses += 1
                return None

            self._store.move_to_end(key)
            self._hits += 1
            return entry.value

    def set(self, key: str, value: T, ttl_seconds: float | None = None) -> None:
        """Insert or update a key in the cache, evicting the LRU entry if full."""
        effective_ttl = ttl_seconds if ttl_seconds is not None else self.ttl_seconds
        expires_at = (time.monotonic() + effective_ttl) if effective_ttl is not None else None

        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = _CacheEntry(value=value, expires_at=expires_at)

            while len(self._store) > self.max_size:
                self._store.popitem(last=False)
                self._evictions += 1

    def get_or_set(
        self,
        key: str,
        loader: Callable[[], T],
        ttl_seconds: float | None = None,
    ) -> tuple[T, bool]:
        """Return cached item or execute loader on miss.

        Returns:
            Tuple of (value, cache_hit).
        """
        cached = self.get(key)
        if cached is not None:
            return cached, True

        fresh_value = loader()
        self.set(key, fresh_value, ttl_seconds=ttl_seconds)
        return fresh_value, False

    def invalidate(self, key: str) -> bool:
        """Remove a specific key from the cache if present."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Remove all entries from the cache and reset counters."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._expirations = 0

    def stats(self) -> dict[str, Any]:
        """Return cache telemetry including size, hits, misses, and hit rate."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = round(self._hits / total_requests, 4) if total_requests > 0 else 0.0
            return {
                "size": len(self._store),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "expirations": self._expirations,
                "hit_rate": hit_rate,
            }


class DuckDBConnectionPool:
    """Thread-safe DuckDB read-only connection manager enforcing hardware caps.

    Always opens connections with read_only=True, threads=1, and memory_limit='1GB'.
    Uses bounded semaphore concurrency and deterministic try/finally teardown so
    Windows file locks (EBUSY) never persist across requests or pipelines.
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        max_connections: int = 4,
        threads: int = 1,
        memory_limit: str = "1GB",
        reuse_connections: bool = False,
        max_retries: int = 3,
        retry_delay_seconds: float = 0.02,
    ) -> None:
        if max_connections < 1:
            raise ValueError("max_connections must be at least 1.")
        if threads < 1:
            raise ValueError("threads must be at least 1.")

        self.db_path = Path(db_path) if db_path is not None else settings.db_path
        self.max_connections = max_connections
        self.threads = int(threads)
        self.memory_limit = str(memory_limit)
        self.reuse_connections = reuse_connections
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

        self._semaphore = threading.BoundedSemaphore(self.max_connections)
        self._lock = threading.Lock()
        self._idle_connections: list[duckdb.DuckDBPyConnection] = []
        self._active_leases = 0
        self._total_acquired = 0
        self._total_closed = 0
        self._lock_retries = 0
        self._closed = False

    @property
    def read_only(self) -> bool:
        """Confirm pool strictly enforces read-only connections."""
        return True

    def _open_new_connection(self) -> duckdb.DuckDBPyConnection:
        """Open a single hardware-governed read-only connection with lock retry."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"DuckDB database not found at {self.db_path}")

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                con = duckdb.connect(str(self.db_path), read_only=True)
                try:
                    con.execute(f"SET threads = {self.threads};")
                    con.execute(f"SET memory_limit = '{self.memory_limit}';")
                except Exception:
                    con.close()
                    raise
                return con
            except duckdb.IOException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    with self._lock:
                        self._lock_retries += 1
                    time.sleep(self.retry_delay_seconds * (attempt + 1))
                    continue
                raise

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Failed to establish DuckDB read-only connection.")

    @contextmanager
    def acquire(self, timeout: float = 5.0) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Lease a read-only DuckDB connection inside a deterministic try/finally block."""
        if self._closed:
            raise RuntimeError("DuckDBConnectionPool has been closed.")

        acquired = self._semaphore.acquire(timeout=timeout)
        if not acquired:
            raise TimeoutError(
                f"Timed out waiting for DuckDB connection lease after {timeout:.2f}s."
            )

        con: duckdb.DuckDBPyConnection | None = None
        lease_registered = False
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("DuckDBConnectionPool has been closed.")
                self._active_leases += 1
                lease_registered = True
                if self.reuse_connections and self._idle_connections:
                    con = self._idle_connections.pop()

            if con is None:
                con = self._open_new_connection()

            with self._lock:
                self._total_acquired += 1

            yield con
        finally:
            try:
                if con is not None:
                    with self._lock:
                        should_return = (
                            self.reuse_connections
                            and not self._closed
                            and len(self._idle_connections) < self.max_connections
                        )
                        if should_return:
                            self._idle_connections.append(con)
                            con = None
                    if con is not None:
                        con.close()
                        with self._lock:
                            self._total_closed += 1
            finally:
                if lease_registered:
                    with self._lock:
                        self._active_leases = max(0, self._active_leases - 1)
                self._semaphore.release()

    def close_all(self) -> None:
        """Close all idle pooled connections and prevent new acquisitions."""
        with self._lock:
            self._closed = True
            idle = list(self._idle_connections)
            self._idle_connections.clear()

        for con in idle:
            try:
                con.close()
            finally:
                with self._lock:
                    self._total_closed += 1

    def stats(self) -> dict[str, Any]:
        """Return pool telemetry including active leases, closures, and hardware caps."""
        with self._lock:
            return {
                "db_path": str(self.db_path),
                "read_only": self.read_only,
                "threads": self.threads,
                "memory_limit": self.memory_limit,
                "max_connections": self.max_connections,
                "reuse_connections": self.reuse_connections,
                "active_leases": self._active_leases,
                "idle_connections": len(self._idle_connections),
                "total_acquired": self._total_acquired,
                "total_closed": self._total_closed,
                "lock_retries": self._lock_retries,
                "closed": self._closed,
            }

    def __enter__(self) -> DuckDBConnectionPool:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close_all()


class AtomicArtifactSlot(Generic[T]):
    """Thread-safe container supporting zero-downtime atomic pointer swaps."""

    def __init__(self, initial_value: T | None = None) -> None:
        self._lock = threading.RLock()
        self._value = initial_value
        self._version = 0

    def get(self) -> T | None:
        """Return the currently active artifact reference."""
        with self._lock:
            return self._value

    def swap(self, new_value: T) -> int:
        """Atomically replace the active artifact and increment the version counter."""
        with self._lock:
            self._value = new_value
            self._version += 1
            return self._version

    @property
    def version(self) -> int:
        """Return monotonic swap version counter."""
        with self._lock:
            return self._version


class PlayerProfileCacheManager:
    """Coordinated LRU profile cache and read-only DuckDB pool manager."""

    def __init__(
        self,
        pool: DuckDBConnectionPool | None = None,
        cache: LRUProfileCache[dict[str, Any]] | None = None,
    ) -> None:
        self.pool = pool if pool is not None else DuckDBConnectionPool()
        self.cache = cache if cache is not None else LRUProfileCache()

    def get_player_profile(
        self,
        player_id: int,
        edition_year: int = 2024,
    ) -> tuple[dict[str, Any] | None, bool, float]:
        """Fetch player profile via LRU cache or pooled DuckDB read-only fallback.

        Returns:
            Tuple of (profile_dict_or_none, cache_hit, latency_ms).
        """
        t0 = time.perf_counter()
        cache_key = f"player:{player_id}:{edition_year}"

        cached = self.cache.get(cache_key)
        if cached is not None:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
            return cached, True, elapsed_ms

        sql = """
            SELECT
                player_id,
                fifa_edition_year,
                short_name,
                long_name,
                age,
                nationality,
                club,
                league_name,
                positions,
                overall,
                potential,
                value_eur,
                pace,
                shooting,
                passing,
                dribbling,
                defending,
                physic,
                COALESCE(annual_growth_velocity, 0.0) AS annual_growth_velocity
            FROM main.fct_player_seasons
            WHERE player_id = ? AND fifa_edition_year = ?
            LIMIT 1;
        """
        profile: dict[str, Any] | None = None
        with self.pool.acquire() as con:
            cursor = con.execute(sql, [player_id, edition_year])
            col_names = [desc[0] for desc in cursor.description]
            row = cursor.fetchone()
            if row is not None:
                profile = dict(zip(col_names, row))
                profile["primary_pos"] = get_primary_position(profile.get("positions"))
                profile["tactical_sector"] = get_sector_for_position(profile.get("positions"))

        if profile is not None:
            self.cache.set(cache_key, profile)

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
        return profile, False, elapsed_ms

    def close(self) -> None:
        """Tear down pool and clear cache cleanly."""
        self.pool.close_all()
        self.cache.clear()
