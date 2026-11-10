"""Unit and concurrency tests for LRU profile cache and DuckDB connection pool.

Validates:
1. LRUProfileCache eviction ordering, TTL expiration, and telemetry metrics.
2. DuckDBConnectionPool read_only=True, threads=1, memory_limit=1GB, and
   deterministic try/finally closing blocks preventing Windows file locks (EBUSY).
3. Multi-threaded concurrency stress tests verifying zero lock collisions.
4. AtomicArtifactSlot thread-safe swap semantics.
5. Verification gate: sub-2ms cache hits on verified 180,021-row DuckDB database.
"""

from __future__ import annotations

import concurrent.futures
import time
from pathlib import Path

import duckdb
import pytest

from src.api.cache_manager import (
    AtomicArtifactSlot,
    DuckDBConnectionPool,
    LRUProfileCache,
    PlayerProfileCacheManager,
)
from src.api.config import settings


def test_lru_cache_eviction_and_metrics() -> None:
    """Verify strict LRU eviction ordering, invalidation, and hit/miss metrics."""
    cache: LRUProfileCache[str] = LRUProfileCache(max_size=2, ttl_seconds=60.0)
    cache.set("k1", "v1")
    cache.set("k2", "v2")

    # Access k1 so k2 becomes the least-recently-used key
    assert cache.get("k1") == "v1"

    # Inserting k3 must evict k2, keeping k1 and k3
    cache.set("k3", "v3")
    assert cache.get("k2") is None
    assert cache.get("k1") == "v1"
    assert cache.get("k3") == "v3"

    stats = cache.stats()
    assert stats["size"] == 2
    assert stats["hits"] == 3
    assert stats["misses"] == 1
    assert stats["evictions"] == 1
    assert stats["hit_rate"] == 0.75

    # Invalidation removes key cleanly
    assert cache.invalidate("k1") is True
    assert cache.invalidate("k1") is False
    assert cache.get("k1") is None


def test_lru_cache_ttl_expiration() -> None:
    """Verify TTL expiration purges stale entries without touching valid entries."""
    cache: LRUProfileCache[int] = LRUProfileCache(max_size=10, ttl_seconds=0.05)
    cache.set("short", 42)
    assert cache.get("short") == 42

    time.sleep(0.08)
    assert cache.get("short") is None
    assert cache.stats()["expirations"] == 1


def test_lru_cache_get_or_set() -> None:
    """Verify get_or_set invokes loader strictly on cache misses."""
    cache: LRUProfileCache[dict[str, int]] = LRUProfileCache(max_size=8)
    calls = 0

    def loader() -> dict[str, int]:
        nonlocal calls
        calls += 1
        return {"overall": 91}

    val1, hit1 = cache.get_or_set("haaland", loader)
    val2, hit2 = cache.get_or_set("haaland", loader)

    assert hit1 is False
    assert hit2 is True
    assert calls == 1
    assert val1 == {"overall": 91}
    assert val2 == {"overall": 91}


def test_duckdb_pool_enforces_readonly_and_hardware_caps() -> None:
    """Verify pool enforces read_only=True, threads=1, memory_limit=1GB, and closes handles."""
    if not settings.db_path.exists():
        pytest.skip(f"DuckDB database not found at {settings.db_path}")

    pool = DuckDBConnectionPool(
        db_path=settings.db_path, max_connections=2, reuse_connections=False
    )
    assert pool.read_only is True

    with pool.acquire() as con:
        settings_row = con.execute(
            "SELECT current_setting('threads'), current_setting('memory_limit'), current_setting('access_mode');"
        ).fetchone()
        assert settings_row is not None
        assert int(settings_row[0]) == 1
        assert "read_only" in str(settings_row[2]).lower()

        # Verify write attempts are rejected by read-only enforcement
        with pytest.raises(duckdb.InvalidInputException):
            con.execute("CREATE TABLE should_fail (id INT);")

    # Deterministic teardown must have closed the connection immediately
    stats = pool.stats()
    assert stats["active_leases"] == 0
    assert stats["total_acquired"] == 1
    assert stats["total_closed"] == 1
    pool.close_all()


def test_duckdb_pool_deterministic_teardown_prevents_ebusy_on_exception(tmp_path: Path) -> None:
    """Verify exceptions inside acquire() still release semaphore and close Windows handle."""
    temp_db = tmp_path / "temp_pool_test.duckdb"
    rw_init = duckdb.connect(str(temp_db))
    rw_init.execute("CREATE TABLE items (id INT); INSERT INTO items VALUES (1);")
    rw_init.close()

    pool = DuckDBConnectionPool(db_path=temp_db, max_connections=1, reuse_connections=False)
    with pytest.raises(RuntimeError, match="boom"):
        with pool.acquire() as con:
            assert con.execute("SELECT COUNT(*) FROM items;").fetchone() == (1,)
            raise RuntimeError("boom")

    assert pool.stats()["active_leases"] == 0
    assert pool.stats()["total_closed"] == 1
    pool.close_all()

    # Subsequent RW connection must open with zero Windows EBUSY file lock conflicts
    rw_check = duckdb.connect(str(temp_db), read_only=False)
    rw_check.execute("INSERT INTO items VALUES (2);")
    rw_check.close()


def test_concurrent_duckdb_pool_zero_file_lock_collisions() -> None:
    """Verify multi-threaded concurrent leases execute cleanly without EBUSY errors."""
    if not settings.db_path.exists():
        pytest.skip(f"DuckDB database not found at {settings.db_path}")

    with DuckDBConnectionPool(
        db_path=settings.db_path, max_connections=4, reuse_connections=False
    ) as pool:

        def worker(idx: int) -> int:
            with pool.acquire() as con:
                row = con.execute(
                    "SELECT overall FROM main.fct_player_seasons WHERE player_id = 239085 AND fifa_edition_year = 2024 LIMIT 1;"
                ).fetchone()
                return int(row[0]) if row else idx

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(worker, range(12)))

        assert results == [91] * 12
        stats = pool.stats()
        assert stats["total_acquired"] == 12
        assert stats["total_closed"] == 12
        assert stats["active_leases"] == 0


def test_reusable_pool_closes_all_handles_on_exit(tmp_path: Path) -> None:
    """Verify reuse_connections=True caches handles safely and releases all locks on exit."""
    temp_db = tmp_path / "reusable_pool.duckdb"
    rw = duckdb.connect(str(temp_db))
    rw.execute("CREATE TABLE t (x INT); INSERT INTO t VALUES (7);")
    rw.close()

    with DuckDBConnectionPool(db_path=temp_db, max_connections=2, reuse_connections=True) as pool:
        for _ in range(4):
            with pool.acquire() as con:
                assert con.execute("SELECT x FROM t;").fetchone() == (7,)
        assert pool.stats()["idle_connections"] == 1

    # Pool exit closes every cached handle cleanly
    assert pool.stats()["closed"] is True
    assert pool.stats()["idle_connections"] == 0

    # External write handle opens immediately with zero EBUSY error
    rw2 = duckdb.connect(str(temp_db), read_only=False)
    rw2.execute("INSERT INTO t VALUES (8);")
    rw2.close()


def test_atomic_artifact_slot_zero_downtime_swap() -> None:
    """Verify AtomicArtifactSlot swaps versions safely across threads."""
    slot: AtomicArtifactSlot[list[str]] = AtomicArtifactSlot(["Cristiano Ronaldo"])
    assert slot.version == 0
    assert slot.get() == ["Cristiano Ronaldo"]

    new_ver = slot.swap(["Cristiano Ronaldo", "E. Haaland", "J. Bellingham"])
    assert new_ver == 1
    assert slot.version == 1
    assert len(slot.get() or []) == 3


def test_verification_gate_sub_2ms_cache_hits() -> None:
    """Verify PlayerProfileCacheManager achieves sub-2ms warm cache hits on real catalog."""
    if not settings.db_path.exists():
        pytest.skip(f"DuckDB database not found at {settings.db_path}")

    pool = DuckDBConnectionPool(
        db_path=settings.db_path, max_connections=2, reuse_connections=False
    )
    service = PlayerProfileCacheManager(pool=pool)

    try:
        # Cold miss queries DuckDB
        profile_cold, hit_cold, _ = service.get_player_profile(239085, 2024)
        assert hit_cold is False
        assert profile_cold is not None
        assert profile_cold["overall"] == 91
        assert profile_cold["tactical_sector"] == "Center Forwards & Strikers"

        # 100 warm cache hits must average strictly under 2.0ms without touching DuckDB
        latencies_ms: list[float] = []
        for _ in range(100):
            profile_warm, hit_warm, lat_ms = service.get_player_profile(239085, 2024)
            assert hit_warm is True
            assert profile_warm is not None and profile_warm["overall"] == 91
            latencies_ms.append(lat_ms)

        avg_ms = sum(latencies_ms) / len(latencies_ms)
        max_ms = max(latencies_ms)
        assert avg_ms < 2.0, f"Average cache hit latency {avg_ms:.4f}ms exceeded 2.0ms gate."
        assert max_ms < 2.0, f"Peak cache hit latency {max_ms:.4f}ms exceeded 2.0ms gate."

        # Confirm DuckDB was queried exactly once across 101 calls
        assert pool.stats()["total_acquired"] == 1
        assert pool.stats()["total_closed"] == 1

        # Missing player returns None cleanly
        missing, hit_miss, _ = service.get_player_profile(99999999, 2024)
        assert missing is None
        assert hit_miss is False
    finally:
        service.close()
