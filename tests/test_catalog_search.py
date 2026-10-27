"""Automated contract tests and performance benchmarks for catalog search engine.

Validates query correctness, SQL injection immunity, pagination partitions,
tactical sector facets, complete skill attribute vectors, and sub-100ms p95 latency.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from src.catalog_search import (
    ALL_SKILL_COLUMNS,
    CORE_SEARCH_COLUMNS,
    SECTOR_ORDER,
    get_catalog_stats,
    get_connection,
    get_player_by_id_and_year,
    search_players,
    search_players_paginated,
)


@pytest.fixture(scope="module")
def shared_con():
    """Shared DuckDB connection fixture with hardware limits."""
    con = get_connection()
    # Warm up buffer cache
    con.execute("SELECT 1 FROM stg_fifa_granular WHERE short_name ILIKE '%a%'").fetchall()
    yield con
    con.close()


def test_catalog_stats_invariants(shared_con):
    """Verify underlying catalog row counts, distinct players, and edition boundaries."""
    stats = get_catalog_stats(con=shared_con)
    assert stats["total_rows"] == 142079
    assert stats["distinct_players"] == 45629
    assert stats["min_year"] == 2015
    assert stats["max_year"] == 2022


def test_search_players_filtering(shared_con):
    """Verify parameterized search by name, edition year, and overall rating thresholds."""
    results = search_players(
        query="Haaland",
        edition_year=2022,
        min_overall=85,
        limit=10,
        con=shared_con,
    )
    assert len(results) > 0
    for r in results:
        assert "haaland" in r["short_name"].lower() or "haaland" in r["long_name"].lower()
        assert r["fifa_edition_year"] == 2022
        assert r["overall"] >= 85


def test_search_players_injection_safety(shared_con):
    """Verify SQL injection payloads are safely parameterized and executed without error."""
    payloads = [
        "' OR '1'='1",
        "'; DROP TABLE stg_fifa_granular; --",
        "admin'--",
        "UNION SELECT NULL, NULL, NULL",
    ]
    for payload in payloads:
        results = search_players(query=payload, limit=5, con=shared_con)
        assert isinstance(results, list)
        assert len(results) == 0


def test_paginated_partitions_deterministic(shared_con):
    """Verify multi-page queries produce mutually exclusive partitions with zero duplicate keys."""
    page_1 = search_players_paginated(query="Saka", limit=10, offset=0, con=shared_con)
    page_2 = search_players_paginated(query="Saka", limit=10, offset=10, con=shared_con)
    page_3 = search_players_paginated(query="Saka", limit=10, offset=20, con=shared_con)

    assert len(page_1["items"]) == 10
    assert len(page_2["items"]) == 10
    assert len(page_3["items"]) == 10
    assert page_1["total_matching"] == page_2["total_matching"] == page_3["total_matching"]
    assert page_1["total_matching"] > 20

    k1 = {(r["player_id"], r["fifa_edition_year"]) for r in page_1["items"]}
    k2 = {(r["player_id"], r["fifa_edition_year"]) for r in page_2["items"]}
    k3 = {(r["player_id"], r["fifa_edition_year"]) for r in page_3["items"]}

    # Zero overlap between partition boundaries
    assert len(k1 & k2) == 0
    assert len(k2 & k3) == 0
    assert len(k1 & k3) == 0
    assert page_1["has_more"] is True


def test_tactical_sector_facets(shared_con):
    """Verify tactical sector filtering returns uniform roles and complete facet counts."""
    res = search_players_paginated(query="Saka", sector="Defenders", limit=10, con=shared_con)
    assert len(res["items"]) > 0
    for item in res["items"]:
        assert item["tactical_sector"] == "Defenders"

    # Facet counts must report across all canonical sectors
    for sec in SECTOR_ORDER:
        assert sec in res["sector_counts"]
        assert res["sector_counts"][sec] >= 0


def test_complete_player_profile_retrieval(shared_con):
    """Verify complete player profile extraction contains all 34 tactical skill attributes."""
    search_res = search_players("Benzema", limit=1, con=shared_con)
    assert len(search_res) == 1
    top_cand = search_res[0]

    profile = get_player_by_id_and_year(
        top_cand["player_id"],
        top_cand["fifa_edition_year"],
        con=shared_con,
    )
    assert profile is not None

    for col in CORE_SEARCH_COLUMNS:
        assert col in profile
    for col in ALL_SKILL_COLUMNS:
        assert col in profile
        assert profile[col] is not None


def test_catalog_search_latency_benchmark_p95(shared_con):
    """Benchmark 100 realistic player queries asserting p95 latency under 100ms."""
    sample_names = [
        "Haaland", "Mbapp", "Bellingham", "Musiala", "Pedri",
        "Saka", "Foden", "Kane", "Benzema", "Modri",
        "Vinicius", "Rodri", "Rice", "Silva", "Dias",
        "Alisson", "Courtois", "Casemiro", "Kroos", "Griezmann",
        "Neymar", "Messi", "Ronaldo", "Salah", "Davies",
    ]
    benchmark_queries = (sample_names * 4)[:100]
    durations_ms: list[float] = []

    for query in benchmark_queries:
        t0 = time.perf_counter()
        results = search_players(query=query, limit=20, con=shared_con)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        durations_ms.append(elapsed_ms)
        assert len(results) > 0

    assert len(durations_ms) == 100
    durations_ms.sort()

    p50 = durations_ms[int(100 * 0.50)]
    p90 = durations_ms[int(100 * 0.90)]
    p95 = durations_ms[int(100 * 0.95)]
    mean_ms = sum(durations_ms) / 100.0

    print(f"\n[BENCHMARK] 100 Catalog Queries: mean={mean_ms:.2f}ms, p50={p50:.2f}ms, p90={p90:.2f}ms, p95={p95:.2f}ms")

    # Strict SLA Assertion: p95 latency must stay strictly below 100ms
    assert p95 < 100.0, f"p95 latency exceeded threshold: {p95:.2f}ms >= 100ms"
