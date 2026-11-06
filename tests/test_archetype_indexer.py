"""Unit tests for high-dimensional archetype vector index.

Validates:
1. Loading 100 canonical superstar profiles into normalized numpy matrices.
2. Gaussian RBF kernel similarity scoring and bounds [0.0, 100.0].
3. Tactical role incompatibility distance penalties.
4. Positional isolation between outfield players and goalkeepers.
5. Sub-5ms query latency verification gate.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from src.models.archetype_indexer import (
    GK_SKILL_COLS,
    OUTFIELD_SKILL_COLS,
    ArchetypeMatchResult,
    ArchetypeVectorIndex,
)


@pytest.fixture(scope="module")
def indexer() -> ArchetypeVectorIndex:
    """Instantiate and load the archetype vector index once for the test module."""
    gallery_path = Path("data/processed/superstar_gallery.json")
    if not gallery_path.exists():
        pytest.skip(f"Superstar gallery {gallery_path} not found.")
    return ArchetypeVectorIndex(gallery_path=gallery_path)


def test_indexer_loads_gallery(indexer: ArchetypeVectorIndex) -> None:
    """Verify indexer loads 100 superstars into normalized outfield and GK matrices."""
    assert indexer.is_loaded is True
    assert len(indexer.gallery) == 100
    assert len(indexer.outfield_metadata) + len(indexer.gk_metadata) == 100

    # Outfield matrix shape and normalization
    assert indexer.outfield_matrix.shape[1] == len(OUTFIELD_SKILL_COLS)
    outfield_norms = np.linalg.norm(indexer.outfield_matrix, axis=1)
    np.testing.assert_allclose(outfield_norms, 1.0, rtol=1e-5)

    # GK matrix shape and normalization
    assert indexer.gk_matrix.shape[1] == len(GK_SKILL_COLS)
    gk_norms = np.linalg.norm(indexer.gk_matrix, axis=1)
    np.testing.assert_allclose(gk_norms, 1.0, rtol=1e-5)


def test_gaussian_rbf_similarity_identity(indexer: ArchetypeVectorIndex) -> None:
    """Verify identical vectors yield exactly 100.0% similarity."""
    zero_dist = np.array([0.0])
    sim = indexer.calculate_rbf_similarity(zero_dist)
    assert round(float(sim[0]), 1) == 100.0


def test_gaussian_rbf_similarity_bounds(indexer: ArchetypeVectorIndex) -> None:
    """Verify similarity values lie strictly within [0.0, 100.0]."""
    test_dists = np.linspace(0.0, 3.0, 50)
    sims = indexer.calculate_rbf_similarity(test_dists)
    assert np.all(sims >= 0.0)
    assert np.all(sims <= 100.0)
    # Monotonically decreasing with distance
    assert np.all(np.diff(sims) <= 0.0)


def test_tactical_role_incompatibility_penalties(indexer: ArchetypeVectorIndex) -> None:
    """Verify tactical sector distance penalties reduce similarity for incompatible roles."""
    # Use Cristiano Ronaldo attributes as query
    cr7_star = next(s for s in indexer.gallery if s["short_name"] == "Cristiano Ronaldo")
    cr7_attrs = cr7_star["attributes"]

    # Match as primary position LW (identical sector)
    matches_lw = indexer.query(
        attributes=cr7_attrs,
        primary_position="LW",
        top_k=5,
        apply_penalties=True,
    )
    assert len(matches_lw) > 0
    top_match = matches_lw[0]
    assert top_match.superstar_name == "Cristiano Ronaldo"
    assert top_match.similarity_score_pct == 100.0
    assert top_match.raw_distance == 0.0
    assert top_match.penalized_distance == 0.0

    # Now simulate a central defender query with the identical attribute vector
    matches_cb = indexer.query(
        attributes=cr7_attrs,
        primary_position="CB",
        top_k=5,
        apply_penalties=True,
    )
    # Finding Cristiano Ronaldo as a CB query should incur a tactical role penalty
    cr7_cb_match = next(m for m in matches_cb if m.superstar_name == "Cristiano Ronaldo")
    assert cr7_cb_match.penalized_distance > cr7_cb_match.raw_distance
    assert cr7_cb_match.similarity_score_pct < top_match.similarity_score_pct


def test_goalkeeper_isolation(indexer: ArchetypeVectorIndex) -> None:
    """Verify goalkeepers query only GK archetypes with 5 skill traits."""
    gk_query = {
        "goalkeeping_diving": 88.0,
        "goalkeeping_handling": 85.0,
        "goalkeeping_kicking": 80.0,
        "goalkeeping_positioning": 90.0,
        "goalkeeping_reflexes": 89.0,
    }
    matches = indexer.query(attributes=gk_query, primary_position="GK", top_k=3)
    assert len(matches) <= 3
    for match in matches:
        assert match.tactical_sector == "Goalkeepers"
        assert match.primary_position == "GK"
        assert isinstance(match, ArchetypeMatchResult)


def test_exclude_player_id_filter(indexer: ArchetypeVectorIndex) -> None:
    """Verify exclude_player_id prevents self-matching."""
    cr7_star = next(s for s in indexer.gallery if s["short_name"] == "Cristiano Ronaldo")
    cr7_id = cr7_star["player_id"]
    matches = indexer.query(
        attributes=cr7_star["attributes"],
        primary_position="LW",
        top_k=3,
        exclude_player_id=cr7_id,
    )
    assert all(m.player_id != cr7_id for m in matches)


def test_verification_gate_sub_5ms_query_latency(indexer: ArchetypeVectorIndex) -> None:
    """Verify Gaussian RBF vector search completes in < 5ms per candidate query."""
    dummy_query = {col: 75.0 for col in OUTFIELD_SKILL_COLS}

    # Warm-up run
    indexer.query(attributes=dummy_query, primary_position="CAM", top_k=3)

    # Benchmark 100 queries
    num_runs = 100
    t0 = time.perf_counter()
    for _ in range(num_runs):
        indexer.query(attributes=dummy_query, primary_position="CAM", top_k=3)
    elapsed_total = time.perf_counter() - t0
    avg_latency_ms = (elapsed_total / num_runs) * 1000.0

    assert avg_latency_ms < 5.0, (
        f"Query latency {avg_latency_ms:.3f}ms exceeded 5.0ms verification budget."
    )
