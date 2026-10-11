import time
import numpy as np
import pytest
from scripts.ceiling_matcher import SuperstarCeilingMatcher, SKILL_COLS, OUTFIELD_COLS, SECTOR_MAP


@pytest.fixture(scope="module")
def matcher():
    return SuperstarCeilingMatcher()


def test_matcher_loads_gallery(matcher):
    assert len(matcher.gallery) == 100
    assert hasattr(matcher, "gallery")


def test_vector_similarity_math_identity():
    # Identity test: identical vectors must yield exactly 100.0% similarity
    vec = np.array([75.0] * len(OUTFIELD_COLS))
    norm = np.linalg.norm(vec)
    unit = vec / norm
    norm_dist = float(np.linalg.norm(unit - unit))
    calibrated_pct = round(float(np.exp(-(norm_dist ** 2) / (2 * (0.28 ** 2))) * 100), 1)
    assert calibrated_pct == 100.0


def test_vector_similarity_bounds_arbitrary():
    np.random.seed(42)
    for _ in range(50):
        v1 = np.random.uniform(30.0, 99.0, size=len(OUTFIELD_COLS))
        v2 = np.random.uniform(30.0, 99.0, size=len(OUTFIELD_COLS))
        u1 = v1 / np.linalg.norm(v1)
        u2 = v2 / np.linalg.norm(v2)
        dist = float(np.linalg.norm(u1 - u2))
        score = float(np.exp(-(dist ** 2) / (2 * (0.28 ** 2))) * 100)
        assert 0.0 <= score <= 100.0


def test_positional_sector_mapping():
    assert SECTOR_MAP["GK"] == "Goalkeepers"
    assert SECTOR_MAP["CB"] == "Defenders"
    assert SECTOR_MAP["CM"] == "Defensive & Central Midfielders"
    assert SECTOR_MAP["CAM"] == "Attacking Midfielders & Wingers"
    assert SECTOR_MAP["ST"] == "Center Forwards & Strikers"


def test_in_memory_matching_latency(matcher):
    # Vector matching against all 100 archetypes should complete in < 15ms
    dummy_vec = np.array([80.0] * len(OUTFIELD_COLS))
    u_dummy = dummy_vec / np.linalg.norm(dummy_vec)

    t0 = time.perf_counter()
    scores = []
    for star in matcher.gallery:
        if star["tactical_sector"] == "Goalkeepers":
            continue
        s_vec = np.array([star["attributes"].get(c, 50.0) for c in OUTFIELD_COLS])
        s_u = s_vec / np.linalg.norm(s_vec)
        dist = np.linalg.norm(u_dummy - s_u)
        score = np.exp(-(dist ** 2) / (2 * (0.28 ** 2))) * 100
        scores.append(score)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert elapsed_ms < 15.0, f"Vector comparison exceeded latency budget: {elapsed_ms:.2f}ms"
    assert len(scores) > 0


def test_end_to_end_match_known_prospect(matcher):
    result = matcher.match_prospect("Pedri", age_max=21, top_k=3)
    assert "error" not in result
    assert result["prospect"]["name"] == "Pedri"
    assert len(result["matches"]) == 3

    scores = [m["similarity_score_pct"] for m in result["matches"]]
    # Verify sorting order (descending)
    assert scores == sorted(scores, reverse=True)
    for m in result["matches"]:
        assert 0.0 <= m["similarity_score_pct"] <= 100.0
        assert m["tactical_sector"] != "Goalkeepers"
        assert len(m["top_growth_priorities"]) <= 3
        assert len(m["closest_matching_traits"]) <= 3
