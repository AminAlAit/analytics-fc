"""Unit tests for cross-source player entity resolution engine."""

import pandas as pd
import pytest

from src.data.entity_resolution import (
    PlayerEntityResolver,
    build_composite_key,
    normalize_string,
)


def test_normalize_string():
    """Verify string normalization strips accents, punctuation, and casing."""
    assert normalize_string("Martin Ødegaard") == "martin odegaard"
    assert normalize_string("Luka Modrić") == "luka modric"
    assert normalize_string("Thomas Müller") == "thomas muller"
    assert normalize_string("  Kylian   Mbappé!  ") == "kylian mbappe"
    assert normalize_string(None) == ""


def test_build_composite_key():
    """Verify composite key joins normalized name, dob, and nationality."""
    key = build_composite_key("Lionel Messi", "1987-06-24", "Argentina")
    assert key == "lionel messi|1987-06-24|argentina"


@pytest.fixture
def sample_catalog():
    """Build a sample reference catalog dataframe."""
    return pd.DataFrame(
        [
            {
                "player_id": 158023,
                "short_name": "L. Messi",
                "long_name": "Lionel Andrés Messi Cuccittini",
                "dob": "1987-06-24",
                "nationality_name": "Argentina",
                "fifa_version": 24,
            },
            {
                "player_id": 252371,
                "short_name": "J. Bellingham",
                "long_name": "Jude Bellingham",
                "dob": "2003-06-29",
                "nationality_name": "England",
                "fifa_version": 24,
            },
            {
                "player_id": 20801,
                "short_name": "Cristiano Ronaldo",
                "long_name": "Cristiano Ronaldo dos Santos Aveiro",
                "dob": "1985-02-05",
                "nationality_name": "Portugal",
                "fifa_version": 24,
            },
        ]
    )


def test_exact_composite_resolution(sample_catalog):
    """Verify exact composite match achieves 1.0 confidence."""
    resolver = PlayerEntityResolver()
    resolver.index_catalog(sample_catalog)

    match = resolver.resolve_player(name="Jude Bellingham", dob="2003-06-29", nationality="England")
    assert match is not None
    assert match.player_id == 252371
    assert match.confidence == 1.00
    assert match.match_type == "EXACT_COMPOSITE"


def test_fuzzy_name_resolution(sample_catalog):
    """Verify typo / accented name resolves via fuzzy fallback."""
    resolver = PlayerEntityResolver()
    resolver.index_catalog(sample_catalog)

    # Minor misspelling "Jood Bellingham"
    match = resolver.resolve_player(name="Jude Bellinghem")
    assert match is not None
    assert match.player_id == 252371
    assert match.confidence >= 0.85
    assert match.match_type == "FUZZY_NAME_SIMILARITY"


def test_unknown_player_returns_none(sample_catalog):
    """Verify uncataloged player query cleanly returns None."""
    resolver = PlayerEntityResolver()
    resolver.index_catalog(sample_catalog)

    match = resolver.resolve_player(name="Completely Unknown Player XYZ")
    assert match is None
