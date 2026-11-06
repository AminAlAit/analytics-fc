"""Unit tests for Pydantic V2 data contracts and schema validation."""

import pandas as pd
import pytest
from pydantic import ValidationError

from src.data.contracts import (
    PlayerSeasonRecord,
    validate_season_dataframe,
)


def test_player_season_record_valid():
    """Verify clean instantiation of valid player record."""
    data = {
        "player_id": 252371,
        "fifa_version": 24,
        "short_name": "J. Bellingham",
        "age": 20,
        "overall": 86,
        "potential": 91,
        "pace": 76.0,
        "shooting": 78.0,
        "passing": 80.0,
        "dribbling": 85.0,
        "defending": 78.0,
        "physic": 82.0,
        "value_eur": 100500000.0,
        "wage_eur": 175000.0,
    }
    rec = PlayerSeasonRecord.model_validate(data)
    assert rec.player_id == 252371
    assert rec.overall == 86
    assert rec.potential == 91
    assert rec.short_name == "J. Bellingham"


def test_player_season_record_bounds():
    """Verify attribute bounds [35, 99] and age bounds are strictly enforced."""
    base = {
        "player_id": 100,
        "fifa_version": 24,
        "short_name": "Test Player",
        "age": 25,
        "overall": 75,
        "potential": 80,
    }

    # Invalid overall < 35
    with pytest.raises(ValidationError):
        PlayerSeasonRecord.model_validate({**base, "overall": 34})

    # Invalid overall > 99
    with pytest.raises(ValidationError):
        PlayerSeasonRecord.model_validate({**base, "overall": 100})

    # Invalid age < 15
    with pytest.raises(ValidationError):
        PlayerSeasonRecord.model_validate({**base, "age": 14})

    # Invalid age > 55
    with pytest.raises(ValidationError):
        PlayerSeasonRecord.model_validate({**base, "age": 56})


def test_player_season_record_empty_name():
    """Verify empty or whitespace short_name is rejected."""
    base = {
        "player_id": 100,
        "fifa_version": 24,
        "short_name": "   ",
        "age": 25,
        "overall": 75,
        "potential": 80,
    }
    with pytest.raises(ValidationError):
        PlayerSeasonRecord.model_validate(base)


def test_validate_season_dataframe_success():
    """Verify dataframe batch validator on valid dataset."""
    df = pd.DataFrame(
        [
            {
                "player_id": 1,
                "fifa_version": 23,
                "short_name": "L. Messi",
                "age": 35,
                "overall": 91,
                "potential": 91,
                "pace": 81.0,
            },
            {
                "player_id": 2,
                "fifa_version": 24,
                "short_name": "K. De Bruyne",
                "age": 32,
                "overall": 91,
                "potential": 91,
                "pace": 72.0,
            },
        ]
    )
    summary = validate_season_dataframe(df)
    assert summary.status == "PASSED"
    assert summary.valid_records == 2
    assert summary.invalid_records == 0


def test_validate_season_dataframe_detects_errors():
    """Verify dataframe batch validator catches out-of-bound rows."""
    df = pd.DataFrame(
        [
            {
                "player_id": 1,
                "fifa_version": 24,
                "short_name": "Valid",
                "age": 22,
                "overall": 80,
                "potential": 85,
            },
            {
                "player_id": 2,
                "fifa_version": 24,
                "short_name": "Invalid",
                "age": 22,
                "overall": 105,  # Out of bounds
                "potential": 85,
            },
        ]
    )
    summary = validate_season_dataframe(df)
    assert summary.status == "FAILED"
    assert summary.valid_records == 1
    assert summary.invalid_records == 1
    assert len(summary.error_samples) == 1
