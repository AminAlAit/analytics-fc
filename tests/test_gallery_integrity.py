import json
from pathlib import Path

import pytest

GALLERY_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "superstar_gallery.json"
)

REQUIRED_SKILL_COLS = [
    "attacking_crossing",
    "attacking_finishing",
    "attacking_heading_accuracy",
    "attacking_short_passing",
    "attacking_volleys",
    "skill_dribbling",
    "skill_curve",
    "skill_fk_accuracy",
    "skill_long_passing",
    "skill_ball_control",
    "movement_acceleration",
    "movement_sprint_speed",
    "movement_agility",
    "movement_reactions",
    "movement_balance",
    "power_shot_power",
    "power_jumping",
    "power_stamina",
    "power_strength",
    "power_long_shots",
    "mentality_aggression",
    "mentality_interceptions",
    "mentality_positioning",
    "mentality_vision",
    "mentality_penalties",
    "mentality_composure",
    "defending_marking_awareness",
    "defending_standing_tackle",
    "defending_sliding_tackle",
    "goalkeeping_diving",
    "goalkeeping_handling",
    "goalkeeping_kicking",
    "goalkeeping_positioning",
    "goalkeeping_reflexes",
]

VALID_SECTORS = {
    "Goalkeepers",
    "Defenders",
    "Defensive & Central Midfielders",
    "Attacking Midfielders & Wingers",
    "Center Forwards & Strikers",
}


@pytest.fixture(scope="module")
def gallery_data():
    assert GALLERY_PATH.exists(), f"Gallery artifact missing at {GALLERY_PATH}"
    with open(GALLERY_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data


def test_gallery_cardinality_and_uniqueness(gallery_data):
    assert len(gallery_data) == 100, f"Expected exactly 100 superstars, found {len(gallery_data)}"
    player_ids = [star["player_id"] for star in gallery_data]
    assert len(player_ids) == len(set(player_ids)), "Duplicate player IDs found in gallery"


def test_gallery_required_fields_and_sectors(gallery_data):
    for star in gallery_data:
        assert "player_id" in star
        assert "short_name" in star and len(star["short_name"]) > 0
        assert "tactical_sector" in star and star["tactical_sector"] in VALID_SECTORS
        assert "peak_overall" in star and star["peak_overall"] >= 85
        assert "attributes" in star and isinstance(star["attributes"], dict)


def test_gallery_attribute_completeness_and_bounds(gallery_data):
    for star in gallery_data:
        attrs = star["attributes"]
        for col in REQUIRED_SKILL_COLS:
            assert col in attrs, f"Missing attribute {col} for {star['short_name']}"
            val = attrs[col]
            assert val is not None, f"Null attribute {col} for {star['short_name']}"
            assert 1.0 <= float(val) <= 99.0, (
                f"Attribute {col} out of range ({val}) for {star['short_name']}"
            )


def test_gallery_sector_representation(gallery_data):
    sector_counts = {}
    for star in gallery_data:
        sec = star["tactical_sector"]
        sector_counts[sec] = sector_counts.get(sec, 0) + 1

    assert sector_counts.get("Goalkeepers", 0) >= 5, "Insufficient goalkeeper archetypes"
    assert sector_counts.get("Defenders", 0) >= 15, "Insufficient defender archetypes"
    assert sector_counts.get("Center Forwards & Strikers", 0) >= 20, (
        "Insufficient striker archetypes"
    )
