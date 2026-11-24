"""Social card size, retina scale, and timing contracts."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.publisher.card_generator import render_card_pair, render_social_card

PAYLOAD = {
    "player_name": "E. Haaland",
    "benchmark_name": "R. Lewandowski",
    "radar": [
        {"dimension": "Pace", "prospect_value": None, "benchmark_value": 80},
        {"dimension": "Shooting", "prospect_value": 91, "benchmark_value": 92},
        {"dimension": "Passing", "prospect_value": float("nan"), "benchmark_value": 75},
        {"dimension": "Physical", "prospect_value": 88, "benchmark_value": 84},
    ],
    "forecast": [{"q10": 86, "q50": 90, "q90": 93}],
}


def test_retina_pair_sizes_and_speed(tmp_path: Path) -> None:
    """2x retina cards are twice the logical canvas and finish inside 800ms."""
    info = render_card_pair(PAYLOAD, tmp_path, retina=True)
    assert info["elapsed_ms"] < 800.0
    landscape = Image.open(tmp_path / "card_landscape.png")
    square = Image.open(tmp_path / "card_square.png")
    assert landscape.size == (2400, 1350)
    assert square.size == (2160, 2160)
    assert info["landscape"]["logical_size"] == (1200, 675)
    assert info["square"]["logical_size"] == (1080, 1080)


def test_logical_canvas_without_retina(tmp_path: Path) -> None:
    info = render_social_card(PAYLOAD, tmp_path / "card.png", layout="landscape", retina=False)
    assert Image.open(info["path"]).size == (1200, 675)
    square = render_social_card(PAYLOAD, tmp_path / "square.png", layout="square", retina=False)
    assert square["pixel_size"] == (1080, 1080)


def test_unknown_layout_raises(tmp_path: Path) -> None:
    try:
        render_social_card(PAYLOAD, tmp_path / "card.png", layout="story")
    except ValueError as exc:
        assert "story" in str(exc)
    else:
        raise AssertionError("Unknown layout should raise ValueError")


def test_empty_payload_still_renders(tmp_path: Path) -> None:
    info = render_social_card({}, tmp_path / "empty.png", retina=False)
    assert Image.open(info["path"]).size == (1200, 675)
    assert info["size_bytes"] > 500
