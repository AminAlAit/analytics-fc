"""Dual-polygon radar contracts: finite coordinates and client-backed matches."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from dashboard.components.tactical_radar import (
    build_radar_from_ceiling,
    build_tactical_radar_altair,
    dual_radar_coordinates,
    finite_score,
    normalize_macro_radar,
)

OUTFIELD = ["Pace", "Shooting", "Passing", "Dribbling", "Defending", "Physical"]
GOALKEEPER = ["Diving", "Handling", "Kicking", "Reflexes", "Positioning", "Reactions"]


def _assert_finite_frame(frame: pd.DataFrame) -> None:
    assert frame[["x", "y", "value"]].isna().sum().sum() == 0
    assert np.isfinite(frame[["x", "y", "value"]].to_numpy(dtype=float)).all()


def test_finite_score_replaces_nulls() -> None:
    """Null, NaN, infinity, and junk text become the neutral score."""
    assert finite_score(None) == 50.0
    assert finite_score(float("nan")) == 50.0
    assert finite_score(float("inf")) == 50.0
    assert finite_score("n/a") == 50.0
    assert finite_score(0) == 0.0
    assert finite_score(87.4) == 87.4


def test_dual_polygon_has_no_null_coordinates() -> None:
    """Prospect and benchmark polygons stay finite when inputs contain nulls."""
    frame = dual_radar_coordinates(
        OUTFIELD,
        [91, None, float("nan"), 70, 40, "bad"],
        [88, 90, 85, float("inf"), 60, 77],
        prospect_name="Haaland",
        benchmark_name="Ronaldo",
    )
    _assert_finite_frame(frame)
    assert set(frame["series"]) == {"Haaland", "Ronaldo"}
    assert len(frame) == 14
    assert frame.groupby("series").size().tolist() == [7, 7]
    closed = frame[frame["order"] == 6]
    assert len(closed) == 2


def test_chart_layers_keep_finite_polygon_data() -> None:
    """Altair layers for both polygons expose only finite x and y values."""
    chart = build_tactical_radar_altair(
        GOALKEEPER,
        [80, None, 70, 65, 60, 75],
        [90, 88, 84, 86, 82, 80],
        p_name="Courtois",
        s_name="Neuer",
    )
    spec = chart.to_dict()
    assert spec["width"] == "container"
    assert spec["height"] == 420
    encoded = str(spec)
    assert "#10B981" in encoded
    assert "#06B6D4" in encoded

    polygon_frames = [
        layer.data
        for layer in chart.layer
        if isinstance(layer.data, pd.DataFrame) and {"x", "y", "value"}.issubset(layer.data.columns)
    ]
    assert len(polygon_frames) == 4
    for frame in polygon_frames:
        _assert_finite_frame(frame)


def test_normalize_macro_radar_keeps_gap_and_replaces_null_axis() -> None:
    """Macro rows from match_ceiling become comparison rows without null scores."""
    categories, prospect, benchmark, rows = normalize_macro_radar(
        [
            {
                "dimension": "Pace",
                "prospect_value": None,
                "benchmark_value": 90,
                "gap": None,
            },
            {"dimension": "Shooting", "prospect_value": 88, "benchmark_value": 91, "gap": 3},
            {"dimension": "Passing", "prospect_value": 70, "benchmark_value": 80, "gap": 10},
        ]
    )
    assert categories == ["Pace", "Shooting", "Passing"]
    assert prospect[0] == 50.0
    assert rows[0]["delta"] == 40.0
    assert rows[1]["delta"] == 3.0
    frame = dual_radar_coordinates(categories, prospect, benchmark)
    _assert_finite_frame(frame)


def test_length_mismatch_and_short_axis_list_raise() -> None:
    """Uneven series and fewer than three axes are rejected before drawing."""
    with pytest.raises(ValueError):
        dual_radar_coordinates(["Pace", "Shooting"], [80, 70], [90])
    with pytest.raises(ValueError):
        build_tactical_radar_altair(["Pace", "Shooting"], [80, 70], [90, 85])


def test_build_radar_from_ceiling_uses_client_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """The component asks api_client.match_ceiling for the radar payload."""
    calls: list[tuple[int, int, int]] = []

    def _fake_match(player_id: int, edition_year: int, top_k: int = 3) -> dict[str, Any]:
        calls.append((player_id, edition_year, top_k))
        return {
            "prospect": {"short_name": "Haaland"},
            "matches": [{"name": "Lewandowski"}],
            "macro_radar": [
                {
                    "dimension": name,
                    "prospect_value": 80 if index else None,
                    "benchmark_value": 90,
                    "gap": 10,
                }
                for index, name in enumerate(OUTFIELD)
            ],
        }

    monkeypatch.setattr("dashboard.components.tactical_radar.match_ceiling", _fake_match)
    chart = build_radar_from_ceiling(239085, 2022, top_k=3)
    assert calls == [(239085, 2022, 3)]
    assert chart is not None
    frames = [
        layer.data
        for layer in chart.layer
        if isinstance(layer.data, pd.DataFrame) and "value" in layer.data.columns
    ]
    for frame in frames:
        _assert_finite_frame(frame)


def test_build_radar_from_ceiling_returns_none_without_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing ceiling match does not draw an empty chart."""
    monkeypatch.setattr(
        "dashboard.components.tactical_radar.match_ceiling",
        lambda *_args, **_kwargs: None,
    )
    assert build_radar_from_ceiling(1, 2022) is None


def test_live_embedded_ceiling_radar_has_six_finite_axes() -> None:
    """Embedded match_ceiling produces a six-axis radar with finite coordinates."""
    chart = build_radar_from_ceiling(239085, 2022, top_k=3)
    if chart is None:
        pytest.skip("Embedded ceiling match is unavailable.")
    frames = [
        layer.data
        for layer in chart.layer
        if isinstance(layer.data, pd.DataFrame) and {"x", "y", "value"}.issubset(layer.data.columns)
    ]
    assert frames
    for frame in frames:
        _assert_finite_frame(frame)
        assert frame["dimension"].nunique() == 6
