"""Dual-polygon Altair radar for prospect versus prime benchmark attributes.

Inference stays in dashboard.api_client.match_ceiling. This module only shapes
already returned macro radar rows into finite coordinates.
"""

from __future__ import annotations

import math
from typing import Any

import altair as alt
import numpy as np
import pandas as pd

from dashboard.api_client import match_ceiling

NULL_SCORE = 50.0
MIN_AXES = 3


def finite_score(value: Any, default: float = NULL_SCORE) -> float:
    """Return a finite attribute score, replacing nulls and non-numeric values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def normalize_macro_radar(
    rows: list[dict[str, Any]],
) -> tuple[list[str], list[float], list[float], list[dict[str, Any]]]:
    """Turn match_ceiling macro_radar rows into finite axes and gap rows."""
    categories: list[str] = []
    prospect_values: list[float] = []
    benchmark_values: list[float] = []
    comparison: list[dict[str, Any]] = []
    for row in rows:
        label = str(row.get("dimension") or "Axis")
        prospect = finite_score(row.get("prospect_value"))
        benchmark = finite_score(row.get("benchmark_value"))
        gap = finite_score(row.get("gap"), default=round(benchmark - prospect, 1))
        categories.append(label)
        prospect_values.append(prospect)
        benchmark_values.append(benchmark)
        comparison.append(
            {
                "dimension": label,
                "p_val": round(prospect, 1),
                "s_val": round(benchmark, 1),
                "delta": round(gap, 1),
            }
        )
    return categories, prospect_values, benchmark_values, comparison


def dual_radar_coordinates(
    categories: list[str],
    prospect_values: list[float],
    benchmark_values: list[float],
    prospect_name: str = "Prospect",
    benchmark_name: str = "Prime Benchmark",
) -> pd.DataFrame:
    """Closed polygon coordinates for both series. Every x and y value is finite."""
    if len(categories) < MIN_AXES:
        raise ValueError(f"Radar requires at least {MIN_AXES} dimensions.")
    if not (len(categories) == len(prospect_values) == len(benchmark_values)):
        raise ValueError("Radar categories and both value series must have equal length.")

    count = len(categories)
    angles = [index * 2 * math.pi / count - math.pi / 2 for index in range(count)]
    cleaned_prospect = [finite_score(value) for value in prospect_values]
    cleaned_benchmark = [finite_score(value) for value in benchmark_values]

    points: list[dict[str, Any]] = []
    for series_name, values in (
        (prospect_name, cleaned_prospect),
        (benchmark_name, cleaned_benchmark),
    ):
        for index, (label, value, angle) in enumerate(zip(categories, values, angles)):
            points.append(
                {
                    "series": series_name,
                    "dimension": label,
                    "value": value,
                    "order": index,
                    "x": value * math.cos(angle),
                    "y": value * math.sin(angle),
                }
            )
        points.append(
            {
                "series": series_name,
                "dimension": categories[0],
                "value": values[0],
                "order": count,
                "x": values[0] * math.cos(angles[0]),
                "y": values[0] * math.sin(angles[0]),
            }
        )

    frame = pd.DataFrame(points)
    if not np.isfinite(frame[["x", "y", "value"]].to_numpy(dtype=float)).all():
        raise ValueError("Radar coordinates contain a null or non-finite value.")
    return frame


def build_tactical_radar_altair(
    categories: list[str],
    p_vals: list[float],
    s_vals: list[float],
    p_name: str = "Prospect",
    s_name: str = "Prime Benchmark",
) -> alt.LayerChart:
    """Render a responsive pitch-dark radar with prospect and benchmark polygons."""
    points = dual_radar_coordinates(categories, p_vals, s_vals, p_name, s_name)
    count = len(categories)
    angles = [index * 2 * math.pi / count - math.pi / 2 for index in range(count)]

    grid_rows: list[dict[str, Any]] = []
    for level in (20, 40, 60, 80, 100):
        for angle in angles:
            grid_rows.append(
                {"x": level * math.cos(angle), "y": level * math.sin(angle), "level": str(level)}
            )
        grid_rows.append(
            {
                "x": level * math.cos(angles[0]),
                "y": level * math.sin(angles[0]),
                "level": str(level),
            }
        )
    grid_chart = (
        alt.Chart(pd.DataFrame(grid_rows))
        .mark_line(color="#223147", strokeWidth=1, strokeDash=[3, 3])
        .encode(
            x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-150, 150])),
            y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-150, 150])),
            detail="level:N",
        )
    )

    spoke_rows: list[dict[str, Any]] = []
    for index, angle in enumerate(angles):
        spoke_rows.append({"x": 0.0, "y": 0.0, "spoke": str(index)})
        spoke_rows.append(
            {"x": 100 * math.cos(angle), "y": 100 * math.sin(angle), "spoke": str(index)}
        )
    spoke_chart = (
        alt.Chart(pd.DataFrame(spoke_rows))
        .mark_line(color="#1E293B", strokeWidth=1)
        .encode(x="x:Q", y="y:Q", detail="spoke:N")
    )

    prospect = points[points["series"] == p_name]
    benchmark = points[points["series"] == s_name]
    prospect_line = (
        alt.Chart(prospect)
        .mark_line(color="#10B981", strokeWidth=3)
        .encode(x="x:Q", y="y:Q", order="order:Q")
    )
    prospect_points = (
        alt.Chart(prospect)
        .mark_circle(color="#10B981", size=60)
        .encode(x="x:Q", y="y:Q", tooltip=["dimension:N", "value:Q"])
    )
    benchmark_line = (
        alt.Chart(benchmark)
        .mark_line(color="#06B6D4", strokeWidth=2.5, strokeDash=[4, 4])
        .encode(x="x:Q", y="y:Q", order="order:Q")
    )
    benchmark_points = (
        alt.Chart(benchmark)
        .mark_circle(color="#06B6D4", size=50)
        .encode(x="x:Q", y="y:Q", tooltip=["dimension:N", "value:Q"])
    )

    label_rows = []
    for label, prospect_value, benchmark_value, angle in zip(
        categories,
        [finite_score(value) for value in p_vals],
        [finite_score(value) for value in s_vals],
        angles,
    ):
        label_rows.append(
            {
                "x": 122 * math.cos(angle),
                "y": 122 * math.sin(angle),
                "label": (f"{label}: {int(round(prospect_value))} | {int(round(benchmark_value))}"),
            }
        )
    labels_chart = (
        alt.Chart(pd.DataFrame(label_rows))
        .mark_text(color="#94A3B8", fontSize=11, fontWeight=700)
        .encode(x="x:Q", y="y:Q", text="label:N")
    )

    chart = (
        grid_chart
        + spoke_chart
        + prospect_line
        + prospect_points
        + benchmark_line
        + benchmark_points
        + labels_chart
    ).properties(height=420, width="container")
    chart = chart.configure_view(strokeWidth=0).configure(background="#0A0E17")
    return chart


def build_radar_from_ceiling(
    player_id: int,
    edition_year: int,
    *,
    top_k: int = 3,
) -> alt.LayerChart | None:
    """Load a ceiling match through the dashboard client and draw its radar."""
    payload = match_ceiling(player_id, edition_year, top_k=top_k)
    if not payload:
        return None
    macro_radar = payload.get("macro_radar") or []
    if len(macro_radar) < MIN_AXES:
        return None
    categories, prospect_values, benchmark_values, _comparison = normalize_macro_radar(macro_radar)
    matches = payload.get("matches") or []
    prospect = payload.get("prospect") or {}
    prospect_name = str(prospect.get("short_name") or "Prospect")
    benchmark_name = str(matches[0]["name"]) if matches else "Prime Benchmark"
    return build_tactical_radar_altair(
        categories,
        prospect_values,
        benchmark_values,
        p_name=prospect_name,
        s_name=benchmark_name,
    )
