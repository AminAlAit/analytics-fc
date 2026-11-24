"""Pitch-dark social cards at standard and 2x retina sizes."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

CANVAS_SPECS: dict[str, tuple[int, int]] = {
    "landscape": (1200, 675),
    "square": (1080, 1080),
}
NULL_SCORE = 50.0


def finite_score(value: Any, default: float = NULL_SCORE) -> float:
    """Return a finite attribute score."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def render_social_card(
    payload: dict[str, Any],
    dest: Path | str,
    *,
    layout: str = "landscape",
    retina: bool = True,
) -> dict[str, Any]:
    """Draw a radar card. Retina output is twice the logical canvas on each edge."""
    if layout not in CANVAS_SPECS:
        raise ValueError(f"Unknown card layout: {layout}")
    started = time.perf_counter()
    logical_w, logical_h = CANVAS_SPECS[layout]
    scale = 2 if retina else 1
    width, height = logical_w * scale, logical_h * scale
    image = Image.new("RGB", (width, height), "#0A0E17")
    draw = ImageDraw.Draw(image)
    player_name = str(payload.get("player_name") or "Prospect")
    benchmark_name = str(payload.get("benchmark_name") or "Benchmark")
    draw.text((32 * scale, 28 * scale), "ANALYTICS FC", fill="#10B981")
    draw.text((32 * scale, 64 * scale), player_name, fill="#F8FAFC")
    draw.text((32 * scale, 108 * scale), f"vs {benchmark_name}", fill="#94A3B8")

    radar = list(payload.get("radar") or [])
    if radar:
        center_x = int(width * 0.72)
        center_y = int(height * 0.58)
        radius = int(min(width, height) * 0.28)
        count = len(radar)
        prospect_points: list[tuple[float, float]] = []
        benchmark_points: list[tuple[float, float]] = []
        for index, row in enumerate(radar):
            angle = index * 2 * math.pi / count - math.pi / 2
            prospect = finite_score(row.get("prospect_value")) / 100.0
            benchmark = finite_score(row.get("benchmark_value")) / 100.0
            prospect_points.append(
                (
                    center_x + radius * prospect * math.cos(angle),
                    center_y + radius * prospect * math.sin(angle),
                )
            )
            benchmark_points.append(
                (
                    center_x + radius * benchmark * math.cos(angle),
                    center_y + radius * benchmark * math.sin(angle),
                )
            )
        draw.polygon(benchmark_points, outline="#06B6D4")
        draw.polygon(prospect_points, outline="#10B981")

    forecast = list(payload.get("forecast") or [])
    if forecast:
        median = finite_score(forecast[-1].get("q50", forecast[-1].get("q50_overall")), default=0.0)
        floor = finite_score(forecast[-1].get("q10", forecast[-1].get("q10_overall")), default=0.0)
        ceiling = finite_score(
            forecast[-1].get("q90", forecast[-1].get("q90_overall")), default=0.0
        )
        draw.text(
            (32 * scale, height - 72 * scale),
            f"Q10 {floor:.0f}   Q50 {median:.0f}   Q90 {ceiling:.0f}",
            fill="#F8FAFC",
        )

    target = Path(dest)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="PNG")
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "path": str(target),
        "layout": layout,
        "logical_size": (logical_w, logical_h),
        "pixel_size": image.size,
        "retina": retina,
        "elapsed_ms": elapsed_ms,
        "size_bytes": target.stat().st_size,
    }


def render_card_pair(
    payload: dict[str, Any], directory: Path | str, *, retina: bool = True
) -> dict[str, Any]:
    """Render landscape and square cards into one directory."""
    root = Path(directory)
    started = time.perf_counter()
    landscape = render_social_card(
        payload, root / "card_landscape.png", layout="landscape", retina=retina
    )
    square = render_social_card(payload, root / "card_square.png", layout="square", retina=retina)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {"landscape": landscape, "square": square, "elapsed_ms": elapsed_ms}
