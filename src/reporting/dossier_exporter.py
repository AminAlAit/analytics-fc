"""Executive PDF and PNG dossier sheets.

Chart and table inputs are plain records. Live matches, when requested, go
through dashboard.api_client. PDF output uses the matplotlib Agg backend, so
no browser binary is required.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw

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


def _radar_pairs(radar: list[dict[str, Any]]) -> list[tuple[str, float, float]]:
    pairs: list[tuple[str, float, float]] = []
    for row in radar:
        label = str(row.get("dimension") or "Axis")
        pairs.append(
            (
                label,
                finite_score(row.get("prospect_value")),
                finite_score(row.get("benchmark_value")),
            )
        )
    return pairs


def _forecast_rows(forecast: list[dict[str, Any]]) -> list[tuple[int, float, float, float]]:
    rows: list[tuple[int, float, float, float]] = []
    for point in forecast:
        rows.append(
            (
                int(point.get("year") or point.get("target_edition_year") or 0),
                finite_score(point.get("q10", point.get("q10_overall")), default=0.0),
                finite_score(point.get("q50", point.get("q50_overall")), default=0.0),
                finite_score(point.get("q90", point.get("q90_overall")), default=0.0),
            )
        )
    return rows


def export_dossier_pdf(payload: dict[str, Any], dest: Path | str) -> dict[str, Any]:
    """Write a one-page executive PDF and return path, bytes, and elapsed time."""
    started = time.perf_counter()
    target = Path(dest)
    target.parent.mkdir(parents=True, exist_ok=True)
    player_name = str(payload.get("player_name") or "Prospect")
    benchmark_name = str(payload.get("benchmark_name") or "Prime Benchmark")
    edition_year = int(payload.get("edition_year") or 0)
    overall = finite_score(payload.get("overall"), default=0.0)
    potential = finite_score(payload.get("potential"), default=0.0)
    age = int(finite_score(payload.get("age"), default=0.0))
    radar = _radar_pairs(list(payload.get("radar") or []))
    forecast = _forecast_rows(list(payload.get("forecast") or []))

    figure = plt.figure(figsize=(8.27, 11.69), dpi=100, facecolor="#0A0E17")
    figure.suptitle(
        f"Analytics FC  |  {player_name}",
        color="#F8FAFC",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    header = figure.add_axes((0.08, 0.88, 0.84, 0.06))
    header.set_axis_off()
    header.set_facecolor("#0A0E17")
    header.text(
        0.0,
        0.5,
        (
            f"FIFA {edition_year}   Age {age}   "
            f"Overall {overall:.0f}   Potential {potential:.0f}   vs {benchmark_name}"
        ),
        color="#10B981",
        fontsize=10,
        va="center",
        fontfamily="monospace",
    )

    if radar:
        labels = [item[0] for item in radar]
        prospect_vals = [item[1] for item in radar]
        benchmark_vals = [item[2] for item in radar]
        positions = list(range(len(labels)))
        axes = figure.add_axes((0.1, 0.48, 0.8, 0.36))
        axes.set_facecolor("#111827")
        axes.bar(
            [pos - 0.18 for pos in positions],
            prospect_vals,
            width=0.36,
            color="#10B981",
            label=player_name,
        )
        axes.bar(
            [pos + 0.18 for pos in positions],
            benchmark_vals,
            width=0.36,
            color="#06B6D4",
            label=benchmark_name,
        )
        axes.set_ylim(0, 100)
        axes.set_xticks(positions, labels, color="#94A3B8")
        axes.tick_params(colors="#94A3B8")
        axes.set_title("Tactical comparison", color="#F8FAFC", loc="left")
        axes.legend(facecolor="#111827", edgecolor="#1E293B", labelcolor="#F8FAFC")
        for spine in axes.spines.values():
            spine.set_color("#1E293B")

    if forecast:
        years = [row[0] for row in forecast]
        axes = figure.add_axes((0.1, 0.12, 0.8, 0.28))
        axes.set_facecolor("#111827")
        axes.fill_between(
            years,
            [row[1] for row in forecast],
            [row[3] for row in forecast],
            color="#10B981",
            alpha=0.25,
        )
        axes.plot(years, [row[2] for row in forecast], color="#10B981", linewidth=2)
        axes.set_ylim(40, 99)
        axes.tick_params(colors="#94A3B8")
        axes.set_title("Trajectory median and Q10 to Q90", color="#F8FAFC", loc="left")
        for spine in axes.spines.values():
            spine.set_color("#1E293B")

    with PdfPages(target) as pdf:
        pdf.savefig(figure, facecolor=figure.get_facecolor())
    plt.close(figure)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    data = target.read_bytes()
    return {
        "path": str(target),
        "size_bytes": len(data),
        "elapsed_ms": elapsed_ms,
        "header": data[:5],
    }


def export_dossier_png(payload: dict[str, Any], dest: Path | str) -> dict[str, Any]:
    """Write a pitch-dark PNG summary sheet."""
    started = time.perf_counter()
    target = Path(dest)
    target.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1200, 1600), "#0A0E17")
    draw = ImageDraw.Draw(image)
    player_name = str(payload.get("player_name") or "Prospect")
    benchmark_name = str(payload.get("benchmark_name") or "Prime Benchmark")
    draw.text((48, 48), "ANALYTICS FC", fill="#10B981")
    draw.text((48, 96), player_name, fill="#F8FAFC")
    draw.text(
        (48, 150),
        (
            f"FIFA {int(payload.get('edition_year') or 0)}  |  "
            f"OVR {finite_score(payload.get('overall'), default=0.0):.0f}  |  "
            f"vs {benchmark_name}"
        ),
        fill="#94A3B8",
    )
    top = 240
    for label, prospect, benchmark in _radar_pairs(list(payload.get("radar") or [])):
        draw.text((48, top), f"{label}", fill="#F8FAFC")
        draw.rectangle((280, top, 280 + prospect * 3, top + 18), fill="#10B981")
        draw.rectangle((620, top, 620 + benchmark * 3, top + 18), fill="#06B6D4")
        draw.text((280, top + 24), f"{prospect:.0f} vs {benchmark:.0f}", fill="#94A3B8")
        top += 70
    image.save(target, format="PNG")
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "path": str(target),
        "size_bytes": target.stat().st_size,
        "elapsed_ms": elapsed_ms,
        "size": image.size,
    }


def export_dossier_from_client(
    player_id: int,
    edition_year: int,
    *,
    pdf_path: Path | str,
    png_path: Path | str | None = None,
    horizon_years: int = 3,
) -> dict[str, Any]:
    """Build a dossier from match_ceiling and forecast_trajectory, then export it."""
    from dashboard.api_client import forecast_trajectory, match_ceiling

    match = match_ceiling(player_id, edition_year, top_k=1) or {}
    forecast = forecast_trajectory(
        player_id,
        edition_year,
        horizon_years=horizon_years,
        include_conformal=False,
    )
    prospect = match.get("prospect") or {}
    matches = match.get("matches") or []
    benchmark_name = str(matches[0]["name"]) if matches else "Prime Benchmark"
    payload = {
        "player_name": prospect.get("short_name") or f"Player {player_id}",
        "edition_year": edition_year,
        "overall": prospect.get("overall"),
        "potential": prospect.get("potential"),
        "age": prospect.get("age"),
        "benchmark_name": benchmark_name,
        "radar": match.get("macro_radar") or [],
        "forecast": [
            {
                "year": point.get("target_edition_year"),
                "q10": point.get("q10_overall"),
                "q50": point.get("q50_overall"),
                "q90": point.get("q90_overall"),
            }
            for point in (forecast or {}).get("forecast", [])
        ],
    }
    pdf_info = export_dossier_pdf(payload, pdf_path)
    png_info = export_dossier_png(payload, png_path) if png_path is not None else None
    return {"payload": payload, "pdf": pdf_info, "png": png_info}
