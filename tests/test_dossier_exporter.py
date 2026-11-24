"""Executive dossier export contracts."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.reporting.dossier_exporter import (
    export_dossier_from_client,
    export_dossier_pdf,
    export_dossier_png,
    finite_score,
)

SAMPLE = {
    "player_name": "E. Haaland",
    "benchmark_name": "R. Lewandowski",
    "edition_year": 2022,
    "overall": 88,
    "potential": 94,
    "age": 21,
    "radar": [
        {"dimension": "Pace", "prospect_value": None, "benchmark_value": 90},
        {"dimension": "Shooting", "prospect_value": 91, "benchmark_value": 92},
        {"dimension": "Passing", "prospect_value": float("nan"), "benchmark_value": 75},
        {"dimension": "Physical", "prospect_value": 86, "benchmark_value": 84},
    ],
    "forecast": [
        {"year": 2022, "q10": 88, "q50": 88, "q90": 88},
        {"year": 2023, "q10": 87, "q50": 89, "q90": 92},
        {"year": 2024, "q10": 86, "q50": 90, "q90": 93},
    ],
}


def test_finite_score_replaces_nulls() -> None:
    assert finite_score(None) == 50.0
    assert finite_score(float("inf")) == 50.0
    assert finite_score(76) == 76.0


def test_pdf_compiles_under_two_seconds(tmp_path: Path) -> None:
    """PDF bytes start with the PDF header and compile inside 2 seconds."""
    dest = tmp_path / "dossier.pdf"
    info = export_dossier_pdf(SAMPLE, dest)
    assert info["elapsed_ms"] < 2000.0
    data = dest.read_bytes()
    assert data.startswith(b"%PDF")
    assert info["size_bytes"] == len(data)


def test_png_sheet_size_and_mode(tmp_path: Path) -> None:
    dest = tmp_path / "dossier.png"
    info = export_dossier_png(SAMPLE, dest)
    image = Image.open(dest)
    assert image.size == (1200, 1600)
    assert image.mode == "RGB"
    assert info["elapsed_ms"] < 2000.0
    assert dest.stat().st_size > 1000


def test_empty_payload_still_writes_files(tmp_path: Path) -> None:
    pdf_path = tmp_path / "empty.pdf"
    png_path = tmp_path / "empty.png"
    export_dossier_pdf({}, pdf_path)
    export_dossier_png({}, png_path)
    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert Image.open(png_path).size == (1200, 1600)


def test_client_export_uses_match_and_forecast(tmp_path: Path, monkeypatch) -> None:
    """The live helper calls the dashboard client and writes both files."""

    def _match(player_id: int, edition_year: int, top_k: int = 3) -> dict:
        assert player_id == 239085
        assert edition_year == 2022
        assert top_k == 1
        return {
            "prospect": {
                "short_name": "E. Haaland",
                "overall": 88,
                "potential": 94,
                "age": 21,
            },
            "matches": [{"name": "R. Lewandowski"}],
            "macro_radar": [
                {"dimension": "Pace", "prospect_value": 89, "benchmark_value": 78},
                {"dimension": "Shooting", "prospect_value": 91, "benchmark_value": 92},
                {"dimension": "Physical", "prospect_value": 86, "benchmark_value": 84},
            ],
        }

    def _forecast(*args: object, **kwargs: object) -> dict:
        assert args[0] == 239085
        return {
            "forecast": [
                {
                    "target_edition_year": 2023,
                    "q10_overall": 87,
                    "q50_overall": 89,
                    "q90_overall": 92,
                }
            ]
        }

    monkeypatch.setattr("dashboard.api_client.match_ceiling", _match)
    monkeypatch.setattr("dashboard.api_client.forecast_trajectory", _forecast)
    result = export_dossier_from_client(
        239085,
        2022,
        pdf_path=tmp_path / "client.pdf",
        png_path=tmp_path / "client.png",
    )
    assert result["payload"]["benchmark_name"] == "R. Lewandowski"
    assert (tmp_path / "client.pdf").read_bytes().startswith(b"%PDF")
    assert Image.open(tmp_path / "client.png").size == (1200, 1600)
