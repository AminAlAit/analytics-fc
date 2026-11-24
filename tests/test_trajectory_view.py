"""Fan chart contracts for the 10th to 90th percentile trajectory cone."""

from __future__ import annotations

import time
from typing import Any

import altair as alt

from dashboard.components.trajectory_view import (
    build_forecast_fan,
    build_trajectory_fan_chart,
    build_trajectory_fan_data,
)


def _sample_forecast(*, include_conformal: bool) -> dict[str, Any]:
    points = []
    for step, year in enumerate((2023, 2024, 2025), start=1):
        point: dict[str, Any] = {
            "target_edition_year": year,
            "projected_age": 21 + step,
            "predicted_growth": 1.2,
            "q10_overall": 80 + step,
            "q50_overall": 84 + step,
            "q90_overall": 88 + step,
        }
        if include_conformal:
            point["conformal_lower"] = 79 + step
            point["conformal_upper"] = 90 + step
        points.append(point)
    return {
        "reference_edition_year": 2022,
        "current_age": 21,
        "current_overall": 82,
        "potential": 91,
        "forecast": points,
    }


def test_fan_data_orders_quantile_cone() -> None:
    """Each step keeps the floor under the median and the median under the ceiling."""
    frame = build_trajectory_fan_data(_sample_forecast(include_conformal=False))
    assert len(frame) == 4
    assert int(frame.iloc[0]["year"]) == 2022
    assert (frame["q10"] <= frame["q50"]).all()
    assert (frame["q50"] <= frame["q90"]).all()
    assert frame["conformal_lower"].isna().all()


def test_fan_chart_cone_renders_under_100ms() -> None:
    """The 10th to 90th area layer compiles inside the chart budget."""
    payload = _sample_forecast(include_conformal=True)
    build_trajectory_fan_chart(payload)
    started = time.perf_counter()
    chart = build_trajectory_fan_chart(payload)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    spec = chart.to_dict()

    assert isinstance(chart, alt.LayerChart)
    assert elapsed_ms < 100.0
    encoded = str(spec)
    assert "q10" in encoded
    assert "q90" in encoded
    assert "q50" in encoded
    assert "#10B981" in encoded


def test_build_forecast_fan_uses_client(monkeypatch) -> None:
    """The view asks forecast_trajectory instead of loading models itself."""
    calls: list[dict[str, Any]] = []

    def _fake_forecast(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return _sample_forecast(include_conformal=kwargs.get("include_conformal", True))

    monkeypatch.setattr("dashboard.components.trajectory_view.forecast_trajectory", _fake_forecast)
    payload, chart = build_forecast_fan(
        239085,
        2022,
        horizon_years=2,
        include_conformal=False,
        quantile_model="quantile",
        conformal_model=None,
    )
    assert payload is not None
    assert chart is not None
    assert calls[0]["player_id"] == 239085
    assert calls[0]["horizon_years"] == 2
    assert calls[0]["include_conformal"] is False
    assert calls[0]["quantile_model"] == "quantile"
