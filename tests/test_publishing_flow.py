"""Publishing sequence pauses and task.fn() execution without a Prefect server."""

from __future__ import annotations

from pipelines.publishing_flow import (
    draft_caption_task,
    extract_spotlight_task,
    render_card_task,
)
from pipelines.tasks.publish_task import THERMAL_PAUSE_SECONDS, run_publishing_sequence


def test_sequence_pauses_two_seconds_between_steps() -> None:
    pauses: list[float] = []
    result = run_publishing_sequence(
        {"player_id": 239085, "player_name": "E. Haaland"},
        sleep=pauses.append,
    )
    assert THERMAL_PAUSE_SECONDS == 2.0
    assert pauses == [2.0, 2.0, 2.0]
    assert result["spotlight"]["player_name"] == "E. Haaland"
    assert result["card"]["layout"] == "landscape"
    assert result["caption"] == "E. Haaland spotlight"
    assert result["result"]["status"] == "skipped"


def test_tasks_execute_through_fn() -> None:
    spotlight = extract_spotlight_task.fn({"player_id": 7, "player_name": "Saka"})
    card = render_card_task.fn(spotlight)
    caption = draft_caption_task.fn(spotlight)
    assert card["player_id"] == 7
    assert caption == "Saka spotlight"


def test_missing_name_raises() -> None:
    try:
        run_publishing_sequence({"player_id": 1}, sleep=lambda _seconds: None)
    except ValueError as exc:
        assert "player_name" in str(exc)
    else:
        raise AssertionError("missing name should raise")
