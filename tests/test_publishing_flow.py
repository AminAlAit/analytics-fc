"""Publishing sequence pauses, wired publisher integration, and task.fn() execution."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from pipelines.publishing_flow import (
    dispatch_post_task,
    draft_caption_task,
    extract_spotlight_task,
    render_card_task,
)
from pipelines.tasks.publish_task import THERMAL_PAUSE_SECONDS, run_publishing_sequence
from src.publisher.compliance_guard import ComplianceLedger
from src.publisher.x_client import XClient


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
    skipped = dispatch_post_task.fn(caption)
    assert card["player_id"] == 7
    assert caption == "Saka spotlight"
    assert skipped["status"] == "skipped"


def test_missing_name_raises() -> None:
    try:
        run_publishing_sequence({"player_id": 1}, sleep=lambda _seconds: None)
    except ValueError as exc:
        assert "player_name" in str(exc)
    else:
        raise AssertionError("missing name should raise")


def test_wired_sequence_renders_drafts_uploads_and_enforces_compliance(tmp_path: Path) -> None:
    """Exercise card generator, grounded copy engine, XClient media upload, and ledger."""
    calls: list[tuple[str, str]] = []

    def _mock_transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
    ) -> tuple[int, dict[str, Any], dict[str, str]]:
        calls.append((method, url))
        if url.endswith("/initialize"):
            return 200, {"data": {"id": "media-99"}}, {}
        if url.endswith("/tweets"):
            return 201, {"data": {"id": "post-42"}}, {}
        return 200, {"ok": True}, {}

    client = XClient("tok-123", transport=_mock_transport, sleep=lambda _s: None)
    ledger = ComplianceLedger(tmp_path / "ledger.json")
    moment = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    card_path = tmp_path / "haaland_card.png"

    payload: dict[str, Any] = {
        "player_id": 239085,
        "player_name": "E. Haaland",
        "benchmark_name": "R. Lewandowski",
        "stats": {"player": "E. Haaland", "age": 21, "overall": 88},
        "radar": [
            {"dimension": "Pace", "prospect_value": 89, "benchmark_value": 80},
            {"dimension": "Shooting", "prospect_value": 91, "benchmark_value": 92},
            {"dimension": "Passing", "prospect_value": 65, "benchmark_value": 79},
        ],
    }

    def _fake_complete(_prompt: str, *, model: str) -> str:
        assert model == "hetzner-flash"
        return "E. Haaland is 21 \u2014 rated 88 overall."

    res = run_publishing_sequence(
        payload,
        sleep=lambda _s: None,
        client=client,
        card_dest=card_path,
        complete=_fake_complete,
        ledger=ledger,
        moment=moment,
    )

    assert card_path.exists()
    with Image.open(card_path) as img:
        assert img.size == (2400, 1350)
    assert res["caption"] == "E. Haaland is 21 ,  rated 88 overall."
    assert res["result"]["status"] == "posted"
    assert res["result"]["media_ids"] == ["media-99"]
    assert len(calls) == 4  # INIT, APPEND, FINALIZE, POST /tweets

    # Second run inside 90-day window must be blocked without invoking XClient
    calls.clear()
    blocked = run_publishing_sequence(
        payload,
        sleep=lambda _s: None,
        client=client,
        card_dest=card_path,
        complete=_fake_complete,
        ledger=ledger,
        moment=moment,
    )
    assert blocked["result"]["status"] == "blocked"
    assert blocked["result"]["reason"] == "duplicate_within_window"
    assert calls == []
