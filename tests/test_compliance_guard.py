"""Duplicate window, weekly quota, and atomic ledger writes."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.publisher.compliance_guard import ComplianceLedger

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)


def test_blocks_duplicate_inside_90_days(tmp_path: Path) -> None:
    ledger = ComplianceLedger(tmp_path / "ledger.json")
    ledger.record(239085, NOW)
    with pytest.raises(ValueError, match="duplicate_within_window"):
        ledger.record(239085, NOW + timedelta(days=89))
    decision = ledger.evaluate(239085, NOW + timedelta(days=91))
    assert decision["allowed"] is True


def test_weekly_quota_blocks_eighth_player(tmp_path: Path) -> None:
    ledger = ComplianceLedger(tmp_path / "ledger.json", weekly_quota=7)
    for player_id in range(7):
        ledger.record(player_id, NOW + timedelta(hours=player_id))
    decision = ledger.evaluate(99, NOW + timedelta(hours=8))
    assert decision["allowed"] is False
    assert decision["reason"] == "weekly_quota"


def test_replace_leaves_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    ledger = ComplianceLedger(path)
    ledger.record(1, NOW)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["entries"][0]["player_id"] == 1
    assert list(path.parent.glob("ledger-*.json")) == []
