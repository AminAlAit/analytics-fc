"""Atomic spotlight ledger.

Blocks a repeat player inside 90 days and blocks a post once the weekly
quota is full. Writes land through a temp file and os.replace.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def _utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


class ComplianceLedger:
    """Persistent publish history."""

    def __init__(
        self,
        path: Path | str,
        *,
        window_days: int = 90,
        weekly_quota: int = 7,
    ) -> None:
        if window_days < 1:
            raise ValueError("window_days must be at least 1.")
        if weekly_quota < 1:
            raise ValueError("weekly_quota must be at least 1.")
        self.path = Path(path)
        self.window_days = window_days
        self.weekly_quota = weekly_quota

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        entries = raw.get("entries", [])
        return list(entries)

    def _save(self, entries: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"entries": entries}, indent=2)
        fd, tmp_name = tempfile.mkstemp(prefix="ledger-", suffix=".json", dir=str(self.path.parent))
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            tmp_path.write_text(payload, encoding="utf-8")
            os.replace(tmp_path, self.path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def evaluate(self, player_id: int, moment: datetime) -> dict[str, Any]:
        """Return whether a spotlight is allowed and why."""
        now = _utc(moment)
        entries = self._load()
        window_start = now - timedelta(days=self.window_days)
        week_start = now - timedelta(days=now.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        recent_same = 0
        week_count = 0
        for entry in entries:
            published = _utc(datetime.fromisoformat(str(entry["published_at"])))
            if published >= week_start:
                week_count += 1
            if int(entry["player_id"]) == int(player_id) and published >= window_start:
                recent_same += 1
        if recent_same:
            return {"allowed": False, "reason": "duplicate_within_window", "week_count": week_count}
        if week_count >= self.weekly_quota:
            return {"allowed": False, "reason": "weekly_quota", "week_count": week_count}
        return {"allowed": True, "reason": "ok", "week_count": week_count}

    def record(self, player_id: int, moment: datetime) -> dict[str, Any]:
        """Append a spotlight when the guard allows it."""
        decision = self.evaluate(player_id, moment)
        if not decision["allowed"]:
            raise ValueError(str(decision["reason"]))
        entries = self._load()
        entries.append(
            {
                "player_id": int(player_id),
                "published_at": _utc(moment).isoformat(),
            }
        )
        self._save(entries)
        return decision
