"""Publishing pipeline step functions wired to src.publisher modules.

The Prefect flow calls these functions. Tests can call them directly or
through task.fn() without starting a Prefect server.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.publisher.card_generator import render_social_card
from src.publisher.compliance_guard import ComplianceLedger
from src.publisher.copy_engine import DEFAULT_MODEL, draft_scouting_copy

THERMAL_PAUSE_SECONDS = 2.0


def pause(sleep: Callable[[float], None] | None = None) -> None:
    """Thermal pause between publishing steps."""
    sleeper = sleep if sleep is not None else time.sleep
    sleeper(THERMAL_PAUSE_SECONDS)


def extract_spotlight(payload: dict[str, Any]) -> dict[str, Any]:
    """Pull player identity and optional spotlight payload fields."""
    player_id = payload.get("player_id")
    name = payload.get("player_name")
    if player_id is None or not str(name or "").strip():
        raise ValueError("Spotlight payload needs player_id and player_name.")
    spotlight: dict[str, Any] = {
        "player_id": int(player_id),
        "player_name": str(name),
    }
    for key in ("stats", "radar", "forecast", "benchmark_name", "layout"):
        if key in payload and payload[key] is not None:
            spotlight[key] = payload[key]
    return spotlight


def render_card_payload(
    spotlight: dict[str, Any],
    *,
    dest: Path | str | None = None,
    layout: str | None = None,
    retina: bool = True,
) -> dict[str, Any]:
    """Render a retina PNG card when dest is given, else describe metadata."""
    resolved_layout = str(layout or spotlight.get("layout") or "landscape")
    result: dict[str, Any] = {
        "layout": resolved_layout,
        "player_name": spotlight["player_name"],
        "player_id": spotlight["player_id"],
    }
    if dest is not None:
        rendered = render_social_card(spotlight, dest, layout=resolved_layout, retina=retina)
        result.update(rendered)
    return result


def draft_caption(
    spotlight: dict[str, Any],
    *,
    complete: Callable[..., str] | None = None,
    model: str | None = None,
) -> str:
    """Draft grounded copy via copy_engine when complete is supplied, else return a one-line note."""
    if complete is None:
        return f"{spotlight['player_name']} spotlight"
    stats_payload = spotlight.get("stats")
    if isinstance(stats_payload, dict) and stats_payload:
        stats = dict(stats_payload)
    else:
        stats = {"player_id": spotlight["player_id"], "player_name": spotlight["player_name"]}
    active_model = model if model is not None else DEFAULT_MODEL
    return draft_scouting_copy(stats, model=active_model, complete=complete)


def dispatch_post(
    caption: str,
    *,
    client: Any = None,
    media_path: Path | str | None = None,
    player_id: int | None = None,
    ledger: ComplianceLedger | None = None,
    moment: datetime | None = None,
) -> dict[str, Any]:
    """Check compliance ledger, upload optional card media, and send post via client."""
    now = moment if moment is not None else datetime.now(UTC)
    if ledger is not None and player_id is not None:
        decision = ledger.evaluate(int(player_id), now)
        if not decision["allowed"]:
            return {
                "status": "blocked",
                "reason": decision["reason"],
                "text": caption,
            }

    if client is None:
        return {"status": "skipped", "text": caption}

    media_ids: list[str] | None = None
    if media_path is not None and hasattr(client, "upload_media"):
        raw_bytes = Path(media_path).read_bytes()
        media_id = str(client.upload_media(raw_bytes))
        media_ids = [media_id]

    if media_ids is not None:
        created = client.create_post(caption, media_ids=media_ids)
    else:
        created = client.create_post(caption)

    if ledger is not None and player_id is not None:
        ledger.record(int(player_id), now)

    return {
        "status": "posted",
        "text": caption,
        "media_ids": media_ids,
        "response": created,
    }


def run_publishing_sequence(
    payload: dict[str, Any],
    *,
    sleep: Callable[[float], None] | None = None,
    client: Any = None,
    card_dest: Path | str | None = None,
    layout: str | None = None,
    retina: bool = True,
    complete: Callable[..., str] | None = None,
    model: str | None = None,
    ledger: ComplianceLedger | None = None,
    moment: datetime | None = None,
) -> dict[str, Any]:
    """Run extract, render, copy, compliance check, and dispatch with pauses between steps."""
    sleeper = sleep if sleep is not None else time.sleep
    spotlight = extract_spotlight(payload)
    sleeper(THERMAL_PAUSE_SECONDS)
    card = render_card_payload(spotlight, dest=card_dest, layout=layout, retina=retina)
    sleeper(THERMAL_PAUSE_SECONDS)
    caption = draft_caption(spotlight, complete=complete, model=model)
    sleeper(THERMAL_PAUSE_SECONDS)
    result = dispatch_post(
        caption,
        client=client,
        media_path=card.get("path"),
        player_id=spotlight["player_id"],
        ledger=ledger,
        moment=moment,
    )
    return {
        "spotlight": spotlight,
        "card": card,
        "caption": caption,
        "result": result,
    }
