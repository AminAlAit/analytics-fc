"""Plain publishing steps used by the editorial flow.

The Prefect flow calls these functions. Tests can call them directly or
through task.fn() without starting a Prefect server.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

THERMAL_PAUSE_SECONDS = 2.0


def pause(sleep: Callable[[float], None] | None = None) -> None:
    """Thermal pause between publishing steps."""
    sleeper = sleep if sleep is not None else time.sleep
    sleeper(THERMAL_PAUSE_SECONDS)


def extract_spotlight(payload: dict[str, Any]) -> dict[str, Any]:
    """Pull the player identity required for a spotlight."""
    player_id = payload.get("player_id")
    name = payload.get("player_name")
    if player_id is None or not str(name or "").strip():
        raise ValueError("Spotlight payload needs player_id and player_name.")
    return {"player_id": int(player_id), "player_name": str(name)}


def render_card_payload(spotlight: dict[str, Any]) -> dict[str, Any]:
    """Describe the card to render. Image bytes stay with the card generator."""
    return {
        "layout": "landscape",
        "player_name": spotlight["player_name"],
        "player_id": spotlight["player_id"],
    }


def draft_caption(spotlight: dict[str, Any]) -> str:
    """One-line caption using only the supplied name."""
    return f"{spotlight['player_name']} spotlight"


def dispatch_post(caption: str, *, client: Any = None) -> dict[str, Any]:
    """Send the caption when a client is supplied. Otherwise record a skip."""
    if client is None:
        return {"status": "skipped", "text": caption}
    created = client.create_post(caption)
    return {"status": "posted", "text": caption, "response": created}


def run_publishing_sequence(
    payload: dict[str, Any],
    *,
    sleep: Callable[[float], None] | None = None,
    client: Any = None,
) -> dict[str, Any]:
    """Run extract, render, copy, and dispatch with a pause between each step."""
    sleeper = sleep if sleep is not None else time.sleep
    spotlight = extract_spotlight(payload)
    sleeper(THERMAL_PAUSE_SECONDS)
    card = render_card_payload(spotlight)
    sleeper(THERMAL_PAUSE_SECONDS)
    caption = draft_caption(spotlight)
    sleeper(THERMAL_PAUSE_SECONDS)
    result = dispatch_post(caption, client=client)
    return {
        "spotlight": spotlight,
        "card": card,
        "caption": caption,
        "result": result,
    }
