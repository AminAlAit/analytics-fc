"""Weekly editorial flow.

pipelines/flows.py already owns the master pipeline, so this module stays
beside it instead of adding a pipelines/flows package that would shadow that
file. Tasks are thin wrappers. Call task.fn() in tests so Prefect does not
boot a server.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from prefect import flow, task

from pipelines.tasks.publish_task import (
    dispatch_post,
    draft_caption,
    extract_spotlight,
    render_card_payload,
    run_publishing_sequence,
)
from src.publisher.compliance_guard import ComplianceLedger


@task(name="extract-spotlight")
def extract_spotlight_task(payload: dict[str, Any]) -> dict[str, Any]:
    """Task wrapper around spotlight extraction."""
    return extract_spotlight(payload)


@task(name="render-card")
def render_card_task(
    spotlight: dict[str, Any],
    *,
    dest: Path | str | None = None,
    layout: str | None = None,
    retina: bool = True,
) -> dict[str, Any]:
    """Task wrapper around card payload rendering."""
    return render_card_payload(spotlight, dest=dest, layout=layout, retina=retina)


@task(name="draft-caption")
def draft_caption_task(
    spotlight: dict[str, Any],
    *,
    complete: Callable[..., str] | None = None,
    model: str | None = None,
) -> str:
    """Task wrapper around caption drafting."""
    return draft_caption(spotlight, complete=complete, model=model)


@task(name="dispatch-post")
def dispatch_post_task(
    caption: str,
    *,
    client: Any = None,
    media_path: Path | str | None = None,
    player_id: int | None = None,
    ledger: ComplianceLedger | None = None,
    moment: datetime | None = None,
) -> dict[str, Any]:
    """Task wrapper around post dispatch."""
    return dispatch_post(
        caption,
        client=client,
        media_path=media_path,
        player_id=player_id,
        ledger=ledger,
        moment=moment,
    )


@flow(name="publishing-flow")
def publishing_flow(payload: dict[str, Any]) -> dict[str, Any]:
    """Sequential publishing flow with thermal pauses between steps."""
    return run_publishing_sequence(payload)
