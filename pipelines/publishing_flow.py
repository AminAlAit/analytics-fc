"""Weekly editorial flow.

pipelines/flows.py already owns the master pipeline, so this module stays
beside it instead of adding a pipelines/flows package that would shadow that
file. Tasks are thin wrappers. Call task.fn() in tests so Prefect does not
boot a server.
"""

from __future__ import annotations

from typing import Any

from prefect import flow, task

from pipelines.tasks.publish_task import run_publishing_sequence


@task(name="extract-spotlight")
def extract_spotlight_task(payload: dict[str, Any]) -> dict[str, Any]:
    """Task wrapper around spotlight extraction."""
    from pipelines.tasks.publish_task import extract_spotlight

    return extract_spotlight(payload)


@task(name="render-card")
def render_card_task(spotlight: dict[str, Any]) -> dict[str, Any]:
    """Task wrapper around card payload rendering."""
    from pipelines.tasks.publish_task import render_card_payload

    return render_card_payload(spotlight)


@task(name="draft-caption")
def draft_caption_task(spotlight: dict[str, Any]) -> str:
    """Task wrapper around caption drafting."""
    from pipelines.tasks.publish_task import draft_caption

    return draft_caption(spotlight)


@task(name="dispatch-post")
def dispatch_post_task(caption: str) -> dict[str, Any]:
    """Task wrapper around post dispatch."""
    from pipelines.tasks.publish_task import dispatch_post

    return dispatch_post(caption)


@flow(name="publishing-flow")
def publishing_flow(payload: dict[str, Any]) -> dict[str, Any]:
    """Sequential publishing flow with thermal pauses between steps."""
    return run_publishing_sequence(payload)
