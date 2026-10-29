"""Automated verification suite to prevent architectural drift between LikeC4 and Trello."""

import re
from pathlib import Path

import pytest

from architecture.scripts.sync_status import (
    BOARD_ID,
    COMPONENT_TICKET_MAP,
    MODEL_FILE,
    VIEWS_FILE,
    fetch_trello_cards,
    load_trello_credentials,
)


def test_architecture_files_exist():
    """Verify core LikeC4 files exist and are populated."""
    assert MODEL_FILE.exists(), f"Missing {MODEL_FILE}"
    assert VIEWS_FILE.exists(), f"Missing {VIEWS_FILE}"
    assert len(MODEL_FILE.read_text(encoding="utf-8")) > 1000
    assert len(VIEWS_FILE.read_text(encoding="utf-8")) > 500


def test_mapped_components_in_model():
    """Verify all 28 tracked components exist in model.c4 with tags, links, and metadata."""
    content = MODEL_FILE.read_text(encoding="utf-8")

    for comp_name, ticket_id in COMPONENT_TICKET_MAP.items():
        assert comp_name in content, f"Component '{comp_name}' missing in model.c4"

        # Check tag exists
        tag_match = re.search(rf"{comp_name}\s*=\s*(?:component|container)\s*'[^']+'\s*\{{\s*(#[a-z_]+)", content)
        assert tag_match is not None, f"Status tag missing for '{comp_name}'"
        assert tag_match.group(1) in ["#built", "#in_progress", "#backlog"], f"Invalid tag for '{comp_name}'"

        # Find block bounds for accurate content inspection
        decl_idx = content.find(f"{comp_name} =")
        assert decl_idx != -1
        # Check link and metadata within the next 500 characters
        comp_slice = content[decl_idx:decl_idx + 600]
        assert "link https://trello.com/c/" in comp_slice, f"Trello link missing for '{comp_name}'"
        assert f"ticket '{ticket_id}'" in comp_slice, f"Metadata ticket '{ticket_id}' missing for '{comp_name}'"


def test_view_links_configured():
    """Verify views contain top-level navigation links to the master Trello board."""
    views_content = VIEWS_FILE.read_text(encoding="utf-8")
    assert f"https://trello.com/b/{BOARD_ID}" in views_content


def test_trello_live_status_parity():
    """Verify live parity between Trello card states and LikeC4 model tags."""
    try:
        cards_by_ticket, counts = fetch_trello_cards()
    except Exception as exc:
        pytest.skip(f"Trello API credentials unavailable or network offline: {exc}")

    content = MODEL_FILE.read_text(encoding="utf-8")
    discrepancies = []

    for comp_name, ticket_id in COMPONENT_TICKET_MAP.items():
        card = cards_by_ticket.get(ticket_id)
        if not card:
            continue

        tag_match = re.search(rf"{comp_name}\s*=\s*(?:component|container)\s*'[^']+'\s*\{{\s*(#[a-z_]+)", content)
        if not tag_match:
            discrepancies.append(f"{comp_name} ({ticket_id}): missing tag in model.c4")
            continue

        current_tag = tag_match.group(1)
        if current_tag != card.status_tag:
            discrepancies.append(
                f"{comp_name} ({ticket_id}): model has {current_tag}, but Trello is in list '{card.list_name}' ({card.status_tag})"
            )

    assert not discrepancies, "Architectural drift detected:\n" + "\n".join(discrepancies)
