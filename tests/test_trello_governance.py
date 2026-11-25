"""Unit tests for Trello governance auditing and lane transition enforcement."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from scripts.sync_trello_governance import (
    audit_board_compliance,
    extract_commit_hashes,
    governed_transition_card,
    main,
    validate_lane_transition,
    verify_git_commit_exists,
)


class MockTrelloClient:
    """In-memory Trello client mock for offline governance tests."""

    def __init__(
        self,
        lists: list[dict[str, Any]],
        cards: list[dict[str, Any]],
        checklists: list[dict[str, Any]],
        actions: list[dict[str, Any]],
    ) -> None:
        self.lists = lists
        self.cards = cards
        self.checklists = checklists
        self.actions = actions
        self.moves: list[tuple[str, str]] = []

    def get_lists(self, board_id: str) -> list[dict[str, Any]]:
        return self.lists

    def get_cards(self, board_id: str) -> list[dict[str, Any]]:
        return self.cards

    def get_checklists(self, board_id: str) -> list[dict[str, Any]]:
        return self.checklists

    def get_comment_actions(self, board_id: str) -> list[dict[str, Any]]:
        return self.actions

    def move_card_to_list(self, card_id: str, list_id: str) -> dict[str, Any]:
        self.moves.append((card_id, list_id))
        for card in self.cards:
            if card["id"] == card_id:
                card["idList"] = list_id
        return {"id": card_id, "idList": list_id}


def _sample_lists() -> list[dict[str, Any]]:
    return [
        {"id": "list_backlog", "name": "Backlog", "closed": False},
        {"id": "list_progress", "name": "In Progress", "closed": False},
        {"id": "list_review", "name": "In Review", "closed": False},
        {"id": "list_done", "name": "Done", "closed": False},
    ]


def test_lane_transition_policy() -> None:
    """Assert canonical forward progression is enforced and skips/rollbacks fail."""
    assert validate_lane_transition("Backlog", "In Progress") == (True, [])
    assert validate_lane_transition("To Do", "In Progress") == (True, [])
    assert validate_lane_transition("In Progress", "In Review") == (True, [])
    assert validate_lane_transition("In Review", "Done") == (True, [])
    assert validate_lane_transition("In Review", "In Progress") == (True, [])

    ok_skip, errs_skip = validate_lane_transition("Backlog", "Done")
    assert ok_skip is False
    assert "disallowed_transition:Backlog->Done" in errs_skip

    ok_reopen, errs_reopen = validate_lane_transition("Done", "In Progress")
    assert ok_reopen is False
    assert "disallowed_transition:Done->In Progress" in errs_reopen

    ok_same, errs_same = validate_lane_transition("In Review", "In Review")
    assert ok_same is False
    assert errs_same == ["noop_transition:In Review"]


def test_extract_commit_hashes_ignores_plain_numbers_and_trello_ids() -> None:
    """Confirm hash extraction skips 4-digit years and 24-char Trello IDs."""
    text = (
        "Verified board 6add00820e13a7c7024609e8 in 2026 on commit 60cd10e, "
        "Commit: 9547594, and full hash a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2."
    )
    hashes = extract_commit_hashes(text)
    assert hashes == ["60cd10e", "9547594", "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"]


def test_verify_git_commit_exists_against_temp_repo(tmp_path: Path) -> None:
    """Confirm git commit lookup validates real SHAs and rejects fabricated ones."""
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Amin Al-Ait"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "aminalait.de@gmail.com"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    sample_file = tmp_path / "README.md"
    sample_file.write_text("Analytics FC\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Add initial test file"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert verify_git_commit_exists(sha, repo_root=tmp_path) is True
    assert verify_git_commit_exists("deadbee", repo_root=tmp_path) is False


def test_audit_board_compliance_detects_violations_and_clean_cards() -> None:
    """Verify Done card auditing catches incomplete checklists, tainted text, and missing logs."""
    client = MockTrelloClient(
        lists=_sample_lists(),
        cards=[
            {"id": "c1", "name": "AFC-628: Watchdog", "idList": "list_done", "closed": False},
            {"id": "c2", "name": "AFC-629: Git Hygiene", "idList": "list_done", "closed": False},
            {"id": "c3", "name": "AFC-630: Board Sync", "idList": "list_progress", "closed": False},
        ],
        checklists=[
            {
                "id": "chk1",
                "idCard": "c1",
                "checkItems": [
                    {"id": "i1", "name": "Build module", "state": "complete"},
                    {"id": "i2", "name": "Agent 30 review", "state": "complete"},
                ],
            },
            {
                "id": "chk2",
                "idCard": "c2",
                "checkItems": [
                    {"id": "i3", "name": "Build hook", "state": "complete"},
                    {"id": "i4", "name": "Review hook", "state": "incomplete"},
                ],
            },
        ],
        actions=[
            {
                "type": "commentCard",
                "data": {
                    "card": {"id": "c1"},
                    "text": (
                        "Completed on commit e335210. Pytest passed 4/4. "
                        "Independent review signed off APPROVED."
                    ),
                },
            },
            {
                "type": "commentCard",
                "data": {
                    "card": {"id": "c2"},
                    "text": ("Generated " + "with A" + "I \u2014 missing hash and test metrics."),
                },
            },
        ],
    )

    report = audit_board_compliance("board1", client=client, check_commits=False)
    assert report.compliant is False
    assert report.total_cards_audited == 3
    assert report.done_cards_audited == 2
    assert report.violations_count == 1

    c1_res = next(c for c in report.cards if c.card_id == "c1")
    assert c1_res.compliant is True
    assert c1_res.commit_hashes == ["e335210"]

    c2_res = next(c for c in report.cards if c.card_id == "c2")
    assert c2_res.compliant is False
    assert "incomplete_checklist:1/2" in c2_res.violations
    assert "missing_commit_hash_in_signoff" in c2_res.violations
    assert "missing_reviewer_signoff_in_comment" in c2_res.violations
    assert any("prohibited_em_dash" in v for v in c2_res.violations)
    assert any("prohibited_marker_gen" in v for v in c2_res.violations)


def test_governed_transition_blocks_unready_done_and_allows_compliant() -> None:
    """Confirm Done transition is blocked until checklists and sign-off logs pass."""
    client = MockTrelloClient(
        lists=_sample_lists(),
        cards=[
            {"id": "c_rev", "name": "AFC-630: Governance", "idList": "list_review", "closed": False}
        ],
        checklists=[
            {
                "id": "chk_rev",
                "idCard": "c_rev",
                "checkItems": [{"id": "i1", "name": "Verify rules", "state": "incomplete"}],
            }
        ],
        actions=[],
    )

    blocked = governed_transition_card(
        "AFC-630", "Done", board_id="board1", client=client, check_commits=False
    )
    assert blocked.moved is False
    assert "incomplete_checklist:0/1" in blocked.violations
    assert "missing_signoff_comment" in blocked.violations
    assert client.moves == []

    # Complete checklist and attach clean sign-off comment
    client.checklists[0]["checkItems"][0]["state"] = "complete"
    client.actions.append(
        {
            "type": "commentCard",
            "data": {
                "card": {"id": "c_rev"},
                "text": "Commit abc1234 merged. Pytest passed 6/6. Agent 30 review APPROVED.",
            },
        }
    )

    allowed = governed_transition_card(
        "AFC-630", "Done", board_id="board1", client=client, check_commits=False
    )
    assert allowed.moved is True
    assert allowed.violations == []
    assert client.moves == [("c_rev", "list_done")]


def test_cli_exit_codes_with_mocked_board(monkeypatch: Any, capsys: Any) -> None:
    """Verify CLI emits valid JSON and exits 0 for clean boards, 1 for non-compliant."""
    clean_client = MockTrelloClient(
        lists=_sample_lists(),
        cards=[{"id": "c1", "name": "AFC-630: Clean", "idList": "list_done", "closed": False}],
        checklists=[
            {
                "id": "chk1",
                "idCard": "c1",
                "checkItems": [{"id": "i1", "name": "Unit tests", "state": "complete"}],
            }
        ],
        actions=[
            {
                "type": "commentCard",
                "data": {
                    "card": {"id": "c1"},
                    "text": "Commit deadc0de verified. Pytest passed 5/5. Reviewer APPROVED.",
                },
            }
        ],
    )
    monkeypatch.setattr("scripts.sync_trello_governance.LiveTrelloClient", lambda: clean_client)
    rc_ok = main(["--audit", "--json"])
    payload_ok = json.loads(capsys.readouterr().out)
    assert rc_ok == 0
    assert payload_ok["compliant"] is True

    clean_client.checklists[0]["checkItems"][0]["state"] = "incomplete"
    rc_fail = main(["--audit"])
    out_fail = capsys.readouterr().out
    assert rc_fail == 1
    assert "NON-COMPLIANT" in out_fail

    move_client = MockTrelloClient(
        lists=_sample_lists(),
        cards=[{"id": "c_mv", "name": "AFC-630: Move", "idList": "list_review", "closed": False}],
        checklists=[
            {
                "id": "chk_mv",
                "idCard": "c_mv",
                "checkItems": [{"id": "i1", "name": "Unit tests", "state": "incomplete"}],
            }
        ],
        actions=[],
    )
    monkeypatch.setattr("scripts.sync_trello_governance.LiveTrelloClient", lambda: move_client)
    rc_blocked = main(["--move", "AFC-630", "Done"])
    out_blocked = capsys.readouterr().out
    assert rc_blocked == 1
    assert "BLOCKED:" in out_blocked

    move_client.checklists[0]["checkItems"][0]["state"] = "complete"
    move_client.actions.append(
        {
            "type": "commentCard",
            "data": {
                "card": {"id": "c_mv"},
                "text": "Commit abc1234 merged. Pytest passed 6/6. Reviewer APPROVED.",
            },
        }
    )
    rc_moved = main(["--move", "AFC-630", "Done", "--json"])
    payload_moved = json.loads(capsys.readouterr().out)
    assert rc_moved == 0
    assert payload_moved["moved"] is True
