"""Trello board lane transition and governance compliance auditor.

Validates Kanban lane progressions (Backlog / To Do -> In Progress -> In Review
-> Done), verifies 100 percent checklist completion on Done cards, enforces clean
sign-off logs containing valid Git commit hashes and test verification notes,
and scans comments for prohibited markers via scripts.verify_git_hygiene.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

try:
    from scripts.trello_sync import BOARD_ID, trello_api
    from scripts.verify_git_hygiene import scan_text_for_violations
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.modules.pop("scripts", None)
    from scripts.trello_sync import BOARD_ID, trello_api
    from scripts.verify_git_hygiene import scan_text_for_violations

REPO_ROOT = Path(__file__).resolve().parent.parent

# Canonical allowed Kanban lane transitions
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "backlog": frozenset({"in progress"}),
    "to do": frozenset({"in progress"}),
    "in progress": frozenset({"in review", "to do", "backlog"}),
    "in review": frozenset({"done", "in progress"}),
    "done": frozenset(),
}

_COMMIT_HASH_RE = re.compile(
    r"\b(?![0-9]+\b)(?:[0-9a-f]{7,12}|[0-9a-f]{40})\b",
    re.IGNORECASE,
)
_TEST_EVIDENCE_RE = re.compile(
    r"\b(?:test(?:s|ed|ing)?|pytest|ruff|mypy|passed|verified|verification|check(?:s|ed)?)\b",
    re.IGNORECASE,
)
_REVIEW_EVIDENCE_RE = re.compile(
    r"\b(?:review(?:ed|er)?|approved|signed\s*off|sign-off|agent\s*30|purist)\b",
    re.IGNORECASE,
)


class TrelloClient(Protocol):
    """Protocol abstracting Trello board queries and list moves for testability."""

    def get_lists(self, board_id: str) -> list[dict[str, Any]]: ...

    def get_cards(self, board_id: str) -> list[dict[str, Any]]: ...

    def get_checklists(self, board_id: str) -> list[dict[str, Any]]: ...

    def get_comment_actions(self, board_id: str) -> list[dict[str, Any]]: ...

    def move_card_to_list(self, card_id: str, list_id: str) -> dict[str, Any]: ...


class LiveTrelloClient:
    """Default Trello API client delegating to scripts.trello_sync.trello_api."""

    def get_lists(self, board_id: str) -> list[dict[str, Any]]:
        """Fetch active lists for a Trello board."""
        return cast(list[dict[str, Any]], trello_api("GET", f"boards/{board_id}/lists"))

    def get_cards(self, board_id: str) -> list[dict[str, Any]]:
        """Fetch open cards for a Trello board."""
        return cast(list[dict[str, Any]], trello_api("GET", f"boards/{board_id}/cards"))

    def get_checklists(self, board_id: str) -> list[dict[str, Any]]:
        """Fetch all checklists on a Trello board in one request."""
        return cast(list[dict[str, Any]], trello_api("GET", f"boards/{board_id}/checklists"))

    def get_comment_actions(self, board_id: str) -> list[dict[str, Any]]:
        """Fetch recent card comment actions on a Trello board."""
        return cast(
            list[dict[str, Any]],
            trello_api(
                "GET",
                f"boards/{board_id}/actions",
                params={"filter": "commentCard", "limit": 1000},
            ),
        )

    def move_card_to_list(self, card_id: str, list_id: str) -> dict[str, Any]:
        """Move a Trello card to the target list ID at top position."""
        return cast(
            dict[str, Any],
            trello_api("PUT", f"cards/{card_id}", params={"idList": list_id, "pos": "top"}),
        )


@dataclass(frozen=True)
class CardAuditResult:
    """Governance audit outcome for a single Trello card."""

    card_id: str
    card_name: str
    lane: str
    compliant: bool
    checklist_completed: int
    checklist_total: int
    commit_hashes: list[str] = field(default_factory=list)
    verified_hashes: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BoardAuditReport:
    """Aggregated compliance audit report across a Trello board."""

    board_id: str
    compliant: bool
    total_cards_audited: int
    done_cards_audited: int
    violations_count: int
    cards: list[CardAuditResult] = field(default_factory=list)


@dataclass(frozen=True)
class TransitionResult:
    """Outcome of a governed card lane transition request."""

    moved: bool
    card_id: str
    card_name: str
    from_lane: str
    to_lane: str
    violations: list[str] = field(default_factory=list)


def normalize_lane_name(lane_name: str) -> str:
    """Normalize Trello lane name for rule comparisons."""
    return " ".join(lane_name.strip().lower().split())


def validate_lane_transition(from_lane: str, to_lane: str) -> tuple[bool, list[str]]:
    """Validate whether transitioning a card between Kanban lanes follows project policy."""
    src = normalize_lane_name(from_lane)
    dst = normalize_lane_name(to_lane)
    if src not in ALLOWED_TRANSITIONS:
        return False, [f"unknown_source_lane:{from_lane}"]
    if dst not in ALLOWED_TRANSITIONS:
        return False, [f"unknown_target_lane:{to_lane}"]
    if src == dst:
        return False, [f"noop_transition:{from_lane}"]
    if dst not in ALLOWED_TRANSITIONS[src]:
        return False, [f"disallowed_transition:{from_lane}->{to_lane}"]
    return True, []


def extract_commit_hashes(text: str) -> list[str]:
    """Extract unique candidate Git commit SHAs (7-12 or 40 hex chars) from text."""
    seen: list[str] = []
    for match in _COMMIT_HASH_RE.findall(text):
        lower = match.lower()
        if lower not in seen:
            seen.append(lower)
    return seen


def verify_git_commit_exists(commit_hash: str, repo_root: Path | None = None) -> bool:
    """Verify whether a commit SHA resolves to an existing commit in the Git repository."""
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", commit_hash):
        return False
    cwd = repo_root if repo_root is not None else REPO_ROOT
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{commit_hash}^{{commit}}"],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def validate_checklist_completion(
    checklists: list[dict[str, Any]],
) -> tuple[int, int, list[str]]:
    """Verify every item across all checklists on a card is marked complete."""
    total = 0
    completed = 0
    for chk in checklists:
        for item in chk.get("checkItems", []):
            total += 1
            if str(item.get("state", "")).lower() == "complete":
                completed += 1
    violations: list[str] = []
    if total == 0:
        violations.append("missing_acceptance_checklist")
    elif completed < total:
        violations.append(f"incomplete_checklist:{completed}/{total}")
    return completed, total, violations


def validate_comment_texts(
    comment_texts: list[str],
    *,
    require_signoff: bool,
    check_commits: bool = False,
    repo_root: Path | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Audit card comment logs for clean text, commit SHAs, test reports, and review approval."""
    violations: list[str] = []
    all_hashes: list[str] = []
    verified_hashes: list[str] = []

    if not comment_texts:
        if require_signoff:
            violations.append("missing_signoff_comment")
        return all_hashes, verified_hashes, violations

    for idx, text in enumerate(comment_texts):
        for tag in scan_text_for_violations(text):
            violations.append(f"comment_{idx}_{tag}")
        for h in extract_commit_hashes(text):
            if h not in all_hashes:
                all_hashes.append(h)

    if require_signoff:
        combined = "\n".join(comment_texts)
        if not all_hashes:
            violations.append("missing_commit_hash_in_signoff")
        if not _TEST_EVIDENCE_RE.search(combined):
            violations.append("missing_test_report_in_signoff")
        if not _REVIEW_EVIDENCE_RE.search(combined):
            violations.append("missing_reviewer_signoff_in_comment")

    if check_commits and all_hashes:
        for h in all_hashes:
            if verify_git_commit_exists(h, repo_root=repo_root):
                verified_hashes.append(h)
        if require_signoff and not verified_hashes:
            violations.append("unverified_commit_hash_in_git")

    return all_hashes, verified_hashes, violations


def audit_single_card(
    card: dict[str, Any],
    *,
    lane_name: str,
    checklists: list[dict[str, Any]],
    comment_texts: list[str],
    check_commits: bool = False,
    repo_root: Path | None = None,
) -> CardAuditResult:
    """Audit one card's checklists and comments against Analytics FC governance rules."""
    is_done = normalize_lane_name(lane_name) == "done"
    completed, total, chk_violations = validate_checklist_completion(checklists)
    hashes, verified, cmt_violations = validate_comment_texts(
        comment_texts,
        require_signoff=is_done,
        check_commits=check_commits,
        repo_root=repo_root,
    )

    violations: list[str] = []
    if is_done:
        violations.extend(chk_violations)
    violations.extend(cmt_violations)

    return CardAuditResult(
        card_id=str(card.get("id", "")),
        card_name=str(card.get("name", "")),
        lane=lane_name,
        compliant=len(violations) == 0,
        checklist_completed=completed,
        checklist_total=total,
        commit_hashes=hashes,
        verified_hashes=verified,
        violations=violations,
    )


def _group_checklists_by_card(checklists: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group board checklists by owning card ID."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for chk in checklists:
        card_id = str(chk.get("idCard", ""))
        if card_id:
            grouped.setdefault(card_id, []).append(chk)
    return grouped


def _group_comments_by_card(actions: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Group commentCard action texts by owning card ID."""
    grouped: dict[str, list[str]] = {}
    for action in actions:
        if action.get("type") not in (None, "commentCard"):
            continue
        data = action.get("data") or {}
        card_info = data.get("card") or {}
        card_id = str(card_info.get("id", ""))
        text = str(data.get("text", "")).strip()
        if card_id and text:
            grouped.setdefault(card_id, []).append(text)
    return grouped


def audit_board_compliance(
    board_id: str = BOARD_ID,
    *,
    client: TrelloClient | None = None,
    check_commits: bool = False,
    repo_root: Path | None = None,
    ticket_prefix: str = "AFC-",
) -> BoardAuditReport:
    """Audit all active cards on a Trello board for lane, checklist, and sign-off compliance."""
    active_client = client if client is not None else LiveTrelloClient()
    lists = active_client.get_lists(board_id)
    list_names: dict[str, str] = {str(lst["id"]): str(lst["name"]) for lst in lists}

    cards = active_client.get_cards(board_id)
    checklists_by_card = _group_checklists_by_card(active_client.get_checklists(board_id))
    comments_by_card = _group_comments_by_card(active_client.get_comment_actions(board_id))

    audited: list[CardAuditResult] = []
    done_count = 0
    violation_count = 0
    prefix_upper = ticket_prefix.upper()

    for card in cards:
        if card.get("closed"):
            continue
        name = str(card.get("name", ""))
        if prefix_upper and prefix_upper not in name.upper():
            continue
        lane_name = list_names.get(str(card.get("idList", "")), "Unknown")
        if normalize_lane_name(lane_name) == "done":
            done_count += 1
        card_id = str(card.get("id", ""))
        res = audit_single_card(
            card,
            lane_name=lane_name,
            checklists=checklists_by_card.get(card_id, []),
            comment_texts=comments_by_card.get(card_id, []),
            check_commits=check_commits,
            repo_root=repo_root,
        )
        audited.append(res)
        if not res.compliant:
            violation_count += 1

    return BoardAuditReport(
        board_id=board_id,
        compliant=violation_count == 0,
        total_cards_audited=len(audited),
        done_cards_audited=done_count,
        violations_count=violation_count,
        cards=audited,
    )


def governed_transition_card(
    ticket_id: str,
    target_lane: str,
    *,
    board_id: str = BOARD_ID,
    client: TrelloClient | None = None,
    check_commits: bool = True,
    repo_root: Path | None = None,
) -> TransitionResult:
    """Move a ticket between Kanban lanes only when governance rules are satisfied."""
    active_client = client if client is not None else LiveTrelloClient()
    lists = active_client.get_lists(board_id)
    list_by_norm: dict[str, dict[str, Any]] = {
        normalize_lane_name(str(lst["name"])): lst for lst in lists if not lst.get("closed", False)
    }
    list_name_by_id: dict[str, str] = {str(lst["id"]): str(lst["name"]) for lst in lists}

    cards = active_client.get_cards(board_id)
    card = next((c for c in cards if ticket_id.upper() in str(c.get("name", "")).upper()), None)
    if card is None:
        return TransitionResult(
            moved=False,
            card_id="",
            card_name=ticket_id,
            from_lane="Unknown",
            to_lane=target_lane,
            violations=[f"card_not_found:{ticket_id}"],
        )

    card_id = str(card["id"])
    card_name = str(card["name"])
    from_lane = list_name_by_id.get(str(card.get("idList", "")), "Unknown")

    ok_lane, lane_errs = validate_lane_transition(from_lane, target_lane)
    if not ok_lane:
        return TransitionResult(
            moved=False,
            card_id=card_id,
            card_name=card_name,
            from_lane=from_lane,
            to_lane=target_lane,
            violations=lane_errs,
        )

    target_norm = normalize_lane_name(target_lane)
    target_list = list_by_norm.get(target_norm)
    if target_list is None:
        return TransitionResult(
            moved=False,
            card_id=card_id,
            card_name=card_name,
            from_lane=from_lane,
            to_lane=target_lane,
            violations=[f"board_missing_list:{target_lane}"],
        )

    if target_norm == "done":
        checklists = _group_checklists_by_card(active_client.get_checklists(board_id)).get(
            card_id, []
        )
        comments = _group_comments_by_card(active_client.get_comment_actions(board_id)).get(
            card_id, []
        )
        preview = audit_single_card(
            card,
            lane_name="Done",
            checklists=checklists,
            comment_texts=comments,
            check_commits=check_commits,
            repo_root=repo_root,
        )
        if not preview.compliant:
            return TransitionResult(
                moved=False,
                card_id=card_id,
                card_name=card_name,
                from_lane=from_lane,
                to_lane=target_lane,
                violations=preview.violations,
            )

    active_client.move_card_to_list(card_id, str(target_list["id"]))
    return TransitionResult(
        moved=True,
        card_id=card_id,
        card_name=card_name,
        from_lane=from_lane,
        to_lane=str(target_list["name"]),
        violations=[],
    )


def _format_audit_summary(report: BoardAuditReport) -> str:
    """Format a plain-text board governance report with zero prohibited dashes."""
    lines = [
        (
            f"Trello Governance Audit | board={report.board_id} | "
            f"cards={report.total_cards_audited} | done={report.done_cards_audited} | "
            f"violations={report.violations_count} | compliant={report.compliant}"
        )
    ]
    for card in report.cards:
        if not card.compliant:
            joined = ", ".join(card.violations)
            lines.append(
                f"  [NON-COMPLIANT] {card.card_name} ({card.lane}) "
                f"[{card.checklist_completed}/{card.checklist_total}]: {joined}"
            )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for Trello governance auditing and lane transitions."""
    parser = argparse.ArgumentParser(
        description="Audit Trello board governance compliance and validate lane transitions.",
    )
    parser.add_argument(
        "--board-id",
        default=BOARD_ID,
        help=f"Target Trello board ID (default: {BOARD_ID}).",
    )
    parser.add_argument(
        "--ticket-prefix",
        default="AFC-",
        help="Filter cards containing this ticket prefix (empty string audits all cards).",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Audit all board cards for Done checklist and sign-off compliance.",
    )
    parser.add_argument(
        "--check-commits",
        action="store_true",
        help="Verify extracted commit hashes against local Git objects.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output.",
    )
    parser.add_argument(
        "--move",
        nargs=2,
        metavar=("TICKET_ID", "TARGET_LANE"),
        help="Perform a governed lane transition for TICKET_ID to TARGET_LANE.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point returning 0 on compliance and 1 on governance violations."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.move:
        ticket_id, target_lane = args.move
        res = governed_transition_card(
            ticket_id,
            target_lane,
            board_id=args.board_id,
            check_commits=args.check_commits,
        )
        if args.json:
            print(json.dumps(asdict(res), indent=2))
        else:
            status = "MOVED" if res.moved else "BLOCKED"
            print(
                f"{status}: {res.card_name} ({res.from_lane} -> {res.to_lane}) "
                f"violations={res.violations}"
            )
        return 0 if res.moved else 1

    # Default action is board compliance audit
    report = audit_board_compliance(
        board_id=args.board_id,
        check_commits=args.check_commits,
        ticket_prefix=args.ticket_prefix,
    )
    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        print(_format_audit_summary(report))
    return 0 if report.compliant else 1


if __name__ == "__main__":
    sys.exit(main())
