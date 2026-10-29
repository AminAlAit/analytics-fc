"""Continuous synchronization engine between Trello board state and LikeC4 architecture.

Fetches live ticket states from Trello board 6add00820e13a7c7024609e8 and updates:
1. Status tags (#built, #in_progress, #backlog)
2. Live Trello card deep-links (link https://trello.com/c/<id> 'Trello: AFC-xxx')
3. Structured metadata blocks (ticket, card, status)
4. Stage burndown telemetry
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("analytics_fc.architecture.sync")

ENV_FILE = Path(r"C:\Users\AminE\.secrets\trello.env")
BOARD_ID = "6add00820e13a7c7024609e8"
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_FILE = BASE_DIR / "architecture" / "src" / "model.c4"
VIEWS_FILE = BASE_DIR / "architecture" / "src" / "views.c4"

# Complete mapping from model component identifier to Trello ticket ID
COMPONENT_TICKET_MAP = {
    # Stage 1 & 4: Streamlit Scouting Terminal
    "catalogSelector": "AFC-501",
    "archetypeView": "AFC-503",
    "tacticalRadar": "AFC-504",
    "vectorAudit": "AFC-505",
    "apiClient": "AFC-519",
    "driftIndicator": "AFC-532",
    # Stage 2 & 3: DuckDB Data Lake & dbt Models
    "duckdbBronze": "AFC-506",
    "dvcStore": "AFC-507",
    "dbtSilver": "AFC-508",
    "dbtGold": "AFC-510",
    # Stage 4: FastAPI REST Inference Microservice
    "healthProbe": "AFC-514",
    "catalogSearch": "AFC-515",
    "profileLookup": "AFC-516",
    "catalogStats": "AFC-517",
    "vectorMatcher": "AFC-518",
    # Stage 5: Prefect Orchestration
    "envSentinel": "AFC-520",
    "ingestTask": "AFC-521",
    "dbtTask": "AFC-522",
    "galleryTask": "AFC-523",
    "qualityGateTask": "AFC-524",
    # Stage 6: Code Quality & CI/CD
    "ruffLinter": "AFC-525",
    "mypyChecker": "AFC-526",
    "githubActions": "AFC-527",
    # Stage 7: Evidently AI & Drift
    "evidentlyDrift": "AFC-529",
    # Stage 8: Flagship Model 2 (Career Trajectory)
    "featureContract": "AFC-533",
    "trajectoryDataset": "AFC-534",
    "baselineRegressor": "AFC-535",
    "evaluationSuite": "AFC-536",
}


@dataclass(frozen=True)
class TrelloCardInfo:
    ticket_id: str
    card_name: str
    short_url: str
    list_name: str
    status_tag: str


def set_low_process_priority() -> None:
    """Throttle process priority to protect laptop hardware."""
    try:
        if sys.platform == "win32":
            import ctypes

            # BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000
            )
    except Exception:
        pass


def load_trello_credentials() -> tuple[str, str]:
    """Extract Trello API key and token from local secrets."""
    if not ENV_FILE.exists():
        raise FileNotFoundError(f"Missing Trello credentials file at {ENV_FILE}")
    env_vars: dict[str, str] = {}
    with open(ENV_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip().strip("\"'")
    return env_vars["TRELLO_KEY"], env_vars["TRELLO_TOKEN"]


def fetch_trello_cards() -> tuple[dict[str, TrelloCardInfo], dict[str, int]]:
    """Query Trello API to retrieve current status and URLs for all board cards."""
    key, token = load_trello_credentials()
    params = urllib.parse.urlencode({"key": key, "token": token, "cards": "open"})
    url = f"https://api.trello.com/1/boards/{BOARD_ID}/lists?{params}"

    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        lists = json.loads(resp.read().decode("utf-8"))

    cards_by_ticket: dict[str, TrelloCardInfo] = {}
    counts = {"done": 0, "in_progress": 0, "backlog": 0, "total": 0}

    for lst in lists:
        raw_list_name = lst["name"]
        list_name_lower = raw_list_name.lower()

        if "done" in list_name_lower:
            tag = "#built"
            category = "done"
        elif "progress" in list_name_lower or "review" in list_name_lower:
            tag = "#in_progress"
            category = "in_progress"
        else:
            tag = "#backlog"
            category = "backlog"

        for card in lst.get("cards", []):
            match = re.search(r"(AFC-\d+)", card["name"])
            if match:
                ticket_id = match.group(1)
                clean_name = re.sub(r"^AFC-\d+:\s*", "", card["name"]).strip()
                cards_by_ticket[ticket_id] = TrelloCardInfo(
                    ticket_id=ticket_id,
                    card_name=clean_name,
                    short_url=card.get("shortUrl", f"https://trello.com/c/{card.get('shortLink')}"),
                    list_name=raw_list_name,
                    status_tag=tag,
                )
                counts[category] += 1
                counts["total"] += 1

    return cards_by_ticket, counts


def find_block_bounds(content: str, start_index: int) -> tuple[int, int]:
    """Find the start and end character positions of a brace-enclosed block."""
    brace_start = content.find("{", start_index)
    if brace_start == -1:
        return -1, -1

    depth = 0
    i = brace_start
    while i < len(content):
        char = content[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return brace_start, i
        i += 1
    return brace_start, -1


def update_element_block(block_body: str, card: TrelloCardInfo, indent: str) -> str:
    """Update tag, link, and metadata within an element block body."""
    body = block_body

    # 1. Update or set status tag (must be first line inside element)
    tag_pattern = r"(^\s*)(#[a-z_]+)"
    if re.search(tag_pattern, body, flags=re.MULTILINE):
        body = re.sub(tag_pattern, rf"\g<1>{card.status_tag}", body, count=1, flags=re.MULTILINE)
    else:
        body = f"\n{indent}  {card.status_tag}" + body

    # 2. Update or insert Trello link
    link_line = f"{indent}  link {card.short_url} 'Trello: {card.ticket_id}'"
    link_pattern = r"^\s*link\s+https://trello\.com/c/\S+(\s+'[^']*')?"
    if re.search(link_pattern, body, flags=re.MULTILINE):
        body = re.sub(link_pattern, link_line, body, count=1, flags=re.MULTILINE)
    else:
        # Insert after technology line if present
        tech_pattern = r"(technology\s+'[^']*')"
        if re.search(tech_pattern, body):
            body = re.sub(tech_pattern, rf"\g<1>\n{link_line}", body, count=1)
        else:
            body = body.rstrip() + f"\n{link_line}\n{indent}"

    # 3. Update or insert metadata block
    sanitized_name = card.card_name.replace("'", "\\'")
    metadata_block = (
        f"{indent}  metadata {{\n"
        f"{indent}    ticket '{card.ticket_id}'\n"
        f"{indent}    card '{sanitized_name}'\n"
        f"{indent}    status '{card.list_name}'\n"
        f"{indent}  }}"
    )

    metadata_pattern = r"^\s*metadata\s*\{[^}]*\}"
    if re.search(metadata_pattern, body, flags=re.MULTILINE | re.DOTALL):
        body = re.sub(
            metadata_pattern, metadata_block, body, count=1, flags=re.MULTILINE | re.DOTALL
        )
    else:
        # Place before style block or at end of element
        style_pattern = r"(^\s*style\s*\{)"
        if re.search(style_pattern, body, flags=re.MULTILINE):
            body = re.sub(
                style_pattern, rf"{metadata_block}\n\n\g<1>", body, count=1, flags=re.MULTILINE
            )
        else:
            body = body.rstrip() + f"\n\n{metadata_block}\n{indent}"

    return body


def sync_likec4_model(dry_run: bool = False) -> tuple[int, dict[str, int]]:
    """Synchronize LikeC4 model tags, links, and metadata against live Trello state."""
    cards_by_ticket, counts = fetch_trello_cards()
    content = MODEL_FILE.read_text(encoding="utf-8")

    updates_count = 0
    for comp_name, ticket_id in COMPONENT_TICKET_MAP.items():
        card = cards_by_ticket.get(ticket_id)
        if not card:
            continue

        decl_pattern = rf"([ \t]*){comp_name}\s*=\s*(?:component|container)\s*'[^']+'\s*\{{"
        match = re.search(decl_pattern, content)
        if not match:
            continue

        indent = match.group(1)
        brace_start, brace_end = find_block_bounds(content, match.start())
        if brace_start == -1 or brace_end == -1:
            continue

        block_body = content[brace_start + 1 : brace_end]
        updated_body = update_element_block(block_body, card, indent)

        if block_body != updated_body:
            content = content[: brace_start + 1] + updated_body + content[brace_end:]
            updates_count += 1
            print(f"Synchronized {comp_name} [{ticket_id}] -> {card.status_tag} | {card.short_url}")

    if not dry_run and updates_count > 0:
        MODEL_FILE.write_text(content, encoding="utf-8")
        print(f"Applied updates to {updates_count} components in {MODEL_FILE.name}")
    elif updates_count == 0:
        print("All LikeC4 components are already up to date with Trello.")

    # Calculate telemetry
    pct = round((counts["done"] / counts["total"] * 100.0), 1) if counts["total"] > 0 else 0.0
    print(
        f"Board Progress: {counts['done']}/{counts['total']} Done ({pct}%), {counts['in_progress']} In Progress, {counts['backlog']} Backlog"
    )

    return updates_count, counts


def sync_views_file() -> int:
    """Ensure views contain direct Trello board navigation links."""
    content = VIEWS_FILE.read_text(encoding="utf-8")
    board_url = f"https://trello.com/b/{BOARD_ID}"
    updated = content

    view_names = [
        "systemContext",
        "containerOverview",
        "prefectEngine",
        "apiService",
        "scoutingTerminal",
        "dataArchitecture",
        "qualitySuite",
        "careerModelSuite",
    ]
    for vname in view_names:
        view_pattern = (
            rf"(view\s+{vname}\s+of\s+\w+\s*\{{\s*title\s+'[^']+'\s*description\s+'[^']+')"
        )
        if (
            re.search(view_pattern, updated)
            and f"link {board_url}"
            not in updated[updated.find(f"view {vname}") : updated.find(f"view {vname}") + 400]
        ):
            updated = re.sub(
                view_pattern,
                rf"\g<1>\n    link {board_url} 'Trello: Analytics FC Master Board'",
                updated,
                count=1,
            )

    if updated != content:
        VIEWS_FILE.write_text(updated, encoding="utf-8")
        print(f"Synchronized top-level Trello board links in {VIEWS_FILE.name}")
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize Trello tickets with LikeC4 architecture."
    )
    parser.add_argument(
        "--once",
        action="store_true",
        default=True,
        help="Execute single synchronization run and exit.",
    )
    parser.add_argument("--watch", action="store_true", help="Continuous polling watcher mode.")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval in seconds.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Evaluate changes without writing to disk."
    )
    args = parser.parse_args()

    set_low_process_priority()

    if args.watch:
        print(f"Starting Trello-LikeC4 background watcher (interval: {args.interval}s)...")
        while True:
            try:
                sync_likec4_model(dry_run=args.dry_run)
                sync_views_file()
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nWatcher stopped.")
                break
            except Exception as e:
                print(f"Sync error: {e}")
                time.sleep(args.interval)
    else:
        sync_likec4_model(dry_run=args.dry_run)
        sync_views_file()


if __name__ == "__main__":
    main()
