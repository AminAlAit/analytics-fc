"""Automated synchronization script between Trello card states and LikeC4 model tags.

Reads live ticket lists from Trello and updates status tags in architecture/src/model.c4
so that LikeC4 diagrams automatically reflect green, amber, or gray states.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger("analytics_fc.architecture.sync")

ENV_FILE = Path(r"C:\Users\AminE\.secrets\trello.env")
BOARD_ID = "6add00820e13a7c7024609e8"
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_FILE = BASE_DIR / "architecture" / "src" / "model.c4"

# Map component identifiers to corresponding ticket keys
COMPONENT_TICKET_MAP = {
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
    "dbtTestGate": "AFC-528",
    # Stage 7: Evidently AI & Drift
    "evidentlyDrift": "AFC-529",
    "driftIndicator": "AFC-532",
    # Stage 8: Flagship Model 2
    "featureContract": "AFC-533",
    "trajectoryDataset": "AFC-534",
    "baselineRegressor": "AFC-535",
    "evaluationSuite": "AFC-536",
}


def load_trello_credentials() -> tuple[str, str]:
    """Extract Trello API key and token from local secrets."""
    if not ENV_FILE.exists():
        raise FileNotFoundError(f"Missing Trello credentials file at {ENV_FILE}")
    env_vars: dict[str, str] = {}
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip()
    return env_vars["TRELLO_KEY"], env_vars["TRELLO_TOKEN"]


def fetch_trello_ticket_statuses() -> dict[str, str]:
    """Query Trello API to retrieve current status for each AFC ticket."""
    key, token = load_trello_credentials()
    params = urllib.parse.urlencode({"key": key, "token": token, "cards": "open"})
    url = f"https://api.trello.com/1/boards/{BOARD_ID}/lists?{params}"

    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req) as resp:
        lists = json.loads(resp.read().decode("utf-8"))

    ticket_status: dict[str, str] = {}
    for lst in lists:
        list_name = lst["name"].lower()
        if "done" in list_name:
            tag = "#built"
        elif "progress" in list_name or "review" in list_name:
            tag = "#in_progress"
        else:
            tag = "#backlog"

        for card in lst.get("cards", []):
            match = re.search(r"(AFC-\d+)", card["name"])
            if match:
                ticket_status[match.group(1)] = tag

    return ticket_status


def sync_likec4_model() -> int:
    """Synchronize LikeC4 model tags in place against live Trello state."""
    ticket_statuses = fetch_trello_ticket_statuses()
    content = MODEL_FILE.read_text(encoding="utf-8")

    updates_count = 0
    for comp_name, ticket_id in COMPONENT_TICKET_MAP.items():
        current_tag = ticket_statuses.get(ticket_id, "#backlog")

        pattern = rf"({comp_name}\s*=\s*(?:component|container)\s*'[^']+'\s*\{{\s*)(#[a-z_]+)"
        match = re.search(pattern, content)
        if match:
            existing_tag = match.group(2)
            if existing_tag != current_tag:
                content = re.sub(
                    pattern,
                    rf"\g<1>{current_tag}",
                    content,
                    count=1,
                )
                updates_count += 1
                print(f"Updated {comp_name} ({ticket_id}): {existing_tag} -> {current_tag}")

    if updates_count > 0:
        MODEL_FILE.write_text(content, encoding="utf-8")
        print(f"Synchronized {updates_count} component tags in {MODEL_FILE.name}")
    else:
        print("All LikeC4 component tags are already in sync with Trello.")

    return updates_count


if __name__ == "__main__":
    sync_likec4_model()
