"""Factual scouting copy through the Hetzner router.

Default model is hetzner-flash. Output is scrubbed of dash characters that
are not hyphens, and rejected when it contains a number that was not supplied.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "http://100.116.103.104:8080/v1"
DEFAULT_MODEL = "hetzner-flash"
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def resolve_token_path() -> Path:
    """Token file from HETZNER_ROUTER_TOKEN_FILE, otherwise ~/.secrets."""
    return Path(
        os.environ.get("HETZNER_ROUTER_TOKEN_FILE")
        or (Path.home() / ".secrets" / "hetzner-router-token.txt")
    )


def scrub_copy(text: str) -> str:
    """Replace em and en dashes so published copy stays hyphen-only."""
    return text.replace("\u2014", ", ").replace("\u2013", "-").strip()


def _stat_numbers(stats: dict[str, Any]) -> set[str]:
    allowed: set[str] = set()
    for value in stats.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            allowed.add(str(int(value)) if float(value).is_integer() else str(value))
            allowed.add(str(value))
        elif isinstance(value, str):
            allowed.update(_NUMBER.findall(value))
    return allowed


def numbers_are_grounded(text: str, stats: dict[str, Any]) -> bool:
    """True when every number in the copy appears in the supplied stats."""
    allowed = _stat_numbers(stats)
    return all(number in allowed for number in _NUMBER.findall(text))


def build_prompt(stats: dict[str, Any]) -> str:
    """Ask for two factual sentences that may use only the supplied stats."""
    lines = [f"{key}: {value}" for key, value in stats.items()]
    return (
        "Write two short factual scouting sentences. "
        "Use only the numbers and names below. Do not add any other number. "
        "Do not use an em dash.\n" + "\n".join(lines)
    )


def load_router_token(path: Path | str | None = None) -> str:
    """Read the bearer token from the environment or the secrets file."""
    if path is None:
        env_token = os.environ.get("HETZNER_ROUTER_TOKEN", "").strip()
        if env_token:
            return env_token
        token_path = resolve_token_path()
    else:
        token_path = Path(path)
    return token_path.read_text(encoding="utf-8").strip()


def router_health(base_url: str = DEFAULT_BASE_URL, timeout: float = 3.0) -> bool:
    """Return True when GET /health on the router responds 200."""
    root = base_url.rstrip("/")
    health_url = root[: -len("/v1")] + "/health" if root.endswith("/v1") else root + "/health"
    request = urllib.request.Request(health_url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status) == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def complete_copy(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    token: str | None = None,
    timeout: float = 30.0,
) -> str:
    """Call the OpenAI-compatible chat completions endpoint."""
    bearer = token if token is not None else load_router_token()
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "You write short factual football scouting notes."},
                {"role": "user", "content": prompt},
            ],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {bearer}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str(payload["choices"][0]["message"]["content"])


def draft_scouting_copy(
    stats: dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    complete: Any = None,
) -> str:
    """Draft copy and reject text that invents a number or keeps a long dash."""
    prompt = build_prompt(stats)
    sender = complete if complete is not None else complete_copy
    raw = str(sender(prompt, model=model))
    cleaned = scrub_copy(raw)
    if "\u2014" in cleaned or "\u2013" in cleaned:
        raise ValueError("Copy still contains a long dash.")
    if not numbers_are_grounded(cleaned, stats):
        raise ValueError("Copy contains a number that is not in the supplied stats.")
    if not cleaned:
        raise ValueError("Copy is empty.")
    return cleaned
