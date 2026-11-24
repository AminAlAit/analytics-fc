"""Scouting copy stays grounded and free of long dashes."""

from __future__ import annotations

import json

import pytest

from src.publisher.copy_engine import (
    build_prompt,
    draft_scouting_copy,
    numbers_are_grounded,
    router_health,
    scrub_copy,
)

STATS = {"player": "E. Haaland", "overall": 88, "age": 21, "edition_year": 2022}


def test_scrub_replaces_long_dashes() -> None:
    assert scrub_copy("88\u2014a finisher") == "88, a finisher"
    assert scrub_copy("age 21\u201322") == "age 21-22"
    assert "\u2014" not in scrub_copy("88\u2014a finisher")


def test_draft_rejects_invented_numbers() -> None:
    def _fake(prompt: str, model: str) -> str:
        assert model == "hetzner-flash"
        assert "overall: 88" in prompt
        return "Haaland is 99 overall."

    with pytest.raises(ValueError, match="not in the supplied stats"):
        draft_scouting_copy(STATS, complete=_fake)


def test_draft_keeps_grounded_flash_copy() -> None:
    def _fake(prompt: str, model: str) -> str:
        assert model == "hetzner-flash"
        return "E. Haaland is 88 overall at age 21 in 2022\u2014a centre forward."

    text = draft_scouting_copy(STATS, complete=_fake)
    assert "\u2014" not in text
    assert "\u2013" not in text
    assert numbers_are_grounded(text, STATS)
    assert "88" in text
    assert "overall: 88" in build_prompt(STATS)


def test_router_health_reads_status() -> None:
    class _Response:
        status = 200

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"{}"

    def _open(request, timeout=0):  # type: ignore[no-untyped-def]
        assert request.full_url.endswith("/health")
        assert timeout == 3.0
        return _Response()

    import src.publisher.copy_engine as copy_engine

    original = copy_engine.urllib.request.urlopen
    copy_engine.urllib.request.urlopen = _open
    try:
        assert router_health() is True
    finally:
        copy_engine.urllib.request.urlopen = original


def test_complete_copy_posts_json(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class _Response:
        status = 200

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8")

    def _open(request, timeout=0):  # type: ignore[no-untyped-def]
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["auth"] = request.get_header("Authorization")
        return _Response()

    monkeypatch.setattr("src.publisher.copy_engine.urllib.request.urlopen", _open)
    token_file = tmp_path / "token.txt"
    token_file.write_text("secret-token\n", encoding="utf-8")
    monkeypatch.setattr("src.publisher.copy_engine.TOKEN_PATH", token_file)
    from src.publisher.copy_engine import complete_copy

    text = complete_copy("hello", model="hetzner-flash")
    assert text == "ok"
    assert str(captured["url"]).endswith("/chat/completions")
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "hetzner-flash"
    assert captured["auth"] == "Bearer secret-token"
