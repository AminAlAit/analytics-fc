"""X client chunking and 429 backoff, with no live network calls."""

from __future__ import annotations

import json

import pytest

from src.publisher.x_client import XClient


def test_upload_is_chunked_and_authorized() -> None:
    calls: list[tuple[str, str, bytes | None]] = []

    def transport(method, url, headers, body):  # type: ignore[no-untyped-def]
        assert headers["Authorization"] == "Bearer oauth-token"
        calls.append((method, url, body))
        if url.endswith("/initialize"):
            return 200, {"data": {"id": "media-1"}}, {}
        if url.endswith("/append"):
            return 200, {"data": {"id": "media-1"}}, {}
        if url.endswith("/finalize"):
            return 200, {"data": {"id": "media-1"}}, {}
        raise AssertionError(url)

    client = XClient("oauth-token", transport=transport, chunk_size=4)
    media_id = client.upload_media(b"abcdefghij")
    assert media_id == "media-1"
    appends = [
        json.loads(body.decode("utf-8")) for _method, url, body in calls if url.endswith("/append")
    ]
    assert [item["segment_index"] for item in appends] == [0, 1, 2]
    assert appends[0]["media"]
    assert any(url.endswith("/finalize") for _method, url, _body in calls)
    init_body = json.loads(calls[0][2].decode("utf-8"))
    assert init_body["total_bytes"] == 10


def test_429_uses_retry_after_then_succeeds() -> None:
    sleeps: list[float] = []
    attempts = {"count": 0}

    def transport(method, url, headers, body):  # type: ignore[no-untyped-def]
        attempts["count"] += 1
        if attempts["count"] == 1:
            return 429, {"error": "slow down"}, {"retry-after": "2"}
        return 200, {"data": {"id": "post-9"}}, {}

    client = XClient("oauth-token", transport=transport, sleep=sleeps.append)
    payload = client.create_post("Haaland note", media_ids=["media-1"])
    assert payload["data"]["id"] == "post-9"
    assert sleeps == [2.0]
    assert attempts["count"] == 2


def test_rate_limit_exhaustion_raises() -> None:
    def transport(method, url, headers, body):  # type: ignore[no-untyped-def]
        return 429, {"error": "slow down"}, {"retry-after": "1"}

    client = XClient("oauth-token", transport=transport, sleep=lambda _seconds: None, max_retries=1)
    with pytest.raises(RuntimeError, match="rate limit"):
        client.create_post("again")


def test_urllib_transport_returns_http_429(monkeypatch: pytest.MonkeyPatch) -> None:
    """urlopen HTTPError 429 becomes a status tuple so the client can retry."""
    import urllib.error
    from email.message import Message
    from io import BytesIO

    from src.publisher.x_client import _urllib_transport

    headers = Message()
    headers["Retry-After"] = "3"
    error = urllib.error.HTTPError(
        url="https://api.x.com/2/tweets",
        code=429,
        msg="Too Many Requests",
        hdrs=headers,
        fp=BytesIO(b'{"error":"slow"}'),
    )

    def _open(request, timeout=0):  # type: ignore[no-untyped-def]
        raise error

    monkeypatch.setattr("urllib.request.urlopen", _open)
    status, payload, response_headers = _urllib_transport(
        "POST",
        "https://api.x.com/2/tweets",
        {},
        b"{}",
    )
    assert status == 429
    assert payload["error"] == "slow"
    assert response_headers["retry-after"] == "3"


def test_missing_token_and_empty_media_raise() -> None:
    with pytest.raises(ValueError):
        XClient("")
    client = XClient("oauth-token", transport=lambda *args: (200, {}, {}))
    with pytest.raises(ValueError):
        client.upload_media(b"")
    with pytest.raises(ValueError):
        client.create_post("   ")
