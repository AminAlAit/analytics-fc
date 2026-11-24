"""X API v2 client for media upload and posts.

Calls stay behind an injectable transport so tests never hit the live API.
429 responses sleep and retry. Media upload uses INIT, APPEND, and FINALIZE.
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Callable
from typing import Any

Transport = Callable[
    [str, str, dict[str, str], bytes | None], tuple[int, dict[str, Any], dict[str, str]]
]


class XClient:
    """Authenticated X API v2 client."""

    def __init__(
        self,
        access_token: str,
        *,
        transport: Transport | None = None,
        sleep: Callable[[float], None] | None = None,
        chunk_size: int = 1024 * 1024,
        max_retries: int = 3,
    ) -> None:
        if not access_token:
            raise ValueError("OAuth 2.0 access token is required.")
        self.access_token = access_token
        self.transport = transport if transport is not None else _urllib_transport
        self.sleep = sleep if sleep is not None else time.sleep
        self.chunk_size = chunk_size
        self.max_retries = max_retries

    def _headers(self, content_type: str = "application/json") -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": content_type,
            "Accept": "application/json",
        }

    def _request(
        self,
        method: str,
        url: str,
        *,
        body: bytes | None = None,
        content_type: str = "application/json",
    ) -> dict[str, Any]:
        delay = 1.0
        last_status = 0
        for attempt in range(self.max_retries + 1):
            status, payload, headers = self.transport(
                method, url, self._headers(content_type), body
            )
            last_status = status
            if status != 429:
                if status >= 400:
                    raise RuntimeError(f"X API {status}: {payload}")
                return payload
            if attempt == self.max_retries:
                break
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            wait = float(retry_after) if retry_after else delay
            self.sleep(wait)
            delay *= 2
        raise RuntimeError(f"X API {last_status}: rate limit retries exhausted")

    def upload_media(self, media: bytes, *, media_type: str = "image/png") -> str:
        """Upload bytes in chunks and return the media id."""
        if not media:
            raise ValueError("Media payload is empty.")
        init = self._request(
            "POST",
            "https://api.x.com/2/media/upload/initialize",
            body=json.dumps(
                {
                    "media_type": media_type,
                    "total_bytes": len(media),
                    "media_category": "tweet_image",
                }
            ).encode("utf-8"),
        )
        media_id = str(init["data"]["id"])
        for index, start in enumerate(range(0, len(media), self.chunk_size)):
            chunk = media[start : start + self.chunk_size]
            self._request(
                "POST",
                f"https://api.x.com/2/media/upload/{media_id}/append",
                body=json.dumps(
                    {
                        "segment_index": index,
                        "media": base64.b64encode(chunk).decode("ascii"),
                    }
                ).encode("utf-8"),
            )
        self._request("POST", f"https://api.x.com/2/media/upload/{media_id}/finalize")
        return media_id

    def create_post(self, text: str, media_ids: list[str] | None = None) -> dict[str, Any]:
        """Create a post, optionally attaching uploaded media ids."""
        if not text.strip():
            raise ValueError("Post text is empty.")
        body: dict[str, Any] = {"text": text}
        if media_ids:
            body["media"] = {"media_ids": media_ids}
        return self._request(
            "POST",
            "https://api.x.com/2/tweets",
            body=json.dumps(body).encode("utf-8"),
        )


def _decode_payload(raw: bytes) -> dict[str, Any]:
    if not raw:
        return {}
    loaded = json.loads(raw.decode("utf-8"))
    return loaded if isinstance(loaded, dict) else {"data": loaded}


def _urllib_transport(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            response_headers = {key.lower(): value for key, value in response.headers.items()}
            return int(response.status), _decode_payload(raw), response_headers
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        response_headers = {key.lower(): value for key, value in exc.headers.items()}
        return int(exc.code), _decode_payload(raw), response_headers
