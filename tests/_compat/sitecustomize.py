"""Auto-loaded sitecustomize hook for child test subprocesses (Prefect server).

Translates legacy FastAPI Body(..., min_items=..., max_items=...) kwargs used by
Prefect 3.x API routes into FastAPI >= 0.115 min_length/max_length kwargs without
modifying third-party packages inside .venv.
"""

from __future__ import annotations

import functools
from typing import Any


def _normalize_legacy_collection_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Map legacy FastAPI Body min_items/max_items kwargs to min_length/max_length."""
    if "min_items" not in kwargs and "max_items" not in kwargs:
        return kwargs
    updated = dict(kwargs)
    if "min_items" in updated:
        min_items = updated.pop("min_items")
        updated.setdefault("min_length", min_items)
    if "max_items" in updated:
        max_items = updated.pop("max_items")
        updated.setdefault("max_length", max_items)
    return updated


def install_fastapi_body_compat_shim() -> None:
    """Idempotently wrap FastAPI Body helpers to accept min_items/max_items."""
    try:
        import fastapi
        import fastapi.param_functions
        import fastapi.params
    except ImportError:
        return

    if getattr(fastapi.params.Body.__init__, "_afc_min_items_patched", False):
        return

    orig_init = fastapi.params.Body.__init__
    orig_body_fn = fastapi.Body

    @functools.wraps(orig_init)
    def _compat_body_init(self: Any, default: Any = ..., **kwargs: Any) -> None:
        normalized = _normalize_legacy_collection_kwargs(kwargs)
        orig_init(self, default=default, **normalized)

    @functools.wraps(orig_body_fn)
    def _compat_body_fn(default: Any = ..., **kwargs: Any) -> Any:
        normalized = _normalize_legacy_collection_kwargs(kwargs)
        return orig_body_fn(default=default, **normalized)

    _compat_body_init._afc_min_items_patched = True  # type: ignore[attr-defined]
    _compat_body_fn._afc_min_items_patched = True  # type: ignore[attr-defined]

    fastapi.params.Body.__init__ = _compat_body_init  # type: ignore[method-assign]
    fastapi.Body = _compat_body_fn  # type: ignore[assignment]
    fastapi.param_functions.Body = _compat_body_fn  # type: ignore[assignment]


install_fastapi_body_compat_shim()
