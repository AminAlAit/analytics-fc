"""Atomic startup asset bootstrapper for standalone and Streamlit Cloud deployments.

Ensures required serving artifacts (DuckDB catalog, superstar_gallery.json, and
Model 2 joblib weights) are present locally before any @st.cache_data loaders run,
downloading from AFC_BUNDLE_URL when deployed without gitignored binary assets.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import streamlit as st

logger = logging.getLogger("analytics_fc.dashboard.bootstrap")

BASE_DIR = Path(__file__).resolve().parent.parent
MIN_VALID_DUCKDB_BYTES = 100_000

REQUIRED_CORE_ASSETS: dict[str, int] = {
    "dbt_project/analytics_fc.duckdb": MIN_VALID_DUCKDB_BYTES,
    "data/processed/superstar_gallery.json": 100,
}

OPTIONAL_MODEL_ASSETS: dict[str, int] = {
    "models/trajectory_regressor.joblib": 100,
    "models/conformal_predictor.joblib": 100,
}


def _resolve_bundle_url(explicit_url: str | None = None) -> str | None:
    """Resolve serving bundle URL from explicit arg, environment, or Streamlit secrets."""
    if explicit_url and explicit_url.strip():
        return explicit_url.strip()

    env_url = os.environ.get("AFC_BUNDLE_URL", "").strip()
    if env_url:
        return env_url

    try:
        secret_val = st.secrets.get("AFC_BUNDLE_URL")
        if isinstance(secret_val, str) and secret_val.strip():
            return secret_val.strip()
    except Exception:
        pass
    return None


def _inspect_missing_assets(
    base_dir: Path,
    specs: dict[str, int],
) -> list[str]:
    """Return relative paths of assets that are absent or smaller than expected."""
    missing: list[str] = []
    for rel_path, min_bytes in specs.items():
        candidate = base_dir / rel_path
        if not candidate.exists() or candidate.stat().st_size < min_bytes:
            missing.append(rel_path)
    return missing


def _validate_archive_member(member: zipfile.ZipInfo, dest_root: Path) -> Path:
    """Validate zip member path against traversal, absolute paths, and symlinks."""
    if member.is_dir():
        raise ValueError("Directory entry should be skipped before validation.")

    # Reject POSIX symlink mode bits if set in external_attr
    mode = (member.external_attr >> 16) & 0o170000
    if mode == 0o120000:
        raise ValueError(f"Unsafe symlink entry in serving bundle: {member.filename}")

    rel_path = Path(member.filename)
    if rel_path.is_absolute() or rel_path.drive or ".." in rel_path.parts:
        raise ValueError(f"Unsafe archive member path: {member.filename}")

    resolved_root = dest_root.resolve()
    target = (resolved_root / rel_path).resolve()
    if resolved_root not in target.parents:
        raise ValueError(f"Archive member escapes target directory: {member.filename}")
    return target


def _safe_extract_bundle(archive_path: Path, base_dir: Path) -> list[str]:
    """Safely extract zip bundle into a staging dir and atomically move into base_dir."""
    extracted_paths: list[str] = []
    with tempfile.TemporaryDirectory(dir=str(base_dir)) as stage_dir:
        stage_root = Path(stage_dir)
        with zipfile.ZipFile(archive_path, mode="r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                staged_target = _validate_archive_member(member, stage_root)
                staged_target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member, "r") as src, open(staged_target, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)

                final_target = (base_dir / member.filename).resolve()
                final_target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged_target, final_target)
                extracted_paths.append(member.filename)
    return extracted_paths


def prepare_serving_assets(
    base_dir: Path | str | None = None,
    bundle_url: str | None = None,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Verify local serving artifacts or atomically download them from AFC_BUNDLE_URL."""
    root = Path(base_dir).resolve() if base_dir is not None else BASE_DIR.resolve()
    missing_core = _inspect_missing_assets(root, REQUIRED_CORE_ASSETS)
    missing_models = _inspect_missing_assets(root, OPTIONAL_MODEL_ASSETS)

    if not missing_core and not missing_models:
        return {
            "ready": True,
            "source": "local",
            "missing_core": [],
            "missing_models": [],
            "error": None,
        }

    resolved_url = _resolve_bundle_url(bundle_url)
    if not resolved_url:
        return {
            "ready": len(missing_core) == 0,
            "source": "local" if len(missing_core) == 0 else "missing",
            "missing_core": missing_core,
            "missing_models": missing_models,
            "error": (
                "Missing serving artifacts and AFC_BUNDLE_URL is not configured."
                if missing_core
                else None
            ),
        }

    if not resolved_url.startswith(("https://", "http://")):
        return {
            "ready": False,
            "source": "error",
            "missing_core": missing_core,
            "missing_models": missing_models,
            "error": "AFC_BUNDLE_URL must use http:// or https:// scheme.",
        }

    root.mkdir(parents=True, exist_ok=True)
    fd, tmp_zip_name = tempfile.mkstemp(suffix=".zip", dir=str(root))
    os.close(fd)
    tmp_zip = Path(tmp_zip_name)

    try:
        req = urllib.request.Request(resolved_url, headers={"User-Agent": "analytics-fc/1.0"})
        with (
            urllib.request.urlopen(req, timeout=timeout_seconds) as resp,
            open(tmp_zip, "wb") as out_file,
        ):
            shutil.copyfileobj(resp, out_file, length=1024 * 1024)

        extracted = _safe_extract_bundle(tmp_zip, root)
        missing_after = _inspect_missing_assets(root, REQUIRED_CORE_ASSETS)
        models_after = _inspect_missing_assets(root, OPTIONAL_MODEL_ASSETS)
        return {
            "ready": len(missing_after) == 0,
            "source": "downloaded",
            "extracted": extracted,
            "missing_core": missing_after,
            "missing_models": models_after,
            "error": None if not missing_after else f"Bundle missing files: {missing_after}",
        }
    except Exception as exc:
        logger.warning("Failed to bootstrap serving bundle from %s: %s", resolved_url, exc)
        return {
            "ready": False,
            "source": "error",
            "missing_core": missing_core,
            "missing_models": missing_models,
            "error": str(exc),
        }
    finally:
        if tmp_zip.exists():
            tmp_zip.unlink()


@st.cache_resource
def _cached_prepare_serving_assets() -> dict[str, Any]:
    """Cached single-flight startup bootstrapper for Streamlit Cloud."""
    return prepare_serving_assets(base_dir=BASE_DIR)


def ensure_serving_assets() -> dict[str, Any]:
    """Return cached startup status, clearing the resource cache on failed attempts."""
    status = _cached_prepare_serving_assets()
    if not status.get("ready", False):
        _cached_prepare_serving_assets.clear()
    return status
