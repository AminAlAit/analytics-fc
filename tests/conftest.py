"""Shared pytest startup hooks and dependency compatibility shims."""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _register_compat_pythonpath() -> None:
    """Ensure repo root and tests/_compat are on sys.path and PYTHONPATH."""
    repo_str = str(REPO_ROOT)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    compat_dir = str(REPO_ROOT / "tests" / "_compat")
    existing = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    for entry in (repo_str, compat_dir):
        if entry not in existing:
            existing.append(entry)
    # Keep tests/_compat first so sitecustomize hooks load cleanly in child processes
    ordered = [compat_dir, repo_str, *[p for p in existing if p not in (compat_dir, repo_str)]]
    os.environ["PYTHONPATH"] = os.pathsep.join(ordered)


def _configure_isolated_prefect_home() -> None:
    """Isolate Prefect SQLite state per pytest session away from user ~/.prefect."""
    tmp_dir = tempfile.mkdtemp(prefix="afc_prefect_")
    atexit.register(shutil.rmtree, tmp_dir, ignore_errors=True)
    os.environ.setdefault("PREFECT_HOME", tmp_dir)
    os.environ.setdefault(
        "PREFECT_API_DATABASE_CONNECTION_URL",
        f"sqlite+aiosqlite:///{(Path(tmp_dir) / 'prefect.db').as_posix()}",
    )


def _bootstrap_test_environment() -> None:
    """Register paths and install compatibility shims before test collection."""
    _register_compat_pythonpath()

    from src.monitoring import ensure_pandas_evidently_compat
    from tests._compat.sitecustomize import install_fastapi_body_compat_shim

    ensure_pandas_evidently_compat()
    install_fastapi_body_compat_shim()
    _configure_isolated_prefect_home()


_bootstrap_test_environment()
