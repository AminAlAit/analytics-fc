"""Shared pytest startup hooks and dependency compatibility shims."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

from src.monitoring import ensure_pandas_evidently_compat
from tests._compat.sitecustomize import install_fastapi_body_compat_shim


def _register_compat_pythonpath() -> None:
    """Prepend tests/_compat to PYTHONPATH so child subprocesses load sitecustomize."""
    compat_dir = str(Path(__file__).resolve().parent / "_compat")
    existing = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if compat_dir not in existing:
        os.environ["PYTHONPATH"] = os.pathsep.join([compat_dir, *existing])


def _configure_isolated_prefect_home() -> None:
    """Isolate Prefect SQLite state per pytest session away from user ~/.prefect."""
    tmp_dir = tempfile.mkdtemp(prefix="afc_prefect_")
    atexit.register(shutil.rmtree, tmp_dir, ignore_errors=True)
    os.environ.setdefault("PREFECT_HOME", tmp_dir)
    os.environ.setdefault(
        "PREFECT_API_DATABASE_CONNECTION_URL",
        f"sqlite+aiosqlite:///{(Path(tmp_dir) / 'prefect.db').as_posix()}",
    )


ensure_pandas_evidently_compat()
install_fastapi_body_compat_shim()
_register_compat_pythonpath()
_configure_isolated_prefect_home()
