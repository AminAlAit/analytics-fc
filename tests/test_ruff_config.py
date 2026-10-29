"""Unit and integration tests verifying Ruff configuration and repository linting hygiene."""

import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = BASE_DIR / "pyproject.toml"
RUFF_EXE = BASE_DIR / ".venv" / "Scripts" / "ruff.exe"


def test_pyproject_toml_configuration():
    """Verify pyproject.toml defines required Ruff lint and format rule sets."""
    assert PYPROJECT_PATH.exists(), "pyproject.toml missing in repository root"
    content = PYPROJECT_PATH.read_text(encoding="utf-8")

    assert "[tool.ruff]" in content
    assert "line-length = 100" in content
    assert "[tool.ruff.lint]" in content

    # Verify required rule groups: E, F, W, I, UP
    for rule_group in ['"E"', '"F"', '"W"', '"I"', '"UP"']:
        assert rule_group in content, f"Rule group {rule_group} missing from pyproject.toml"


def test_ruff_check_repository_clean():
    """Verify ruff check . exits with 0 clean status across repository."""
    cmd = [str(RUFF_EXE if RUFF_EXE.exists() else "ruff"), "check", "."]
    proc = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"ruff check failed with output:\n{proc.stdout}\n{proc.stderr}"


def test_ruff_format_check_compliant():
    """Verify ruff format --check . confirms zero unformatted files."""
    cmd = [str(RUFF_EXE if RUFF_EXE.exists() else "ruff"), "format", "--check", "."]
    proc = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"ruff format check failed with output:\n{proc.stdout}\n{proc.stderr}"
    )
