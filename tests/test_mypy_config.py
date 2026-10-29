"""Unit and integration tests verifying Mypy configuration and static type safety."""

import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = BASE_DIR / "pyproject.toml"
MYPY_EXE = BASE_DIR / ".venv" / "Scripts" / "mypy.exe"


def test_pyproject_toml_mypy_configuration():
    """Verify pyproject.toml defines strict static typing configuration for Mypy."""
    assert PYPROJECT_PATH.exists(), "pyproject.toml missing in repository root"
    content = PYPROJECT_PATH.read_text(encoding="utf-8")

    assert "[tool.mypy]" in content
    assert 'python_version = "3.11"' in content
    assert "src" in content
    assert "scripts" in content
    assert "pipelines" in content
    assert "check_untyped_defs = true" in content
    assert "strict_optional = true" in content


def test_mypy_execution_zero_errors():
    """Verify mypy execution passes with zero type errors across configured modules."""
    cmd = [str(MYPY_EXE if MYPY_EXE.exists() else "mypy")]
    proc = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"mypy check failed with output:\n{proc.stdout}\n{proc.stderr}"
    assert "Success: no issues found" in proc.stdout
