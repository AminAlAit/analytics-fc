"""Unit tests verifying GitHub Actions CI workflow configuration and schema validity."""

from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
CI_WORKFLOW_PATH = BASE_DIR / ".github" / "workflows" / "ci.yml"


def test_ci_workflow_file_exists():
    """Verify GitHub Actions CI workflow file exists in expected directory."""
    assert CI_WORKFLOW_PATH.exists(), f"CI workflow missing at {CI_WORKFLOW_PATH}"


def test_ci_workflow_valid_yaml():
    """Verify GitHub Actions CI workflow parses as valid YAML."""
    assert CI_WORKFLOW_PATH.exists()
    content = CI_WORKFLOW_PATH.read_text(encoding="utf-8")
    data = yaml.safe_load(content)
    assert isinstance(data, dict), "CI workflow YAML must be a mapping"
    assert "name" in data
    assert "jobs" in data


def test_ci_workflow_triggers():
    """Verify CI workflow is configured for push and pull_request on main."""
    content = CI_WORKFLOW_PATH.read_text(encoding="utf-8")
    data = yaml.safe_load(content)

    # In PyYAML 'on' might be parsed as boolean True or string 'on'
    on_trigger = data.get(True) or data.get("on")
    assert on_trigger is not None, "Missing trigger configuration"
    assert "push" in on_trigger
    assert "pull_request" in on_trigger


def test_ci_workflow_quality_gate_job():
    """Verify CI workflow contains required linting, typing, and test execution steps."""
    content = CI_WORKFLOW_PATH.read_text(encoding="utf-8")
    data = yaml.safe_load(content)

    jobs = data.get("jobs", {})
    assert "quality-gate" in jobs, "Missing 'quality-gate' job in CI workflow"

    job = jobs["quality-gate"]
    assert job.get("runs-on") == "ubuntu-latest"

    steps = job.get("steps", [])
    step_runs = " ".join([s.get("run", "") for s in steps])

    assert "ruff check" in step_runs, "Missing ruff check step in CI workflow"
    assert "ruff format --check" in step_runs, "Missing ruff format check step in CI workflow"
    assert "mypy" in step_runs, "Missing mypy step in CI workflow"
    assert "pytest" in step_runs, "Missing pytest step in CI workflow"
