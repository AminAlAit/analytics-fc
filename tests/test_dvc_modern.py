"""Tests verifying DVC hash tracking and bronze lakehouse configuration."""

from pathlib import Path

import yaml


def test_bronze_dvc_tracking_exists():
    """Verify that data/bronze.dvc exists and defines valid md5 hash."""
    candidate = Path("analytics-fc/data/bronze.dvc")
    if not candidate.exists():
        candidate = Path("data/bronze.dvc")
    assert candidate.exists(), "data/bronze.dvc must exist"

    with open(candidate, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "outs" in data
    assert len(data["outs"]) >= 1
    out_entry = data["outs"][0]
    assert out_entry["path"] == "bronze"
    assert "md5" in out_entry
    assert len(out_entry["md5"]) > 0


def test_bronze_data_ignored_by_git():
    """Verify that data/.gitignore excludes bronze binaries from Git."""
    candidate = Path("analytics-fc/data/.gitignore")
    if not candidate.exists():
        candidate = Path("data/.gitignore")
    assert candidate.exists()

    with open(candidate, encoding="utf-8") as f:
        content = f.read()

    assert "/bronze" in content or "bronze" in content
