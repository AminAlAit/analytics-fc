import subprocess
import sys
from pathlib import Path
import pytest
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DVC_PATH = BASE_DIR / "data" / "raw.dvc"
PROCESSED_DVC_PATH = BASE_DIR / "data" / "processed.dvc"
DVC_CONFIG_PATH = BASE_DIR / ".dvc" / "config"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


def test_raw_dvc_pointer_contract():
    assert RAW_DVC_PATH.exists(), f"Missing raw DVC pointer at {RAW_DVC_PATH}"
    with open(RAW_DVC_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "outs" in data and len(data["outs"]) == 1
    out = data["outs"][0]
    assert out["path"] == "raw"
    assert "md5" in out and out["md5"].endswith(".dir")
    assert out.get("nfiles", 0) >= 80, f"Expected at least 80 raw files tracked, found {out.get('nfiles')}"
    assert out.get("size", 0) > 2_000_000_000, f"Expected size > 2GB, found {out.get('size')}"

    # Verify tracked filesystem assets exist
    assert RAW_DATA_DIR.exists() and RAW_DATA_DIR.is_dir()
    archive_file = RAW_DATA_DIR / "fifaaccuracy_archive" / "fifatable.csv"
    assert archive_file.exists(), f"Historical archive missing at {archive_file}"


def test_processed_dvc_pointer_contract():
    assert PROCESSED_DVC_PATH.exists(), f"Missing processed DVC pointer at {PROCESSED_DVC_PATH}"
    with open(PROCESSED_DVC_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "outs" in data and len(data["outs"]) == 1
    out = data["outs"][0]
    assert out["path"] == "processed"
    assert "md5" in out and len(out["md5"]) > 0
    assert out.get("nfiles", 0) == 1

    # Verify underlying gallery file exists
    gallery_file = PROCESSED_DATA_DIR / "superstar_gallery.json"
    assert gallery_file.exists(), f"Gallery artifact missing at {gallery_file}"


def test_dvc_configuration_invariants():
    assert DVC_CONFIG_PATH.exists(), f"Missing DVC configuration at {DVC_CONFIG_PATH}"
    with open(DVC_CONFIG_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    assert "type = copy" in content, "DVC cache.type copy must be configured for Windows filesystem safety"
    assert "analytics = false" in content, "DVC telemetry must be disabled"
    assert "remote = local_storage" in content, "Default remote local_storage must be set"


def test_dvc_status_clean():
    dvc_exe = BASE_DIR / ".venv" / "Scripts" / "dvc.exe"
    if not dvc_exe.exists():
        pytest.skip("dvc executable not found in .venv/Scripts")

    proc = subprocess.run(
        [str(dvc_exe), "status"],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        check=False
    )
    assert proc.returncode == 0, f"dvc status failed with stderr: {proc.stderr}"
    assert "Data and pipelines are up to date." in proc.stdout, f"Unexpected dvc status output: {proc.stdout}"
