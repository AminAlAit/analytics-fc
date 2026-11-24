"""Unit tests for git hygiene verification, commit-msg hook, and history audit."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from scripts.verify_git_hygiene import (
    EXPECTED_AUTHOR_EMAIL,
    EXPECTED_AUTHOR_NAME,
    audit_git_history,
    install_commit_msg_hook,
    main,
    scan_staged_diff,
    scan_text_for_violations,
    validate_author_identity,
    validate_commit_message,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_git(repo: Path, args: list[str], extra_env: dict[str, str] | None = None) -> str:
    """Execute a deterministic git command inside a temporary test repository."""
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": EXPECTED_AUTHOR_NAME,
            "GIT_AUTHOR_EMAIL": EXPECTED_AUTHOR_EMAIL,
            "GIT_COMMITTER_NAME": EXPECTED_AUTHOR_NAME,
            "GIT_COMMITTER_EMAIL": EXPECTED_AUTHOR_EMAIL,
        }
    )
    if extra_env:
        env.update(extra_env)
    res = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return res.stdout.strip()


def test_validate_commit_message_rules() -> None:
    """Verify single-line 3-6 word messages pass while invalid forms fail."""
    ok, errs = validate_commit_message("Enforce git hygiene commit standards\n# git comment\n")
    assert ok is True
    assert errs == []

    assert validate_commit_message("Fix bug")[0] is False
    assert validate_commit_message("Add one two three four five six")[0] is False
    assert validate_commit_message("Add one two")[0] is True
    assert validate_commit_message("Add one two three four five")[0] is True

    ok_multi, errs_multi = validate_commit_message("Add feature now\n\nExtra body line")
    assert ok_multi is False
    assert any(e.startswith("multiline_commit_message") for e in errs_multi)

    bad_dash = "Add feature " + "\u2014" + " cleanly"
    assert validate_commit_message(bad_dash)[0] is False

    bad_marker = "Add feature by " + "anti" + "gravity"
    assert validate_commit_message(bad_marker)[0] is False

    bad_trailer = "Add feature now " + "co-" + "authored-" + "by me"
    assert validate_commit_message(bad_trailer)[0] is False


def test_word_boundary_scanner_avoids_false_positives() -> None:
    """Ensure benign technical words like bootstrap do not trigger false alarms."""
    assert scan_text_for_violations("Use dashboard.bootstrap.ensure_serving_assets cleanly") == []
    flagged = scan_text_for_violations("Automated commit from test[" + "bo" + "t]")
    assert len(flagged) >= 1


def test_validate_author_identity() -> None:
    """Enforce verified Amin Al-Ait author identity."""
    ok, errs = validate_author_identity(EXPECTED_AUTHOR_NAME, EXPECTED_AUTHOR_EMAIL)
    assert ok is True
    assert errs == []

    bad_ok, bad_errs = validate_author_identity("Other Dev", "other@example.com")
    assert bad_ok is False
    assert len(bad_errs) == 2


def test_install_commit_msg_hook_and_cli(tmp_path: Path) -> None:
    """Verify hook installation on temp repo (LF endings) and CLI commit-msg validation."""
    fake_repo = tmp_path / "repo"
    (fake_repo / ".git").mkdir(parents=True)
    hook_file = install_commit_msg_hook(repo_root=fake_repo)
    assert hook_file.exists()
    raw_bytes = hook_file.read_bytes()
    assert b"\r\n" not in raw_bytes
    assert "--commit-msg-file" in raw_bytes.decode("utf-8")

    msg_good = tmp_path / "COMMIT_EDITMSG_GOOD"
    msg_good.write_text("Add unit test validation\n", encoding="utf-8")
    assert main(["--commit-msg-file", str(msg_good)]) == 0

    msg_bad = tmp_path / "COMMIT_EDITMSG_BAD"
    msg_bad.write_text("Too short\n", encoding="utf-8")
    assert main(["--commit-msg-file", str(msg_bad)]) == 1


def test_audit_git_history_and_staged_diff_scanner(tmp_path: Path) -> None:
    """Verify audit_git_history and scan_staged_diff on clean and violating repos."""
    temp_repo = tmp_path / "audit_repo"
    temp_repo.mkdir()
    _run_git(temp_repo, ["init"])
    (temp_repo / "sample.txt").write_text("v1\n", encoding="utf-8")
    _run_git(temp_repo, ["add", "sample.txt"])
    assert scan_staged_diff(repo_root=temp_repo) == []
    _run_git(temp_repo, ["commit", "-m", "Initialize sample tracking file"])

    clean_report = audit_git_history(
        repo_root=temp_repo,
        max_count=10,
        strict_word_count_recent=10,
    )
    assert clean_report["commits_audited"] == 1
    assert clean_report["clean"] is True

    # Verify staged diff scanner flags em-dash additions before commit
    (temp_repo / "sample.txt").write_text("v2 " + "\u2014" + " bad dash\n", encoding="utf-8")
    _run_git(temp_repo, ["add", "sample.txt"])
    assert "prohibited_em_dash" in scan_staged_diff(repo_root=temp_repo)

    # Replace with clean staged content and commit a prohibited trailer to verify history catch
    (temp_repo / "sample.txt").write_text("v2\n", encoding="utf-8")
    _run_git(temp_repo, ["add", "sample.txt"])
    trailer_line = "Co-" + "authored-" + "by: Helper <helper@example.com>"
    _run_git(
        temp_repo,
        ["commit", "-m", f"Update sample file\n\n{trailer_line}"],
    )

    dirty_report = audit_git_history(
        repo_root=temp_repo,
        max_count=10,
        strict_word_count_recent=10,
    )
    assert dirty_report["clean"] is False
    assert dirty_report["violation_count"] == 1
    assert "prohibited_marker_coauth" in dirty_report["findings"][0]["violations"]


def test_live_repository_identity_and_self_scan() -> None:
    """Audit live repo identity and verify verify_git_hygiene.py has zero self-hits."""
    source_text = (REPO_ROOT / "scripts" / "verify_git_hygiene.py").read_text(encoding="utf-8")
    assert scan_text_for_violations(source_text) == []

    report = audit_git_history(
        repo_root=REPO_ROOT,
        max_count=120,
        strict_word_count_recent=100,
    )
    assert report["commits_audited"] >= 100
    for item in report["findings"]:
        for v in item["violations"]:
            assert "author" not in v, f"Unexpected author violation in {item['commit']}: {v}"

    real_hook = REPO_ROOT / ".git" / "hooks" / "commit-msg"
    if real_hook.exists():
        assert b"\r\n" not in real_hook.read_bytes()
        if os.name != "nt":
            assert os.access(real_hook, os.X_OK)
