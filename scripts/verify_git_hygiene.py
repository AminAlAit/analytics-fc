"""Git hygiene, author identity, and commit standard verifier.

Enforces single-line 3 to 6 word imperative commit messages, verified author
metadata (Amin Al-Ait <aminalait.de@gmail.com>), zero em/en-dashes, and zero
prohibited attribution or tool tags across diffs, commits, and hooks.
"""

from __future__ import annotations

import argparse
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

EXPECTED_AUTHOR_NAME = "Amin Al-Ait"
EXPECTED_AUTHOR_EMAIL = "aminalait.de@gmail.com"
MIN_WORDS = 3
MAX_WORDS = 6

# Construct prohibited patterns modularly to avoid false self-scan matches.
_PROHIBITED_WORD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("marker_ag", re.compile(r"\b" + "anti" + "gravity" + r"\b", re.IGNORECASE)),
    ("marker_cl", re.compile(r"\b" + "cla" + "ude" + r"\b", re.IGNORECASE)),
    ("marker_gm", re.compile(r"\b" + "gem" + "ini" + r"\b", re.IGNORECASE)),
    ("marker_cu", re.compile(r"\b" + "cur" + "sor" + r"\b", re.IGNORECASE)),
    ("marker_gr", re.compile(r"\b" + "gr" + "ok" + r"\b", re.IGNORECASE)),
    ("marker_cg", re.compile(r"\b" + "chat" + "gpt" + r"\b", re.IGNORECASE)),
    ("marker_cp", re.compile(r"\b" + "copi" + "lot" + r"\b", re.IGNORECASE)),
    ("marker_oa", re.compile(r"\b" + "open" + "ai" + r"\b", re.IGNORECASE)),
    ("marker_an", re.compile(r"\b" + "anthr" + "opic" + r"\b", re.IGNORECASE)),
    ("marker_bot", re.compile(r"(?:\b" + "bo" + r"t\b|\[" + "bo" + r"t\])", re.IGNORECASE)),
    (
        "marker_coauth",
        re.compile("co" + "-" + "authored" + "-" + "by", re.IGNORECASE),
    ),
    (
        "marker_gen",
        re.compile(r"\b" + "generated" + r"\s+" + "with" + r"\s+" + "a" + r"i\b", re.IGNORECASE),
    ),
)

_DASH_CHARS: tuple[tuple[str, str], ...] = (
    ("em_dash", "\u2014"),
    ("en_dash", "\u2013"),
)


def scan_text_for_violations(text: str) -> list[str]:
    """Scan commit message or diff text for prohibited tags and typography."""
    violations: list[str] = []
    for label, ch in _DASH_CHARS:
        if ch in text:
            violations.append(f"prohibited_{label}")
    for label, pattern in _PROHIBITED_WORD_PATTERNS:
        if pattern.search(text):
            violations.append(f"prohibited_{label}")
    return violations


def _strip_comment_lines(raw_msg: str) -> list[str]:
    """Remove git comment lines and trailing blank lines from a commit message."""
    lines = [line for line in raw_msg.splitlines() if not line.startswith("#")]
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and not lines[0].strip():
        lines.pop(0)
    return lines


def validate_commit_message(
    msg: str,
    check_word_count: bool = True,
) -> tuple[bool, list[str]]:
    """Validate commit message format, word count (3 to 6), and content hygiene."""
    violations: list[str] = []
    lines = _strip_comment_lines(msg)
    if not lines:
        return False, ["empty_commit_message"]

    if len(lines) > 1:
        violations.append(f"multiline_commit_message:{len(lines)}_lines")

    subject = lines[0].strip()
    words = [w for w in subject.split() if w]
    if check_word_count and not (MIN_WORDS <= len(words) <= MAX_WORDS):
        violations.append(f"word_count_out_of_range:{len(words)}")

    if subject and not subject[0].isupper():
        violations.append("must_start_capitalized")

    violations.extend(scan_text_for_violations("\n".join(lines)))
    return len(violations) == 0, violations


def validate_author_identity(author_name: str, author_email: str) -> tuple[bool, list[str]]:
    """Verify commit author identity matches the repository owner."""
    violations: list[str] = []
    if author_name.strip() != EXPECTED_AUTHOR_NAME:
        violations.append(f"unexpected_author_name:{author_name.strip()}")
    if author_email.strip() != EXPECTED_AUTHOR_EMAIL:
        violations.append(f"unexpected_author_email:{author_email.strip()}")
    return len(violations) == 0, violations


def scan_staged_diff(repo_root: Path | None = None) -> list[str]:
    """Scan added lines in staged git diff for prohibited markers and dashes."""
    root = repo_root or Path(__file__).resolve().parent.parent
    res = subprocess.run(
        ["git", "diff", "--cached", "-U0"],
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    if res.returncode != 0:
        raise RuntimeError(f"git diff --cached failed: {res.stderr.strip()}")

    added_lines = [
        line[1:]
        for line in res.stdout.splitlines()
        if line.startswith("+") and not line.startswith("+++ ")
    ]
    if not added_lines:
        return []
    return scan_text_for_violations("\n".join(added_lines))


def audit_git_history(
    repo_root: Path | None = None,
    max_count: int = 120,
    strict_word_count_recent: int = 100,
) -> dict[str, Any]:
    """Audit git commit history for identity, message standards, and zero violations."""
    root = repo_root or Path(__file__).resolve().parent.parent
    res = subprocess.run(
        [
            "git",
            "log",
            "-n",
            str(max_count),
            "--format=%H%x1f%an%x1f%ae%x1f%cn%x1f%ce%x1f%B%x1e",
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    if res.returncode != 0:
        raise RuntimeError(f"git log failed: {res.stderr.strip()}")

    raw_records = [rec for rec in res.stdout.split("\x1e") if rec.strip()]
    findings: list[dict[str, Any]] = []

    for idx, rec in enumerate(raw_records):
        parts = rec.lstrip("\r\n").split("\x1f")
        if len(parts) < 6:
            continue
        sha, aname, aemail, cname, cemail, body = parts[:6]
        commit_issues: list[str] = []

        ok_author, author_errs = validate_author_identity(aname, aemail)
        if not ok_author:
            commit_issues.extend(author_errs)

        ok_committer, committer_errs = validate_author_identity(cname, cemail)
        if not ok_committer:
            commit_issues.extend([f"committer_{e}" for e in committer_errs])

        enforce_wc = idx < strict_word_count_recent
        ok_msg, msg_errs = validate_commit_message(body, check_word_count=enforce_wc)
        if not ok_msg:
            commit_issues.extend(msg_errs)

        if commit_issues:
            findings.append(
                {
                    "commit": sha[:7],
                    "subject": body.strip().splitlines()[0] if body.strip() else "",
                    "violations": commit_issues,
                }
            )

    return {
        "commits_audited": len(raw_records),
        "strict_word_count_window": strict_word_count_recent,
        "violation_count": len(findings),
        "findings": findings,
        "clean": len(findings) == 0,
    }


def _resolve_git_hooks_dir(repo_root: Path) -> Path:
    """Resolve the .git/hooks directory for standard repos or git worktrees."""
    dot_git = repo_root / ".git"
    if dot_git.is_dir():
        return dot_git / "hooks"
    if dot_git.is_file():
        content = dot_git.read_text(encoding="utf-8").strip()
        if content.startswith("gitdir:"):
            git_dir = Path(content.split(":", 1)[1].strip())
            if not git_dir.is_absolute():
                git_dir = (repo_root / git_dir).resolve()
            return git_dir / "hooks"
    raise FileNotFoundError(f"Could not locate .git directory under {repo_root}")


def install_commit_msg_hook(repo_root: Path | None = None) -> Path:
    """Install .git/hooks/commit-msg invoking verify_git_hygiene.py."""
    root = repo_root or Path(__file__).resolve().parent.parent
    hooks_dir = _resolve_git_hooks_dir(root)
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook_path = hooks_dir / "commit-msg"

    script_rel = "scripts/verify_git_hygiene.py"
    hook_body = (
        "#!/usr/bin/env sh\n"
        'REPO_ROOT="$(git rev-parse --show-toplevel)"\n'
        'if [ -x "$REPO_ROOT/.venv/Scripts/python.exe" ]; then\n'
        '  PY_BIN="$REPO_ROOT/.venv/Scripts/python.exe"\n'
        'elif [ -x "$REPO_ROOT/.venv/bin/python" ]; then\n'
        '  PY_BIN="$REPO_ROOT/.venv/bin/python"\n'
        "else\n"
        '  PY_BIN="$(command -v python3 || command -v python)"\n'
        "fi\n"
        f'exec "$PY_BIN" "$REPO_ROOT/{script_rel}" --commit-msg-file "$1" --check-env-author\n'
    )
    hook_path.write_text(hook_body, encoding="utf-8", newline="\n")
    current_mode = hook_path.stat().st_mode
    hook_path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return hook_path


def _resolve_active_git_author(repo_root: Path) -> tuple[str, str]:
    """Read author name and email from env vars or git config."""
    name = os.environ.get("GIT_AUTHOR_NAME", "").strip()
    email = os.environ.get("GIT_AUTHOR_EMAIL", "").strip()
    if not name:
        name = subprocess.check_output(
            ["git", "config", "user.name"],
            cwd=str(repo_root),
            text=True,
            encoding="utf-8",
        ).strip()
    if not email:
        email = subprocess.check_output(
            ["git", "config", "user.email"],
            cwd=str(repo_root),
            text=True,
            encoding="utf-8",
        ).strip()
    return name, email


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for commit-msg hook, staged diff scan, and history audit."""
    parser = argparse.ArgumentParser(description="Verify Analytics FC git hygiene standards.")
    parser.add_argument("--commit-msg-file", type=Path, default=None)
    parser.add_argument("--check-env-author", action="store_true")
    parser.add_argument("--install-hook", action="store_true")
    parser.add_argument("--audit-history", action="store_true")
    parser.add_argument("--scan-staged", action="store_true")
    parser.add_argument("--max-count", type=int, default=120)
    parser.add_argument("--strict-recent", type=int, default=100)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent

    if args.install_hook:
        hook = install_commit_msg_hook(repo_root=repo_root)
        print(f"Installed commit-msg hook at {hook}")

    if args.commit_msg_file is not None:
        raw_msg = args.commit_msg_file.read_text(encoding="utf-8")
        ok_msg, errs = validate_commit_message(raw_msg, check_word_count=True)
        if not ok_msg:
            print(f"Commit message rejected: {errs}", file=sys.stderr)
            return 1

    if args.check_env_author:
        aname, aemail = _resolve_active_git_author(repo_root)
        ok_auth, aerrs = validate_author_identity(aname, aemail)
        if not ok_auth:
            print(f"Author identity rejected: {aerrs}", file=sys.stderr)
            return 1

    if args.scan_staged:
        staged_errs = scan_staged_diff(repo_root=repo_root)
        if staged_errs:
            print(f"Staged diff violations: {staged_errs}", file=sys.stderr)
            return 1

    if args.audit_history:
        report = audit_git_history(
            repo_root=repo_root,
            max_count=args.max_count,
            strict_word_count_recent=args.strict_recent,
        )
        print(
            f"Audited {report['commits_audited']} commits: "
            f"violations={report['violation_count']} clean={report['clean']}"
        )
        if not report["clean"]:
            for item in report["findings"]:
                print(f"  {item['commit']}: {item['violations']}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
