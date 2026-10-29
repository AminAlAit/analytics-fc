"""Installs git automation hooks for continuous LikeC4 synchronization."""

import os
import stat
import sys
from pathlib import Path

HOOK_CONTENT = """#!/bin/sh
# Analytics FC: Automated LikeC4 synchronization hook
python architecture/scripts/sync_status.py --once > /dev/null 2>&1 &
exit 0
"""


def install() -> None:
    git_dir = Path(".git")
    if not git_dir.exists():
        sys.stderr.write("Error: .git directory not found. Run from repository root.\n")
        sys.exit(1)

    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)
    post_commit_hook = hooks_dir / "post-commit"

    post_commit_hook.write_text(HOOK_CONTENT, encoding="utf-8")

    # Add executable permissions on Unix systems
    if sys.platform != "win32":
        current_stat = os.stat(post_commit_hook)
        os.chmod(post_commit_hook, current_stat.st_mode | stat.S_IEXEC)

    print(f"Installed post-commit hook successfully to {post_commit_hook}")


if __name__ == "__main__":
    install()
