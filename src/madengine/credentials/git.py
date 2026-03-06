"""Git token handling utilities."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def configure_git_credentials(token: str) -> None:
    """Configure git to use a token for HTTPS authentication."""
    os.environ["GITHUB_TOKEN"] = token
    os.environ["GIT_ASKPASS"] = "echo"
    os.environ["GIT_TERMINAL_PROMPT"] = "0"


def get_git_token(env_var: str = "GITHUB_TOKEN", token_file: str | None = None) -> str | None:
    """Retrieve git token from environment or file."""
    token = os.environ.get(env_var)
    if token:
        return token

    if token_file:
        path = Path(os.path.expanduser(token_file))
        if path.exists():
            return path.read_text().strip()

    return None


def verify_git_access(repo_url: str) -> bool:
    """Verify git can access a repository."""
    try:
        result = subprocess.run(
            ["git", "ls-remote", repo_url],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
