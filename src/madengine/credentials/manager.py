"""Unified credential interface."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Credentials:
    """Container for credential values."""

    git_token: str | None = None
    registry_auth: dict | None = None


class CredentialManager:
    """Manages credentials for git and container registries.

    Loads credentials from:
    1. Environment variables
    2. Credential files
    3. System credential stores (docker config)
    """

    def __init__(self, cfg: dict[str, Any] | Any):
        if hasattr(cfg, "__getitem__"):
            self.cfg = cfg
        else:
            self.cfg = {}
        self._credentials: Credentials | None = None

    def load(self) -> Credentials:
        if self._credentials:
            return self._credentials

        git_token = self._load_git_token()
        registry_auth = self._load_registry_auth()

        self._credentials = Credentials(
            git_token=git_token,
            registry_auth=registry_auth,
        )
        return self._credentials

    def setup_environment(self) -> None:
        """Set up environment variables for child processes (AIImageBuilder, AIWorkloads)."""
        creds = self.load()

        if creds.git_token:
            os.environ["GITHUB_TOKEN"] = creds.git_token
            os.environ["GIT_ASKPASS"] = "echo"
            os.environ["GIT_TERMINAL_PROMPT"] = "0"

    def _load_git_token(self) -> str | None:
        git_cfg = self.cfg.get("git", {}) if hasattr(self.cfg, "get") else {}

        env_var = git_cfg.get("token_env", "GITHUB_TOKEN") if git_cfg else "GITHUB_TOKEN"
        token = os.environ.get(env_var)
        if token:
            return token

        token_file = git_cfg.get("token_file") if git_cfg else None
        if token_file:
            path = Path(os.path.expanduser(token_file))
            if path.exists():
                return path.read_text().strip()

        return None

    def _load_registry_auth(self) -> dict | None:
        registry_cfg = self.cfg.get("registry", {}) if hasattr(self.cfg, "get") else {}

        docker_config = (
            registry_cfg.get("docker_config", "~/.docker/config.json")
            if registry_cfg
            else "~/.docker/config.json"
        )
        path = Path(os.path.expanduser(docker_config))

        if path.exists():
            with open(path) as f:
                return json.load(f)

        return None
