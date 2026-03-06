"""Container registry authentication."""

from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def load_docker_config(config_path: str = "~/.docker/config.json") -> dict[str, Any] | None:
    """Load Docker config.json for registry auth."""
    path = Path(os.path.expanduser(config_path))
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def docker_login(registry: str, username: str, password: str) -> bool:
    """Log in to a Docker registry."""
    try:
        result = subprocess.run(
            ["docker", "login", registry, "-u", username, "--password-stdin"],
            input=password,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_registry_credentials(
    registry: str, docker_config: dict[str, Any] | None = None
) -> tuple[str, str] | None:
    """Extract credentials for a specific registry from Docker config."""
    if docker_config is None:
        docker_config = load_docker_config()
    if docker_config is None:
        return None

    auths = docker_config.get("auths", {})
    auth_entry = auths.get(registry, {})
    auth_str = auth_entry.get("auth")
    if auth_str:
        decoded = base64.b64decode(auth_str).decode("utf-8")
        if ":" in decoded:
            username, password = decoded.split(":", 1)
            return username, password

    return None
