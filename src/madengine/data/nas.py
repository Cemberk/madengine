"""NAS data provider via SSH/rsync."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from .base import DataProvider, StagingResult


class NASDataProvider(DataProvider):
    """Data provider for NAS via SSH/rsync."""

    def __init__(self, cfg: dict[str, Any]):
        self.host = cfg.get("host")
        self.user = cfg.get("user", os.environ.get("USER"))
        self.key_file = Path(os.path.expanduser(cfg.get("key_file", "~/.ssh/id_rsa")))
        self.remote_path = cfg.get("remote_path", "/shared/datasets")
        self.local_cache = Path(
            os.path.expanduser(cfg.get("local_cache", "/tmp/data_cache"))
        )

    @property
    def name(self) -> str:
        return "nas"

    def stage(self, dataset: str) -> StagingResult:
        if not self.host:
            return StagingResult(success=False, error="NAS host not configured")

        remote = f"{self.user}@{self.host}:{self.remote_path}/{dataset}"
        local = self.local_cache / dataset
        local.parent.mkdir(parents=True, exist_ok=True)

        cmd = ["rsync", "-avz", "--progress"]

        if self.key_file.exists():
            cmd.extend(["-e", f"ssh -i {self.key_file}"])

        cmd.extend([f"{remote}/", str(local)])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

            if result.returncode != 0:
                return StagingResult(success=False, error=f"rsync failed: {result.stderr}")

            size = sum(f.stat().st_size for f in local.rglob("*") if f.is_file())

            return StagingResult(success=True, local_path=str(local), size_bytes=size)

        except subprocess.TimeoutExpired:
            return StagingResult(success=False, error="rsync timed out after 1 hour")

    def exists(self, dataset: str) -> bool:
        if not self.host:
            return False

        cmd = ["ssh"]
        if self.key_file.exists():
            cmd.extend(["-i", str(self.key_file)])
        cmd.extend(
            [f"{self.user}@{self.host}", f"test -e {self.remote_path}/{dataset}"]
        )

        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0
