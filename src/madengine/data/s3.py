"""AWS S3 data provider."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from .base import DataProvider, StagingResult


class S3DataProvider(DataProvider):
    """Data provider for AWS S3."""

    def __init__(self, cfg: dict[str, Any]):
        self.bucket = cfg.get("bucket")
        self.region = cfg.get("region", "us-east-1")
        self.local_cache = Path(
            os.path.expanduser(cfg.get("local_cache", "/tmp/data_cache"))
        )

        access_key_env = cfg.get("access_key_env", "AWS_ACCESS_KEY_ID")
        secret_key_env = cfg.get("secret_key_env", "AWS_SECRET_ACCESS_KEY")
        self.access_key = os.environ.get(access_key_env)
        self.secret_key = os.environ.get(secret_key_env)

    @property
    def name(self) -> str:
        return "s3"

    def stage(self, dataset: str) -> StagingResult:
        if not self.bucket:
            return StagingResult(success=False, error="S3 bucket not configured")

        s3_path = f"s3://{self.bucket}/{dataset}"
        local = self.local_cache / dataset
        local.parent.mkdir(parents=True, exist_ok=True)

        cmd = ["aws", "s3", "sync", s3_path, str(local), "--region", self.region]

        env = os.environ.copy()
        if self.access_key:
            env["AWS_ACCESS_KEY_ID"] = self.access_key
        if self.secret_key:
            env["AWS_SECRET_ACCESS_KEY"] = self.secret_key

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=7200, env=env
            )

            if result.returncode != 0:
                return StagingResult(
                    success=False, error=f"aws s3 sync failed: {result.stderr}"
                )

            size = sum(f.stat().st_size for f in local.rglob("*") if f.is_file())
            return StagingResult(success=True, local_path=str(local), size_bytes=size)

        except subprocess.TimeoutExpired:
            return StagingResult(
                success=False, error="S3 sync timed out after 2 hours"
            )

    def exists(self, dataset: str) -> bool:
        if not self.bucket:
            return False

        cmd = [
            "aws", "s3", "ls",
            f"s3://{self.bucket}/{dataset}",
            "--region", self.region,
        ]

        env = os.environ.copy()
        if self.access_key:
            env["AWS_ACCESS_KEY_ID"] = self.access_key
        if self.secret_key:
            env["AWS_SECRET_ACCESS_KEY"] = self.secret_key

        result = subprocess.run(cmd, capture_output=True, env=env)
        return result.returncode == 0
