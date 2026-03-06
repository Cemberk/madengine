"""MinIO S3-compatible data provider."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from .base import DataProvider, StagingResult


class MinIODataProvider(DataProvider):
    """Data provider for MinIO S3-compatible storage."""

    def __init__(self, cfg: dict[str, Any]):
        self.endpoint = cfg.get("endpoint")
        self.bucket = cfg.get("bucket")
        self.local_cache = Path(
            os.path.expanduser(cfg.get("local_cache", "/tmp/data_cache"))
        )

        access_key_env = cfg.get("access_key_env", "MINIO_ACCESS_KEY")
        secret_key_env = cfg.get("secret_key_env", "MINIO_SECRET_KEY")
        self.access_key = os.environ.get(access_key_env)
        self.secret_key = os.environ.get(secret_key_env)

    @property
    def name(self) -> str:
        return "minio"

    def stage(self, dataset: str) -> StagingResult:
        if not self.endpoint or not self.bucket:
            return StagingResult(
                success=False, error="MinIO endpoint and bucket must be configured"
            )

        s3_path = f"s3://{self.bucket}/{dataset}"
        local = self.local_cache / dataset
        local.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "aws", "s3", "sync",
            s3_path,
            str(local),
            "--endpoint-url", self.endpoint,
        ]

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
                    success=False, error=f"MinIO sync failed: {result.stderr}"
                )

            size = sum(f.stat().st_size for f in local.rglob("*") if f.is_file())
            return StagingResult(success=True, local_path=str(local), size_bytes=size)

        except subprocess.TimeoutExpired:
            return StagingResult(
                success=False, error="MinIO sync timed out after 2 hours"
            )

    def exists(self, dataset: str) -> bool:
        if not self.endpoint or not self.bucket:
            return False

        cmd = [
            "aws", "s3", "ls",
            f"s3://{self.bucket}/{dataset}",
            "--endpoint-url", self.endpoint,
        ]

        env = os.environ.copy()
        if self.access_key:
            env["AWS_ACCESS_KEY_ID"] = self.access_key
        if self.secret_key:
            env["AWS_SECRET_ACCESS_KEY"] = self.secret_key

        result = subprocess.run(cmd, capture_output=True, env=env)
        return result.returncode == 0
