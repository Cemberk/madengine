"""Local filesystem data provider."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import DataProvider, StagingResult


class LocalDataProvider(DataProvider):
    """Data provider for local filesystem - no staging needed, just verification."""

    def __init__(self, cfg: dict[str, Any] | Any):
        base_path = cfg.get("base_path", "/data") if hasattr(cfg, "get") else "/data"
        self.base_path = Path(base_path)

    @property
    def name(self) -> str:
        return "local"

    def stage(self, dataset: str) -> StagingResult:
        path = self.base_path / dataset

        if not path.exists():
            return StagingResult(success=False, error=f"Dataset not found: {path}")

        if path.is_file():
            size = path.stat().st_size
        else:
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

        return StagingResult(success=True, local_path=str(path), size_bytes=size)

    def exists(self, dataset: str) -> bool:
        return (self.base_path / dataset).exists()
