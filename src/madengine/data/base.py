"""Abstract data provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class StagingResult:
    """Result of data staging operation."""

    success: bool
    local_path: str | None = None
    size_bytes: int | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "local_path": self.local_path,
            "size_bytes": self.size_bytes,
            "error": self.error,
        }


class DataProvider(ABC):
    """Abstract base class for data providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""

    @abstractmethod
    def stage(self, dataset: str) -> StagingResult:
        """Stage dataset to local filesystem."""

    @abstractmethod
    def exists(self, dataset: str) -> bool:
        """Check if dataset exists in this provider."""


def get_data_provider(cfg: dict[str, Any] | Any) -> DataProvider:
    """Factory function to create data provider from config."""
    if not hasattr(cfg, "get"):
        cfg = {}

    provider_type = cfg.get("provider", "local")
    provider_cfg = cfg.get(provider_type, {})

    if not hasattr(provider_cfg, "get"):
        provider_cfg = {}

    if provider_type == "local":
        from .local import LocalDataProvider

        return LocalDataProvider(provider_cfg)
    elif provider_type == "nas":
        from .nas import NASDataProvider

        return NASDataProvider(provider_cfg)
    elif provider_type == "s3":
        from .s3 import S3DataProvider

        return S3DataProvider(provider_cfg)
    elif provider_type == "minio":
        from .minio import MinIODataProvider

        return MinIODataProvider(provider_cfg)
    else:
        raise ValueError(f"Unknown data provider: {provider_type}")
