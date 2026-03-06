"""Abstract adapter interface for black-box tool integrations."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AdapterResult:
    """Result from adapter execution."""

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "returncode": self.returncode,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class Adapter(ABC):
    """Abstract base class for tool adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Adapter name for logging."""

    @abstractmethod
    def execute(self, config: dict[str, Any]) -> AdapterResult:
        """Execute the adapter with given configuration."""

    @abstractmethod
    def validate_config(self, config: dict[str, Any]) -> list[str]:
        """Validate configuration before execution. Returns error messages."""
