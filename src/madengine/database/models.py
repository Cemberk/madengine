"""Pydantic schemas for database records."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class RunRecord(BaseModel):
    """Database record for a pipeline run."""

    model_config = {"extra": "allow"}

    id: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    recipe: Optional[str] = None
    config: dict[str, Any] = Field(default_factory=dict)

    success: bool = True
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

    image: Optional[str] = None
    duration_seconds: Optional[float] = None
