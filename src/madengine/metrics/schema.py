"""Metrics data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class JobMetrics:
    """Schema for collected job metrics."""

    # Identification
    job_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Configuration reference
    recipe: str | None = None
    model: str | None = None
    image: str | None = None

    # Resource allocation
    num_nodes: int | None = None
    gpus_per_node: int | None = None
    total_gpus: int | None = None

    # Performance metrics
    throughput: float | None = None
    throughput_unit: str = "tokens/sec"
    loss: float | None = None
    gpu_memory_gb: float | None = None
    gpu_utilization_pct: float | None = None

    # Timing
    duration_seconds: float | None = None
    training_time_seconds: float | None = None

    # Status
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "timestamp": self.timestamp.isoformat(),
            "recipe": self.recipe,
            "model": self.model,
            "image": self.image,
            "num_nodes": self.num_nodes,
            "gpus_per_node": self.gpus_per_node,
            "total_gpus": self.total_gpus,
            "throughput": self.throughput,
            "throughput_unit": self.throughput_unit,
            "loss": self.loss,
            "gpu_memory_gb": self.gpu_memory_gb,
            "gpu_utilization_pct": self.gpu_utilization_pct,
            "duration_seconds": self.duration_seconds,
            "training_time_seconds": self.training_time_seconds,
            "success": self.success,
            "error": self.error,
        }
