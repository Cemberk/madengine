"""Parse job output and extract performance metrics."""

from __future__ import annotations

import re
from typing import Any

from .schema import JobMetrics

DEFAULT_PATTERNS: dict[str, list[str]] = {
    "throughput": [
        r"throughput[:\s]+([0-9.]+)",
        r"tokens[/\s]sec[:\s]+([0-9.]+)",
        r"samples[/\s]sec[:\s]+([0-9.]+)",
        r"([0-9.]+)\s*tokens/s",
    ],
    "loss": [
        r"loss[:\s]+([0-9.]+)",
        r"train[_\s]loss[:\s]+([0-9.]+)",
    ],
    "gpu_memory": [
        r"memory[:\s]+([0-9.]+)\s*GB",
        r"gpu[_\s]memory[:\s]+([0-9.]+)",
        r"allocated[:\s]+([0-9.]+)\s*GB",
    ],
    "gpu_utilization": [
        r"gpu[_\s]util[:\s]+([0-9.]+)",
        r"utilization[:\s]+([0-9.]+)%",
    ],
}


class MetricsCollector:
    """Collects metrics from job output by parsing stdout/stderr for known patterns."""

    def __init__(self, cfg: dict[str, Any] | Any = None):
        if cfg is None or not hasattr(cfg, "get"):
            cfg = {}
        self.cfg = cfg

        self.patterns: dict[str, list[str]] = dict(DEFAULT_PATTERNS)
        custom_patterns = cfg.get("patterns", {})
        if hasattr(custom_patterns, "items"):
            for key, patterns in custom_patterns.items():
                if isinstance(patterns, str):
                    patterns = [patterns]
                self.patterns[key] = list(patterns) if not isinstance(patterns, list) else patterns

    def collect(self, exec_result: dict[str, Any]) -> dict[str, Any]:
        """Collect metrics from execution result dict (stdout, stderr, duration_seconds, etc.)."""
        output = exec_result.get("stdout", "") + "\n" + exec_result.get("stderr", "")

        metrics = JobMetrics(
            duration_seconds=exec_result.get("duration_seconds"),
            success=exec_result.get("success", True),
            error=exec_result.get("error"),
        )

        metrics.throughput = self._extract_metric(output, "throughput")
        metrics.loss = self._extract_metric(output, "loss")
        metrics.gpu_memory_gb = self._extract_metric(output, "gpu_memory")
        metrics.gpu_utilization_pct = self._extract_metric(output, "gpu_utilization")

        return metrics.to_dict()

    def _extract_metric(self, output: str, metric_name: str) -> float | None:
        patterns = self.patterns.get(metric_name, [])
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, IndexError):
                    continue
        return None
