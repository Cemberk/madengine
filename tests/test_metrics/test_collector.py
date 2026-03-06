"""Tests for metrics collector."""

from __future__ import annotations

import pytest

from madengine.metrics.collector import MetricsCollector


class TestMetricsCollector:
    def test_collect_throughput(self):
        collector = MetricsCollector()
        result = collector.collect(
            {"stdout": "throughput: 1234.5 tokens/sec", "stderr": "", "success": True}
        )
        assert result["throughput"] == 1234.5

    def test_collect_loss(self):
        collector = MetricsCollector()
        result = collector.collect(
            {"stdout": "train_loss: 0.0042", "stderr": "", "success": True}
        )
        assert result["loss"] == 0.0042

    def test_collect_gpu_memory(self):
        collector = MetricsCollector()
        result = collector.collect(
            {"stdout": "memory: 78.5 GB", "stderr": "", "success": True}
        )
        assert result["gpu_memory_gb"] == 78.5

    def test_collect_gpu_utilization(self):
        collector = MetricsCollector()
        result = collector.collect(
            {"stdout": "gpu_util: 95.2", "stderr": "", "success": True}
        )
        assert result["gpu_utilization_pct"] == 95.2

    def test_collect_duration(self):
        collector = MetricsCollector()
        result = collector.collect(
            {"stdout": "", "stderr": "", "success": True, "duration_seconds": 3600.0}
        )
        assert result["duration_seconds"] == 3600.0

    def test_collect_no_metrics(self):
        collector = MetricsCollector()
        result = collector.collect(
            {"stdout": "no metrics here", "stderr": "", "success": True}
        )
        assert result["throughput"] is None
        assert result["loss"] is None

    def test_collect_from_stderr(self):
        collector = MetricsCollector()
        result = collector.collect(
            {"stdout": "", "stderr": "loss: 0.123", "success": True}
        )
        assert result["loss"] == 0.123

    def test_custom_patterns(self):
        collector = MetricsCollector(
            {"patterns": {"throughput": r"perf=([0-9.]+)"}}
        )
        result = collector.collect(
            {"stdout": "perf=5678.9", "stderr": "", "success": True}
        )
        assert result["throughput"] == 5678.9

    def test_tokens_per_sec_pattern(self):
        collector = MetricsCollector()
        result = collector.collect(
            {"stdout": "achieved 9876.5 tokens/s", "stderr": "", "success": True}
        )
        assert result["throughput"] == 9876.5
