"""Tests for metrics schema."""

from __future__ import annotations

from madengine.metrics.schema import JobMetrics


class TestJobMetrics:
    def test_defaults(self):
        m = JobMetrics()
        assert m.success is True
        assert m.throughput is None
        assert m.throughput_unit == "tokens/sec"

    def test_to_dict(self):
        m = JobMetrics(throughput=1234.5, loss=0.01, duration_seconds=600.0)
        d = m.to_dict()
        assert d["throughput"] == 1234.5
        assert d["loss"] == 0.01
        assert d["duration_seconds"] == 600.0
        assert "timestamp" in d

    def test_with_error(self):
        m = JobMetrics(success=False, error="OOM")
        d = m.to_dict()
        assert d["success"] is False
        assert d["error"] == "OOM"
