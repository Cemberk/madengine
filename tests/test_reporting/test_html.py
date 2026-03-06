"""Tests for HTML exporter."""

from __future__ import annotations

import pytest

from madengine.reporting.html import HTMLExporter


class TestHTMLExporter:
    def test_export_generates_html(self, tmp_path):
        records = [
            {
                "timestamp": "2026-01-01T00:00:00",
                "recipe": "test",
                "config": {"model": "llama3"},
                "success": True,
                "metrics": {"throughput": 1234.5, "loss": 0.01},
                "duration_seconds": 600,
            }
        ]

        output = tmp_path / "report.html"
        exporter = HTMLExporter()
        exporter.export(output_path=output, records=records)

        html = output.read_text()
        assert "<!DOCTYPE html>" in html
        assert "madengine" in html
        assert "1234.5" in html
        assert "Success" in html

    def test_export_shows_failure(self, tmp_path):
        records = [
            {
                "timestamp": "2026-01-01",
                "success": False,
                "config": {},
                "metrics": {},
            }
        ]

        output = tmp_path / "fail.html"
        exporter = HTMLExporter()
        exporter.export(output_path=output, records=records)

        html = output.read_text()
        assert "Failed" in html
        assert "failure" in html

    def test_export_empty_raises(self, tmp_path):
        exporter = HTMLExporter()
        with pytest.raises(ValueError, match="No records"):
            exporter.export(output_path=tmp_path / "empty.html", records=[])
