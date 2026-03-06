"""Tests for CSV exporter."""

from __future__ import annotations

import csv

import pytest

from madengine.reporting.csv import CSVExporter


class TestCSVExporter:
    def test_export_basic(self, tmp_path):
        records = [
            {"name": "run1", "throughput": 100.0},
            {"name": "run2", "throughput": 200.0},
        ]

        output = tmp_path / "output.csv"
        exporter = CSVExporter()
        exporter.export(output_path=output, records=records)

        with open(output) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["name"] == "run1"
        assert rows[1]["throughput"] == "200.0"

    def test_export_nested(self, tmp_path):
        records = [{"config": {"model": "llama3"}, "metrics": {"throughput": 1234}}]

        output = tmp_path / "nested.csv"
        exporter = CSVExporter()
        exporter.export(output_path=output, records=records)

        with open(output) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert "config.model" in rows[0]
        assert rows[0]["config.model"] == "llama3"

    def test_export_empty_raises(self, tmp_path):
        exporter = CSVExporter()
        with pytest.raises(ValueError, match="No records"):
            exporter.export(output_path=tmp_path / "empty.csv", records=[])
