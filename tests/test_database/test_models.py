"""Tests for database models."""

from __future__ import annotations

from madengine.database.models import RunRecord


class TestRunRecord:
    def test_defaults(self):
        r = RunRecord()
        assert r.success is True
        assert r.config == {}
        assert r.metrics == {}

    def test_with_data(self):
        r = RunRecord(
            recipe="test_recipe",
            config={"model": "llama3"},
            metrics={"throughput": 1234.5},
            success=True,
        )
        assert r.recipe == "test_recipe"
        assert r.config["model"] == "llama3"

    def test_extra_fields_allowed(self):
        r = RunRecord(custom_field="hello")
        assert r.custom_field == "hello"
