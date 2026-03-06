"""Tests for AIWorkloads adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from madengine.adapters.aiworkloads import AIWorkloadsAdapter


class TestAIWorkloadsAdapter:
    def test_validate_config_missing_image(self):
        adapter = AIWorkloadsAdapter()
        errors = adapter.validate_config({})
        assert "container.image is required" in errors

    def test_validate_config_valid(self):
        adapter = AIWorkloadsAdapter()
        errors = adapter.validate_config({"container": {"image": "test:latest"}})
        assert errors == []

    def test_name(self):
        adapter = AIWorkloadsAdapter()
        assert adapter.name == "AIWorkloads"

    @patch("subprocess.run")
    def test_execute_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"success": true, "submit_command": "sbatch job.sh"}',
            stderr="",
        )

        adapter = AIWorkloadsAdapter()
        result = adapter.execute({"container": {"image": "test:latest"}})

        assert result.success
        assert result.data["submit_command"] == "sbatch job.sh"

    @patch("subprocess.run")
    def test_execute_failure(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout='{"success": false, "error": {"message": "Config error"}}',
            stderr="",
        )

        adapter = AIWorkloadsAdapter()
        result = adapter.execute({"container": {"image": "test:latest"}})

        assert not result.success
        assert "Config error" in result.error

    @patch("subprocess.run")
    def test_execute_timeout(self, mock_run):
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="aiworkloads", timeout=300)

        adapter = AIWorkloadsAdapter(timeout=300)
        result = adapter.execute({"container": {"image": "test:latest"}})

        assert not result.success
        assert "timed out" in result.error

    def test_execute_validation_failure(self):
        adapter = AIWorkloadsAdapter()
        result = adapter.execute({})

        assert not result.success
        assert "validation failed" in result.error

    @patch("subprocess.run")
    def test_execute_invalid_json(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="not valid json",
            stderr="",
        )

        adapter = AIWorkloadsAdapter()
        result = adapter.execute({"container": {"image": "test:latest"}})

        assert not result.success
        assert "parse" in result.error.lower()
