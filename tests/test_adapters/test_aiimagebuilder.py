"""Tests for AIImageBuilder adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from madengine.adapters.aiimagebuilder import AIImageBuilderAdapter


class TestAIImageBuilderAdapter:
    def test_validate_config_missing_tag_prefix(self):
        adapter = AIImageBuilderAdapter()
        errors = adapter.validate_config({})
        assert "tag_prefix is required" in errors

    def test_validate_config_valid(self):
        adapter = AIImageBuilderAdapter()
        errors = adapter.validate_config({"tag_prefix": "my-image"})
        assert errors == []

    def test_name(self):
        adapter = AIImageBuilderAdapter()
        assert adapter.name == "AIImageBuilder"

    @patch("subprocess.run")
    def test_execute_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"success": true, "container": {"image": "my-image:abc123"}}',
            stderr="",
        )

        adapter = AIImageBuilderAdapter()
        result = adapter.execute({"tag_prefix": "my-image", "runtime": "rocm"})

        assert result.success
        assert result.data["container"]["image"] == "my-image:abc123"

    @patch("subprocess.run")
    def test_execute_failure(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="Build failed",
            stderr="Error: Dockerfile not found",
        )

        adapter = AIImageBuilderAdapter()
        result = adapter.execute({"tag_prefix": "my-image"})

        assert not result.success

    def test_transform_config(self):
        adapter = AIImageBuilderAdapter()
        transformed = adapter._transform_config(
            {
                "tag_prefix": "test",
                "runtime": "rocm",
                "components": {"pytorch": {"enabled": True}},
            }
        )

        assert transformed["build"]["tag_prefix"] == "test"
        assert transformed["runtime"]["type"] == "rocm"
        assert transformed["component"]["pytorch"]["enabled"] is True
