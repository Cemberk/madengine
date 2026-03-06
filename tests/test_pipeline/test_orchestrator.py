"""Tests for the pipeline orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from madengine.adapters.base import AdapterResult
from madengine.pipeline.orchestrator import Orchestrator, PipelineResult


class TestPipelineResult:
    def test_to_dict(self):
        result = PipelineResult(success=True, image="test:latest", duration_seconds=10.0)
        d = result.to_dict()
        assert d["success"] is True
        assert d["image"] == "test:latest"
        assert d["duration_seconds"] == 10.0

    def test_to_json(self):
        result = PipelineResult(success=True)
        j = result.to_json()
        assert '"success": true' in j

    def test_to_text(self):
        result = PipelineResult(
            success=True,
            image="test:latest",
            submit_command="sbatch job.sh",
            duration_seconds=42.5,
        )
        text = result.to_text()
        assert "Success: True" in text
        assert "test:latest" in text
        assert "42.5s" in text

    def test_to_text_with_error(self):
        result = PipelineResult(success=False, error="something broke")
        text = result.to_text()
        assert "something broke" in text


class TestOrchestrator:
    @patch("madengine.adapters.aiworkloads.AIWorkloadsAdapter.execute")
    def test_dry_run_skips_execution(self, mock_execute, minimal_config):
        mock_execute.return_value = AdapterResult(
            success=True,
            data={"submit_command": "sbatch job.sh"},
        )

        orchestrator = Orchestrator(minimal_config)
        result = orchestrator.run()

        assert result.success
        assert result.stages.get("execution", {}).get("skipped")

    @patch("madengine.adapters.aiworkloads.AIWorkloadsAdapter.execute")
    def test_dry_run_returns_submit_command(self, mock_execute, minimal_config):
        mock_execute.return_value = AdapterResult(
            success=True,
            data={"submit_command": "sbatch job.sh"},
        )

        orchestrator = Orchestrator(minimal_config)
        result = orchestrator.run()

        assert result.submit_command == "sbatch job.sh"

    @patch("madengine.adapters.aiworkloads.AIWorkloadsAdapter.execute")
    def test_script_generation_failure(self, mock_execute, minimal_config):
        mock_execute.return_value = AdapterResult(
            success=False,
            error="Invalid model config",
        )

        orchestrator = Orchestrator(minimal_config)
        result = orchestrator.run()

        assert not result.success
        assert "Script generation failed" in result.error

    def test_image_build_skipped_when_configured(self, minimal_config):
        with patch(
            "madengine.adapters.aiworkloads.AIWorkloadsAdapter.execute"
        ) as mock_execute:
            mock_execute.return_value = AdapterResult(
                success=True, data={"submit_command": "echo ok"}
            )

            orchestrator = Orchestrator(minimal_config)
            result = orchestrator.run()

            assert result.stages.get("image", {}).get("skipped") is True
