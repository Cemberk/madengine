"""Tests for the adapter bridge between legacy and new-path flows."""

from __future__ import annotations

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from madengine.adapters.base import AdapterResult
from madengine.tooling.bridge import AdapterBridge, ImageBuildResult, ScriptRunResult
from madengine.tooling.workload_config import WorkloadConfig


def _make_args(**kwargs):
    defaults = {
        "use_aiworkloads": False,
        "use_aiimagebuilder": False,
        "workload_config": None,
        "dry_run": False,
    }
    defaults.update(kwargs)
    return Namespace(**defaults)


class TestAdapterBridge:
    def test_not_new_path_by_default(self):
        bridge = AdapterBridge(_make_args())
        assert bridge.is_new_path is False

    def test_is_new_path_with_aiworkloads(self):
        bridge = AdapterBridge(_make_args(use_aiworkloads=True))
        assert bridge.is_new_path is True

    def test_is_new_path_with_aiimagebuilder(self):
        bridge = AdapterBridge(_make_args(use_aiimagebuilder=True))
        assert bridge.is_new_path is True

    def test_load_workload_config_from_file(self, tmp_path):
        cfg_path = tmp_path / "madengine.yaml"
        cfg_path.write_text(OmegaConf.to_yaml(OmegaConf.create({
            "image_build": {"enabled": True, "tag_prefix": "test"},
        })))

        bridge = AdapterBridge(_make_args(
            use_aiimagebuilder=True,
            workload_config=str(cfg_path),
        ))

        model_info = {"name": "test", "scripts": "scripts/test/run.sh"}
        cfg = bridge.load_workload_config(model_info)
        assert cfg is not None
        assert cfg.image_build.enabled is True

    def test_load_workload_config_auto_discovery(self, tmp_path):
        scripts_dir = tmp_path / "scripts" / "myworkload"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "run.sh").write_text("#!/bin/bash\necho ok")
        (scripts_dir / "madengine.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create({
            "workload_run": {"enabled": True, "model": "auto-discovered"},
        })))

        bridge = AdapterBridge(_make_args(use_aiworkloads=True))
        model_info = {"name": "myworkload", "scripts": str(scripts_dir / "run.sh")}
        cfg = bridge.load_workload_config(model_info)
        assert cfg is not None
        assert cfg.workload_run.model == "auto-discovered"

    @patch("madengine.adapters.aiimagebuilder.AIImageBuilderAdapter.execute")
    def test_build_image_success(self, mock_execute):
        mock_execute.return_value = AdapterResult(
            success=True,
            data={"container": {"image": "ci-test:abc123"}},
        )

        bridge = AdapterBridge(_make_args(use_aiimagebuilder=True))
        result = bridge.build_image(
            model_info={"name": "test", "dockerfile": "docker/test"},
            context={"docker_env_vars": {"MAD_GPU_VENDOR": "AMD"}},
        )

        assert result.success
        assert result.image_tag == "ci-test:abc123"

    @patch("madengine.adapters.aiimagebuilder.AIImageBuilderAdapter.execute")
    def test_build_image_failure(self, mock_execute):
        mock_execute.return_value = AdapterResult(
            success=False,
            error="Dockerfile not found",
        )

        bridge = AdapterBridge(_make_args(use_aiimagebuilder=True))
        result = bridge.build_image(
            model_info={"name": "test"},
            context={"docker_env_vars": {"MAD_GPU_VENDOR": "AMD"}},
        )

        assert not result.success
        assert "Dockerfile not found" in result.error

    def test_build_image_dry_run(self):
        bridge = AdapterBridge(_make_args(use_aiimagebuilder=True, dry_run=True))
        result = bridge.build_image(
            model_info={"name": "test"},
            context={"docker_env_vars": {"MAD_GPU_VENDOR": "AMD"}},
        )

        assert result.success
        assert "dry-run" in result.image_tag

    @patch("madengine.adapters.aiworkloads.AIWorkloadsAdapter.execute")
    def test_run_workload_success(self, mock_execute):
        mock_execute.return_value = AdapterResult(
            success=True,
            data={"submit_command": "bash generated_run.sh"},
        )

        bridge = AdapterBridge(_make_args(use_aiworkloads=True))
        result = bridge.run_workload(
            model_info={"name": "test", "n_gpus": "8"},
            image_tag="ci-test:latest",
            context={"docker_env_vars": {"MAD_GPU_VENDOR": "AMD"}},
        )

        assert result.success
        assert result.submit_command == "bash generated_run.sh"

    def test_run_workload_dry_run(self):
        bridge = AdapterBridge(_make_args(use_aiworkloads=True, dry_run=True))
        result = bridge.run_workload(
            model_info={"name": "test"},
            image_tag="ci-test:latest",
            context={"docker_env_vars": {"MAD_GPU_VENDOR": "AMD"}},
        )

        assert result.success
        assert "dry-run" in result.submit_command

    def test_get_extensions(self):
        bridge = AdapterBridge(_make_args(use_aiworkloads=True))
        cfg = WorkloadConfig.from_dict({
            "image_build": {"enabled": True},
            "custom_nccl": {"NCCL_ALGO": "Ring"},
            "profiling": {"rocprof": True},
        })
        extensions = bridge.get_extensions(cfg)
        assert "custom_nccl" in extensions
        assert "profiling" in extensions
        assert "image_build" not in extensions

    @patch("madengine.adapters.aiworkloads.AIWorkloadsAdapter.execute")
    def test_workload_config_extensions_passed(self, mock_execute):
        mock_execute.return_value = AdapterResult(
            success=True, data={"submit_command": "sbatch job.sh"}
        )

        bridge = AdapterBridge(_make_args(use_aiworkloads=True))
        workload_cfg = WorkloadConfig.from_dict({
            "workload_run": {"enabled": True, "scheduler": "slurm"},
            "benchmark": {"framework": "sglang", "precision": "fp8"},
        })

        bridge.run_workload(
            model_info={"name": "test"},
            image_tag="ci-test:latest",
            context={"docker_env_vars": {"MAD_GPU_VENDOR": "AMD"}},
            workload_cfg=workload_cfg,
        )

        call_config = mock_execute.call_args[0][0]
        assert call_config["scheduler"] == "slurm"
        assert call_config["extensions"]["benchmark"]["framework"] == "sglang"

    def test_build_not_enabled_returns_error(self):
        bridge = AdapterBridge(_make_args())
        result = bridge.build_image(
            model_info={"name": "test"},
            context={},
        )
        assert not result.success
        assert "not enabled" in result.error

    def test_run_not_enabled_returns_error(self):
        bridge = AdapterBridge(_make_args())
        result = bridge.run_workload(
            model_info={"name": "test"},
            image_tag="test:latest",
            context={},
        )
        assert not result.success
        assert "not enabled" in result.error
