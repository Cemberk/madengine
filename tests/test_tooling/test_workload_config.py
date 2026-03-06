"""Tests for OmegaConf workload config with arbitrary dict extension."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from madengine.tooling.workload_config import WorkloadConfig


class TestWorkloadConfig:
    def test_from_dict_defaults(self):
        cfg = WorkloadConfig.from_dict({})
        assert cfg.image_build.enabled is False
        assert cfg.workload_run.enabled is False

    def test_from_dict_with_image_build(self):
        cfg = WorkloadConfig.from_dict({
            "image_build": {
                "enabled": True,
                "tag_prefix": "my-image",
                "runtime": "rocm",
                "components": {"pytorch": {"enabled": True}},
            }
        })
        assert cfg.image_build.enabled is True
        assert cfg.image_build.tag_prefix == "my-image"
        assert cfg.image_build.components["pytorch"]["enabled"] is True

    def test_from_dict_with_workload_run(self):
        cfg = WorkloadConfig.from_dict({
            "workload_run": {
                "enabled": True,
                "scheduler": "slurm",
                "model": "llama3/70B",
                "distributed": {"num_nodes": 8, "gpus_per_node": 8},
            }
        })
        assert cfg.workload_run.enabled is True
        assert cfg.workload_run.scheduler == "slurm"
        assert cfg.workload_run.model == "llama3/70B"

    def test_arbitrary_extensions_preserved(self):
        """Unknown keys are preserved (struct=False)."""
        cfg = WorkloadConfig.from_dict({
            "custom_profiling": {"rocprof": True, "rpd_trace": True},
            "nccl_tuning": {"NCCL_ALGO": "Ring"},
        })
        container = OmegaConf.to_container(cfg, resolve=True)
        assert container["custom_profiling"]["rocprof"] is True
        assert container["nccl_tuning"]["NCCL_ALGO"] == "Ring"

    def test_from_yaml(self, tmp_path):
        yaml_content = """
image_build:
  enabled: true
  tag_prefix: test
workload_run:
  enabled: true
  scheduler: local
my_custom_section:
  key1: value1
  nested:
    key2: 42
"""
        config_path = tmp_path / "madengine.yaml"
        config_path.write_text(yaml_content)

        cfg = WorkloadConfig.from_yaml(str(config_path))
        assert cfg.image_build.enabled is True
        assert cfg.workload_run.scheduler == "local"

        container = OmegaConf.to_container(cfg, resolve=True)
        assert container["my_custom_section"]["key1"] == "value1"
        assert container["my_custom_section"]["nested"]["key2"] == 42

    def test_mixed_known_and_unknown(self):
        cfg = WorkloadConfig.from_dict({
            "image_build": {"enabled": True, "tag_prefix": "test"},
            "workload_run": {"enabled": False},
            "pytest_fkit": {"timeout": 600, "workers": "auto"},
            "benchmark": {"framework": "sglang", "precision": "fp8"},
        })
        assert cfg.image_build.enabled is True
        container = OmegaConf.to_container(cfg, resolve=True)
        assert container["pytest_fkit"]["timeout"] == 600
        assert container["benchmark"]["framework"] == "sglang"
