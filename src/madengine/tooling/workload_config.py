"""Structured config for per-workload settings with arbitrary extension.

A workload folder's madengine.yaml can contain any of these sections.
Unknown keys are preserved through OmegaConf's open-dict feature, so teams
can add their own tooling sections (e.g., profiling, nccl, flash_attn)
without changing madengine core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig, OmegaConf, MISSING


# --- Structured config schemas (validated sections) ---

@dataclass
class ImageBuildConfig:
    """Config for AIImageBuilder integration (replaces Dockerfile build)."""
    enabled: bool = False
    tag_prefix: str = ""
    runtime: str = "rocm"
    max_jobs: int = 16
    components: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkloadRunConfig:
    """Config for AIWorkloads integration (replaces run.sh)."""
    enabled: bool = False
    scheduler: str = "local"
    model: str = ""
    distributed: dict[str, Any] = field(default_factory=dict)
    profiling: dict[str, Any] = field(default_factory=dict)
    container: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkloadConfig:
    """Top-level per-workload config.

    Known sections are typed; anything else is preserved as arbitrary dicts
    through OmegaConf's struct=False mode.

    Example madengine.yaml in a workload folder:

        image_build:
          enabled: true
          tag_prefix: "llama3-training"
          runtime: rocm
          components:
            pytorch:
              enabled: true

        workload_run:
          enabled: true
          scheduler: slurm
          model: megatron-lm/llama3/70B
          distributed:
            num_nodes: 8
            gpus_per_node: 8

        # Arbitrary extension -- madengine passes through untouched
        custom_profiling:
          rocprof: true
          rpd_trace: true
          output_dir: /tmp/traces

        nccl_tuning:
          NCCL_ALGO: Ring
          NCCL_PROTO: Simple
    """
    image_build: ImageBuildConfig = field(default_factory=ImageBuildConfig)
    workload_run: WorkloadRunConfig = field(default_factory=WorkloadRunConfig)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> DictConfig:
        """Create a WorkloadConfig DictConfig from a plain dict.

        Validates known sections against the structured schema while
        preserving arbitrary extension keys.
        """
        schema = OmegaConf.structured(WorkloadConfig)
        OmegaConf.set_struct(schema, False)
        user = OmegaConf.create(d)
        merged = OmegaConf.merge(schema, user)
        OmegaConf.set_struct(merged, False)
        return merged

    @staticmethod
    def from_yaml(path: str) -> DictConfig:
        """Load a WorkloadConfig from a YAML file."""
        user = OmegaConf.load(path)
        schema = OmegaConf.structured(WorkloadConfig)
        OmegaConf.set_struct(schema, False)
        merged = OmegaConf.merge(schema, user)
        OmegaConf.set_struct(merged, False)
        return merged
