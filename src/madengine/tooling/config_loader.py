"""Discovers and loads OmegaConf configs from workload folders.

Walks the scripts/ directory tree looking for madengine.yaml files. Each one
is loaded, validated against the WorkloadConfig schema, and merged with the
base config -- with arbitrary extra keys preserved.

This lets workload owners drop a madengine.yaml next to their run.sh to opt in
to AIWorkloads, AIImageBuilder, or any custom tooling extension.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from madengine.tooling.workload_config import WorkloadConfig

WORKLOAD_CONFIG_FILENAME = "madengine.yaml"


@dataclass
class DiscoveredConfig:
    """A workload config discovered in the filesystem."""
    name: str
    path: Path
    config: DictConfig

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": str(self.path),
            "config": OmegaConf.to_container(self.config, resolve=True),
        }


class WorkloadConfigLoader:
    """Discovers and loads per-workload OmegaConf configs.

    Usage:
        loader = WorkloadConfigLoader(scripts_dir="./scripts")
        configs = loader.discover()
        cfg = loader.load("inferencemax")
    """

    def __init__(
        self,
        scripts_dir: str | Path = "./scripts",
        config_filename: str = WORKLOAD_CONFIG_FILENAME,
        base_config: DictConfig | None = None,
    ):
        self.scripts_dir = Path(scripts_dir)
        self.config_filename = config_filename
        self.base_config = base_config

    def discover(self) -> list[DiscoveredConfig]:
        """Walk scripts_dir and find all workload folders with a madengine.yaml."""
        results = []
        if not self.scripts_dir.exists():
            return results

        for root, dirs, files in os.walk(self.scripts_dir):
            if self.config_filename in files:
                config_path = Path(root) / self.config_filename
                workload_name = Path(root).name
                try:
                    cfg = WorkloadConfig.from_yaml(str(config_path))
                    results.append(DiscoveredConfig(
                        name=workload_name,
                        path=config_path,
                        config=cfg,
                    ))
                except Exception as e:
                    print(f"Warning: failed to load {config_path}: {e}")

        return results

    def load(self, workload_name: str) -> DictConfig | None:
        """Load config for a specific workload by folder name."""
        for discovered in self.discover():
            if discovered.name == workload_name:
                if self.base_config:
                    return OmegaConf.merge(self.base_config, discovered.config)
                return discovered.config
        return None

    def load_from_path(self, config_path: str | Path) -> DictConfig:
        """Load a specific workload config file."""
        return WorkloadConfig.from_yaml(str(config_path))

    def get_extensions(self, config: DictConfig) -> dict[str, Any]:
        """Extract arbitrary extension keys (non-schema keys) from a workload config.

        Returns only the keys that aren't part of the standard WorkloadConfig schema,
        enabling pass-through of custom tooling sections.
        """
        known_keys = {"image_build", "workload_run"}
        container = OmegaConf.to_container(config, resolve=True)
        return {k: v for k, v in container.items() if k not in known_keys}


def discover_workload_configs(
    scripts_dir: str | Path = "./scripts",
) -> list[DiscoveredConfig]:
    """Convenience function to discover all workload configs."""
    loader = WorkloadConfigLoader(scripts_dir=scripts_dir)
    return loader.discover()


def load_workload_config(
    workload_name: str,
    scripts_dir: str | Path = "./scripts",
    base_config: DictConfig | None = None,
) -> DictConfig | None:
    """Convenience function to load a specific workload config."""
    loader = WorkloadConfigLoader(
        scripts_dir=scripts_dir,
        base_config=base_config,
    )
    return loader.load(workload_name)
