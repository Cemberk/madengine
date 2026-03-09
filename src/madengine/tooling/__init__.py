"""OmegaConf-based tooling config discovery for workload folders.

Each workload folder (under scripts/) can carry a madengine.yaml that extends
the base config with arbitrary dicts. This enables per-workload specialization
of image building, script generation, data staging, and any future tooling.

The config_converter module provides 1:1 conversion between the YAML configs
(configs/workloads/*.yaml) and the legacy models.json format, enabling:
  madengine run --workload-config configs/workloads/transformers_ut.yaml
"""

from madengine.tooling.config_loader import (
    WorkloadConfigLoader,
    discover_workload_configs,
    load_workload_config,
)
from madengine.tooling.config_converter import (
    config_to_model_info,
    config_to_build_args,
    config_to_dockerfile,
    config_to_context_dict,
    load_all_configs,
    load_single_config,
    load_yaml_config,
)
from madengine.tooling.workload_config import WorkloadConfig

__all__ = [
    "WorkloadConfigLoader",
    "WorkloadConfig",
    "discover_workload_configs",
    "load_workload_config",
    "config_to_model_info",
    "config_to_build_args",
    "config_to_dockerfile",
    "config_to_context_dict",
    "load_all_configs",
    "load_single_config",
    "load_yaml_config",
]
