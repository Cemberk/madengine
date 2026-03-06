"""OmegaConf-based tooling config discovery for workload folders.

Each workload folder (under scripts/) can carry a madengine.yaml that extends
the base config with arbitrary dicts. This enables per-workload specialization
of image building, script generation, data staging, and any future tooling.
"""

from madengine.tooling.config_loader import (
    WorkloadConfigLoader,
    discover_workload_configs,
    load_workload_config,
)
from madengine.tooling.workload_config import WorkloadConfig

__all__ = [
    "WorkloadConfigLoader",
    "WorkloadConfig",
    "discover_workload_configs",
    "load_workload_config",
]
