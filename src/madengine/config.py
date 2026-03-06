"""Unified configuration system using OmegaConf."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def get_default_config_path() -> Path:
    return Path(__file__).parent.parent.parent / "configs" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> DictConfig:
    """Load configuration with defaults.

    Priority (highest to lowest):
    1. User config file
    2. Default config.yaml
    """
    default_path = get_default_config_path()
    if default_path.exists():
        default_cfg = OmegaConf.load(default_path)
    else:
        default_cfg = OmegaConf.create({})

    if config_path:
        user_cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(default_cfg, user_cfg)
    else:
        cfg = default_cfg

    return _expand_paths(cfg)


def merge_overrides(cfg: DictConfig, overrides: tuple[str, ...]) -> DictConfig:
    """Merge CLI overrides (key=value dotlist) into config."""
    if not overrides:
        return cfg
    override_cfg = OmegaConf.from_dotlist(list(overrides))
    return OmegaConf.merge(cfg, override_cfg)


def _expand_paths(cfg: DictConfig) -> DictConfig:
    """Expand ~ and environment variables in path fields."""
    container = OmegaConf.to_container(cfg, resolve=True)

    def expand(obj: Any) -> Any:
        if isinstance(obj, str):
            return os.path.expanduser(os.path.expandvars(obj))
        elif isinstance(obj, dict):
            return {k: expand(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [expand(v) for v in obj]
        return obj

    return OmegaConf.create(expand(container))


def validate_config(cfg: DictConfig) -> list[str]:
    """Validate configuration. Returns list of error messages (empty if valid)."""
    errors = []

    if not cfg.get("skip_image_build", False):
        if not cfg.get("image"):
            errors.append("'image' section required when building images")

    if not cfg.get("training"):
        errors.append("'training' section is required")

    provider = cfg.get("data", {}).get("provider", "local")
    provider_cfg = cfg.get("data", {}).get(provider, {})

    if provider == "nas":
        if not provider_cfg.get("host"):
            errors.append("data.nas.host is required for NAS provider")
    elif provider == "s3":
        if not provider_cfg.get("bucket"):
            errors.append("data.s3.bucket is required for S3 provider")
    elif provider == "minio":
        if not provider_cfg.get("endpoint") or not provider_cfg.get("bucket"):
            errors.append("data.minio.endpoint and bucket required for MinIO provider")

    return errors
