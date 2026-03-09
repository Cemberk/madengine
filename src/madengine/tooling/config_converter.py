"""Converts workload YAML configs to models.json-compatible dicts.

This module bridges the gap between the new YAML-first config format
(configs/workloads/*.yaml) and the legacy models.json format that
run_models.py expects. It enables:

  madengine run --workload-config configs/workloads/transformers_ut.yaml

to produce the exact same docker build + docker run as:

  madengine run --tags transformers_ut  (with models.json)

The conversion is 1:1 and reversible -- you can round-trip between
models.json and YAML without data loss.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_yaml_config(config_path: str | Path) -> dict:
    """Load a YAML workload config file and return as plain dict."""
    try:
        import yaml
    except ImportError:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(str(config_path))
        return OmegaConf.to_container(cfg, resolve=True)

    with open(config_path) as f:
        return yaml.safe_load(f)


def config_to_model_info(config: dict, platform: str = "amd") -> dict:
    """Convert a workload YAML config to a models.json-compatible dict.

    The returned dict has the exact same keys and structure as a models.json
    entry, so it can be fed directly to RunModels.run_model().

    Args:
        config: Parsed YAML config dict (from configs/workloads/*.yaml).
        platform: Which platform variant to select when multiple Dockerfiles
                  exist (e.g., "amd" or "nvidia"). Default is "amd".

    Returns:
        dict compatible with models.json entries.
    """
    model_info = {}

    model_info["name"] = config["name"]
    model_info["owner"] = config.get("owner", "")
    model_info["tags"] = config.get("tags", [])
    if "timeout" in config:
        model_info["timeout"] = config["timeout"]
    model_info["n_gpus"] = config.get("n_gpus", "-1")
    model_info["training_precision"] = config.get("training_precision", "amp_fp16")
    if "skip_gpu_arch" in config:
        model_info["skip_gpu_arch"] = config["skip_gpu_arch"]
    model_info["args"] = config.get("run", {}).get("args", "")

    docker = config.get("docker", {})
    model_info["dockerfile"] = docker.get("dockerfile", "")
    model_info["dockercontext"] = docker.get("context", "docker")

    run = config.get("run", {})
    model_info["scripts"] = run.get("scripts", "")
    model_info["multiple_results"] = run.get("multiple_results", "")

    if config.get("url"):
        model_info["url"] = config["url"]
    if config.get("cred"):
        model_info["cred"] = config["cred"]
    if config.get("data"):
        model_info["data"] = config["data"]
    if config.get("additional_docker_run_options"):
        model_info["additional_docker_run_options"] = config["additional_docker_run_options"]

    custom = config.get("custom", {})
    for key, value in custom.items():
        model_info[key] = value

    return model_info


def config_to_build_args(config: dict, platform: str = "amd") -> dict:
    """Extract docker build args from a workload config.

    Selects the appropriate Dockerfile variant based on platform.
    Returns a dict of build_arg_name -> value suitable for
    --build-arg KEY=VALUE in docker build.
    """
    docker = config.get("docker", {})

    if "build_args" in docker:
        return dict(docker["build_args"])

    variants = docker.get("variants", [])
    for variant in variants:
        ctx = variant.get("context", "")
        if platform.lower() in ctx.lower():
            return dict(variant.get("build_args", {}))

    if variants:
        return dict(variants[0].get("build_args", {}))

    return {}


def config_to_dockerfile(config: dict, platform: str = "amd") -> str:
    """Resolve the Dockerfile path for a platform from a workload config.

    For single-variant configs, returns docker.file directly.
    For multi-variant configs, picks the variant matching the platform.
    """
    docker = config.get("docker", {})

    if "file" in docker:
        return docker["file"]

    variants = docker.get("variants", [])
    for variant in variants:
        ctx = variant.get("context", "")
        if platform.lower() in ctx.lower():
            return variant.get("file", "")

    if variants:
        return variants[0].get("file", "")

    return docker.get("dockerfile", "")


def config_to_context_dict(config: dict, platform: str = "amd") -> dict:
    """Build a madengine additional_context dict from custom fields.

    This produces the same JSON structure that the Jenkinsfile helpers
    (generateInferenceMaxContext, generateExecDashboardContext, etc.)
    construct at runtime.
    """
    custom = config.get("custom", {})
    if not custom:
        return {}

    context = {}
    docker_env_vars = {}
    docker_build_arg = {}

    if "model" in custom:
        docker_env_vars["MODEL"] = custom["model"]
    if "model_prefix" in custom:
        docker_env_vars["MODEL_PREFIX"] = custom["model_prefix"]
    if "framework" in custom:
        docker_env_vars["FRAMEWORK"] = custom["framework"]
    if "precision" in custom:
        docker_env_vars["PRECISION"] = custom["precision"]

    if "image" in custom:
        docker_build_arg["BASE_DOCKER"] = custom["image"]

    if "afo_benchmarks" in custom:
        afo = custom["afo_benchmarks"]
        bool_keys = [
            "run_gemm", "run_gemm_models", "run_flash_attention", "run_conv",
            "run_rccl", "run_aiter_rmsnorm", "run_moe", "run_hipblaslt",
        ]
        for key in bool_keys:
            if key in afo:
                docker_env_vars[key.upper()] = str(afo[key]).lower()

        str_keys = [
            "gemm_sections", "gemm_models", "fa_sections", "conv_sections",
            "rccl_sections", "rmsnorm_sections", "moe_sections",
        ]
        for key in str_keys:
            if key in afo:
                docker_env_vars[key.upper()] = afo[key]

    if docker_env_vars:
        context["docker_env_vars"] = docker_env_vars
    if docker_build_arg:
        context["docker_build_arg"] = docker_build_arg

    return context


def load_all_configs(configs_dir: str | Path) -> list[dict]:
    """Load all YAML workload configs from a directory.

    Returns a list of models.json-compatible dicts, suitable for
    replacing the models.json file entirely.
    """
    configs_dir = Path(configs_dir)
    if not configs_dir.exists():
        return []

    results = []
    for yaml_file in sorted(configs_dir.glob("*.yaml")):
        config = load_yaml_config(yaml_file)
        model_info = config_to_model_info(config)
        results.append(model_info)

    return results


def load_single_config(config_path: str | Path, platform: str = "amd") -> dict:
    """Load a single YAML config and return models.json-compatible dict.

    This is the primary entry point for:
        madengine run --workload-config path/to/workload.yaml
    """
    config = load_yaml_config(config_path)
    return config_to_model_info(config, platform=platform)
