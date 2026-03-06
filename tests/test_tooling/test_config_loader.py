"""Tests for workload config discovery from filesystem."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from madengine.tooling.config_loader import (
    WorkloadConfigLoader,
    discover_workload_configs,
    load_workload_config,
    WORKLOAD_CONFIG_FILENAME,
)


@pytest.fixture
def scripts_dir(tmp_path):
    """Create a scripts directory with workload configs."""
    scripts = tmp_path / "scripts"

    # Workload with config
    w1 = scripts / "inferencemax"
    w1.mkdir(parents=True)
    (w1 / "run.sh").write_text("#!/bin/bash\necho hello")
    (w1 / WORKLOAD_CONFIG_FILENAME).write_text(OmegaConf.to_yaml(OmegaConf.create({
        "image_build": {"enabled": True, "tag_prefix": "inferencemax"},
        "workload_run": {"enabled": True, "model": "deepseek-r1"},
        "benchmark": {"framework": "sglang"},
    })))

    # Workload without config (legacy)
    w2 = scripts / "legacy_model"
    w2.mkdir(parents=True)
    (w2 / "run.sh").write_text("#!/bin/bash\necho legacy")

    # Workload with config + nested extensions
    w3 = scripts / "transformers_ut"
    w3.mkdir(parents=True)
    (w3 / "run.sh").write_text("#!/bin/bash\necho tests")
    (w3 / WORKLOAD_CONFIG_FILENAME).write_text(OmegaConf.to_yaml(OmegaConf.create({
        "workload_run": {"enabled": True, "model": "transformers_ut"},
        "pytest_fkit": {"timeout": 600, "workers": "auto"},
    })))

    return scripts


class TestWorkloadConfigLoader:
    def test_discover_finds_configs(self, scripts_dir):
        loader = WorkloadConfigLoader(scripts_dir=scripts_dir)
        configs = loader.discover()
        names = {c.name for c in configs}
        assert "inferencemax" in names
        assert "transformers_ut" in names
        assert "legacy_model" not in names

    def test_discover_empty_dir(self, tmp_path):
        loader = WorkloadConfigLoader(scripts_dir=tmp_path / "nonexistent")
        assert loader.discover() == []

    def test_load_specific_workload(self, scripts_dir):
        loader = WorkloadConfigLoader(scripts_dir=scripts_dir)
        cfg = loader.load("inferencemax")
        assert cfg is not None
        assert cfg.image_build.enabled is True
        assert cfg.image_build.tag_prefix == "inferencemax"

    def test_load_nonexistent_returns_none(self, scripts_dir):
        loader = WorkloadConfigLoader(scripts_dir=scripts_dir)
        assert loader.load("nonexistent") is None

    def test_get_extensions(self, scripts_dir):
        loader = WorkloadConfigLoader(scripts_dir=scripts_dir)
        cfg = loader.load("inferencemax")
        extensions = loader.get_extensions(cfg)
        assert "benchmark" in extensions
        assert extensions["benchmark"]["framework"] == "sglang"
        assert "image_build" not in extensions
        assert "workload_run" not in extensions

    def test_load_from_path(self, scripts_dir):
        loader = WorkloadConfigLoader(scripts_dir=scripts_dir)
        path = scripts_dir / "inferencemax" / WORKLOAD_CONFIG_FILENAME
        cfg = loader.load_from_path(path)
        assert cfg.workload_run.model == "deepseek-r1"

    def test_convenience_discover(self, scripts_dir):
        configs = discover_workload_configs(scripts_dir=scripts_dir)
        assert len(configs) == 2

    def test_convenience_load(self, scripts_dir):
        cfg = load_workload_config("transformers_ut", scripts_dir=scripts_dir)
        assert cfg is not None
        container = OmegaConf.to_container(cfg, resolve=True)
        assert container["pytest_fkit"]["workers"] == "auto"

    def test_base_config_merge(self, scripts_dir):
        base = OmegaConf.create({"database": {"backend": "mongodb"}})
        loader = WorkloadConfigLoader(scripts_dir=scripts_dir, base_config=base)
        cfg = loader.load("inferencemax")
        container = OmegaConf.to_container(cfg, resolve=True)
        assert container.get("database", {}).get("backend") == "mongodb"
