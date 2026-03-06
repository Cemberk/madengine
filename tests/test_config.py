"""Tests for configuration loading and validation."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from madengine.config import load_config, merge_overrides, validate_config


class TestLoadConfig:
    def test_load_defaults(self):
        cfg = load_config()
        assert cfg.dry_run is False
        assert cfg.data.provider == "local"

    def test_load_user_config(self, tmp_yaml):
        path = tmp_yaml({"dry_run": True, "data": {"provider": "s3"}})
        cfg = load_config(path)
        assert cfg.dry_run is True
        assert cfg.data.provider == "s3"

    def test_user_config_merges_defaults(self, tmp_yaml):
        path = tmp_yaml({"dry_run": True})
        cfg = load_config(path)
        assert cfg.dry_run is True
        assert cfg.data.provider == "local"


class TestMergeOverrides:
    def test_single_override(self):
        cfg = OmegaConf.create({"dry_run": False})
        result = merge_overrides(cfg, ("dry_run=true",))
        assert result.dry_run is True

    def test_dotted_override(self):
        cfg = OmegaConf.create({"data": {"provider": "local"}})
        result = merge_overrides(cfg, ("data.provider=s3",))
        assert result.data.provider == "s3"

    def test_empty_overrides(self):
        cfg = OmegaConf.create({"dry_run": False})
        result = merge_overrides(cfg, ())
        assert result.dry_run is False


class TestValidateConfig:
    def test_valid_with_skip_image_build(self):
        cfg = OmegaConf.create(
            {
                "skip_image_build": True,
                "training": {"model": "test"},
                "data": {"provider": "local"},
            }
        )
        errors = validate_config(cfg)
        assert errors == []

    def test_missing_image_section(self):
        cfg = OmegaConf.create(
            {"training": {"model": "test"}, "data": {"provider": "local"}}
        )
        errors = validate_config(cfg)
        assert any("image" in e for e in errors)

    def test_missing_training_section(self):
        cfg = OmegaConf.create({"skip_image_build": True, "data": {"provider": "local"}})
        errors = validate_config(cfg)
        assert any("training" in e for e in errors)

    def test_nas_missing_host(self):
        cfg = OmegaConf.create(
            {
                "skip_image_build": True,
                "training": {"model": "test"},
                "data": {"provider": "nas", "nas": {}},
            }
        )
        errors = validate_config(cfg)
        assert any("nas.host" in e for e in errors)

    def test_s3_missing_bucket(self):
        cfg = OmegaConf.create(
            {
                "skip_image_build": True,
                "training": {"model": "test"},
                "data": {"provider": "s3", "s3": {}},
            }
        )
        errors = validate_config(cfg)
        assert any("s3.bucket" in e for e in errors)
