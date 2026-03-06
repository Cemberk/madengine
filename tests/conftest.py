"""Shared test fixtures for madengine."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf


@pytest.fixture
def minimal_config():
    """Minimal valid config for testing."""
    return OmegaConf.create(
        {
            "dry_run": True,
            "skip_image_build": True,
            "training": {
                "container": {"image": "test:latest"},
                "model": "megatron-lm/llama3/8B",
                "scheduler": "slurm",
            },
            "credentials": {},
            "data": {"provider": "local", "local": {"base_path": "/data"}},
            "metrics": {},
            "database": {"backend": "none"},
            "tools": {
                "aiimagebuilder": {"executable": "aiimagebuilder"},
                "aiworkloads": {"executable": "aiworkloads"},
            },
            "timeouts": {"image_build": 7200, "job_execution": 86400},
        }
    )


@pytest.fixture
def tmp_yaml(tmp_path):
    """Factory to create temporary YAML files."""

    def _create(content: dict, name: str = "config.yaml") -> Path:
        p = tmp_path / name
        p.write_text(OmegaConf.to_yaml(OmegaConf.create(content)))
        return p

    return _create


@pytest.fixture
def recipe_dir(tmp_path):
    """Create a temporary recipe directory with sample recipes."""
    recipes = tmp_path / "recipes"
    recipes.mkdir()

    quick = recipes / "quick_test.yaml"
    quick.write_text(
        OmegaConf.to_yaml(
            OmegaConf.create(
                {
                    "name": "quick_test",
                    "description": "Quick test recipe",
                    "skip_image_build": True,
                    "training": {
                        "container": {"image": "test:latest"},
                        "model": "test-model",
                    },
                }
            )
        )
    )

    return recipes
