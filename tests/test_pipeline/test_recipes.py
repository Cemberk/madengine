"""Tests for recipe registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from madengine.pipeline.recipes import RecipeRegistry


class TestRecipeRegistry:
    def test_list_all_empty(self, tmp_path):
        empty = tmp_path / "empty_recipes"
        empty.mkdir()
        registry = RecipeRegistry(recipe_paths=[empty])
        assert registry.list_all() == []

    def test_list_all_finds_recipes(self, recipe_dir):
        registry = RecipeRegistry(recipe_paths=[recipe_dir])
        recipes = registry.list_all()
        assert len(recipes) == 1
        assert recipes[0].name == "quick_test"
        assert recipes[0].description == "Quick test recipe"

    def test_find_existing_recipe(self, recipe_dir):
        registry = RecipeRegistry(recipe_paths=[recipe_dir])
        path = registry.find("quick_test")
        assert path is not None
        assert path.name == "quick_test.yaml"

    def test_find_missing_recipe(self, recipe_dir):
        registry = RecipeRegistry(recipe_paths=[recipe_dir])
        path = registry.find("nonexistent")
        assert path is None

    def test_load_existing(self, recipe_dir):
        registry = RecipeRegistry(recipe_paths=[recipe_dir])
        cfg = registry.load("quick_test")
        assert cfg.name == "quick_test"

    def test_load_missing_raises(self, recipe_dir):
        registry = RecipeRegistry(recipe_paths=[recipe_dir])
        with pytest.raises(FileNotFoundError, match="not found"):
            registry.load("nonexistent")

    def test_exists(self, recipe_dir):
        registry = RecipeRegistry(recipe_paths=[recipe_dir])
        assert registry.exists("quick_test") is True
        assert registry.exists("nonexistent") is False
