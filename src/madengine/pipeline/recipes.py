"""Recipe loading, discovery, and expansion."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


@dataclass
class RecipeInfo:
    """Recipe metadata."""

    name: str
    path: Path
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": str(self.path),
            "description": self.description,
        }


class RecipeRegistry:
    """Registry for pre-defined job recipes.

    Recipes are YAML files that expand to full madengine configs.
    """

    def __init__(self, recipe_paths: list[Path] | None = None):
        self.recipe_paths = recipe_paths if recipe_paths is not None else self._default_paths()

    def _default_paths(self) -> list[Path]:
        paths = []

        builtin = Path(__file__).parent.parent.parent.parent / "configs" / "recipes"
        if builtin.exists():
            paths.append(builtin)

        user_dir = Path(os.path.expanduser("~/.config/madengine/recipes"))
        if user_dir.exists():
            paths.append(user_dir)

        env_path = os.environ.get("MADENGINE_RECIPES")
        if env_path:
            p = Path(env_path)
            if p.exists():
                paths.append(p)

        return paths

    def list_all(self) -> list[RecipeInfo]:
        recipes = []
        for path in self.recipe_paths:
            for recipe_file in path.glob("**/*.yaml"):
                cfg = OmegaConf.load(recipe_file)
                recipes.append(
                    RecipeInfo(
                        name=recipe_file.stem,
                        path=recipe_file,
                        description=cfg.get("description", ""),
                    )
                )
        return recipes

    def find(self, name: str) -> Path | None:
        for path in self.recipe_paths:
            recipe_file = path / f"{name}.yaml"
            if recipe_file.exists():
                return recipe_file

            for recipe_file in path.glob(f"**/{name}.yaml"):
                return recipe_file

        return None

    def load(self, name: str) -> DictConfig:
        """Load and expand a recipe into full config.

        Raises:
            FileNotFoundError: If recipe not found.
        """
        recipe_path = self.find(name)
        if not recipe_path:
            available = [r.name for r in self.list_all()]
            raise FileNotFoundError(
                f"Recipe '{name}' not found. Available: {', '.join(available) or '(none)'}"
            )

        from madengine.config import load_config

        return load_config(recipe_path)

    def exists(self, name: str) -> bool:
        return self.find(name) is not None
