"""Adapter for AIWorkloads black-box integration."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .base import Adapter, AdapterResult


class AIWorkloadsAdapter(Adapter):
    """Invokes the aiworkloads CLI as a subprocess with JSON I/O."""

    def __init__(self, executable: str = "aiworkloads", timeout: int = 300):
        self.executable = executable
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "AIWorkloads"

    def execute(self, config: dict[str, Any]) -> AdapterResult:
        errors = self.validate_config(config)
        if errors:
            return AdapterResult(
                success=False,
                error=f"Config validation failed: {'; '.join(errors)}",
            )

        config_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                aiw_config = self._transform_config(config)
                f.write(OmegaConf.to_yaml(OmegaConf.create(aiw_config)))
                config_path = f.name

            cmd = [
                self.executable,
                f"user_conf={config_path}",
                "output_format=json",
            ]

            if config.get("container", {}).get("image"):
                cmd.append(f"container.image={config['container']['image']}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.returncode != 0:
                try:
                    error_data = json.loads(result.stdout)
                    error_msg = error_data.get("error", {}).get("message", result.stderr)
                except (json.JSONDecodeError, AttributeError):
                    error_msg = result.stderr or result.stdout

                return AdapterResult(
                    success=False,
                    error=error_msg,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    returncode=result.returncode,
                )

            output = json.loads(result.stdout)

            return AdapterResult(
                success=output.get("success", True),
                data=output,
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
            )

        except subprocess.TimeoutExpired:
            return AdapterResult(
                success=False,
                error=f"AIWorkloads timed out after {self.timeout}s",
            )
        except json.JSONDecodeError as e:
            return AdapterResult(
                success=False,
                error=f"Failed to parse AIWorkloads output: {e}",
            )
        finally:
            if config_path:
                Path(config_path).unlink(missing_ok=True)

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        errors = []
        if not config.get("container", {}).get("image"):
            errors.append("container.image is required")
        return errors

    def _transform_config(self, config: dict[str, Any]) -> dict:
        """Transform madengine config to AIWorkloads format."""
        return {
            "container": config.get("container", {}),
            "scheduler": config.get("scheduler", "slurm"),
            "model": config.get("model"),
            "distributed": config.get("distributed", {}),
            "profiling": config.get("profiling", {}),
            "paths": config.get("paths", {}),
        }
