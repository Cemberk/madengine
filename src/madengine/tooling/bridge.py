"""Bridge between the legacy RunModels flow and the new adapter-based path.

When --use-aiimagebuilder or --use-aiworkloads flags are set, the legacy
run_models.py delegates to this bridge instead of calling docker build / run.sh
directly. The bridge:

1. Discovers per-workload madengine.yaml configs (OmegaConf, arbitrary dicts)
2. Calls AIImageBuilder adapter to build images (replaces docker build)
3. Calls AIWorkloads adapter to generate + execute scripts (replaces run.sh)
4. Returns results in a format run_models.py can consume

This keeps run_models.py's existing flow intact -- the bridge is called at
the exact points where docker build and run.sh would normally execute.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from madengine.adapters import AIImageBuilderAdapter, AIWorkloadsAdapter, AdapterResult
from madengine.tooling.config_loader import WorkloadConfigLoader, WORKLOAD_CONFIG_FILENAME
from madengine.tooling.workload_config import WorkloadConfig


@dataclass
class ImageBuildResult:
    """Result from the AIImageBuilder bridge call."""
    success: bool
    image_tag: str = ""
    build_duration: float = 0.0
    error: str = ""


@dataclass
class ScriptRunResult:
    """Result from the AIWorkloads bridge call."""
    success: bool
    submit_command: str = ""
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    duration: float = 0.0
    error: str = ""


class AdapterBridge:
    """Bridges the legacy madengine flow to the new adapter-based tooling.

    Usage from run_models.py:

        bridge = AdapterBridge(args)

        # Instead of docker build:
        if bridge.use_aiimagebuilder:
            result = bridge.build_image(model_info, context)
            image_tag = result.image_tag

        # Instead of bash run.sh:
        if bridge.use_aiworkloads:
            result = bridge.run_workload(model_info, image_tag, context)
    """

    def __init__(self, args: Any):
        self.args = args
        self.use_aiimagebuilder = getattr(args, "use_aiimagebuilder", False)
        self.use_aiworkloads = getattr(args, "use_aiworkloads", False)
        self.dry_run = getattr(args, "dry_run", False)
        self._workload_config_path = getattr(args, "workload_config", None)

        self._image_adapter = AIImageBuilderAdapter() if self.use_aiimagebuilder else None
        self._workload_adapter = AIWorkloadsAdapter() if self.use_aiworkloads else None
        self._config_loader = WorkloadConfigLoader()

    @property
    def is_new_path(self) -> bool:
        """True if any new-path flag is active."""
        return self.use_aiimagebuilder or self.use_aiworkloads

    def load_workload_config(self, model_info: dict[str, Any]) -> DictConfig | None:
        """Load OmegaConf config for a workload.

        Priority:
        1. --workload-config CLI arg (explicit path)
        2. madengine.yaml in the model's scripts directory
        3. None (fall back to legacy behavior)
        """
        if self._workload_config_path:
            return WorkloadConfig.from_yaml(self._workload_config_path)

        scripts_path = model_info.get("scripts", "")
        if scripts_path:
            if scripts_path.endswith(".sh"):
                scripts_dir = os.path.dirname(scripts_path)
            else:
                scripts_dir = scripts_path

            config_path = Path(scripts_dir) / WORKLOAD_CONFIG_FILENAME
            if config_path.exists():
                return WorkloadConfig.from_yaml(str(config_path))

        return None

    def build_image(
        self,
        model_info: dict[str, Any],
        context: dict[str, Any],
        workload_cfg: DictConfig | None = None,
    ) -> ImageBuildResult:
        """Build a container image via AIImageBuilder (replaces docker build).

        Args:
            model_info: Model dict from models.json.
            context: The current context.ctx dict.
            workload_cfg: Optional per-workload OmegaConf config.
        """
        if not self._image_adapter:
            return ImageBuildResult(success=False, error="AIImageBuilder not enabled")

        image_cfg = self._build_image_config(model_info, context, workload_cfg)

        if self.dry_run:
            print(f"[DRY-RUN] Would call aiimagebuilder with:")
            print(f"  tag_prefix: {image_cfg.get('tag_prefix')}")
            print(f"  runtime: {image_cfg.get('runtime')}")
            print(f"  components: {image_cfg.get('components')}")
            return ImageBuildResult(
                success=True,
                image_tag=f"{image_cfg.get('tag_prefix', 'dry-run')}:dry-run",
            )

        start = time.time()
        result = self._image_adapter.execute(image_cfg)
        duration = time.time() - start

        if result.success:
            container = result.data.get("container", {})
            return ImageBuildResult(
                success=True,
                image_tag=container.get("image", ""),
                build_duration=duration,
            )
        else:
            return ImageBuildResult(
                success=False,
                error=result.error or "AIImageBuilder failed",
                build_duration=duration,
            )

    def run_workload(
        self,
        model_info: dict[str, Any],
        image_tag: str,
        context: dict[str, Any],
        workload_cfg: DictConfig | None = None,
    ) -> ScriptRunResult:
        """Generate and execute a workload via AIWorkloads (replaces run.sh).

        Args:
            model_info: Model dict from models.json.
            image_tag: Container image to use.
            context: The current context.ctx dict.
            workload_cfg: Optional per-workload OmegaConf config.
        """
        if not self._workload_adapter:
            return ScriptRunResult(success=False, error="AIWorkloads not enabled")

        run_cfg = self._build_workload_config(model_info, image_tag, context, workload_cfg)

        if self.dry_run:
            print(f"[DRY-RUN] Would call aiworkloads with:")
            print(f"  container.image: {run_cfg.get('container', {}).get('image')}")
            print(f"  model: {run_cfg.get('model')}")
            print(f"  scheduler: {run_cfg.get('scheduler')}")
            return ScriptRunResult(
                success=True,
                submit_command="[dry-run] would execute workload",
            )

        result = self._workload_adapter.execute(run_cfg)

        if result.success:
            submit_command = result.data.get("submit_command", "")
            return ScriptRunResult(
                success=True,
                submit_command=submit_command,
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
            )
        else:
            return ScriptRunResult(
                success=False,
                error=result.error or "AIWorkloads failed",
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
            )

    def get_extensions(self, workload_cfg: DictConfig) -> dict[str, Any]:
        """Get arbitrary extension dicts from workload config.

        These are keys beyond image_build and workload_run -- custom tooling
        sections that workload owners define. They can be injected as
        environment variables, passed to scripts, etc.
        """
        return self._config_loader.get_extensions(workload_cfg)

    # --- Private helpers ---

    def _build_image_config(
        self,
        model_info: dict[str, Any],
        context: dict[str, Any],
        workload_cfg: DictConfig | None,
    ) -> dict[str, Any]:
        """Assemble config dict for AIImageBuilder from model_info + context + workload_cfg."""
        model_name = model_info.get("name", "workload").replace("/", "_").lower()
        dockerfile = model_info.get("dockerfile", "")

        cfg: dict[str, Any] = {
            "tag_prefix": f"ci-{model_name}",
            "runtime": "rocm",
            "components": {},
        }

        if context.get("docker_build_arg", {}).get("BASE_DOCKER"):
            cfg["base_image"] = context["docker_build_arg"]["BASE_DOCKER"]

        gpu_vendor = context.get("docker_env_vars", {}).get("MAD_GPU_VENDOR", "")
        if "NVIDIA" in gpu_vendor:
            cfg["runtime"] = "cuda"

        if workload_cfg:
            ib_cfg = OmegaConf.to_container(
                workload_cfg.get("image_build", {}), resolve=True
            )
            if isinstance(ib_cfg, dict):
                if ib_cfg.get("tag_prefix"):
                    cfg["tag_prefix"] = ib_cfg["tag_prefix"]
                if ib_cfg.get("runtime"):
                    cfg["runtime"] = ib_cfg["runtime"]
                if ib_cfg.get("components"):
                    cfg["components"] = ib_cfg["components"]
                if ib_cfg.get("max_jobs"):
                    cfg["max_jobs"] = ib_cfg["max_jobs"]

        return cfg

    def _build_workload_config(
        self,
        model_info: dict[str, Any],
        image_tag: str,
        context: dict[str, Any],
        workload_cfg: DictConfig | None,
    ) -> dict[str, Any]:
        """Assemble config dict for AIWorkloads from model_info + context + workload_cfg."""
        gpu_vendor = context.get("docker_env_vars", {}).get("MAD_GPU_VENDOR", "")
        runtime = "cuda" if "NVIDIA" in gpu_vendor else "rocm"

        cfg: dict[str, Any] = {
            "container": {
                "image": image_tag,
                "type": "docker",
                "runtime": runtime,
            },
            "scheduler": "local",
            "model": model_info.get("name", ""),
            "distributed": {},
            "profiling": {},
            "paths": {},
        }

        n_gpus = model_info.get("n_gpus", "-1")
        if n_gpus and n_gpus != "-1":
            cfg["distributed"]["gpus_per_node"] = int(n_gpus)

        if model_info.get("args"):
            cfg["model_args"] = model_info["args"]

        if workload_cfg:
            wr_cfg = OmegaConf.to_container(
                workload_cfg.get("workload_run", {}), resolve=True
            )
            if isinstance(wr_cfg, dict):
                if wr_cfg.get("scheduler"):
                    cfg["scheduler"] = wr_cfg["scheduler"]
                if wr_cfg.get("model"):
                    cfg["model"] = wr_cfg["model"]
                if wr_cfg.get("distributed"):
                    cfg["distributed"].update(wr_cfg["distributed"])
                if wr_cfg.get("profiling"):
                    cfg["profiling"] = wr_cfg["profiling"]
                if wr_cfg.get("container"):
                    for k, v in wr_cfg["container"].items():
                        if k != "image":
                            cfg["container"][k] = v

            extensions = self._config_loader.get_extensions(workload_cfg)
            if extensions:
                cfg["extensions"] = extensions

        return cfg
