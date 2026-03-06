"""Main pipeline coordinator."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from omegaconf import DictConfig, OmegaConf

from madengine.adapters import AIImageBuilderAdapter, AIWorkloadsAdapter, AdapterResult
from madengine.credentials import CredentialManager
from madengine.data import get_data_provider
from madengine.metrics import MetricsCollector


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""

    success: bool
    job_id: str | None = None
    image: str | None = None
    submit_command: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_seconds: float = 0.0
    stages: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "job_id": self.job_id,
            "image": self.image,
            "submit_command": self.submit_command,
            "metrics": self.metrics,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "stages": self.stages,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_text(self) -> str:
        lines = [
            "=" * 60,
            "PIPELINE RESULT",
            "=" * 60,
            f"Success: {self.success}",
            f"Duration: {self.duration_seconds:.1f}s",
        ]
        if self.image:
            lines.append(f"Image: {self.image}")
        if self.submit_command:
            lines.append(f"Submit: {self.submit_command}")
        if self.error:
            lines.append(f"Error: {self.error}")
        if self.metrics:
            lines.append("Metrics:")
            for k, v in self.metrics.items():
                lines.append(f"  {k}: {v}")
        lines.append("=" * 60)
        return "\n".join(lines)


class Orchestrator:
    """Main pipeline orchestrator.

    Coordinates:
    1. Credentials setup
    2. Data staging
    3. Image building (AIImageBuilder)
    4. Script generation (AIWorkloads)
    5. Job execution
    6. Metrics collection
    7. Database storage
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        tools_cfg = cfg.get("tools", {})

        self.image_adapter = AIImageBuilderAdapter(
            executable=tools_cfg.get("aiimagebuilder", {}).get("executable", "aiimagebuilder"),
            timeout=cfg.get("timeouts", {}).get("image_build", 7200),
        )
        self.workload_adapter = AIWorkloadsAdapter(
            executable=tools_cfg.get("aiworkloads", {}).get("executable", "aiworkloads"),
            timeout=300,
        )

        self.credentials = CredentialManager(cfg.get("credentials", {}))
        self.data_provider = get_data_provider(cfg.get("data", {}))
        self.metrics_collector = MetricsCollector(cfg.get("metrics", {}))

        db_cfg = cfg.get("database", {})
        if db_cfg.get("backend") and db_cfg.get("backend") != "none":
            from madengine.database import get_database_backend

            self.database = get_database_backend(db_cfg)
        else:
            self.database = None

    def run(self) -> PipelineResult:
        """Execute the full pipeline."""
        start_time = time.time()
        stages: dict[str, dict] = {}

        try:
            # Stage 1: Setup credentials
            self.credentials.setup_environment()
            stages["credentials"] = {"success": True}

            # Stage 2: Stage data
            if self.cfg.get("data", {}).get("dataset"):
                data_result = self._stage_data()
                stages["data"] = data_result
                if not data_result["success"]:
                    return PipelineResult(
                        success=False,
                        error=f"Data staging failed: {data_result.get('error')}",
                        stages=stages,
                        duration_seconds=time.time() - start_time,
                    )
                data_path = data_result["local_path"]
            else:
                data_path = None

            # Stage 3: Build image (unless skipped)
            if not self.cfg.get("skip_image_build", False):
                image_result = self._build_image()
                stages["image"] = image_result.to_dict()
                if not image_result.success:
                    return PipelineResult(
                        success=False,
                        error=f"Image build failed: {image_result.error}",
                        stages=stages,
                        duration_seconds=time.time() - start_time,
                    )
                container_info = image_result.data.get("container", {})
            else:
                container_info = OmegaConf.to_container(
                    self.cfg.get("training", {}).get("container", {}), resolve=True
                )
                stages["image"] = {"skipped": True}

            # Stage 4: Generate scripts
            workload_result = self._generate_scripts(container_info, data_path)
            stages["workload"] = workload_result.to_dict()
            if not workload_result.success:
                return PipelineResult(
                    success=False,
                    error=f"Script generation failed: {workload_result.error}",
                    stages=stages,
                    duration_seconds=time.time() - start_time,
                )

            submit_command = workload_result.data.get("submit_command")

            # Stage 5: Execute job (unless dry-run)
            if self.cfg.get("dry_run", False):
                stages["execution"] = {"skipped": True, "reason": "dry_run"}
                return PipelineResult(
                    success=True,
                    image=container_info.get("image") if isinstance(container_info, dict) else None,
                    submit_command=submit_command,
                    stages=stages,
                    duration_seconds=time.time() - start_time,
                )

            exec_result = self._execute_job(submit_command)
            stages["execution"] = exec_result

            if not exec_result["success"]:
                return PipelineResult(
                    success=False,
                    image=container_info.get("image") if isinstance(container_info, dict) else None,
                    submit_command=submit_command,
                    error=f"Job execution failed: {exec_result.get('error')}",
                    stages=stages,
                    duration_seconds=time.time() - start_time,
                )

            # Stage 6: Collect metrics
            metrics = self.metrics_collector.collect(exec_result)
            stages["metrics"] = {"success": True, "data": metrics}

            # Stage 7: Store in database
            if self.database:
                self._store_metrics(metrics)
                stages["database"] = {"success": True}

            return PipelineResult(
                success=True,
                image=container_info.get("image") if isinstance(container_info, dict) else None,
                submit_command=submit_command,
                metrics=metrics,
                stages=stages,
                duration_seconds=time.time() - start_time,
            )

        except Exception as e:
            return PipelineResult(
                success=False,
                error=str(e),
                stages=stages,
                duration_seconds=time.time() - start_time,
            )

    def _stage_data(self) -> dict:
        dataset = self.cfg.data.dataset
        result = self.data_provider.stage(dataset)
        return result.to_dict()

    def _build_image(self) -> AdapterResult:
        image_cfg = OmegaConf.to_container(self.cfg.get("image", {}), resolve=True)
        return self.image_adapter.execute(image_cfg)

    def _generate_scripts(
        self,
        container_info: dict,
        data_path: str | None,
    ) -> AdapterResult:
        training_cfg = OmegaConf.to_container(self.cfg.get("training", {}), resolve=True)
        training_cfg["container"] = container_info

        if data_path:
            training_cfg.setdefault("paths", {})
            training_cfg["paths"]["data"] = data_path

        return self.workload_adapter.execute(training_cfg)

    def _execute_job(self, submit_command: str) -> dict:
        timeout = self.cfg.get("timeouts", {}).get("job_execution", 86400)
        start = time.time()
        try:
            result = subprocess.run(
                submit_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration_seconds": time.time() - start,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Job timed out after {timeout}s",
                "duration_seconds": time.time() - start,
            }

    def _store_metrics(self, metrics: dict) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": OmegaConf.to_container(self.cfg, resolve=True),
            "metrics": metrics,
        }
        self.database.insert(record)
