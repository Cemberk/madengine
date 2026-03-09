"""Bridge between madengine and optimizer backends (aitemplate, tensorrt, migraphx).

Loads optimizer based on config or CLI args, executes optimization,
and generates comparison reports. Optionally imports optimizer implementations
from aise when available.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Optional import from aise - try common paths
_AISE_OPTIMIZERS_AVAILABLE = False
_AITemplateOptimizer = None
_get_registry = None

try:
    # Try aise optimizers (aise/scripts/optimizers/)
    _aise_scripts = Path(__file__).resolve().parents[4] / "aise" / "scripts"
    if _aise_scripts.exists() and str(_aise_scripts) not in sys.path:
        sys.path.insert(0, str(_aise_scripts))
    from optimizers import AITemplateOptimizer as _AIT
    from optimizers import get_registry as _get_registry_fn
    _AITemplateOptimizer = _AIT
    _get_registry = _get_registry_fn
    _AISE_OPTIMIZERS_AVAILABLE = True
except ImportError:
    pass


@dataclass
class OptimizationResult:
    """Result from an optimizer run."""
    success: bool
    model_name: str = ""
    backend: str = ""
    original_performance: str = ""
    optimized_performance: str = ""
    speedup: float | None = None
    compilation_time_s: float | None = None
    error: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonReport:
    """Report comparing original vs optimized runs."""
    model_name: str
    original: dict[str, Any]
    optimized: dict[str, Any]
    speedup: float | None = None
    summary: str = ""


class OptimizerBridge:
    """Bridge between madengine and the optimizer registry.

    Loads optimizer based on config or CLI args, executes optimization,
    and generates comparison reports.
    """

    SUPPORTED_BACKENDS = {"aitemplate", "tensorrt", "migraphx"}
    DEFAULT_BACKEND = "aitemplate"

    def __init__(self, args: Any):
        self.args = args
        self.optimizer_arg = getattr(args, "optimizer", None)
        self.optimize_only = getattr(args, "optimize_only", False)
        self.compare = getattr(args, "compare", False)

    def _resolve_backend(self, config: dict[str, Any] | None) -> str | None:
        """Resolve optimizer backend from CLI or config."""
        backend = self.optimizer_arg
        if backend == "auto" and config:
            opt = config.get("optimization", {})
            backend = opt.get("backend") or opt.get("optimizer")
            if isinstance(backend, str):
                backend = backend.lower()
        if backend and backend != "auto":
            return backend
        if config and self._has_optimization_config(config):
            return config.get("optimization", {}).get("backend") or self.DEFAULT_BACKEND
        return None

    def _has_optimization_config(self, config: dict[str, Any]) -> bool:
        """Check if config has an optimization section."""
        return bool(config.get("optimization")) or bool(
            config.get("workload_run", {}).get("optimizer")
        )

    def load_optimizer(self, backend: str, model_type: str = "resnet", gpu_vendor: str | None = None):
        """Load optimizer instance for the given backend."""
        backend = backend.lower()
        if backend not in self.SUPPORTED_BACKENDS and backend != "auto":
            raise ValueError(
                f"Unknown optimizer backend: {backend}. "
                f"Supported: {', '.join(self.SUPPORTED_BACKENDS)}, auto"
            )
        if backend == "auto" and _get_registry:
            registry = _get_registry()
            opt = registry.get_optimizer(model_type=model_type, gpu_vendor=gpu_vendor)
            if opt is None:
                raise RuntimeError(
                    f"No optimizer available for model_type={model_type}. "
                    "Use --optimizer aitemplate explicitly."
                )
            return opt
        if backend == "aitemplate":
            if _AITemplateOptimizer is None:
                raise RuntimeError(
                    "AITemplate optimizer not available. "
                    "Ensure aise/scripts is on PYTHONPATH or install aise."
                )
            return _AITemplateOptimizer()
        if backend in ("tensorrt", "migraphx"):
            raise NotImplementedError(
                f"Optimizer '{backend}' is not yet implemented. "
                "Use --optimizer aitemplate for AITemplate."
            )
        raise ValueError(f"Unsupported backend: {backend}")

    def execute_optimization(
        self,
        model_info: dict[str, Any],
        workload_config: dict[str, Any] | None,
        context: dict[str, Any],
    ) -> OptimizationResult:
        """Execute optimization for a model.

        Args:
            model_info: Model dict from models.json discovery.
            workload_config: Per-workload YAML config (with optional optimization section).
            context: madengine context.ctx.

        Returns:
            OptimizationResult with success/error and metrics.
        """
        config = workload_config or model_info
        backend = self._resolve_backend(config)
        if not backend:
            return OptimizationResult(
                success=False,
                model_name=model_info.get("name", "unknown"),
                error="No optimizer specified. Use --optimizer or add optimization section to config.",
            )

        opt_cfg = config.get("optimization", {})
        workload_run = config.get("workload_run", {})
        model_type = opt_cfg.get("model_type") or workload_run.get("model_type") or "resnet"
        gpu_vendor = None
        if context.get("docker_env_vars", {}).get("MAD_GPU_VENDOR", "").find("AMD") != -1:
            gpu_vendor = "amd"
        elif context.get("docker_env_vars", {}).get("MAD_GPU_VENDOR", "").find("NVIDIA") != -1:
            gpu_vendor = "nvidia"

        try:
            optimizer = self.load_optimizer(backend, model_type=model_type, gpu_vendor=gpu_vendor)
        except (ValueError, RuntimeError, NotImplementedError) as e:
            return OptimizationResult(
                success=False,
                model_name=model_info.get("name", "unknown"),
                backend=backend,
                error=str(e),
            )

        model_name = model_info.get("name", "unknown")

        # Registry returns Optimizer with optimize/benchmark; raw AIT uses optimize_model
        if hasattr(optimizer, "optimize") and hasattr(optimizer, "benchmark"):
            return self._run_registry_optimizer(optimizer, model_info, config)

        if backend == "aitemplate":
            return self._run_aitemplate(optimizer, model_info, config)

        return OptimizationResult(
            success=False,
            model_name=model_name,
            backend=backend,
            error=f"Backend {backend} execution not implemented",
        )

    def _run_registry_optimizer(
        self,
        optimizer: Any,
        model_info: dict[str, Any],
        config: dict[str, Any],
    ) -> OptimizationResult:
        """Run optimizer from aise registry (optimize/benchmark dict interface)."""
        opt_cfg = config.get("optimization", {})
        workload_run = config.get("workload_run", {})
        bp = workload_run.get("benchmark_params", {})
        model_config = {
            "name": model_info.get("name", "unknown"),
            "model_type": opt_cfg.get("model_type") or workload_run.get("model_type") or "resnet",
            "batch_sizes": bp.get("batch_sizes", [1, 8, 32]),
            "sequence_lengths": bp.get("sequence_lengths", [128, 512]),
            "image_sizes": bp.get("image_sizes", [224]),
            "precision": opt_cfg.get("target_precision", "fp16"),
            "use_fx2ait": opt_cfg.get("use_fx2ait", False),
        }
        try:
            if self.optimize_only:
                result = optimizer.optimize(model_config)
                perf = getattr(result, "optimized_latency_ms", None)
            else:
                result = optimizer.benchmark(model_config)
                perf = getattr(result, "latency_ms", None) or getattr(result, "optimized_latency_ms", None)
            return OptimizationResult(
                success=result.status == "success",
                model_name=result.model_name,
                backend=optimizer.name() if hasattr(optimizer, "name") and callable(optimizer.name) else "",
                optimized_performance=str(perf) if perf is not None else "",
                speedup=getattr(result, "speedup", None),
                compilation_time_s=getattr(result, "compilation_time_s", None),
                error=result.error or "",
                details={},
            )
        except Exception as e:
            return OptimizationResult(
                success=False,
                model_name=model_info.get("name", "unknown"),
                backend=getattr(optimizer, "name", lambda: "")(),
                error=str(e),
            )

    def _run_aitemplate(
        self,
        optimizer: Any,
        model_info: dict[str, Any],
        config: dict[str, Any],
    ) -> OptimizationResult:
        """Run AITemplate optimizer."""
        from dataclasses import asdict

        opt_cfg = config.get("optimization", {})
        workload_run = config.get("workload_run", {})
        model_type = opt_cfg.get("model_type") or workload_run.get("model_type") or "resnet"
        bp = workload_run.get("benchmark_params", {})

        try:
            AITModelConfig = getattr(optimizer, "AITModelConfig", None)
            if not AITModelConfig:
                return OptimizationResult(
                    success=False,
                    model_name=model_info.get("name", "unknown"),
                    backend="aitemplate",
                    error="AITemplateOptimizer missing AITModelConfig",
                )

            model_config = AITModelConfig(
                name=model_info.get("name", "unknown"),
                model_type=model_type,
                hf_model_id=opt_cfg.get("hf_model_id"),
                batch_sizes=bp.get("batch_sizes", [1, 8, 32]),
                sequence_lengths=bp.get("sequence_lengths", [128, 512]),
                image_sizes=bp.get("image_sizes", [224]),
                precision=opt_cfg.get("target_precision", "fp16"),
                use_fx2ait=opt_cfg.get("use_fx2ait", False),
            )

            benchmark = not self.optimize_only
            result = optimizer.optimize_model(model_config, benchmark=benchmark)

            return OptimizationResult(
                success=result.status == "success",
                model_name=result.model_name,
                backend="aitemplate",
                optimized_performance=str(result.optimized_latency_ms) if result.optimized_latency_ms else "",
                speedup=result.speedup,
                compilation_time_s=result.compilation_time_s,
                error=result.error or "",
                details=asdict(result) if hasattr(result, "__dataclass_fields__") else {},
            )
        except Exception as e:
            return OptimizationResult(
                success=False,
                model_name=model_info.get("name", "unknown"),
                backend="aitemplate",
                error=str(e),
            )

    def generate_comparison_report(
        self,
        original_result: dict[str, Any],
        optimized_result: OptimizationResult,
        output_path: str | Path = "optimization_comparison_report.json",
    ) -> str:
        """Generate comparison report between original and optimized runs.

        Args:
            original_result: RunDetails-like dict (performance, metric, status, etc.).
            optimized_result: OptimizationResult from execute_optimization.
            output_path: Where to write the report.

        Returns:
            Path to the written report file.
        """
        report = {
            "model": optimized_result.model_name,
            "original": {
                "performance": original_result.get("performance", ""),
                "metric": original_result.get("metric", ""),
                "status": original_result.get("status", ""),
                "test_duration": original_result.get("test_duration", ""),
            },
            "optimized": {
                "performance": optimized_result.optimized_performance,
                "backend": optimized_result.backend,
                "status": "SUCCESS" if optimized_result.success else "FAILURE",
                "compilation_time_s": optimized_result.compilation_time_s,
            },
            "comparison": {},
        }

        speedup = optimized_result.speedup
        if speedup is not None:
            report["comparison"]["speedup"] = speedup
        elif original_result.get("performance") and optimized_result.optimized_performance:
            try:
                orig = float(original_result["performance"])
                opt = float(optimized_result.optimized_performance)
                if orig > 0:
                    report["comparison"]["speedup"] = opt / orig
            except (ValueError, TypeError):
                pass

        summary_parts = [
            f"Model: {optimized_result.model_name}",
            f"Original: {original_result.get('performance', 'N/A')} {original_result.get('metric', '')}",
            f"Optimized: {optimized_result.optimized_performance or 'N/A'}",
        ]
        if "speedup" in report["comparison"]:
            summary_parts.append(f"Speedup: {report['comparison']['speedup']:.2f}x")
        report["summary"] = " | ".join(summary_parts)

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        return str(path)


def get_optimizer_registry() -> dict[str, str]:
    """Return available optimizers from aise (when present) or built-in list."""
    registry = {}
    if _AISE_OPTIMIZERS_AVAILABLE:
        registry["aitemplate"] = "AITemplate (from aise)"
    else:
        registry["aitemplate"] = "AITemplate (aise not on PYTHONPATH)"
    registry["tensorrt"] = "TensorRT (planned)"
    registry["migraphx"] = "MIGraphX (planned)"
    return registry
