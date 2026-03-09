"""Tests for config_converter: YAML config <-> models.json 1:1 conversion."""

import json
import os
import tempfile

import pytest

from madengine.tooling.config_converter import (
    config_to_model_info,
    config_to_build_args,
    config_to_dockerfile,
    config_to_context_dict,
    load_all_configs,
    load_single_config,
    load_yaml_config,
)


SAMPLE_INFERENCEMAX_CONFIG = {
    "name": "inferencemax_dsr1_fp8_mi300x_sglang",
    "owner": "inferencemax@amd.com",
    "tags": ["inferencemax", "benchmark", "sglang", "mi300x", "dsr1"],
    "timeout": 28400,
    "n_gpus": "8",
    "training_precision": "fp8",
    "skip_gpu_arch": "",
    "docker": {
        "dockerfile": "docker/inferencemax",
        "context": "docker",
        "file": "docker/inferencemax.ubuntu.amd.Dockerfile",
        "base_image": "rocm/7.0:rocm7.0_ubuntu_22.04_sgl-dev-v0.5.2-rocm7.0-mi30x-20250915",
        "build_args": {
            "BASE_DOCKER": "rocm/7.0:rocm7.0_ubuntu_22.04_sgl-dev-v0.5.2-rocm7.0-mi30x-20250915",
        },
    },
    "run": {
        "scripts": "scripts/inferencemax",
        "args": "",
        "multiple_results": "results_inferencemax.csv",
    },
    "custom": {
        "image": "rocm/7.0:rocm7.0_ubuntu_22.04_sgl-dev-v0.5.2-rocm7.0-mi30x-20250915",
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "model_prefix": "dsr1",
        "framework": "sglang",
        "precision": "fp8",
    },
}


SAMPLE_TRANSFORMERS_UT_CONFIG = {
    "name": "transformers_ut",
    "owner": "cem.bozkus@amd.com",
    "tags": ["unit-tests", "Monday"],
    "timeout": -1,
    "n_gpus": "-1",
    "training_precision": "amp_fp16",
    "skip_gpu_arch": "",
    "docker": {
        "dockerfile": "docker/upstream_huggingface_ut",
        "context": "docker",
        "variants": [
            {
                "file": "docker/upstream_huggingface_ut.ubuntu.amd.Dockerfile",
                "context": "{'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}",
                "base_image": "rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.7.1",
                "build_args": {
                    "BASE_DOCKER": "rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.7.1",
                    "HFTRANSFORMERS_REPO": "https://github.com/huggingface/transformers",
                    "HFTRANSFORMERS_BRANCH": "main",
                },
            },
            {
                "file": "docker/upstream_huggingface_ut.ubuntu.nvidia.Dockerfile",
                "context": "{'gpu_vendor': 'NVIDIA', 'guest_os': 'UBUNTU'}",
                "base_image": "nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04",
                "build_args": {
                    "BASE_DOCKER": "nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04",
                },
            },
        ],
    },
    "run": {
        "scripts": "scripts/transformers_ut.sh",
        "args": "",
        "multiple_results": "results_huggingface_ut.csv",
    },
}


SAMPLE_EXEC_DASHBOARD_CONFIG = {
    "name": "exec_dashboard",
    "owner": "cem.bozkus@amd.com",
    "tags": ["exec-dashboard", "afo"],
    "timeout": 1500000,
    "n_gpus": "8",
    "docker": {
        "dockerfile": "docker/exec_dashboard",
        "context": "docker",
        "file": "docker/exec_dashboard.ubuntu.amd.Dockerfile",
        "base_image": "rocm/aigmodels-private:exec_dashboard_vLLM_nightly",
    },
    "run": {
        "scripts": "scripts/exec_dashboard",
        "args": "",
        "multiple_results": "exec_dashboard_results.csv",
    },
    "custom": {
        "afo_benchmarks": {
            "run_gemm": True,
            "run_flash_attention": True,
            "gemm_sections": "genericLLM_m cube",
        },
    },
}


class TestConfigToModelInfo:
    def test_basic_fields(self):
        result = config_to_model_info(SAMPLE_INFERENCEMAX_CONFIG)
        assert result["name"] == "inferencemax_dsr1_fp8_mi300x_sglang"
        assert result["owner"] == "inferencemax@amd.com"
        assert result["tags"] == ["inferencemax", "benchmark", "sglang", "mi300x", "dsr1"]
        assert result["timeout"] == 28400
        assert result["n_gpus"] == "8"
        assert result["training_precision"] == "fp8"
        assert result["skip_gpu_arch"] == ""

    def test_docker_fields(self):
        result = config_to_model_info(SAMPLE_INFERENCEMAX_CONFIG)
        assert result["dockerfile"] == "docker/inferencemax"
        assert result["dockercontext"] == "docker"

    def test_run_fields(self):
        result = config_to_model_info(SAMPLE_INFERENCEMAX_CONFIG)
        assert result["scripts"] == "scripts/inferencemax"
        assert result["args"] == ""
        assert result["multiple_results"] == "results_inferencemax.csv"

    def test_custom_fields_flattened(self):
        result = config_to_model_info(SAMPLE_INFERENCEMAX_CONFIG)
        assert result["image"] == "rocm/7.0:rocm7.0_ubuntu_22.04_sgl-dev-v0.5.2-rocm7.0-mi30x-20250915"
        assert result["model"] == "deepseek-ai/DeepSeek-R1-0528"
        assert result["model_prefix"] == "dsr1"
        assert result["framework"] == "sglang"
        assert result["precision"] == "fp8"

    def test_transformers_ut_no_custom(self):
        result = config_to_model_info(SAMPLE_TRANSFORMERS_UT_CONFIG)
        assert result["name"] == "transformers_ut"
        assert result["dockerfile"] == "docker/upstream_huggingface_ut"
        assert result["scripts"] == "scripts/transformers_ut.sh"
        assert "image" not in result
        assert "model" not in result


class TestConfigToBuildArgs:
    def test_single_variant(self):
        args = config_to_build_args(SAMPLE_INFERENCEMAX_CONFIG)
        assert args["BASE_DOCKER"] == "rocm/7.0:rocm7.0_ubuntu_22.04_sgl-dev-v0.5.2-rocm7.0-mi30x-20250915"

    def test_multi_variant_amd(self):
        args = config_to_build_args(SAMPLE_TRANSFORMERS_UT_CONFIG, platform="amd")
        assert args["BASE_DOCKER"] == "rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.7.1"
        assert args["HFTRANSFORMERS_BRANCH"] == "main"

    def test_multi_variant_nvidia(self):
        args = config_to_build_args(SAMPLE_TRANSFORMERS_UT_CONFIG, platform="nvidia")
        assert args["BASE_DOCKER"] == "nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04"


class TestConfigToDockerfile:
    def test_single_file(self):
        result = config_to_dockerfile(SAMPLE_INFERENCEMAX_CONFIG)
        assert result == "docker/inferencemax.ubuntu.amd.Dockerfile"

    def test_multi_variant_amd(self):
        result = config_to_dockerfile(SAMPLE_TRANSFORMERS_UT_CONFIG, platform="amd")
        assert result == "docker/upstream_huggingface_ut.ubuntu.amd.Dockerfile"

    def test_multi_variant_nvidia(self):
        result = config_to_dockerfile(SAMPLE_TRANSFORMERS_UT_CONFIG, platform="nvidia")
        assert result == "docker/upstream_huggingface_ut.ubuntu.nvidia.Dockerfile"


class TestConfigToContextDict:
    def test_inferencemax_context(self):
        ctx = config_to_context_dict(SAMPLE_INFERENCEMAX_CONFIG)
        assert ctx["docker_env_vars"]["MODEL"] == "deepseek-ai/DeepSeek-R1-0528"
        assert ctx["docker_env_vars"]["FRAMEWORK"] == "sglang"
        assert ctx["docker_env_vars"]["PRECISION"] == "fp8"
        assert ctx["docker_env_vars"]["MODEL_PREFIX"] == "dsr1"
        assert ctx["docker_build_arg"]["BASE_DOCKER"] == "rocm/7.0:rocm7.0_ubuntu_22.04_sgl-dev-v0.5.2-rocm7.0-mi30x-20250915"

    def test_exec_dashboard_afo_context(self):
        ctx = config_to_context_dict(SAMPLE_EXEC_DASHBOARD_CONFIG)
        assert ctx["docker_env_vars"]["RUN_GEMM"] == "true"
        assert ctx["docker_env_vars"]["RUN_FLASH_ATTENTION"] == "true"
        assert ctx["docker_env_vars"]["GEMM_SECTIONS"] == "genericLLM_m cube"

    def test_no_custom_returns_empty(self):
        ctx = config_to_context_dict(SAMPLE_TRANSFORMERS_UT_CONFIG)
        assert ctx == {}


class TestLoadYamlConfig:
    def test_load_from_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("name: test_workload\ntimeout: 3600\n")
            f.flush()
            config = load_yaml_config(f.name)
            assert config["name"] == "test_workload"
            assert config["timeout"] == 3600
            os.unlink(f.name)


class TestLoadAllConfigs:
    def test_load_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["wl1", "wl2"]:
                path = os.path.join(tmpdir, f"{name}.yaml")
                with open(path, "w") as f:
                    f.write(f"name: {name}\ntimeout: 3600\ndocker:\n  dockerfile: docker/{name}\nrun:\n  scripts: scripts/{name}\n")

            results = load_all_configs(tmpdir)
            assert len(results) == 2
            names = [r["name"] for r in results]
            assert "wl1" in names
            assert "wl2" in names


class TestLoadSingleConfig:
    def test_roundtrip_inferencemax(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            try:
                import yaml
                yaml.dump(SAMPLE_INFERENCEMAX_CONFIG, f)
            except ImportError:
                import json as json_mod
                content = json_mod.dumps(SAMPLE_INFERENCEMAX_CONFIG)
                f.write(content)
            f.flush()

            result = load_single_config(f.name)
            assert result["name"] == "inferencemax_dsr1_fp8_mi300x_sglang"
            assert result["model"] == "deepseek-ai/DeepSeek-R1-0528"
            assert result["framework"] == "sglang"
            os.unlink(f.name)


class TestRoundTrip:
    """Verify that models.json -> YAML -> model_info produces the same data."""

    MODELS_JSON_ENTRY = {
        "name": "inferencemax_dsr1_fp8_mi300x_sglang",
        "dockerfile": "docker/inferencemax",
        "scripts": "scripts/inferencemax",
        "timeout": 28400,
        "n_gpus": "8",
        "owner": "inferencemax@amd.com",
        "training_precision": "fp8",
        "multiple_results": "results_inferencemax.csv",
        "tags": ["inferencemax", "benchmark", "sglang", "mi300x", "dsr1"],
        "args": "",
        "skip_gpu_arch": "",
        "image": "rocm/7.0:rocm7.0_ubuntu_22.04_sgl-dev-v0.5.2-rocm7.0-mi30x-20250915",
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "model_prefix": "dsr1",
        "framework": "sglang",
        "precision": "fp8",
    }

    def test_roundtrip_preserves_key_fields(self):
        result = config_to_model_info(SAMPLE_INFERENCEMAX_CONFIG)
        original = self.MODELS_JSON_ENTRY

        assert result["name"] == original["name"]
        assert result["dockerfile"] == original["dockerfile"]
        assert result["scripts"] == original["scripts"]
        assert result["timeout"] == original["timeout"]
        assert str(result["n_gpus"]) == str(original["n_gpus"])
        assert result["owner"] == original["owner"]
        assert result["training_precision"] == original["training_precision"]
        assert result["multiple_results"] == original["multiple_results"]
        assert result["tags"] == original["tags"]
        assert result["args"] == original["args"]
        assert result["model"] == original["model"]
        assert result["framework"] == original["framework"]
        assert result["precision"] == original["precision"]
        assert result["image"] == original["image"]
        assert result["model_prefix"] == original["model_prefix"]
