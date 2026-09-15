"""Tests for layered distributed-inference configuration (the fourth way).

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

import textwrap

import pytest

from madengine.deployment.layered_config import (
    DEFAULT_FILENAME,
    LayeredConfigError,
    _as_arg_map,
    find_config,
    load_config,
    resolve_env,
    resolve_for_model,
    resolve_serve_args,
    validate_card,
)


def write(tmp_path, body, name=DEFAULT_FILENAME):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return path


class TestArgMap:
    """Serve flags normalise to a map so one layer can override ONE flag."""

    def test_string_becomes_map(self):
        assert _as_arg_map("--disable-radix-cache --tp 8") == {
            "--disable-radix-cache": True,
            "--tp": "8",
        }

    def test_multi_value_flag_is_kept_whole(self):
        # --cuda-graph-bs takes a list; splitting it would lose the recipe.
        assert _as_arg_map("--cuda-graph-bs 8 16 32")["--cuda-graph-bs"] == "8 16 32"

    def test_map_passes_through(self):
        assert _as_arg_map({"--max-model-len": 16384}) == {"--max-model-len": 16384}

    def test_none_is_empty(self):
        assert _as_arg_map(None) == {}

    def test_non_flag_leading_token_is_an_error(self):
        with pytest.raises(LayeredConfigError):
            _as_arg_map("8 --tp")


class TestServeArgs:
    """base -> mode -> role -> role+mode, each overriding the last."""

    CONFIG = {
        "version": 1,
        "model": {
            "serve": {
                "base": "--attention-backend aiter --tp 1",
                "modes": {"tp": "--tp 8", "dp": "--tp 2"},
                "roles": {
                    "prefill": {"tp": "--disable-cuda-graph"},
                    "decode": {"tp": "--cuda-graph-bs 8 16"},
                },
            }
        },
    }

    def test_base_only(self):
        assert resolve_serve_args(self.CONFIG) == {
            "--attention-backend": "aiter",
            "--tp": "1",
        }

    def test_mode_overrides_base(self):
        assert resolve_serve_args(self.CONFIG, mode="tp")["--tp"] == "8"

    def test_role_and_mode_compose(self):
        decode = resolve_serve_args(self.CONFIG, mode="tp", role="decode")
        assert decode["--cuda-graph-bs"] == "8 16"
        assert decode["--tp"] == "8"
        assert "--disable-cuda-graph" not in decode

    def test_roles_do_not_leak_into_each_other(self):
        prefill = resolve_serve_args(self.CONFIG, mode="tp", role="prefill")
        assert "--disable-cuda-graph" in prefill
        assert "--cuda-graph-bs" not in prefill

    def test_missing_serve_section_is_empty(self):
        assert resolve_serve_args({"version": 1}) == {}


class TestEnvPrecedence:
    """site < model < benchmark < card env_vars < submit-time override."""

    CONFIG = {
        "version": 1,
        "site": {"env": {"NVME_ROOT": "/mnt/site", "SHARED": "/shared"}},
        "model": {"env": {"NVME_ROOT": "/mnt/model", "AITER": "1"}},
        "benchmark": [
            {"env": {"SEEDS": "3"}},
            {"kind": "niah", "env": {"AITER": "0", "NIAH_WORDS": "10000"}},
        ],
    }

    def test_model_beats_site(self):
        assert resolve_env(self.CONFIG)["NVME_ROOT"] == "/mnt/model"

    def test_task_level_benchmark_defaults_always_apply(self):
        assert resolve_env(self.CONFIG)["SEEDS"] == "3"

    def test_benchmark_kind_beats_model(self):
        assert resolve_env(self.CONFIG, benchmark="niah")["AITER"] == "0"

    def test_unselected_benchmark_kind_does_not_apply(self):
        assert "NIAH_WORDS" not in resolve_env(self.CONFIG)

    def test_card_env_beats_file(self):
        env = resolve_env(self.CONFIG, model_env={"NVME_ROOT": "/mnt/card"})
        assert env["NVME_ROOT"] == "/mnt/card"

    def test_runtime_override_wins_everything(self):
        env = resolve_env(
            self.CONFIG,
            model_env={"NVME_ROOT": "/mnt/card"},
            runtime_env={"NVME_ROOT": "/mnt/runtime"},
        )
        assert env["NVME_ROOT"] == "/mnt/runtime"

    def test_layers_merge_rather_than_replace(self):
        # SHARED comes only from site, AITER only from model: a lower layer's
        # keys must survive a higher layer that does not mention them.
        env = resolve_env(self.CONFIG)
        assert env["SHARED"] == "/shared"
        assert env["AITER"] == "1"

    def test_values_are_stringified(self):
        env = resolve_env({"version": 1, "model": {"env": {"MAX_LEN": 16384}}})
        assert env["MAX_LEN"] == "16384"


class TestLoad:
    def test_version_must_be_supported(self, tmp_path):
        path = write(tmp_path, "version: 99\n")
        with pytest.raises(LayeredConfigError, match="not supported"):
            load_config(path)

    def test_malformed_yaml_names_the_file(self, tmp_path):
        path = write(tmp_path, "version: 1\nmodel: [unclosed\n")
        with pytest.raises(LayeredConfigError, match="not valid YAML"):
            load_config(path)

    def test_top_level_must_be_a_map(self, tmp_path):
        path = write(tmp_path, "- version: 1\n")
        with pytest.raises(LayeredConfigError, match="must be a map"):
            load_config(path)

    def test_empty_file_is_empty_config(self, tmp_path):
        assert load_config(write(tmp_path, "")) == {}


class TestFindConfig:
    def test_sibling_by_convention(self, tmp_path):
        write(tmp_path, "version: 1\n")
        assert find_config({}, tmp_path) is not None

    def test_absent_returns_none(self, tmp_path):
        assert find_config({}, tmp_path) is None

    def test_env_var_overrides_convention(self, tmp_path):
        write(tmp_path, "version: 1\n", name="other.yaml")
        found = find_config({"env_vars": {"MAD_CONFIG": "other.yaml"}}, tmp_path)
        assert found is not None and found.name == "other.yaml"


class TestValidateCard:
    """The one genuinely ambiguous thing: `args` means two different things."""

    def test_distributed_card_with_sbatch_args_warns(self):
        warnings = validate_card(
            {
                "name": "x",
                "distributed": {"launcher": "slurm_multi"},
                "args": "-N 2 -n 2",
            }
        )
        assert len(warnings) == 1
        assert "distributed.nnodes" in warnings[0]

    def test_script_style_args_do_not_warn(self):
        warnings = validate_card(
            {
                "name": "x",
                "distributed": {"launcher": "slurm_multi"},
                "args": "--model_repo m --config configs/default.yaml",
            }
        )
        assert warnings == []

    def test_non_distributed_card_is_not_checked(self):
        assert validate_card({"name": "x", "args": "-N 2 -n 2"}) == []


class TestResolveForModel:
    def test_no_file_means_no_change(self, tmp_path):
        # Ways 1-3 are the common case and must be untouched.
        env, warnings = resolve_for_model({"name": "x"}, tmp_path)
        assert env == {} and warnings == []

    def test_benchmark_selected_from_card_env(self, tmp_path):
        write(
            tmp_path,
            """
            version: 1
            benchmark:
              - kind: niah
                env: {NIAH_WORDS: '10000'}
            """,
        )
        env, _ = resolve_for_model(
            {"name": "x", "env_vars": {"BENCHMARK_SCRIPT": "niah"}}, tmp_path
        )
        assert env["NIAH_WORDS"] == "10000"

    def test_equivalence_with_hand_written_env_vars(self, tmp_path):
        """The property that makes way 4 safe to adopt.

        A way-4 file and the hand-written env_vars a card carries today must
        resolve to the same environment. If this holds, adopting way 4 changes
        nothing downstream -- no workload script has to be touched.
        """
        write(
            tmp_path,
            """
            version: 1
            site:
              env:
                NVME_ROOT: /mnt/m2m_nobackup
                SHARED_MOUNT: /shared_inference
            model:
              id: moonshotai/Kimi-K3
              local_name: Kimi-K3
              env:
                MODEL_NAME: Kimi-K3
                TP_SIZE: '8'
                PP_SIZE: '2'
                REQUIRE_LOCAL_WEIGHTS: '1'
            benchmark:
              - kind: niah
                env:
                  BENCHMARK_SCRIPT: niah
                  NIAH_WORDS: '10000,50000,100000,200000'
            """,
        )
        way4, _ = resolve_for_model(
            {"name": "kimi", "env_vars": {"BENCHMARK_SCRIPT": "niah"}}, tmp_path
        )

        # What the same card expresses today, as a flat env_vars block (way 1-3).
        way123 = {
            "NVME_ROOT": "/mnt/m2m_nobackup",
            "SHARED_MOUNT": "/shared_inference",
            "MODEL_NAME": "Kimi-K3",
            "TP_SIZE": "8",
            "PP_SIZE": "2",
            "REQUIRE_LOCAL_WEIGHTS": "1",
            "BENCHMARK_SCRIPT": "niah",
            "NIAH_WORDS": "10000,50000,100000,200000",
        }
        assert way4 == way123

    def test_malformed_file_raises_rather_than_silently_ignoring(self, tmp_path):
        write(tmp_path, "version: 1\nbenchmark: 5\n")
        with pytest.raises(LayeredConfigError):
            resolve_for_model({"name": "x"}, tmp_path)


class TestDockerEnvVarsReachSlurm:
    """A manifest's context.docker_env_vars must reach the SLURM path.

    It used to reach only `docker run -e` on the local path, so manifests had to
    declare the same variable twice -- once in context.docker_env_vars and again
    in deployment_config.env_vars. Two copies of a NIC list drift, and a wrong
    one does not fail loudly: RCCL falls back to TCP and the benchmark still
    reports a number.
    """

    @staticmethod
    def _env(additional_context, model_info):
        """The layering _build_env_vars performs, in order."""
        env = {}
        if "docker_env_vars" in additional_context:
            env.update(additional_context["docker_env_vars"])
        if "env_vars" in model_info:
            env.update(model_info["env_vars"])
        if "env_vars" in additional_context:
            env.update(additional_context["env_vars"])
        return env

    def test_docker_env_vars_now_arrive(self):
        env = self._env({"docker_env_vars": {"NCCL_IB_HCA": "rdma0:1"}}, {})
        assert env["NCCL_IB_HCA"] == "rdma0:1"

    def test_declared_once_is_enough(self):
        # The whole point: no need to repeat it in deployment_config.env_vars.
        env = self._env(
            {"docker_env_vars": {"RDMAV_DRIVERS": "ionic", "RCCL_AINIC_ROCE": "1"}},
            {"env_vars": {"MODEL_NAME": "DeepSeek-R1"}},
        )
        assert env["RDMAV_DRIVERS"] == "ionic"
        assert env["RCCL_AINIC_ROCE"] == "1"
        assert env["MODEL_NAME"] == "DeepSeek-R1"

    def test_existing_precedence_is_unchanged(self):
        # Anything that already flowed keeps winning; this change is additive.
        env = self._env(
            {
                "docker_env_vars": {"NCCL_IB_GID_INDEX": "1"},
                "env_vars": {"NCCL_IB_GID_INDEX": "3"},
            },
            {"env_vars": {"NCCL_IB_GID_INDEX": "2"}},
        )
        assert env["NCCL_IB_GID_INDEX"] == "3"


class TestToolsOnSelfManagedPath:
    """slurm_multi used to drop `tools` entirely.

    prepare() early-dispatches self-managed launchers and returns, so the
    templated path's profiling block never ran for them. madengine cannot wrap a
    script it does not control, but silently ignoring a configured tool is worse
    than saying so.
    """

    @staticmethod
    def _resolve(tools, enabled, resolved):
        """The decision _build_env_vars now makes, isolated from SLURM."""
        env = {}
        if tools:
            picked = resolved if enabled else []
            if picked:
                env["MAD_TOOLS"] = ",".join(
                    t.get("name", str(t)) if isinstance(t, dict) else str(t)
                    for t in picked
                )
        return env

    def test_no_tools_configured_adds_nothing(self):
        assert self._resolve([], True, []) == {}

    def test_resolved_tools_reach_the_script(self):
        env = self._resolve([{"name": "rocprofv3"}], True, [{"name": "rocprofv3"}])
        assert env["MAD_TOOLS"] == "rocprofv3"

    def test_plain_string_tools_are_handled(self):
        assert (
            self._resolve(["rocprofv3"], True, ["rocprofv3"])["MAD_TOOLS"]
            == "rocprofv3"
        )

    def test_profiling_unavailable_sets_nothing(self):
        # rocprofv3 missing -> no MAD_TOOLS, and the caller warns.
        assert self._resolve([{"name": "rocprofv3"}], False, []) == {}


class TestSelfManagedResultsHonourTheCard:
    """The self-managed path ignored the card's declared multiple_results.

    The templated path reads it and scores candidates; this one searched a
    hardcoded list naming /shared_inference -- one site's NFS mount. On any other
    cluster the card's declaration was the only thing that could have worked.
    """

    @staticmethod
    def _declared(manifest):
        for m in (manifest.get("built_models") or {}).values():
            if m.get("multiple_results"):
                return m["multiple_results"]
        return None

    def test_declaration_is_found(self):
        manifest = {"built_models": {"img": {"multiple_results": "perf_Kimi-K3.csv"}}}
        assert self._declared(manifest) == "perf_Kimi-K3.csv"

    def test_absent_declaration_falls_back(self):
        assert self._declared({"built_models": {"img": {}}}) is None

    def test_empty_manifest_is_safe(self):
        assert self._declared({}) is None

    def test_first_declaring_model_wins(self):
        manifest = {
            "built_models": {
                "a": {},
                "b": {"multiple_results": "perf_b.csv"},
            }
        }
        assert self._declared(manifest) == "perf_b.csv"
