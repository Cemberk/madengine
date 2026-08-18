"""Unit tests for reporting: update_perf_csv and PERF_CSV_HEADER."""

import os
import pytest
import tempfile

import json
import pandas as pd

from madengine.reporting.update_perf_csv import (
    PERF_CSV_HEADER,
    update_perf_csv,
)


class TestPerfCsvHeader:
    """PERF_CSV_HEADER constant and compatibility."""

    def test_perf_csv_header_contains_required_columns(self):
        """Header must contain model, status, and other required columns for perf table."""
        assert "model" in PERF_CSV_HEADER
        assert "status" in PERF_CSV_HEADER
        assert "performance" in PERF_CSV_HEADER
        assert "gpu_architecture" in PERF_CSV_HEADER

    def test_perf_csv_header_is_comma_separated(self):
        """Header is a single line of comma-separated column names."""
        parts = PERF_CSV_HEADER.split(",")
        assert len(parts) >= 20


class TestUpdatePerfCsvCreatesFileWhenMissing:
    """update_perf_csv creates perf CSV with header when file does not exist."""

    def test_exception_result_creates_perf_csv_if_missing(self):
        """When perf_csv does not exist and exception_result is provided, file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            perf_csv = os.path.join(tmpdir, "perf.csv")
            exception_json = os.path.join(tmpdir, "exception.json")
            # Minimal exception entry (status FAILURE for failed run)
            minimal = {
                "model": "test/model",
                "status": "FAILURE",
                "tags": "tag1",
                "performance": "",
                "metric": "",
            }
            with open(exception_json, "w") as f:
                json.dump(minimal, f)

            assert not os.path.exists(perf_csv)
            update_perf_csv(perf_csv, exception_result=exception_json)

            assert os.path.exists(perf_csv)
            df = pd.read_csv(perf_csv)
            assert "model" in df.columns
            assert "status" in df.columns
            assert len(df) == 1
            assert df.iloc[0]["model"] == "test/model"
            assert df.iloc[0]["status"] == "FAILURE"


class TestResolveResultStatus:
    """An explicitly-reported status beats deriving one from `performance`.

    Deriving status from performance treats any non-null number as SUCCESS. That
    holds for throughput, where a missing number is the only failure, but not for
    accuracy benchmarks where zero is a real measurement of a real failure.
    """

    def test_explicit_status_wins(self):
        from madengine.reporting.update_perf_csv import resolve_result_status
        assert resolve_result_status("FAILURE", 0) == "FAILURE"
        assert resolve_result_status("SUCCESS", None) == "SUCCESS"

    def test_explicit_status_is_normalised(self):
        from madengine.reporting.update_perf_csv import resolve_result_status
        assert resolve_result_status(" failure ", 0) == "FAILURE"

    @pytest.mark.parametrize("blank", [None, "", "   ", float("nan")])
    def test_blank_status_falls_back_to_derivation(self, blank):
        from madengine.reporting.update_perf_csv import resolve_result_status
        assert resolve_result_status(blank, 1.5) == "SUCCESS"
        assert resolve_result_status(blank, None) == "FAILURE"

    def test_derivation_unchanged_for_producers_without_status(self):
        """The 98 model cards on the templated path emit no status column."""
        from madengine.reporting.update_perf_csv import resolve_result_status
        assert resolve_result_status(None, 1234.5) == "SUCCESS"
        assert resolve_result_status(None, 0) == "SUCCESS"   # legacy behaviour
        assert resolve_result_status(None, float("nan")) == "FAILURE"


class TestMultipleResultsStatusPreservation:
    """End-to-end: a zero-score FAILURE row survives update_perf_csv."""

    def _run(self, tmp_path, csv_body):
        common = tmp_path / "common_info.json"
        common.write_text(json.dumps({"nnodes": "2", "n_gpus": "16",
                                      "launcher": "slurm_multi", "tags": ["pyt", "vllm"]}))
        results = tmp_path / "perf_WL.csv"
        results.write_text(csv_body)
        perf = tmp_path / "perf.csv"
        update_perf_csv(
            perf_csv=str(perf),
            multiple_results=str(results),
            common_info=str(common),
            model_name="wl",
        )
        return pd.read_csv(perf)

    def test_zero_score_failure_is_preserved(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        df = self._run(tmp_path,
            "model,performance,metric,status\n"
            "m,10,needles /10,SUCCESS\n"
            "m,0,needles /10,FAILURE\n")
        assert list(df["status"]) == ["SUCCESS", "FAILURE"], \
            "a 0-score row reported as FAILURE must not be recomputed to SUCCESS"

    def test_csv_without_status_column_derives_as_before(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        df = self._run(tmp_path,
            "model,performance,metric\n"
            "m,1234.5,tok/s\n")
        assert list(df["status"]) == ["SUCCESS"]

    def test_madengine_metadata_is_merged_in(self, tmp_path, monkeypatch):
        """The point of the narrow contract: madengine supplies what it knows."""
        monkeypatch.chdir(tmp_path)
        df = self._run(tmp_path, "model,performance,metric\nm,1.0,tok/s\n")
        assert df.iloc[0]["nnodes"] == 2
        assert df.iloc[0]["n_gpus"] == 16
        assert df.iloc[0]["launcher"] == "slurm_multi"
