"""Tests for local data provider."""

from __future__ import annotations

import pytest

from madengine.data.local import LocalDataProvider


class TestLocalDataProvider:
    def test_name(self):
        provider = LocalDataProvider({"base_path": "/tmp"})
        assert provider.name == "local"

    def test_stage_missing_dataset(self, tmp_path):
        provider = LocalDataProvider({"base_path": str(tmp_path)})
        result = provider.stage("nonexistent")
        assert not result.success
        assert "not found" in result.error

    def test_stage_existing_file(self, tmp_path):
        data_file = tmp_path / "dataset.txt"
        data_file.write_text("hello world")

        provider = LocalDataProvider({"base_path": str(tmp_path)})
        result = provider.stage("dataset.txt")

        assert result.success
        assert result.local_path == str(data_file)
        assert result.size_bytes == 11

    def test_stage_existing_directory(self, tmp_path):
        data_dir = tmp_path / "my_dataset"
        data_dir.mkdir()
        (data_dir / "file1.txt").write_text("aaa")
        (data_dir / "file2.txt").write_text("bbb")

        provider = LocalDataProvider({"base_path": str(tmp_path)})
        result = provider.stage("my_dataset")

        assert result.success
        assert result.size_bytes == 6

    def test_exists(self, tmp_path):
        (tmp_path / "present.txt").write_text("ok")
        provider = LocalDataProvider({"base_path": str(tmp_path)})

        assert provider.exists("present.txt") is True
        assert provider.exists("absent.txt") is False
