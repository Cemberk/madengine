"""Tests for CLI entry point."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from madengine.cli import cli


class TestCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "madengine" in result.output

    def test_run_requires_recipe_or_config(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run"])
        assert result.exit_code != 0
        assert "Either --recipe or --config" in result.output

    def test_recipes_list(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["recipes", "list"])
        assert result.exit_code == 0

    def test_recipes_show_missing(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["recipes", "show", "nonexistent"])
        assert result.exit_code != 0

    def test_validate_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0
        assert "Validate" in result.output

    def test_db_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["db", "--help"])
        assert result.exit_code == 0
        assert "Database" in result.output

    def test_report_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--help"])
        assert result.exit_code == 0
