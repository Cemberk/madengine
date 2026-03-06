"""Tests for credential manager."""

from __future__ import annotations

import os

import pytest

from madengine.credentials.manager import CredentialManager, Credentials


class TestCredentialManager:
    def test_load_git_token_from_env(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token-123")
        manager = CredentialManager({"git": {"token_env": "GITHUB_TOKEN"}})
        creds = manager.load()
        assert creds.git_token == "test-token-123"

    def test_load_git_token_from_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        token_file = tmp_path / "token.txt"
        token_file.write_text("file-token-456\n")

        manager = CredentialManager({"git": {"token_file": str(token_file)}})
        creds = manager.load()
        assert creds.git_token == "file-token-456"

    def test_load_returns_none_when_no_token(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        manager = CredentialManager({})
        creds = manager.load()
        assert creds.git_token is None

    def test_setup_environment_sets_vars(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "env-token")
        manager = CredentialManager({"git": {"token_env": "GITHUB_TOKEN"}})
        manager.setup_environment()
        assert os.environ.get("GIT_TERMINAL_PROMPT") == "0"

    def test_credentials_cached(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "cached-token")
        manager = CredentialManager({"git": {"token_env": "GITHUB_TOKEN"}})
        creds1 = manager.load()
        creds2 = manager.load()
        assert creds1 is creds2

    def test_empty_config(self):
        manager = CredentialManager({})
        creds = manager.load()
        assert isinstance(creds, Credentials)
