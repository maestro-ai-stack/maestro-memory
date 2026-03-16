"""Tests for CLI commands using typer.testing.CliRunner."""
from __future__ import annotations

import os

import pytest
from typer.testing import CliRunner

from maestro_memory.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _use_tmp_storage(tmp_path, monkeypatch):
    """Point maestro-memory storage to a temp dir for all CLI tests."""
    base = tmp_path / "memory"
    base.mkdir()
    monkeypatch.setattr("maestro_memory.core.config.BASE_DIR", base)
    monkeypatch.setattr("maestro_memory.core.config.CONFIG_PATH", base / "config.toml")
    # Also patch DEFAULT_CONFIG
    monkeypatch.setattr(
        "maestro_memory.core.config.DEFAULT_CONFIG",
        {
            "embedding": {"provider": "local", "model": "all-MiniLM-L6-v2"},
            "llm": {"provider": "none"},
            "storage": {"base_dir": str(base)},
        },
    )


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "maestro-memory" in result.stdout


def test_status() -> None:
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Facts" in result.stdout or "facts" in result.stdout.lower()


def test_add_and_search() -> None:
    add_result = runner.invoke(app, ["add", "User prefers snake_case"])
    assert add_result.exit_code == 0
    assert "stored" in add_result.stdout.lower() or "Episode" in add_result.stdout

    search_result = runner.invoke(app, ["search", "snake_case"])
    assert search_result.exit_code == 0
    assert "snake_case" in search_result.stdout


def test_config_show() -> None:
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "embedding" in result.stdout


def test_graph_list_entities() -> None:
    result = runner.invoke(app, ["graph", "--list-entities"])
    assert result.exit_code == 0
    # On empty DB it should say no entities
    assert "No entities" in result.stdout or result.stdout.strip() == ""
