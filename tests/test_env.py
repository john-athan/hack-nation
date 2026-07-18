"""Hermetic tests for the ~/.hack.env loader. No real home file is touched."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from genome_firewall.env import load_hack_env


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / ".hack.env"
    path.write_text(body, encoding="utf-8")
    return path


def test_loads_key_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GF_SMOKE_KEY", raising=False)
    load_hack_env(_write(tmp_path, "GF_SMOKE_KEY=sk-abc123\n"))
    assert os.environ["GF_SMOKE_KEY"] == "sk-abc123"


def test_existing_env_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GF_SMOKE_KEY", "already-set")
    load_hack_env(_write(tmp_path, "GF_SMOKE_KEY=from-file\n"))
    assert os.environ["GF_SMOKE_KEY"] == "already-set"


def test_ignores_comments_blanks_and_quotes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GF_QUOTED", raising=False)
    body = '# a comment\n\n  \nGF_QUOTED="quoted value"\nNOEQUALS\n'
    load_hack_env(_write(tmp_path, body))
    assert os.environ["GF_QUOTED"] == "quoted value"
    assert "NOEQUALS" not in os.environ


def test_missing_file_is_noop(tmp_path: Path) -> None:
    load_hack_env(tmp_path / "does-not-exist.env")  # must not raise
