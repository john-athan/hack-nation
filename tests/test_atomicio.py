"""Tests for crash-safe artifact writes (the unattended-finalize torn-write guard)."""

from __future__ import annotations

from pathlib import Path

import pytest

from genome_firewall.atomicio import atomic_write


def test_writes_content_and_leaves_no_temp(tmp_path: Path) -> None:
    target = tmp_path / "artifact.txt"
    atomic_write(target, lambda p: p.write_text("payload"))
    assert target.read_text() == "payload"
    # No stray *.tmp.* sibling left behind.
    assert list(tmp_path.glob("artifact.txt.tmp.*")) == []


def test_creates_missing_parent(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir" / "artifact.txt"
    atomic_write(target, lambda p: p.write_text("x"))
    assert target.read_text() == "x"


def test_failed_write_preserves_prior_and_cleans_temp(tmp_path: Path) -> None:
    target = tmp_path / "artifact.txt"
    target.write_text("original")

    def boom(p: Path) -> None:
        p.write_text("half-written")  # partial output lands in the TEMP file...
        raise RuntimeError("killed mid-write")

    with pytest.raises(RuntimeError, match="killed mid-write"):
        atomic_write(target, boom)

    # ...so the caller ever sees the whole prior artifact, never the torn one, and no temp survives.
    assert target.read_text() == "original"
    assert list(tmp_path.glob("artifact.txt.tmp.*")) == []
