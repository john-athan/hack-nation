"""run_amrfinder failure modes — the live-upload path must degrade, never hang the demo."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from genome_firewall import amrfinder
from genome_firewall.amrfinder import run_amrfinder
from genome_firewall.errors import AMRFinderError, GenomeFirewallError


class _FakeProc:
    def __init__(self, *, timeout: bool = False, returncode: int = 0) -> None:
        self.pid = 424242
        self._timeout = timeout
        self.returncode = returncode

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        if self._timeout:
            raise subprocess.TimeoutExpired(cmd="amrfinder", timeout=timeout or 0)
        return "", "boom" if self.returncode else ""

    def wait(self) -> int:
        return self.returncode


def _fasta(tmp_path: Path) -> Path:
    f = tmp_path / "g.fna"
    f.write_text(">c\nACGT\n")
    return f


def test_timeout_raises_and_kills_process_group(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    killed: list[int] = []
    monkeypatch.setattr(amrfinder.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(amrfinder.os, "killpg", lambda pgid, sig: killed.append(pgid))
    monkeypatch.setattr(
        amrfinder.subprocess, "Popen", lambda *a, **k: _FakeProc(timeout=True)
    )

    out = tmp_path / "g.tsv"
    with pytest.raises(AMRFinderError, match="timed out"):
        run_amrfinder(_fasta(tmp_path), out)
    assert killed == [424242]  # the whole tree was signalled, not just the child


def test_nonzero_returncode_raises(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        amrfinder.subprocess, "Popen", lambda *a, **k: _FakeProc(returncode=1)
    )
    with pytest.raises(AMRFinderError, match="failed"):
        run_amrfinder(_fasta(tmp_path), tmp_path / "g.tsv")


def test_missing_fasta_raises(tmp_path: Path) -> None:
    with pytest.raises(AMRFinderError, match="not found"):
        run_amrfinder(tmp_path / "nope.fna", tmp_path / "g.tsv")


def test_missing_micromamba_binary_degrades_to_typed_error(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    """A missing micromamba/`amr` env must surface as AMRFinderError, not a bare FileNotFoundError.

    On the LIVE upload path app.py only catches GenomeFirewallError; if Popen's raw
    FileNotFoundError escaped, a judge uploading their own genome would see a stage traceback
    instead of the clean "use a cached genome" fallback. This is the case AMRFinderError's own
    docstring names as the most common — so it must be the typed error the UI already handles.
    """

    def _boom(*_a, **_k):  # noqa: ANN002, ANN003
        raise FileNotFoundError(2, "No such file or directory", "micromamba")

    monkeypatch.setattr(amrfinder.subprocess, "Popen", _boom)
    with pytest.raises(AMRFinderError, match="unavailable") as exc_info:
        run_amrfinder(_fasta(tmp_path), tmp_path / "g.tsv")
    # app.py's guard is `except GenomeFirewallError`; the typed error MUST be catchable there.
    assert isinstance(exc_info.value, GenomeFirewallError)
