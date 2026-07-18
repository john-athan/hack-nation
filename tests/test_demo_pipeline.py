"""Test the demo pipeline's content-addressed upload naming (the live-upload fabrication guard)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))

from pipeline import upload_fasta_path  # noqa: E402  (demo-local module)

_DIR = Path("/tmp/gf-fastas")


def test_same_content_same_name_is_stable() -> None:
    # Identical re-upload must reuse the cache → same path (fast, deterministic on stage).
    a = upload_fasta_path(_DIR, "genome.fasta", b">c1\nACGT\n")
    b = upload_fasta_path(_DIR, "genome.fasta", b">c1\nACGT\n")
    assert a == b


def test_different_content_same_name_never_collides() -> None:
    # THE fabrication guard: two different genomes sharing a basename must NOT share a cache
    # key, or the second upload is served the first's determinants.
    a = upload_fasta_path(_DIR, "genome.fasta", b">g_A\nACGT\n")
    b = upload_fasta_path(_DIR, "genome.fasta", b">g_B\nTTTT\n")
    assert a != b


def test_path_layout() -> None:
    p = upload_fasta_path(_DIR, "Sample_1.fna", b">x\nACGT\n")
    assert p.parent == _DIR
    assert p.stem.startswith("upload_Sample_1_")  # human-readable basename kept
    assert p.suffix == ".fna"


def test_empty_filename_is_handled() -> None:
    # A pathological empty upload name must still yield a valid, content-unique path.
    p = upload_fasta_path(_DIR, "", b">x\nACGT\n")
    assert p.stem.startswith("upload_genome_")
    assert p.parent == _DIR
