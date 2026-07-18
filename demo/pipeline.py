"""Demo pipeline glue: an uploaded/selected FASTA → honest per-drug report.

Kept separate from the Streamlit UI so it is unit-testable and reusable. Caches AMRFinderPlus
output per genome so a repeated demo run is instant and deterministic (protecting the happy path).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from genome_firewall.amrfinder import parse_tsv, run_amrfinder
from genome_firewall.constants import AMRFINDER_DIR
from genome_firewall.report import build_report
from genome_firewall.schema import GenomeReport

_UPLOAD_DIGEST_LEN = 12


def upload_fasta_path(fasta_dir: Path, filename: str, content: bytes) -> Path:
    """Content-addressed on-disk path for an uploaded FASTA.

    `analyze_fasta` keys the AMRFinderPlus cache by file stem, so naming an upload by its
    filename alone lets two *different* genomes that happen to share a basename (one user and
    the next both uploading "genome.fasta") collide on that cache — the second would be
    served the first's TSV, surfacing genome A's determinants for genome B. That is a
    fabrication on the demo's live TRIGGER beat, so we fold a content hash into the stem: a
    cache hit is then provably the *same* bytes, while an identical re-upload still reuses the
    cache (fast + deterministic on stage). The human-readable basename is kept for the report.
    """
    digest = hashlib.sha256(content).hexdigest()[:_UPLOAD_DIGEST_LEN]
    safe_stem = Path(filename).stem or "genome"
    return fasta_dir / f"upload_{safe_stem}_{digest}.fna"


def analyze_fasta(
    fasta: Path, *, threads: int = 4, use_cache: bool = True
) -> tuple[GenomeReport, pd.DataFrame]:
    """Annotate a FASTA (cached) and build the honest per-drug report. Returns (report, determinants)."""
    genome_id = fasta.stem
    out_tsv = AMRFINDER_DIR / f"{genome_id}.tsv"
    if not (use_cache and out_tsv.exists() and out_tsv.stat().st_size > 0):
        run_amrfinder(fasta, out_tsv, threads=threads)
    determinants = parse_tsv(out_tsv)
    report = build_report(determinants, genome_id)
    return report, determinants
