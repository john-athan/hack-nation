"""Smoke-test the REAL live-annotate beat: FASTA → live AMRFinderPlus → honest report.

This is the demo's TRIGGER beat and its central external dependency (a micromamba `amr`
env + the pinned AMRFinderPlus DB). The cached happy path is covered by `preflight.py`,
but preflight reads CACHED TSVs by design — so it can never catch a broken live env/DB.
This drives the exact code path a judge's upload hits (`analyze_fasta(..., use_cache=False)`
forces a real annotation instead of the cache short-circuit) and fails loudly if:
  - the micromamba env / amrfinder binary / DB is missing or broken (would hang the stage),
  - the live annotation yields NO determinants on a genome known to carry blaSHV-2A (a
    silently mis-installed DB looks exactly like a clean, susceptible genome), or
  - a prediction's supporting genes are not grounded in the determinants AMRFinderPlus
    actually reported live (the beat-① narration would cite a gene the run didn't find).

On-demand stage-day smoke — NOT part of the pytest gate: it shells out to AMRFinderPlus
(~11s, competes with the annotation daemon for cores). Mirrors `scripts/smoke_rationale.py`.

Run: `uv run python scripts/smoke_annotate.py [GENOME_ID]`
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from genome_firewall.constants import (
    AMRFINDER_TIMEOUT_S,
    CALL_RESISTANT,
    EVIDENCE_KNOWN_GENE,
    FASTA_DIR,
)

# demo/ is a sibling package (pipeline glue lives with the UI); make it importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))

from pipeline import analyze_fasta  # noqa: E402  (after sys.path tweak)

# Cached ESBL carrier: blaSHV-2A + blaTEM-1 + tet(A). Beat-① narrates this genome, so a
# live run MUST reproduce its known-gene resistant calls — that is the assertion below.
_DEFAULT_GENOME = "1079901.3"


def _pick_genome(requested: str | None) -> str:
    if requested:
        return requested
    default = FASTA_DIR / f"{_DEFAULT_GENOME}.fna"
    if default.exists():
        return _DEFAULT_GENOME
    fastas = sorted(FASTA_DIR.glob("*.fna"))
    if not fastas:
        raise SystemExit(f"No FASTAs in {FASTA_DIR} — run scripts/pull_data.py first.")
    return fastas[0].stem


def main() -> int:
    genome = _pick_genome(sys.argv[1] if len(sys.argv) > 1 else None)
    fasta = FASTA_DIR / f"{genome}.fna"
    if not (fasta.exists() and fasta.stat().st_size > 0):
        raise SystemExit(
            f"{fasta} missing/empty — pick a genome with a downloaded FASTA."
        )

    print(f"genome {genome}: driving LIVE AMRFinderPlus (use_cache=False)…")
    t0 = time.time()
    # use_cache=False forces run_amrfinder — the exact live path an uploaded FASTA takes.
    # run_amrfinder is bounded (AMRFINDER_TIMEOUT_S) and raises on timeout/nonzero rc, so a
    # broken env surfaces as a loud AMRFinderError here, never a silent hang.
    report, determinants = analyze_fasta(fasta, threads=4, use_cache=False)
    dt = time.time() - t0
    print(f"live annotate + report in {dt:.1f}s (ceiling {AMRFINDER_TIMEOUT_S}s)")

    if determinants.empty:
        raise SystemExit(
            f"{genome} annotated to ZERO determinants — {_DEFAULT_GENOME} carries blaSHV-2A/"
            "blaTEM-1/tet(A), so an empty result means a broken AMRFinderPlus DB or env, not a "
            "clean genome. The beat-① known-gene story would silently vanish."
        )
    found_symbols = set(determinants["symbol"])
    print(f"determinants ({len(determinants)}): {sorted(found_symbols)}")

    # No fabrication: every supporting gene a prediction cites must be a determinant the live
    # run actually reported (same grounding rule the rationale smoke enforces on LLM text).
    for p in report.predictions:
        stray = set(p.supporting_genes) - found_symbols
        if stray:
            raise SystemExit(
                f"UNGROUNDED: {p.antibiotic} cites {sorted(stray)} not in the live "
                f"determinants {sorted(found_symbols)}."
            )

    known_gene_hits = [
        p
        for p in report.predictions
        if p.call == CALL_RESISTANT and p.evidence_category == EVIDENCE_KNOWN_GENE
    ]
    for p in known_gene_hits:
        genes = ", ".join(p.supporting_genes) or "—"
        print(f"[known_gene R] {p.antibiotic}  ({genes})")

    if not known_gene_hits:
        raise SystemExit(
            f"{genome} produced NO known-gene resistant call — beat-① ('resistant via a "
            "characterized gene') would not render. Live env/DB regression."
        )

    print(
        f"\nOK: live AMRFinderPlus path verified — {len(determinants)} determinants, "
        f"{len(known_gene_hits)} known-gene resistant call(s), all supporting genes grounded."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
