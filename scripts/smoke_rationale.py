"""Smoke-test the REAL OpenAI constrained-rationale path on a cached demo genome.

Owner directive (INBOX 2026-07-18): wire + smoke-test the real path — verify a real
rationale renders for a demo genome and cites ONLY genes present in the payload. This
exercises the LLM branch (not the fallback) and fails loudly if:
  - no genome is annotated (nothing to explain),
  - OpenAI never actually answered (we'd silently be on the template), or
  - a rationale names a determinant outside its prediction's payload.

Run: `uv run python scripts/smoke_rationale.py [GENOME_ID]`
The driver sources ~/.hack.env; this also loads it directly so a bare run works.
"""

from __future__ import annotations

import os
import sys

from genome_firewall.amrfinder import parse_tsv
from genome_firewall.constants import AMRFINDER_DIR
from genome_firewall.env import load_hack_env
from genome_firewall.rationale import (
    _ENV_API_KEY,
    _mentions_foreign_gene,
    _template,
    explain,
)
from genome_firewall.report import build_report

_DEFAULT_GENOME = "1079901.3"  # cached: ampicillin + tetracycline R via known genes


def _pick_genome(requested: str | None) -> str:
    if requested:
        return requested
    tsvs = sorted(AMRFINDER_DIR.glob("*.tsv"))
    if not tsvs:
        raise SystemExit(
            "No annotated genomes in data/amrfinder_out — run the pipeline first."
        )
    # Prefer the documented demo genome if it is annotated; else the first available.
    stems = {p.stem for p in tsvs}
    return _DEFAULT_GENOME if _DEFAULT_GENOME in stems else tsvs[0].stem


def main() -> int:
    load_hack_env()
    if not (key := os.environ.get(_ENV_API_KEY)):
        raise SystemExit(
            f"{_ENV_API_KEY} not set (not in env or ~/.hack.env) — cannot smoke the real path."
        )
    print(f"key: {key[:6]}…{key[-4:]} (len {len(key)})")

    genome = _pick_genome(sys.argv[1] if len(sys.argv) > 1 else None)
    tsv = AMRFINDER_DIR / f"{genome}.tsv"
    if not (tsv.exists() and tsv.stat().st_size > 0):
        raise SystemExit(f"{tsv} missing/empty — pick an annotated genome.")
    report = build_report(parse_tsv(tsv), genome)
    print(f"genome {genome}: {len(report.predictions)} predictions\n")

    llm_hits = 0
    for p in report.predictions:
        text = explain(p, genome, use_llm=True)
        # Fabrication gate is the whole point — assert it holds on the live output.
        if _mentions_foreign_gene(text, set(p.supporting_genes)):
            raise SystemExit(
                f"FABRICATION: {p.antibiotic} rationale named a gene outside {p.supporting_genes}: {text!r}"
            )
        differed = text != _template(p)
        llm_hits += differed
        tag = "LLM" if differed else "template"
        genes = ", ".join(p.supporting_genes) or "—"
        print(f"[{tag}] {p.antibiotic} ({p.call}; genes: {genes})\n    {text}\n")

    if llm_hits == 0:
        raise SystemExit(
            "Every rationale equaled its template — the OpenAI path was NOT exercised (cost cap? empty responses?)."
        )
    print(
        f"OK: {llm_hits} rationale(s) came from OpenAI; no fabricated genes. Real path verified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
