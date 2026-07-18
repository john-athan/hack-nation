"""Pre-bake the curated demo beats' OpenAI rationales to data/rationale_cache.json.

WHY: the live OpenAI call is the one un-cached external dependency left on the CURATED demo
path. On flaky venue wifi each resistance row waits up to the 12s timeout before falling back
to a template, so a beat with several resistance rows can stall ~1–2 min BEFORE the firewall
money slide renders. Baking the phrasings to disk makes the curated path instant, deterministic,
and OFFLINE — while keeping the "phrased by OpenAI" caption truthful (these WERE phrased by
OpenAI, here, once). At demo time `rationale.explain()` serves the disk hit before any network
call; a cache miss falls through to live, so this only ever removes latency.

Run once with a live key (needs OPENAI_API_KEY): `uv run python scripts/bake_rationales.py`.
Re-run after a finalize that moves a curated genome's calls. The output
(data/rationale_cache.json) IS committed — a fresh clone serves these real, gene-checked
gpt-5.4-mini phrasings offline with no key, so the hosted demo needs neither a key nor a
per-visit call. Re-baking needs a key; serving the baked cache does not.

Fails LOUDLY rather than baking a fallback template: a template masquerading in the cache would
defeat the whole point (it's not an OpenAI phrasing) and hide a real breakage, so any row that
can't get a genuine, fabrication-clean live answer aborts the bake.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))

from genome_firewall.amrfinder import parse_tsv
from genome_firewall.constants import AMRFINDER_DIR, DATA_DIR
from genome_firewall.env import load_hack_env
from genome_firewall.rationale import (
    _ENV_API_KEY,
    _llm_rationale,
    _llm_worthy,
    _mentions_foreign_gene,
    _template,
    make_disk_record,
    write_disk_cache,
)
from genome_firewall.report import build_report
from presets import supported_presets  # type: ignore[import-not-found]

_DEMO_GENOMES_JSON = DATA_DIR / "demo_genomes.json"


def _curated_genome_ids() -> list[str]:
    """The renderable curated beats' genome ids, from the file the demo/preflight/smoke read.

    Filtered through the same allowlist the app uses (`supported_presets`) so we never bake a
    stale/reverted beat's genome that the demo won't actually surface (cycle-41/42 doctrine)."""
    raw = json.loads(_DEMO_GENOMES_JSON.read_text(encoding="utf-8"))
    ids = [g["id"] for g in supported_presets(raw["genomes"])]
    if not ids:
        raise SystemExit(
            f"{_DEMO_GENOMES_JSON} lists no renderable curated genomes — nothing to bake."
        )
    return ids


def main() -> int:
    load_hack_env()
    if not os.environ.get(_ENV_API_KEY):
        raise SystemExit(
            f"{_ENV_API_KEY} not set (not in env or ~/.hack.env) — cannot bake real phrasings."
        )

    records: list[dict[str, object]] = []
    for genome in _curated_genome_ids():
        tsv = AMRFINDER_DIR / f"{genome}.tsv"
        if not (tsv.exists() and tsv.stat().st_size > 0):
            raise SystemExit(f"{tsv} missing/empty — annotate {genome} before baking.")
        report = build_report(parse_tsv(tsv), genome)
        worthy = [p for p in report.predictions if _llm_worthy(p)]
        print(f"genome {genome}: {len(worthy)} worthy row(s) to bake")
        for p in worthy:
            text = _llm_rationale(p)  # raw live call, bypassing every cache
            if text is None:
                raise SystemExit(
                    f"OpenAI returned no phrasing for {genome}/{p.antibiotic} "
                    f"(cost cap or empty response) — refusing to bake a template."
                )
            if text == _template(p):
                raise SystemExit(
                    f"{genome}/{p.antibiotic} phrasing equals its template — the live path "
                    f"was NOT exercised; refusing to bake a non-OpenAI answer."
                )
            if _mentions_foreign_gene(text, set(p.supporting_genes)):
                raise SystemExit(
                    f"FABRICATION: {genome}/{p.antibiotic} named a gene outside "
                    f"{p.supporting_genes}: {text!r} — not baking."
                )
            records.append(make_disk_record(p, genome, text))
            print(f"  [baked] {p.antibiotic} ({p.call}): {text}")

    path = write_disk_cache(records)
    print(f"\nOK: baked {len(records)} rationale(s) to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
