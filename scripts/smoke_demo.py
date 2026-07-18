"""Repeat-determinism smoke test for the demo happy path — one command, RED on any drift.

The reliability mandate is "smoke-test the happy path repeatedly": a live stage demo must
render *byte-identically* every time a judge clicks Analyze, or the pitch's own honesty story
wobbles on screen. `preflight.py` proves ONE run of each beat has the right *properties*; this
proves REPEATED runs are *identical* across the whole rendered surface a judge actually sees —
the mechanism report, the naive-vs-firewall verdicts + knockout deltas, the untrained-drug
reconciliation, and the shared collapse money slide. That is exactly the output-drift regression
class hardening cycles kept catching by hand; here it is a durable, repeatable guard.

Like preflight it is Streamlit-free, runs off the CACHED annotations, retrains nothing, and
mutates nothing — safe to run at 8am before going on stage:

    uv run python scripts/smoke_demo.py

Beats come from data/demo_genomes.json (never hardcoded), so the smoke set self-updates with the
curated presets and can never drift from what the sidebar shows. Exits non-zero with the first
drifting surface named, so it drops cleanly into a pre-demo shell guard.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))

import pandas as pd

from collapse import (  # type: ignore[import-not-found]
    COLLAPSE_COL,
    collapse_frame,
    non_discriminating_drugs,
    sub_target_coverage_drugs,
    underpowered_drugs,
)
from genome_firewall.conformal import CONFORMAL_ALPHA
from genome_firewall.constants import AMRFINDER_DIR, DATA_DIR, FASTA_DIR
from genome_firewall.train import MODELS_PATH, load
from pipeline import analyze_fasta  # type: ignore[import-not-found]
from report_table import (  # type: ignore[import-not-found]
    report_table,
    untrained_reported_drugs,
)
from verdict import all_verdicts, format_delta, naive_confidence  # type: ignore[import-not-found]

_RESULTS_CSV = DATA_DIR / "results.csv"
_DEMO_GENOMES_JSON = DATA_DIR / "demo_genomes.json"
# How many times to render each surface. Small but >1 — a single extra render already exposes any
# ordering / float / cache-staleness nondeterminism; more just burns cache-warm CPU for no signal.
_REPEATS = 5
_COVERAGE_TARGET = 1 - CONFORMAL_ALPHA


def _cached(genome_id: str) -> bool:
    tsv = AMRFINDER_DIR / f"{genome_id}.tsv"
    fasta = FASTA_DIR / f"{genome_id}.fna"
    return fasta.exists() and tsv.exists() and tsv.stat().st_size > 0


def _beat_surface(models: dict, results: pd.DataFrame | None, genome_id: str) -> str:
    """Canonical string of the FULL per-genome demo surface — everything a judge sees for a beat.

    Deterministic by construction (cached TSV → fitted models → temperature-free formatting), so a
    hash mismatch across runs means a real regression, not sampling noise. The OpenAI rationale is
    deliberately excluded: it is a phrasing nicety with its own template fallback, not part of the
    deterministic core this guard protects.
    """
    report, dets = analyze_fasta(FASTA_DIR / f"{genome_id}.fna")
    verdicts = all_verdicts(models, set(dets["symbol"].astype(str)), dets, report)
    # Reproduce _firewall_section's row order + formatting so a change to either is caught here.
    rows = [
        (
            v.drug,
            f"{v.naive_call} ({naive_confidence(v):.0%})",
            v.firewall_verdict,
            v.firewall_holding,
            format_delta(v.knockout_delta),
            v.evidence,
            v.therapeutic,
        )
        for v in sorted(verdicts, key=lambda x: (not x.firewall_holding, x.drug))
    ]
    untrained = untrained_reported_drugs(report, [v.drug for v in verdicts], results)
    parts = [
        report_table(report).to_csv(index=False),
        json.dumps(rows),
        json.dumps(untrained),
    ]
    return "\n".join(parts)


def _collapse_surface(results: pd.DataFrame | None) -> str:
    """Canonical string of the shared collapse money slide (rendered once, below every beat)."""
    if results is None:
        return "NO_RESULTS"
    ok = collapse_frame(results)
    frame = "NONE" if ok is None else ok.to_csv(index=False)
    return "\n".join(
        [
            frame,
            f"col={COLLAPSE_COL}",
            json.dumps(non_discriminating_drugs(results)),
            json.dumps(underpowered_drugs(results)),
            json.dumps(sub_target_coverage_drugs(results, _COVERAGE_TARGET)),
        ]
    )


def _digests(render) -> tuple[str, bool]:  # noqa: ANN001
    """Render `_REPEATS` times; return (first digest, all-identical?)."""
    hashes = [
        hashlib.sha256(render().encode()).hexdigest()[:16] for _ in range(_REPEATS)
    ]
    return hashes[0], len(set(hashes)) == 1


def main() -> int:
    if not MODELS_PATH.exists():
        print(f"❌ {MODELS_PATH} missing — train models before smoke-testing.")
        return 1
    if not _DEMO_GENOMES_JSON.exists():
        print(f"❌ {_DEMO_GENOMES_JSON} missing — no curated beats to smoke-test.")
        return 1

    models = load(MODELS_PATH)
    results = pd.read_csv(_RESULTS_CSV) if _RESULTS_CSV.exists() else None
    try:
        entries = json.loads(_DEMO_GENOMES_JSON.read_text())["genomes"]
    except (json.JSONDecodeError, KeyError, OSError) as exc:
        print(f"❌ {_DEMO_GENOMES_JSON} unreadable: {exc}")
        return 1

    print(f"Genome Firewall — demo happy-path smoke ({_REPEATS}× each)\n")
    failed = False

    for entry in entries:
        # A "② collapse" marker beat has no genome; only per-genome beats render a surface.
        genome_id = entry.get("id")
        if not genome_id:
            continue
        beat = entry.get("beat", "?")
        if not _cached(genome_id):
            print(
                f"  ❌  {beat} {genome_id}: FASTA/annotation not cached — beat runs live"
            )
            failed = True
            continue
        digest, stable = _digests(
            lambda gid=genome_id: _beat_surface(models, results, gid)
        )
        mark = "✅" if stable else "❌"
        print(
            f"  {mark}  {beat} {genome_id}: {'stable' if stable else 'DRIFT'} ({digest})"
        )
        failed = failed or not stable

    digest, stable = _digests(lambda: _collapse_surface(results))
    mark = "✅" if stable else "❌"
    print(f"  {mark}  collapse slide: {'stable' if stable else 'DRIFT'} ({digest})")
    failed = failed or not stable

    if failed:
        print("\n❌ Demo path NOT reproducible — a surface drifts between renders.")
        return 1
    print(
        "\n✅ Demo happy path reproducible — every surface byte-identical across renders."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
