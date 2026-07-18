"""Demo doctor: a fast, READ-ONLY check that the demo is stage-ready. Exit 0 = green.

Run right before demoing (or in the driver): confirms the three artifacts load, the collapse money
slide is non-empty, and BOTH curated beats render the way the pitch narrates — in seconds, off the
CACHED annotations, mutating nothing. Unlike pick_demo_genomes.py (a slow ~300-genome *write* step
inside finalize) this never retrains, never rewrites a preset, and never boots Streamlit, so it is
safe to run at 8am before going on stage.

    uv run python scripts/preflight.py

Exits non-zero and prints ❌ with the reason on the first failed check, so it drops cleanly into a
pre-demo shell guard. Streamlit-free (imports only the pandas/genome_firewall demo glue).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))

import pandas as pd

from collapse import collapse_frame  # type: ignore[import-not-found]
from genome_firewall.constants import (
    AMRFINDER_DIR,
    CALL_RESISTANT,
    DATA_DIR,
    EVIDENCE_KNOWN_GENE,
    FASTA_DIR,
)
from genome_firewall.train import MODELS_PATH, load
from pipeline import analyze_fasta  # type: ignore[import-not-found]
from preflight import (  # type: ignore[import-not-found]
    Check,
    check_known_gene_beat,
    check_ood_beat,
    check_qrdr_beat,
)
from verdict import all_verdicts  # type: ignore[import-not-found]

_RESULTS_CSV = DATA_DIR / "results.csv"
_DEMO_GENOMES_JSON = DATA_DIR / "demo_genomes.json"
_MIN_DRUGS = 5  # the pitch shows ~6-10 drugs; fewer means a half-trained artifact


def _fail(name: str, detail: str) -> Check:
    return Check(name, False, detail)


def _check_artifacts() -> list[Check]:
    checks: list[Check] = []
    for name, path in (
        ("models.joblib present", MODELS_PATH),
        ("results.csv present", _RESULTS_CSV),
        ("demo_genomes.json present", _DEMO_GENOMES_JSON),
    ):
        checks.append(Check(name, path.exists(), str(path)))
    return checks


def _check_collapse() -> Check:
    if not _RESULTS_CSV.exists():
        return _fail("collapse money slide", "results.csv missing — slide is blank")
    ok = collapse_frame(pd.read_csv(_RESULTS_CSV))
    if ok is None:
        return _fail("collapse money slide", "no evaluable drugs — slide is blank")
    top = ok.iloc[0]
    return Check(
        "collapse money slide",
        True,
        f"{len(ok)} drugs; leads with {top['drug']} "
        f"{top['random_bal_acc']:.2f}→{top['grouped_bal_acc']:.2f}",
    )


def _cached(genome_id: str) -> bool:
    """Both the FASTA and its AMRFinder TSV must be cached so the beat runs instant + offline."""
    tsv = AMRFINDER_DIR / f"{genome_id}.tsv"
    fasta = FASTA_DIR / f"{genome_id}.fna"
    return fasta.exists() and tsv.exists() and tsv.stat().st_size > 0


def _drive_beat(models: dict, genome_id: str, beat: str) -> Check:
    if not _cached(genome_id):
        return _fail(
            f"{beat} genome {genome_id}",
            "FASTA/annotation not cached — beat would run live AMRFinder (slow, needs network)",
        )
    report, dets = analyze_fasta(FASTA_DIR / f"{genome_id}.fna")
    verdicts = all_verdicts(models, set(dets["symbol"].astype(str)), dets, report)
    if beat in ("known_gene", "qrdr"):
        resistant = {
            p.antibiotic
            for p in report.predictions
            if p.call == CALL_RESISTANT and p.evidence_category == EVIDENCE_KNOWN_GENE
        }
        if beat == "known_gene":
            return check_known_gene_beat(verdicts, resistant)
        return check_qrdr_beat(verdicts, resistant)
    return check_ood_beat(verdicts)


def _check_beats(models: dict) -> list[Check]:
    if not _DEMO_GENOMES_JSON.exists():
        return [
            _fail("curated beats", "demo_genomes.json missing — no curated presets")
        ]
    try:
        genomes = json.loads(_DEMO_GENOMES_JSON.read_text())["genomes"]
    except (json.JSONDecodeError, KeyError, OSError) as exc:
        return [_fail("curated beats", f"demo_genomes.json unreadable: {exc}")]
    checks = []
    for entry in genomes:
        beat = entry.get("beat", "?")
        # "known_gene" → beat ①; "ood" → beat ③; "qrdr" → beat ④. Marker "② collapse" has no genome.
        if beat in {"known_gene", "ood", "qrdr"}:
            checks.append(_drive_beat(models, entry["id"], beat))
    if not checks:
        checks.append(
            _fail("curated beats", "no known_gene/ood/qrdr beat in demo_genomes.json")
        )
    return checks


def main() -> int:
    checks = _check_artifacts()
    missing = [c for c in checks if not c.ok]
    if not missing:
        models = load(MODELS_PATH)
        if len(models) < _MIN_DRUGS:
            checks.append(
                _fail("models drug count", f"only {len(models)} drugs (<{_MIN_DRUGS})")
            )
        else:
            checks.append(
                Check("models drug count", True, f"{len(models)} drugs trained")
            )
        checks.append(_check_collapse())
        checks.extend(_check_beats(models))

    print("Genome Firewall — demo preflight\n")
    for c in checks:
        print(f"  {'✅' if c.ok else '❌'}  {c.name}: {c.detail}")
    failed = [c for c in checks if not c.ok]
    if failed:
        print(f"\n❌ NOT demo-ready — {len(failed)} check(s) failed.")
        return 1
    print(f"\n✅ Demo-ready — all {len(checks)} checks green.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
