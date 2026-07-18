"""T3 honest-core run: cohort → MIC labels + determinants → dataset → random-vs-grouped report.

Run: uv run python scripts/run_evaluation.py
Writes data/results.csv (the per-drug comparison table) and prints the collapse summary.
Uses serovar×MLST as the coarse honest CV group (ADR-0005) — no all-vs-all Mash needed.
"""

from __future__ import annotations

import sys

import pandas as pd

from genome_firewall.amrfinder import parse_tsv
from genome_firewall.atomicio import atomic_write
from genome_firewall.constants import AMRFINDER_DIR, COHORT_CSV, DATA_DIR, LABELS_CSV
from genome_firewall.dataset import build_dataset
from genome_firewall.evaluate import evaluate_all
from genome_firewall.labels import canonical_drug
from genome_firewall.mic import rederive

_RESULTS_CSV = DATA_DIR / "results.csv"


def _load_determinants(genome_ids: set[str]) -> pd.DataFrame:
    frames = [
        parse_tsv(t)
        for t in sorted(AMRFINDER_DIR.glob("*.tsv"))
        if t.stem in genome_ids
    ]
    if not frames:
        print(
            "[eval] no annotated TSVs for cohort — run annotate.py first",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    cohort = pd.read_csv(COHORT_CSV, dtype=str)
    genome_ids = set(cohort["genome_id"].astype(str))
    print(f"[eval] cohort: {len(genome_ids)} genomes", file=sys.stderr)

    raw = pd.read_csv(LABELS_CSV, dtype=str)
    raw["antibiotic"] = raw["antibiotic"].map(canonical_drug)
    # ADR-0004: re-derive labels from raw MIC, not BV-BRC's mixed-era phenotype. Binary model
    # drops Intermediate (reported separately); R/S feed the classifier.
    mic_labels = rederive(raw)
    mic_labels = mic_labels[mic_labels["label"].isin({"Resistant", "Susceptible"})]

    determinants = _load_determinants(genome_ids)
    annotated = set(determinants["genome_id"].astype(str))
    cohort = cohort[cohort["genome_id"].astype(str).isin(annotated)]
    print(f"[eval] annotated & usable: {len(cohort)} genomes", file=sys.stderr)

    ds = build_dataset(cohort, determinants, mic_labels)
    print(
        f"[eval] {ds.x_mech.shape[1]} mechanism features, {ds.meta['cluster'].nunique()} groups",
        file=sys.stderr,
    )

    table = evaluate_all(ds, determinants=determinants)
    # Atomic: the demo's collapse slide reads this unguarded (pd.read_csv); a torn write from an
    # interrupted finalize must never reach it.
    atomic_write(_RESULTS_CSV, lambda tmp: table.to_csv(tmp, index=False))

    print(
        "\n=== PER-DRUG: random (dishonest) vs grouped (honest) AUROC ===",
        file=sys.stderr,
    )
    show = table[table["status"] == "ok"].copy()
    for _, r in show.iterrows():
        ra, ga = r.get("random_auroc"), r.get("grouped_auroc")
        drop = (ra - ga) if pd.notna(ra) and pd.notna(ga) else float("nan")
        flag = "" if r["therapeutic"] else " [marker-only]"
        print(
            f"  {r['drug']:28s}{flag:13s} random={ra} grouped={ga} Δ={drop:+.3f} "
            f"| cover={r.get('coverage')} no_call={r.get('frac_no_call')} (nR={r['n_resistant']})",
            file=sys.stderr,
        )
    skipped = table[table["status"] != "ok"]
    for _, r in skipped.iterrows():
        print(
            f"  {r['drug']:28s} SKIP: {r['status']} (nR={r['n_resistant']})",
            file=sys.stderr,
        )
    print(f"\n[done] wrote {_RESULTS_CSV}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
