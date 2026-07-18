"""Train + persist per-drug models for the demo. Run after annotate + run_evaluation.

Run: uv run python scripts/train_models.py
Reads the cohort + MIC labels + determinants, builds the dataset, trains every eligible drug,
and saves to data/models.joblib for the Streamlit demo to load.
"""

from __future__ import annotations

import sys

import pandas as pd

from genome_firewall.amrfinder import parse_tsv
from genome_firewall.constants import AMRFINDER_DIR, COHORT_CSV, LABELS_CSV
from genome_firewall.dataset import build_dataset
from genome_firewall.labels import canonical_drug
from genome_firewall.mic import rederive
from genome_firewall.train import MODELS_PATH, save, train_all


def main() -> int:
    cohort = pd.read_csv(COHORT_CSV, dtype=str)
    genome_ids = set(cohort["genome_id"].astype(str))
    dets = pd.concat(
        [
            parse_tsv(t)
            for t in sorted(AMRFINDER_DIR.glob("*.tsv"))
            if t.stem in genome_ids
        ],
        ignore_index=True,
    )
    annotated = set(dets["genome_id"].astype(str))
    cohort = cohort[cohort["genome_id"].astype(str).isin(annotated)]

    raw = pd.read_csv(LABELS_CSV, dtype=str)
    raw["antibiotic"] = raw["antibiotic"].map(canonical_drug)
    labels = rederive(raw)
    labels = labels[labels["label"].isin({"Resistant", "Susceptible"})]

    ds = build_dataset(cohort, dets, labels)
    models = train_all(ds)
    save(models)
    print(f"[train] {len(models)} drug models → {MODELS_PATH}", file=sys.stderr)
    for drug, td in models.items():
        print(
            f"  {drug:28s} n={td.n} nR={td.n_resistant} feats={len(td.feature_columns)}",
            file=sys.stderr,
        )
    return 0 if models else 1


if __name__ == "__main__":
    raise SystemExit(main())
