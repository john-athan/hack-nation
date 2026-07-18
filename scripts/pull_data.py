"""T1 data slice: pull lab labels, harmonize, sample ~30 genomes, download FASTAs.

Run: uv run python scripts/pull_data.py [--n 30]
Idempotent-ish: skips FASTA downloads that already exist on disk.
"""

from __future__ import annotations

import argparse
import io
import sys

import pandas as pd

from genome_firewall import bvbrc
from genome_firewall.constants import FASTA_DIR, LABELS_CLEAN_CSV, LABELS_CSV
from genome_firewall.drugs import DRUG_DB
from genome_firewall.errors import EmptyFastaError
from genome_firewall.labels import harmonize

# The one drug we take end-to-end in T1; we balance the sample on its phenotype.
_ANCHOR_DRUG = "ampicillin"


def _load_or_fetch_labels() -> pd.DataFrame:
    if LABELS_CSV.exists():
        print(f"[labels] using cached {LABELS_CSV}", file=sys.stderr)
        return pd.read_csv(LABELS_CSV, dtype=str)
    print(
        "[labels] fetching from BV-BRC (this pulls the full lab-measured table)…",
        file=sys.stderr,
    )
    csv_text = bvbrc.fetch_labels_csv()
    LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    LABELS_CSV.write_text(csv_text)
    return pd.read_csv(io.StringIO(csv_text), dtype=str)


def _select_genomes(clean: pd.DataFrame, n: int) -> list[str]:
    """Pick n genomes with an ampicillin label, balanced R/S, richest drug coverage."""
    panel = set(DRUG_DB)
    on_panel = clean[clean["antibiotic"].isin(panel)]
    coverage = on_panel.groupby("genome_id")["antibiotic"].nunique()

    anchor = clean[clean["antibiotic"] == _ANCHOR_DRUG].set_index("genome_id")["label"]
    ranked = coverage.to_frame("cov").join(anchor.rename("amp"), how="inner").dropna()
    ranked = ranked.sort_values("cov", ascending=False)

    half = n // 2
    chosen: list[str] = []
    for phenotype, k in (("Resistant", half), ("Susceptible", n - half)):
        gids = ranked[ranked["amp"] == phenotype].head(k).index.tolist()
        chosen.extend(gids)
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    raw = _load_or_fetch_labels()
    print(f"[labels] {len(raw):,} raw rows", file=sys.stderr)
    clean = harmonize(raw)
    print(f"[labels] {len(clean):,} clean genome×drug labels", file=sys.stderr)

    gids = _select_genomes(clean, args.n)
    print(
        f"[select] {len(gids)} genomes chosen (balanced on {_ANCHOR_DRUG})",
        file=sys.stderr,
    )

    clean[clean["genome_id"].isin(gids)].to_csv(LABELS_CLEAN_CSV, index=False)

    FASTA_DIR.mkdir(parents=True, exist_ok=True)
    ok = 0
    for gid in gids:
        dest = FASTA_DIR / f"{gid}.fna"
        if dest.exists() and dest.stat().st_size > 0:
            ok += 1
            continue
        try:
            dest.write_text(bvbrc.fetch_fasta(gid))
            ok += 1
            print(f"[fasta] {gid} ✓", file=sys.stderr)
        except EmptyFastaError as exc:
            print(f"[fasta] {gid} SKIP: {exc}", file=sys.stderr)
    print(f"[done] {ok}/{len(gids)} FASTAs in {FASTA_DIR}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
