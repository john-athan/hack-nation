"""T1 end-to-end demo: annotated genome → honest per-drug report (JSON + table).

Run: uv run python scripts/report.py [--genome GID] [--drug ampicillin]
With no --genome, reports the first annotated genome. With --drug, filters to one drug
(the T1 acceptance path). Cross-checks each call against the lab label when available.
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from genome_firewall.amrfinder import parse_tsv
from genome_firewall.constants import AMRFINDER_DIR, LABELS_CLEAN_CSV
from genome_firewall.labels import canonical_drug
from genome_firewall.report import build_report


def _load_determinants() -> pd.DataFrame:
    tsvs = sorted(AMRFINDER_DIR.glob("*.tsv"))
    if not tsvs:
        print(
            f"[report] no TSVs in {AMRFINDER_DIR}; run annotate.py first",
            file=sys.stderr,
        )
        raise SystemExit(1)
    frames = [parse_tsv(t) for t in tsvs]
    return pd.concat(frames, ignore_index=True)


def _labels_for(genome_id: str) -> dict[str, str]:
    if not LABELS_CLEAN_CSV.exists():
        return {}
    df = pd.read_csv(LABELS_CLEAN_CSV, dtype=str)
    sub = df[df["genome_id"].astype(str) == str(genome_id)]
    return dict(zip(sub["antibiotic"], sub["label"], strict=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", default=None)
    ap.add_argument("--drug", default=None)
    args = ap.parse_args()

    rows = _load_determinants()
    genomes = sorted(rows["genome_id"].astype(str).unique()) or []
    if not genomes:
        # No determinants at all still lets us report (everything → no-call).
        genomes = [t.stem for t in sorted(AMRFINDER_DIR.glob("*.tsv"))]
    gid = args.genome or genomes[0]

    # Canonicalize so "--drug Ampicillin" or a synonym maps to the panel key, not a silent miss.
    drugs = [canonical_drug(args.drug)] if args.drug else None
    report = build_report(rows, gid, drugs=drugs)
    labels = _labels_for(gid)

    print(json.dumps(report.to_dict(), indent=2))
    print("\n--- verdict vs lab label ---", file=sys.stderr)
    for p in report.predictions:
        lab = labels.get(p.antibiotic, "—")
        genes = ",".join(p.supporting_genes) or "—"
        print(
            f"  {p.antibiotic:28s} {p.call:11s} conf={p.confidence:.2f} "
            f"[{p.evidence_category:15s}] genes={genes:20s} lab={lab}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
