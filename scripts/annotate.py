"""Run AMRFinderPlus over every FASTA in data/fasta → data/amrfinder_out/*.tsv.

Run: uv run python scripts/annotate.py [--jobs 4]
Skips genomes already annotated. Parallelized with a thread pool (each amrfinder
call is its own process; we just fan out the subprocess launches).
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from genome_firewall.amrfinder import run_amrfinder
from genome_firewall.cohort import stratified_order
from genome_firewall.constants import AMRFINDER_DIR, COHORT_CSV, FASTA_DIR
from genome_firewall.errors import AMRFinderError


def _serovar_by_genome() -> dict[str, str]:
    """genome_id → serovar from the cohort table (empty if it isn't built yet)."""
    if not COHORT_CSV.exists():
        return {}
    with COHORT_CSV.open(newline="") as fh:
        return {
            row["genome_id"]: (row.get("serovar") or "").strip()
            for row in csv.DictReader(fh)
        }


def _ordered_fastas() -> list[str]:
    """FASTA filenames in serovar-round-robin order so a partial annotate stays diverse.

    Falls back to a loud sorted() only if the cohort table is missing (never silent) —
    stratification needs the serovar map and cohort.csv is present in every real run.
    """
    names = [p.name for p in FASTA_DIR.glob("*.fna")]
    serovar = _serovar_by_genome()
    if not serovar:
        print(
            f"[annotate] {COHORT_CSV} absent → sorted (non-stratified) order",
            file=sys.stderr,
        )
        return sorted(names)
    by_stem = {Path(n).stem: n for n in names}
    order = stratified_order(list(by_stem), serovar)
    return [by_stem[stem] for stem in order]


def _annotate_one(fasta_name: str) -> tuple[str, bool, str]:
    fasta = FASTA_DIR / fasta_name
    out = AMRFINDER_DIR / f"{fasta.stem}.tsv"
    if out.exists() and out.stat().st_size > 0:
        return fasta.stem, True, "cached"
    try:
        run_amrfinder(fasta, out, threads=1)
        return fasta.stem, True, "ok"
    except AMRFinderError as exc:
        return fasta.stem, False, str(exc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    fastas = _ordered_fastas()
    if not fastas:
        print(
            f"[annotate] no FASTAs in {FASTA_DIR}; run pull_data.py first",
            file=sys.stderr,
        )
        return 1
    AMRFINDER_DIR.mkdir(parents=True, exist_ok=True)

    ok = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_annotate_one, f): f for f in fastas}
        for fut in as_completed(futures):
            stem, success, msg = fut.result()
            ok += success
            print(f"[annotate] {stem}: {msg}", file=sys.stderr)
    print(f"[done] {ok}/{len(fastas)} annotated in {AMRFINDER_DIR}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
