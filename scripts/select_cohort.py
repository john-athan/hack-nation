"""T2 step 1: select the cohort (ADR-0005) and download its FASTAs.

Keeps every QC-passing resistant isolate + a lineage-capped susceptible sample — NO
row-budget subsample. Run: uv run python scripts/select_cohort.py [--jobs 12]
Writes data/cohort.csv and fills data/fasta. Idempotent: cached labels/metadata + FASTAs reused.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from genome_firewall import bvbrc
from genome_firewall.cohort import select_cohort
from genome_firewall.constants import COHORT_CSV, FASTA_DIR, LABELS_CSV
from genome_firewall.drugs import DRUG_DB
from genome_firewall.errors import BVBRCError, EmptyFastaError
from genome_firewall.labels import harmonize

# Metadata for ALL panel-labeled candidates (not just the well-covered ones — we must not
# lose a resistant isolate just because it has few labels).
_META_CACHE = LABELS_CSV.parent / "candidate_meta_full.csv"


def _candidate_metadata(clean: pd.DataFrame) -> pd.DataFrame:
    if _META_CACHE.exists():
        print(f"[meta] using cached {_META_CACHE}", file=sys.stderr)
        return pd.read_csv(_META_CACHE, dtype=str)
    panel = set(DRUG_DB)
    candidates = (
        clean[clean["antibiotic"].isin(panel)]["genome_id"]
        .astype(str)
        .unique()
        .tolist()
    )
    print(
        f"[meta] fetching metadata for {len(candidates)} candidates…", file=sys.stderr
    )
    meta = bvbrc.fetch_metadata(candidates)
    meta.to_csv(_META_CACHE, index=False)
    return meta


def _download(gid: str) -> tuple[str, bool, str]:
    dest = FASTA_DIR / f"{gid}.fna"
    if dest.exists() and dest.stat().st_size > 0:
        return gid, True, "cached"
    try:
        dest.write_text(bvbrc.fetch_fasta(gid))
        return gid, True, "ok"
    except (EmptyFastaError, BVBRCError) as exc:
        return gid, False, str(exc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()

    clean = harmonize(pd.read_csv(LABELS_CSV, dtype=str))
    meta = _candidate_metadata(clean)
    print(f"[meta] {len(meta)} genomes with metadata", file=sys.stderr)

    cohort = select_cohort(clean, meta)
    cohort.to_csv(COHORT_CSV, index=False)
    n_res = int(cohort["is_resistant"].sum())
    print(
        f"[cohort] {len(cohort)} genomes ({n_res} resistant, {len(cohort) - n_res} susceptible) "
        f"across {cohort['serovar'].nunique()} serovars",
        file=sys.stderr,
    )

    FASTA_DIR.mkdir(parents=True, exist_ok=True)
    gids = cohort["genome_id"].astype(str).tolist()
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_download, g): g for g in gids}
        for done, fut in enumerate(as_completed(futures), 1):
            gid, success, msg = fut.result()
            ok += success
            fail += not success
            if not success:
                print(f"[fasta {done}/{len(gids)}] {gid}: FAIL {msg}", file=sys.stderr)
            elif done % 500 == 0:
                print(f"[fasta] {done}/{len(gids)} done", file=sys.stderr)
    print(
        f"[done] {ok}/{len(gids)} FASTAs available ({fail} failed) in {FASTA_DIR}",
        file=sys.stderr,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
