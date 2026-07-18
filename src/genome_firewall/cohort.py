"""Cohort selection for the scaled feature build (T2) — ADR-0005 policy.

The honesty story lives on the split, and the split lives on NOT throwing away signal.
So the rule is: keep EVERY resistant isolate (rare positives drive per-class recall,
PR-AUC and calibration — subsampling them silently wrecks the scarce-R drugs), and only
trim the redundant SUSCEPTIBLE majority, capped per lineage group (serovar×MLST as the
coarse-cluster proxy). Lineage leakage is a job for the grouped CV split, not for dropping
rows. Fully deterministic — no RNG (determinism-first).
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

from .constants import (
    COHORT_SUS_CAP_PER_GROUP,
    GENOME_LEN_MAX,
    GENOME_LEN_MIN,
    GENOME_MAX_CONTIGS,
    PHENOTYPE_RESISTANT,
)

_UNKNOWN = "__unknown__"


def stratified_order(
    ids: Sequence[str], group_of: Mapping[str, str], *, fallback: str = _UNKNOWN
) -> list[str]:
    """Interleave `ids` round-robin across their group so ANY prefix is group-representative.

    WHY: annotation is a multi-hour job that can be cut short (dead cycle, slow box). The
    demo's collapse figure is only honest on a phylogenetically DIVERSE slice, but sorted
    genome-id order clusters by serovar — so a partial cohort over-represents whichever
    serovar sorts first (observed: Kentucky at 160/311 while Typhimurium 793 sat untouched)
    and mutes the collapse. Round-robin guarantees breadth as annotation progresses: rare
    serovars are exhausted early, dominant ones keep contributing, every prefix stays broad.
    Fully deterministic (sorted buckets, sorted group keys) — no RNG, matches this module's
    determinism-first doctrine.
    """
    buckets: dict[str, deque[str]] = defaultdict(deque)
    for item in sorted(ids):
        buckets[group_of.get(item, fallback)].append(item)

    keys = sorted(buckets)
    order: list[str] = []
    while any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k]:
                order.append(buckets[k].popleft())
    return order


def _per_genome(labels: pd.DataFrame, panel: set[str]) -> pd.DataFrame:
    """Per-genome coverage + resistant flag/count over the panel drugs."""
    on_panel = labels[labels["antibiotic"].isin(panel)]
    grp = on_panel.groupby("genome_id")
    return pd.DataFrame(
        {
            "cov": grp["antibiotic"].nunique(),
            "n_resistant": grp["label"].apply(
                lambda s: int((s == PHENOTYPE_RESISTANT).sum())
            ),
        }
    )


def _qc_pass(meta: pd.DataFrame) -> pd.DataFrame:
    """Keep only well-formed assemblies (plausible Salmonella length, not too fragmented)."""
    m = meta.copy()
    length = pd.to_numeric(m["genome_length"], errors="coerce")
    contigs = pd.to_numeric(m["contigs"], errors="coerce")
    ok = (
        length.between(GENOME_LEN_MIN, GENOME_LEN_MAX)
        & (contigs <= GENOME_MAX_CONTIGS)
        & contigs.notna()
    )
    return m[ok]


def select_cohort(
    labels: pd.DataFrame,
    meta: pd.DataFrame,
    sus_cap_per_group: int = COHORT_SUS_CAP_PER_GROUP,
) -> pd.DataFrame:
    """Keep every QC-passing resistant isolate + a capped susceptible sample per lineage.

    `labels` is the harmonized frame (genome_id, antibiotic, label); `meta` is BV-BRC
    genome metadata (genome_id, serovar, mlst, genome_length, contigs). Returns the chosen
    rows with cov/serovar/mlst/n_resistant/is_resistant for downstream steps to reuse.
    """
    panel = set(labels["antibiotic"].unique())
    per = _per_genome(labels, panel)
    qc = _qc_pass(meta).set_index("genome_id")

    joined = qc.join(per, how="inner")
    joined = joined[joined["cov"].notna()].copy()
    joined["serovar"] = joined["serovar"].fillna(_UNKNOWN).replace("", _UNKNOWN)
    joined["mlst"] = joined["mlst"].fillna(_UNKNOWN).replace("", _UNKNOWN)
    joined["is_resistant"] = joined["n_resistant"] > 0
    joined = joined.reset_index()

    # KEEP ALL resistant isolates — never capped.
    resistant = joined[joined["is_resistant"]]

    # Susceptible-only: cap per (serovar, mlst) group, richest-coverage first (deterministic).
    susceptible = joined[~joined["is_resistant"]].sort_values(
        ["serovar", "mlst", "cov", "genome_id"], ascending=[True, True, False, True]
    )
    capped_sus = susceptible.groupby(["serovar", "mlst"], sort=False).head(
        sus_cap_per_group
    )

    cohort = pd.concat([resistant, capped_sus], ignore_index=True)
    return cohort.sort_values("genome_id").reset_index(drop=True)
