"""Assemble the modeling table: two feature blocks + labels + grouping, for the cohort.

This is the integration hub every T3 step reads. Two blocks stay physically separate
(USP): the MECHANISM block (AMRFinderPlus determinants) is what the knockout probe zeroes;
the LINEAGE block (serovar/MLST one-hot) is the lineage signal the honest split must not
leak and that shift-weighting reads. Labels are {1=Resistant, 0=Susceptible, NaN=untested}.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .constants import (
    LABEL_NEG,
    LABEL_POS,
    LINEAGE_PREFIX,
    MECH_PREFIX,
    NO_CLUSTER,
    PHENOTYPE_RESISTANT,
    PHENOTYPE_SUSCEPTIBLE,
)
from .drugs import DRUG_DB
from .features import build_matrix

_LABEL_ENCODE = {PHENOTYPE_RESISTANT: LABEL_POS, PHENOTYPE_SUSCEPTIBLE: LABEL_NEG}


@dataclass(frozen=True, slots=True)
class Dataset:
    """Cohort modeling table. All frames share the same genome_id index/order."""

    genome_ids: list[str]
    x_mech: pd.DataFrame  # binary determinant matrix (mech__* columns)
    x_lineage: pd.DataFrame  # serovar/MLST one-hot (lin__* columns)
    y: pd.DataFrame  # genome × drug, values {1.0, 0.0, NaN}
    meta: pd.DataFrame  # serovar, mlst, cluster per genome

    def features(self, block: str = "both") -> pd.DataFrame:
        """Feature matrix for modeling: 'mech', 'lineage', or 'both' (concatenated)."""
        if block == "mech":
            return self.x_mech
        if block == "lineage":
            return self.x_lineage
        return pd.concat([self.x_mech, self.x_lineage], axis=1)

    def drug_xy(
        self, drug: str, block: str = "mech"
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """(X, y, groups) for one drug with untested genomes dropped. groups = cluster id."""
        if drug not in self.y.columns:
            raise KeyError(f"no labels for drug {drug!r}")
        labeled = self.y[drug].dropna()
        idx = labeled.index
        x = self.features(block).loc[idx]
        groups = self.meta.loc[idx, "cluster"]
        return x, labeled.astype(int), groups


def _mechanism_block(determinants: pd.DataFrame, genome_ids: list[str]) -> pd.DataFrame:
    mat = build_matrix(determinants)
    # Reindex to the full cohort: a genome with no determinants is all-zeros, not missing.
    mat = mat.reindex(genome_ids, fill_value=0).astype("int8")
    mat.columns = [f"{MECH_PREFIX}{c}" for c in mat.columns]
    return mat


def _lineage_block(meta: pd.DataFrame) -> pd.DataFrame:
    # One-hot serovar + MLST. These encode lineage explicitly so the honest split can be
    # judged against them and the knockout probe can weigh mechanism vs lineage.
    cols = [c for c in ("serovar", "mlst") if c in meta.columns]
    dummies = pd.get_dummies(meta[cols].fillna("unknown"), prefix=cols).astype("int8")
    dummies.columns = [f"{LINEAGE_PREFIX}{c}" for c in dummies.columns]
    return dummies


def _label_block(clean_labels: pd.DataFrame, genome_ids: list[str]) -> pd.DataFrame:
    panel = list(DRUG_DB)
    lab = clean_labels[clean_labels["antibiotic"].isin(panel)].copy()
    lab["val"] = lab["label"].map(_LABEL_ENCODE)
    wide = lab.pivot_table(
        index="genome_id", columns="antibiotic", values="val", aggfunc="max"
    )
    return wide.reindex(index=genome_ids, columns=panel)


def build_dataset(
    cohort: pd.DataFrame,
    determinants: pd.DataFrame,
    clean_labels: pd.DataFrame,
    clusters: dict[str, int] | None = None,
) -> Dataset:
    """Join cohort metadata + determinants + labels (+ optional Mash clusters) → Dataset."""
    genome_ids = cohort["genome_id"].astype(str).tolist()
    meta = cohort.set_index(cohort["genome_id"].astype(str))[["serovar", "mlst"]].copy()
    # Grouping for the honest split: Mash cluster when available, else fall back to serovar
    # (still a lineage-aware group — never a random split).
    if clusters:
        meta["cluster"] = [clusters.get(g, NO_CLUSTER) for g in genome_ids]
    else:
        # ADR-0005 coarse group = serovar × 7-gene MLST (sequence-type level). This is the
        # recommended honest grouping and needs no all-vs-all Mash.
        combo = meta["serovar"].fillna("unknown") + "|" + meta["mlst"].fillna("unknown")
        meta["cluster"] = pd.factorize(combo)[0]

    x_mech = _mechanism_block(determinants, genome_ids)
    x_lineage = _lineage_block(meta)
    y = _label_block(clean_labels, genome_ids)
    return Dataset(
        genome_ids=genome_ids, x_mech=x_mech, x_lineage=x_lineage, y=y, meta=meta
    )
