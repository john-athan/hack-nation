"""Feature matrix from AMRFinderPlus determinants.

The USP treats features as two BLOCKS: mechanism (this file — resistance/POINT
determinants) vs lineage (MLST/serovar/Mash — added in T2/T3). Keeping them separate
is what makes the knockout probe and shift-weighting possible later.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Columns of the tidy determinant frame produced by amrfinder.parse_tsv.
_REQUIRED = ("genome_id", "symbol", "subtype", "drug_class", "subclass")


@dataclass(frozen=True, slots=True)
class Determinant:
    """One resistance/POINT determinant call for a genome."""

    symbol: str
    subtype: str
    drug_class: str
    subclass: str


def build_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    """Binary mechanism matrix: genome_id (index) × determinant symbol (columns), 0/1.

    aggfunc=max collapses duplicate hits of the same symbol in one genome to a single 1.
    """
    missing = [c for c in _REQUIRED if c not in rows.columns]
    if missing:
        raise ValueError(f"determinant frame missing columns: {missing}")
    if rows.empty:
        return pd.DataFrame()
    mat = (
        rows.assign(present=1)
        .pivot_table(
            index="genome_id",
            columns="symbol",
            values="present",
            aggfunc="max",
            fill_value=0,
        )
        .astype("int8")
    )
    mat.columns.name = None
    return mat


def determinants_for_genome(rows: pd.DataFrame, genome_id: str) -> list[Determinant]:
    """All determinants called for one genome (used to build the evidence in a report)."""
    sub = rows[rows["genome_id"].astype(str) == str(genome_id)]
    return [
        Determinant(
            symbol=str(r.symbol),
            subtype=str(r.subtype),
            drug_class=str(r.drug_class),
            subclass=str(r.subclass),
        )
        for r in sub.itertuples(index=False)
    ]
