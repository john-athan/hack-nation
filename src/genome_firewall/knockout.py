"""Knockout evidence probe (ADR-0007, T3c) — mechanism vs lineage, by intervention.

Gene PRESENCE is weak evidence; INTERVENTION is strong. We zero out the drug's mechanism
determinants in a genome's feature vector and re-predict. If the call collapses R→S, the
model's confidence was mechanism-grounded (those genes caused it). If it stays R without
them, the signal is lineage/statistical — "dark AMR" or a lineage co-traveler the model
leaned on. This turns the PDF's three evidence categories into a rigorous, causal test
rather than a lookup, and powers the dual-oracle evidence quadrants.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from .constants import (
    EVIDENCE_KNOWN_GENE,
    EVIDENCE_NO_SIGNAL,
    EVIDENCE_STATISTICAL,
    MECH_PREFIX,
)
from .drugs import Drug, drug_matches_determinant
from .model import predict_resistant_proba

# A call counts as "Resistant" for the probe above this P(R); mechanism is credited when
# the knockout drops it below the same line.
CALL_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class KnockoutResult:
    baseline_p: float
    knockout_p: float
    delta: float  # baseline_p − knockout_p: how much the mechanism genes drove the call
    zeroed: list[str]
    evidence: str  # constants.EVIDENCE_*


def build_catalog(determinants: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Map each mechanism feature column (mech__<symbol>) → its (Class, Subclass).

    Uses the most frequent class/subclass seen for a symbol (they are near-constant per symbol).
    """
    catalog: dict[str, tuple[str, str]] = {}
    for symbol, grp in determinants.groupby("symbol"):
        cls = (
            str(grp["drug_class"].mode().iloc[0])
            if not grp["drug_class"].mode().empty
            else ""
        )
        sub = (
            str(grp["subclass"].mode().iloc[0])
            if not grp["subclass"].mode().empty
            else ""
        )
        catalog[f"{MECH_PREFIX}{symbol}"] = (cls, sub)
    return catalog


def drug_mech_columns(drug: Drug, catalog: dict[str, tuple[str, str]]) -> list[str]:
    """The mechanism feature columns that confer resistance to `drug` (the knockout target set)."""
    return [
        col
        for col, (cls, sub) in catalog.items()
        if drug_matches_determinant(drug, cls, sub)
    ]


def probe(
    model: CalibratedClassifierCV, x_row: pd.DataFrame, columns: list[str]
) -> KnockoutResult:
    """Re-predict one genome with its drug-mechanism columns zeroed → attribution + evidence tier."""
    if len(x_row) != 1:
        raise ValueError("probe expects exactly one genome row")
    baseline_p = float(predict_resistant_proba(model, x_row)[0])

    present = [c for c in columns if c in x_row.columns]
    x_ko = x_row.copy()
    for c in present:
        x_ko[c] = 0
    knockout_p = float(predict_resistant_proba(model, x_ko)[0])

    return KnockoutResult(
        baseline_p=baseline_p,
        knockout_p=knockout_p,
        delta=baseline_p - knockout_p,
        zeroed=present,
        evidence=_evidence(baseline_p, knockout_p),
    )


def _evidence(baseline_p: float, knockout_p: float) -> str:
    if baseline_p < CALL_THRESHOLD:
        return EVIDENCE_NO_SIGNAL  # not calling R at all
    # Called R: did removing the mechanism genes actually change the mind?
    if knockout_p < CALL_THRESHOLD:
        return EVIDENCE_KNOWN_GENE  # mechanism-grounded — the genes caused the call
    return (
        EVIDENCE_STATISTICAL  # stayed R without them → lineage/statistical / dark-AMR
    )
