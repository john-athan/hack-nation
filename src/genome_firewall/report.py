"""T1 verdict engine: determinants + drug DB → an honest per-drug GenomeReport.

This is deliberately a RULE-BASED dummy, not the calibrated model (that arrives in T3).
Its honesty is the point: it only asserts *resistant* when a curated determinant explains
it. Absence of a resistance gene is NEVER read as "susceptible" — it becomes a no-call,
because a genome missing a known gene can still resist by an unknown mechanism. The
calibrated statistical layer is what will (cautiously) produce susceptible calls later.
"""

from __future__ import annotations

import pandas as pd

from .constants import (
    CALL_NO_CALL,
    CALL_RESISTANT,
    EVIDENCE_KNOWN_GENE,
    EVIDENCE_NO_SIGNAL,
)
from .drugs import DRUG_DB, Drug, drug_matches_determinant, target_present
from .errors import UnknownDrugError
from .features import Determinant, determinants_for_genome
from .schema import DrugPrediction, GenomeReport

# T1 dummy confidences. The real, calibrated numbers come from the T3 model.
_CONF_KNOWN_GENE = 0.9
_CONF_INTRINSIC = 0.95
_CONF_NO_CALL = 0.0


def _predict_drug(drug: Drug, dets: list[Determinant]) -> DrugPrediction:
    # Intrinsic resistance (EUCAST expected phenotype) is a deterministic R, no gene needed.
    if drug.intrinsic_resistant:
        return DrugPrediction(
            antibiotic=drug.name,
            call=CALL_RESISTANT,
            confidence=_CONF_INTRINSIC,
            evidence_category=EVIDENCE_KNOWN_GENE,
            supporting_genes=[],
            target_present=True,
        )

    hits = [d for d in dets if drug_matches_determinant(drug, d.drug_class, d.subclass)]
    # Empty set = no core-gene screen available for T1; the gate assumes essential targets present.
    present = target_present(drug, set())

    if hits:
        return DrugPrediction(
            antibiotic=drug.name,
            call=CALL_RESISTANT,
            confidence=_CONF_KNOWN_GENE,
            evidence_category=EVIDENCE_KNOWN_GENE,
            supporting_genes=sorted({d.symbol for d in hits}),
            target_present=present,
        )

    # No determinant, target present → we do NOT know. Honest no-call, not "susceptible".
    return DrugPrediction(
        antibiotic=drug.name,
        call=CALL_NO_CALL,
        confidence=_CONF_NO_CALL,
        evidence_category=EVIDENCE_NO_SIGNAL,
        supporting_genes=[],
        target_present=present,
    )


def build_report(
    rows: pd.DataFrame, genome_id: str, drugs: list[str] | None = None
) -> GenomeReport:
    """Assemble the per-drug report for one genome from its determinant rows.

    Raises UnknownDrugError on any requested drug not on the panel — a silent empty
    report (no-silent-fallback doctrine) would be a confusing failure in a live demo.
    """
    dets = determinants_for_genome(rows, genome_id)
    names = drugs if drugs is not None else list(DRUG_DB)
    unknown = [n for n in names if n not in DRUG_DB]
    if unknown:
        raise UnknownDrugError(f"not on panel: {unknown}. Known: {sorted(DRUG_DB)}")
    preds = [_predict_drug(DRUG_DB[n], dets) for n in names]
    return GenomeReport(genome_id=genome_id, predictions=preds)
