"""The mechanism (rule-based) per-drug report table for the demo.

Extracted from app.py so it's unit-testable off Streamlit (cf. collapse.py, verdict.py) —
a fabricated column can't slip back in past the gate.

WHY there is no confidence column here: this table is the DETERMINISTIC mechanism oracle.
It asserts *resistant* only when a curated determinant (or an intrinsic-resistance rule)
explains it, and no-calls otherwise. The T1 rule layer carries only PLACEHOLDER confidences
(report._CONF_KNOWN_GENE = 0.9, _CONF_INTRINSIC = 0.95) — not calibrated numbers. Surfacing
"90%" under an "Honest by construction" caption invited the fair question "where does 90%
come from?" whose only honest answer is "a constant." So the grounding shown here is
qualitative — the evidence category plus the actual supporting genes — and the CALIBRATED
probability is the separate statistical firewall table's job (the second oracle). This is the
NO-FABRICATION doctrine applied to a display: never print a number we cannot justify.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from genome_firewall.constants import CALL_RESISTANT
from genome_firewall.evaluate import NON_THERAPEUTIC

if TYPE_CHECKING:
    from collections.abc import Iterable

    from genome_firewall.schema import GenomeReport

VERDICT_STYLE = {
    "resistant": "🔴 likely to FAIL",
    "susceptible": "🟢 likely to WORK",
    "no_call": "⚪ NO-CALL",
}

# The mechanism report covers EVERY panel drug; the calibrated firewall table only lists drugs
# with a trained model. When a drug the report calls resistant (e.g. streptomycin) has no model,
# it must not just vanish from the second table — an unexplained omission reads as an internal
# contradiction on the product literally named "The Honest One". These map results.csv's `status`
# (produced by evaluate.evaluate_drug) to the honest reason no classifier was trained; keyed by the
# exact status literals evaluate.py emits so the caption DERIVES from the artifact and never staler.
_UNTRAINED_REASON = {
    "no_call_insufficient_positives": (
        "too few resistant isolates in the cohort to calibrate a classifier"
    ),
    "no_labels": "no lab-measured susceptibility labels for this drug in the cohort",
}
_STATUS_SINGLE_CLASS = "no_call_single_class"
_UNTRAINED_FALLBACK = "no calibrated model was trained for this drug"


def _single_class_reason(row: pd.Series) -> str:
    """Why a single-class (only-R or only-S) drug can't be calibrated — named by direction."""
    n, n_res = row.get("n"), row.get("n_resistant")
    if pd.notna(n) and pd.notna(n_res) and n == n_res:
        return "every cohort isolate tested resistant → single-class, no susceptible contrast to calibrate against"
    if pd.notna(n_res) and n_res == 0:
        return "every cohort isolate tested susceptible → single-class, no resistant contrast to calibrate against"
    return "only one phenotype observed in the cohort → single-class, cannot train a classifier"


def untrained_reported_drugs(
    report: GenomeReport,
    firewall_drugs: Iterable[str],
    results: pd.DataFrame | None,
) -> list[tuple[str, str]]:
    """Report drugs the report calls RESISTANT that have NO calibrated firewall model, each with why.

    Reconciles the mechanism report (all panel drugs) against the firewall table (trained drugs
    only) so a resistant call can never silently disappear between them — the NO-FABRICATION
    doctrine applied to an *omission*. Reasons derive from results.csv's `status` column, so they
    self-update with the cohort and never go stale. Graceful when results is None / lacks columns.

    ONLY resistant calls are named: a no-call the report can't ground has nothing to reconcile —
    it legitimately isn't in the firewall — so narrating it here as a vanished call would itself be
    the contradiction. (The same drug is single-class cohort-wide, but on THIS genome the report
    may still no-call it; the caption must match the row shown above, not the cohort.)
    """
    trained = set(firewall_drugs)
    by_drug: dict[str, pd.Series] = {}
    if results is not None and "drug" in results.columns:
        by_drug = {str(r["drug"]): r for _, r in results.iterrows()}
    out: list[tuple[str, str]] = []
    for p in report.predictions:
        if p.antibiotic in trained or p.call != CALL_RESISTANT:
            continue
        row = by_drug.get(p.antibiotic)
        if (
            row is None
        ):  # pre-status / missing results.csv → graceful fallback, never a crash
            out.append((p.antibiotic, _UNTRAINED_FALLBACK))
            continue
        status = str(row["status"]) if "status" in row else ""
        if status == _STATUS_SINGLE_CLASS:
            reason = _single_class_reason(row)
        else:
            reason = _UNTRAINED_REASON.get(status, _UNTRAINED_FALLBACK)
        out.append((p.antibiotic, reason))
    return out


def report_table(report: GenomeReport) -> pd.DataFrame:
    """One row per drug: the grounded call, its evidence category, and the genes behind it.

    No confidence column — see the module docstring (the rule layer's confidences are
    uncalibrated placeholders; the calibrated number lives in the statistical firewall table).
    """
    rows = [
        {
            "antibiotic": p.antibiotic,
            "verdict": VERDICT_STYLE.get(p.call, p.call),
            "evidence": p.evidence_category,
            "supporting genes": ", ".join(p.supporting_genes) or "—",
            "role": "marker-only" if p.antibiotic in NON_THERAPEUTIC else "therapeutic",
        }
        for p in report.predictions
    ]
    return pd.DataFrame(rows)
