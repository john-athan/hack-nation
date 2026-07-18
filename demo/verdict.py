"""Firewall verdicts for the demo: naive vs conformal-firewall + knockout evidence per drug.

This is the USP centerpiece made concrete. For each drug with a trained model we contrast the
NAIVE call (a bare P(R) threshold — the confident-green number a normal AMR model would show)
against the FIREWALL verdict (the conformal prediction set — which abstains as {R,S} or the empty
set {} when neither label clears the coverage bar), and we attach the knockout evidence tier (mechanism vs
lineage/statistical). A row is "the firewall holding" whenever the firewall DIVERGES from a
committed naive call — either by withholding (abstaining) or by overriding it to resistant on a
characterized mechanism. Holding is NOT gated on the naive being *over*-confident: `is_holding`
takes no confidence, so a coin-flip naive call the firewall withholds on holds too. (The demo
caption must therefore not universally claim the firewall "catches an overconfident call".)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from genome_firewall import conformal, knockout
from genome_firewall.constants import (
    CALL_RESISTANT,
    CALL_SUSCEPTIBLE,
    EVIDENCE_KNOWN_GENE,
)
from genome_firewall.drugs import DRUG_DB
from genome_firewall.evaluate import NON_THERAPEUTIC
from genome_firewall.train import TrainedDrug, predict, vectorize_genome

if TYPE_CHECKING:
    from genome_firewall.schema import GenomeReport

_NAIVE_THRESHOLD = 0.5


def is_holding(naive_call: str, firewall_verdict: str) -> bool:
    """Does the firewall DIVERGE from a committed naive call? (The 🛡️ marker's meaning.)

    Holding = the naive model committed to R or S *and* the firewall landed somewhere else — it
    withheld (a NO-CALL / empty set) or overrode to a characterized-mechanism resistant. Note the
    signature: this takes NO confidence. Holding is deliberately independent of *how* confident the
    naive was — the firewall diverges whether the naive was 91% sure or a 51% coin flip — so the UI
    must not claim every 🛡️ row is "catching an overconfident call" (it isn't; see the caption).
    """
    return (
        naive_call in (CALL_RESISTANT, CALL_SUSCEPTIBLE)
        and firewall_verdict != naive_call
    )


@dataclass(frozen=True, slots=True)
class Verdict:
    drug: str
    therapeutic: bool
    naive_p: float
    naive_call: str
    firewall_verdict: str  # constants.CALL_* or conformal.VERDICT_OOD
    firewall_holding: bool  # naive confident but firewall abstains
    knockout_delta: float
    evidence: str


def _naive_call(p: float) -> str:
    return CALL_RESISTANT if p >= _NAIVE_THRESHOLD else CALL_SUSCEPTIBLE


def format_delta(delta: float) -> str:
    """Render a knockout Δ for the firewall table. A NaN Δ marks a report-grounded override (a
    curated known gene the model's own probe did not corroborate) — there is no meaningful
    model-knockout attribution, so we show an em dash instead of a misleading near-zero number."""
    return "—" if math.isnan(delta) else f"{delta:+.2f}"


def naive_confidence(v: Verdict) -> float:
    """Confidence of the naive model's CALLED class. `naive_p` is P(resistant), so a *susceptible*
    call's confidence is `1 - naive_p`. The demo pairs this with the call word ("susceptible (78%)")
    instead of raw P(resistant); showing P(resistant) next to "susceptible" made the signature OOD
    beat read 22% while the narration says the naive "works (78%)" — a contradiction on the USP slide.
    """
    return v.naive_p if v.naive_call == CALL_RESISTANT else 1 - v.naive_p


def verdict_for_drug(
    td: TrainedDrug,
    drug: str,
    symbols: set[str],
    catalog: dict[str, tuple[str, str]],
    known_genes: list[str] | None = None,
) -> Verdict:
    p, pred_set = predict(td, symbols)
    fw = conformal.set_to_verdict(pred_set)
    naive = _naive_call(p)
    # Knockout only over columns the model actually has for this drug.
    ko_cols = [
        c
        for c in knockout.drug_mech_columns(DRUG_DB[drug], catalog)
        if c in td.feature_columns
    ]
    x = vectorize_genome(symbols, td.feature_columns)
    ko = knockout.probe(td.clf, x, ko_cols)
    evidence = ko.evidence
    knockout_delta = ko.delta
    # Mechanism dominates statistics. If the honest rule-based report explains this drug with a
    # curated determinant (or an intrinsic-resistance rule → known_genes == []), the firewall must
    # NOT echo a contrary "susceptible" from a model that simply never learned that gene. A safety
    # interlock that calls a drug "works" over a characterized resistance gene is the exact failure
    # this product exists to prevent — so the firewall becomes mechanism-grounded resistant, which
    # also guarantees it can never contradict the honest report shown right above it.
    if known_genes is not None:
        fw = CALL_RESISTANT
        evidence = EVIDENCE_KNOWN_GENE
        # This known_gene comes from the curated RULE report, not this model's knockout probe. Keep
        # Δ only when the probe INDEPENDENTLY corroborates the mechanism (its own tier is already
        # known_gene → a large Δ as zeroing the gene drops the call). When it does NOT — the rule
        # knows a gene the model never learned, so baseline_p < 0.5 and Δ ≈ 0 — a near-zero Δ printed
        # beside "known_gene" contradicts the column's own definition (Δ = how much the genes drove
        # the call). The verdict is report-grounded, not model-knockout-grounded, so there is no
        # honest model-attribution number to show; NaN renders as "—" rather than a misleading +0.06.
        if ko.evidence != EVIDENCE_KNOWN_GENE:
            knockout_delta = float("nan")
    holding = is_holding(naive, fw)
    return Verdict(
        drug=drug,
        therapeutic=drug not in NON_THERAPEUTIC,
        naive_p=p,
        naive_call=naive,
        firewall_verdict=fw,
        firewall_holding=holding,
        knockout_delta=knockout_delta,
        evidence=evidence,
    )


def _known_gene_resistant(report: GenomeReport) -> dict[str, list[str]]:
    """Drugs the honest rule-based report calls resistant via a curated determinant → its genes.

    Intrinsic-resistance calls carry the same evidence tier with no named gene, so their value is
    an empty list (still a real known-gene override — distinguished from "absent" by dict membership).
    """
    return {
        p.antibiotic: list(p.supporting_genes)
        for p in report.predictions
        if p.call == CALL_RESISTANT and p.evidence_category == EVIDENCE_KNOWN_GENE
    }


def all_verdicts(
    models: dict[str, TrainedDrug],
    symbols: set[str],
    determinants: pd.DataFrame,
    report: GenomeReport | None = None,
) -> list[Verdict]:
    """Every trained drug's naive-vs-firewall verdict for one genome's determinant symbols.

    When `report` (the honest rule-based per-drug report) is supplied, a drug it explains with a
    curated determinant forces the firewall to a mechanism-grounded resistant verdict, so the two
    tables can never disagree. Omitted → pure statistical verdicts (used by the unit tests).
    """
    catalog = knockout.build_catalog(determinants)
    known = _known_gene_resistant(report) if report is not None else {}
    return [
        verdict_for_drug(td, drug, symbols, catalog, known.get(drug))
        for drug, td in models.items()
    ]
