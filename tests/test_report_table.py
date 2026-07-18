"""Lock the mechanism report table's honesty contract.

The rule-based report carries only PLACEHOLDER confidences (report._CONF_KNOWN_GENE = 0.9),
so surfacing them as a "confidence" column under an "Honest by construction" caption is a
fabrication smell — the number isn't calibrated. These tests pin that the table shows the
grounded call + evidence + genes and NEVER a confidence number, so it can't creep back in.
"""

from __future__ import annotations

import sys
from pathlib import Path

from genome_firewall.constants import (
    CALL_NO_CALL,
    CALL_RESISTANT,
    EVIDENCE_KNOWN_GENE,
    EVIDENCE_NO_SIGNAL,
)
from genome_firewall.schema import DrugPrediction, GenomeReport

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))

import pandas as pd  # noqa: E402

from report_table import report_table, untrained_reported_drugs  # noqa: E402  (demo-local module)


def _report() -> GenomeReport:
    return GenomeReport(
        genome_id="test",
        predictions=[
            DrugPrediction(
                antibiotic="ampicillin",
                call=CALL_RESISTANT,
                confidence=0.9,  # placeholder — must NOT reach the table
                evidence_category=EVIDENCE_KNOWN_GENE,
                supporting_genes=["blaTEM-1"],
                target_present=True,
            ),
            DrugPrediction(
                antibiotic="ciprofloxacin",
                call=CALL_NO_CALL,
                confidence=0.0,
                evidence_category=EVIDENCE_NO_SIGNAL,
                supporting_genes=[],
                target_present=True,
            ),
        ],
    )


def test_no_fabricated_confidence_column() -> None:
    # The core guard: the uncalibrated placeholder confidence must never surface here.
    table = report_table(_report())
    assert "confidence" not in table.columns


def test_grounding_is_shown() -> None:
    table = report_table(_report()).set_index("antibiotic")
    # The resistant call carries its real driving gene and evidence category.
    assert table.loc["ampicillin", "supporting genes"] == "blaTEM-1"
    assert table.loc["ampicillin", "evidence"] == EVIDENCE_KNOWN_GENE
    # A no-call shows no genes but stays honest about why (no signal).
    assert table.loc["ciprofloxacin", "supporting genes"] == "—"
    assert table.loc["ciprofloxacin", "evidence"] == EVIDENCE_NO_SIGNAL


def test_row_per_prediction() -> None:
    assert len(report_table(_report())) == 2


def _report_with_streptomycin() -> GenomeReport:
    return GenomeReport(
        genome_id="beat3",
        predictions=[
            DrugPrediction(
                antibiotic="ampicillin",
                call=CALL_RESISTANT,
                confidence=0.9,
                evidence_category=EVIDENCE_KNOWN_GENE,
                supporting_genes=["blaTEM-1"],
                target_present=True,
            ),
            DrugPrediction(
                antibiotic="streptomycin",
                call=CALL_RESISTANT,
                confidence=0.9,
                evidence_category=EVIDENCE_KNOWN_GENE,
                supporting_genes=["aph(3'')-Ib", "aph(6)-Id"],
                target_present=True,
            ),
        ],
    )


def test_reported_resistant_drug_without_a_model_is_named_not_dropped() -> None:
    # THE gap: a drug the mechanism report calls resistant (streptomycin) has no trained model,
    # so it vanishes from the firewall table. It must be surfaced with an honest reason instead.
    results = pd.DataFrame(
        {
            "drug": ["ampicillin", "streptomycin"],
            "status": ["ok", "no_call_single_class"],
            "n": [1685, 688],
            "n_resistant": [856, 688],
        }
    )
    untrained = untrained_reported_drugs(
        _report_with_streptomycin(), firewall_drugs={"ampicillin"}, results=results
    )
    assert [d for d, _ in untrained] == [
        "streptomycin"
    ]  # ampicillin has a model → not named
    (_, reason) = untrained[0]
    # n == n_resistant → the honest single-class reason names the direction (all resistant).
    assert "resistant" in reason and "single-class" in reason


def test_untrained_reason_covers_each_status() -> None:
    def _one(status: str, n: int, n_res: int) -> str:
        rep = GenomeReport(
            genome_id="g",
            predictions=[
                DrugPrediction(
                    antibiotic="streptomycin",
                    call=CALL_RESISTANT,
                    confidence=0.9,
                    evidence_category=EVIDENCE_KNOWN_GENE,
                    supporting_genes=["aph(6)-Id"],
                    target_present=True,
                )
            ],
        )
        results = pd.DataFrame(
            {
                "drug": ["streptomycin"],
                "status": [status],
                "n": [n],
                "n_resistant": [n_res],
            }
        )
        return untrained_reported_drugs(rep, firewall_drugs=set(), results=results)[0][
            1
        ]

    assert "susceptible" in _one("no_call_single_class", 500, 0)  # all-S single class
    assert "too few resistant" in _one("no_call_insufficient_positives", 500, 3)
    assert "labels" in _one("no_labels", 0, 0)


def test_untrained_no_call_drug_is_not_named() -> None:
    # Regression guard: reconciliation names a drug ONLY where the report calls it RESISTANT. A
    # single-class drug the report NO-CALLs on this genome (streptomycin on the ESBL beat: no genes
    # hit → no_signal) must NOT be narrated "every isolate tested resistant → see the report" while
    # the row above shows ⚪ NO-CALL — that caption/report contradiction is the exact honesty hole.
    report = GenomeReport(
        genome_id="beat1",
        predictions=[
            DrugPrediction(
                antibiotic="ampicillin",
                call=CALL_RESISTANT,
                confidence=0.9,
                evidence_category=EVIDENCE_KNOWN_GENE,
                supporting_genes=["blaTEM-1"],
                target_present=True,
            ),
            DrugPrediction(
                antibiotic="streptomycin",  # single-class cohort-wide, but NO-CALL on THIS genome
                call=CALL_NO_CALL,
                confidence=0.0,
                evidence_category=EVIDENCE_NO_SIGNAL,
                supporting_genes=[],
                target_present=True,
            ),
        ],
    )
    results = pd.DataFrame(
        {
            "drug": ["ampicillin", "streptomycin"],
            "status": ["ok", "no_call_single_class"],
            "n": [1685, 688],
            "n_resistant": [856, 688],
        }
    )
    untrained = untrained_reported_drugs(
        report, firewall_drugs={"ampicillin"}, results=results
    )
    assert (
        untrained == []
    )  # ampicillin trained; streptomycin no-call here → nothing to reconcile


def test_untrained_reported_drugs_graceful_without_results() -> None:
    # A pre-status / missing results.csv still names the untrained drug with the fallback reason,
    # never crashes (mirrors the collapse helpers' backward-compat doctrine).
    untrained = untrained_reported_drugs(
        _report_with_streptomycin(), firewall_drugs={"ampicillin"}, results=None
    )
    assert [d for d, _ in untrained] == ["streptomycin"]
    assert untrained[0][1]  # a non-empty fallback reason, no KeyError
