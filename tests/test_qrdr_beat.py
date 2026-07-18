"""Lock beat ④ (gyrA-QRDR) — the mechanism-grounded knockout beat added over genome 54388.377.

Two contracts a future retrain / edit must not silently break:
  1. The ground-truth gate (cycle-41/42 rule): a curated beat that COMMITS fluoroquinolone
     resistance is only allowed when the genome's SHIPPED lab phenotype for that drug is Resistant.
     A lineage over-generalization (no lab support) is the exact defect that nearly shipped a
     self-refuting ceftriaxone-R beat in cycle 41.
  2. The beat's display contract: ≥1 fluoroquinolone is called known-gene resistant AND the knockout
     probe CORROBORATES it (a load-bearing positive Δ), with NO firewall/report contradiction.
     A report-only override (the rule knows a gene the model never learned → NaN / tiny Δ) is not
     enough — the beat's whole point is a model-corroborated mechanism.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

# demo/ and scripts/ each contain a preflight.py; insert scripts first so demo wins index 0 and
# a bare `from preflight import ...` resolves to the demo module (scripts/preflight is the CLI shim).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))

from genome_firewall.constants import (  # noqa: E402
    CALL_RESISTANT,
    CALL_SUSCEPTIBLE,
    EVIDENCE_KNOWN_GENE,
    EVIDENCE_STATISTICAL,
)
from preflight import (  # noqa: E402  (demo-local module)
    QRDR_MIN_KNOCKOUT_DELTA,
    check_qrdr_beat,
)
from verdict import Verdict  # noqa: E402  (demo-local module)


def _v(
    drug: str,
    firewall_verdict: str,
    evidence: str,
    knockout_delta: float,
    *,
    naive_call: str = CALL_RESISTANT,
) -> Verdict:
    return Verdict(
        drug=drug,
        therapeutic=True,
        naive_p=0.8,
        naive_call=naive_call,
        firewall_verdict=firewall_verdict,
        firewall_holding=False,
        knockout_delta=knockout_delta,
        evidence=evidence,
    )


def test_qrdr_beat_passes_on_knockout_corroborated_fq() -> None:
    # The real demo genome: cipro + nalidixic known-gene R with a large corroborating knockout Δ.
    vs = [
        _v("ciprofloxacin", CALL_RESISTANT, EVIDENCE_KNOWN_GENE, 0.51),
        _v("nalidixic acid", CALL_RESISTANT, EVIDENCE_KNOWN_GENE, 0.47),
    ]
    check = check_qrdr_beat(vs, {"ciprofloxacin", "nalidixic acid"})
    assert check.ok
    assert "ciprofloxacin" in check.detail


def test_qrdr_beat_fails_when_knockout_does_not_corroborate() -> None:
    # A report-only override: the rule knows the gene but the model never learned it, so zeroing it
    # barely moves P(R). Below the threshold → the beat has no model-corroborated mechanism to show.
    vs = [_v("ciprofloxacin", CALL_RESISTANT, EVIDENCE_KNOWN_GENE, 0.02)]
    assert not check_qrdr_beat(vs, {"ciprofloxacin"}).ok
    # And a NaN Δ (report-grounded override the probe did not corroborate) also fails.
    vs_nan = [_v("ciprofloxacin", CALL_RESISTANT, EVIDENCE_KNOWN_GENE, float("nan"))]
    assert not check_qrdr_beat(vs_nan, {"ciprofloxacin"}).ok


def test_qrdr_beat_fails_on_firewall_report_contradiction() -> None:
    # The one thing a safety interlock must never do: the report calls cipro resistant while the
    # firewall says susceptible. Even with a valid FQ elsewhere, the contradiction fails the beat.
    vs = [
        _v("ciprofloxacin", CALL_SUSCEPTIBLE, EVIDENCE_STATISTICAL, 0.0),
        _v("nalidixic acid", CALL_RESISTANT, EVIDENCE_KNOWN_GENE, 0.47),
    ]
    check = check_qrdr_beat(vs, {"ciprofloxacin"})
    assert not check.ok
    assert "CONTRADICTS" in check.detail


def test_qrdr_beat_requires_a_fluoroquinolone() -> None:
    # A non-FQ known-gene resistant drug with a big Δ is a fine beat-① catch but is NOT this beat.
    vs = [_v("ceftriaxone", CALL_RESISTANT, EVIDENCE_KNOWN_GENE, 0.9)]
    assert not check_qrdr_beat(vs, set()).ok


def test_qrdr_beat_requires_known_gene_evidence() -> None:
    # A statistical (lineage) FQ resistant call is exactly the cycle-41 false-positive class — the
    # beat must be mechanism-grounded, so statistical evidence does not qualify.
    vs = [_v("ciprofloxacin", CALL_RESISTANT, EVIDENCE_STATISTICAL, 0.5)]
    assert not check_qrdr_beat(vs, set()).ok


def test_lab_gate_returns_only_resistant_fluoroquinolones() -> None:
    from pick_demo_genomes import lab_resistant_drugs  # noqa: PLC0415  # ty: ignore[unresolved-import]
    from preflight import FLUOROQUINOLONES  # noqa: PLC0415

    labels = pd.DataFrame(
        {
            "genome_id": ["g1", "g1", "g1", "g1", "g2"],
            "antibiotic": [
                "ciprofloxacin",
                "nalidixic acid",
                "tetracycline",
                "ampicillin",
                "ciprofloxacin",
            ],
            "resistant_phenotype": [
                "Resistant",
                "Susceptible",  # a Susceptible FQ must NOT leak through the gate
                "Resistant",  # a Resistant NON-FQ must NOT be returned (drug filter)
                "Susceptible",
                "Resistant",  # a different genome must NOT be returned
            ],
        }
    )
    assert lab_resistant_drugs(labels, "g1", FLUOROQUINOLONES) == {"ciprofloxacin"}
    assert lab_resistant_drugs(labels, "g2", FLUOROQUINOLONES) == {"ciprofloxacin"}
    # A genome with no lab-Resistant fluoroquinolone yields an empty set (gate would reject the beat).
    assert lab_resistant_drugs(labels, "absent", FLUOROQUINOLONES) == set()


def test_min_delta_is_a_sane_threshold() -> None:
    # A guard on the shared constant so a future edit can't quietly drop the corroboration bar to ~0.
    assert 0.0 < QRDR_MIN_KNOCKOUT_DELTA < 0.5
    assert not math.isnan(QRDR_MIN_KNOCKOUT_DELTA)
