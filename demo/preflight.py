"""Pure demo-readiness checks for the preflight "doctor" (scripts/preflight.py).

The 90-second demo hangs on three artifacts (models.joblib, results.csv, demo_genomes.json)
and two curated genomes rendering the way the pitch narrates. Every cycle re-verifies this by
hand; this isolates the *decision* logic — is a beat holding, does the OOD genome abstain, is the
money slide non-empty — so one command (and a unit test) can confirm the demo is stage-ready
without retraining or booting Streamlit. Pure functions here; all I/O lives in the script.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from genome_firewall.conformal import VERDICT_OOD
from genome_firewall.constants import CALL_NO_CALL, CALL_RESISTANT, EVIDENCE_KNOWN_GENE

if TYPE_CHECKING:
    from collections.abc import Iterable

    from verdict import Verdict

# The firewall "abstains" when its conformal set won't commit: {R,S}=no-call or {}=OOD-novel.
# These are exactly the verdicts that make the beat-③ catch (naive confident, firewall holds back).
ABSTAIN_VERDICTS = frozenset({CALL_NO_CALL, VERDICT_OOD})

# Fluoroquinolones a single gyrA QRDR point mutation drives — the drugs beat ④ commits on.
FLUOROQUINOLONES = frozenset({"ciprofloxacin", "nalidixic acid"})
# Beat ④'s knockout must CORROBORATE the mechanism: zeroing the QRDR feature drops P(R) by at least
# this much. A report-only override (the rule knows a gene the model never learned) carries a NaN Δ
# and would fail here — the beat's whole point is a model-corroborated mechanism (the demo Δ is ≈+0.5).
QRDR_MIN_KNOCKOUT_DELTA = 0.10


@dataclass(frozen=True, slots=True)
class Check:
    """One preflight assertion: whether it passed and a one-line human detail."""

    name: str
    ok: bool
    detail: str


def check_known_gene_beat(
    verdicts: Iterable[Verdict], report_resistant_drugs: set[str]
) -> Check:
    """Beat ①: the firewall must HOLD ≥1 overconfident call and NEVER contradict the honest report.

    A contradiction is a drug the mechanism report calls resistant (a curated determinant explains
    it) while the firewall says anything other than resistant — the one thing a safety interlock
    must never do. `report_resistant_drugs` is the set of such drugs, computed by the caller.
    """
    vs = list(verdicts)
    contradictions = sorted(
        v.drug
        for v in vs
        if v.drug in report_resistant_drugs and v.firewall_verdict != CALL_RESISTANT
    )
    holding = sorted(v.drug for v in vs if v.firewall_holding)
    ok = not contradictions and bool(holding)
    if contradictions:
        detail = (
            f"firewall CONTRADICTS the report on {contradictions} — must never happen"
        )
    elif not holding:
        detail = "no firewall-holding catch — beat ① has nothing to show"
    else:
        detail = f"{len(holding)} holding {holding}, 0 contradictions"
    return Check("beat① known-gene resistance", ok, detail)


def check_qrdr_beat(
    verdicts: Iterable[Verdict], report_resistant_drugs: set[str]
) -> Check:
    """Beat ④ (gyrA-QRDR): ≥1 fluoroquinolone called known-gene resistant that the knockout probe
    CORROBORATES (a real positive Δ — the QRDR mutation drives the call), and NO contradiction.

    The mechanism-grounded knockout story: one gyrA point mutation → clinically relevant
    fluoroquinolone resistance, and zeroing the QRDR feature moves the call. Unlike beat ①, the
    firewall need not DIVERGE from the naive call — a QRDR mutation is strong enough that the naive
    model already commits resistant, so `firewall_holding` is False here; the beat's contract is the
    corroborated mechanism (evidence + a load-bearing knockout Δ), not a hold. `report_resistant_drugs`
    is the caller-computed set the honest report explains with a curated determinant (for the
    contradiction check — the one thing a safety interlock must never do).
    """
    vs = list(verdicts)
    contradictions = sorted(
        v.drug
        for v in vs
        if v.drug in report_resistant_drugs and v.firewall_verdict != CALL_RESISTANT
    )
    driven = sorted(
        v.drug
        for v in vs
        if v.drug in FLUOROQUINOLONES
        and v.firewall_verdict == CALL_RESISTANT
        and v.evidence == EVIDENCE_KNOWN_GENE
        and not math.isnan(v.knockout_delta)
        and v.knockout_delta >= QRDR_MIN_KNOCKOUT_DELTA
    )
    ok = not contradictions and bool(driven)
    if contradictions:
        detail = (
            f"firewall CONTRADICTS the report on {contradictions} — must never happen"
        )
    elif not driven:
        detail = "no knockout-corroborated fluoroquinolone resistance — beat ④ has nothing to show"
    else:
        detail = f"{len(driven)} FQ knockout-corroborated {driven}, 0 contradictions"
    return Check("beat④ gyrA-QRDR knockout", ok, detail)


def check_ood_beat(verdicts: Iterable[Verdict]) -> Check:
    """Beat ③: ≥1 therapeutic drug the naive model calls "works" while the firewall abstains.

    This is the exact criterion pick_demo_genomes.py picks the OOD genome by, so it stays true
    across retrains — no brittle parsing of the preset label's drug name.
    """
    catches = sorted(
        v.drug
        for v in verdicts
        if v.therapeutic
        and v.naive_call != CALL_RESISTANT
        and v.firewall_verdict in ABSTAIN_VERDICTS
    )
    ok = bool(catches)
    detail = (
        f"naive says works, firewall abstains on {catches}"
        if catches
        else "no naive-confident / firewall-abstain catch — beat ③ falls flat"
    )
    return Check("beat③ firewall abstains (won't commit)", ok, detail)
