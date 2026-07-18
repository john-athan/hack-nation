"""Contract tests for the demo preflight checks (demo/preflight.py).

The preflight's whole job is to go RED when a beat would flop on stage, so both the green and the
red branch of each check are locked here — a green-only test would pass even if the doctor could
never fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preflight import check_known_gene_beat, check_ood_beat  # type: ignore[import-not-found]
from verdict import Verdict  # type: ignore[import-not-found]

from genome_firewall.conformal import VERDICT_OOD
from genome_firewall.constants import (
    CALL_NO_CALL,
    CALL_RESISTANT,
    CALL_SUSCEPTIBLE,
)


def _verdict(
    drug: str,
    *,
    therapeutic: bool = True,
    naive_call: str = CALL_SUSCEPTIBLE,
    firewall_verdict: str = CALL_SUSCEPTIBLE,
    firewall_holding: bool = False,
) -> Verdict:
    return Verdict(
        drug=drug,
        therapeutic=therapeutic,
        naive_p=0.1 if naive_call == CALL_SUSCEPTIBLE else 0.9,
        naive_call=naive_call,
        firewall_verdict=firewall_verdict,
        firewall_holding=firewall_holding,
        knockout_delta=0.0,
        evidence="statistical",
    )


def test_known_gene_beat_green_when_holding_and_no_contradiction() -> None:
    verdicts = [
        _verdict("ceftriaxone", firewall_verdict=CALL_RESISTANT, firewall_holding=True),
        _verdict("gentamicin"),
    ]
    check = check_known_gene_beat(verdicts, report_resistant_drugs={"ceftriaxone"})
    assert check.ok
    assert "1 holding" in check.detail


def test_known_gene_beat_red_on_contradiction() -> None:
    # Report calls ceftriaxone resistant (known gene) but the firewall says susceptible — the one
    # thing a safety interlock must never do. Must go RED even though another drug is holding.
    verdicts = [
        _verdict("ceftriaxone", firewall_verdict=CALL_SUSCEPTIBLE),
        _verdict("cefoxitin", firewall_verdict=CALL_RESISTANT, firewall_holding=True),
    ]
    check = check_known_gene_beat(verdicts, report_resistant_drugs={"ceftriaxone"})
    assert not check.ok
    assert "CONTRADICTS" in check.detail


def test_known_gene_beat_red_when_nothing_holds() -> None:
    verdicts = [_verdict("gentamicin"), _verdict("tetracycline")]
    check = check_known_gene_beat(verdicts, report_resistant_drugs=set())
    assert not check.ok
    assert "no firewall-holding" in check.detail


def test_ood_beat_green_on_naive_confident_firewall_abstains() -> None:
    verdicts = [
        _verdict(
            "ciprofloxacin", naive_call=CALL_SUSCEPTIBLE, firewall_verdict=VERDICT_OOD
        ),
        _verdict(
            "ceftriaxone", naive_call=CALL_SUSCEPTIBLE, firewall_verdict=CALL_NO_CALL
        ),
    ]
    check = check_ood_beat(verdicts)
    assert check.ok
    assert "ciprofloxacin" in check.detail


def test_ood_beat_red_when_firewall_commits() -> None:
    # Firewall commits (susceptible) instead of abstaining → no catch, beat falls flat.
    verdicts = [_verdict("ciprofloxacin", firewall_verdict=CALL_SUSCEPTIBLE)]
    check = check_ood_beat(verdicts)
    assert not check.ok


def test_ood_beat_ignores_marker_only_drugs() -> None:
    # A non-therapeutic (marker-only) abstain is not a demo catch — the pitch is about a drug you'd
    # actually give a patient. Must not count.
    verdicts = [
        _verdict(
            "nalidixic acid",
            therapeutic=False,
            naive_call=CALL_SUSCEPTIBLE,
            firewall_verdict=VERDICT_OOD,
        )
    ]
    check = check_ood_beat(verdicts)
    assert not check.ok
