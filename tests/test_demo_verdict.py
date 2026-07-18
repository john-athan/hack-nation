"""Test the demo's naive-vs-firewall verdict helper against synthetic trained models."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))

from genome_firewall.constants import (  # noqa: E402
    CALL_RESISTANT,
    EVIDENCE_KNOWN_GENE,
    MECH_PREFIX,
)
from genome_firewall.dataset import Dataset  # noqa: E402
from genome_firewall.schema import DrugPrediction, GenomeReport  # noqa: E402
from genome_firewall.train import train_all  # noqa: E402

import inspect  # noqa: E402

from genome_firewall.conformal import VERDICT_OOD  # noqa: E402
from genome_firewall.constants import CALL_NO_CALL, CALL_SUSCEPTIBLE  # noqa: E402
from verdict import (  # noqa: E402  (demo-local module)
    Verdict,
    all_verdicts,
    format_delta,
    is_holding,
    naive_confidence,
)


def test_is_holding_is_confidence_independent_and_covers_both_hold_kinds() -> None:
    # The 🛡️ caption must not claim every hold is "catching an overconfident call": holding is
    # defined WITHOUT any confidence input. The signature proves it — a future refactor that gates
    # holding on a confidence threshold (changing this contract) fails here, not on stage.
    assert list(inspect.signature(is_holding).parameters) == [
        "naive_call",
        "firewall_verdict",
    ]
    # A committed naive call the firewall WITHHOLDS on holds — whether the empty-set abstention...
    assert is_holding(CALL_SUSCEPTIBLE, VERDICT_OOD) is True
    # ...or the {R,S} no-call. (On beat-③ five such holds sit at 51–58% naive: coin flips, not
    # overconfidence — the exact rows the old caption misdescribed.)
    assert is_holding(CALL_SUSCEPTIBLE, CALL_NO_CALL) is True
    # A mechanism OVERRIDE (naive says susceptible, firewall → resistant) also holds — so "abstains
    # rather than guess" is not the whole story either.
    assert is_holding(CALL_SUSCEPTIBLE, CALL_RESISTANT) is True
    # Agreement is not a hold; and a naive that never committed (its own no-call) cannot be "held".
    assert is_holding(CALL_SUSCEPTIBLE, CALL_SUSCEPTIBLE) is False
    assert is_holding(CALL_NO_CALL, VERDICT_OOD) is False


def _verdict(naive_p: float, naive_call: str) -> Verdict:
    return Verdict(
        drug="ampicillin",
        therapeutic=True,
        naive_p=naive_p,
        naive_call=naive_call,
        firewall_verdict=naive_call,
        firewall_holding=False,
        knockout_delta=0.0,
        evidence="statistical",
    )


def test_naive_confidence_shows_called_class_not_p_resistant() -> None:
    # A susceptible call at P(R)=0.22 is 78% confident in *susceptible* — not 22%. Pairing the
    # "susceptible" label with raw P(R) contradicted the demo's "naive says it works (78%)" beat.
    assert naive_confidence(_verdict(0.22, CALL_SUSCEPTIBLE)) == 0.78
    # A resistant call reports P(R) directly.
    assert naive_confidence(_verdict(0.9, CALL_RESISTANT)) == 0.9


def _dataset(n: int = 240) -> Dataset:
    rng = np.random.default_rng(0)
    gids = [f"g{i}" for i in range(n)]
    y = rng.integers(0, 2, n)
    x_mech = pd.DataFrame(
        {f"{MECH_PREFIX}blaTEM-1": y, f"{MECH_PREFIX}noise": rng.integers(0, 2, n)},
        index=gids,
    )
    x_lin = pd.DataFrame({"lin__serovar_A": np.ones(n, dtype="int8")}, index=gids)
    y_df = pd.DataFrame({"ampicillin": y.astype(float)}, index=gids)
    meta = pd.DataFrame(
        {"serovar": ["A"] * n, "mlst": ["m"] * n, "cluster": rng.integers(0, 12, n)},
        index=gids,
    )
    return Dataset(genome_ids=gids, x_mech=x_mech, x_lineage=x_lin, y=y_df, meta=meta)


def _determinants() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "genome_id": ["g0"],
            "symbol": ["blaTEM-1"],
            "subtype": ["AMR"],
            "drug_class": ["BETA-LACTAM"],
            "subclass": ["BETA-LACTAM"],
        }
    )


def test_naive_vs_firewall_and_knockout() -> None:
    models = train_all(_dataset())
    assert "ampicillin" in models
    dets = _determinants()

    carrier = all_verdicts(models, {"blaTEM-1"}, dets)
    v = next(x for x in carrier if x.drug == "ampicillin")
    # Carrier of the causal gene → naive leans resistant; knockout removes the driver → delta>0.
    assert v.naive_p > 0.5
    assert v.knockout_delta > 0
    assert v.evidence  # some evidence tier assigned

    clean = all_verdicts(models, set(), dets)
    vc = next(x for x in clean if x.drug == "ampicillin")
    assert vc.naive_p < v.naive_p  # no driver gene → lower P(R)


def test_firewall_defers_to_known_gene_over_statistical_susceptible() -> None:
    """A safety interlock must never echo 'susceptible' when the honest report has a curated gene."""
    models = train_all(_dataset())
    dets = _determinants()

    # No carrier gene present → the model alone leans susceptible and does NOT commit to resistant.
    stat = next(x for x in all_verdicts(models, set(), dets) if x.drug == "ampicillin")
    assert stat.naive_p < 0.5
    assert stat.firewall_verdict != CALL_RESISTANT

    # But when the honest rule-based report explains the drug with a determinant, mechanism wins:
    # the firewall becomes resistant and flags 🛡️ holding (it corrected the naive 'works').
    report = GenomeReport(
        "g0",
        [
            DrugPrediction(
                "ampicillin",
                CALL_RESISTANT,
                0.9,
                EVIDENCE_KNOWN_GENE,
                ["blaTEM-1"],
                True,
            )
        ],
    )
    v = next(
        x for x in all_verdicts(models, set(), dets, report) if x.drug == "ampicillin"
    )
    assert v.firewall_verdict == CALL_RESISTANT
    assert v.firewall_holding is True
    assert v.evidence == EVIDENCE_KNOWN_GENE


def test_uncorroborated_override_blanks_knockout_delta_but_corroborated_keeps_it() -> (
    None
):
    """A known_gene override the model's own probe does NOT corroborate must not print a near-zero Δ
    beside the 'known_gene' label — the column means 'how much the genes drove the call', so a small
    Δ there flatly contradicts the label (the beat-① ceftriaxone seam). It is NaN → '—'. But when the
    carrier gene IS present the probe corroborates, and that genuine attribution Δ is kept."""
    models = train_all(_dataset())
    dets = _determinants()
    report = GenomeReport(
        "g0",
        [
            DrugPrediction(
                "ampicillin",
                CALL_RESISTANT,
                0.9,
                EVIDENCE_KNOWN_GENE,
                ["blaTEM-1"],
                True,
            )
        ],
    )

    # Uncorroborated: no carrier symbol → model leans susceptible → probe evidence is not known_gene.
    # The rule forces resistant, but the model-knockout Δ is not the basis, so it is blanked to NaN.
    uncorroborated = next(
        x for x in all_verdicts(models, set(), dets, report) if x.drug == "ampicillin"
    )
    assert uncorroborated.evidence == EVIDENCE_KNOWN_GENE
    assert math.isnan(uncorroborated.knockout_delta)
    assert format_delta(uncorroborated.knockout_delta) == "—"

    # Corroborated: the carrier gene IS present, the probe independently grounds the mechanism, so
    # the real positive attribution Δ survives the override (ampicillin/tetracycline on the hero row).
    corroborated = next(
        x
        for x in all_verdicts(models, {"blaTEM-1"}, dets, report)
        if x.drug == "ampicillin"
    )
    assert corroborated.evidence == EVIDENCE_KNOWN_GENE
    assert not math.isnan(corroborated.knockout_delta)
    assert corroborated.knockout_delta > 0
    assert format_delta(corroborated.knockout_delta).startswith("+")
