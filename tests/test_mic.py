"""Tests for MIC → S/I/R re-derivation, focusing on the safety-critical censoring rules."""

from __future__ import annotations

import pandas as pd

from genome_firewall import mic
from genome_firewall.mic import BREAKPOINTS, Breakpoint, interpret_mic


def test_exact_values() -> None:
    bp = Breakpoint(s_max=8, r_min=32)  # ampicillin-like
    assert interpret_mic(4, "=", bp) == mic.S
    assert interpret_mic(16, "=", bp) == mic.INT
    assert interpret_mic(32, "=", bp) == mic.R
    assert interpret_mic(8, "", bp) == mic.S  # empty sign → exact


def test_censored_upper_bound_never_false_susceptible() -> None:
    bp = Breakpoint(s_max=8, r_min=32)
    assert interpret_mic(4, "<=", bp) == mic.S  # <=4 ≤ s_max → S
    # "<=16" is above s_max: could be S or I → ambiguous, must NOT be called S.
    assert interpret_mic(16, "<=", bp) is None


def test_censored_lower_bound_resistant() -> None:
    bp = Breakpoint(s_max=8, r_min=32)
    assert interpret_mic(32, ">", bp) == mic.R  # >32 ≥ r_min → R
    assert interpret_mic(16, ">", bp) == mic.INT  # >16: non-susceptible, in I band
    # ">4" with 4<s_max: MIC could still be ≤8 → uninterpretable, never a false R.
    assert interpret_mic(4, ">", bp) is None


def test_cipro_trap_single_mutant_is_nonsusceptible() -> None:
    bp = BREAKPOINTS["ciprofloxacin"]  # S≤0.06, R≥1
    assert interpret_mic(0.03, "=", bp) == mic.S  # wild-type
    assert (
        interpret_mic(0.25, "=", bp) == mic.INT
    )  # single QRDR mutant → NOT susceptible
    assert interpret_mic(2, "=", bp) == mic.R


def test_rederive_drops_non_mic_units_and_dedups() -> None:
    raw = pd.DataFrame(
        {
            "genome_id": ["g1", "g1", "g2", "g3"],
            "antibiotic": ["ampicillin", "ampicillin", "ampicillin", "ampicillin"],
            "resistant_phenotype": [
                "Resistant",
                "Resistant",
                "Susceptible",
                "Resistant",
            ],
            "measurement_value": ["32", "32", "2", "20"],
            "measurement_sign": [">", ">", "<=", "="],
            "measurement_unit": [
                "mg/L",
                "mg/L",
                "mg/L",
                "mm",
            ],  # g3 is disk diffusion → dropped
        }
    )
    out = mic.rederive(raw)
    got = dict(zip(out["genome_id"], out["label"], strict=True))
    assert got == {"g1": mic.R, "g2": mic.S}  # g3 dropped (mm), g1 deduped


def test_label_churn_reports_categorical_disagreement() -> None:
    raw = pd.DataFrame(
        {
            "genome_id": ["g1", "g2"],
            "antibiotic": ["ciprofloxacin", "ciprofloxacin"],
            "resistant_phenotype": ["Susceptible", "Susceptible"],
            "measurement_value": ["0.25", "0.03"],  # g1 → I (trap), g2 → S
            "measurement_sign": ["=", "="],
            "measurement_unit": ["mg/L", "mg/L"],
        }
    )
    churn = mic.label_churn(raw)
    row = churn[churn["antibiotic"] == "ciprofloxacin"].iloc[0]
    assert row["s_to_nonS"] == 1
    assert row["cat_disagree"] == 1
