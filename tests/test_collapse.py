"""Lock the demo collapse money-slide's data contract.

This surface shipped a display regression in three separate cycles (metric choice, chart sort,
table-vs-chart order) because it was welded to Streamlit render calls and untested. These tests
pin the invariants the pitch depends on so a fourth can't slip through: largest drop leads, only
evaluable drugs count, NaNs and non-ok statuses drop out, and empty input is a None (empty-state).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from pytest import approx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))

from collapse import (  # noqa: E402  (demo-local module)
    COLLAPSE_COL,
    MIN_EVAL_N,
    MIN_RANDOM_BAL_ACC,
    collapse_frame,
    non_discriminating_drugs,
    sub_target_coverage_drugs,
    underpowered_drugs,
)


def _row(drug: str, status: str, rnd: float | None, grp: float | None) -> dict:
    return {
        "drug": drug,
        "status": status,
        "random_bal_acc": rnd,
        "grouped_bal_acc": grp,
    }


def test_sorted_largest_collapse_first() -> None:
    # The pitch points at the drug whose honest-split balanced accuracy craters most; it must lead.
    table = pd.DataFrame(
        [
            _row("ampicillin", "ok", 0.90, 0.88),  # Δ 0.02 (robust known-gene drug)
            _row("trimethoprim-sulfa", "ok", 0.83, 0.57),  # Δ 0.26 (the collapse)
            _row("chloramphenicol", "ok", 0.85, 0.67),  # Δ 0.18
        ]
    )
    out = collapse_frame(table)
    assert out is not None
    assert list(out["drug"]) == [
        "trimethoprim-sulfa",
        "chloramphenicol",
        "ampicillin",
    ]
    assert out[COLLAPSE_COL].iloc[0] == approx(0.26, abs=1e-9)
    assert out[COLLAPSE_COL].is_monotonic_decreasing


def test_non_ok_status_excluded() -> None:
    table = pd.DataFrame(
        [
            _row("ampicillin", "ok", 0.90, 0.70),
            _row("meropenem", "no_call_insufficient_positives", 0.99, 0.99),
        ]
    )
    out = collapse_frame(table)
    assert out is not None
    assert list(out["drug"]) == ["ampicillin"]


def test_nan_scores_dropped() -> None:
    # A drug with no honest number to compare has no place on the collapse slide.
    table = pd.DataFrame(
        [
            _row("ampicillin", "ok", 0.90, 0.70),
            _row("azithromycin", "ok", 0.80, None),
        ]
    )
    out = collapse_frame(table)
    assert out is not None
    assert list(out["drug"]) == ["ampicillin"]


def test_no_evaluable_drugs_returns_none() -> None:
    # Caller renders an empty-state on None — never a blank chart.
    table = pd.DataFrame(
        [_row("meropenem", "no_call_insufficient_positives", None, None)]
    )
    assert collapse_frame(table) is None


def test_chance_on_random_split_excluded_from_collapse() -> None:
    # A model at chance on the RANDOM split (never learned a signal) has no collapse to show and
    # is a counterexample to the slide's thesis — it must not appear on the collapse chart/table.
    table = pd.DataFrame(
        [
            _row("ciprofloxacin", "ok", 0.83, 0.65),  # real collapse
            _row("azithromycin", "ok", 0.50, 0.50),  # coin-flip on random → excluded
        ]
    )
    out = collapse_frame(table)
    assert out is not None
    assert list(out["drug"]) == ["ciprofloxacin"]


def test_non_discriminating_drugs_named_for_the_honest_note() -> None:
    # The excluded drugs are surfaced (not hidden): the caption names them from this list.
    table = pd.DataFrame(
        [
            _row("ciprofloxacin", "ok", 0.83, 0.65),
            _row("azithromycin", "ok", 0.50, 0.50),
            _row(
                "streptomycin", "no_call_single_class", None, None
            ),  # not evaluable, not here
        ]
    )
    assert non_discriminating_drugs(table) == ["azithromycin"]


def test_all_below_threshold_returns_none() -> None:
    # If every evaluable drug is at chance on random, there is no honest collapse to chart.
    table = pd.DataFrame([_row("azithromycin", "ok", MIN_RANDOM_BAL_ACC - 0.01, 0.50)])
    assert collapse_frame(table) is None


def _cov_row(drug: str, coverage: float | None) -> dict:
    # An evaluable ("ok", both bal-accs present) row carrying a realized-coverage value.
    return {**_row(drug, "ok", 0.90, 0.70), "coverage": coverage}


def test_sub_target_coverage_drugs_named_worst_first() -> None:
    # The demo prints a ≥90% guarantee AND shows realized coverage; where it dips under target the
    # caption must own it, worst-coverage-first, so the money slide can't read as a broken promise.
    table = pd.DataFrame(
        [
            _cov_row("ampicillin", 0.862),  # dips
            _cov_row("ceftriaxone", 0.943),  # holds — not named
            _cov_row("ciprofloxacin", 0.832),  # dips hardest → leads
            _cov_row("gentamicin", 0.887),  # dips
        ]
    )
    assert sub_target_coverage_drugs(table, 0.90) == [
        "ciprofloxacin",
        "ampicillin",
        "gentamicin",
    ]


def test_sub_target_coverage_excludes_non_discriminating_drugs() -> None:
    # A drug at chance on the random split is excluded from the collapse table (collapse_frame), so
    # its coverage dip must NOT be named in the table's caption — else the caption lists a drug the
    # table doesn't show, and since the app formats values from the table frame a lone such dip
    # would render an empty "(...)". Only discriminating (table) drugs may be named.
    table = pd.DataFrame(
        [
            _cov_row("ciprofloxacin", 0.832),  # discriminating, dips → named
            {  # at chance on random → excluded from the slide → must NOT be named even though it dips
                **_row("azithromycin", "ok", 0.50, 0.50),
                "coverage": 0.88,
            },
        ]
    )
    assert sub_target_coverage_drugs(table, 0.90) == ["ciprofloxacin"]


def test_sub_target_coverage_ignores_nan_and_missing_column() -> None:
    # A drug with no measured coverage (NaN) isn't a dip; a pre-coverage results.csv → [] (no crash).
    with_nan = pd.DataFrame(
        [_cov_row("streptomycin", None), _cov_row("ampicillin", 0.862)]
    )
    assert sub_target_coverage_drugs(with_nan, 0.90) == ["ampicillin"]
    no_col = pd.DataFrame([_row("ampicillin", "ok", 0.90, 0.70)])
    assert sub_target_coverage_drugs(no_col, 0.90) == []


def _row_n(drug: str, rnd: float, grp: float, n: int) -> dict:
    # An evaluable, discriminating row carrying its cohort size (the `n` column results.csv ships).
    return {**_row(drug, "ok", rnd, grp), "n": n}


def test_collapse_frame_excludes_underpowered_drugs() -> None:
    # A drug with a real random-split signal but too few isolates (< MIN_EVAL_N) posts a degenerate
    # small-n estimate; it must NOT chart beside the well-powered drugs. RED without the n-gate: an
    # AUROC-perfect meropenem-like row (n=116) would otherwise appear on the money slide.
    table = pd.DataFrame(
        [
            _row_n(
                "meropenem", 0.957, 0.90, MIN_EVAL_N - 1
            ),  # under-powered → excluded
            _row_n("ampicillin", 0.94, 0.80, MIN_EVAL_N + 1),  # powered → kept
        ]
    )
    frame = collapse_frame(table)
    assert frame is not None
    assert frame["drug"].tolist() == ["ampicillin"]


def test_collapse_frame_missing_n_column_is_backward_compatible() -> None:
    # A pre-`n` results.csv (or a test table without the column) must still chart — the n-gate is
    # graceful-degradation, like the coverage/Brier columns, never a hard dependency.
    table = pd.DataFrame([_row("ampicillin", "ok", 0.94, 0.80)])
    frame = collapse_frame(table)
    assert frame is not None
    assert frame["drug"].tolist() == ["ampicillin"]


def test_underpowered_drugs_named_and_disjoint_from_non_discriminating() -> None:
    # The excluded under-powered drug is surfaced (not hidden), and a low-n low-signal drug is named
    # once — by its more fundamental reason (non-discriminating), never in both caption lists.
    table = pd.DataFrame(
        [
            _row_n("meropenem", 0.957, 0.90, 116),  # discriminating but under-powered
            _row_n(
                "azithromycin", 0.50, 0.50, 50
            ),  # at chance AND low-n → non-discriminating only
            _row_n("ampicillin", 0.94, 0.80, 1685),  # kept
        ]
    )
    assert underpowered_drugs(table) == ["meropenem"]
    assert non_discriminating_drugs(table) == ["azithromycin"]
    assert underpowered_drugs(pd.DataFrame([_row("x", "ok", 0.90, 0.70)])) == []


def test_sub_target_coverage_excludes_underpowered_drugs() -> None:
    # A meropenem-like row dips hardest on coverage (0.54) but is under-powered → excluded from the
    # table, so it must NOT be named in the table's caption (else the caption's worst-first lead is a
    # small-n artifact the table doesn't even show).
    table = pd.DataFrame(
        [
            {
                **_row_n("meropenem", 0.957, 0.90, 116),
                "coverage": 0.541,
            },  # under-powered
            {
                **_row_n("ciprofloxacin", 0.89, 0.89, 1565),
                "coverage": 0.873,
            },  # kept, dips
        ]
    )
    assert sub_target_coverage_drugs(table, 0.90) == ["ciprofloxacin"]
