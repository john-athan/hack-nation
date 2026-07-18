"""Pure data-prep for the demo's collapse money-slide, extracted from app.py's Streamlit
render so it can be unit-tested.

This is the demo's most regression-prone surface: three separate cycles shipped display bugs
here (charting the metric that does NOT collapse; sorting bars alphabetically instead of by the
drop; ordering the table but not the chart). Every one shipped because the logic was welded to
`st.*` calls and had no test. Isolating the frame prep — which drugs are evaluable, the collapse
magnitude, and the order the pitch depends on (largest drop first) — lets a test lock the contract.
"""

from __future__ import annotations

import pandas as pd

# Balanced accuracy, not AUROC: AUROC barely moves under the honest grouped split (it rewards
# ranking, which lineage leakage preserves), so it reads as "no collapse". Balanced accuracy is
# the operating-point metric a safety interlock lives or dies on — the number that actually craters.
_RANDOM = "random_bal_acc"
_GROUPED = "grouped_bal_acc"
COLLAPSE_COL = "collapse_bal_acc"

# A model at (or barely above) chance on the RANDOM split never learned a signal — there is no
# lineage-driven collapse to demonstrate for it, only noise. Worse, such a drug (typically a
# rare-resistance phenotype with too few positives to learn) can crater on AUROC while its
# balanced accuracy sits flat at ~0.5 — a direct counterexample to the slide's thesis ("AUROC
# barely moves while balanced accuracy craters"). We EXCLUDE these from the collapse comparison
# and surface them separately as "insufficient signal" (never hide them: results.csv keeps every
# drug, and the caption names the excluded ones). Which drugs fall here is derived from the live
# frame and shifts as the annotated cohort grows, so none is named here (an earlier, smaller slice
# had azithromycin excluded — the current cohort makes it the collapse LEADER). Chance is 0.5;
# require a margin above it.
MIN_RANDOM_BAL_ACC = 0.55

# A leave-clade-out estimate needs a minimum cohort to be STABLE. Below it, the test folds hold a
# handful of minority-class isolates, so a drug posts a degenerate AUROC=1.000 (perfect separation
# on a tiny fold) beside wildly unstable conformal coverage — noise that reads on the money slide as
# "too good to be true" next to the slide's scariest number. That is pure downside: an under-powered
# drug is not part of any demo beat, and charting it beside the 1,500+-isolate drugs implies an
# equal footing it does not have. We EXCLUDE these and surface them separately as "too few isolates"
# (never hide them — results.csv keeps every drug and the caption names the excluded ones). The
# threshold is stable across finalizes: cohorts only grow, so a drug that clears it never re-drops,
# and one that starts under it (meropenem, n≈116) re-enters honestly once it has enough data. Chosen
# to exclude the one under-powered drug while keeping the next-smallest evaluable drug (n≈473).
MIN_EVAL_N = 300


def _evaluable(table: pd.DataFrame) -> pd.DataFrame:
    """status == "ok" drugs with both bal-acc scores present (a no-call/insufficient drug has no
    honest number to compare)."""
    return table[table["status"] == "ok"].dropna(subset=[_RANDOM, _GROUPED])


def collapse_frame(table: pd.DataFrame) -> pd.DataFrame | None:
    """Evaluable, *discriminating* drugs with a balanced-accuracy collapse column, sorted
    largest-drop-first so the real lineage-driven collapses lead the chart and table.

    Drugs at/near chance on the random split (< MIN_RANDOM_BAL_ACC) are excluded — they never
    learned a signal, so they have no honest collapse to show (see the constant's rationale). Drugs
    with too few isolates for a stable estimate (< MIN_EVAL_N) are likewise excluded (degenerate
    small-n AUROC / unstable coverage). Returns None when no drug qualifies — the caller renders an
    empty-state, never a blank chart.
    """
    ok = _evaluable(table)
    ok = ok[ok[_RANDOM] >= MIN_RANDOM_BAL_ACC]
    if "n" in ok.columns:
        ok = ok[ok["n"] >= MIN_EVAL_N]
    if ok.empty:
        return None
    return ok.assign(**{COLLAPSE_COL: ok[_RANDOM] - ok[_GROUPED]}).sort_values(
        COLLAPSE_COL, ascending=False
    )


def non_discriminating_drugs(table: pd.DataFrame) -> list[str]:
    """Evaluable drugs whose random-split model sits at/near chance (< MIN_RANDOM_BAL_ACC): no
    learned signal, so excluded from the collapse slide. Named in the caption for honesty."""
    ok = _evaluable(table)
    return sorted(ok[ok[_RANDOM] < MIN_RANDOM_BAL_ACC]["drug"].tolist())


def underpowered_drugs(table: pd.DataFrame) -> list[str]:
    """Evaluable, *discriminating* drugs excluded from the collapse slide for having too few
    isolates (< MIN_EVAL_N) for a stable leave-clade-out estimate. Restricted to discriminating
    drugs so a low-signal low-n drug is named once, by its more fundamental reason (see
    `non_discriminating_drugs`). Named in the caption for honesty. Returns [] if no `n` column."""
    if "n" not in table.columns:
        return []
    ok = _evaluable(table)
    ok = ok[ok[_RANDOM] >= MIN_RANDOM_BAL_ACC]
    return sorted(ok[ok["n"] < MIN_EVAL_N]["drug"].tolist())


def sub_target_coverage_drugs(table: pd.DataFrame, target: float) -> list[str]:
    """Discriminating collapse-slide drugs whose REALIZED conformal coverage on the honest split
    dips below `target` (the ≥1−α guarantee, e.g. 0.90), worst-coverage-first.

    Restricted to the SAME set as `collapse_frame` (discriminating >= MIN_RANDOM_BAL_ACC AND
    adequately powered >= MIN_EVAL_N): a drug absent from the table would be incoherent to name in
    the table's caption — and, since the caption formats values from the table's frame, a
    dip named here but absent from the table would render an empty "(...)". The dip is a symptom of
    the same broken exchangeability the collapse measures (the clade holdout breaks the marginal
    guarantee's exchangeability assumption); balanced accuracy craters for the same reason, but
    drug-by-drug the coverage dip and the bal-acc collapse need NOT track one another (correlation
    is weak), so the caption must not claim they move together. Naming the dipping drugs (never
    hiding them) owns the sub-target number honestly. Returns [] when no coverage column exists
    (older results.csv) or none dip."""
    ok = _evaluable(table)
    ok = ok[ok[_RANDOM] >= MIN_RANDOM_BAL_ACC]
    if "n" in ok.columns:
        ok = ok[ok["n"] >= MIN_EVAL_N]
    if "coverage" not in ok.columns:
        return []
    cov = ok.dropna(subset=["coverage"])
    below = cov[cov["coverage"] < target]
    return below.sort_values("coverage")["drug"].tolist()
