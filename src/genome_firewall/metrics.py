"""Honest scoring for per-drug AMR classifiers (1=Resistant, 0=Susceptible).

Four things a naive `accuracy` hides and that decide whether this tool is safe to ship:

- Clinical error asymmetry (ADR-0004): calling a truly *Resistant* isolate Susceptible (a
  Very Major Error) can kill a patient; the reverse (Major Error) wastes a drug. CLSI grades
  these separately, so we report VME/ME/CA, not one blended number.
- Rank quality independent of a threshold (AUROC / PR-AUC) — a calibrated cut comes later.
- Selective prediction (ADR-0005): the no-call operating point is chosen from a risk-coverage
  curve to *bound VME*, not from a fixed probability band. This module produces that curve.
- Honest uncertainty (ADR-0005): genomes within a lineage are pseudo-replicates, so an i.i.d.
  row bootstrap understates variance. We resample whole clusters instead.

Pure code: sklearn + numpy, deterministic (seeded Generator), no model and no network.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike

    # A scalar metric of (y_true, y_score); may return NaN on a degenerate (single-class) input.
    MetricFn = Callable[[np.ndarray, np.ndarray], float]

# Class encoding: Resistant is the positive/dangerous-to-miss class throughout.
RESISTANT = 1
SUSCEPTIBLE = 0
DEFAULT_THRESHOLD = 0.5
# AUROC and average precision are undefined on a single-class truth vector — report NaN there
# rather than raising, so a per-drug loop over sparse drugs never crashes.
_MIN_CLASSES_FOR_RANKING = 2

# Risk-coverage sweep: keep the most-confident fraction, from all rows down to a tenth.
_FULL_COVERAGE = 1.0
_MIN_COVERAGE = 0.1
_COVERAGE_STEP = 0.05

# Cluster-bootstrap defaults. 1729 (Hardy-Ramanujan) is arbitrary; only its fixedness matters.
DEFAULT_N_BOOT = 1000
DEFAULT_ALPHA = 0.05
DEFAULT_SEED = 1729
_PERCENTILE_SCALE = 100.0


def binary_metrics(
    y_true: ArrayLike,
    y_score: ArrayLike,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Threshold + ranking metrics for one drug's classifier.

    `y_score` are P(Resistant) probabilities; the hard prediction is `y_score >= threshold`.
    Ranking metrics (AUROC, PR-AUC) are NaN when the truth vector is single-class — undefined,
    not a crash — because sparse drugs routinely have a fold with only susceptible isolates.
    Per-class recalls are NaN when that class is absent (`zero_division`) for the same honesty.
    """
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score, dtype=float)
    y_pred = (y_score_arr >= threshold).astype(int)
    has_both_classes = len(np.unique(y_true_arr)) >= _MIN_CLASSES_FOR_RANKING

    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
        # Sensitivity: of truly Resistant isolates, the fraction caught.
        "recall_resistant": float(
            recall_score(y_true_arr, y_pred, pos_label=RESISTANT, zero_division=np.nan)
        ),
        # Specificity: of truly Susceptible isolates, the fraction correctly cleared.
        "recall_susceptible": float(
            recall_score(
                y_true_arr, y_pred, pos_label=SUSCEPTIBLE, zero_division=np.nan
            )
        ),
        "f1_resistant": float(
            f1_score(y_true_arr, y_pred, pos_label=RESISTANT, zero_division=np.nan)
        ),
        "auroc": float(roc_auc_score(y_true_arr, y_score_arr))
        if has_both_classes
        else float("nan"),
        "pr_auc": float(average_precision_score(y_true_arr, y_score_arr))
        if has_both_classes
        else float("nan"),
        "brier": float(brier_score_loss(y_true_arr, y_score_arr)),
        "n": float(len(y_true_arr)),
        "n_resistant": float(np.sum(y_true_arr == RESISTANT)),
    }


def clinical_agreement(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """CLSI-style categorical error rates (ADR-0004).

    - very_major_error (VME): truly Resistant called Susceptible, as a fraction of truly
      Resistant isolates. The dangerous error — it withholds an effective drug from a patient.
    - major_error (ME): truly Susceptible called Resistant, as a fraction of truly Susceptible.
    - categorical_agreement (CA): overall S/R agreement.

    Essential Agreement (EA, MIC within one doubling dilution) is intentionally omitted: it
    needs raw MIC dilutions, which a binary S/R classifier does not produce.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    true_r = y_true_arr == RESISTANT
    true_s = y_true_arr == SUSCEPTIBLE
    n_true_r = int(true_r.sum())
    n_true_s = int(true_s.sum())

    # Rate is undefined with no denominator (a fold lacking that truth class) — report NaN.
    vme = (
        float(np.sum(y_pred_arr[true_r] == SUSCEPTIBLE) / n_true_r)
        if n_true_r
        else float("nan")
    )
    me = (
        float(np.sum(y_pred_arr[true_s] == RESISTANT) / n_true_s)
        if n_true_s
        else float("nan")
    )

    return {
        "very_major_error": vme,
        "major_error": me,
        "categorical_agreement": float(np.mean(y_true_arr == y_pred_arr)),
    }


def risk_coverage(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    confidence: ArrayLike,
) -> pd.DataFrame:
    """Selective-prediction curve: error vs. how much of the cohort we choose to answer.

    Sort predictions by descending confidence, then at each coverage level keep that top
    fraction and measure its error. Backs the no-call decision (ADR-0005): a per-drug operating
    point is picked where retained error (and specifically VME) is acceptably low. Coverage 1.0
    (answer everything) is always the first row, and its error equals the full-cohort error.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    conf_arr = np.asarray(confidence, dtype=float)
    n = len(y_true_arr)

    # Most-confident first; a stable sort keeps ties in input order for reproducibility.
    order = np.argsort(-conf_arr, kind="stable")
    wrong_sorted = (y_pred_arr[order] != y_true_arr[order]).astype(float)

    # Inclusive descending sweep 1.00, 0.95, ... 0.10; round to kill float-accumulation drift.
    n_levels = int(round((_FULL_COVERAGE - _MIN_COVERAGE) / _COVERAGE_STEP)) + 1
    levels = np.round(_FULL_COVERAGE - _COVERAGE_STEP * np.arange(n_levels), decimals=2)

    records: list[dict[str, float]] = []
    for level in levels:
        # At least one retained prediction, so error is always defined.
        n_retained = max(1, int(round(float(level) * n)))
        error = float(wrong_sorted[:n_retained].mean())
        records.append(
            {
                "coverage": float(level),
                "selective_error": error,
                "selective_accuracy": 1.0 - error,
                "n_retained": float(n_retained),
            }
        )
    return pd.DataFrame.from_records(records)


def cluster_bootstrap_ci(
    y_true: ArrayLike,
    y_score: ArrayLike,
    groups: ArrayLike,
    metric_fn: MetricFn,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """Percentile CI for a metric, resampling whole CLUSTERS with replacement (ADR-0005).

    Genomes within one lineage are pseudo-replicates, so resampling rows i.i.d. would pretend
    we have far more independent evidence than we do and shrink the interval dishonestly.
    Instead we draw K clusters (K = number of distinct groups) with replacement and pool their
    rows. Boots that come out single-class (metric_fn returns NaN) are dropped, since a ranking
    metric is undefined there. Deterministic: a fixed-seed Generator gives identical output.
    """
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score, dtype=float)
    groups_arr = np.asarray(groups)

    unique_groups = np.unique(groups_arr)
    # Precompute each cluster's row indices once so a boot is just a gather over sampled groups.
    group_to_rows = {g: np.flatnonzero(groups_arr == g) for g in unique_groups}
    n_groups = len(unique_groups)

    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=n_groups, replace=True)
        rows = np.concatenate([group_to_rows[g] for g in sampled])
        value = metric_fn(y_true_arr[rows], y_score_arr[rows])
        if not np.isnan(value):
            values.append(value)

    if not values:
        return float("nan"), float("nan")

    lower = float(np.percentile(values, _PERCENTILE_SCALE * (alpha / 2.0)))
    upper = float(np.percentile(values, _PERCENTILE_SCALE * (1.0 - alpha / 2.0)))
    return lower, upper
