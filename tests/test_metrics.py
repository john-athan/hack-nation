"""Hermetic tests for AMR classifier scoring (synthetic data, no model/network)."""

from __future__ import annotations

import numpy as np

from genome_firewall.metrics import (
    binary_metrics,
    clinical_agreement,
    cluster_bootstrap_ci,
    risk_coverage,
)


def test_binary_metrics_perfect_classifier() -> None:
    y_true = np.array([0, 0, 1, 1])
    # Scores perfectly separate the classes and sit either side of 0.5.
    y_score = np.array([0.05, 0.2, 0.8, 0.95])
    m = binary_metrics(y_true, y_score)
    assert m["auroc"] == 1.0
    assert m["pr_auc"] == 1.0
    assert m["recall_resistant"] == 1.0
    assert m["recall_susceptible"] == 1.0
    assert m["f1_resistant"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["n"] == 4.0
    assert m["n_resistant"] == 2.0


def test_binary_metrics_single_class_is_nan_not_crash() -> None:
    # All-susceptible truth: ranking metrics undefined, must be NaN and must not raise.
    y_true = np.array([0, 0, 0, 0])
    y_score = np.array([0.1, 0.4, 0.6, 0.9])
    m = binary_metrics(y_true, y_score)
    assert np.isnan(m["auroc"])
    assert np.isnan(m["pr_auc"])
    assert m["n_resistant"] == 0.0


def test_clinical_agreement_known_vme_and_me() -> None:
    # 4 truly R: one predicted S -> VME = 1/4. 4 truly S: two predicted R -> ME = 2/4.
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0, 1, 1])
    a = clinical_agreement(y_true, y_pred)
    assert a["very_major_error"] == 0.25
    assert a["major_error"] == 0.5
    # 5 of 8 agree (3 true-R correct + 2 true-S correct).
    assert a["categorical_agreement"] == 5 / 8


def test_risk_coverage_full_coverage_row_and_monotone() -> None:
    # Confidence-ordered so that as we relax coverage we admit the wrong prediction last.
    y_true = np.array([1, 1, 1, 1, 0])
    y_pred = np.array([1, 1, 1, 1, 1])  # single error on the least-confident row
    confidence = np.array([0.99, 0.95, 0.9, 0.85, 0.55])
    rc = risk_coverage(y_true, y_pred, confidence)

    top = rc.iloc[0]
    assert top["coverage"] == 1.0
    assert top["n_retained"] == 5.0
    # Full-coverage error equals the whole-cohort error (1 wrong of 5).
    assert top["selective_error"] == 0.2
    # Dropping the least-confident (wrong) prediction can only reduce error: monotone-ish.
    errors = rc["selective_error"].to_numpy()
    assert np.all(np.diff(errors) <= 1e-9)
    assert rc.iloc[-1]["selective_error"] == 0.0


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return binary_metrics(y_true, y_score)["auroc"]


def _bootstrap_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n_groups = 12
    y_true: list[int] = []
    y_score: list[float] = []
    groups: list[int] = []
    for g in range(n_groups):
        label = g % 2
        for _ in range(5):
            y_true.append(label)
            # Informative-but-noisy scores so AUROC is neither 1.0 nor degenerate.
            y_score.append(float(np.clip(label * 0.6 + rng.normal(0.2, 0.2), 0.0, 1.0)))
            groups.append(g)
    return np.asarray(y_true), np.asarray(y_score), np.asarray(groups)


def test_cluster_bootstrap_ci_deterministic_and_ordered() -> None:
    y_true, y_score, groups = _bootstrap_dataset()
    lo1, hi1 = cluster_bootstrap_ci(y_true, y_score, groups, _auroc, n_boot=200)
    lo2, hi2 = cluster_bootstrap_ci(y_true, y_score, groups, _auroc, n_boot=200)
    # Same seed -> byte-identical interval.
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 <= hi1
    # A different seed generally moves the interval, proving the seed is actually used.
    lo3, _ = cluster_bootstrap_ci(y_true, y_score, groups, _auroc, n_boot=200, seed=7)
    assert lo3 != lo1 or hi1 == hi2
