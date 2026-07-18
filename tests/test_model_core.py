"""Tests for the T3 core: calibrated model, baselines, conformal sets, knockout probe."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from genome_firewall import conformal, knockout, model
from genome_firewall.constants import (
    CALL_NO_CALL,
    CALL_RESISTANT,
    EVIDENCE_KNOWN_GENE,
    MECH_PREFIX,
)
from genome_firewall.drugs import DRUG_DB
from genome_firewall.errors import InsufficientDataError


def _separable(n: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    # Feature 0 perfectly separates classes; add noise features.
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * (n // 2))
    x = pd.DataFrame(
        {
            f"{MECH_PREFIX}driver": y,  # the causal gene
            f"{MECH_PREFIX}noise": rng.integers(0, 2, n),
        }
    )
    return x, pd.Series(y)


def test_fit_requires_two_classes_and_positive_floor() -> None:
    x, y = _separable()
    with pytest.raises(InsufficientDataError):
        model.fit_calibrated_lr(
            x, pd.Series(np.zeros(len(y), dtype=int))
        )  # single class
    with pytest.raises(InsufficientDataError):
        few = y.copy()
        few[few == 1] = 0
        few.iloc[:3] = 1  # only 3 positives (< floor)
        model.fit_calibrated_lr(x, few)


def test_fit_and_predict_separable() -> None:
    x, y = _separable()
    m = model.fit_calibrated_lr(x, y)
    p = model.predict_resistant_proba(m, x)
    assert p[y == 1].mean() > p[y == 0].mean()


def test_mash_nn_baseline_transfers_neighbor_label() -> None:
    y_train = pd.Series({"a": 1, "b": 0})
    nn = {"t1": ("a", 0.001), "t2": ("b", 0.002), "t3": ("zzz", 0.5)}
    pred = model.mash_nn_baseline(y_train, nn, ["t1", "t2", "t3"])
    assert pred[0] == 1 and pred[1] == 0  # transferred
    assert pred[2] in (0, 1)  # no in-train neighbor → majority fallback


def test_conformal_set_shapes_and_verdict() -> None:
    cm = conformal.ConformalModel(alpha=0.1, global_q=0.5)
    assert cm.predict_set(0.95) == frozenset({"R"})  # confident R
    assert cm.predict_set(0.05) == frozenset({"S"})  # confident S
    assert cm.predict_set(0.5) == frozenset({"R", "S"})  # ambiguous → no-call
    assert conformal.set_to_verdict(cm.predict_set(0.95)) == CALL_RESISTANT
    assert conformal.set_to_verdict(cm.predict_set(0.5)) == CALL_NO_CALL
    assert conformal.set_to_verdict(frozenset()) == conformal.VERDICT_OOD


def test_serving_empty_set_is_uncertainty_band_not_novelty() -> None:
    """Lock the honest serving semantics: with the served global quantile <0.5 (true for EVERY
    trained model — global_q ranges ~0.04–0.34), the {R,S} shape is UNREACHABLE and the empty set
    {} fires exactly for the uncertain middle band q < P(R) < 1−q. So the empty-set verdict encodes
    PROBABILITY uncertainty (the strongest abstention), NOT an off-manifold/novelty signal — nothing
    in predict_set measures distance to the training manifold. A future 'fix' that relabels {} as
    'OOD / novel' must confront this test (NO-FABRICATION; see conformal.py + app.py _FIREWALL_STYLE).
    """
    q = 0.3  # a representative served global quantile (<0.5)
    cm = conformal.ConformalModel(alpha=0.1, global_q=q)
    ps = np.linspace(0.0, 1.0, 201)
    shapes = [cm.predict_set(float(p)) for p in ps]
    # {R,S} is mathematically unreachable when q < 0.5 (needs 1−p ≤ q AND p ≤ q ⇒ q ≥ 0.5).
    assert frozenset({"R", "S"}) not in shapes
    # The empty set appears ONLY strictly inside the uncertain band (q, 1−q).
    empties = [p for p, s in zip(ps, shapes, strict=True) if s == frozenset()]
    assert empties, "empty set must be reachable in the uncertain band"
    assert all(q < p < 1 - q for p in empties)
    assert (
        cm.predict_set(0.5) == frozenset()
    )  # coin-flip P(R) → strongest abstention, not a call
    # Contrast: only at q ≥ 0.5 does {R,S} ('both plausible') become reachable at all.
    assert conformal.ConformalModel(alpha=0.1, global_q=0.5).predict_set(
        0.5
    ) == frozenset({"R", "S"})


def test_conformal_fit_coverage_guarantee() -> None:
    # Well-separated calibrated scores → coverage should meet the 1−alpha target.
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = np.where(y == 1, rng.uniform(0.8, 0.99, 500), rng.uniform(0.01, 0.2, 500))
    cm = conformal.fit(p, y, alpha=0.1)
    cov = conformal.empirical_coverage(cm, p, y)
    assert cov["coverage"] >= 0.9 - 0.05  # meets target within sampling slack


def test_conformal_mondrian_falls_back_for_thin_groups() -> None:
    p = np.linspace(0.6, 0.99, 40)
    y = np.ones(40, dtype=int)
    groups = np.array(["big"] * 35 + ["thin"] * 5)  # thin < MIN_GROUP_CALIB
    cm = conformal.fit(p, y, groups=groups, alpha=0.1)
    assert "big" in cm.group_q
    assert "thin" not in cm.group_q  # uses global_q


def test_knockout_credits_mechanism_when_gene_flips_call() -> None:
    x, y = _separable()
    m = model.fit_calibrated_lr(x, y)
    # A resistant genome carrying the causal driver gene.
    row = pd.DataFrame({f"{MECH_PREFIX}driver": [1], f"{MECH_PREFIX}noise": [0]})
    res = knockout.probe(m, row, [f"{MECH_PREFIX}driver"])
    assert res.baseline_p > res.knockout_p  # zeroing the gene lowers P(R)
    assert res.evidence == EVIDENCE_KNOWN_GENE
    assert res.zeroed == [f"{MECH_PREFIX}driver"]


def test_knockout_catalog_and_drug_columns() -> None:
    determinants = pd.DataFrame(
        {
            "genome_id": ["g1", "g2"],
            "symbol": ["blaTEM-1", "gyrA_S83F"],
            "subtype": ["AMR", "POINT"],
            "drug_class": ["BETA-LACTAM", "QUINOLONE"],
            "subclass": ["BETA-LACTAM", "QUINOLONE"],
        }
    )
    catalog = knockout.build_catalog(determinants)
    cols = knockout.drug_mech_columns(DRUG_DB["ampicillin"], catalog)
    assert f"{MECH_PREFIX}blaTEM-1" in cols
    assert f"{MECH_PREFIX}gyrA_S83F" not in cols  # quinolone, not a beta-lactam driver
