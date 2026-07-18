"""Per-drug calibrated classifier + the two mandatory baselines (ADR-0005).

The model is deliberately boring — L2 logistic regression with sigmoid calibration — because
the story is honesty, not a fancy learner (LightGBM is an optional escalation only if it beats
this under the grouped split). ML "ships" for a drug ONLY where it beats BOTH baselines here:
the deterministic known-gene rule and Mash nearest-neighbor label transfer. That guards against
the tautology of AMR genes predicting themselves, and against a model that has merely memorized
lineage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from .drugs import Drug, drug_matches_determinant
from .errors import InsufficientDataError
from .features import determinants_for_genome

# Drug-level floor: below this many positives, calibration is a fantasy → no-call the drug.
MIN_POSITIVES = 10
MAX_CALIB_FOLDS = 5
RANDOM_SEED = 1729


def fit_calibrated_lr(x: pd.DataFrame, y: pd.Series) -> CalibratedClassifierCV:
    """L2 logistic (class-weight balanced) + sigmoid calibration via internal CV.

    Raises InsufficientDataError if a class is missing or positives are below the floor — the
    honest response is a drug-level no-call, never a falsely-confident model on starved data.

    NOTE: the OUTER split (grouped/leave-clade-out) is where lineage leakage is controlled; the
    inner calibration CV is stratified — acceptable, and flagged so it isn't mistaken for leakage.
    """
    classes = set(y.unique())
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if len(classes) < 2:
        raise InsufficientDataError(f"single-class training set (classes={classes})")
    if n_pos < MIN_POSITIVES:
        raise InsufficientDataError(f"only {n_pos} positives (< {MIN_POSITIVES} floor)")

    # liblinear defaults to L2; we don't pass penalty= explicitly because sklearn 1.8
    # deprecated that argument (removed in 1.10) in favour of the default + C.
    base = LogisticRegression(
        class_weight="balanced", solver="liblinear", max_iter=1000
    )
    folds = min(MAX_CALIB_FOLDS, n_pos, n_neg)
    model = CalibratedClassifierCV(base, method="sigmoid", cv=folds)
    model.fit(x.to_numpy(), y.to_numpy())
    return model


def predict_resistant_proba(
    model: CalibratedClassifierCV, x: pd.DataFrame
) -> np.ndarray:
    """P(Resistant) for each row, aligned to the model's class order."""
    classes = list(model.classes_)
    pos_col = classes.index(1)
    return model.predict_proba(x.to_numpy())[:, pos_col]


def known_gene_baseline(
    determinants: pd.DataFrame, drug: Drug, genome_ids: list[str]
) -> np.ndarray:
    """Deterministic baseline: predict Resistant iff a matching curated determinant is present.

    This is the rule the ML must beat to justify itself — if a lookup table does as well, the
    'model' is just re-reading known genes (the tautology ADR-0005 warns about).
    """
    out = np.zeros(len(genome_ids), dtype=int)
    for i, gid in enumerate(genome_ids):
        dets = determinants_for_genome(determinants, gid)
        if any(drug_matches_determinant(drug, d.drug_class, d.subclass) for d in dets):
            out[i] = 1
    return out


def mash_nn_baseline(
    y_train: pd.Series, nn_map: dict[str, tuple[str, float]], test_ids: list[str]
) -> np.ndarray:
    """Mash nearest-neighbor label transfer: predict each test genome's closest TRAIN genome's label.

    `nn_map` maps genome_id → (nearest_other_id, distance) over the full cohort; we walk it to the
    closest neighbor that is actually in the training set. Genomes with no in-train neighbor default
    to the training majority class (a defensible prior, flagged as such).
    """
    train_labels = y_train.to_dict()
    majority = int(round(float(y_train.mean())))
    out = np.full(len(test_ids), majority, dtype=int)
    for i, gid in enumerate(test_ids):
        neighbor = nn_map.get(gid)
        if neighbor is not None and neighbor[0] in train_labels:
            out[i] = int(train_labels[neighbor[0]])
    return out
