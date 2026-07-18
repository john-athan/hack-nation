"""Train + persist per-drug models so the demo can serve calibrated conformal verdicts live.

Each drug gets a final calibrated model fit on ALL its labeled cohort genomes, plus a conformal
calibrator fit on honest grouped out-of-fold scores (so the prediction sets keep their coverage
guarantee on unseen lineages). We freeze the feature column order so a freshly-uploaded genome is
vectorized identically — determinants the model never saw are simply absent (and drive OOD).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from . import conformal, model, split
from .atomicio import atomic_write
from .constants import DATA_DIR, MECH_PREFIX
from .dataset import Dataset
from .errors import InsufficientDataError

MODELS_PATH = DATA_DIR / "models.joblib"


@dataclass(frozen=True, slots=True)
class TrainedDrug:
    """A frozen per-drug predictor: calibrated model + conformal sets + its feature columns."""

    drug: str
    clf: CalibratedClassifierCV
    conformal: conformal.ConformalModel
    feature_columns: list[str]
    n: int
    n_resistant: int


def train_drug(ds: Dataset, drug: str, block: str = "mech") -> TrainedDrug | None:
    """Fit final model on all data + conformal on honest grouped OOF. None if below the floor."""
    try:
        x, y, groups = ds.drug_xy(drug, block=block)
    except KeyError:
        return None
    if len(set(y)) < 2 or int((y == 1).sum()) < model.MIN_POSITIVES:
        return None

    # Conformal calibration from grouped out-of-fold predictions (no lineage leakage).
    folds = split.grouped_folds(y.to_numpy(), groups.to_numpy())
    oof_y: list[int] = []
    oof_p: list[float] = []
    oof_g: list[object] = []
    for tr, te in folds:
        try:
            m = model.fit_calibrated_lr(x.iloc[tr], y.iloc[tr])
        except InsufficientDataError:
            continue
        oof_p.extend(model.predict_resistant_proba(m, x.iloc[te]).tolist())
        oof_y.extend(y.iloc[te].tolist())
        oof_g.extend(groups.iloc[te].tolist())
    if not oof_p:
        return None
    cm = conformal.fit(np.array(oof_p), np.array(oof_y), groups=np.array(oof_g))

    final = model.fit_calibrated_lr(x, y)  # final model uses all data
    return TrainedDrug(
        drug=drug,
        clf=final,
        conformal=cm,
        feature_columns=list(x.columns),
        n=int(len(y)),
        n_resistant=int((y == 1).sum()),
    )


def train_all(ds: Dataset) -> dict[str, TrainedDrug]:
    from .drugs import DRUG_DB

    trained: dict[str, TrainedDrug] = {}
    for drug in DRUG_DB:
        td = train_drug(ds, drug)
        if td is not None:
            trained[drug] = td
    return trained


def vectorize_genome(
    determinant_symbols: set[str], feature_columns: list[str]
) -> pd.DataFrame:
    """Build a one-row feature frame for a new genome aligned to a model's columns.

    A determinant the model never trained on is simply not a column here — it contributes no
    positive evidence. Be honest about the consequence: a genome whose ONLY signal is a truly
    novel determinant vectorizes identically to a susceptible one, so the serving conformal set
    reflects PROBABILITY uncertainty, not a measured novelty/OOD signal — no distance to the
    training manifold is computed on the upload path (see conformal.py).
    """
    present = {f"{MECH_PREFIX}{s}" for s in determinant_symbols}
    row = {c: (1 if c in present else 0) for c in feature_columns}
    return pd.DataFrame([row], columns=feature_columns).astype("int8")


def predict(
    td: TrainedDrug, determinant_symbols: set[str]
) -> tuple[float, frozenset[str]]:
    """(P(Resistant), conformal prediction set) for one genome under a trained drug model."""
    x = vectorize_genome(determinant_symbols, td.feature_columns)
    p = float(model.predict_resistant_proba(td.clf, x)[0])
    # No `group` on purpose: a served genome (esp. a live upload) is NOT localized into the training
    # Mash lineage groups, so there is no honest per-lineage quantile to apply here — serving uses the
    # GLOBAL marginal quantile (still distribution-free ≥1−α). The Mondrian per-group quantiles drive
    # the offline coverage evaluation (conformal.empirical_coverage), not this per-verdict path.
    return p, td.conformal.predict_set(p)


def save(models: dict[str, TrainedDrug], path: Path = MODELS_PATH) -> Path:
    # Atomic so an OOM/SIGKILL during the unattended finalize can't leave a truncated model that
    # crashes the morning demo on load (a partial write is discarded; the prior good file survives).
    return atomic_write(path, lambda tmp: joblib.dump(models, tmp))


def load(path: Path = MODELS_PATH) -> dict[str, TrainedDrug]:
    return joblib.load(path)
