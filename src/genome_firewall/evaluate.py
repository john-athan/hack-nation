"""The honest core (T3): run each drug through random vs grouped CV and report the truth.

This is where the "honest one" story is proven, not asserted. For every drug we compute the
same metrics under a RANDOM split (the dishonest number that leaks lineage and looks great)
and under a GROUPED / leave-clade-out split (the honest number), and we lead with the honest
one. We also pit the calibrated model against the two mandatory baselines — a model that only
ties the known-gene lookup or Mash-nearest-neighbor hasn't earned the word "learned".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import conformal, metrics, model, split
from .dataset import Dataset
from .drugs import DRUG_DB
from .errors import InsufficientDataError

# Drug therapeutic status (ADR-0004): these are internal markers, never a "works" output.
NON_THERAPEUTIC = frozenset({"gentamicin", "cefoxitin", "nalidixic acid"})


@dataclass(frozen=True, slots=True)
class OofPredictions:
    """Pooled out-of-fold predictions, aligned by genome id across model + baselines."""

    genome_ids: list[str]
    y_true: np.ndarray
    model_p: np.ndarray  # calibrated P(R)
    groups: np.ndarray
    known_gene: np.ndarray | None = None
    mash_nn: np.ndarray | None = None


def _model_oof(
    x: pd.DataFrame, y: pd.Series, folds: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Pooled (y_true, P(R)) for the model alone. Starved folds are skipped honestly."""
    y_true: list[int] = []
    scores: list[float] = []
    for train_idx, test_idx in folds:
        try:
            m = model.fit_calibrated_lr(x.iloc[train_idx], y.iloc[train_idx])
        except InsufficientDataError:
            continue
        scores.extend(model.predict_resistant_proba(m, x.iloc[test_idx]).tolist())
        y_true.extend(y.iloc[test_idx].tolist())
    if not scores:
        return None
    return np.array(y_true), np.array(scores)


def _grouped_oof(
    ds: Dataset,
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    drug: str,
    determinants: pd.DataFrame | None,
    nn_map: dict[str, tuple[str, float]] | None,
) -> OofPredictions | None:
    """One honest grouped-CV pass collecting model + both baselines, all aligned by genome id."""
    folds = split.grouped_folds(y.to_numpy(), groups.to_numpy())
    gids: list[str] = []
    y_true: list[int] = []
    mp: list[float] = []
    grp: list[object] = []
    kg: list[int] = []
    nn: list[int] = []
    want_kg = determinants is not None and drug in DRUG_DB
    want_nn = nn_map is not None
    for train_idx, test_idx in folds:
        try:
            m = model.fit_calibrated_lr(x.iloc[train_idx], y.iloc[train_idx])
        except InsufficientDataError:
            continue
        test_ids = x.index[test_idx].astype(str).tolist()
        gids.extend(test_ids)
        y_true.extend(y.iloc[test_idx].tolist())
        mp.extend(model.predict_resistant_proba(m, x.iloc[test_idx]).tolist())
        grp.extend(groups.iloc[test_idx].tolist())
        if want_kg:
            kg.extend(
                model.known_gene_baseline(
                    determinants, DRUG_DB[drug], test_ids
                ).tolist()
            )
        if want_nn:
            nn.extend(
                model.mash_nn_baseline(y.iloc[train_idx], nn_map, test_ids).tolist()
            )
    if not gids:
        return None
    return OofPredictions(
        genome_ids=gids,
        y_true=np.array(y_true),
        model_p=np.array(mp),
        groups=np.array(grp),
        known_gene=np.array(kg) if want_kg else None,
        mash_nn=np.array(nn) if want_nn else None,
    )


def evaluate_drug(
    ds: Dataset,
    drug: str,
    determinants: pd.DataFrame | None = None,
    nn_map: dict[str, tuple[str, float]] | None = None,
    block: str = "mech",
) -> dict[str, object]:
    """Full honest read-out for one drug: collapse comparison + baselines + conformal."""
    result: dict[str, object] = {
        "drug": drug,
        "therapeutic": drug not in NON_THERAPEUTIC,
    }
    try:
        x, y, groups = ds.drug_xy(drug, block=block)
    except KeyError:
        result["status"] = "no_labels"
        return result

    result["n"] = int(len(y))
    result["n_resistant"] = int((y == 1).sum())
    if len(set(y)) < 2:
        result["status"] = (
            "no_call_single_class"  # only R or only S observed → can't train
        )
        return result
    if int((y == 1).sum()) < model.MIN_POSITIVES:
        result["status"] = "no_call_insufficient_positives"
        return result
    result["status"] = "ok"

    # The money slide: same model, dishonest random split vs honest grouped split.
    rnd = _model_oof(x, y, split.random_folds(y.to_numpy()))
    if rnd is not None:
        result["random"] = metrics.binary_metrics(*rnd)

    oof = _grouped_oof(ds, x, y, groups, drug, determinants, nn_map)
    if oof is not None:
        result["grouped"] = metrics.binary_metrics(oof.y_true, oof.model_p)
        if oof.known_gene is not None:
            result["baseline_known_gene"] = metrics.binary_metrics(
                oof.y_true, oof.known_gene.astype(float)
            )
        if oof.mash_nn is not None:
            result["baseline_mash_nn"] = metrics.binary_metrics(
                oof.y_true, oof.mash_nn.astype(float)
            )
        cm = conformal.fit(oof.model_p, oof.y_true, groups=oof.groups)
        result["conformal"] = conformal.empirical_coverage(
            cm, oof.model_p, oof.y_true, oof.groups
        )
    return result


def evaluate_all(
    ds: Dataset,
    determinants: pd.DataFrame | None = None,
    nn_map: dict[str, tuple[str, float]] | None = None,
) -> pd.DataFrame:
    """Evaluate every panel drug → a tidy comparison table (random vs grouped, per drug)."""
    rows: list[dict[str, object]] = []
    for drug in DRUG_DB:
        res = evaluate_drug(ds, drug, determinants=determinants, nn_map=nn_map)
        row: dict[str, object] = {
            "drug": drug,
            "therapeutic": res.get("therapeutic"),
            "status": res.get("status"),
            "n": res.get("n"),
            "n_resistant": res.get("n_resistant"),
        }
        for kind in ("random", "grouped", "baseline_known_gene", "baseline_mash_nn"):
            block = res.get(kind, {})
            if isinstance(block, dict) and "auroc" in block:
                row[f"{kind}_auroc"] = _round(block.get("auroc"))
                row[f"{kind}_bal_acc"] = _round(block.get("balanced_accuracy"))
                # Brier (a proper score: lower = better-calibrated, not just better-ranked) for
                # the two model splits — the calibration dimension of the collapse slide. AUROC
                # rewards ranking, which lineage leakage preserves; Brier penalises the honest
                # split's over-confident probabilities, so it collapses alongside balanced
                # accuracy. metrics.binary_metrics already computes it; we only stopped dropping
                # it. Baselines are 0/1 lookups (no calibrated probability), so we skip them.
                if kind in ("random", "grouped"):
                    row[f"{kind}_brier"] = _round(block.get("brier"))
                    # The remaining Success-Criteria metrics the brief names explicitly (F1,
                    # PR-AUC, and per-class recall reported separately). binary_metrics already
                    # computes them per drug; we only stopped dropping them from the row. PR-AUC
                    # matters under the class imbalance the brief calls out; the two recalls are
                    # sensitivity (R caught) and specificity (S cleared).
                    row[f"{kind}_f1_resistant"] = _round(block.get("f1_resistant"))
                    row[f"{kind}_pr_auc"] = _round(block.get("pr_auc"))
                    row[f"{kind}_recall_resistant"] = _round(
                        block.get("recall_resistant")
                    )
                    row[f"{kind}_recall_susceptible"] = _round(
                        block.get("recall_susceptible")
                    )
        conf = res.get("conformal", {})
        if isinstance(conf, dict):
            row["coverage"] = _round(conf.get("coverage"))
            row["frac_no_call"] = _round(conf.get("frac_no_call"))
            row["frac_ood"] = _round(conf.get("frac_ood"))
        rows.append(row)
    return pd.DataFrame(rows)


def _round(v: object, ndigits: int = 3) -> float:
    return round(float(v), ndigits) if isinstance(v, (int, float)) else float("nan")
