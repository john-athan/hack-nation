"""Phylogenetically-localized conformal prediction sets (ADR-0007, T3b) — the USP core.

Instead of a bare probability, each drug emits a PREDICTION SET over {S, R} with a
distribution-free coverage guarantee (P(true ∈ set) ≥ 1−α). The set shapes carry the epistemology:
    {R} / {S}  → a confident call
    {R, S}     → NO-CALL: both labels clear the bar → both plausible, won't commit
    {}         → NO-CALL (strongest): NEITHER label clears the 1−α nonconformity bar, i.e. P(R)
                 sits in the uncertain middle band (q < P(R) < 1−q) — the model won't commit.
NOT a novelty/OOD verdict. The empty set reflects PROBABILITY uncertainty, not a measured distance
to the training manifold — nothing on the serving path computes such a distance, and a genome whose
only signal is a truly novel determinant vectorizes like a susceptible one (train.vectorize_genome).
Note also that with the served global quantile <0.5 (true for every trained model) the {R,S} shape
is unreachable, so the empty set is the OPERATIVE abstention (test_serving_empty_set_*).
Mondrian stratification computes the calibration quantile PER lineage group; the per-group
quantiles drive the coverage EVALUATION (`empirical_coverage`), which reports that the
guarantee holds across lineages and not just marginally — the set widens where a lineage's
calibration support is thin. A genome served live has NO assigned lineage group (an uploaded
FASTA is not localized into the training Mash clustering), so the serving path (`train.predict`
→ `predict_set(p)`) uses the GLOBAL marginal quantile — still a distribution-free ≥1−α set.
Pure split conformal: no model internals, no retraining, just held-out calibration scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import CALL_NO_CALL, CALL_RESISTANT, CALL_SUSCEPTIBLE

CONFORMAL_ALPHA = 0.1  # target miscoverage → ≥90% coverage guarantee
MIN_GROUP_CALIB = (
    10  # below this many calibration points, fall back to the global quantile
)
# Empty set → strongest abstention (neither label clears 1−α), NOT a novelty assertion. The value
# string is kept for schema / demo_genomes.json / preflight compatibility (see set_to_verdict).
VERDICT_OOD = "ood_novel"

_R, _S = "R", "S"


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile of nonconformity scores (higher interpolation)."""
    n = scores.size
    if n == 0:
        return 1.0  # no calibration data → include everything (widest, safest set)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


@dataclass(frozen=True, slots=True)
class ConformalModel:
    """Calibrated conformal thresholds: one global quantile + per-group (Mondrian) quantiles."""

    alpha: float
    global_q: float
    group_q: dict[object, float] = field(default_factory=dict)
    min_group_calib: int = MIN_GROUP_CALIB

    def _q_for(self, group: object) -> float:
        return self.group_q.get(group, self.global_q)

    def predict_set(self, p_resistant: float, group: object = None) -> frozenset[str]:
        """Prediction set ⊆ {'R','S'}: include a class when its nonconformity ≤ the group quantile."""
        q = self._q_for(group)
        nonconf = {_R: 1.0 - p_resistant, _S: p_resistant}  # 1 − p(class)
        return frozenset(c for c, s in nonconf.items() if s <= q)


def fit(
    p_resistant: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray | None = None,
    alpha: float = CONFORMAL_ALPHA,
    min_group_calib: int = MIN_GROUP_CALIB,
) -> ConformalModel:
    """Calibrate from held-out predictions. `p_resistant`=P(R), `y_true`∈{0,1}, `groups` optional.

    Nonconformity of a calibration point is 1 − p(true class). Mondrian: a per-group quantile
    is stored only where the group has ≥ min_group_calib points (else it uses the global one).
    """
    p_resistant = np.asarray(p_resistant, dtype=float)
    y_true = np.asarray(y_true)
    p_true = np.where(y_true == 1, p_resistant, 1.0 - p_resistant)
    scores = 1.0 - p_true

    global_q = _conformal_quantile(scores, alpha)
    group_q: dict[object, float] = {}
    if groups is not None:
        groups = np.asarray(groups)
        for g in np.unique(groups):
            mask = groups == g
            if int(mask.sum()) >= min_group_calib:
                group_q[g] = _conformal_quantile(scores[mask], alpha)
    return ConformalModel(
        alpha=alpha, global_q=global_q, group_q=group_q, min_group_calib=min_group_calib
    )


def set_to_verdict(pred_set: frozenset[str]) -> str:
    """Map a prediction set to a human verdict (constants.CALL_* or VERDICT_OOD)."""
    if pred_set == frozenset({_R}):
        return CALL_RESISTANT
    if pred_set == frozenset({_S}):
        return CALL_SUSCEPTIBLE
    if pred_set == frozenset({_R, _S}):
        return CALL_NO_CALL  # both plausible → abstain
    return VERDICT_OOD  # empty → neither label clears the coverage bar (strongest abstention)


def empirical_coverage(
    model: ConformalModel,
    p_resistant: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray | None = None,
) -> dict[str, float]:
    """Coverage + set-size distribution on a test set (the money-slide numbers)."""
    p_resistant = np.asarray(p_resistant, dtype=float)
    y_true = np.asarray(y_true)
    grp = np.asarray(groups) if groups is not None else np.full(len(y_true), None)

    covered = 0
    sizes = {0: 0, 1: 0, 2: 0}
    for p, y, g in zip(p_resistant, y_true, grp, strict=True):
        s = model.predict_set(float(p), g)
        sizes[len(s)] += 1
        true_label = _R if y == 1 else _S
        covered += true_label in s
    n = len(y_true)
    return {
        "coverage": covered / n if n else float("nan"),
        "target_coverage": 1 - model.alpha,
        "frac_singleton": sizes[1] / n if n else float("nan"),  # confident calls
        "frac_no_call": sizes[2] / n if n else float("nan"),  # {R,S}
        "frac_ood": sizes[0] / n if n else float("nan"),  # {}
        "n": float(n),
    }
