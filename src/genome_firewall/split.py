"""Cross-validation splitters that separate mechanism-learning from lineage-memorization.

An AMR classifier can look excellent for the wrong reason: bacterial genomes cluster by
clade, resistance co-travels with clade, so a model that merely recognizes the lineage scores
high on a random split while having learned no resistance *mechanism*. That is the "collapse
slide" (ADR-0005): random StratifiedKFold reports an inflated ~95%, and the honest
grouped/leave-one-clade-out splits collapse it toward the true generalization number.

This module is pure code — deterministic splits (fixed seed) plus a leakage probe on the Mash
distance frame. No model, no unseeded RNG.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import pandas as pd

# One fixed seed for every splitter here so a rerun reproduces identical folds; 1729 (the
# Hardy-Ramanujan taxicab number) is an arbitrary constant, only its fixedness matters.
RANDOM_SEED = 1729
DEFAULT_N_SPLITS = 5
# Leave-one-clade-out needs both classes in train to train a binary model, and a test fold
# large enough to estimate a rate — below these it yields no usable signal, so we skip.
MIN_TRAIN_CLASSES = 2
DEFAULT_MIN_TEST = 5
# `dist` is a tidy all-vs-all Mash frame: two genome-id columns plus the distance.
_DIST_A, _DIST_B, _DIST = "a", "b", "dist"


def random_folds(
    y: Sequence[object] | np.ndarray, n_splits: int = DEFAULT_N_SPLITS
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified K-fold that IGNORES lineage — the dishonest, score-inflating baseline.

    Near-identical clones of one clade land on both sides of the split, so the model is
    graded partly on genomes it has effectively already seen. Kept precisely to quantify that
    inflation against the grouped split.
    """
    y_arr = np.asarray(y)
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED
    )
    return list(splitter.split(np.zeros(len(y_arr)), y_arr))


def grouped_folds(
    y: Sequence[object] | np.ndarray,
    groups: Sequence[object] | np.ndarray,
    n_splits: int = DEFAULT_N_SPLITS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """StratifiedGroupKFold: no lineage group straddles train/test — the HONEST split.

    `groups` is the lineage key (serovar / 7-gene MLST / coarse Mash cluster id). Keeping a
    whole clade on one side forces the model to generalize to *unseen* lineages, which is the
    number that matters clinically. If distinct groups < n_splits we cannot form that many
    disjoint group-holdouts, so we reduce n_splits to the group count rather than crash.
    """
    y_arr = np.asarray(y)
    groups_arr = np.asarray(groups)
    n_groups = len(np.unique(groups_arr))
    effective_splits = min(n_splits, n_groups)
    splitter = StratifiedGroupKFold(
        n_splits=effective_splits, shuffle=True, random_state=RANDOM_SEED
    )
    return list(splitter.split(np.zeros(len(y_arr)), y_arr, groups_arr))


def leave_one_clade_out(
    y: Sequence[object] | np.ndarray,
    groups: Sequence[object] | np.ndarray,
    min_test: int = DEFAULT_MIN_TEST,
) -> Iterator[tuple[np.ndarray, np.ndarray, object]]:
    """Yield (train_idx, test_idx, held_out_group), holding out one whole lineage as test.

    The strongest generalization probe: each clade is scored by a model that never saw it.
    Groups too small to estimate a rate (test < min_test) or that leave the training set
    single-class are skipped — a degenerate fold measures nothing.
    """
    y_arr = np.asarray(y)
    groups_arr = np.asarray(groups)
    all_idx = np.arange(len(y_arr))
    # Sort for a deterministic yield order independent of input row order.
    for held_out in np.unique(groups_arr):
        test_mask = groups_arr == held_out
        test_idx = all_idx[test_mask]
        train_idx = all_idx[~test_mask]
        if len(test_idx) < min_test:
            continue
        if len(np.unique(y_arr[train_idx])) < MIN_TRAIN_CLASSES:
            continue
        # Hand back a plain Python scalar, not a numpy box, so callers can key dicts on it.
        label = held_out.item() if isinstance(held_out, np.generic) else held_out
        yield train_idx, test_idx, label


def min_cross_distance(
    train_ids: list[str], test_ids: list[str], dist: pd.DataFrame
) -> float:
    """Minimum train↔test Mash distance — the leakage check for one split.

    A tiny value means a near-clone spans the split (the honest grouping failed to separate a
    sister sublineage), so a "grouped" score is still partly memorization. `dist` is the tidy
    [a, b, dist] frame; it may be symmetric or upper-triangular, so we test both column
    orientations. Returns NaN when the two sides share no measured pair.
    """
    train_set, test_set = set(train_ids), set(test_ids)
    a, b = dist[_DIST_A].to_numpy(), dist[_DIST_B].to_numpy()
    # A cross pair is any row with one endpoint in train and the other in test, either way.
    a_in_train = np.fromiter((x in train_set for x in a), dtype=bool, count=len(a))
    b_in_test = np.fromiter((x in test_set for x in b), dtype=bool, count=len(b))
    a_in_test = np.fromiter((x in test_set for x in a), dtype=bool, count=len(a))
    b_in_train = np.fromiter((x in train_set for x in b), dtype=bool, count=len(b))
    cross = (a_in_train & b_in_test) | (a_in_test & b_in_train)
    if not cross.any():
        return float("nan")
    return float(dist[_DIST].to_numpy()[cross].min())
