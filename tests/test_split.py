"""Hermetic tests for the honest-vs-dishonest CV splitters (synthetic data, no mash/network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from genome_firewall.split import (
    DEFAULT_N_SPLITS,
    grouped_folds,
    leave_one_clade_out,
    min_cross_distance,
    random_folds,
)


def _grouped_dataset() -> tuple[np.ndarray, np.ndarray]:
    # Ten lineages, each nearly single-class (resistance co-travels with clade). Enough rows
    # per group that a group is never split across folds by construction, so any leak is a bug.
    rows_per_group = 6
    n_groups = 10
    y: list[int] = []
    groups: list[int] = []
    for g in range(n_groups):
        label = g % 2  # even clades resistant, odd susceptible — perfectly clade-linked
        y.extend([label] * rows_per_group)
        groups.extend([g] * rows_per_group)
    return np.asarray(y), np.asarray(groups)


def test_grouped_folds_never_split_a_group() -> None:
    y, groups = _grouped_dataset()
    folds = grouped_folds(y, groups, n_splits=DEFAULT_N_SPLITS)
    assert len(folds) == DEFAULT_N_SPLITS
    for train_idx, test_idx in folds:
        train_groups = set(groups[train_idx].tolist())
        test_groups = set(groups[test_idx].tolist())
        # The honest split's whole point: no lineage appears on both sides.
        assert train_groups.isdisjoint(test_groups)


def test_grouped_folds_reduces_when_fewer_groups_than_splits() -> None:
    # Only 3 distinct groups but 5 requested splits — must reduce, not raise.
    y = np.asarray([0, 1, 0, 1, 0, 1])
    groups = np.asarray(["A", "A", "B", "B", "C", "C"])
    folds = grouped_folds(y, groups, n_splits=5)
    assert len(folds) == 3
    for train_idx, test_idx in folds:
        assert set(groups[train_idx].tolist()).isdisjoint(groups[test_idx].tolist())


def test_leave_one_clade_out_holds_out_each_eligible_group() -> None:
    y, groups = _grouped_dataset()
    folds = list(leave_one_clade_out(y, groups, min_test=5))
    # Every group has 6 rows (>= min_test) and the other 9 groups keep both classes present.
    assert len(folds) == 10
    seen_holdouts = []
    for train_idx, test_idx, held_out in folds:
        seen_holdouts.append(held_out)
        assert held_out not in set(groups[train_idx].tolist())
        assert set(groups[test_idx].tolist()) == {held_out}
    assert sorted(seen_holdouts) == list(range(10))


def test_leave_one_clade_out_skips_small_and_single_class() -> None:
    # Group Z has 1 row (< min_test) → skipped. Removing group Y (the only class-1 rows)
    # leaves a single-class train set, so Y is skipped too. Only X yields.
    y = np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0])
    groups = np.asarray(["X", "X", "X", "X", "X", "X", "Y", "Y", "Y", "Y", "Y", "Z"])
    folds = list(leave_one_clade_out(y, groups, min_test=5))
    held = [h for _, _, h in folds]
    assert held == ["X"]


def test_random_folds_cover_all_indices_once_as_test() -> None:
    y = np.asarray([0, 1] * 25)
    folds = random_folds(y, n_splits=DEFAULT_N_SPLITS)
    assert len(folds) == DEFAULT_N_SPLITS
    covered = np.concatenate([test_idx for _, test_idx in folds])
    # Each index is a test index in exactly one fold.
    assert sorted(covered.tolist()) == list(range(len(y)))


def _dist_frame() -> pd.DataFrame:
    rows = [
        ("g1", "g2", 0.001),
        ("g1", "g3", 0.30),
        ("g2", "g3", 0.28),
        ("g3", "g4", 0.002),
        ("g1", "g4", 0.31),
    ]
    return pd.DataFrame(rows, columns=["a", "b", "dist"])


def test_min_cross_distance_finds_minimum_across_split() -> None:
    dist = _dist_frame()
    # train {g1,g2} vs test {g3,g4}: cross pairs are (g1,g3)=.30, (g2,g3)=.28, (g1,g4)=.31.
    assert min_cross_distance(["g1", "g2"], ["g3", "g4"], dist) == 0.28


def test_min_cross_distance_honors_either_column_orientation() -> None:
    dist = _dist_frame()
    # (g3,g4)=0.002 is stored as a=g3,b=g4; with g3 in test and g4 in train it must be found.
    assert min_cross_distance(["g4"], ["g3"], dist) == 0.002


def test_min_cross_distance_nan_when_disjoint() -> None:
    dist = _dist_frame()
    # No measured pair connects these two id sets.
    result = min_cross_distance(["g1"], ["g99"], dist)
    assert np.isnan(result)
