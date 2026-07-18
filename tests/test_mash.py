"""Tests for single-linkage clustering and nearest-neighbor logic (hermetic, no mash call)."""

from __future__ import annotations

import pandas as pd

from genome_firewall.mash import nearest_neighbors, single_linkage_clusters


def _synthetic() -> pd.DataFrame:
    # Four genomes, two tight pairs: {g1,g2} and {g3,g4} are within 0.005; the pairs are far
    # apart. Includes self-pairs (dist 0) as real mash output does, to prove they're ignored.
    rows = [
        ("g1", "g1", 0.0),
        ("g2", "g2", 0.0),
        ("g3", "g3", 0.0),
        ("g4", "g4", 0.0),
        ("g1", "g2", 0.001),
        ("g2", "g1", 0.001),
        ("g3", "g4", 0.002),
        ("g4", "g3", 0.002),
        ("g1", "g3", 0.30),
        ("g3", "g1", 0.30),
        ("g1", "g4", 0.31),
        ("g4", "g1", 0.31),
        ("g2", "g3", 0.29),
        ("g3", "g2", 0.29),
        ("g2", "g4", 0.33),
        ("g4", "g2", 0.33),
    ]
    return pd.DataFrame(rows, columns=["a", "b", "dist"])


def test_single_linkage_forms_two_clusters() -> None:
    clusters = single_linkage_clusters(_synthetic(), threshold=0.005)
    assert set(clusters) == {"g1", "g2", "g3", "g4"}
    assert clusters["g1"] == clusters["g2"]
    assert clusters["g3"] == clusters["g4"]
    assert clusters["g1"] != clusters["g3"]
    assert len(set(clusters.values())) == 2


def test_cluster_ids_are_deterministic_and_dense() -> None:
    clusters = single_linkage_clusters(_synthetic(), threshold=0.005)
    # Cluster ordered by smallest member: {g1,g2}->0, {g3,g4}->1.
    assert clusters == {"g1": 0, "g2": 0, "g3": 1, "g4": 1}


def test_tight_threshold_makes_all_singletons() -> None:
    clusters = single_linkage_clusters(_synthetic(), threshold=0.0)
    assert len(set(clusters.values())) == 4


def test_loose_threshold_merges_everything() -> None:
    clusters = single_linkage_clusters(_synthetic(), threshold=0.5)
    assert len(set(clusters.values())) == 1


def test_nearest_neighbors_excludes_self_and_picks_closest() -> None:
    nn = nearest_neighbors(_synthetic())
    assert set(nn) == {"g1", "g2", "g3", "g4"}
    assert nn["g1"] == ("g2", 0.001)
    assert nn["g2"] == ("g1", 0.001)
    assert nn["g3"] == ("g4", 0.002)
    assert nn["g4"] == ("g3", 0.002)
    # No genome is ever its own nearest neighbor.
    assert all(neighbor != g for g, (neighbor, _) in nn.items())
