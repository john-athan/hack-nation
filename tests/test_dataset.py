"""Tests for the modeling-table assembly (two blocks + labels + grouping)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from genome_firewall.constants import LINEAGE_PREFIX, MECH_PREFIX
from genome_firewall.dataset import build_dataset


def _cohort() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "genome_id": ["g1", "g2", "g3"],
            "serovar": ["Typhimurium", "Typhimurium", "Newport"],
            "mlst": ["MLST.19", "MLST.19", "MLST.45"],
        }
    )


def _determinants() -> pd.DataFrame:
    # g1 has a beta-lactamase, g2 a POINT mutation, g3 has NO determinants (→ all-zero row).
    return pd.DataFrame(
        {
            "genome_id": ["g1", "g2"],
            "symbol": ["blaTEM-1", "gyrA_S83F"],
            "subtype": ["AMR", "POINT"],
            "drug_class": ["BETA-LACTAM", "QUINOLONE"],
            "subclass": ["BETA-LACTAM", "QUINOLONE"],
        }
    )


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "genome_id": ["g1", "g2", "g3"],
            "antibiotic": ["ampicillin", "ampicillin", "ciprofloxacin"],
            "label": ["Resistant", "Susceptible", "Resistant"],
        }
    )


def test_blocks_labels_and_grouping() -> None:
    ds = build_dataset(_cohort(), _determinants(), _labels())
    assert ds.genome_ids == ["g1", "g2", "g3"]

    # Mechanism block: g3 (no determinants) is a real all-zero row, not missing.
    assert all(c.startswith(MECH_PREFIX) for c in ds.x_mech.columns)
    assert ds.x_mech.loc["g3"].sum() == 0
    assert ds.x_mech.loc["g1", f"{MECH_PREFIX}blaTEM-1"] == 1

    # Lineage block one-hot, namespaced.
    assert all(c.startswith(LINEAGE_PREFIX) for c in ds.x_lineage.columns)

    # Labels encoded {1,0,NaN}; ampicillin untested for g3 → NaN.
    assert ds.y.loc["g1", "ampicillin"] == 1
    assert ds.y.loc["g2", "ampicillin"] == 0
    assert np.isnan(ds.y.loc["g3", "ampicillin"])

    # Grouping falls back to serovar factorization (Typhimurium shares a cluster).
    assert ds.meta.loc["g1", "cluster"] == ds.meta.loc["g2", "cluster"]
    assert ds.meta.loc["g1", "cluster"] != ds.meta.loc["g3", "cluster"]


def test_drug_xy_drops_untested_and_returns_groups() -> None:
    ds = build_dataset(_cohort(), _determinants(), _labels())
    x, y, groups = ds.drug_xy("ampicillin", block="mech")
    assert list(x.index) == ["g1", "g2"]  # g3 (NaN) dropped
    assert list(y) == [1, 0]
    assert len(groups) == 2


def test_clusters_override_serovar() -> None:
    ds = build_dataset(
        _cohort(), _determinants(), _labels(), clusters={"g1": 7, "g2": 8, "g3": 7}
    )
    assert ds.meta.loc["g1", "cluster"] == 7
    assert ds.meta.loc["g3", "cluster"] == 7
