"""Smoke tests for the honest-core orchestrator on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from genome_firewall import evaluate
from genome_firewall.constants import MECH_PREFIX
from genome_firewall.dataset import Dataset


def _synthetic_dataset(n: int = 200) -> Dataset:
    rng = np.random.default_rng(0)
    gids = [f"g{i}" for i in range(n)]
    y_amp = rng.integers(0, 2, n)
    x_mech = pd.DataFrame(
        {f"{MECH_PREFIX}driver": y_amp, f"{MECH_PREFIX}noise": rng.integers(0, 2, n)},
        index=gids,
    )
    x_lin = pd.DataFrame({"lin__serovar_A": np.ones(n, dtype="int8")}, index=gids)
    y = pd.DataFrame({"ampicillin": y_amp.astype(float)}, index=gids)
    meta = pd.DataFrame(
        {"serovar": ["A"] * n, "mlst": ["m"] * n, "cluster": rng.integers(0, 10, n)},
        index=gids,
    )
    return Dataset(genome_ids=gids, x_mech=x_mech, x_lineage=x_lin, y=y, meta=meta)


def test_evaluate_drug_produces_collapse_and_conformal() -> None:
    ds = _synthetic_dataset()
    res = evaluate.evaluate_drug(ds, "ampicillin")
    assert res["status"] == "ok"
    assert "random" in res and "grouped" in res
    assert "conformal" in res
    # A perfectly-separable synthetic driver → both splits score well; structure is what we assert.
    assert 0.0 <= res["conformal"]["coverage"] <= 1.0  # ty: ignore[not-subscriptable]  # deep-index into dict[str, object]


def test_evaluate_drug_insufficient_positives_is_no_call() -> None:
    ds = _synthetic_dataset(n=40)
    # Force almost all-susceptible → below the positive floor.
    ds.y["ampicillin"] = 0.0
    ds.y.iloc[:2, ds.y.columns.get_loc("ampicillin")] = 1.0
    res = evaluate.evaluate_drug(ds, "ampicillin")
    assert res["status"] == "no_call_insufficient_positives"


def test_evaluate_all_surfaces_brier_for_both_splits() -> None:
    """The collapse table's calibration columns: binary_metrics computes Brier, evaluate_all must
    keep it (it was computed then dropped before results.csv). A valid Brier is a probability MSE
    in [0, 1]. Locks the extraction so a refactor can't silently drop the calibration story."""
    ds = _synthetic_dataset()
    table = evaluate.evaluate_all(ds)
    assert {"random_brier", "grouped_brier"} <= set(table.columns)
    amp = table[table["drug"] == "ampicillin"].iloc[0]
    for col in ("random_brier", "grouped_brier"):
        assert 0.0 <= float(amp[col]) <= 1.0


def test_non_therapeutic_flag() -> None:
    ds = _synthetic_dataset(n=20)
    assert evaluate.evaluate_drug(ds, "cefoxitin")["therapeutic"] is False
    assert evaluate.evaluate_drug(ds, "ampicillin")["therapeutic"] is True
