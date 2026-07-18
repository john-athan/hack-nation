"""Tests for model training, persistence, and new-genome vectorization."""

from __future__ import annotations

from dataclasses import replace
from inspect import signature
from pathlib import Path

import numpy as np
import pandas as pd

from genome_firewall import train
from genome_firewall.constants import MECH_PREFIX
from genome_firewall.dataset import Dataset


def _synthetic_dataset(n: int = 200) -> Dataset:
    rng = np.random.default_rng(0)
    gids = [f"g{i}" for i in range(n)]
    y_amp = rng.integers(0, 2, n)
    x_mech = pd.DataFrame(
        {f"{MECH_PREFIX}blaTEM-1": y_amp, f"{MECH_PREFIX}noise": rng.integers(0, 2, n)},
        index=gids,
    )
    x_lin = pd.DataFrame({"lin__serovar_A": np.ones(n, dtype="int8")}, index=gids)
    y = pd.DataFrame({"ampicillin": y_amp.astype(float)}, index=gids)
    meta = pd.DataFrame(
        {"serovar": ["A"] * n, "mlst": ["m"] * n, "cluster": rng.integers(0, 10, n)},
        index=gids,
    )
    return Dataset(genome_ids=gids, x_mech=x_mech, x_lineage=x_lin, y=y, meta=meta)


def test_train_drug_and_predict() -> None:
    ds = _synthetic_dataset()
    td = train.train_drug(ds, "ampicillin")
    assert td is not None
    assert td.feature_columns == [f"{MECH_PREFIX}blaTEM-1", f"{MECH_PREFIX}noise"]
    # A genome carrying the driver gene should score higher than one without.
    p_res, _ = train.predict(td, {"blaTEM-1"})
    p_sus, _ = train.predict(td, set())
    assert p_res > p_sus


def test_serving_uses_global_marginal_quantile_not_mondrian() -> None:
    # Honest serving contract (ADR-0007): a served genome has no assigned lineage group, so
    # predict() uses the GLOBAL marginal conformal quantile. A per-lineage (Mondrian) quantile
    # would need localization we deliberately don't do at inference. Locked so a silent change
    # (threading a group → different demo verdicts) fails pytest, not the stage.
    # Airtight lock: serving takes NO lineage input, so there is no group to honestly pass. A
    # "fix" that adds a group param (the likely regression after reading ADR-0007) fails here.
    assert list(signature(train.predict).parameters) == ["td", "determinant_symbols"]

    ds = _synthetic_dataset()
    td = train.train_drug(ds, "ampicillin")
    assert td is not None
    p_res, served_set = train.predict(td, {"blaTEM-1"})
    assert served_set == td.conformal.predict_set(
        p_res
    )  # == the global (group=None) set

    # The choice is not vacuous: an injected thin lineage's quantile yields a DIFFERENT set, so
    # serving via the global quantile is a real decision, not an accident of equal quantiles.
    cm = replace(
        td.conformal,
        group_q={**td.conformal.group_q, "thin_lineage": p_res + 1e-3},
    )
    assert cm.predict_set(p_res, group="thin_lineage") != served_set


def test_vectorize_aligns_and_ignores_novel() -> None:
    cols = [f"{MECH_PREFIX}blaTEM-1", f"{MECH_PREFIX}gyrA_S83F"]
    row = train.vectorize_genome({"blaTEM-1", "novelGene999"}, cols)
    assert list(row.columns) == cols
    assert row.iloc[0][f"{MECH_PREFIX}blaTEM-1"] == 1
    assert (
        row.iloc[0][f"{MECH_PREFIX}gyrA_S83F"] == 0
    )  # absent → 0; novelGene999 ignored


def test_save_load_roundtrip(tmp_path: Path) -> None:
    ds = _synthetic_dataset()
    models = train.train_all(ds)
    assert "ampicillin" in models
    path = train.save(models, tmp_path / "m.joblib")
    loaded = train.load(path)
    assert set(loaded) == set(models)
    p1, _ = train.predict(models["ampicillin"], {"blaTEM-1"})
    p2, _ = train.predict(loaded["ampicillin"], {"blaTEM-1"})
    assert p1 == p2
