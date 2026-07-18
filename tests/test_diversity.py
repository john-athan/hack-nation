"""Lock the diversity panel's seams: lineage load degrades to None on any bad artifact, and the
phenotype spread reads straight from results.csv columns. The app renders both unguarded, so a
fresh checkout (no baked JSON) must yield an empty-state, never an exception.

Also pins the bake summary shape so a schema drift between bake and reader is caught."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from diversity import load_diversity, phenotype_spread  # noqa: E402  (demo-local module)

_GOOD = {
    "n_genomes": 5402,
    "n_serovars_named": 174,
    "n_mlst": 320,
    "top_serovar": "Typhimurium",
    "top_serovar_share": 0.1468,
    "top_mlst_share": 0.1683,
}


def test_missing_file_returns_none(tmp_path: Path) -> None:
    assert load_diversity(tmp_path / "nope.json") is None


def test_good_file_loads(tmp_path: Path) -> None:
    p = tmp_path / "diversity.json"
    p.write_text(json.dumps(_GOOD))
    assert load_diversity(p) == _GOOD


def test_corrupt_file_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "diversity.json"
    p.write_text("{not json")
    assert load_diversity(p) is None


def test_partial_dict_is_rejected(tmp_path: Path) -> None:
    # A half-summary would let the panel render a claim it can't back — reject it whole.
    p = tmp_path / "diversity.json"
    p.write_text(json.dumps({"n_genomes": 5402}))
    assert load_diversity(p) is None


def test_phenotype_spread_from_results() -> None:
    df = pd.DataFrame(
        {
            "drug": ["azithromycin", "tetracycline", "streptomycin"],
            "status": ["ok", "ok", "no_call_single_class"],
            "n": [1849, 1602, 688],
            "n_resistant": [84, 1077, 688],
        }
    )
    spread = phenotype_spread(df)
    assert spread is not None
    # only the two 'ok' drugs count; streptomycin (single-class) is excluded
    assert spread["n_drugs"] == 2
    assert spread["min_drug"] == "azithromycin"
    assert spread["max_drug"] == "tetracycline"
    assert round(spread["min_frac"], 3) == round(84 / 1849, 3)
    assert round(spread["max_frac"], 3) == round(1077 / 1602, 3)


def test_phenotype_spread_no_ok_drugs_returns_none() -> None:
    df = pd.DataFrame(
        {
            "drug": ["x"],
            "status": ["no_call_single_class"],
            "n": [10],
            "n_resistant": [10],
        }
    )
    assert phenotype_spread(df) is None


def test_phenotype_spread_missing_columns_returns_none() -> None:
    assert phenotype_spread(pd.DataFrame({"drug": ["x"]})) is None


def test_bake_summary_shape() -> None:
    # scripts/ is on sys.path at runtime (top of file); ty can't see it statically.
    from bake_diversity import build_summary  # noqa: PLC0415  # ty: ignore[unresolved-import]

    cohort = pd.DataFrame(
        {
            "serovar": ["Typhimurium", "Typhimurium", "Enteritidis", "__unknown__"],
            "mlst": ["ST1", "ST19", "ST11", "ST1"],
        }
    )
    summary = build_summary(cohort)
    assert summary["n_genomes"] == 4
    # __unknown__ is not a real clade, so it is excluded from the named-serovar count
    assert summary["n_serovars_named"] == 2
    assert summary["n_mlst"] == 3
    assert summary["top_serovar"] == "Typhimurium"
    # share is of the FULL cohort denominator (2 of 4), not named-only
    assert summary["top_serovar_share"] == 0.5
