"""Pure seams for the cohort-diversity panel, kept Streamlit-free so app.py wires them and a test
locks the contracts.

Guards against the data-bias trap with two committed sources: lineage spread from
the baked data/diversity.json (scripts/bake_diversity.py), and per-drug phenotype spread computed
live from the committed results.csv. Both degrade to None (never raise) when their artifact is
absent, so a fresh clone renders an empty-state instead of crashing."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DIVERSITY_JSON = _DATA_DIR / "diversity.json"

_REQUIRED_KEYS = (
    "n_genomes",
    "n_serovars_named",
    "n_mlst",
    "top_serovar",
    "top_serovar_share",
    "top_mlst_share",
)


def load_diversity(path: Path | str | None = None) -> dict | None:
    """Load the baked lineage summary, or None if it is missing/corrupt/incomplete.

    Never raises: a missing file is a legitimate fresh-clone state, and a hand-corrupted file is an
    optional-feature absence, not a demo-killer. A partial dict is rejected whole (returning it
    would let the panel render a half-claim)."""
    p = Path(path) if path is not None else DIVERSITY_JSON
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict) or any(k not in data for k in _REQUIRED_KEYS):
        return None
    return data


def phenotype_spread(results: pd.DataFrame) -> dict | None:
    """Per-drug resistant-fraction spread over the trained (status=='ok') drugs, or None if none.

    The honest answer to "is your data one phenotype?": resistance prevalence runs from the least-
    to most-resistant drug. Uses n_resistant/n straight from the committed results.csv, so it can
    never drift from the reported numbers."""
    if results is None or not {"status", "n", "n_resistant", "drug"} <= set(
        results.columns
    ):
        return None
    ok = results[(results["status"] == "ok") & (results["n"] > 0)].copy()
    if ok.empty:
        return None
    ok["frac_r"] = ok["n_resistant"] / ok["n"]
    lo = ok.loc[ok["frac_r"].idxmin()]
    hi = ok.loc[ok["frac_r"].idxmax()]
    return {
        "n_drugs": int(len(ok)),
        "min_frac": float(lo["frac_r"]),
        "min_drug": str(lo["drug"]),
        "max_frac": float(hi["frac_r"]),
        "max_drug": str(hi["drug"]),
    }
