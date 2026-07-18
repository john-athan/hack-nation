"""Bake the cohort's lineage-diversity summary into a committed artifact.

Why a bake step: the diversity panel answers the challenge author's data-bias concern ("if all
your genomes have the same AMR phenotype / lineage, your data is heavily biased") with real
serovar / MLST spread. That spread lives in `data/cohort.csv`, which is a gitignored box-local
build artifact, so it is absent on a fresh clone and on the Streamlit Cloud deploy. This script
renders the handful of summary counts the panel needs into `data/diversity.json`, which IS
committed, exactly the way scripts/coverage_novelty.py bakes its figure into docs/assets/. The
phenotype half of the panel comes from the already-committed results.csv, so only lineage needs
baking.

Fails LOUDLY when cohort.csv is missing rather than emitting an empty/partial summary: a baked
artifact that silently claimed "0 serovars" would be a fabrication on "The Honest One".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genome_firewall.constants import COHORT_CSV, DATA_DIR  # noqa: E402

# BV-BRC writes this literal when its serovar predictor abstains; it is not a real clade, so it is
# excluded from the named-serovar count (counting it would inflate diversity with a non-answer).
UNKNOWN_SEROVAR = "__unknown__"
DIVERSITY_JSON = DATA_DIR / "diversity.json"


def build_summary(cohort: pd.DataFrame) -> dict[str, object]:
    """Reduce the per-genome cohort to the few counts the panel renders. Pure, so a test can pin it."""
    n = len(cohort)
    if n == 0:
        raise ValueError(
            "cohort.csv has no rows; refusing to bake an empty diversity summary"
        )
    named = cohort[cohort["serovar"] != UNKNOWN_SEROVAR]
    top_serovar = named["serovar"].value_counts()
    top_mlst = cohort["mlst"].astype(str).value_counts()
    return {
        "n_genomes": int(n),
        "n_serovars_named": int(named["serovar"].nunique()),
        "n_mlst": int(cohort["mlst"].nunique()),
        "top_serovar": str(top_serovar.index[0]),
        # Shares are of the FULL cohort (the honest denominator): "the single biggest clade is only
        # X% of the data" is the point, so the fraction must not be relative to named-only.
        "top_serovar_share": round(float(top_serovar.iloc[0]) / n, 4),
        "top_mlst_share": round(float(top_mlst.iloc[0]) / n, 4),
    }


def main() -> None:
    if not COHORT_CSV.exists():
        raise SystemExit(
            f"{COHORT_CSV} not found. It is a box-local build artifact; run the cohort build "
            "(scripts/select_cohort.py) before baking the diversity summary."
        )
    summary = build_summary(pd.read_csv(COHORT_CSV))
    DIVERSITY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {DIVERSITY_JSON}: {summary}")


if __name__ == "__main__":
    main()
