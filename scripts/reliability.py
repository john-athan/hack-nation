"""Reliability diagram (calibration curve) on the HONEST leave-clade-out split.

The Success Criteria (challenge brief, page 5) ask for a Brier score AND a reliability plot as the
"confidence quality" evidence. data/results.csv already carries the per-drug Brier; this bakes the
visual: pooled predicted P(resistant) vs the observed resistant frequency on the grouped
out-of-fold split — the same leak-free OOF predictions data/results.csv is built from (fixed seeds,
so the reconstruction is deterministic and reproduces the published numbers).

Post-hoc over frozen artifacts: it re-fits the ephemeral per-fold LRs the OOF loop always builds and
discards; it never retrains or touches data/models.joblib or data/results.csv. Only therapeutic
drugs are pooled — the ones the tool actually returns a "works / fails" call for.

Run: uv run --with matplotlib python scripts/reliability.py
Writes docs/assets/reliability.{csv,png}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from genome_firewall import evaluate, model
from genome_firewall.amrfinder import parse_tsv
from genome_firewall.constants import AMRFINDER_DIR, COHORT_CSV, LABELS_CSV
from genome_firewall.dataset import build_dataset
from genome_firewall.drugs import DRUG_DB
from genome_firewall.labels import canonical_drug
from genome_firewall.mic import rederive

_ASSETS = Path("docs/assets")
_PNG = _ASSETS / "reliability.png"
_CSV = _ASSETS / "reliability.csv"
# Ten quantile bins: equal-count strata are robust to the heavy mass of low P(R) predictions that
# uniform bins would leave nearly empty, so each point on the curve rests on a comparable sample.
_N_BINS = 10


def _load_dataset() -> tuple[object, pd.DataFrame]:
    """Reproduce run_evaluation.py's loader → the same Dataset the published table was built from."""
    cohort = pd.read_csv(COHORT_CSV, dtype=str)
    genome_ids = set(cohort["genome_id"].astype(str))
    raw = pd.read_csv(LABELS_CSV, dtype=str)
    raw["antibiotic"] = raw["antibiotic"].map(canonical_drug)
    mic = rederive(raw)
    mic = mic[mic["label"].isin({"Resistant", "Susceptible"})]
    frames = [
        parse_tsv(t)
        for t in sorted(AMRFINDER_DIR.glob("*.tsv"))
        if t.stem in genome_ids
    ]
    determinants = pd.concat(frames, ignore_index=True)
    annotated = set(determinants["genome_id"].astype(str))
    cohort = cohort[cohort["genome_id"].astype(str).isin(annotated)]
    return build_dataset(cohort, determinants, mic), determinants


def _pool_honest_predictions(ds: object) -> tuple[np.ndarray, np.ndarray]:
    """Pooled (P(resistant), true label) over the grouped OOF split, therapeutic drugs only."""
    all_p: list[float] = []
    all_y: list[int] = []
    for drug in DRUG_DB:
        if drug in evaluate.NON_THERAPEUTIC:
            continue
        try:
            x, y, groups = ds.drug_xy(drug)
        except KeyError:
            continue
        if len(set(y)) < 2 or int((y == 1).sum()) < model.MIN_POSITIVES:
            continue
        # determinants=None → skip the known-gene baseline; we only need the model's P(R).
        oof = evaluate._grouped_oof(ds, x, y, groups, drug, None, None)
        if oof is None:
            continue
        all_p.extend(oof.model_p.tolist())
        all_y.extend(oof.y_true.tolist())
    return np.asarray(all_p, dtype=float), np.asarray(all_y, dtype=int)


def _calibration_table(p: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Per-bin mean predicted P(R), observed resistant frequency, and count (quantile bins)."""
    edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, _N_BINS + 1)))
    # Interior edges only for digitize; clip so the max value lands in the last bin, not out of range.
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    rows: list[dict[str, float]] = []
    for b in range(len(edges) - 1):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": int(b),
                "mean_predicted": round(float(p[mask].mean()), 4),
                "observed_frequency": round(float(y[mask].mean()), 4),
                "n": int(mask.sum()),
            }
        )
    return pd.DataFrame.from_records(rows)


def main() -> int:
    ds, _ = _load_dataset()
    p, y = _pool_honest_predictions(ds)
    if len(p) == 0:
        print("[rel] no pooled predictions — nothing to plot", file=sys.stderr)
        return 1
    cal = _calibration_table(p, y)
    brier = float(np.mean((p - y) ** 2))
    # Expected Calibration Error: sample-weighted mean gap between confidence and accuracy.
    ece = float(
        np.sum(cal["n"] / cal["n"].sum() * (cal["observed_frequency"] - cal["mean_predicted"]).abs())
    )
    print(
        f"[rel] pooled {len(p)} honest-split predictions; Brier={brier:.4f} ECE={ece:.4f}",
        file=sys.stderr,
    )

    _ASSETS.mkdir(parents=True, exist_ok=True)
    cal.to_csv(_CSV, index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], "--", color="#888", label="perfectly calibrated")
    ax.plot(
        cal["mean_predicted"],
        cal["observed_frequency"],
        "o-",
        color="#0b6d3b",
        label="Genome Firewall (leave-clade-out split)",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted P(resistant)")
    ax.set_ylabel("Observed resistant frequency")
    ax.set_title(
        f"Reliability diagram — leave-clade-out split\nBrier={brier:.3f}  ECE={ece:.3f}  "
        f"(n={len(p):,})"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(_PNG, dpi=130)
    print(f"[rel] wrote {_PNG} and {_CSV}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
