"""Bake the README hero: one 2-panel figure from the frozen artifacts, nothing hand-drawn.

Left  — honest evaluation: per-drug balanced accuracy on a random split (inflated, leaks lineage)
        vs the honest leave-clade-out split, sorted by collapse. The point is the GAP: azithromycin
        craters while gene-driven drugs hold. Read straight from data/results.csv.
Right — calibration: the pooled reliability curve on the honest split, read from
        docs/assets/reliability.csv (baked by scripts/reliability.py). ECE is recomputed from those
        same bins, so every number on the figure traces back to a committed artifact.

Deliberately makes NO claim that the model out-predicts the known-gene baseline (it often does not);
the supported story is honest evaluation + gene-driven hold + calibrated confidence.

Run: uv run --with matplotlib python scripts/make_hero.py   →   docs/assets/hero.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_ASSETS = Path("docs/assets")
_RESULTS = Path("data/results.csv")
_REL_CSV = _ASSETS / "reliability.csv"
_HERO = _ASSETS / "hero.png"

# Same evaluability gate as the demo's collapse slide (demo/collapse.py): discriminating on the
# random split, and enough isolates for a stable leave-clade-out estimate. Keeps the panel honest.
_MIN_RANDOM_BAL_ACC = 0.55
_MIN_EVAL_N = 300

_RED = "#c0392b"  # random split (the inflated, dishonest number)
_GREEN = "#0b6d3b"  # honest leave-clade-out split


def _collapse_frame() -> pd.DataFrame:
    res = pd.read_csv(_RESULTS)
    ok = res[res["status"] == "ok"].dropna(subset=["random_bal_acc", "grouped_bal_acc"])
    ok = ok[(ok["random_bal_acc"] >= _MIN_RANDOM_BAL_ACC) & (ok["n"] >= _MIN_EVAL_N)].copy()
    ok["collapse"] = ok["random_bal_acc"] - ok["grouped_bal_acc"]
    return ok.sort_values("collapse", ascending=False).reset_index(drop=True)


def _ece(rel: pd.DataFrame) -> float:
    w = rel["n"] / rel["n"].sum()
    return float((w * (rel["observed_frequency"] - rel["mean_predicted"]).abs()).sum())


def main() -> int:
    ok = _collapse_frame()
    rel = pd.read_csv(_REL_CSV)
    ece = _ece(rel)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(12.0, 4.6), gridspec_kw={"width_ratios": [1.6, 1.0]}
    )
    fig.patch.set_facecolor("white")

    # --- Left: random vs honest balanced accuracy, per drug -------------------
    y = np.arange(len(ok))
    h = 0.38
    axL.barh(y - h / 2, ok["random_bal_acc"], height=h, color=_RED, label="random split (leaks lineage)")
    axL.barh(
        y + h / 2, ok["grouped_bal_acc"], height=h, color=_GREEN, label="leave-clade-out"
    )
    axL.set_yticks(y)
    axL.set_yticklabels(ok["drug"], fontsize=8)
    axL.invert_yaxis()
    axL.axvline(0.5, color="#999", ls=":", lw=1)
    axL.text(0.5, len(ok) - 0.3, "chance", color="#999", fontsize=7, ha="center", va="top")
    axL.set_xlim(0.4, 1.0)
    axL.set_xlabel("Balanced accuracy")
    axL.set_title(
        "Random split leaks lineage; leave-clade-out doesn't",
        fontsize=11,
        fontweight="bold",
    )
    axL.legend(loc="lower right", fontsize=8, framealpha=0.9)

    # Draw the eye to the biggest collapse (it leads the sort) by reddening its tick label — no
    # floating text to overlap the bars; the numeric callout lives in the README caption instead.
    if not ok.empty:
        lead = ok.iloc[0]["drug"]
        for lbl in axL.get_yticklabels():
            if lbl.get_text() == lead:
                lbl.set_color(_RED)
                lbl.set_fontweight("bold")

    # --- Right: reliability / calibration on the honest split -----------------
    axR.plot([0, 1], [0, 1], "--", color="#999", lw=1, label="perfectly calibrated")
    axR.plot(
        rel["mean_predicted"],
        rel["observed_frequency"],
        "o-",
        color=_GREEN,
        label="Genome Firewall",
    )
    axR.set_xlim(0, 1)
    axR.set_ylim(0, 1)
    axR.set_xlabel("Predicted P(resistant)")
    axR.set_ylabel("Observed resistant frequency")
    axR.set_title(
        f"Calibrated on the leave-clade-out split (ECE {ece:.3f})",
        fontsize=11,
        fontweight="bold",
    )
    axR.legend(loc="upper left", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(_HERO, dpi=140, facecolor="white")
    print(f"[hero] wrote {_HERO}  (drugs={len(ok)}, ECE={ece:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
