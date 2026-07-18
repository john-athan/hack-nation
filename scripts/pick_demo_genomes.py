"""Pick + verify the demo's curated genomes → data/demo_genomes.json.

The 90-second demo has three beats and must NEVER be improvised on stage by scrolling a flat list
of ~5k genome ids. This picks them deterministically from the CURRENT trained models (so they can
never go stale after a retrain), verifies each beat renders, and writes a small preset file the
Streamlit sidebar reads. Regenerate whenever the models are retrained (finalization step).

Beats:
  ① known-gene resistance — the primary ESBL carrier (stable: characterized genes don't move).
  ③ Firewall abstains — a genome the naive model calls "works" (confident green) while the firewall
     won't commit (empty conformal set — neither label clears the coverage bar) and HOLDS.
     Discovered fresh; the starkest, most-robust catch wins.
  ④ gyrA-QRDR knockout — a genome carrying a single gyrA point mutation whose fluoroquinolone
     resistance the knockout probe corroborates (zeroing the QRDR feature moves the call), lab-
     Resistant on the FQ drug. Ground-truth-gated on labels_raw.csv (cycle-41/42 rule).

Run: uv run python scripts/pick_demo_genomes.py [sample_n]
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))

import pandas as pd

from collapse import COLLAPSE_COL, collapse_frame  # type: ignore[import-not-found]
from genome_firewall.atomicio import atomic_write
from genome_firewall.conformal import VERDICT_OOD
from genome_firewall.constants import (
    AMRFINDER_DIR,
    CALL_NO_CALL,
    CALL_RESISTANT,
    COHORT_CSV,
    DATA_DIR,
    EVIDENCE_KNOWN_GENE,
    FASTA_DIR,
)
from genome_firewall.train import MODELS_PATH, load
from pipeline import analyze_fasta  # type: ignore[import-not-found]
from preflight import (  # type: ignore[import-not-found]
    FLUOROQUINOLONES,
    QRDR_MIN_KNOCKOUT_DELTA,
)
from verdict import all_verdicts  # type: ignore[import-not-found]

_PRIMARY = "1079901.3"  # documented ESBL carrier: blaSHV-2A + blaTEM-1 + tet(A)
_QRDR = "54388.377"  # Paratyphi A: single gyrA_S83F QRDR mutation; lab cipro+nalidixic Resistant
_ABSTAIN = frozenset({CALL_NO_CALL, VERDICT_OOD})
_OUT = DATA_DIR / "demo_genomes.json"
_RESULTS_CSV = DATA_DIR / "results.csv"
_LABELS_RAW = DATA_DIR / "labels_raw.csv"
_LAB_RESISTANT = (
    "Resistant"  # the shipped resistant_phenotype value the ground-truth gate requires
)


def _cached_stems(limit: int) -> list[str]:
    stems = sorted(
        p.stem
        for p in AMRFINDER_DIR.glob("*.tsv")
        if p.stat().st_size > 0 and (FASTA_DIR / f"{p.stem}.fna").exists()
    )
    step = max(
        1, len(stems) // limit
    )  # interleave → a truncated sample stays serovar-diverse
    return stems[::step][:limit]


def _serovar_lookup() -> dict[str, str]:
    if not COHORT_CSV.exists():
        return {}
    df = pd.read_csv(COHORT_CSV, usecols=["genome_id", "serovar"], dtype=str)
    return dict(zip(df["genome_id"], df["serovar"].fillna("unknown"), strict=False))


def _verify_primary(models: dict) -> None:
    report, dets = analyze_fasta(FASTA_DIR / f"{_PRIMARY}.fna")
    symbols = set(dets["symbol"].astype(str))
    verdicts = all_verdicts(models, symbols, dets, report)
    contradictions = [
        v.drug
        for v in verdicts
        for p in report.predictions
        if p.antibiotic == v.drug
        and p.call == CALL_RESISTANT
        and v.firewall_verdict != CALL_RESISTANT
    ]
    assert not contradictions, (
        f"demo integrity broken: firewall contradicts report {contradictions}"
    )
    holding = [v.drug for v in verdicts if v.firewall_holding]
    print(
        f"[beat①] {_PRIMARY}: {len(verdicts)} verdicts, holding={holding}, contradictions=[] ✓"
    )


def lab_resistant_drugs(
    labels: pd.DataFrame, genome_id: str, drugs: frozenset[str]
) -> set[str]:
    """Drugs in `drugs` whose SHIPPED lab phenotype is Resistant for `genome_id`.

    The cycle-41/42 ground-truth gate: a curated beat that COMMITS a resistant call must be backed
    by a real lab-Resistant phenotype, never a lineage over-generalization (the exact defect that
    nearly shipped a self-refuting ceftriaxone-R beat in cycle 41). Pure so a unit test can prove a
    Susceptible row never leaks through.
    """
    hit = labels[
        (labels["genome_id"] == genome_id)
        & (labels["resistant_phenotype"] == _LAB_RESISTANT)
    ]
    return set(hit["antibiotic"]) & set(drugs)


def _verify_qrdr(models: dict) -> list[str]:
    """Beat ④ (gyrA-QRDR): a single QRDR point mutation → knockout-corroborated FQ resistance.

    Ground-truth-gated FIRST (a committed FQ-resistant beat must be lab-real), then verified through
    the SAME report/verdict/knockout seams the app renders, so the beat can never go stale after a
    retrain. Returns the lab-Resistant fluoroquinolone drug names (for the preset label)."""
    if not _LABELS_RAW.exists():
        raise SystemExit(
            f"[beat④] {_LABELS_RAW} missing — cannot ground-truth-gate the QRDR beat; refusing to "
            "emit an FQ-resistant beat without lab phenotype (cycle-41/42 rule)."
        )
    labels = pd.read_csv(_LABELS_RAW, dtype=str)
    lab_r = lab_resistant_drugs(labels, _QRDR, FLUOROQUINOLONES)
    assert lab_r, (
        f"[beat④] {_QRDR} has no lab-Resistant fluoroquinolone in labels_raw.csv — refusing to "
        "commit an FQ-resistant beat without ground truth (cycle-41/42 rule)."
    )
    report, dets = analyze_fasta(FASTA_DIR / f"{_QRDR}.fna")
    verdicts = all_verdicts(models, set(dets["symbol"].astype(str)), dets, report)
    contradictions = [
        v.drug
        for v in verdicts
        for p in report.predictions
        if p.antibiotic == v.drug
        and p.call == CALL_RESISTANT
        and v.firewall_verdict != CALL_RESISTANT
    ]
    assert not contradictions, (
        f"[beat④] demo integrity broken: firewall contradicts report {contradictions}"
    )
    driven = [
        v
        for v in verdicts
        if v.drug in lab_r
        and v.firewall_verdict == CALL_RESISTANT
        and v.evidence == EVIDENCE_KNOWN_GENE
        and not math.isnan(v.knockout_delta)
        and v.knockout_delta >= QRDR_MIN_KNOCKOUT_DELTA
    ]
    assert driven, (
        f"[beat④] no knockout-corroborated FQ resistance on {_QRDR} — nothing to show."
    )
    print(
        f"[beat④] {_QRDR}: lab-R FQ {sorted(lab_r)}, knockout-driven "
        f"{[(v.drug, round(v.knockout_delta, 2)) for v in driven]}, contradictions=[] ✓"
    )
    return sorted(v.drug for v in driven)


def _pick_ood(
    models: dict, sample_n: int
) -> tuple[str, list[tuple[str, float]]] | None:
    """The genome with the MOST therapeutic abstain-catches (robust across retrains), then the
    most confident naive 'works' (the starkest catch). Returns (genome_id, [(drug, naive_p), …])."""
    stems = _cached_stems(sample_n)
    print(f"[beat③] scanning {len(stems)} cached genomes for a firewall-abstain catch…")
    catches: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for stem in stems:
        try:
            rep, d = analyze_fasta(FASTA_DIR / f"{stem}.fna")
        except Exception as exc:  # noqa: BLE001 — one bad genome must not abort the scan
            print(f"  skip {stem}: {exc}")
            continue
        syms = set(d["symbol"].astype(str))
        for v in all_verdicts(models, syms, d, rep):
            if (
                v.therapeutic
                and v.firewall_verdict in _ABSTAIN
                and v.naive_call != CALL_RESISTANT
            ):
                catches[stem].append((v.drug, round(v.naive_p, 3)))
    if not catches:
        return None
    # Most catches first; tie-break on the single most-confident naive "works" (lowest P(R)).
    best = min(catches.items(), key=lambda kv: (-len(kv[1]), min(p for _, p in kv[1])))
    return best[0], sorted(best[1], key=lambda dp: dp[1])


def _verify_collapse() -> None:
    """Exercise the collapse money-slide's data path on the FRESH results.csv the finalize just
    wrote — the one app-side load `pick_demo` doesn't otherwise touch — and surface its real
    numbers in the (unattended, 3am) finalize log. The slide is the centerpiece; a blank one is a
    demo failure, so make the actual top collapses visible where the owner reads the run's output."""
    if not _RESULTS_CSV.exists():
        print(
            "[collapse] results.csv missing — collapse slide will show its empty-state ⚠️"
        )
        return
    ok = collapse_frame(pd.read_csv(_RESULTS_CSV))
    if ok is None:
        print(
            "[collapse] no evaluable drugs — money slide is BLANK ⚠️ (check the cohort)"
        )
        return
    top = ", ".join(
        f"{r['drug']} {r['random_bal_acc']:.2f}→{r['grouped_bal_acc']:.2f} "
        f"(Δ{r[COLLAPSE_COL]:.2f})"
        for r in ok.head(3).to_dict("records")
    )
    print(f"[collapse] {len(ok)} drugs on the money slide; top collapses: {top} ✓")


def main() -> None:
    sample_n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    models = load(MODELS_PATH)
    print(f"models: {sorted(models)}")
    serovar = _serovar_lookup()

    _verify_primary(models)
    _verify_collapse()
    qrdr_fq = _verify_qrdr(models)

    genomes = [
        {
            "id": _PRIMARY,
            "beat": "known_gene",
            "label": f"① ESBL carrier — resistance via known genes  ({serovar.get(_PRIMARY, '?')})",
        }
    ]

    picked = _pick_ood(models, sample_n)
    if picked is None:
        print("[beat③] no abstain-catch found in sample — writing beats ①/② only")
    else:
        gid, drugs = picked
        top_drug, top_p = drugs[0]
        print(
            f"[beat③] picked {gid}: {len(drugs)} therapeutic abstain-catches "
            f"({', '.join(f'{dn} {p:.0%}' for dn, p in drugs)})"
        )
        genomes.append(
            {
                "id": gid,
                "beat": "ood",
                "label": (
                    f"③ Firewall abstains — naive says {top_drug} works ({1 - top_p:.0%}), "
                    f"firewall won't commit  ({serovar.get(gid, '?')})"
                ),
            }
        )

    fq = " + ".join(qrdr_fq) if qrdr_fq else "fluoroquinolone"
    genomes.append(
        {
            "id": _QRDR,
            "beat": "qrdr",
            "label": (
                # "traces the call to it", not "confirms": the knockout corroborates the model's
                # MECHANISM ATTRIBUTION (this call rests on gyrA_S83F), not the phenotype — the lab
                # confirms the phenotype. Precise wording on "The Honest One".
                f"④ gyrA S83F — one QRDR mutation; the knockout traces the {fq} resistance call to it  "
                f"({serovar.get(_QRDR, '?')})"
            ),
        }
    )

    # Atomic: the sidebar reads this at startup; a torn write from an interrupted finalize must
    # never reach it (a corrupt read there silently drops the curated presets).
    atomic_write(
        _OUT,
        lambda tmp: tmp.write_text(
            json.dumps({"genomes": genomes}, indent=2, ensure_ascii=False) + "\n"
        ),
    )
    print(f"\nwrote {_OUT} ({len(genomes)} curated demo genomes)")


if __name__ == "__main__":
    main()
