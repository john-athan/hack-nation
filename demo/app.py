"""Genome Firewall — Streamlit demo.

Run: uv run --extra demo streamlit run demo/app.py
The happy path: pick (or upload) a Salmonella genome → live AMRFinderPlus → honest per-drug
report + the random-vs-grouped "collapse" evidence. Strictly defensive decision support.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow `import genome_firewall` when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# `streamlit run` implicitly puts this file's dir on sys.path for the sibling local
# modules (collapse, coverage, ...); do it explicitly so other launchers (AppTest, a
# different cwd on Streamlit Cloud) resolve them too. Launcher-independent, same imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genome_firewall.constants import (  # noqa: E402
    CALL_NO_CALL,
    CALL_RESISTANT,
    DATA_DIR,
    EVIDENCE_STATISTICAL,
    FASTA_DIR,
)
from genome_firewall.conformal import CONFORMAL_ALPHA, VERDICT_OOD  # noqa: E402
from genome_firewall.env import load_hack_env  # noqa: E402
from genome_firewall.errors import GenomeFirewallError  # noqa: E402
from genome_firewall.rationale import (  # noqa: E402
    _ENV_API_KEY,
    MODEL,
    disk_cache_available,
    explain,
)
from genome_firewall.train import MODELS_PATH, load  # noqa: E402

# Make the OpenAI key available whether launched by the driver or by hand (see env.py).
load_hack_env()

from collapse import (  # noqa: E402  (local module)
    COLLAPSE_COL,
    collapse_frame,
    non_discriminating_drugs,
    sub_target_coverage_drugs,
    underpowered_drugs,
)
from coverage import COVERAGE_PNG, load_coverage_table  # noqa: E402  (local module)
from reliability import (  # noqa: E402  (local module)
    RELIABILITY_PNG,
    load_reliability_table,
)
from diversity import load_diversity, phenotype_spread  # noqa: E402  (local module)
from pipeline import analyze_fasta, upload_fasta_path  # noqa: E402  (local module)
from presets import supported_presets  # noqa: E402  (local module)
from report_table import report_table, untrained_reported_drugs  # noqa: E402  (local module)
from verdict import all_verdicts, format_delta, naive_confidence  # noqa: E402  (local module)

_RESULTS_CSV = DATA_DIR / "results.csv"
_DEMO_GENOMES_JSON = DATA_DIR / "demo_genomes.json"
# The conformal set's target coverage (≥1−α) — the "90%" the firewall/collapse captions quote.
_COVERAGE_TARGET = 1 - CONFORMAL_ALPHA
# Public-tunnel mode (set by the systemd unit behind the cloudflared URL). The curated genomes are
# the zero-external-call hero path; on the PUBLIC url we hide live FASTA upload by default so an
# anonymous visitor can't spend our OpenAI budget or contend for the box's 4 cores with a live
# AMRFinderPlus run. Cached-only is the default (uploader hidden). Set GENOME_FIREWALL_PUBLIC=0 locally (with the bio tools) to re-enable the uploader. Off-switch, never a silent one.
_PUBLIC_MODE = os.environ.get("GENOME_FIREWALL_PUBLIC", "1").strip().lower() not in (
    "",
    "0",
    "false",
    "no",
)
_FIREWALL_STYLE = {
    "resistant": "🔴 resistant",
    "susceptible": "🟢 susceptible",
    "no_call": "⚪ NO-CALL ({R,S}, both labels plausible)",
    # The empty conformal set is the STRONGEST abstention, not a novelty/manifold verdict: on the
    # serving path every model's global quantile is <0.5, so {R,S} is unreachable and {} fires
    # exactly when P(R) sits in the uncertain middle band — neither label clears the 90%-coverage
    # nonconformity bar. Nothing here measures distance to the training manifold, so the label must
    # not claim "OOD / novel" (NO-FABRICATION; see conformal.py + test_serving_empty_set_*).
    VERDICT_OOD: "🚫 NO-CALL: neither R nor S clears 90% coverage",
}
_LAB_BANNER = (
    "⚠️ **Decision support only. Not a diagnosis.** Every result must be confirmed with "
    "standard laboratory susceptibility testing before any treatment decision."
)


def _demo_genomes() -> list[str]:
    return sorted(p.stem for p in FASTA_DIR.glob("*.fna"))


def _demo_presets() -> list[dict[str, str]]:
    """Curated demo genomes (one per beat) picked + verified by scripts/pick_demo_genomes.py.

    Keeps the presenter off a flat 5k-id list on stage; regenerated on every retrain so it can
    never point at a genome the (re-picked) firewall no longer flags. Missing file → no presets.
    """
    if not _DEMO_GENOMES_JSON.exists():
        return []
    try:
        entries = json.loads(_DEMO_GENOMES_JSON.read_text())["genomes"]
    except (json.JSONDecodeError, KeyError, OSError):
        return []
    # Drop any beat the current code no longer renders (stale gitignored data guard — see
    # presets.py), THEN keep only presets whose FASTA is cached (instant, deterministic on stage).
    return [
        e for e in supported_presets(entries) if (FASTA_DIR / f"{e['id']}.fna").exists()
    ]


# Which predictions get a rationale LINE shown at all: a resistance call, an honest no-call, or a
# mechanism-free statistical signal (a susceptible-no-signal row would be self-evident, but the
# rule-based report never emits one). This is only the DISPLAY filter — the OpenAI spend is gated
# separately inside explain(), which phrases only resistance/statistical calls and renders every
# no-call from the fixed deterministic template (a no-call's one honest sentence is never rephrased).
_RATIONALE_WORTHY_CALLS = frozenset({CALL_RESISTANT, CALL_NO_CALL})


def _rationale_section(report, use_llm: bool) -> None:  # noqa: ANN001
    """One- or two-sentence plain-language rationale per meaningful prediction.

    OpenAI phrases it when `use_llm` and a key are set; otherwise the deterministic template.
    The constrained layer can never surface a gene outside the prediction's own payload, so
    this is safe to show live (NO-FABRICATION doctrine). `explain()` never raises.
    """
    worthy = [
        p
        for p in report.predictions
        if p.call in _RATIONALE_WORTHY_CALLS
        or p.evidence_category == EVIDENCE_STATISTICAL
    ]
    if not worthy:
        return
    key_present = bool(os.environ.get(_ENV_API_KEY))
    llm_live = use_llm and key_present
    # OpenAI-sourced whenever we serve a phrasing (a live call, or the pre-baked cache which was
    # itself phrased by OpenAI). Only when the LLM is off entirely do we fall to the template.
    served_openai = use_llm and (key_present or disk_cache_available())
    label = "OpenAI" if served_openai else "deterministic template"
    st.subheader(f"Plain-language rationale ({label})")
    with st.spinner("Phrasing rationales…" if llm_live else "Building rationales…"):
        lines = [
            (p.antibiotic, explain(p, report.genome_id, use_llm=use_llm))
            for p in worthy
        ]
    for drug, text in lines:
        st.markdown(f"- **{drug}**: {text}")
    if llm_live:
        st.caption(
            f"Resistance and statistical calls are phrased by OpenAI `{MODEL}`, "
            "hard-constrained to the genes in each prediction's payload (any fabricated gene is "
            "rejected back to the template) and cost-capped. **No-call rows use the fixed "
            "deterministic template.** A no-call's single template sentence is never rephrased toward "
            "susceptibility. OpenAI phrases the verdict; it never computes it."
        )
    elif served_openai:
        st.caption(
            f"These are real OpenAI `{MODEL}` rationales, **pre-baked** into the repo — phrased "
            "by the model, gene-checked at bake time and re-checked on load — so the hosted demo "
            "shows real model output with no key and no per-visit cost. Set `OPENAI_API_KEY` and run "
            "locally to phrase novel genomes live. OpenAI phrases the verdict; it never computes it."
        )
    else:
        st.caption(
            f"These are the built-in **deterministic template** rationales. The OpenAI `{MODEL}` "
            "phrasing layer (and its pre-baked cache) is not available here. It is fully implemented "
            "— run locally with `OPENAI_API_KEY` set to watch it phrase each verdict, hard-constrained "
            "to the genes actually found. The wording changes; the verdict does not."
        )


def _firewall_section(report, determinants: pd.DataFrame) -> None:  # noqa: ANN001
    """The USP centerpiece: naive confident call vs the conformal firewall + knockout evidence."""
    if not MODELS_PATH.exists():
        st.info(
            "Train per-drug models (`scripts/train_models.py`) to enable the calibrated "
            "**naive-vs-firewall** verdicts and the gene-knockout evidence probe."
        )
        return
    models = load(MODELS_PATH)
    symbols = set(determinants["symbol"].astype(str))
    # Pass the honest report so a curated-determinant drug can never read "susceptible" here while
    # the report above calls it resistant — the firewall defers to characterized mechanism.
    verdicts = all_verdicts(models, symbols, determinants, report)
    st.subheader("Naive model vs the Firewall")
    st.caption(
        "A normal AMR model always commits. The **naive** column shows its call and confidence, "
        "even where the genome gives it little to go on (a 51% probability still reads as a firm "
        "call). The **Firewall** commits only when a label clears the *target* 90%-coverage bar; "
        "otherwise "
        "it emits a **NO-CALL** rather than guess, or overrides to 🔴 resistant on a characterized "
        "mechanism. Rows marked 🛡️ are where *the firewall diverges from the naive call*: "
        "withholding where the naive committed (sometimes at a confidence the genome cannot "
        "justify), or overriding it on a known gene."
    )
    rows = []
    for v in sorted(verdicts, key=lambda x: (not x.firewall_holding, x.drug)):
        # Show the naive model's confidence in the class it CALLED (see verdict.naive_confidence):
        # pairing "susceptible" with raw P(resistant) made the OOD beat read 22% vs the "works (78%)"
        # narration — a self-contradiction on the USP slide.
        rows.append(
            {
                "antibiotic": v.drug,
                "naive": f"{v.naive_call} ({naive_confidence(v):.0%})",
                "firewall": _FIREWALL_STYLE.get(v.firewall_verdict, v.firewall_verdict),
                "🛡️": "🛡️ HOLDING" if v.firewall_holding else "",
                "knockout Δ": format_delta(v.knockout_delta),
                "evidence": v.evidence,
                "role": "marker-only" if not v.therapeutic else "therapeutic",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    # A drug the mechanism report calls resistant (e.g. streptomycin) but with no trained model
    # must not just vanish from this table — name it and say WHY, rather than let the two tables
    # silently disagree on the product named "The Honest One". Reason derives from results.csv.
    results = pd.read_csv(_RESULTS_CSV) if _RESULTS_CSV.exists() else None
    untrained = untrained_reported_drugs(report, [v.drug for v in verdicts], results)
    if untrained:
        named = "; ".join(f"**{drug}**: {why}" for drug, why in untrained)
        st.caption(
            f"Not in the calibrated firewall: {named}. Each is called 🔴 resistant in the "
            "mechanism report above but has no trained model, so the firewall stays **silent here "
            "rather than fake a calibrated probability it cannot compute.** See the mechanism "
            "report for its evidence."
        )


def _render_collapse_chart(ok: pd.DataFrame) -> None:
    """Render the random-vs-grouped balanced-accuracy bars GROUPED (side-by-side), not stacked.

    The money slide's whole point is the *gap* between the two splits. Two render gotchas erase it,
    both defaults of `st.bar_chart` on a multi-series frame — and both must be overridden:
      • stack=False — without it, Vega stacks the two series, so each drug shows one bar of height
        random+grouped (axis runs past 1.0 for a [0,1] metric) and a bigger collapse merely shrinks
        the top segment: no visible gap, the opposite of the narration. stack=False dodges them
        side-by-side so a tall `random` next to a short `grouped` reads as the collapse at a glance.
      • sort=False — keeps `ok`'s largest-collapse-first row order; the default (sort=True) would
        re-sort the drug axis ALPHABETICALLY, scattering the dramatic collapses instead of leading
        with them. (`ok` is already sorted by descending bal-acc collapse in collapse_frame.)
    """
    chart = ok.set_index("drug")[["random_bal_acc", "grouped_bal_acc"]]
    st.bar_chart(chart, sort=False, stack=False)


def _collapse_section() -> None:
    st.subheader("The collapse: random split vs leave-clade-out")
    if not _RESULTS_CSV.exists():
        st.info(
            "Run `scripts/run_evaluation.py` to populate the random-vs-grouped comparison."
        )
        return
    table = pd.read_csv(_RESULTS_CSV)
    ok = collapse_frame(table)
    if ok is None:
        st.info("No evaluated drugs yet.")
        return
    st.caption(
        "Same model, two splits. **Random** leaks bacterial lineage and looks great; the "
        "**grouped / leave-clade-out** split is the one that generalizes. We chart **balanced "
        "accuracy**, the operating-point metric a safety interlock depends on. "
        "Notice AUROC (in the table) barely moves while balanced accuracy drops sharply on the "
        "leave-clade-out split: that gap is the lineage leakage the 2025 PLOS finding named. "
        "The **Brier** columns (a proper score: lower is better-*calibrated*, not only "
        "better-ranked) worsen on the leave-clade-out split too: the over-confidence a lineage-leaking "
        "split hides is exactly what a patient-facing tool must not ship."
    )
    excluded = non_discriminating_drugs(table)
    if excluded:
        # Honest scoping, stated out loud: a drug at chance on the random split never learned a
        # signal, so it has no collapse to show and would otherwise sit here as a counterexample.
        st.caption(
            f"_Excluded (insufficient signal, model at chance on the random split, so no "
            f"collapse to show): {', '.join(excluded)}. Full per-drug metrics remain in "
            f"`results.csv`._"
        )
    underpowered = underpowered_drugs(table)
    if underpowered:
        # Same honest-scoping move for the other exclusion axis: a drug with too few isolates posts
        # a degenerate small-n AUROC/coverage that would read as "too good to be true" beside the
        # well-powered drugs. Name it (with n) rather than silently charting or dropping it.
        counts = dict(zip(table["drug"], table["n"], strict=False))
        named = ", ".join(
            f"{d} (n={int(counts[d])})" for d in underpowered if d in counts
        )
        st.caption(
            f"_Excluded (too few isolates for a stable leave-clade-out estimate): {named}. "
            f"Full per-drug metrics remain in `results.csv`._"
        )
    _render_collapse_chart(ok)
    # Brier columns are additive (older results.csv predate them) — only show what's present so a
    # stale artifact still renders the table instead of raising a KeyError on stage.
    brier_cols = [c for c in ("random_brier", "grouped_brier") if c in ok.columns]
    display_cols = [
        "drug",
        "random_bal_acc",
        "grouped_bal_acc",
        COLLAPSE_COL,
        "random_auroc",
        "grouped_auroc",
        *brier_cols,
        "coverage",
        "n_resistant",
    ]
    st.dataframe(
        ok[display_cols].rename(columns={COLLAPSE_COL: "Δ (bal-acc collapse)"}),
        hide_index=True,
    )
    # The rest of the challenge Success Criteria, per drug on the honest split: per-class recall
    # (sensitivity for R, specificity for S) reported SEPARATELY, F1, and PR-AUC (the metric that
    # matters under class imbalance). Rendered from the same results.csv and guarded on presence,
    # so an older artifact without these columns still renders the collapse table above.
    crit_cols = [
        c
        for c in (
            "grouped_recall_resistant",
            "grouped_recall_susceptible",
            "grouped_f1_resistant",
            "grouped_pr_auc",
        )
        if c in ok.columns
    ]
    if crit_cols:
        st.caption(
            "Operating-point metrics on the leave-clade-out split, per drug — the rest of the challenge "
            "Success Criteria: recall for resistant (sensitivity) and susceptible (specificity) "
            "reported separately, F1, and PR-AUC (the metric that matters under class imbalance)."
        )
        st.dataframe(
            ok[["drug", *crit_cols]].rename(
                columns={
                    "grouped_recall_resistant": "recall R (sensitivity)",
                    "grouped_recall_susceptible": "recall S (specificity)",
                    "grouped_f1_resistant": "F1 (R)",
                    "grouped_pr_auc": "PR-AUC",
                }
            ),
            hide_index=True,
        )
    # The 'coverage' column shows realized conformal coverage on the honest split; for some drugs
    # it dips below the 90% target the firewall caption quotes. Own it out loud (self-updating — the
    # caveat only renders while a drug actually dips): unstated, a conformal-literate reviewer reads a
    # sub-90% number as a broken guarantee. Owned, it is a symptom of the SAME exchangeability break
    # the collapse measures. But coverage-dip and bal-acc-collapse only correlate weakly drug-by-drug
    # (our biggest collapse dips hardest, yet flat-collapse drugs also dip and some big collapses
    # don't), so the caption must NOT claim they track "exactly" — that would be an overclaim the
    # table beside it refutes on the one slide whose whole job is honest calibration.
    dipping = sub_target_coverage_drugs(table, _COVERAGE_TARGET)
    if dipping:
        cov = dict(zip(ok["drug"], ok["coverage"], strict=False))
        named = ", ".join(f"{d} {cov[d]:.2f}" for d in dipping if d in cov)
        st.caption(
            f"**Coverage** is the conformal set's realized coverage on the "
            f"leave-clade-out split. It dips below the {_COVERAGE_TARGET:.0%} *target* for "
            f"{named}. That is a symptom of the same lineage shift the collapse measures, not a "
            f"broken guarantee: conformal's marginal coverage assumes exchangeability, which "
            f"the clade holdout deliberately breaks. Balanced accuracy drops sharply for the same "
            f"reason, though drug-by-drug the two need not move together. On the exchangeable "
            f"random split the ≥1−α guarantee holds."
        )


def _coverage_section() -> None:
    """The coverage-vs-novelty money plot: does the 90% guarantee still hold as a genome drifts
    away from anything we trained on? Baked offline by scripts/coverage_novelty.py; here we only
    load it and degrade to an empty state when the asset hasn't been built (fresh clone)."""
    if not COVERAGE_PNG.exists():
        return
    st.subheader("Does the guarantee hold as genomes get more novel?")
    st.caption(
        "The real question for any safety layer: the coverage promise is easy to keep on "
        "genomes that look like the training set. Does it survive on the strange ones? We split "
        "every leave-clade-out prediction into four bins by how far the genome sits from its "
        f"own fold's training set (Mash distance), then measure delivered coverage per bin against "
        f"the {_COVERAGE_TARGET:.0%} target."
    )
    st.image(str(COVERAGE_PNG), width="stretch")
    st.caption(
        "Coverage stays at the target in every bin, from the genomes nearest the training set to "
        "the most novel. That is the guarantee holding conditionally, not just on average. The "
        "cost is abstention: the tool declines to commit on 11 to 19% of predictions (right "
        "panel), holding its coverage by saying no-call instead of guessing. Within one species "
        "the novelty range is narrow (Mash distance under 0.05), so this shows the guarantee is "
        "robust across the cohort we have. It is not a claim about a brand-new pathogen."
    )
    table = load_coverage_table()
    if table is not None:
        st.caption("Per-bin numbers:")
        st.dataframe(table, hide_index=True)


def _reliability_section() -> None:
    """The other half of the Success Criteria's 'confidence quality': the Brier score made visual.
    A reliability diagram on the honest split — predicted P(resistant) vs observed frequency. Baked
    offline by scripts/reliability.py; load-only here, empty-state when the asset is absent."""
    if not RELIABILITY_PNG.exists():
        return
    st.subheader("Are the confidence scores calibrated?")
    st.caption(
        "Reliability diagram on the leave-clade-out split: predicted P(resistant) vs the "
        "observed resistant frequency, pooled across the therapeutic drugs. Points on the diagonal "
        "mean a stated 70% really is 70%. This is the Brier score in the collapse table made "
        "visual — the confidence-quality evidence the challenge Success Criteria asks for by name."
    )
    st.image(str(RELIABILITY_PNG), width="stretch")
    table = load_reliability_table()
    if table is not None:
        st.caption("Per-bin numbers:")
        st.dataframe(table, hide_index=True)


def _applicability_section() -> None:
    """The dangerous failure mode: a model that sees no resistance marker and declares a drug works,
    even when the drug cannot work here regardless of the genome (e.g. vancomycin on E. coli). Pure
    presentation over the frozen gates already in report.py / drugs.py (target_present,
    intrinsic_resistant, absence -> no-call). Renders no new computation."""
    st.subheader("🛡️ The failure this refuses: 'no gene found, so the drug works'")
    st.markdown(
        "A genome model can find no resistance gene for a drug and conclude the drug will work, when "
        "in fact the drug cannot work on that species at all. The textbook case is vancomycin against "
        "*E. coli*: there is no acquired resistance gene to find, yet vancomycin never treats a "
        "Gram-negative bug like *E. coli*. Absence of a resistance gene is not evidence that a drug "
        "works. Catching this needs gates beyond gene-matching. We built them."
    )
    st.markdown(
        "- **Absence of a marker is a NO-CALL, never a blind 'susceptible'.** No known resistance "
        "gene means we do not know, so we say so. A missing gene never becomes 'this drug works'.\n"
        "- **A drug-applicability gate.** Each drug carries its molecular target. If the target the "
        "drug acts on is not even present, absence of resistance genes cannot imply susceptibility.\n"
        "- **An intrinsic-resistance gate (EUCAST expected phenotype).** A drug a species is "
        "intrinsically resistant to is a deterministic call, not a statistical guess."
    )
    st.caption(
        "Together these gates refuse the one dangerous mistake: calling a bug treatable when it is "
        "not, the mistake this tool is built to catch."
    )


def _diversity_section() -> None:
    """Guards against the data-bias trap: a single-clone or single-phenotype cohort makes any
    accuracy number a lineage artifact. Answer it up front with real spread from the committed
    sources (baked lineage summary + results.csv). Presentation only, no computation on the frozen
    core; degrades to nothing when either artifact is absent (fresh clone)."""
    div = load_diversity()
    if div is None:
        return
    st.subheader("Is the cohort one clone? No.")
    lineage, mlst, clade = st.columns(3)
    lineage.metric("Named serovars", f"{div['n_serovars_named']}")
    mlst.metric("MLST sequence types", f"{div['n_mlst']}")
    clade.metric(
        f"Biggest clade ({div['top_serovar']})",
        f"{div['top_serovar_share']:.0%} of cohort",
    )
    st.caption(
        f"The {div['n_genomes']:,} Salmonella genomes span {div['n_serovars_named']} named serovars "
        f"and {div['n_mlst']} MLST sequence types. The single largest lineage is only "
        f"{div['top_serovar_share']:.0%} of the cohort, so no one clade dominates the training set."
    )
    if not _RESULTS_CSV.exists():
        return
    spread = phenotype_spread(pd.read_csv(_RESULTS_CSV))
    if spread is not None:
        st.caption(
            f"Resistance is not one phenotype either. Across the {spread['n_drugs']} drugs, "
            f"prevalence runs from {spread['min_frac']:.0%} ({spread['min_drug']}) to "
            f"{spread['max_frac']:.0%} ({spread['max_drug']}), so every drug carries both resistant "
            "and susceptible isolates to learn from. A cohort biased to one lineage or one phenotype "
            "would turn any accuracy score into an artifact. This one is diverse on both axes."
        )


def _provenance_section() -> None:
    """Responsible-design credibility: state where the ground-truth labels come from
    and how Intermediate is handled, plainly. Grounded in the ACTIVE label path (mic.rederive,
    used by run_evaluation.py): raw MIC re-interpreted against one fixed CLSI M100 35th ed. (2025)
    breakpoint set, censored readings bounded toward the safe side, ties toward Resistant. Pure
    presentation over committed code, no computation, no data artifact, no frozen-core dependency."""
    st.subheader("Where the ground-truth labels come from")
    st.markdown(
        "The labels the model learns from come from lab MIC measurements, not from a ready-made "
        "resistant/susceptible call. Public databases ship a ready-made call, but it mixes "
        "breakpoint standards from decades of submissions. That inconsistency would show up as fake "
        "miscalibration and quietly break the coverage guarantee this tool is built on. So we re-read "
        "every raw MIC against one fixed breakpoint table (CLSI M100, 35th edition, 2025)."
    )
    st.markdown(
        "- **Intermediate is handled openly.** Re-reading each MIC gives three outcomes: susceptible, "
        "intermediate, resistant. Intermediate is the ambiguous zone right around the breakpoint. We "
        "do not force it to one side. The model trains on the clear resistant and susceptible cases, "
        "and intermediate isolates are held out.\n"
        "- **Censored readings stay on the safe side.** A raw MIC like '>32' or '<=0.5' has no exact "
        "value. We never read an upper-bounded MIC as susceptible, because calling a resistant bug "
        "treatable is the error that harms a patient.\n"
        "- **One clean label per genome and drug.** When a genome has several measurements for one "
        "drug, we take the majority, and a tie breaks toward resistant, the safer call for a firewall."
    )
    st.caption(
        "Re-deriving labels from raw MIC is more work than trusting the ready-made column. We do it so "
        "the numbers mean what they say. The labels are still lab phenotypes, so every report "
        "carries a confirm-with-standard-lab-testing note."
    )


def _plain_language_section() -> None:
    """A plain-words explainer for non-expert readers so the target coverage reads as a chosen SAFETY
    GUARANTEE, not a weak accuracy score, plus the VME framing and the one measurable-impact line.
    Pure presentation, no computation, no frozen-core dependency."""
    st.subheader("What the numbers mean (in plain words)")
    left, right = st.columns(2)
    with left:
        st.markdown(
            f"**The {_COVERAGE_TARGET:.0%} is a safety promise, not our accuracy.**\n\n"
            f"It is the coverage the tool guarantees: across many genomes, the true answer sits "
            f"inside what it reports at least {_COVERAGE_TARGET:.0%} of the time. When a genome is "
            "too novel to keep that promise, the tool says NO-CALL instead of guessing. A guaranteed "
            f"{_COVERAGE_TARGET:.0%} beats a confident 99% that is sometimes wrong, because a wrong "
            "'this drug works' can kill a patient."
        )
    with right:
        st.markdown(
            "**The error we work hardest to avoid.**\n\n"
            "Calling a resistant infection treatable is the deadly mistake. The patient gets a drug "
            "that cannot work while the infection runs on. So when the tool is unsure, it would "
            "rather return NO-CALL than call a resistant bug susceptible."
        )
    st.caption(
        "Why it matters: antibiotic-resistant infections kill more than 1 million people a year, "
        "and standard lab susceptibility testing takes 1 to 3 days. The genome holds much of the "
        "answer in seconds. The hard part is knowing when to trust that answer, which is the job "
        "of this tool."
    )


def main() -> None:
    st.set_page_config(page_title="Genome Firewall", page_icon="🧬", layout="wide")
    st.title("🧬 Genome Firewall")
    st.markdown(
        "**Calibrated antibiotic-resistance predictions, with a hard safety interlock.** "
        "Antibiotic-resistant infections kill "
        ">1M people/year; standard susceptibility testing takes 1–3 days. The genome has much "
        "of the answer in minutes. What's missing is a *trustworthy* way to read it."
    )
    st.warning(_LAB_BANNER)

    with st.sidebar:
        st.header("Input")
        genomes = _demo_genomes()
        presets = _demo_presets()
        choice: str | None
        if presets:
            # Curated, per-beat presets up top so each demo beat is one click, never a scroll.
            label_to_id = {p["label"]: p["id"] for p in presets}
            choice = label_to_id[
                st.selectbox("🎬 Curated demo genome", list(label_to_id))
            ]
            with st.expander("…or browse all cached genomes"):
                browsed = st.selectbox("All cached", ["(use curated above)", *genomes])
                if browsed != "(use curated above)":
                    choice = browsed
        else:
            choice = (
                st.selectbox("Cached Salmonella genome", genomes) if genomes else None
            )
        if _PUBLIC_MODE:
            upload = None
            st.caption(
                "🔒 Live FASTA upload is disabled on the public demo. The curated genomes above "
                "run the full pipeline instantly. Clone the repo to analyse your own genome."
            )
        else:
            upload = st.file_uploader("…or upload a FASTA", type=["fna", "fasta", "fa"])
        key_present = bool(os.environ.get(_ENV_API_KEY))
        baked_available = disk_cache_available()
        openai_available = key_present or baked_available
        use_llm = st.checkbox(
            "✨ OpenAI plain-language rationale",
            value=openai_available,
            disabled=not openai_available,
            help=f"Phrase each verdict with OpenAI {MODEL} (constrained to payload genes, "
            "cost-capped)."
            if key_present
            else f"Curated genomes serve pre-baked OpenAI {MODEL} rationales; set OPENAI_API_KEY "
            "to phrase novel genomes live."
            if baked_available
            else "Set OPENAI_API_KEY (or ~/.hack.env) to enable. Falls back to templates.",
        )
        if not key_present and baked_available:
            st.caption(
                f"✨ Rationales here are real OpenAI `{MODEL}` phrasings **pre-baked** into the "
                f"repo, so this hosted demo shows real model output with no key and no cost. Clone "
                f"and set `OPENAI_API_KEY` to phrase your own genomes live."
            )
        elif not key_present:
            st.caption(
                f"🔒 No OpenAI key and no baked cache here, so rationales use the built-in "
                f"**deterministic template**. The OpenAI `{MODEL}` layer is fully implemented — set "
                f"`OPENAI_API_KEY` to enable it. Same verdict either way; the model only rewords it."
            )
        go = st.button("Analyze genome", type="primary")

    if not go:
        # Landing keeps the signature collapse slide visible; everything else tucks into one
        # collapsed expander so the page reads like an app (pick a genome → analyze), not an essay.
        _collapse_section()
        with st.expander(
            "📊 More evidence & methodology — calibration, coverage, cohort diversity, "
            "label provenance, and the failure it refuses",
            expanded=False,
        ):
            _reliability_section()
            _coverage_section()
            _diversity_section()
            _plain_language_section()
            _applicability_section()
            _provenance_section()
        return

    if upload is not None:
        content = upload.getvalue()
        fasta = upload_fasta_path(FASTA_DIR, upload.name, content)
        fasta.write_bytes(content)
    elif choice:
        fasta = FASTA_DIR / f"{choice}.fna"
    else:
        st.error("Pick a cached genome or upload a FASTA.")
        return

    try:
        with st.spinner(f"Running AMRFinderPlus on {fasta.stem} (~17s)…"):
            report, determinants = analyze_fasta(fasta)
    except GenomeFirewallError as exc:
        # Never let a flaky live annotation crash the demo — degrade to a clear message and
        # point at the pre-cached genomes (which annotate instantly).
        st.error(f"Could not analyze {fasta.stem}: {exc}")
        st.info(
            "Try one of the pre-cached demo genomes in the sidebar. They run instantly."
        )
        return

    _label = {e["id"]: e.get("label", "") for e in _demo_presets()}.get(report.genome_id, "")
    _label = _label.lstrip("①②③④⑤⑥⑦⑧⑨ ").strip()
    if _label:
        st.subheader(f"Per-drug report: {_label}")
        st.caption(f"BV-BRC genome {report.genome_id}")
    else:
        st.subheader(f"Per-drug report: genome {report.genome_id}")
    st.dataframe(report_table(report), hide_index=True)
    st.caption(
        "No fabrication by construction: this **mechanism** report asserts *resistant* only when a "
        "curated determinant explains it, and absence of a known gene becomes a **no-call** here, "
        "never a mechanism-based 'susceptible'. The separate **calibrated statistical** verdict "
        "(next table) is the second oracle: where mechanism is silent it may still commit, with a "
        "conformal no-call when even the statistics lack support."
    )
    _rationale_section(report, use_llm)
    _firewall_section(report, determinants)
    st.warning(_LAB_BANNER)
    # The per-genome result above is the app; the evidence + methodology tuck into one collapsed
    # expander so someone reading their result isn't wading through the whole explainer every time.
    with st.expander(
        "📊 Evidence & methodology — the collapse slide, calibration, coverage, cohort, "
        "labels, and the failure it refuses",
        expanded=False,
    ):
        _collapse_section()
        _reliability_section()
        _coverage_section()
        _applicability_section()
        _diversity_section()
        _plain_language_section()
        _provenance_section()


if __name__ == "__main__":
    main()
