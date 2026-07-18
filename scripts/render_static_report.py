"""Pre-render the flagship demo genome's FULL report to a self-contained static HTML.

This is the demo's black-box flight recorder: if the box, the venue wifi, or OpenAI die on
stage, `docs/assets/flagship_report.html` still shows the whole money path — the per-drug
mechanism report, the naive-vs-firewall verdicts with knockout Δ, and the collapse slide —
with zero network, zero live model calls beyond what is already cached on disk.

It does NOT reimplement any logic: it drives the exact demo library seams the Streamlit app
uses (pipeline.analyze_fasta over the cached AMRFinderPlus TSV, report_table, verdict.*,
collapse.collapse_frame, rationale.explain served from the on-disk cache) and mirrors app.py's
on-screen content into one inline-CSS HTML file. Deterministic and offline by construction:
annotation comes from the cached TSV, rationales from the pre-baked disk cache (a miss falls
back to the deterministic template — never a crash).
"""

from __future__ import annotations

import html
import json
import math
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
# Match app.py: make `genome_firewall` and the local demo modules importable, and run relative
# to the repo root so the constants' relative data/ paths (FASTA_DIR, AMRFINDER_DIR) resolve.
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "demo"))

from collapse import COLLAPSE_COL, collapse_frame  # noqa: E402
from genome_firewall.constants import (  # noqa: E402
    CALL_NO_CALL,
    CALL_RESISTANT,
    CALL_SUSCEPTIBLE,
    DATA_DIR,
    EVIDENCE_STATISTICAL,
    FASTA_DIR,
)
from genome_firewall.conformal import CONFORMAL_ALPHA, VERDICT_OOD  # noqa: E402
from genome_firewall.env import load_hack_env  # noqa: E402
from genome_firewall.rationale import _ENV_API_KEY, explain  # noqa: E402
from genome_firewall.train import MODELS_PATH, load  # noqa: E402
from pipeline import analyze_fasta  # noqa: E402
from report_table import report_table, untrained_reported_drugs  # noqa: E402
from verdict import all_verdicts, format_delta, naive_confidence  # noqa: E402

# Loading the key lets rationale.explain serve the pre-baked disk cache (still offline — the disk
# hit never touches the network); with no key it degrades to the deterministic template. Either
# path is safe and crash-free, so the static artifact renders with or without ~/.hack.env sourced.
load_hack_env()

import os  # noqa: E402  (read AFTER load_hack_env populates the key)

_OUTPUT_PATH = _REPO_ROOT / "docs" / "assets" / "flagship_report.html"
_DEMO_GENOMES_JSON = DATA_DIR / "demo_genomes.json"
_RESULTS_CSV = DATA_DIR / "results.csv"
_FLAGSHIP_BEAT = "known_gene"
_FALLBACK_GENOME = "1079901.3"
_COVERAGE_TARGET = 1 - CONFORMAL_ALPHA

# Mirror of app.py's display strings (kept in sync by reuse of the same seams, not the styling).
_FIREWALL_STYLE = {
    CALL_RESISTANT: "🔴 resistant",
    CALL_SUSCEPTIBLE: "🟢 susceptible",
    CALL_NO_CALL: "⚪ NO-CALL ({R,S} — both labels plausible)",
    VERDICT_OOD: "🚫 NO-CALL — neither R nor S clears 90% coverage",
}
_LAB_BANNER = (
    "⚠️ Decision support only — not a diagnosis. Every result must be confirmed with standard "
    "laboratory susceptibility testing before any treatment decision."
)
_RATIONALE_WORTHY_CALLS = frozenset({CALL_RESISTANT, CALL_NO_CALL})


def _flagship_genome_id() -> str:
    """The beat-① known_gene genome, read from data/demo_genomes.json; fall back if unreadable."""
    try:
        entries = json.loads(_DEMO_GENOMES_JSON.read_text())["genomes"]
    except (OSError, ValueError, KeyError):
        return _FALLBACK_GENOME
    for e in entries:
        if e.get("beat") == _FLAGSHIP_BEAT and (FASTA_DIR / f"{e['id']}.fna").exists():
            return str(e["id"])
    return _FALLBACK_GENOME


def _firewall_rows(report, determinants: pd.DataFrame) -> pd.DataFrame | None:  # noqa: ANN001
    """The naive-vs-firewall verdict rows, mirroring app._firewall_section. None if no models."""
    if not MODELS_PATH.exists():
        return None
    models = load(MODELS_PATH)
    symbols = set(determinants["symbol"].astype(str))
    verdicts = all_verdicts(models, symbols, determinants, report)
    rows = [
        {
            "antibiotic": v.drug,
            "naive": f"{v.naive_call} ({naive_confidence(v):.0%})",
            "firewall": _FIREWALL_STYLE.get(v.firewall_verdict, v.firewall_verdict),
            "🛡️": "🛡️ HOLDING" if v.firewall_holding else "",
            "knockout Δ": format_delta(v.knockout_delta),
            "evidence": v.evidence,
            "role": "marker-only" if not v.therapeutic else "therapeutic",
        }
        for v in sorted(verdicts, key=lambda x: (not x.firewall_holding, x.drug))
    ]
    return pd.DataFrame(rows)


def _untrained_caption(report, determinants: pd.DataFrame) -> str:  # noqa: ANN001
    if not MODELS_PATH.exists():
        return ""
    models = load(MODELS_PATH)
    results = pd.read_csv(_RESULTS_CSV) if _RESULTS_CSV.exists() else None
    untrained = untrained_reported_drugs(report, list(models), results)
    if not untrained:
        return ""
    named = "; ".join(
        f"<b>{html.escape(d)}</b> — {html.escape(w)}" for d, w in untrained
    )
    return (
        f"Not in the calibrated firewall: {named}. Each is called 🔴 resistant in the mechanism "
        "report above but has no trained model, so the firewall stays silent here rather than fake "
        "a calibrated probability it cannot compute."
    )


def _rationale_lines(report) -> tuple[list[tuple[str, str]], bool]:  # noqa: ANN001
    """Per-drug plain-language rationale lines + whether OpenAI phrasings were used."""
    worthy = [
        p
        for p in report.predictions
        if p.call in _RATIONALE_WORTHY_CALLS
        or p.evidence_category == EVIDENCE_STATISTICAL
    ]
    llm_live = bool(os.environ.get(_ENV_API_KEY))
    lines = [
        (p.antibiotic, explain(p, report.genome_id, use_llm=llm_live)) for p in worthy
    ]
    return lines, llm_live


def _collapse_table() -> pd.DataFrame | None:
    if not _RESULTS_CSV.exists():
        return None
    ok = collapse_frame(pd.read_csv(_RESULTS_CSV))
    if ok is None:
        return None
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
    display_cols = [c for c in display_cols if c in ok.columns]
    return ok[display_cols].rename(columns={COLLAPSE_COL: "Δ (bal-acc collapse)"})


def _table_html(df: pd.DataFrame) -> str:
    """Deterministic HTML table from a DataFrame; NaN → em dash, floats trimmed to 3 places."""
    cleaned = df.copy()
    for col in cleaned.columns:
        cleaned[col] = cleaned[col].map(
            lambda v: (
                "—"
                if isinstance(v, float) and math.isnan(v)
                else f"{v:.3f}"
                if isinstance(v, float)
                else v
            )
        )
    return cleaned.to_html(index=False, escape=True, border=0, classes="report")


_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial,
  sans-serif; margin: 0; padding: 2rem; color: #1a1a1a; background: #f6f7f9; line-height: 1.5; }
main { max-width: 1040px; margin: 0 auto; }
h1 { font-size: 1.9rem; margin: 0 0 .25rem; }
h2 { font-size: 1.25rem; margin: 2rem 0 .5rem; border-bottom: 2px solid #e2e5e9; padding-bottom: .3rem; }
.subtitle { color: #555; margin: 0 0 1rem; }
.banner { background: #fff4e0; border: 1px solid #f0c674; border-radius: 8px; padding: .75rem 1rem;
  margin: 1rem 0; font-weight: 600; color: #7a5200; }
.caption { color: #555; font-size: .88rem; margin: .5rem 0 0; }
.static-note { background: #eef2ff; border: 1px solid #c7d2fe; border-radius: 8px; padding: .6rem 1rem;
  font-size: .85rem; color: #3730a3; margin-bottom: 1.5rem; }
table.report { border-collapse: collapse; width: 100%; background: #fff; border-radius: 8px;
  overflow: hidden; box-shadow: 0 1px 2px rgba(0,0,0,.06); font-size: .9rem; }
table.report th, table.report td { text-align: left; padding: .5rem .7rem; border-bottom: 1px solid #eceef1; }
table.report th { background: #f0f2f5; font-weight: 600; }
table.report tr:last-child td { border-bottom: none; }
ul.rationale { margin: .5rem 0; padding-left: 1.2rem; }
ul.rationale li { margin: .3rem 0; }
footer { color: #888; font-size: .8rem; margin-top: 2.5rem; text-align: center; }
"""


def _render_html(
    genome_id: str,
    report_df: pd.DataFrame,
    rationale: tuple[list[tuple[str, str]], bool],
    firewall_df: pd.DataFrame | None,
    firewall_caption: str,
    collapse_df: pd.DataFrame | None,
) -> str:
    lines, llm_live = rationale
    rationale_label = "OpenAI (pre-baked)" if llm_live else "deterministic template"
    rationale_items = "\n".join(
        f"      <li><b>{html.escape(d)}</b> — {html.escape(t)}</li>" for d, t in lines
    )
    firewall_html = (
        _table_html(firewall_df)
        if firewall_df is not None
        else "<p class='caption'>Train per-drug models to enable the firewall verdicts.</p>"
    )
    firewall_caption_html = (
        f"<p class='caption'>{firewall_caption}</p>" if firewall_caption else ""
    )
    collapse_html = (
        _table_html(collapse_df)
        if collapse_df is not None
        else "<p class='caption'>No evaluated drugs yet.</p>"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Genome Firewall — flagship report ({html.escape(genome_id)})</title>
<style>{_CSS}</style>
</head>
<body>
<main>
  <h1>🧬 Genome Firewall — flagship report</h1>
  <p class="subtitle">Calibrated AMR predictions with a hard safety interlock. Static offline snapshot of genome
    <b>{html.escape(genome_id)}</b> (beat ① — resistance via known genes).</p>
  <div class="static-note">Static backup: this page was pre-rendered from the cached AMRFinderPlus
    annotation and the pre-baked rationale cache — no network, no live model calls. It mirrors the
    live Streamlit demo so the money path survives a box/network/OpenAI outage on stage.</div>
  <div class="banner">{html.escape(_LAB_BANNER)}</div>

  <h2>Per-drug mechanism report — genome {html.escape(genome_id)}</h2>
  {_table_html(report_df)}
  <p class="caption">No fabrication by construction: the mechanism report asserts <i>resistant</i> only when a
    curated determinant explains it; absence of a known gene becomes a no-call — never a
    mechanism-based "susceptible".</p>

  <h2>Plain-language rationale — {rationale_label}</h2>
  <ul class="rationale">
{rationale_items}
  </ul>

  <h2>Naive model vs the Firewall</h2>
  {firewall_html}
  <p class="caption">A normal AMR model always commits — the <b>naive</b> column shows its call and
    confidence. The <b>Firewall</b> commits only when a label clears the 90%-coverage bar; otherwise it
    emits a NO-CALL, or overrides to 🔴 resistant on a characterized mechanism. 🛡️ marks where the
    firewall diverges from the naive call.</p>
  {firewall_caption_html}

  <h2>The collapse: random split vs leave-clade-out</h2>
  {collapse_html}
  <p class="caption">Same model, two splits. <b>Random</b> leaks bacterial lineage and looks great; the
    <b>grouped / leave-clade-out</b> split is the one that generalizes. AUROC barely moves while balanced
    accuracy drops sharply on the leave-clade-out split — that gap is the lineage leakage. Coverage dips
    below the {_COVERAGE_TARGET:.0%} target on the leave-clade-out split for the same exchangeability break.</p>
  <div class="banner">{html.escape(_LAB_BANNER)}</div>
  <footer>Genome Firewall · pre-rendered static backup · deterministic &amp; offline</footer>
</main>
</body>
</html>
"""


def main() -> None:
    genome_id = _flagship_genome_id()
    fasta = FASTA_DIR / f"{genome_id}.fna"
    report, determinants = analyze_fasta(fasta)  # cached TSV → no network

    report_df = report_table(report)
    rationale = _rationale_lines(report)
    firewall_df = _firewall_rows(report, determinants)
    firewall_caption = _untrained_caption(report, determinants)
    collapse_df = _collapse_table()

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(
        _render_html(
            genome_id, report_df, rationale, firewall_df, firewall_caption, collapse_df
        ),
        encoding="utf-8",
    )
    print(
        f"Wrote {_OUTPUT_PATH} ({_OUTPUT_PATH.stat().st_size} bytes) for genome {genome_id}"
    )


if __name__ == "__main__":
    main()
