"""Re-derive S/I/R from raw MIC using ONE fixed CLSI M100 breakpoint set (ADR-0004).

Why not trust BV-BRC's `resistant_phenotype`? It mixes breakpoint eras across decades of
submissions — that label noise would masquerade as model miscalibration and quietly wreck
the calibration/conformal USP. So we re-interpret every raw MIC against a single current
breakpoint table. Censored values (">32", "<=0.5") are handled conservatively: we never
call an isolate Susceptible on an ambiguous upper-bounded reading (bounding VME — the
dangerous "called S but really R" error).

Breakpoint numbers below are CLSI M100 35th ed. (2025), read from the source tables
(Salmonella-specific 2A-2 where it exists, else generic Enterobacterales 2A-1).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Result vocabulary (3-class; Intermediate kept as a real class per ADR-0004, not collapsed).
R, INT, S = "Resistant", "Intermediate", "Susceptible"

# MIC is reported in mg/L (== µg/mL). Disk-diffusion (mm) needs zone breakpoints — skipped here.
_MIC_UNITS = frozenset({"mg/l", "ug/ml", "µg/ml", "mcg/ml"})
_DEFAULT_SIGN = "="


@dataclass(frozen=True, slots=True)
class Breakpoint:
    """CLSI MIC breakpoints (mg/L): S if MIC<=s_max, R if MIC>=r_min, else Intermediate."""

    s_max: float | None
    r_min: float | None
    has_clsi: bool = True
    note: str = ""


# Keys must match labels.canonical_drug() output. s_max = highest still-Susceptible MIC,
# r_min = lowest Resistant MIC; Intermediate is the gap between them. Values transcribed from
# CLSI M100 35th ed. (2025): Salmonella-specific Table 2A-2 where it exists, else the generic
# Enterobacterales Table 2A-1 (tagged in the note). Verified against the source PDF, not memory.
BREAKPOINTS: dict[str, Breakpoint] = {
    "ampicillin": Breakpoint(8, 32),  # 2A-2 Salmonella
    "amoxicillin-clavulanic acid": Breakpoint(
        8, 32, note="2A-1 Enterobacterales; amox 8/4→32/16"
    ),
    "ceftriaxone": Breakpoint(1, 4),  # 2A-2 Salmonella
    # 2A-1; CLSI comment 8: cephamycins look active in vitro but are NOT clinically effective
    # for Salmonella — AmpC MARKER only, never a therapeutic S call (ADR-0004).
    "cefoxitin": Breakpoint(8, 32, note="2A-1; AmpC marker only, non-therapeutic"),
    "meropenem": Breakpoint(1, 4),  # 2A-2 Salmonella
    # The lowered Salmonella-specific breakpoint (2A-2) so a single QRDR mutant is correctly
    # non-susceptible (S≤0.06, I 0.12–0.5, R≥1). This is the "cipro trap" — verified exactly.
    "ciprofloxacin": Breakpoint(0.06, 1),
    "nalidixic acid": Breakpoint(
        16, 32, note="2A-1; FQ screen only, not therapy (no I band)"
    ),
    # S. Typhi-only clinical breakpoint; NTS has NO CLSI breakpoint → this is ECOFF/surveillance,
    # not a clinical S call for non-typhoidal Salmonella. Kept as an internal marker.
    "azithromycin": Breakpoint(
        16, 32, has_clsi=False, note="S.Typhi/ECOFF only; NTS surveillance"
    ),
    # Current CLSI Ed35 = S≤2 / I4 / R≥8 (LOWERED from the legacy ≤4/8/≥16 most NARMS records
    # used). We label with current CLSI consistently. In-vitro only — not therapeutic for Salmonella.
    "gentamicin": Breakpoint(2, 8, note="2A-1 Ed35 (was 4/8/16); in-vitro marker only"),
    "tetracycline": Breakpoint(4, 16),  # 2A-2 Salmonella
    "chloramphenicol": Breakpoint(8, 32),  # 2A-2 Salmonella
    # TMP component: S≤2/38, R≥4/76, no I band. 2A-2 Salmonella.
    "trimethoprim-sulfamethoxazole": Breakpoint(2, 4),
    # No CLSI MIC breakpoint (disk-only); NARMS research cutoff R≥32 (was ≥64 pre-2014). Not clinical.
    "streptomycin": Breakpoint(
        None, 32, has_clsi=False, note="no CLSI; NARMS research cutoff"
    ),
}


def interpret_mic(value: float, sign: str, bp: Breakpoint) -> str | None:
    """Classify one MIC reading → R/I/S, or None if the (possibly censored) value is ambiguous."""
    sign = (sign or _DEFAULT_SIGN).strip()
    r_min, s_max = bp.r_min, bp.s_max

    if sign in (">", ">="):  # lower bound: MIC is at least ~value
        if r_min is not None and value >= r_min:
            return R
        if s_max is not None and value >= s_max:
            return INT if r_min is not None else R  # non-susceptible; R if no I band
        return None  # MIC>value but value<s_max → could still be S; uninterpretable
    if sign in ("<", "<="):  # upper bound: MIC is at most ~value
        if s_max is not None and value <= s_max:
            return S
        return None  # value>s_max → could be S or I; ambiguous
    # exact "="
    if r_min is not None and value >= r_min:
        return R
    if s_max is not None and value <= s_max:
        return S
    return INT


def _interpret_row(row: pd.Series) -> str | None:
    bp = BREAKPOINTS.get(str(row["antibiotic"]))
    if bp is None:
        return None
    unit = str(row.get("measurement_unit", "")).strip().lower()
    if unit not in _MIC_UNITS:
        return None  # disk diffusion (mm) or unknown unit → not MIC-interpretable here
    try:
        value = float(str(row["measurement_value"]).strip())
    except (ValueError, TypeError):
        return None
    return interpret_mic(value, str(row.get("measurement_sign", "") or ""), bp)


def rederive(raw: pd.DataFrame) -> pd.DataFrame:
    """Re-derive one MIC-based label per (genome, drug). Returns genome_id, antibiotic, label∈{R,I,S}.

    Rows without an interpretable MIC are dropped. Genome×drug collapses by majority vote,
    ties → Resistant (bounding VME — the safer error for a firewall).
    """
    df = raw.drop_duplicates().copy()
    df["mic_label"] = df.apply(_interpret_row, axis=1)
    df = df.dropna(subset=["mic_label"])

    def _resolve(group: pd.Series) -> str:
        counts = group.value_counts()
        winners = set(counts[counts == counts.max()].index)
        if R in winners and len(winners) > 1:
            return R
        return str(counts.idxmax())

    resolved = (
        df.groupby(["genome_id", "antibiotic"])["mic_label"]
        .apply(_resolve)
        .reset_index()
    )
    return resolved.rename(columns={"mic_label": "label"})


# BV-BRC resistant_phenotype vocabulary → our 3-class categories for comparison.
_BVBRC_TO_CAT = {
    "Susceptible": S,
    "Resistant": R,
    "Intermediate": INT,
    "Nonsusceptible": R,
}


def label_churn(raw: pd.DataFrame) -> pd.DataFrame:
    """Compare re-derived MIC labels against BV-BRC's resistant_phenotype (the churn figure).

    Per drug: `sr_flips` = hard S↔R disagreements; `cat_disagree` = any 3-class change; and
    `s_to_nonS` = BV-BRC-Susceptible calls we reclassify as Intermediate/Resistant — the
    ciprofloxacin "trap" direction (single-QRDR mutants old breakpoints missed) and the one
    that matters for patient safety (bounding VME). Evidence we understand label provenance.
    """
    ours = rederive(raw)
    bvbrc = raw[["genome_id", "antibiotic", "resistant_phenotype"]].drop_duplicates()
    bvbrc = bvbrc.dropna(subset=["resistant_phenotype"])
    merged = ours.merge(bvbrc, on=["genome_id", "antibiotic"], how="inner")
    merged["bvbrc_cat"] = merged["resistant_phenotype"].map(_BVBRC_TO_CAT)
    merged = merged.dropna(subset=["bvbrc_cat"])

    hard = {R, S}
    merged["sr_flip"] = (
        merged["label"].isin(hard)
        & merged["bvbrc_cat"].isin(hard)
        & (merged["label"] != merged["bvbrc_cat"])
    )
    merged["cat_change"] = merged["label"] != merged["bvbrc_cat"]
    merged["s_to_nonS"] = (merged["bvbrc_cat"] == S) & (merged["label"] != S)

    return (
        merged.groupby("antibiotic")
        .agg(
            n=("cat_change", "size"),
            sr_flips=("sr_flip", "sum"),
            cat_disagree=("cat_change", "sum"),
            s_to_nonS=("s_to_nonS", "sum"),
        )
        .assign(cat_disagree_pct=lambda d: (100 * d["cat_disagree"] / d["n"]).round(1))
        .reset_index()
        .sort_values("cat_disagree_pct", ascending=False)
    )
