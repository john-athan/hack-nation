"""AMRFinderPlus adapter: run the annotator on a FASTA and parse its TSV.

We shell out to the pinned `amr` micromamba env (determinism-first: the DB version is
fixed so the determinant matrix is reproducible). Parsing is pure code — no model.
"""

from __future__ import annotations

import io
import os
import signal
import subprocess
from pathlib import Path

import pandas as pd

from .constants import (
    AMRFINDER_ENV,
    AMRFINDER_ORGANISM,
    AMRFINDER_TIMEOUT_S,
    KEPT_SUBTYPES,
    MICROMAMBA,
)
from .errors import AMRFinderError

# AMRFinderPlus v4 renamed several columns from v3. We read v4 names and fall back to v3.
_SYMBOL_COLS = ("Element symbol", "Gene symbol")
_SUBTYPE_COLS = ("Element subtype", "Subtype")
_CLASS_COL = "Class"
_SUBCLASS_COL = "Subclass"


def run_amrfinder(fasta: Path, out_tsv: Path, threads: int = 1) -> Path:
    """Annotate one FASTA, writing AMRFinderPlus TSV to out_tsv. Returns out_tsv."""
    if not fasta.exists():
        raise AMRFinderError(f"FASTA not found: {fasta}")
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(MICROMAMBA),
        "run",
        "-n",
        AMRFINDER_ENV,
        "amrfinder",
        "-n",
        str(fasta),
        "--organism",
        AMRFINDER_ORGANISM,
        "--plus",
        "--threads",
        str(threads),
        "--name",
        fasta.stem,
        "-o",
        str(out_tsv),
    ]
    # start_new_session so a timeout can kill the WHOLE tree: `micromamba run` forks amrfinder,
    # which forks tblastn — killing just the immediate child would orphan those to eat CPU
    # through the rest of the demo. A bounded run means a hung upload degrades, never hangs.
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        # The `micromamba` binary itself is missing/unreachable (wrong $HOME, un-provisioned
        # box, renamed env) — the exact "amr env is missing" case AMRFinderError documents, yet
        # the ONE launch failure Popen raises as a bare FileNotFoundError. On the LIVE upload path
        # that would escape app.py's `except GenomeFirewallError` guard as a raw stage traceback;
        # convert it to the typed error the UI already degrades into a cached-genome hint.
        raise AMRFinderError(
            f"amrfinder is unavailable ({MICROMAMBA} not found) — the `amr` micromamba env is "
            f"likely missing. Use a pre-cached demo genome (annotates instantly), or install it "
            f"per AMRFinderError's install hint."
        ) from exc
    try:
        _, stderr = proc.communicate(timeout=AMRFINDER_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
        raise AMRFinderError(
            f"amrfinder timed out after {AMRFINDER_TIMEOUT_S}s for {fasta.name} — likely an "
            f"oversized or malformed assembly. Try a cached demo genome (annotates instantly)."
        ) from exc
    if proc.returncode != 0 or not out_tsv.exists():
        raise AMRFinderError(
            f"amrfinder failed for {fasta.name} (rc={proc.returncode}): {(stderr or '').strip()}"
        )
    return out_tsv


def _pick(cols: tuple[str, ...], available: pd.Index) -> str:
    for c in cols:
        if c in available:
            return c
    raise AMRFinderError(
        f"none of {cols} present in AMRFinder output columns {list(available)}"
    )


def parse_tsv(tsv: Path) -> pd.DataFrame:
    """Parse one AMRFinderPlus TSV → tidy rows (genome_id, symbol, subtype, class, subclass).

    Keeps only AMR/POINT determinants (drops VIRULENCE/STRESS/METAL). Empty (no hits)
    is valid and returns an empty frame — a susceptible-looking genome, not an error.
    """
    text = tsv.read_text()
    if not text.strip():
        return _empty()
    df = pd.read_csv(io.StringIO(text), sep="\t", dtype=str).fillna("")
    if df.empty:
        return _empty()

    sym = _pick(_SYMBOL_COLS, df.columns)
    sub = _pick(_SUBTYPE_COLS, df.columns)
    df = df[df[sub].isin(KEPT_SUBTYPES)]
    if df.empty:
        return _empty()

    out = pd.DataFrame(
        {
            "genome_id": tsv.stem,
            "symbol": df[sym].astype(str),
            "subtype": df[sub].astype(str),
            "drug_class": _optional_col(df, _CLASS_COL),
            "subclass": _optional_col(df, _SUBCLASS_COL),
        }
    )
    return out.reset_index(drop=True)


def _optional_col(df: pd.DataFrame, name: str) -> pd.Series:
    # A missing Class/Subclass must not silently blank every drug match (→ all no_call)
    # nor crash on DataFrame.get's scalar fallback; return an empty-string column instead.
    if name in df.columns:
        return df[name].astype(str)
    return pd.Series("", index=df.index)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["genome_id", "symbol", "subtype", "drug_class", "subclass"]
    )
