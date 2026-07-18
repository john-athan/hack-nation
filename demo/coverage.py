"""Pure load-seam for the demo's coverage-vs-novelty money plot, kept Streamlit-free so the
orchestrator can wire it into app.py and a test can lock the graceful-missing contract.

Mirrors demo/collapse.py's doctrine: no `st.*` at import time. The heavy figure and per-bin table
are baked offline by scripts/coverage_novelty.py into docs/assets/; here we only load them, and
degrade to None (never raise) when the asset is absent so a fresh clone renders an empty-state
instead of crashing the app."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "docs" / "assets"
COVERAGE_CSV = _ASSETS_DIR / "coverage_novelty.csv"
# Path constant the orchestrator hands to st.image; kept here so app.py never hard-codes a path.
COVERAGE_PNG = _ASSETS_DIR / "coverage_novelty.png"


def load_coverage_table(path: Path | str | None = None) -> pd.DataFrame | None:
    """Load the committed per-bin coverage table, or None if it hasn't been built yet.

    Returns None (not an exception) on a missing file so the caller renders an empty-state — the
    demo must survive a checkout where the offline asset wasn't regenerated."""
    p = Path(path) if path is not None else COVERAGE_CSV
    if not p.exists():
        return None
    return pd.read_csv(p)
