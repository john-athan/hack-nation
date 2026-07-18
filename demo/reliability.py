"""Pure load-seam for the demo's reliability (calibration) diagram, kept Streamlit-free so app.py
wires it and a test can lock the graceful-missing contract.

Mirrors demo/coverage.py's doctrine: no `st.*` at import time. The figure and per-bin table are
baked offline by scripts/reliability.py into docs/assets/; here we only load them, and degrade to
None (never raise) when the asset is absent so a fresh clone renders an empty-state."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "docs" / "assets"
RELIABILITY_CSV = _ASSETS_DIR / "reliability.csv"
# Path constant the orchestrator hands to st.image; kept here so app.py never hard-codes a path.
RELIABILITY_PNG = _ASSETS_DIR / "reliability.png"


def load_reliability_table(path: Path | str | None = None) -> pd.DataFrame | None:
    """Load the committed per-bin calibration table, or None if it hasn't been built yet."""
    p = Path(path) if path is not None else RELIABILITY_CSV
    if not p.exists():
        return None
    return pd.read_csv(p)
