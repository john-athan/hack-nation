"""Lock the coverage money-plot's demo load-seam: graceful-missing → None, present → a frame.

The app renders this unguarded; a missing asset on a fresh checkout must yield an empty-state, not
an exception. These pin that contract the same way test_collapse.py pins the collapse frame's."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))

from coverage import (  # noqa: E402  (demo-local module)
    COVERAGE_PNG,
    load_coverage_table,
)


def test_missing_file_returns_none(tmp_path: Path) -> None:
    assert load_coverage_table(tmp_path / "does_not_exist.csv") is None


def test_present_file_loads_frame(tmp_path: Path) -> None:
    csv = tmp_path / "coverage_novelty.csv"
    frame = pd.DataFrame(
        {"bin": ["Q1 nearest"], "delivered_coverage": [0.91], "frac_abstain": [0.05]}
    )
    frame.to_csv(csv, index=False)
    loaded = load_coverage_table(csv)
    assert loaded is not None
    assert list(loaded["bin"]) == ["Q1 nearest"]
    assert loaded["delivered_coverage"].iloc[0] == 0.91


def test_coverage_png_path_under_assets() -> None:
    assert COVERAGE_PNG.name == "coverage_novelty.png"
    assert COVERAGE_PNG.parent.name == "assets"
