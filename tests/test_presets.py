"""The app must never surface a curated preset for a beat it no longer renders.

Regression guard for cycle 41's incomplete revert: the "dark-AMR" beat's rendering code was
reverted, but its entry survived in the gitignored demo_genomes.json and stayed live on stage.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))

from presets import SUPPORTED_BEATS, supported_presets  # noqa: E402


def test_supported_presets_keeps_known_beats() -> None:
    entries = [
        {"id": "1079901.3", "beat": "known_gene", "label": "①"},
        {"id": "28901.24344", "beat": "ood", "label": "③"},
    ]
    assert supported_presets(entries) == entries


def test_supported_presets_drops_reverted_dark_amr_beat() -> None:
    entries = [
        {"id": "1079901.3", "beat": "known_gene", "label": "①"},
        {"id": "28901.22098", "beat": "dark_amr", "label": "④ Dark-AMR — …"},
    ]
    kept = supported_presets(entries)
    assert {e["id"] for e in kept} == {"1079901.3"}
    assert "dark_amr" not in SUPPORTED_BEATS


def test_supported_presets_drops_entry_without_beat() -> None:
    assert supported_presets([{"id": "x", "label": "y"}]) == []
