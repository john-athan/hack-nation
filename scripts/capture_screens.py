"""Capture full-page screenshots of the running Genome Firewall Streamlit demo.

Headless Chromium (Playwright) drives the live app and writes three PNGs into
docs/assets/ — the money slide plus two analyze-genome beats. It connects to an
ALREADY-RUNNING Streamlit (it launches nothing itself); start the app first with:

    setsid uv run --extra demo streamlit run demo/app.py \
        --server.headless true --server.port 8601 > /tmp/gf_stream.log 2>&1 &

Then regenerate the screenshots with:

    uv run --with playwright python scripts/capture_screens.py

Why an external Streamlit rather than launching it here: the demo's OpenAI key is
loaded from ~/.hack.env by the app process, and keeping the server long-lived (not
per-capture) is how the presenter actually runs it — we screenshot exactly that.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from playwright.sync_api import (  # ty: ignore[unresolved-import]  # dev-only, via `uv run --with playwright`
    Page,
    TimeoutError as PWTimeoutError,
    sync_playwright,
)

# This box ships no browser system libs (libnspr4/libgbm/libX* …). We stage them in a
# private dir and hand it to Chromium via LD_LIBRARY_PATH — a runtime-only shim, nothing
# is installed system-wide. Empty by default so a normal machine with the libs is unaffected.
BROWSER_LIB_DIR = os.environ.get("GF_BROWSER_LIB_DIR", "")

PORT = 8601
BASE_URL = f"http://localhost:{PORT}"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "docs" / "assets"

# A tall viewport plus full_page=True captures the whole scroll surface in one shot.
VIEWPORT = {"width": 1440, "height": 2600}

# Streamlit reruns on every widget change; these are generous so a cold cache or a
# live OpenAI rationale round-trip still lands inside the window.
NAV_TIMEOUT_MS = 30_000
WIDGET_TIMEOUT_MS = 40_000
# Extra settle so Vega charts / st.dataframe finish painting before the snapshot.
RENDER_SETTLE_MS = 2_500


def _select_curated_genome(page: Page, marker: str) -> None:
    """Pick the curated demo genome whose label starts with `marker` ('①' / '③').

    The sidebar '🎬 Curated demo genome' widget is a baseweb select: click it to
    pop the listbox, then click the option carrying the beat's circled-number
    prefix. Streamlit reruns after the choice, so we let the network settle.
    """
    combobox = page.get_by_role("combobox").first
    combobox.wait_for(state="visible", timeout=WIDGET_TIMEOUT_MS)
    combobox.click()
    option = page.get_by_role("option").filter(has_text=re.compile(rf"^{marker}"))
    option.first.wait_for(state="visible", timeout=WIDGET_TIMEOUT_MS)
    option.first.click()
    page.wait_for_load_state("networkidle", timeout=NAV_TIMEOUT_MS)


def _analyze_and_wait(page: Page) -> None:
    """Click 'Analyze genome' and wait for the per-drug report to render."""
    page.get_by_role("button", name="Analyze genome").click()
    page.get_by_text("Per-drug report").first.wait_for(
        state="visible", timeout=WIDGET_TIMEOUT_MS
    )
    page.wait_for_timeout(RENDER_SETTLE_MS)


def _shoot(page: Page, name: str) -> Path:
    out = ASSETS_DIR / name
    page.screenshot(path=str(out), full_page=True)
    return out


def _landing(page: Page) -> Path:
    page.goto(BASE_URL, wait_until="networkidle", timeout=NAV_TIMEOUT_MS)
    # The money slide: title + lab banner + "The collapse" chart/table, pre-click.
    page.get_by_text("The collapse").first.wait_for(
        state="visible", timeout=WIDGET_TIMEOUT_MS
    )
    page.wait_for_timeout(RENDER_SETTLE_MS)
    return _shoot(page, "01-landing-collapse.png")


def _beat(page: Page, marker: str, name: str) -> Path:
    # Reload to a clean slate so each beat starts from the un-analyzed landing state.
    page.goto(BASE_URL, wait_until="networkidle", timeout=NAV_TIMEOUT_MS)
    _select_curated_genome(page, marker)
    _analyze_and_wait(page)
    return _shoot(page, name)


def main() -> int:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    launch_env = None
    if BROWSER_LIB_DIR:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        merged = f"{BROWSER_LIB_DIR}:{existing}" if existing else BROWSER_LIB_DIR
        launch_env = {**os.environ, "LD_LIBRARY_PATH": merged}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, env=launch_env)
        page = browser.new_page(viewport=VIEWPORT)
        page.set_default_timeout(WIDGET_TIMEOUT_MS)
        try:
            written.append(_landing(page))
            written.append(_beat(page, "①", "02-beat1-report-and-firewall.png"))
            written.append(_beat(page, "③", "03-beat3-firewall-abstains.png"))
        finally:
            browser.close()
    for path in written:
        size = path.stat().st_size if path.exists() else 0
        print(f"wrote {path}  ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PWTimeoutError as exc:
        print(f"TIMEOUT waiting on the app — is Streamlit up on {BASE_URL}? {exc}")
        sys.exit(1)
