"""Lock the demo collapse money-slide's CHART RENDER (not just its data prep).

test_collapse.py pins `collapse_frame` — which drugs, what order — but that is only half the slide:
the render call itself shipped this surface's *fourth* display regression (st.bar_chart defaults to
Vega-stacked, summing random+grouped per drug so the axis ran past 1.0 and the gap the pitch points
at vanished). The frame tests can't catch that because it lives in the `st.bar_chart` kwargs, welded
to Streamlit — exactly the class of bug that keeps slipping through. This drives the REAL app render
helper through Streamlit's AppTest and asserts the produced Vega-Lite spec is grouped (side-by-side),
never stacked, so a regression in the render args fails pytest instead of the stage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_DEMO = Path(__file__).resolve().parent.parent / "demo"


def _render_script() -> None:
    # Runs as the AppTest "script": import the real helper and render a synthetic 2-drug frame.
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "demo"))
    import pandas as pd
    from app import _render_collapse_chart  # the real render seam from demo/app.py

    ok = pd.DataFrame(
        {
            "drug": ["ciprofloxacin", "ampicillin"],
            "random_bal_acc": [0.83, 0.95],
            "grouped_bal_acc": [0.65, 0.93],
        }
    )
    _render_collapse_chart(ok)


def test_collapse_chart_is_grouped_not_stacked() -> None:
    sys.path.insert(0, str(_DEMO))
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_render_script)
    at.run()
    assert not at.exception, at.exception

    chart = at.get("vega_lite_chart")
    assert chart, "collapse section rendered no bar chart"
    spec = json.loads(chart[0].proto.spec)
    encoding = spec["encoding"]

    # Grouped side-by-side is signalled by an xOffset channel + an explicit non-stacked y.
    # Stacked (the buggy default) has neither. Assert we are unambiguously in the grouped layout.
    assert "xOffset" in encoding, (
        "collapse chart must be GROUPED (side-by-side) so the random-vs-grouped gap is visible; "
        "a missing xOffset means it reverted to stacked bars (summing the two series)"
    )
    assert encoding["y"].get("stack") is False, (
        "y.stack must be False so balanced accuracy stays on a [0,1] axis; a stacked chart sums "
        "random+grouped per drug and runs the axis past 1.0"
    )
