"""Guard that demo/app.py bootstraps its OWN sibling imports (collapse, coverage, ...).

app.py leans on nothing but its own sys.path setup: it must import `collapse`/`coverage`/
`diversity`/`applicability` no matter who launches it. `streamlit run` happens to put the
script's dir on sys.path, so a demo that only works under that one launcher looks fine on the
box yet can break on a different cwd (Streamlit Cloud, AppTest, a wrapper). This drives the REAL
file through AppTest from the repo root with `demo/` deliberately kept OFF the path, so a
regression that drops the self-bootstrap fails pytest instead of the deploy.

Runs in a subprocess with a clean sys.path: importing app.py mutates the parent process's
sys.path, so an in-process check would pass off polluted state and mask a reverted fix.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

# Executed as a child process from the repo root. `demo/` is NOT on sys.path (only the repo root,
# which pytest/-c add). If app.py did not insert its own dir, the sibling imports would raise
# ModuleNotFoundError and this child would exit non-zero.
_CHILD = """
import sys
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("demo/app.py", default_timeout=90).run()
assert not at.exception, f"app.py raised on landing view: {at.exception}"
assert len(at.markdown) > 0, "landing view rendered no markdown"
print("OK", len(at.markdown))
"""


def test_app_is_launcher_independent() -> None:
    # Clean cwd = repo root; do not leak a `demo/` entry via PYTHONPATH — the whole point is that
    # app.py resolves its siblings unaided.
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, (
        "demo/app.py failed to load from the repo root without `demo/` on sys.path. It must "
        "insert its own directory so sibling modules (collapse, coverage, ...) resolve under any "
        f"launcher.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert proc.stdout.startswith("OK"), proc.stdout
