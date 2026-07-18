"""Guard the submission Live URL path: public mode + NO OpenAI key renders clean.

The owner's submission Live Project URL is a Streamlit Community Cloud deploy. Cloud runs
`demo/app.py` with `GENOME_FIREWALL_PUBLIC=1`, NO `~/.hack.env` (so no OpenAI key), and only
the pip deps in the top-level `requirements.txt` (no mash/amrfinder). The curated beats are the
zero-external-call hero path, so the app MUST render its landing with zero exceptions under those
exact conditions.

Every presentation cycle adds panels to app.py (diversity, coverage, plain-language, the QRDR
beat, ...). Any one of them could reach for the OpenAI key at import/landing, or take a
public-only branch that raises, and the break would only surface when the owner clicks Deploy at
submission time. This drives the REAL app under the Cloud env (public flag on, key genuinely
absent) so such a regression fails pytest instead of the deploy.

Subprocess + a scrubbed env (no key inherited, HOME at an empty temp dir so `load_hack_env`
finds nothing): importing app.py mutates the parent's sys.path/os.environ, and the whole point is
that the child sees the Cloud condition, not this box's exported key.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

# Self-checking child: it first ASSERTS it is genuinely in the Cloud condition (public flag on,
# no key visible) so a mis-scrubbed env fails loudly instead of passing a false green, then proves
# the landing renders. `demo/` is kept OFF sys.path (app.py must self-bootstrap, as on Cloud).
_CHILD = """
import os
from streamlit.testing.v1 import AppTest

assert os.environ.get("GENOME_FIREWALL_PUBLIC", "").lower() in ("1", "true", "yes"), \
    "public flag not set — not a faithful Cloud sim"
assert "OPENAI_API_KEY" not in os.environ, \
    "an OpenAI key leaked into the child — not a faithful no-key sim"

at = AppTest.from_file("demo/app.py", default_timeout=90).run()
assert not at.exception, f"public no-key landing raised: {at.exception}"
assert len(at.markdown) > 0, "public landing rendered no markdown"
print("OK", len(at.markdown))
"""


def test_public_no_key_landing_renders() -> None:
    with tempfile.TemporaryDirectory() as clean_home:
        # A minimal env that mirrors Streamlit Cloud: public flag on, no OpenAI key, HOME at an
        # empty dir (so `~/.hack.env` does not exist and no key is loaded). Explicitly do NOT copy
        # this box's os.environ — the driver exports OPENAI_API_KEY, which would mask the break.
        env = {
            "PATH": _system_path(),
            "HOME": clean_home,
            "GENOME_FIREWALL_PUBLIC": "1",
        }
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD],
            cwd=_REPO,
            capture_output=True,
            text=True,
            timeout=180,
            env=env,
        )
    assert proc.returncode == 0, (
        "demo/app.py failed to render on the Streamlit Cloud path (public mode, no OpenAI key). "
        "This is the owner's submission Live URL — a panel must not require the key or raise in "
        f"public mode.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert proc.stdout.startswith("OK"), proc.stdout


def _system_path() -> str:
    import os

    return os.environ.get("PATH", "/usr/bin:/bin")
