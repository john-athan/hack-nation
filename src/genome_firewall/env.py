"""Load owner-convention secrets (~/.hack.env) into the process env, once.

The overnight driver `set -a; . ~/.hack.env` before every cycle, so agent-launched
processes inherit OPENAI_API_KEY. A *manual* demo run (`streamlit run demo/app.py`, or a
judge cloning the repo) does not. This mirrors that single convention so the OpenAI
rationale path behaves identically however the app is launched — without pulling in
python-dotenv (LEAN doctrine: a six-line hand-rolled parser over a framework dep).

Precedence is deliberate: anything already exported wins (`setdefault`), so an explicit
`OPENAI_API_KEY=… streamlit …` or the driver's own export is never clobbered.
"""

from __future__ import annotations

import os
from pathlib import Path

_HACK_ENV = Path.home() / ".hack.env"
_QUOTES = "\"'"


def load_hack_env(path: Path = _HACK_ENV) -> None:
    """Best-effort load of KEY=VALUE lines from ~/.hack.env into os.environ.

    Silent no-op if the file is absent — this is a convenience, not a requirement, and
    the rationale layer already degrades to templates when the key never shows up.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip(_QUOTES))
