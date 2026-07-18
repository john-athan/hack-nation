"""Crash-safe artifact writes: write to a sibling temp file, then atomically rename into place.

The overnight finalize chain (`bin/finalize.sh`) rewrites the demo's headline artifacts —
`results.csv`, `models.joblib`, `demo_genomes.json` — UNATTENDED at ~03:00 on a memory-constrained
box. A plain in-place write that is SIGKILL'd mid-syscall (OOM, a killed process group) leaves a
TRUNCATED file, and the morning demo then crashes on load (`joblib.load` / `pd.read_csv`). `os.replace`
is atomic within a filesystem, so a concurrent reader ever sees the OLD file whole or the NEW file
whole — never a torn one; a failed write leaves the prior good artifact untouched. This guards torn
writes (the real failure mode here), not power-loss durability (which would additionally need fsync).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def atomic_write(path: Path, writer: Callable[[Path], object]) -> Path:
    """Run `writer` against a sibling temp path, then atomically `os.replace` it onto `path`.

    The temp file lives in the SAME directory as `path` so the rename stays within one filesystem
    (a cross-device `os.replace` would raise). A failure — including a raise inside `writer` — removes
    the temp file and re-raises, never leaving a half-written artifact or clobbering the existing one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        writer(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path
