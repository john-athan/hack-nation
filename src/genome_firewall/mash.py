"""Mash adapter: sketch genomes, compute pairwise distances, and cluster lineages.

Mash distance (~1 - ANI) gives us a cheap, alignment-free genetic-similarity matrix. We
single-linkage cluster near-identical genomes so the train/test split groups by clade
(StratifiedGroupKFold on the cluster id) — otherwise near-duplicate strains leak across the
split and inflate accuracy. Everything here is pure code + a shelled-out mash call: no model,
no RNG, deterministic cluster ids.

Mash runs from the pinned `amr` micromamba env (same env as AMRFinderPlus).
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pandas as pd

from .constants import AMRFINDER_ENV, MICROMAMBA

# Mash sketch parameters. k=21 is Mash's recommended k for bacterial genomes (low collision
# probability at ~5 Mbp); a 100k-hash sketch resolves distances well below our clustering
# threshold. Pinned so the distance matrix is reproducible run-to-run.
MASH_KMER_SIZE = 21
MASH_SKETCH_SIZE = 100_000
# Single-linkage default: D≈0.005 (~99.5% ANI) groups near-identical strains of one clade.
DEFAULT_CLUSTER_THRESHOLD = 0.005
# `mash dist` TSV layout (no header): reference, query, distance, p-value, shared-hashes.
_DIST_COLS = ("ref", "query", "dist", "pvalue", "shared")


class MashError(Exception):
    """Mash failed to sketch/dist, or was called with no input.

    Most common cause: the `amr` micromamba env is missing mash. Install hint:
        ~/bin/micromamba install -n amr -c bioconda mash
    """


def sketch(fastas: list[Path], out_prefix: Path, threads: int = 4) -> Path:
    """Sketch one combined .msh from many FASTAs. Returns the <out_prefix>.msh path."""
    if not fastas:
        raise MashError("sketch requires at least one FASTA")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(MICROMAMBA),
        "run",
        "-n",
        AMRFINDER_ENV,
        "mash",
        "sketch",
        "-k",
        str(MASH_KMER_SIZE),
        "-s",
        str(MASH_SKETCH_SIZE),
        "-p",
        str(threads),
        "-o",
        str(out_prefix),
        *(str(f) for f in fastas),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    msh = out_prefix.with_suffix(".msh")
    if proc.returncode != 0 or not msh.exists():
        raise MashError(
            f"mash sketch failed (rc={proc.returncode}): {proc.stderr.strip()}"
        )
    return msh


def pairwise_dist(msh: Path, threads: int = 4) -> pd.DataFrame:
    """All-vs-all distances from one sketch → tidy frame [a, b, dist] keyed by genome id.

    Genome id is the FASTA file stem (e.g. `1079901.3` from `1079901.3.fna`), not the path
    mash echoes back — downstream joins are on the id, never the on-disk location.
    """
    cmd = [
        str(MICROMAMBA),
        "run",
        "-n",
        AMRFINDER_ENV,
        "mash",
        "dist",
        "-p",
        str(threads),
        str(msh),
        str(msh),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MashError(
            f"mash dist failed (rc={proc.returncode}): {proc.stderr.strip()}"
        )
    raw = pd.read_csv(
        io.StringIO(proc.stdout), sep="\t", header=None, names=list(_DIST_COLS)
    )
    return pd.DataFrame(
        {
            "a": raw["ref"].map(lambda p: Path(p).stem),
            "b": raw["query"].map(lambda p: Path(p).stem),
            "dist": raw["dist"].astype(float),
        }
    )


def single_linkage_clusters(
    dist: pd.DataFrame, threshold: float = DEFAULT_CLUSTER_THRESHOLD
) -> dict[str, int]:
    """Union-find single-linkage clustering: genomes join if dist <= threshold.

    Cluster ids are assigned deterministically (clusters sorted by their smallest member
    genome id → 0, 1, 2, ...) so a rerun yields identical ids. Every genome in the matrix
    gets an id, singletons included.
    """
    genomes = sorted(set(dist["a"]) | set(dist["b"]))
    parent = {g: g for g in genomes}

    def find(x: str) -> str:
        # Path compression keeps repeated lookups near-O(1); order-independent, so no RNG.
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            # Attach the lexicographically larger root under the smaller for a stable forest.
            hi, lo = (rx, ry) if rx > ry else (ry, rx)
            parent[hi] = lo

    for a, b, d in zip(dist["a"], dist["b"], dist["dist"], strict=True):
        if a != b and d <= threshold:
            union(a, b)

    # Map each root to a dense id ordered by the root's own (smallest-member) genome id.
    roots_sorted = sorted({find(g) for g in genomes})
    root_to_id = {root: i for i, root in enumerate(roots_sorted)}
    return {g: root_to_id[find(g)] for g in genomes}


def nearest_neighbors(dist: pd.DataFrame) -> dict[str, tuple[str, float]]:
    """For each genome, its single closest OTHER genome and that distance (self excluded).

    Feeds a nearest-neighbor baseline and out-of-distribution detection: a query far from
    every training genome is one we should not confidently call.
    """
    best: dict[str, tuple[str, float]] = {}
    for a, b, d in zip(dist["a"], dist["b"], dist["dist"], strict=True):
        if a == b:
            continue
        cur = best.get(a)
        # Ties resolve to the smaller genome id for a deterministic neighbour.
        if cur is None or d < cur[1] or (d == cur[1] and b < cur[0]):
            best[a] = (b, float(d))
    return best
