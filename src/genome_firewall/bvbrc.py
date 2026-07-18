"""BV-BRC HTTP client — labels (genome_amr) and assemblies (genome_sequence).

Thin, dependency-light wrapper over the public REST API. Every call retries on
transient failures because we pull thousands of genomes overnight and a single
flaky response should not sink the run.
"""

from __future__ import annotations

import io
import time
from urllib.parse import quote

import pandas as pd
import requests

from .constants import (
    BVBRC_API,
    BVBRC_EVIDENCE,
    BVBRC_METADATA_BATCH,
    BVBRC_PAGE_SIZE,
    BVBRC_SPECIES,
    HTTP_RETRIES,
    HTTP_TIMEOUT_S,
)
from .errors import BVBRCError, EmptyFastaError

# genome_amr columns we need for labels + provenance.
_LABEL_SELECT = (
    "genome_id,antibiotic,resistant_phenotype,measurement_value,"
    "measurement_unit,measurement_sign,laboratory_typing_method,"
    "testing_standard,testing_standard_year"
)

# genome columns for lineage stratification (serovar/MLST) + assembly QC.
_META_SELECT = "genome_id,genome_name,serovar,mlst,genome_length,contigs,genome_status"


def _get(url: str, headers: dict[str, str]) -> requests.Response:
    """GET with linear backoff. Raises BVBRCError once retries are exhausted."""
    last: Exception | None = None
    for attempt in range(HTTP_RETRIES):
        try:
            resp = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT_S)
            if resp.status_code == 200:
                return resp
            # A permanent client error (bad genome_id, malformed clause) won't heal on
            # retry — fail fast so a withdrawn genome doesn't burn 4x backoff in the big pull.
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                raise BVBRCError(f"HTTP {resp.status_code} (permanent) for {url}")
            last = BVBRCError(f"HTTP {resp.status_code} for {url}")
        except requests.RequestException as exc:  # network/timeout
            last = exc
        time.sleep(1.5 * (attempt + 1))
    raise BVBRCError(
        f"BV-BRC request failed after {HTTP_RETRIES} tries: {url}"
    ) from last


def _query(resource: str, clause: str) -> str:
    # Spaces must be percent-encoded inside the RQL clause or the API 400s.
    return f"{BVBRC_API}/{resource}/?{quote(clause, safe='(),=&')}"


def label_count() -> int:
    """Total lab-measured Salmonella phenotype rows available (via Content-Range)."""
    clause = f'and(keyword("{BVBRC_SPECIES}"),eq(evidence,{BVBRC_EVIDENCE}))&limit(1)'
    resp = _get(_query("genome_amr", clause), headers={"Accept": "application/json"})
    rng = resp.headers.get("Content-Range", "items 0-0/0")
    return int(rng.rsplit("/", 1)[-1])


def fetch_labels_csv() -> str:
    """Pull ALL lab-measured Salmonella phenotype rows as CSV text (paged)."""
    total = label_count()
    parts: list[str] = []
    header: str | None = None
    for offset in range(0, total, BVBRC_PAGE_SIZE):
        # sort() is required: Solr-backed offset paging is not stable without an explicit
        # order, so a refetch during index churn could drop/duplicate rows across pages.
        clause = (
            f'and(keyword("{BVBRC_SPECIES}"),eq(evidence,{BVBRC_EVIDENCE}))'
            f"&select({_LABEL_SELECT})&sort(+genome_id,+antibiotic)"
            f"&limit({BVBRC_PAGE_SIZE},{offset})"
        )
        resp = _get(_query("genome_amr", clause), headers={"Accept": "text/csv"})
        text = resp.text.strip("\n")
        if not text:
            continue
        first, _, rest = text.partition("\n")
        if header is None:
            header = first
            parts.append(text)
        else:
            parts.append(rest)
    if header is None:
        raise BVBRCError("BV-BRC returned no label rows")
    return "\n".join(parts) + "\n"


def fetch_metadata(genome_ids: list[str]) -> pd.DataFrame:
    """Fetch lineage/QC metadata (serovar, mlst, length, contigs) for many genomes.

    Batched via in(genome_id,(...)) so a few thousand candidates cost a handful of
    requests. Genomes the API omits (withdrawn) simply don't appear in the result.
    """
    frames: list[pd.DataFrame] = []
    for start in range(0, len(genome_ids), BVBRC_METADATA_BATCH):
        batch = genome_ids[start : start + BVBRC_METADATA_BATCH]
        ids = ",".join(batch)
        clause = (
            f"in(genome_id,({ids}))&select({_META_SELECT})&limit({BVBRC_PAGE_SIZE})"
        )
        resp = _get(_query("genome", clause), headers={"Accept": "text/csv"})
        text = resp.text.strip("\n")
        if text:
            frames.append(pd.read_csv(io.StringIO(text), dtype=str))
    if not frames:
        return pd.DataFrame(columns=_META_SELECT.split(","))
    return pd.concat(frames, ignore_index=True)


def fetch_fasta(genome_id: str) -> str:
    """Fetch a genome assembly as FASTA text. Raises EmptyFastaError if empty."""
    clause = (
        f"eq(genome_id,{genome_id})&select(sequence_id,sequence)"
        f"&sort(+sequence_id)&limit({BVBRC_PAGE_SIZE})"
    )
    resp = _get(
        _query("genome_sequence", clause), headers={"Accept": "application/json"}
    )
    contigs = resp.json()
    if not contigs:
        raise EmptyFastaError(f"no sequence for genome_id {genome_id}")
    lines = [f">{c['sequence_id']}\n{c['sequence']}" for c in contigs]
    return "\n".join(lines) + "\n"
