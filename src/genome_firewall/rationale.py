"""T4 constrained-rationale layer: a THIN OpenAI wrapper over one prediction.

Design rules that are load-bearing, not decoration:
- OpenAI is NEVER the predictor — it only phrases an already-computed `DrugPrediction`.
- It is optional: the deterministic template is the real product; the LLM is a nicety
  behind a flag + an API key. Every failure mode (no key, no `openai` install, API
  error, cost cap, or a fabricated gene) falls back to that template. No exception
  escapes `explain()`, and no unverified gene is ever surfaced (NO-FABRICATION doctrine).
- Cost is hard-bounded: small model, cached by (genome, drug, call), running token
  counter, and a hard USD cap that trips back to templates.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from .atomicio import atomic_write
from .constants import (
    CALL_RESISTANT,
    CALL_SUSCEPTIBLE,
    EVIDENCE_STATISTICAL,
)

if TYPE_CHECKING:
    from .schema import DrugPrediction

# --- OpenAI call parameters --------------------------------------------------
_ENV_API_KEY = "OPENAI_API_KEY"
# The rationale is a constrained phrasing task (rephrase an already-computed verdict, citing only
# genes in the payload), not reasoning, so a small model is the right tier — and a newer small model
# beats gpt-4o-mini on every axis that matters here: cheaper, faster, a fresher knowledge cutoff, and
# a stronger instruction-follower (which tightens the no-fabrication guard). Overridable via env so
# the exact id can be pinned or rolled without a code change. gpt-5.4-nano is the cheaper drop-in if
# cost ever matters — it rarely does, since the curated demo serves pre-baked phrasings offline.
MODEL = os.environ.get("GENOME_FIREWALL_OPENAI_MODEL", "gpt-5.4-mini")
MAX_TOKENS = 300  # one or two sentences never needs more; caps worst-case spend/call
# Bound the live call so flaky venue wifi can't hang the demo. The rationale is rendered
# synchronously in the Streamlit loop; without a timeout the openai SDK defaults to 600s
# (10 min) × 2 retries, so a single dropped connection would freeze the stage for ~30 min.
# Small-model phrasing normally returns in ~1–2s, so 12s is generous yet a hard ceiling; a
# timeout raises → explain() catches it → the deterministic template renders instead. One
# retry rides out a transient blip without the default's ×3 worst-case wait. (Mirrors the
# AMRFinder process-group timeout added for the upload path — same "never hang on stage" rule.)
OPENAI_TIMEOUT_S = 12.0
OPENAI_MAX_RETRIES = 1

# --- Cost guard --------------------------------------------------------------
# List price (USD per 1M tokens) for the spend projection below. Kept at gpt-4o-mini's (higher)
# price as a deliberate CONSERVATIVE ceiling: the newer small models are cheaper, so this over-counts
# spend and trips the cap early — fail-safe, never an under-count. One place to change for the exact
# current price.
_INPUT_USD_PER_1M = 0.15
_OUTPUT_USD_PER_1M = 0.60
_PER_MILLION = 1_000_000
# Hard stop: once projected cumulative spend would cross this, we stop calling OpenAI
# for the rest of the process and serve templates. Owner directive: hard-stop ~$20.
HARD_CAP_USD = 20.0

# Module-level running token totals — the whole process shares one budget.
_prompt_tokens_used = 0
_completion_tokens_used = 0

# Cache: identical (genome, drug, call) triples reuse the first answer for free.
_rationale_cache: dict[tuple[str, str, str], str] = {}

# --- On-disk pre-baked cache -------------------------------------------------
# The live OpenAI call is the one un-cached external dependency left on the CURATED demo
# path. On flaky venue wifi it stalls the money slide: each resistance row waits up to the
# 12s timeout before falling back to a template, so a beat with several resistance rows can
# hang ~1–2 min BEFORE the firewall centerpiece even renders. We pre-bake the curated beats'
# OpenAI phrasings to data/rationale_cache.json (scripts/bake_rationales.py) and serve them
# from disk before ever touching the network — instant, deterministic, and still genuinely
# *phrased by OpenAI*, so the "phrased by OpenAI" caption stays truthful. A cache MISS
# (an un-baked genome, or a finalize that shifted the call) transparently falls through to a
# live call, so this only ever removes latency — it can never change a verdict.
_DISK_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "rationale_cache.json"
_DISK_RECORDS_KEY = "rationales"
_DISK_GENOME_KEY = "genome_id"
_DISK_ANTIBIOTIC_KEY = "antibiotic"
_DISK_CALL_KEY = "call"
_DISK_EVIDENCE_KEY = "evidence_category"
_DISK_GENES_KEY = "supporting_genes"
_DISK_TEXT_KEY = "text"

# Key = (genome, drug, call, evidence_category). `evidence_category` is in the key so a
# genome whose call is unchanged but whose evidence BASIS flips (e.g. a mechanism newly
# appears: statistical_only → known_gene) becomes a clean MISS → live path, never serving a
# stale "from a statistical model signal without a known mechanism" sentence for a prediction
# that now HAS a mechanism (the foreign-gene gate alone can't catch an evidence-basis shift).
_DiskKey = tuple[str, str, str, str]
# Lazily read, then held for the process. None = not yet read; {} = read but absent/empty.
_disk_cache: dict[_DiskKey, str] | None = None

# --- Prompt (data, not inlined) ----------------------------------------------
# The prompt lives at repo-root prompts/rationale.txt; this file is src/genome_firewall/
# so the repo root is two parents up. Path-relative (not importlib.resources) because the
# prompt is a repo asset outside the installed package tree.
_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "rationale.txt"
# The file holds the system prompt, then this delimiter on its own line, then the user
# prompt — one versioned artifact, split once at load.
_PROMPT_DELIMITER = "===USER==="
_NO_GENES = "none"  # rendered into the prompt when supporting_genes is empty

# --- Deterministic template --------------------------------------------------
_KNOWN_DETERMINANT = "a known resistance determinant"
_CONFIRM = "Confirm with standard laboratory susceptibility testing."

# --- No-fabrication gene detector --------------------------------------------
# Gene-like token shapes. Anything matching that is NOT in the allowed set means the
# model named a determinant we did not compute → reject the whole answer. Over-matching
# is the SAFE direction here: a false positive just reverts to the template.
_GENE_PATTERNS: tuple[str, ...] = (
    r"bla[A-Za-z0-9-]+",  # beta-lactamases: blaCTX-M-15, blaTEM-1, blaKPC-3
    r"[a-z]{3,4}_[A-Z]\d+[A-Z]",  # QRDR/point mutations: gyrA_S83F, parC_S80I
    # acquired-gene families: aac(6')-Ib, tet(A), qnrS1, sul1, dfrA14, mcr-1, aph(3')-Ia
    r"(?:aac|aad|aph|ant|tet|qnr|sul|dfr|erm|mph|mef|cat|cml|flo|fos|van|mcr|arr|oqx|qep|sat|ere|str)"
    r"[A-Z0-9(][A-Za-z0-9()'\-]*",
    # resistance target genes, optionally carrying a mutation: gyrA, parC, rpoB, gyrA_S83F
    r"(?:gyr|par|rpo|rrs|pmr|mgr|pho|pbp)[A-Z](?:_[A-Z]\d+[A-Z])?",
)
_GENE_RE = re.compile("|".join(f"(?:{p})" for p in _GENE_PATTERNS))


def _mentions_foreign_gene(text: str, allowed: set[str]) -> bool:
    """True if `text` names any gene-like token outside `allowed` (case-insensitive)."""
    allowed_cf = {g.casefold() for g in allowed}
    return any(m.group(0).casefold() not in allowed_cf for m in _GENE_RE.finditer(text))


def _load_disk_cache() -> dict[_DiskKey, str]:
    """Read the pre-baked rationale cache once, keyed by (genome, drug, call, evidence).

    A missing or unreadable file is a legitimate optional-feature absence — the cache is a
    reliability artifact (data/rationale_cache.json is committed, so a fresh clone has it; an
    un-baked genome is simply absent) — NOT an error: degrade to the live path by returning an
    empty map. Held for the process (the file does not change under a running demo).

    Every record is validated per-record: a record missing a field, or whose key parts / text
    are not plain strings, is SKIPPED (not fatal to the whole cache). This keeps explain()'s
    never-raise contract absolute — a non-str `text` would otherwise raise inside the gene
    regex — and lets one corrupt/old-format record degrade gracefully to a live miss."""
    global _disk_cache
    if _disk_cache is not None:
        return _disk_cache
    cache: dict[_DiskKey, str] = {}
    try:
        records = json.loads(_DISK_CACHE_PATH.read_text(encoding="utf-8"))[
            _DISK_RECORDS_KEY
        ]
    except (OSError, ValueError, KeyError, TypeError):
        _disk_cache = cache
        return cache
    for r in records:
        try:
            key = (
                r[_DISK_GENOME_KEY],
                r[_DISK_ANTIBIOTIC_KEY],
                r[_DISK_CALL_KEY],
                r[_DISK_EVIDENCE_KEY],
            )
            text = r[_DISK_TEXT_KEY]
        except (KeyError, TypeError):
            continue  # malformed / old-format record → skip, don't disable the whole cache
        if all(isinstance(v, str) for v in (*key, text)):
            cache[key] = text
    _disk_cache = cache
    return cache


def disk_cache_available() -> bool:
    """True if any pre-baked rationales are present. The hosted demo can then show genuine OpenAI
    phrasings with NO key — they were phrased by OpenAI at bake time, served from disk here."""
    return bool(_load_disk_cache())


def write_disk_cache(records: list[dict[str, object]]) -> Path:
    """Persist pre-baked rationale records to data/rationale_cache.json (bake script only).

    Sorted + indented so the artifact is stable and diffable across re-bakes. Written
    atomically (torn-write safe, matching the finalize-chain artifacts). Owns the on-disk
    format so `_load_disk_cache` and the bake script agree in exactly one place."""
    ordered = sorted(
        records,
        key=lambda r: (
            r[_DISK_GENOME_KEY],
            r[_DISK_ANTIBIOTIC_KEY],
            r[_DISK_CALL_KEY],
            r[_DISK_EVIDENCE_KEY],
        ),
    )
    payload = {_DISK_RECORDS_KEY: ordered}
    return atomic_write(
        _DISK_CACHE_PATH,
        lambda tmp: tmp.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        ),
    )


def make_disk_record(
    prediction: DrugPrediction, genome_id: str, text: str
) -> dict[str, object]:
    """Build one on-disk cache record from a prediction + its baked OpenAI phrasing."""
    return {
        _DISK_GENOME_KEY: genome_id,
        _DISK_ANTIBIOTIC_KEY: prediction.antibiotic,
        _DISK_CALL_KEY: prediction.call,
        _DISK_EVIDENCE_KEY: prediction.evidence_category,
        _DISK_GENES_KEY: list(prediction.supporting_genes),
        _DISK_TEXT_KEY: text,
    }


def _format_genes(genes: list[str]) -> str:
    """Human list: 'x', 'x and y', 'x, y, and z'."""
    if len(genes) == 1:
        return genes[0]
    if len(genes) == 2:
        return f"{genes[0]} and {genes[1]}"
    return ", ".join(genes[:-1]) + f", and {genes[-1]}"


def _template(prediction: DrugPrediction) -> str:
    """Deterministic rationale built ONLY from the prediction's own fields.

    It can never name a gene absent from `supporting_genes` because those genes are the
    only external strings it interpolates.
    """
    drug = prediction.antibiotic
    if prediction.evidence_category == EVIDENCE_STATISTICAL:
        verdict = prediction.call.replace("_", " ")
        return (
            f"Predicted {verdict} for {drug} from a statistical model signal without a "
            f"single known resistance mechanism. {_CONFIRM}"
        )
    if prediction.call == CALL_RESISTANT:
        if prediction.supporting_genes:
            genes = _format_genes(prediction.supporting_genes)
            return (
                f"Predicted resistant to {drug}: the genome carries {genes}, "
                f"{_KNOWN_DETERMINANT}. {_CONFIRM}"
            )
        # No acquired gene but still resistant → intrinsic/expected phenotype.
        return (
            f"Predicted resistant to {drug}: resistance is expected for this organism "
            f"even without an acquired gene. {_CONFIRM}"
        )
    if prediction.call == CALL_SUSCEPTIBLE:
        return (
            f"Predicted susceptible to {drug}: no resistance determinant was detected. "
            f"{_CONFIRM}"
        )
    # CALL_NO_CALL — the honest default: silence is not susceptibility.
    return (
        f"No call for {drug}: no known resistance determinant was detected, and absence "
        f"of a resistance gene is not evidence of susceptibility. {_CONFIRM}"
    )


# --- Cost accounting ---------------------------------------------------------
def _estimated_spend_usd() -> float:
    return (_prompt_tokens_used / _PER_MILLION) * _INPUT_USD_PER_1M + (
        _completion_tokens_used / _PER_MILLION
    ) * _OUTPUT_USD_PER_1M


def _would_exceed_cap() -> bool:
    """True if one more worst-case call could push cumulative spend past the cap.

    Worst case = MAX_TOKENS of both input and output; conservative on purpose so we
    stop BEFORE crossing $20, never after.
    """
    per_call = (MAX_TOKENS / _PER_MILLION) * (_INPUT_USD_PER_1M + _OUTPUT_USD_PER_1M)
    return _estimated_spend_usd() + per_call > HARD_CAP_USD


def _record_usage(prompt_tokens: int, completion_tokens: int) -> None:
    global _prompt_tokens_used, _completion_tokens_used
    _prompt_tokens_used += prompt_tokens
    _completion_tokens_used += completion_tokens


# --- Prompt rendering + LLM call ---------------------------------------------
def _render_prompt(prediction: DrugPrediction) -> tuple[str, str]:
    """Load the versioned prompt and fill it with ONLY the four safe payload fields."""
    raw = _PROMPT_PATH.read_text(encoding="utf-8")
    system_part, _, user_part = raw.partition(_PROMPT_DELIMITER)
    genes = ", ".join(prediction.supporting_genes) or _NO_GENES
    payload = {
        "antibiotic": prediction.antibiotic,
        "call": prediction.call,
        "evidence_category": prediction.evidence_category,
        "supporting_genes": genes,
    }
    return system_part.strip().format(**payload), user_part.strip().format(**payload)


def _llm_rationale(prediction: DrugPrediction) -> str | None:
    """Call OpenAI once. Returns the text, or None if the cost cap or an empty response
    should force the caller back to the template. Raises on any transport/SDK error —
    the caller catches everything."""
    if _would_exceed_cap():
        return None
    # Lazy import: openai lives in the optional 'demo' extra; absence must degrade, not crash.
    from openai import OpenAI

    system_prompt, user_prompt = _render_prompt(prediction)
    client = OpenAI(timeout=OPENAI_TIMEOUT_S, max_retries=OPENAI_MAX_RETRIES)
    # gpt-5.4-mini takes max_completion_tokens (not max_tokens) and only the default temperature
    # (verified against the API). Demo-path determinism comes from the caches, not a temperature.
    response = client.chat.completions.create(
        model=MODEL,
        max_completion_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    usage = response.usage
    if usage is not None:
        _record_usage(usage.prompt_tokens, usage.completion_tokens)
    content = response.choices[0].message.content
    return content.strip() if content else None


def _llm_worthy(prediction: DrugPrediction) -> bool:
    """Whether an OpenAI phrasing is worth spending on — and SAFE to spend on — this prediction.

    Only a resistance call (name the genes / intrinsic expectation) or a mechanism-free
    statistical signal carries a real narrative for the LLM to phrase. A no-call — and a plain
    no-signal susceptible — has exactly ONE honest sentence: the template's. Letting an LLM
    rephrase it risks softening "we will not commit" (or "no determinant detected") toward
    "probably fine", the single thing a safety interlock must never imply, for zero narrative
    gain. So the library refuses to route those through OpenAI at all. This is also the dominant
    demo-latency guard: the mechanism report is mostly no-calls, so gating here keeps the
    synchronous Streamlit rationale loop to the handful of resistance rows instead of every drug.
    """
    return (
        prediction.call == CALL_RESISTANT
        or prediction.evidence_category == EVIDENCE_STATISTICAL
    )


def explain(
    prediction: DrugPrediction, genome_id: str, *, use_llm: bool = False
) -> str:
    """A 1–2 sentence plain-language rationale for one drug prediction.

    Deterministic template by default. With `use_llm=True` it first serves a pre-baked OpenAI
    phrasing from the on-disk cache when present (curated demo beats — no network, and NO key
    needed, so the hosted demo shows genuine OpenAI text); on a cache MISS it tries a constrained
    LIVE OpenAI phrasing, which needs a key. ANY failure (no key on a miss, no SDK, API error, cost
    cap, or a fabricated gene) silently returns the template. It never raises and never surfaces a
    gene outside `prediction.supporting_genes` (the disk hit is re-validated against the current
    genes too). A no-call / plain-susceptible prediction always returns the template (`_llm_worthy`).
    """
    template = _template(prediction)
    if not use_llm or not _llm_worthy(prediction):
        return template

    cache_key = (genome_id, prediction.antibiotic, prediction.call)
    cached = _rationale_cache.get(cache_key)
    if cached is not None:
        return cached

    # Pre-baked disk cache: an OpenAI phrasing computed offline. Safe to serve WITHOUT a live key
    # (no network, no cost), so the hosted demo shows genuine OpenAI text from the committed cache.
    # Re-run the NO-FABRICATION gate against the CURRENT genes — a finalize could have shifted a
    # genome's determinants under a stale entry, so a cached line that now names a foreign gene is
    # rejected here (fall through to a live call / template), never surfaced on stage.
    disk_key = (*cache_key, prediction.evidence_category)
    disk_text = _load_disk_cache().get(disk_key)
    if disk_text is not None and not _mentions_foreign_gene(
        disk_text, set(prediction.supporting_genes)
    ):
        _rationale_cache[cache_key] = disk_text
        return disk_text

    # A cache MISS needs a fresh phrasing, and only that LIVE call requires a key — no key → template.
    if not os.environ.get(_ENV_API_KEY):
        return template

    try:
        text = _llm_rationale(prediction)
    except Exception:  # noqa: BLE001 — a rationale is never worth crashing the pipeline
        return template
    if text is None:
        return template

    # NO-FABRICATION gate: an answer that names a determinant we did not compute is
    # worse than a plain template — reject it outright, do not surface it.
    if _mentions_foreign_gene(text, set(prediction.supporting_genes)):
        return template

    _rationale_cache[cache_key] = text
    return text
