"""Hermetic tests for the constrained-rationale layer. NO network / no OpenAI calls."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from genome_firewall.constants import (
    CALL_NO_CALL,
    CALL_RESISTANT,
    CALL_SUSCEPTIBLE,
    EVIDENCE_KNOWN_GENE,
    EVIDENCE_NO_SIGNAL,
    EVIDENCE_STATISTICAL,
)
from genome_firewall import rationale
from genome_firewall.rationale import (
    _ENV_API_KEY,
    OPENAI_MAX_RETRIES,
    OPENAI_TIMEOUT_S,
    _mentions_foreign_gene,
    _template,
    explain,
    make_disk_record,
    write_disk_cache,
)
from genome_firewall.schema import DrugPrediction

_GENOME = "GENOME_TEST_1"


def _pred(call: str, evidence: str, genes: list[str]) -> DrugPrediction:
    return DrugPrediction(
        antibiotic="ampicillin",
        call=call,
        confidence=0.9,
        evidence_category=evidence,
        supporting_genes=genes,
        target_present=True,
    )


def test_template_resistant_known_gene_names_only_supporting_genes() -> None:
    genes = ["blaTEM-1", "aac(6')-Ib"]
    pred = _pred(CALL_RESISTANT, EVIDENCE_KNOWN_GENE, genes)
    text = _template(pred)
    assert "ampicillin" in text
    for gene in genes:
        assert gene in text
    # It must not name any gene beyond the supporting set.
    assert not _mentions_foreign_gene(text, set(genes))


def test_template_no_call_states_absence_is_not_susceptibility() -> None:
    pred = _pred(CALL_NO_CALL, EVIDENCE_NO_SIGNAL, [])
    text = _template(pred)
    assert "ampicillin" in text
    assert "not evidence of susceptibility" in text
    # No genes to cite, and none must appear.
    assert not _mentions_foreign_gene(text, set())


def test_template_susceptible_notes_confirmation() -> None:
    pred = _pred(CALL_SUSCEPTIBLE, EVIDENCE_NO_SIGNAL, [])
    text = _template(pred)
    assert "ampicillin" in text
    assert "susceptible" in text.lower()
    assert "laboratory" in text.lower()
    assert not _mentions_foreign_gene(text, set())


def test_mentions_foreign_gene_allows_listed_gene() -> None:
    assert (
        _mentions_foreign_gene("carries blaCTX-M, a beta-lactamase", {"blaCTX-M"})
        is False
    )


def test_mentions_foreign_gene_flags_unlisted_gene() -> None:
    assert _mentions_foreign_gene("carries blaCTX-M-15", {"tet(A)"}) is True


def test_mentions_foreign_gene_ignores_plain_prose() -> None:
    text = "No call: confirm the result with standard laboratory testing."
    assert _mentions_foreign_gene(text, set()) is False


def test_explain_without_key_returns_template_and_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_ENV_API_KEY, raising=False)
    pred = _pred(CALL_RESISTANT, EVIDENCE_KNOWN_GENE, ["blaTEM-1"])
    result = explain(pred, _GENOME, use_llm=True)
    assert result == _template(pred)


def test_explain_default_is_template(monkeypatch: pytest.MonkeyPatch) -> None:
    # Even with a key present, use_llm=False must stay fully offline.
    monkeypatch.setenv(_ENV_API_KEY, "sk-should-not-be-used")
    pred = _pred(CALL_NO_CALL, EVIDENCE_NO_SIGNAL, [])
    assert explain(pred, _GENOME, use_llm=False) == _template(pred)


def _install_fake_openai(
    monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any], *, raises: bool
) -> None:
    """Inject a hermetic `openai` module whose OpenAI() records its constructor kwargs.
    With raises=True, `.create()` raises to simulate a wifi timeout on stage."""

    def _raise(**_: Any) -> Any:
        raise RuntimeError("simulated wifi timeout")

    def _empty_response(**_: Any) -> Any:
        # A well-formed response whose content is empty → _llm_rationale returns None
        # → explain() serves the template, without any real network call.
        message = types.SimpleNamespace(content=None)
        return types.SimpleNamespace(
            usage=None, choices=[types.SimpleNamespace(message=message)]
        )

    class _FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(
                    create=_raise if raises else _empty_response
                )
            )

    module = types.ModuleType("openai")
    module.OpenAI = _FakeOpenAI  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "openai", module)


def test_llm_client_is_bounded_by_timeout_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The live call MUST be constructed with a hard timeout + bounded retries so venue
    # wifi can never hang the synchronous Streamlit render for the SDK's 600s×2 default.
    monkeypatch.setenv(_ENV_API_KEY, "sk-test")
    captured: dict[str, Any] = {}
    _install_fake_openai(monkeypatch, captured, raises=False)
    pred = _pred(CALL_RESISTANT, EVIDENCE_KNOWN_GENE, ["blaTEM-1"])
    # An empty (None) response falls back to template, but the client was still built.
    explain(pred, _GENOME, use_llm=True)
    assert captured["timeout"] == OPENAI_TIMEOUT_S
    assert captured["max_retries"] == OPENAI_MAX_RETRIES


def test_explain_degrades_to_template_when_llm_call_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A timeout (or any transport error) must degrade to the deterministic template,
    # never propagate and crash the demo render.
    monkeypatch.setenv(_ENV_API_KEY, "sk-test")
    _install_fake_openai(monkeypatch, {}, raises=True)
    pred = _pred(CALL_RESISTANT, EVIDENCE_KNOWN_GENE, ["blaTEM-1"])
    assert explain(pred, _GENOME, use_llm=True) == _template(pred)


_LLM_MARKER = "LLM phrasing marker — confirm with lab."  # no gene-like tokens


def _install_counting_openai(monkeypatch: pytest.MonkeyPatch, calls: list[int]) -> None:
    """Fake `openai` whose create() records each invocation and returns a fixed rationale."""

    def _create(**_: Any) -> Any:
        calls.append(1)
        message = types.SimpleNamespace(content=_LLM_MARKER)
        return types.SimpleNamespace(
            usage=None, choices=[types.SimpleNamespace(message=message)]
        )

    class _FakeOpenAI:
        def __init__(self, **_: Any) -> None:
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=_create)
            )

    module = types.ModuleType("openai")
    module.OpenAI = _FakeOpenAI  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "openai", module)


def test_explain_spends_llm_on_resistant_but_never_on_no_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The mechanism report is mostly no-calls; routing each through OpenAI both wastes the
    # synchronous demo loop AND lets the model soften a no-call toward susceptibility. A no-call
    # must ALWAYS return the fixed template without an LLM call; a resistance call must use it.
    # (Regression guard: the demo's display filter admits both, so this gate lives in explain().)
    monkeypatch.setenv(_ENV_API_KEY, "sk-test")
    calls: list[int] = []
    _install_counting_openai(monkeypatch, calls)

    no_call = _pred(CALL_NO_CALL, EVIDENCE_NO_SIGNAL, [])
    assert explain(no_call, _GENOME, use_llm=True) == _template(no_call)
    assert calls == []  # the LLM was never invoked for a no-call

    resistant = _pred(CALL_RESISTANT, EVIDENCE_KNOWN_GENE, ["blaTEM-1"])
    assert explain(resistant, _GENOME, use_llm=True) == _LLM_MARKER
    assert len(calls) == 1  # exactly one live call, for the resistance row


# --- On-disk pre-baked cache -------------------------------------------------
def test_disk_cache_hit_serves_baked_phrasing_without_a_live_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The whole point of the disk cache: a curated beat's rationale is served instantly from
    # disk, so flaky venue wifi can never stall the money slide behind a live OpenAI call.
    monkeypatch.setenv(_ENV_API_KEY, "sk-test")
    calls: list[int] = []
    _install_counting_openai(monkeypatch, calls)
    pred = _pred(CALL_RESISTANT, EVIDENCE_KNOWN_GENE, ["blaTEM-1"])
    genome = "DISK_HIT_GENOME"
    baked = "Predicted resistant to ampicillin; confirm with laboratory testing."
    monkeypatch.setattr(
        rationale,
        "_disk_cache",
        {(genome, pred.antibiotic, pred.call, pred.evidence_category): baked},
    )
    assert explain(pred, genome, use_llm=True) == baked
    assert calls == []  # served from disk — OpenAI was never touched


def test_stale_disk_entry_naming_a_foreign_gene_is_rejected_not_surfaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A finalize can shift a genome's determinants under a stale cache entry. A cached line
    # that now names a gene the current prediction does not carry must be rejected (fall
    # through to the live path), never surfaced — the NO-FABRICATION gate re-runs on disk hits.
    monkeypatch.setenv(_ENV_API_KEY, "sk-test")
    calls: list[int] = []
    _install_counting_openai(
        monkeypatch, calls
    )  # live path returns the gene-free _LLM_MARKER
    pred = _pred(CALL_RESISTANT, EVIDENCE_KNOWN_GENE, ["tet(A)"])
    genome = "STALE_DISK_GENOME"
    stale = "Resistant to ampicillin via blaCTX-M-15."  # a gene NOT in supporting_genes
    monkeypatch.setattr(
        rationale,
        "_disk_cache",
        {(genome, pred.antibiotic, pred.call, pred.evidence_category): stale},
    )
    result = explain(pred, genome, use_llm=True)
    assert result == _LLM_MARKER  # fell through to the live call
    assert result != stale  # the stale foreign-gene line was NOT shown
    assert len(calls) == 1


def test_evidence_flip_misses_the_disk_entry_and_serves_the_live_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # evidence_category is part of the disk key, so a genome whose call is unchanged but whose
    # evidence BASIS flipped (statistical → a mechanism newly appears) can never serve the stale
    # "statistical signal without a known mechanism" phrasing for a now-mechanism-grounded call.
    monkeypatch.setenv(_ENV_API_KEY, "sk-test")
    calls: list[int] = []
    _install_counting_openai(monkeypatch, calls)
    genome = "EVIDENCE_FLIP_GENOME"
    stale = "Predicted resistant to ampicillin from a statistical model signal."
    # Cache was baked when evidence was statistical_only …
    monkeypatch.setattr(
        rationale,
        "_disk_cache",
        {(genome, "ampicillin", CALL_RESISTANT, EVIDENCE_STATISTICAL): stale},
    )
    # … but the current prediction now has a known-gene mechanism → the key misses.
    now = _pred(CALL_RESISTANT, EVIDENCE_KNOWN_GENE, ["blaTEM-1"])
    result = explain(now, genome, use_llm=True)
    assert (
        result == _LLM_MARKER
    )  # clean miss → live path, stale statistical sentence not served
    assert len(calls) == 1


def test_non_string_cache_text_is_skipped_so_explain_never_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    # explain()'s never-raise contract must hold even against a corrupt/hand-edited cache: a
    # non-str `text` (a JSON number/list) would raise inside the gene regex if loaded, so the
    # loader drops such records → the row falls through to the live path, never crashes the demo.
    monkeypatch.setenv(_ENV_API_KEY, "sk-test")
    calls: list[int] = []
    _install_counting_openai(monkeypatch, calls)
    path = tmp_path / "rationale_cache.json"
    path.write_text(
        '{"rationales": [{"genome_id": "G", "antibiotic": "ampicillin", "call": '
        f'"{CALL_RESISTANT}", "evidence_category": "{EVIDENCE_KNOWN_GENE}", '
        '"supporting_genes": [], "text": 42}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(rationale, "_DISK_CACHE_PATH", path)
    monkeypatch.setattr(rationale, "_disk_cache", None)
    assert rationale._load_disk_cache() == {}  # the non-str-text record was dropped
    pred = _pred(CALL_RESISTANT, EVIDENCE_KNOWN_GENE, ["blaTEM-1"])
    assert explain(pred, "G", use_llm=True) == _LLM_MARKER  # live path, no raise


def test_write_disk_cache_roundtrips_through_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    path = tmp_path / "rationale_cache.json"
    monkeypatch.setattr(rationale, "_DISK_CACHE_PATH", path)
    monkeypatch.setattr(rationale, "_disk_cache", None)
    pred = _pred(CALL_RESISTANT, EVIDENCE_KNOWN_GENE, ["blaTEM-1"])
    text = "Predicted resistant to ampicillin; confirm with laboratory testing."
    write_disk_cache([make_disk_record(pred, "G1", text)])
    loaded = rationale._load_disk_cache()
    assert loaded[("G1", "ampicillin", CALL_RESISTANT, EVIDENCE_KNOWN_GENE)] == text


def test_load_disk_cache_missing_or_malformed_file_is_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    # An absent cache is a legitimate optional-feature absence (box-local, gitignored) — it must
    # degrade to the live path, not crash. A corrupt file is treated the same way.
    monkeypatch.setattr(rationale, "_DISK_CACHE_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(rationale, "_disk_cache", None)
    assert rationale._load_disk_cache() == {}

    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json", encoding="utf-8")
    monkeypatch.setattr(rationale, "_DISK_CACHE_PATH", bad)
    monkeypatch.setattr(rationale, "_disk_cache", None)
    assert rationale._load_disk_cache() == {}
