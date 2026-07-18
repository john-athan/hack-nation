"""Tests for cohort selection (ADR-0005: keep all positives, cap susceptible majority)."""

from __future__ import annotations

import pandas as pd

from genome_firewall.cohort import select_cohort, stratified_order


def _meta(serovar_of: dict[str, str], mlst: str = "MLST.x") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "genome_id": list(serovar_of),
            "serovar": list(serovar_of.values()),
            "mlst": [mlst] * len(serovar_of),
            "genome_length": ["4800000"] * len(serovar_of),
            "contigs": ["50"] * len(serovar_of),
            "genome_status": ["WGS"] * len(serovar_of),
        }
    )


def _labels(spec: dict[str, str]) -> pd.DataFrame:
    # spec: genome_id -> phenotype for a single drug "ampicillin".
    return pd.DataFrame(
        {
            "genome_id": list(spec),
            "antibiotic": ["ampicillin"] * len(spec),
            "label": list(spec.values()),
        }
    )


def test_keeps_every_resistant_isolate_uncapped() -> None:
    # 10 resistant clones in one serovar+mlst group must ALL survive (never capped).
    spec = {f"r{i}": "Resistant" for i in range(10)}
    serovar_of = {g: "Typhimurium" for g in spec}
    cohort = select_cohort(_labels(spec), _meta(serovar_of), sus_cap_per_group=2)
    assert set(cohort["genome_id"]) == set(spec)  # all 10 kept
    assert cohort["is_resistant"].all()


def test_caps_susceptible_majority_per_group() -> None:
    spec = {f"s{i}": "Susceptible" for i in range(10)}
    serovar_of = {g: "Enteritidis" for g in spec}
    cohort = select_cohort(_labels(spec), _meta(serovar_of), sus_cap_per_group=3)
    assert len(cohort) == 3  # capped
    assert not cohort["is_resistant"].any()


def test_resistant_kept_susceptible_capped_together() -> None:
    spec = {"r0": "Resistant", "r1": "Resistant"}
    spec |= {f"s{i}": "Susceptible" for i in range(5)}
    serovar_of = dict.fromkeys(spec, "Newport")
    cohort = select_cohort(_labels(spec), _meta(serovar_of), sus_cap_per_group=2)
    got = set(cohort["genome_id"])
    assert {"r0", "r1"} <= got  # both resistant kept
    assert len(got & {f"s{i}" for i in range(5)}) == 2  # susceptible capped to 2


def test_qc_drops_junk_assemblies_even_if_resistant() -> None:
    spec = {"good": "Resistant", "junk": "Resistant"}
    meta = _meta({"good": "A", "junk": "B"})
    meta.loc[meta["genome_id"] == "junk", "contigs"] = "5000"  # too fragmented
    cohort = select_cohort(_labels(spec), meta)
    assert set(cohort["genome_id"]) == {"good"}


def test_stratified_order_is_round_robin_across_groups() -> None:
    # Dominant group A (5) + rare groups B, C (1 each). A short prefix must already
    # touch every group — that is the whole point (partial annotate stays diverse).
    group_of = {f"a{i}": "A" for i in range(5)} | {"b0": "B", "c0": "C"}
    order = stratified_order(list(group_of), group_of)
    assert set(order) == set(group_of)  # permutation, nothing dropped
    assert set(group_of[g] for g in order[:3]) == {"A", "B", "C"}  # breadth-first


def test_stratified_order_deterministic_and_fallback_bucketed() -> None:
    group_of = {"x1": "S", "x2": "S"}  # y1/y2 have no group → fallback bucket
    ids = ["y2", "x2", "y1", "x1"]
    first = stratified_order(ids, group_of, fallback="__none__")
    assert first == stratified_order(list(reversed(ids)), group_of, fallback="__none__")
    assert set(first) == set(ids)  # unmapped ids still included, never lost
