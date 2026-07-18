"""Tests for the determinism-first core: labels, AMRFinder parsing, features, schema."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import pytest

from genome_firewall.amrfinder import parse_tsv
from genome_firewall.errors import UnknownDrugError
from genome_firewall.features import build_matrix, determinants_for_genome
from genome_firewall.labels import canonical_drug, harmonize
from genome_firewall.report import build_report
from genome_firewall.schema import DrugPrediction, GenomeReport

# A minimal AMRFinderPlus v4 TSV: one AMR gene, one POINT mutation, one VIRULENCE (dropped).
_FIXTURE_TSV = (
    "Name\tElement symbol\tElement subtype\tClass\tSubclass\n"
    "g1\tblaTEM-1\tAMR\tBETA-LACTAM\tBETA-LACTAM\n"
    "g1\tgyrA_S83F\tPOINT\tQUINOLONE\tQUINOLONE\n"
    "g1\tstaphylococcal\tVIRULENCE\tVIRULENCE\tVIRULENCE\n"
)


def test_parse_tsv_keeps_amr_and_point_drops_virulence(tmp_path: Path) -> None:
    tsv = tmp_path / "28901.99999.tsv"  # stem becomes genome_id
    tsv.write_text(_FIXTURE_TSV)
    rows = parse_tsv(tsv)
    assert set(rows["symbol"]) == {"blaTEM-1", "gyrA_S83F"}
    assert set(rows["subtype"]) == {"AMR", "POINT"}
    assert (rows["genome_id"] == "28901.99999").all()


def test_parse_tsv_empty_is_ok(tmp_path: Path) -> None:
    tsv = tmp_path / "empty.tsv"
    tsv.write_text("")
    assert parse_tsv(tsv).empty


def test_parse_tsv_missing_class_columns_does_not_crash(tmp_path: Path) -> None:
    # A TSV without Class/Subclass must degrade to empty strings, never crash.
    tsv = tmp_path / "g1.tsv"
    tsv.write_text("Name\tElement symbol\tElement subtype\ng1\tblaTEM-1\tAMR\n")
    rows = parse_tsv(tsv)
    assert list(rows["symbol"]) == ["blaTEM-1"]
    assert list(rows["drug_class"]) == [""]
    assert list(rows["subclass"]) == [""]


def test_build_matrix_and_determinants(tmp_path: Path) -> None:
    tsv = tmp_path / "g1.tsv"
    tsv.write_text(_FIXTURE_TSV)
    rows = parse_tsv(tsv)
    mat = build_matrix(rows)
    assert mat.loc["g1", "blaTEM-1"] == 1
    assert mat.loc["g1", "gyrA_S83F"] == 1
    dets = determinants_for_genome(rows, "g1")
    assert {d.symbol for d in dets} == {"blaTEM-1", "gyrA_S83F"}
    assert any(d.subtype == "POINT" for d in dets)


def test_harmonize_maps_phenotypes_and_dedups() -> None:
    raw = pd.DataFrame(
        {
            "genome_id": ["a", "a", "a", "b", "c"],
            "antibiotic": [
                "ampicillin",
                "ampicillin",
                "ampicillin",
                "ampicillin",
                "ampicillin",
            ],
            "resistant_phenotype": [
                "Resistant",
                "Resistant",
                "Susceptible",  # a: 2R vs 1S → Resistant
                "Nonsusceptible",  # b → Resistant
                "Intermediate",  # c → dropped entirely
            ],
        }
    )
    clean = harmonize(raw)
    got = dict(zip(clean["genome_id"], clean["label"], strict=True))
    assert got == {"a": "Resistant", "b": "Resistant"}


def test_canonical_drug_synonyms() -> None:
    assert (
        canonical_drug("Trimethoprim/Sulfamethoxazole")
        == "trimethoprim-sulfamethoxazole"
    )
    assert canonical_drug("Ampicillin") == "ampicillin"


def test_build_report_rejects_unknown_drug() -> None:
    rows = parse_tsv_stub()
    with pytest.raises(UnknownDrugError):
        build_report(rows, "g1", drugs=["not-a-real-drug"])


def parse_tsv_stub() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["genome_id", "symbol", "subtype", "drug_class", "subclass"]
    )


def test_schema_roundtrip() -> None:
    pred = DrugPrediction(
        "ampicillin", "resistant", 0.9, "known_gene", ["blaTEM-1"], True
    )
    report = GenomeReport("g1", [pred])
    d = report.to_dict()
    assert d["genome_id"] == "g1"
    assert d["predictions"][0]["supporting_genes"] == ["blaTEM-1"]  # ty: ignore[not-subscriptable]  # deep-index into dict[str, object]
    assert "laboratory" in str(d["disclaimer"]).lower()
