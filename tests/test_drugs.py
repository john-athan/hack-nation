"""Tests for the drug-properties DB, determinant matcher, and target-presence gate."""

from __future__ import annotations

from genome_firewall.drugs import (
    DRUG_DB,
    drug_matches_determinant,
    get_drug,
    target_present,
)


def test_get_drug_hit_is_case_insensitive() -> None:
    drug = get_drug("Ciprofloxacin")
    assert drug is not None
    assert drug.name == "ciprofloxacin"
    assert drug.drug_class == "fluoroquinolone"


def test_get_drug_miss_returns_none() -> None:
    assert get_drug("not-a-real-drug") is None


def test_db_key_equals_drug_name() -> None:
    # The dict key is the contract used by callers; it must equal the canonical name.
    for key, drug in DRUG_DB.items():
        assert key == drug.name


def test_expected_narms_drugs_present() -> None:
    expected = {
        "ampicillin",
        "amoxicillin-clavulanic acid",
        "ceftriaxone",
        "cefoxitin",
        "meropenem",
        "ciprofloxacin",
        "nalidixic acid",
        "azithromycin",
        "gentamicin",
        "streptomycin",
        "tetracycline",
        "chloramphenicol",
        "trimethoprim-sulfamethoxazole",
    }
    assert set(DRUG_DB) == expected


def test_beta_lactam_matches_ampicillin_not_ciprofloxacin() -> None:
    amp = get_drug("ampicillin")
    cip = get_drug("ciprofloxacin")
    assert amp is not None and cip is not None
    # A plain penicillinase determinant (blaTEM-1 style) confers ampicillin R.
    assert drug_matches_determinant(amp, "BETA-LACTAM", "BETA-LACTAM")
    assert not drug_matches_determinant(cip, "BETA-LACTAM", "BETA-LACTAM")


def test_quinolone_matches_ciprofloxacin() -> None:
    cip = get_drug("ciprofloxacin")
    assert cip is not None
    assert drug_matches_determinant(cip, "QUINOLONE", "QUINOLONE")
    # Compound class (aac(6')-Ib-cr) still matches via the QUINOLONE token.
    assert drug_matches_determinant(
        cip, "AMINOGLYCOSIDE/QUINOLONE", "AMIKACIN/KANAMYCIN/QUINOLONE/TOBRAMYCIN"
    )


def test_beta_lactam_subclass_gate_discriminates() -> None:
    ceftriaxone = get_drug("ceftriaxone")
    meropenem = get_drug("meropenem")
    assert ceftriaxone is not None and meropenem is not None
    # ESBL/AmpC (CEPHALOSPORIN) confers ceftriaxone R but a plain penicillinase does not.
    assert drug_matches_determinant(ceftriaxone, "BETA-LACTAM", "CEPHALOSPORIN")
    assert not drug_matches_determinant(ceftriaxone, "BETA-LACTAM", "BETA-LACTAM")
    # Meropenem only bows to a carbapenemase, not to an ESBL.
    assert drug_matches_determinant(meropenem, "BETA-LACTAM", "CARBAPENEM")
    assert not drug_matches_determinant(meropenem, "BETA-LACTAM", "CEPHALOSPORIN")


def test_esbl_does_not_over_call_amox_clav_or_cefoxitin() -> None:
    # Regression guard: a bare CEPHALOSPORIN determinant (an ESBL like blaSHV-2A, which is
    # clavulanate-inhibited and does not hydrolyze cephamycins) must NOT confer amox-clav or
    # cefoxitin resistance — both are lab-susceptible on such genomes, and at class granularity
    # an ESBL is indistinguishable from AmpC. Asserting resistance here contradicted the lab
    # phenotype on the flagship demo genome. Only an unambiguous carbapenemase counts.
    amox = get_drug("amoxicillin-clavulanic acid")
    cefoxitin = get_drug("cefoxitin")
    ceftriaxone = get_drug("ceftriaxone")
    assert amox is not None and cefoxitin is not None and ceftriaxone is not None
    assert not drug_matches_determinant(amox, "BETA-LACTAM", "CEPHALOSPORIN")
    assert not drug_matches_determinant(cefoxitin, "BETA-LACTAM", "CEPHALOSPORIN")
    # ceftriaxone is defeated by any CEPHALOSPORIN-subclass beta-lactamase (ESBL or AmpC),
    # unambiguously — it keeps the CEPHALOSPORIN gate.
    assert drug_matches_determinant(ceftriaxone, "BETA-LACTAM", "CEPHALOSPORIN")
    # A genuine carbapenemase does defeat all three (broadest hydrolysis).
    assert drug_matches_determinant(amox, "BETA-LACTAM", "CARBAPENEM")
    assert drug_matches_determinant(cefoxitin, "BETA-LACTAM", "CARBAPENEM")


def test_aminoglycoside_subclass_separates_gentamicin_from_streptomycin() -> None:
    gent = get_drug("gentamicin")
    strep = get_drug("streptomycin")
    assert gent is not None and strep is not None
    # aac(3) -> GENTAMICIN token; aadA -> SPECTINOMYCIN/STREPTOMYCIN.
    assert drug_matches_determinant(
        gent, "AMINOGLYCOSIDE", "GENTAMICIN/KANAMYCIN/TOBRAMYCIN"
    )
    assert not drug_matches_determinant(
        gent, "AMINOGLYCOSIDE", "SPECTINOMYCIN/STREPTOMYCIN"
    )
    assert drug_matches_determinant(
        strep, "AMINOGLYCOSIDE", "SPECTINOMYCIN/STREPTOMYCIN"
    )
    assert not drug_matches_determinant(strep, "AMINOGLYCOSIDE", "GENTAMICIN")


def test_matcher_is_case_insensitive() -> None:
    amp = get_drug("ampicillin")
    assert amp is not None
    assert drug_matches_determinant(amp, "beta-lactam", "beta-lactam")


def test_target_present_defaults_true_on_empty_set() -> None:
    # Empty gene-presence set => assume core targets present (no susceptibility fallacy).
    for drug in DRUG_DB.values():
        assert target_present(drug, set())


def test_target_present_checks_non_empty_set_honestly() -> None:
    cip = get_drug("ciprofloxacin")
    assert cip is not None
    # gyrA is a ciprofloxacin target; a set lacking any target returns False.
    assert target_present(cip, {"gyrA", "some_other_gene"})
    assert not target_present(cip, {"blaTEM-1", "sul1"})
