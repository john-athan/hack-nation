"""Determinism-first drug-properties DB for the NARMS *Salmonella* panel + a target gate.

Two jobs, both pure code (no model — determinism-first):

1. Map an AMRFinderPlus determinant (its `Class` / `Subclass` columns) to the panel
   drugs it confers resistance to. The class/subclass vocabulary here is the EXACT
   uppercase strings from the installed reference DB (2026-05-15.1 `fam.tsv`), read off
   the box — not guessed. AMRFinderPlus emits compound fields (e.g. `AMINOGLYCOSIDE/
   QUINOLONE`, subclass `GENTAMICIN/KANAMYCIN/TOBRAMYCIN`), so matching is done on
   slash-split tokens: a determinant hits a drug when any class token is one the drug
   cares about AND (if the drug narrows by subclass) any subclass token matches. Token
   matching is what lets one `AMINOGLYCOSIDE` class cleanly separate gentamicin from a
   streptomycin-only aadA determinant.

2. A target-presence gate that exists to refuse the "no resistance gene ⇒ susceptible"
   fallacy (see `target_present`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# AMRFinderPlus compound Class/Subclass fields are slash-delimited; split on this to
# get comparable tokens. Hyphens (BETA-LACTAM) and spaces (FUSIDIC ACID) are intact.
_FIELD_SEP = "/"


@dataclass(frozen=True, slots=True)
class Drug:
    """One antibiotic on the NARMS Salmonella panel and how to recognize its resistance.

    `target_genes` are the molecular targets the drug inhibits (used by the presence
    gate, not by the determinant matcher). `amrfinder_classes` / `amrfinder_subclasses`
    are the uppercase AMRFinderPlus vocabulary tokens that flag a determinant as
    conferring resistance to THIS drug.
    """

    name: str
    drug_class: str
    target_genes: tuple[str, ...]
    mechanism: str
    intrinsic_resistant: bool
    amrfinder_classes: frozenset[str]
    amrfinder_subclasses: frozenset[str] = field(default_factory=frozenset)


# --- Target-gene sets (core/essential; universally present in a viable genome) --------
# These are housekeeping/core targets, not accessory genes, so a healthy Salmonella
# genome always carries them. We keep them explicit so the presence gate is honest.
_PBP_TARGETS = ("ftsI", "mrcA", "mrcB")  # penicillin-binding proteins (PBP3/1a/1b)
_GYRASE_TARGETS = ("gyrA", "gyrB", "parC", "parE")  # DNA gyrase + topoisomerase IV
_RRS_16S = ("rrs",)  # 16S rRNA (30S subunit)
_RRL_23S = ("rrl", "23S")  # 23S rRNA (50S subunit)


# The AMRFinderPlus BETA-LACTAM class does NOT carry a PENICILLIN subclass in this DB;
# plain penicillinases land in subclass BETA-LACTAM, AmpC/ESBL in CEPHALOSPORIN,
# carbapenemases in CARBAPENEM. So drugs escalate their subclass gate up that ladder.
_BL = "BETA-LACTAM"
_CEPH = "CEPHALOSPORIN"
_CARB = "CARBAPENEM"


DRUG_DB: dict[str, Drug] = {
    "ampicillin": Drug(
        name="ampicillin",
        drug_class="aminopenicillin",
        target_genes=_PBP_TARGETS,
        mechanism="Binds penicillin-binding proteins, blocking peptidoglycan cross-linking.",
        # EUCAST expected phenotypes list no intrinsic Salmonella resistance to the panel.
        intrinsic_resistant=False,
        amrfinder_classes=frozenset({_BL}),
        # Any beta-lactamase (penicillinase through carbapenemase) hydrolyzes ampicillin,
        # so no subclass gate — the class alone is the evidence.
        amrfinder_subclasses=frozenset(),
    ),
    "amoxicillin-clavulanic acid": Drug(
        name="amoxicillin-clavulanic acid",
        drug_class="beta-lactam/beta-lactamase-inhibitor combination",
        target_genes=_PBP_TARGETS,
        mechanism="Amoxicillin hits PBPs; clavulanate inhibits many class A beta-lactamases.",
        intrinsic_resistant=False,
        amrfinder_classes=frozenset({_BL}),
        # A resistant call here needs a beta-lactamase clavulanate CANNOT neutralize.
        # Plain penicillinases (subclass BETA-LACTAM) are inhibited → excluded. Crucially,
        # ESBLs ARE clavulanate-inhibited yet also report as subclass CEPHALOSPORIN,
        # indistinguishable at class granularity from AmpC (which does evade clavulanate).
        # Trusting a bare CEPHALOSPORIN over-called lab-susceptible ESBL genomes (e.g.
        # blaSHV-2A → amox-clav S) as resistant — a lab-contradicting known-gene call the
        # "honest one" thesis cannot make. So only the unambiguous carbapenemase token
        # counts; a CEPHALOSPORIN-only genome honestly no-calls here (the calibrated
        # statistical oracle still commits). Separating AmpC from ESBL needs gene identity (T3).
        amrfinder_subclasses=frozenset({_CARB}),
    ),
    "ceftriaxone": Drug(
        name="ceftriaxone",
        drug_class="third-generation cephalosporin",
        target_genes=_PBP_TARGETS,
        mechanism="Binds PBPs; a key expanded-spectrum agent, defeated by ESBLs/AmpC.",
        intrinsic_resistant=False,
        amrfinder_classes=frozenset({_BL}),
        # ESBLs and AmpC both land in subclass CEPHALOSPORIN; carbapenemases also
        # hydrolyze it. Plain penicillinases (subclass BETA-LACTAM) do not.
        amrfinder_subclasses=frozenset({_CEPH, _CARB}),
    ),
    "cefoxitin": Drug(
        name="cefoxitin",
        drug_class="cephamycin",
        target_genes=_PBP_TARGETS,
        mechanism="Cephamycin; AmpC-type cephalosporinases are the classic resistance route.",
        intrinsic_resistant=False,
        amrfinder_classes=frozenset({_BL}),
        # Cefoxitin (a cephamycin) is defeated by AmpC cephalosporinases and carbapenemases
        # but NOT by ESBLs — yet AmpC and ESBLs both report as subclass CEPHALOSPORIN, so
        # class granularity cannot tell them apart. Asserting resistance from a bare
        # CEPHALOSPORIN over-called ESBL-only genomes (which stay cefoxitin-susceptible;
        # e.g. blaSHV-2A → cefoxitin S in the lab), so we require the unambiguous CARBAPENEM
        # token and honestly no-call a CEPHALOSPORIN-only genome until gene-level AmpC
        # identification lands (T3). Cefoxitin is a marker-only drug for Salmonella regardless.
        amrfinder_subclasses=frozenset({_CARB}),
    ),
    "meropenem": Drug(
        name="meropenem",
        drug_class="carbapenem",
        target_genes=_PBP_TARGETS,
        mechanism="Carbapenem binding PBPs; only carbapenemases meaningfully defeat it.",
        intrinsic_resistant=False,
        amrfinder_classes=frozenset({_BL}),
        # Only carbapenemases (subclass CARBAPENEM) count — narrowest beta-lactam gate.
        amrfinder_subclasses=frozenset({_CARB}),
    ),
    "ciprofloxacin": Drug(
        name="ciprofloxacin",
        drug_class="fluoroquinolone",
        target_genes=_GYRASE_TARGETS,
        mechanism="Inhibits DNA gyrase and topoisomerase IV, blocking DNA replication.",
        intrinsic_resistant=False,
        # QUINOLONE class covers gyrA/parC POINT mutations and qnr/aac(6')-Ib-cr genes;
        # compound classes like AMINOGLYCOSIDE/QUINOLONE match via the QUINOLONE token.
        amrfinder_classes=frozenset({"QUINOLONE"}),
        amrfinder_subclasses=frozenset(),
    ),
    "nalidixic acid": Drug(
        name="nalidixic acid",
        drug_class="quinolone",
        target_genes=_GYRASE_TARGETS,
        mechanism="First-generation quinolone; gyrA substitutions are the primary driver.",
        intrinsic_resistant=False,
        amrfinder_classes=frozenset({"QUINOLONE"}),
        amrfinder_subclasses=frozenset(),
    ),
    "azithromycin": Drug(
        name="azithromycin",
        drug_class="macrolide",
        target_genes=_RRL_23S,
        mechanism="Binds the 23S rRNA of the 50S subunit, stalling protein synthesis.",
        intrinsic_resistant=False,
        # MACROLIDE token also appears in compound classes (MACROLIDE/PHENICOL,
        # LINCOSAMIDE/MACROLIDE, ...); token matching catches those.
        amrfinder_classes=frozenset({"MACROLIDE"}),
        amrfinder_subclasses=frozenset(),
    ),
    "gentamicin": Drug(
        name="gentamicin",
        drug_class="aminoglycoside",
        target_genes=_RRS_16S,
        mechanism="Binds 16S rRNA of the 30S subunit, causing mistranslation.",
        # NOTE: clinically aminoglycosides are unreliable for systemic salmonellosis,
        # but that is a reporting caveat, not EUCAST intrinsic resistance.
        intrinsic_resistant=False,
        amrfinder_classes=frozenset({"AMINOGLYCOSIDE"}),
        # AMINOGLYCOSIDE class spans many drugs; only subclasses naming GENTAMICIN
        # confer gentamicin R (excludes streptomycin/kanamycin-only determinants).
        amrfinder_subclasses=frozenset({"GENTAMICIN"}),
    ),
    "streptomycin": Drug(
        name="streptomycin",
        drug_class="aminoglycoside",
        target_genes=("rrs", "rpsL"),  # 16S rRNA + ribosomal protein S12
        mechanism="Binds the 30S subunit (16S rRNA / S12), causing misreading.",
        intrinsic_resistant=False,
        amrfinder_classes=frozenset({"AMINOGLYCOSIDE"}),
        # aadA (SPECTINOMYCIN/STREPTOMYCIN) and strAB (STREPTOMYCIN) name the token.
        amrfinder_subclasses=frozenset({"STREPTOMYCIN"}),
    ),
    "tetracycline": Drug(
        name="tetracycline",
        drug_class="tetracycline",
        target_genes=_RRS_16S,
        mechanism="Binds the 30S subunit, blocking aminoacyl-tRNA docking.",
        intrinsic_resistant=False,
        amrfinder_classes=frozenset({"TETRACYCLINE"}),
        # Gate on the TETRACYCLINE subclass to exclude TIGECYCLINE-only determinants
        # (e.g. the NITROFURAN/PHENICOL/QUINOLONE/TETRACYCLINE efflux family).
        amrfinder_subclasses=frozenset({"TETRACYCLINE"}),
    ),
    "chloramphenicol": Drug(
        name="chloramphenicol",
        drug_class="phenicol",
        target_genes=_RRL_23S,
        mechanism="Binds the 50S subunit, inhibiting peptidyl transferase.",
        intrinsic_resistant=False,
        # cat, floR (CHLORAMPHENICOL/FLORFENICOL), cfr and compound PHENICOL classes all
        # confer chloramphenicol R, so class alone is the honest evidence.
        amrfinder_classes=frozenset({"PHENICOL"}),
        amrfinder_subclasses=frozenset(),
    ),
    "trimethoprim-sulfamethoxazole": Drug(
        name="trimethoprim-sulfamethoxazole",
        drug_class="folate-pathway inhibitor combination",
        target_genes=("folA", "folP"),  # DHFR (trimethoprim) + DHPS (sulfonamide)
        mechanism="Trimethoprim inhibits DHFR (folA); sulfamethoxazole inhibits DHPS (folP).",
        intrinsic_resistant=False,
        # GUESS: co-trimoxazole resistance is usually driven by dfrA (TRIMETHOPRIM);
        # sul genes (SULFONAMIDE) alone often leave the combination active. We treat
        # either determinant as evidence — the ML model, not this matcher, sets phenotype.
        amrfinder_classes=frozenset({"TRIMETHOPRIM", "SULFONAMIDE"}),
        amrfinder_subclasses=frozenset(),
    ),
}


def _tokens(field_value: str) -> frozenset[str]:
    """Slash-split an AMRFinderPlus Class/Subclass value into uppercase tokens."""
    return frozenset(
        part.strip().upper() for part in field_value.split(_FIELD_SEP) if part.strip()
    )


def get_drug(name: str) -> Drug | None:
    """Look up a drug by canonical NARMS name (case-insensitive). None if unknown."""
    return DRUG_DB.get(name.strip().lower())


def drug_matches_determinant(drug: Drug, det_class: str, det_subclass: str) -> bool:
    """True if an AMRFinderPlus determinant row confers resistance to `drug`.

    A determinant matches when any of its Class tokens is one the drug recognizes AND,
    if the drug narrows by subclass, any of its Subclass tokens matches. Compound fields
    (slash-delimited) are handled token-wise; comparison is case-insensitive.
    """
    if _tokens(det_class).isdisjoint(drug.amrfinder_classes):
        return False
    if not drug.amrfinder_subclasses:
        return True
    return not _tokens(det_subclass).isdisjoint(drug.amrfinder_subclasses)


def target_present(drug: Drug, present_genes: set[str]) -> bool:
    """Target-presence gate — the guardrail against the "no gene ⇒ susceptible" fallacy.

    Absence of a resistance determinant is NOT evidence of susceptibility; it only means
    we found no known mechanism. This gate instead asks the opposite, honest question:
    is the drug's molecular *target* even present? For the NARMS panel every target here
    is a core/essential gene, so any viable Salmonella genome carries it — hence when the
    caller passes no gene-presence set we assume the core targets are present (True).

    When a non-empty `present_genes` set IS supplied (a future T3 extension that detects
    housekeeping/accessory targets), we check `drug.target_genes` against it honestly and
    do NOT fabricate detection we don't perform. Returns True if any target gene is found.
    """
    if not present_genes:
        return True
    normalized = {gene.strip().lower() for gene in present_genes}
    return any(target.lower() in normalized for target in drug.target_genes)
