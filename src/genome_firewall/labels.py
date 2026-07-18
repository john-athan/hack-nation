"""Phenotype harmonization: raw genome_amr rows → one clean {R,S} label per (genome, drug).

The raw table is messy: mixed phenotype vocabulary, drug-name synonyms, and multiple
measurements per genome×drug. We collapse it deterministically so calibration downstream
sees clean ground truth — no model needed for any of this (determinism-first).
"""

from __future__ import annotations

import pandas as pd

from .constants import PHENOTYPE_MAP

# Drug-name synonyms → canonical NARMS name. Extend as the data surfaces new spellings.
DRUG_SYNONYMS: dict[str, str] = {
    "trimethoprim/sulfamethoxazole": "trimethoprim-sulfamethoxazole",
    "trimethoprim sulfamethoxazole": "trimethoprim-sulfamethoxazole",
    "amoxicillin/clavulanic acid": "amoxicillin-clavulanic acid",
    "amoxicillin/clavulanate": "amoxicillin-clavulanic acid",
    "sulfisoxazole": "sulfamethoxazole",
}


def canonical_drug(name: str) -> str:
    key = name.strip().lower()
    return DRUG_SYNONYMS.get(key, key)


def harmonize(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy frame: genome_id, antibiotic, label ∈ {Resistant, Susceptible}.

    - map phenotype vocabulary to binary (Intermediate dropped),
    - canonicalize drug names,
    - dedup genome×drug by majority vote (ties → Resistant, the safer error for a firewall).
    """
    # Drop physically-identical source rows FIRST: the raw table ships ~500 groups of
    # 4x-duplicated measurements, and counting one record four times would silently skew
    # the majority vote below (and thus T3 calibration ground truth).
    df = raw.drop_duplicates().copy()
    df["antibiotic"] = df["antibiotic"].astype(str).map(canonical_drug)
    df["label"] = df["resistant_phenotype"].astype(str).str.strip().map(PHENOTYPE_MAP)
    df = df.dropna(subset=["label"])

    # Majority vote per genome×drug; break ties toward Resistant.
    def _resolve(group: pd.Series) -> str:
        counts = group.value_counts()
        top = counts.max()
        winners = set(counts[counts == top].index)
        if "Resistant" in winners and len(winners) > 1:
            return "Resistant"
        return str(counts.idxmax())

    resolved = (
        df.groupby(["genome_id", "antibiotic"])["label"].apply(_resolve).reset_index()
    )
    return resolved
