"""Central constants — no magic values scattered across the codebase."""

from __future__ import annotations

from pathlib import Path

# --- Data layout -------------------------------------------------------------
# Everything under data/ is gitignored (see .gitignore); it is a local cache.
DATA_DIR = Path("data")
FASTA_DIR = DATA_DIR / "fasta"
AMRFINDER_DIR = DATA_DIR / "amrfinder_out"
LABELS_CSV = DATA_DIR / "labels_raw.csv"
LABELS_CLEAN_CSV = DATA_DIR / "labels_clean.csv"
COHORT_CSV = DATA_DIR / "cohort.csv"
FEATURES_DIR = DATA_DIR / "features"

# --- T2 cohort (scaled feature build) ----------------------------------------
# ADR-0005: NO row-budget subsample. Keep EVERY resistant isolate (rare positives are
# precious — subsampling silently destroys per-class recall / PR-AUC / calibration on the
# scarce-R drugs). Only trim the redundant SUSCEPTIBLE majority, capped per lineage group
# (serovar×MLST as the coarse-cluster proxy). Lineage leakage is handled at EVALUATION time
# by the grouped split, never by dropping data.
COHORT_SUS_CAP_PER_GROUP = 3
# Assembly QC: drop junk/misassembled genomes before we spend annotate time on them
# (a fragmented assembly yields unreliable determinants — applies to R isolates too).
GENOME_LEN_MIN = 4_000_000
GENOME_LEN_MAX = 5_800_000
GENOME_MAX_CONTIGS = 600

# --- BV-BRC API --------------------------------------------------------------
# Public, no-auth. FTP is firewalled on this box; the HTTP API is the only path.
BVBRC_API = "https://www.bv-brc.org/api"
BVBRC_SPECIES = "Salmonella enterica"
# genome_amr rows are per (genome, antibiotic). We only trust lab-measured phenotypes,
# never in-silico predictions, so our labels are an honest ground truth.
BVBRC_EVIDENCE = "Laboratory Method"
BVBRC_PAGE_SIZE = 25_000
# genome-metadata lookups use in(genome_id,(...)) — keep batches modest so the URL
# stays well under server limits.
BVBRC_METADATA_BATCH = 150
HTTP_TIMEOUT_S = 120
HTTP_RETRIES = 4

# --- AMRFinderPlus -----------------------------------------------------------
# Pinned so the determinant matrix is reproducible across the whole dataset.
AMRFINDER_ENV = "amr"
MICROMAMBA = Path.home() / "bin" / "micromamba"
AMRFINDER_ORGANISM = "Salmonella"
AMRFINDER_DB_VERSION = "2026-05-15.1"
# A normal Salmonella genome annotates in ~17s. This ceiling is generous for any legitimate
# assembly but bounds a pathological live upload so the demo degrades instead of hanging on stage.
AMRFINDER_TIMEOUT_S = 300
# Subtypes we keep as resistance/lineage features; VIRULENCE/STRESS/METAL are dropped.
KEPT_SUBTYPES = frozenset({"AMR", "POINT"})

# --- Phenotype harmonization -------------------------------------------------
# BV-BRC resistant_phenotype vocabulary → our binary {R, S}.
# Nonsusceptible collapses to Resistant (clinically it is not "works").
# Intermediate is dropped (ambiguous; keeping it would poison calibration).
PHENOTYPE_RESISTANT = "Resistant"
PHENOTYPE_SUSCEPTIBLE = "Susceptible"
PHENOTYPE_MAP: dict[str, str | None] = {
    "Resistant": PHENOTYPE_RESISTANT,
    "Nonsusceptible": PHENOTYPE_RESISTANT,
    "Susceptible": PHENOTYPE_SUSCEPTIBLE,
    "Intermediate": None,  # dropped
}

# --- Verdicts & evidence -----------------------------------------------------
CALL_RESISTANT = "resistant"
CALL_SUSCEPTIBLE = "susceptible"
CALL_NO_CALL = "no_call"

# Evidence category surfaced per prediction (PDF requirement).
EVIDENCE_KNOWN_GENE = "known_gene"  # a curated resistance determinant explains the call
EVIDENCE_STATISTICAL = "statistical_only"  # model signal without a known mechanism
EVIDENCE_NO_SIGNAL = "no_signal"  # nothing to go on → drives a no-call

# T1 dummy verdict uses only known_gene / no_signal; statistical arrives with the T3 model.

# --- Modeling table ----------------------------------------------------------
# Binary label encoding for the model matrix (genome × drug); missing label → NaN.
LABEL_POS = 1  # Resistant
LABEL_NEG = 0  # Susceptible
# Feature-block column prefixes keep the mechanism vs lineage split legible end-to-end
# (the USP knockout probe zeroes MECH_PREFIX columns; shift-weighting reads LINEAGE_PREFIX).
MECH_PREFIX = "mech__"
LINEAGE_PREFIX = "lin__"
# A genome with no Mash cluster assignment (e.g. singleton not yet clustered).
NO_CLUSTER = -1
