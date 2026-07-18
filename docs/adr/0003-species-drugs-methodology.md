# 0003. Species, drug set, data source, and methodology

> NOTE: subsample + single-Mash-threshold + intrinsic-only-gate decisions are SUPERSEDED IN PART by ADR-0004 (clinical/labels), ADR-0005 (methodology), ADR-0006 (engineering). Read those.

- Status: Accepted
- Date: 2026-07-18

## Context
Verified live against BV-BRC + AMRFinderPlus on the box (2026-07-18).

## Decisions
- **Species: Salmonella enterica.** Most distinct lab-measured genomes (14,646) and the densest,
  most UNIFORM drug panel (NARMS/Sensititre) → cleanest single-species × drug matrix. (E. coli has
  more rows but patchier per-drug; Klebsiella smaller.) Filter by `keyword("Salmonella enterica")`
  (NOT taxon_id — that is strain-level and undercounts) + `evidence="Laboratory Method"`.
- **Drug set (~6–10, mixed mechanisms):** ampicillin (blaTEM/blaCARB), ceftriaxone + cefoxitin
  (blaCTX-M/blaCMY → cephalosporin), ciprofloxacin + nalidixic acid (gyrA/parC POINT + qnr — showcases
  point-mutations + the no-call story), tetracycline (tetA/B), gentamicin + streptomycin (aac/aph/aadA),
  chloramphenicol (floR/cat), trimethoprim-sulfamethoxazole (sul1/2 + dfrA), azithromycin (mphA).
  Final set = drugs with ≥ ~9k labeled genomes AND a real gene→phenotype link. Drop drugs with < ~15
  resistant isolates after the grouped split (report base rate + no-call instead).
- **Data access:** genome_amr API for labels (evidence=Laboratory Method); genome_sequence API to build
  FASTA (BV-BRC FTP is firewalled from the box → API is the source). Public, no auth. 25k-row page cap.
- **Feature extraction:** run AMRFinderPlus ourselves (`--organism Salmonella --plus`, ~17s/genome,
  verified) → binary presence/absence matrix keyed on `Element symbol`; keep Subtype ∈ {AMR, POINT};
  DROP VIRULENCE/STRESS/METAL. Pin DB version (2026-05-15.1) in metadata. Parallelize with GNU parallel.
- **Scale:** Mash-cluster candidate genomes, subsample to ~1,200–1,500 representatives (kills near-dup
  leakage + bounds runtime to ~4–7h on 4 cores). Optional cpx42 resize if we want more.
- **Split:** Mash sketch `-s 100000` → single-linkage cluster at D≈0.005 → StratifiedGroupKFold; PLUS a
  leave-one-clade-out headline. Report random-split vs grouped-split side by side (the "collapse" slide).
- **Model:** one L2 logistic regression per drug, `class_weight="balanced"`; **sigmoid** calibration
  (isotonic overfits at small n) on a DISJOINT grouped calibration split. Gradient boosting only as a
  comparison, not headline.
- **No-call gate:** abstain if (a) calibrated p in [0.35,0.65] band, (b) genome is OOD (min Mash dist to
  any train cluster > threshold), (c) < min-positives for the drug, or (d) conflicting evidence. Plus a
  deterministic EUCAST intrinsic-resistance table → deterministic R (never "susceptible from absence").
- **Metrics:** balanced accuracy, per-class recall (R and S separately), F1, AUROC, PR-AUC, Brier +
  reliability, risk-coverage/no-call rate — reported per drug AND per genetic group. Include a
  nearest-neighbor (Mash) baseline to prove the ML earns its keep.
- **Demo:** Streamlit — upload FASTA → live AMRFinderPlus (~17s) → per-drug {work/fail/no-call} +
  calibrated confidence + evidence category (known gene / statistical / none) + supporting genes +
  mandatory "confirm with standard lab testing". Plus the collapse slide.
- **OpenAI layer:** clinician-readable per-drug rationale, HARD-constrained to only cite genes
  present in the structured payload (cannot invent a gene); + LLM for messy drug-name harmonization (ETL).
  Model gpt-5.4-mini; $50 credit budget is ample.

## Consequences
- We do NOT claim to beat SOTA accuracy (easy Salmonella cases already ~98% via ResFinder). We win on
  honesty/calibration/no-call/product — exactly the rubric.
- Runtime + install de-risked by live tests. Main residual risk: drug-name harmonization + label hygiene.
