# KB: AMR-from-genome — State of the Art (2023–2026)

Genotype→phenotype AMR prediction is **PARTIALLY solved, unevenly**:
- SOLVED (clinical-grade) for narrow cases: single-gene markers (mecA→MRSA ~99%); acquired-gene resistance
  in Enterobacterales/foodborne (ResFinder 4.0 concordance ≥95–99% for E. coli 97%, Salmonella 98.8%,
  Campylobacter 99.2%); first-line TB (INH/RIF ~97–99%, CRyPTIC/NEJM 2018).
- OPEN: novel/unknown mechanisms; chromosomal/regulatory/efflux/porin-mediated resistance (much of
  Pseudomonas, hard Gram-negatives); new TB drugs (bedaquiline/linezolid ~68% or "no prediction").

## The single most important finding (our narrative hook)
**Yu, Wheeler & Barquist, PLOS Biology 2025** — biased sampling driven by bacterial POPULATION STRUCTURE
confounds ML AMR prediction: under realistic sampling, performance "degraded dramatically", **more training
data did NOT rescue it**, models generalized poorly to unseen clades, and predictive features barely overlapped
across clades → models learn LINEAGE, not mechanism. https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3003539
This is why a RANDOM split inflates accuracy and a GROUPED split collapses it. Our demo weaponizes exactly this.

## Tools
- AMRFinderPlus (NCBI, Feldgarden 2021): BLASTP/BLASTX + curated HMMs; ~5,600 genes + 682 point mutations,
  ~25 drug classes. Genotype detector (separates "gene detected" from "phenotype"). Organism-specific point
  mutations — must pass correct --organism. https://pmc.ncbi.nlm.nih.gov/articles/PMC8208984/
- ResFinder 4.0 (DTU, Bortolaia 2020): benchmarked as phenotype predictor, ≥95% concordance many species.
- CARD/RGI; Pathogenwatch/Kleborate/TB-Profiler (species-specific surveillance).
- PATRIC/BV-BRC AdaBoost on 31-bp k-mers (Davis 2016): mecA 99.5%, A. baumannii carbapenem 94.5%, TB drugs 71–88%.

## Commercial: Day Zero Diagnostics (WGS-from-blood + ML resistance) — acquired by bioMérieux, June 2025.

## Implication for us
Easy catalogued cases (our feasible species) are ALREADY solved → we cannot win on accuracy/novelty.
We win on the rubric: honest grouped-split methodology, calibration, principled no-call, product/explanation UX.
