# Glossary — Genome Firewall (AMR prediction)

- **AMR** — Antimicrobial Resistance: a bacterium survives a drug that should kill/inhibit it.
- **AST** — Antimicrobial Susceptibility Testing: the lab process that measures if a drug works.
- **WGS** — Whole-Genome Sequencing.
- **FASTA** — plain-text format for DNA/protein sequences (our input = one assembled genome).
- **MIC** — Minimum Inhibitory Concentration: lowest drug conc. that stops visible growth (the lab measurement).
- **S / I / R** — Susceptible / Intermediate / Resistant: the phenotype label derived from MIC vs a breakpoint.
- **Breakpoint** — MIC threshold defining S/I/R. Set by **CLSI** (US) or **EUCAST** (EU). **ECOFF** = epidemiological cut-off (wild-type vs not).
- **Phenotype** — observed resistance (what the lab measured). **Genotype** — the genes/mutations present.
- **AMR gene / determinant** — a gene (e.g., blaTEM, mecA) or point mutation (e.g., gyrA S83L) causing resistance.
- **AMRFinderPlus** — NCBI tool: detects known AMR genes + point mutations from a genome. Our default feature extractor.
- **ResFinder / CARD-RGI** — alternative AMR gene detectors / databases (ResFinder is benchmarked as phenotype predictor).
- **BV-BRC (ex-PATRIC)** — Bacterial & Viral Bioinformatics Resource Center: our data source (genomes + lab AMR results).
- **NARMS** — US National Antimicrobial Resistance Monitoring System: high-quality lab MIC data (Salmonella, E. coli...).
- **Grouped / phylogeny-aware split** — train/test split by genetic cluster (not random rows) so near-identical
  genomes don't leak across; the antidote to inflated accuracy.
- **MLST** — Multi-Locus Sequence Typing: assigns a sequence type (ST) — a coarse lineage label.
- **Mash / ANI** — genome-distance measures (MinHash / Average Nucleotide Identity) used to cluster genomes.
- **Clonal / near-identical genomes** — genomes so similar that putting them in both train and test = data leakage.
- **Calibration** — do predicted probabilities match reality? Measured by **Brier score** + **reliability curve**.
  Fixed via **Platt scaling** (logistic) or **isotonic regression**.
- **No-call / abstain** — the system declines to predict when evidence is weak/conflicting (a feature, not a failure).
- **Balanced accuracy / PR-AUC / AUROC / F1** — metrics that survive class imbalance (unlike raw accuracy).
- **Drug-target gate** — deterministic check that the drug's molecular target exists in the genome, so we don't
  call "will work" purely from absence of resistance genes.
- **SHAP / feature importance** — explains model outputs; NOT proof of biological causation (co-located genes confound).
