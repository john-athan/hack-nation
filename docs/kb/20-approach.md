# KB: Our Approach — "The Honest One"

We CANNOT out-bioinformatics domain experts on predictive novelty (easy cases already ~99%).
We WIN the category the rubric actually scores: trustworthy, calibrated, honest, well-presented.

## The wedge (5 pillars)
1. One well-chosen species with abundant LAB-MEASURED MIC + gene-driven resistance (candidate: Salmonella
   or E. coli via NARMS/BV-BRC). 4–6 drugs where gene→phenotype link is real. [decision → ADR-0003]
2. THE MONEY SLIDE: same model under random split (inflated) vs phylogeny-aware grouped split (collapses) →
   we report the grouped number as our real one. Out-rigors everyone.
3. Calibrated confidence: reliability curve + Brier; Platt/isotonic calibration.
4. Principled NO-CALL: driven by the deterministic drug-target gate + thin feature support + uncertain band.
5. OpenAI rationale layer: clinician-readable per-drug explanation, HARD-constrained to only cite genes present
   in the structured payload (physically cannot invent a gene) + "confirm with lab testing" banner. Also LLM for
   ETL/harmonization of messy drug/organism names. LLM = interface/knowledge layer over a deterministic core.

## Pipeline (draft — refine in PLAN)
FASTA → AMRFinderPlus (--organism <species>) → gene/mutation presence-absence matrix
     → join to filtered lab phenotypes (S/R) → grouped split (Mash/MLST clusters)
     → per-antibiotic regularized logistic regression → calibration → no-call policy
     → Streamlit/Gradio report (verdict + confidence + evidence category + genes).

## Infra edge
Build on the Hetzner LINUX box (avoids ARM-Mac/BLAST install hell). Parallelize AMRFinderPlus across cores
(the swarm) — runtime ~1–5 min/genome is the main cost; must parallelize for 1k–3k genomes.
