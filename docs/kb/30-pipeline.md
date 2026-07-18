# KB: Verified Pipeline & Commands (tested live on the box 2026-07-18)

## Environment (done)
- micromamba env `amr` with `ncbi-amrfinderplus` (4.2.7) + `mash` (2.3); DB 2026-05-15.1 downloaded.
- Run tools via: `~/bin/micromamba run -n amr <cmd>`. Python/ML via `uv` (Python 3.14 on box).

## 1. Labels (genome_amr API) — evidence=Laboratory Method
Count (Content-Range header):
  curl -s -D - -o /dev/null 'https://www.bv-brc.org/api/genome_amr/?and(keyword("Salmonella enterica"),eq(evidence,Laboratory Method))&limit(1)'
Pull (CSV, page by 25000):
  curl -s -H 'Accept: text/csv' 'https://www.bv-brc.org/api/genome_amr/?and(keyword("Salmonella enterica"),eq(evidence,Laboratory Method))&select(genome_id,antibiotic,resistant_phenotype,measurement_value,measurement_unit,measurement_sign,laboratory_typing_method,testing_standard,testing_standard_year)&limit(25000,0)'
Label rules: keep evidence="Laboratory Method"; resistant_phenotype ∈ {Susceptible, Resistant, Intermediate, Nonsusceptible}
  → map Nonsusceptible→R; decide Intermediate (drop or →R). Dedup genome×drug (prefer recent / majority vote).
  URL-encode spaces as %20 in scripts.

## 2. FASTA (genome_sequence API — FTP is firewalled)
  curl -s 'https://www.bv-brc.org/api/genome_sequence/?eq(genome_id,GID)&select(sequence_id,sequence)&limit(25000)' \
    | jq -r '.[] | ">" + .sequence_id + "\n" + .sequence' > GID.fna
Typical Salmonella assembly ~4.8 Mbp / ~40 contigs (verified: 28901.21086 = 39 contigs, 4.79 Mbp).

## 3. Features (AMRFinderPlus — VERIFIED 17s/genome, 4 threads, ~190MB RAM)
  ~/bin/micromamba run -n amr amrfinder -n GID.fna --organism Salmonella --plus --threads 4 --name GID -o GID.tsv
Parallelize:
  ls fasta/*.fna | parallel -j 4 '~/bin/micromamba run -n amr amrfinder -n {} --organism Salmonella --plus --threads 1 --name {/.} -o out/{/.}.tsv'
Feature build: concat TSVs; keep rows where Subtype ∈ {AMR, POINT} (drop VIRULENCE/STRESS/METAL);
  pivot to binary matrix on `Element symbol` (v4) / `Gene symbol` (v3), aggfunc=max. POINT rows carry the
  substitution in the symbol (e.g., parC_S80I). Class/Subclass give the drug class the determinant hits.
Feasibility: ~1500 genomes × 17s / 4-way parallel ≈ 1.8h annotate (+ download). Comfortable overnight.

## 4. Split (Mash)
  ~/bin/micromamba run -n amr mash sketch -k 21 -s 100000 -p 4 -o all fasta/*.fna
  ~/bin/micromamba run -n amr mash dist all.msh all.msh -p 4 > mash_dist.tsv   # col3 = distance (~1-ANI)
  single-linkage cluster at D≈0.005 → cluster id per genome → StratifiedGroupKFold(groups=cluster).
  Headline: also leave-one-clade-out. Report random vs grouped side by side.

## 5. Model + calibration + no-call (sklearn)
  per drug: LogisticRegression(penalty=l2, class_weight=balanced, solver=liblinear)
  CalibratedClassifierCV(method="sigmoid") on a disjoint grouped calibration split (no lineage leakage).
  no-call: band [0.35,0.65] ∪ OOD(min Mash dist>thr) ∪ (<min positives) ∪ conflicting; report risk-coverage.
  intrinsic-resistance (EUCAST expected phenotypes) → deterministic R; absence-of-gene → no-call, never S.
  metrics per drug AND per group: balanced acc, per-class recall, F1, AUROC, PR-AUC, Brier, reliability.
  baseline: nearest-neighbor by Mash distance (predict neighbor's phenotype) — prove ML beats it.

## 6. Demo (Streamlit) + OpenAI
  upload FASTA → run amrfinder (17s) → features → per-drug verdict+conf+evidence+genes → "confirm with lab".
  OpenAI: rationale constrained to cite ONLY genes in the payload; + drug-name harmonization ETL.
