# 0004. Clinical & label layer (microbiologist panel)

- Status: Accepted  · Date: 2026-07-18

## Decisions
- **Re-derive S/I/R from RAW MIC** using ONE fixed current CLSI M100 breakpoint version (EUCAST as a
  sensitivity check). Do NOT trust BV-BRC's pre-interpreted SIR (mixed breakpoint eras = label noise
  that would masquerade as model miscalibration). VERIFY current values against M100 (don't trust memory).
- **Reported THERAPEUTIC drugs** (clinically valid for Salmonella): ampicillin, ceftriaxone,
  ciprofloxacin, tetracycline, chloramphenicol, trimethoprim-sulfamethoxazole, azithromycin (WT-vs-non-WT
  vs ECOFF; NTS lacks broad clinical breakpoints — state explicitly).
- **Do NOT report as therapeutic** (CLSI: appear active in vitro but clinically ineffective for Salmonella):
  gentamicin (aminoglycoside), cefoxitin (cephamycin/AmpC marker only), nalidixic acid (fluoroquinolone
  SCREEN, not therapy). Keep these + PMQR (qnr, aac(6')-Ib-cr) + gyrA/parC as INTERNAL features/markers.
- **Ciprofloxacin breakpoint trap:** encode the current lowered Salmonella breakpoint so single-QRDR-mutant
  strains are correctly non-susceptible; keep Intermediate as a real class (don't silently collapse).
- **Primary target = log2(MIC) ordinal** where feasible -> threshold to S/I/R; else classify S/R from
  re-derived labels. Report **VME (R called S = dangerous), ME, essential agreement (EA), categorical
  agreement (CA)** alongside the PDF metrics; tie the no-call threshold to BOUNDING VME.
- **Component-specific genes:** SXT -> dfrA (trimethoprim) vs sul1/2/3 (sulfa) modeled separately; flag
  integron/lineage co-travelers (sul1-qacEdelta1-intI1) as LINKAGE not mechanism in evidence tier (ii).
- **Mechanisms AMRFinderPlus misses** (efflux regulators ramR/acrR/marR; porin ompC/F/D disruption;
  PMQR): add crude "regulator/porin disrupted" flags from the assembly -> route those genomes to NO-CALL
  (this is the direct answer to "gene-absence != susceptible", which the PDF probes).
- **Per-locus assembly QC** (coverage/completeness of gyrA/parC/target genes) feeds no-call, not just
  genome-level Mash OOD.
- **Label-churn figure:** report how many genomes flip S<->R between BV-BRC's call and our re-derived call
  (dramatic for cipro) — proves we understand label provenance.
