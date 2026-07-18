# 0007. USP — conformal "AMR epistemology engine" / safety interlock

- Status: Accepted · Date: 2026-07-18 · Supersedes the ad-hoc no-call band in ADR-0005.

## Decision
The differentiator = "Genome Firewall proves what it can and cannot know." Three composed pieces:

1. **Engine — phylogenetically-localized conformal prediction.** Per drug, a calibratable base model
   (L2 logistic; GBT optional) generates nonconformity scores. Apply split/Mondrian (class- &
   group-conditional) conformal, with calibration WEIGHTED by Mash-distance from the query genome to the
   training set (weighted conformal under covariate shift — Tibshirani/Barber/Candes/Ramdas 2019).
   Output per drug = a PREDICTION SET:
   - {R} or {S}  -> confident, coverage-guaranteed call
   - {R,S}       -> NO-CALL (both labels clear the bar → genuinely ambiguous)
   - {}          -> NO-CALL (strongest): NEITHER label clears the 1−α bar (P(R) in the uncertain
                    middle band) — an abstention from probability uncertainty, NOT a measured
                    off-manifold/novelty signal (see honesty caveats).
   No-call EMERGES from the coverage math and widens where the phylogeny is thin — not a hand-tuned band.

2. **Evidence audit — knockout probe + dual-oracle** (rigorous version of the PDF's 3 evidence categories).
   For each R call: computationally KNOCK OUT (set absent) the drug-relevant AMRFinderPlus determinants,
   hold the lineage feature block fixed, re-predict:
   - prediction flips R->S  => (i) MECHANISM-GROUNDED (show the gene)
   - prediction stays R     => (ii) LINEAGE/STATISTICAL only -> abstain/flag (the 2025 failure mode, caught)
   - R with NO determinant + phenotype R => (iii) DARK-AMR candidate (nuisance nulls listed, not "discovery")
   Dual-oracle quadrants: explained-R / unexplained-R(novel) / silent-gene-S / confident-S.

3. **Requires a TWO-BLOCK feature representation** (enables both knockout and shift-weighting):
   - MECHANISM block: AMRFinderPlus gene + curated point-mutation presence/absence.
   - LINEAGE block: mechanism-free population structure (MLST/serovar one-hot or Mash-sketch/PCA).

## Demo / narrative
- Live "firewall holding": naive model says confident green SUSCEPTIBLE on an OOD/novel-mechanism genome;
  Firewall issues NO-CALL + reason + next confirmatory test. Split-screen.
- Money slide: targeted vs delivered coverage on held-out lineages + prediction-set width vs Mash distance;
  leave-one-clade-out promise-vs-delivery (naive calibration breaks, localized conformal holds).
- Tagline: "the AMR model that knows what it doesn't know — before it kills someone."

## Honesty caveats
Exchangeability breaks on truly novel lineages -> localized conformal degrades gracefully (widens/abstains,
never conjures a call). Per-verdict serving uses the GLOBAL (marginal) conformal quantile: a live-served
genome is not localized into the training Mash groups, so the per-lineage (Mondrian) quantiles drive the
REPORTED coverage evaluation (across-lineage guarantee), not each individually served set — both retain the
distribution-free ≥1−α guarantee. The EMPTY set {} is the strongest abstention (P(R) in the uncertain band
where neither label clears the coverage bar), NOT an out-of-distribution/novel-mechanism detector: nothing on
the serving path measures distance to the training manifold, and a genome whose only signal is a truly novel
determinant vectorizes like a susceptible one — so we present {} as "won't commit", never as "novel/off-manifold".
(With the served global quantile <0.5 for every trained model, {R,S} is unreachable and {} is the operative
abstention.) "mechanism = AMRFinderPlus-detected gene" != expression/function. Knockout is a
counterfactual on the MODEL's inputs, not proof of biology. Dark-AMR = candidate w/ nuisance nulls
(AST error, near-breakpoint, assembly dropout, DB incompleteness). Small drug x clade cells -> looser
guarantees (report where).

## PDF compliance
This IS calibration + no-call + phylogeny-aware evaluation + the 3 evidence categories — implemented more
rigorously than required. Still: AMRFinderPlus default features, per-antibiotic predictions, drug-target/
applicability gate, Streamlit + "confirm with lab testing", strictly defensive.
