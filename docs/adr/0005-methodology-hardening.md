# 0005. Methodology hardening (data-science panel)

- Status: Accepted · Date: 2026-07-18 · Supersedes the subsample/single-threshold parts of ADR-0003.

## Decisions
- **NO row-budget subsampling.** LR trains in seconds on 14k rows. Subsample ONLY to de-duplicate
  near-identical clones; KEEP every distinct cluster and EVERY resistant isolate. (Runtime bound comes
  from AMRFinderPlus, not the model — see feasibility.)
- **Two-level Mash:** tight DEDUP at D~=0.0005 (near-identical); COARSE CV grouping by serovar / 7-gene
  MLST (or D~=0.02-0.05) — NOT 0.005 (too fine; leaks sister sublineages). Report per-fold min train<->test
  Mash distance + cluster-size distribution (single-linkage chaining check).
- **Two mandatory baselines:** deterministic known-gene rule, and Mash-nearest-neighbor label transfer.
  ML "ships" for a drug ONLY where it beats BOTH (esp. on statistical-association drugs / unseen clades).
  Guards against the tautology of AMR genes predicting themselves.
- **Calibration:** grouped-CV ensemble calibration (or beta calibration) instead of one starved split;
  reconsider class_weight vs calibration; positive-count FLOOR per drug -> drug-level no-call if too few.
- **No-call = cost-aware selective prediction** (risk-coverage), per-drug operating point chosen on the
  calibration fold to bound VME; NOT a fixed [0.35,0.65] band. OOD threshold set from train LOO nearest-
  cluster distances, validated on leave-one-clade-out; report OOD-detection AUROC.
- **Cluster/clade-bootstrap CIs** on every metric (not i.i.d. -> pseudo-replication). Report ALL drugs
  incl. failures (pre-commit the metric set); lead with effect sizes + CIs; BH-FDR on any significance claim.
- Per-drug LightGBM = optional 2nd candidate (epistasis) only if it beats calibrated LR under grouped split.
