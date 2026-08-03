# Research log

The single narrative doc.  Historical numbers below were reported from
`experiments/nbody_benchmark.py`, but the referenced `runs/nbody_benchmark.json` artifact is absent
from the current checkout.  They are therefore not independently reproducible here.  Retractions and
scope downgrades are reported in place, loudly, naming the check that failed.

---

## 2026-08-03 — ORBIT F01 upstream-scale replication: **H2, solver, and variance passed**

Scope: this remains a fixed-hyperparameter GP-simulation mechanism test, **not** an N-body,
training, wall-time, or SOTA performance result.  It ran the post-residual-fix commit
`beeeda20fb226f7aac7ab6210679afc3ea3eea97` with TERA submodule `b2382e10...` in one exclusive L40S
allocation (Slurm job 2810279).  The five independent units used upstream de Roos-style seeds
`42 + 1000003 * repeat`, `n=150`, `d=500`, 100 evaluation points, Matérn-5/2, float32,
`m={3,5,10,20,30,50,100}`, and recomputed-residual CG tolerance `1e-5`; TERA was evaluated through
`m=50`.  Wall time was recorded but excluded from every hypothesis test.

The provenance-strict aggregate at BC path
`runs/f01_orbit_official_beeeda2/job-2810279/aggregate.json` reports:

- all 5/5 expected tasks valid and content-distinct, with matching commit, tree, submodule, source,
  package, normalized config, and full-node sole-job exclusivity evidence;
- at the TERA-controlled `m=50`, mean marginal KL was `0.0125718` (bootstrap 95% interval
  `[0.0114981, 0.0134325]`); resource-headroom ORBIT at `m=100<n` achieved `0.00469795`
  (`[0.00419330, 0.00525266]`);
- the paired mean KL improvement was `0.00787385`, with a 10,000-sample paired percentile-bootstrap
  95% interval `[0.00705988, 0.00868200]`, excluding zero on all five resampled dataset units;
- all 35 ORBIT groups converged according to freshly recomputed residuals (maximum relative residual
  `9.98844e-6`) and had valid positive variances (minimum raw variance `0.356503`).

The analytic resource accounting is deliberately separate from wall time.  TERA `m=50` materializes
`6,250,000` reduced-covariance scalars per target and has `5.20833e11` leading dense-Cholesky flops
across 100 targets.  ORBIT `m=100` stores a `60,100`-scalar operator-core proxy per target and used a
mean `3.71379e10` counted operator-plus-preconditioner flops across the same 100 targets (range
`[3.70578e10, 3.71980e10]`).  Thus the larger-`m` arm is inside this accounting envelope by factors
of about 104 in persistent core state and 14.0 in counted leading/iterative flops.  These are
method-specific analytic proxies, not end-to-end hardware-runtime ratios.

This float32 replication was preregistered as **ineligible for H1 same-`m` equivalence**, whose
`1e-6` gate requires float64.  Its aggregate therefore correctly records H1 as `insufficient` and
`overall_mechanism_pass=false`; that bookkeeping must not be misreported as an H2 failure.  The
independent float64 pilot above supplies the H1 equivalence evidence, while this replication supplies
the upstream-scale H2, solver, and variance evidence.  Float32 runs also provide no IEEE-rigorous
posterior certificate (`floating_point_rigorous=false`), and the exact-arithmetic lower-bound
certificate becomes uninformative at `m=100`; neither certificate is used as an empirical gate here.

---

## 2026-08-03 — ORBIT F01 mechanism pilot: **all preregistered gates passed**

Scope: this is a fixed-hyperparameter, exact-GP-simulation mechanism test, **not** an N-body,
training, wall-time, or SOTA performance result.  It ran commit
`15876c78a19be9e79e68a5fc16fd8c69ce600625` with TERA submodule `b2382e10...` on one exclusive
L40S allocation.  The three independent units used seeds 20260803–20260805, `n=60`, `d=50`, 30
evaluation points, Matérn-5/2, dense exact sampling, float64, and `m={5,10,20,30,50}`; TERA was
evaluated through `m=30`.  Wall time was recorded but excluded from every hypothesis test.

The provenance-strict aggregate at BC path
`runs/f01_orbit_pilot_aee694c/job-2810267/aggregate.json` reports:

- all 3/3 expected tasks valid, independent, and consistent in commit, tree, submodule, source,
  package, config, dataset hashes, and full-node exclusivity evidence;
- same-`m` ORBIT/TERA agreement passed all 12 paired comparisons: worst mean difference
  `2.1891e-8` and worst variance difference `8.3816e-13`, below the preregistered `1e-6` threshold;
- all 15 ORBIT groups had valid positive variance and all recomputed relative residuals met `1e-8`;
- enlarging ORBIT from the TERA-controlled `m=30` to the nontrivial `m=50<n` reduced mean marginal
  KL from `0.0107313` to `0.00505893`; the mean paired improvement was `0.00567238`, with a paired
  10,000-sample percentile-bootstrap 95% interval `[0.00277932, 0.00872510]`.

An upstream-scale **single-seed diagnostic only** (job 2810269; `n=150`, `d=500`, 100 evaluation
points, upstream de Roos sampling, float32) likewise moved KL from TERA `m=50` at `0.0106563` to
ORBIT `m=100` at `0.00394131`.  It is not pooled evidence.  It also exposed that a recursive fp32 CG
residual could cross `1e-5` while the recomputed residual remained just above it for one prediction;
the solver was subsequently changed to replace the residual and continue instead of returning a
contradictory convergence status.  The diagnostic's loose-tolerance same-`m` discrepancies are not
used for the float64 equivalence claim.

---

## 2026-08-03 — **SCOPE DOWNGRADE: legacy N-body results are smoke only**

The 2026-07-29 table below must not be used as evidence that one GP framework beats another.  A data
and harness audit found four material problems:

1. the legacy generator redraws particle masses for every trajectory but does not include masses in
   `X`, so pooled rows do not represent one well-defined Hamiltonian regression function;
2. the benchmark splits individual rows, allowing states from the same trajectory into train and
   test rather than measuring trajectory-level generalization;
3. label-based gradient filtering occurs before the split; and
4. the exact arm is under-solved, while the TERA wrapper does not propagate the outer benchmark seed
   and its custom gradient path has no independent regression suite.

The missing run artifact is an additional reproducibility failure.  None of these facts makes the
reported arithmetic imaginary, but together they invalidate any confirmatory or SOTA interpretation.
The table is retained below only as historical smoke output.

A separate companion generator, `data/generate_nbody_confirmatory.py`, now fixes masses per task
replica, persists trajectory/time IDs and full configuration, creates deterministic disjoint
trajectory-group splits, computes normalization from training groups only, performs no label-based
filtering, validates gradients and energy drift, and writes a SHA-256 manifest.  Baselines must be
rerun on frozen companion corpora before a predictive claim is considered.

---

No results recorded yet.
