# Research log

The single narrative doc.  Historical numbers below were reported from
`experiments/nbody_benchmark.py`, but the referenced `runs/nbody_benchmark.json` artifact is absent
from the current checkout.  They are therefore not independently reproducible here.  Retractions and
scope downgrades are reported in place, loudly, naming the check that failed.

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
