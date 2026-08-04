# Research log

The single narrative doc.  Historical numbers below were reported from
`experiments/nbody_benchmark.py`, but the referenced `runs/nbody_benchmark.json` artifact is absent
from the current checkout.  They are therefore not independently reproducible here.  Retractions and
scope downgrades are reported in place, loudly, naming the check that failed.

---

## 2026-08-03 — F02b numerical calibration: **fit chain and probe foundation implemented, execution blocked**

No calibration job has been submitted.  `F02B_NUMERICAL_CALIBRATION_v1` predeclares 45 fixed-budget
fit tasks and 122 reusable-fit probe tasks over development replicas `0,1,2`, all five dimensions,
three optimizer seeds, the `m=50` reference, a seed-11 `m={20,75,100,150,200}` resource sweep, and
two independent-allocation replays.  Its literal fit, probe, and combined matrix hashes are
`e53cabcb...513e`, `b729e755...ae7c`, and `d81aee9b...0114`.  All 100 primary validation inputs enter
the label-independent geometry scan; expensive support64 solves are selected only by predeclared
rank-boundary guard strata.

The design fixes absolute and scale-normalized mean/variance errors, projector metrics, exact-zero
rank semantics, physical-constraint spectral residuals with an analytic roundoff diagnostic, solver
tolerance sweeps, a full-q precision decomposition, higher-precision RBF fixtures, replica-2 holdout
handling, and family-wise maximum gates.  The earlier job 2810629 errors are explicitly excluded from
threshold fitting.  Discovery thresholds may be proposed only from independent error bounds and must
separate registered faults; confirmatory replicas `101..110` remain unopened.

The fit side is now implemented but is still not an execution authorization.  A public immutable
contract separates the prediction-free fit recipe, requested resources, observed allocation, and
raw-byte payload binding.  Before reading corpus bytes or allocating training tensors on CUDA, the
development-only runner independently queries its live Slurm record and process-visible CPU/GPU
hardware.  It then gives strict parsing, catalog authorization, and the authoritative loader the
same no-follow, byte-identical private snapshot of the NPZ, metadata, manifest, and catalog; only the
training split reaches released TERA.  It emits uniquely encoded binary32 parameter JSON plus a
non-recursive execution envelope and rechecks the corpus, catalog, source tree, and dependency files
after fitting.  Its 45-slot aggregator enumerates the identity-derived paths, reads regular
single-link files without following symlinks, requires one common Slurm array with unique per-task
job evidence, rejects mixed deployments/cohorts and every incomplete or unexpected path, and always
emits `freeze_ready=false`.  The registered launcher independently repeats the live `scontrol`
checks for `0-44%1`, one requested L40S, 16 CPUs, 64 GiB, eight hours, and
`OverSubscribe=EXCLUSIVE`; it has not been submitted.

The independent RBF fixture now recomputes projected conditional moments from exact binary32 dyadics
with `mpmath` at 160 and 256 bits.  Its certificate is deliberately narrow: it validates the
conditional for caller-supplied support basis and coordinates, not the support64 SVD, cutoff, or rank
selection, which still require companion N0 evidence.

The first probe-side foundation is now implemented as a fail-closed design but has not run.  An X-only
selector binds the split name `validation` and copies exactly 20 trajectories times five registered
times without reading `E` or `F`; the selector is explicitly not a substitute for future
task/catalog/replica authorization.  Pinned vendor KNN is called only on learned-isotropic
source-fp32 geometry; it persists both training positions and source-row identities, requires
disjoint train/evaluation sources, and fails on short training populations, nonfinite scaling or
distances, ties, and noncanonical vendor order.  No fp64 arm may reselect neighbours.  All 122
task-specific work plans are bound by the domain-separated SHA-256
`c815d848c8866ed085522d56f9db7aedef304a6d3c6e4ef3c24ee0be7f25498e`, including the exact
development/evaluation design, rank/strata/stress/full-q registries, the production
`1e-5` request, the preregistered fp32/fp64 sweep, `min(4*m*r,4096)` cap, strata counts, stress rows,
and the `m=50` four-arm full-q applicability.  Pure diagnostics now recompute dense residuals and
exact dense backward error.  Matrix-free inputs are explicitly caller-claimed rather than
misrepresented as verified: an asserted operator-norm upper bound yields only a lower endpoint, while
the claimed action `b-r` yields a conditional upper endpoint.  The generic pass field was removed.
Cholesky factors must be exactly lower triangular with positive diagonal and are measured by relative
factorization residual, without casting, clipping, or changing the dtype-tiny floor.

The canonical launch manifest fixes three separate submissions (`0-119%1`, singleton 120, singleton
121), requires distinct future `array_job_id` identities for their allocations, and binds the data,
fit stage, fit catalog, output paths, public numerical/resource policy, and catalog hashes.  Crucially,
it records `expected_fit_deployment` separately from `probe_deployment`: probe implementation work
must not retroactively pretend that the earlier fits ran from the later source commit.  Both stages
must share the TERA gitlink and catalog-generation identity, while source trees and dependency locks
may differ.  Private canonical policy bytes, not mutable public views, are the validation authority;
scheduler and probe-plan domains are rehashed on every build and validation.  The manifest still does
not execute or validate a live allocation.

The audited ORBIT build/solve exposure, N0/N1/N2 and full-q computations, stress/fixture execution,
probe artifact runner, strict 122-slot intake, threshold discovery/locked-holdout evaluator, and
probe launcher remain incomplete.
No fit or probe calibration job may be launched until that remaining chain is committed and audited;
every aggregate remains `freeze_ready=false` until a separate review binds its hashes into a frozen
F02b protocol.

---

## 2026-08-03 — F02b fixed-geometry oracle diagnostic: **exploratory decomposition only**

This development-only diagnostic does **not** pass N0, N1, or N2 and does not revive the terminated
F02 protocol.  It ran clean commit `5fa3a72409486c688f53d5de430cfa72808c70f0`, tree
`fc01b25ab0bffe12b85c0785772deebd50a47c50`, with TERA gitlink `b2382e10...` as Slurm job
2810629 on the exclusive four-L40S `interactivegpu` node `g019`.  The job completed with exit code
zero; elapsed time is excluded from all comparisons.  Its strict v3 artifact is
`runs/f02_same_m_diagnostic/replica0-d12-steps20-seed11-5fa3a72.json` (SHA-256
`ffbea890b67afe9e5742ccf3d7209623f040fe999d39baaad23d1dc4c83aa202`).  The artifact binds the
clean source tree, dependency hashes, corpus manifest and file hashes, full registered
configuration, and catalog SHA-256 `2dee429b...942`.

The diagnostic removes the earlier dtype-dependent geometry confounds.  One pinned-vendor fp32 KNN
call selected all 20 `m=50` neighbourhoods, and the identical source-row indices were injected into
every TERA, ORBIT, and independent support64 arm.  Native promoted-fp64 KNN happened to return the
same order and sets on this coordinate, but that observation is not generalized to near-tie cases.
The frozen source-fp32 rank rule retained rank 6 for every support64 and ORBIT arm, matching the six
known center-of-mass and total-momentum constraints.  The retained/discarded singular-value boundary
gap in source-quantized fp64 geometry ranged from `112.5` to `2.86e5` (the fp32 minimum was
`107.7`), but the smallest retained `s6/cutoff` ratio was only `1.255`.  Rank 6 is therefore not
called an N0 pass: the gap, cutoff-margin, constraint-residual, and perturbation-stability thresholds
remain TBD.

The rank bookkeeping also exposes an important precision boundary.  Native fp32 classified all
discarded modes as unresolved, whereas native fp64 resolved rank 12 for every target.  ORBIT64
therefore correctly records `basis_exact=false`: its residual certificate covers the fixed rank-6
support, not the full native-fp64 span.  Relative to support64, the fp64 q-projector was identical to
stored precision; the fp32 projector differed by `2.67476e-4` max-absolute and `6.63791e-4` maximum
relative Frobenius error.  No acceptance threshold is retrofitted to those observations.

Within the selected support, the independent dense oracle strongly validates the ORBIT algebra.
Support64 versus ORBIT64 differed by at most `3.14941e-10` in mean and `1.96024e-15` in latent
variance.  All 20 ORBIT64 solves met the requested `1e-10` tolerance (maximum fresh residual
`7.19647e-11`), while the oracle's maximum direct-solve residual was `1.81038e-15`.  Released TERA64
also agreed with support64 to `6.60870e-10` in mean and `2.56961e-13` in variance.  These are
unscaled, single-coordinate development errors and therefore calibration evidence, not a frozen N1
decision.

The matched-stopping N2 probe gives a mixed result.  At tolerance `1e-5`, both fp32 and fp64
converged on all 20 targets, with maximum fresh residuals `9.50933e-6` and `9.59858e-6`; their
predictions differed by at most `4.89081e-4` in mean and `2.87963e-6` in variance.  At tolerance
`1e-8`, fp64 converged on all targets, but fp32 converged on none: all 20 reached the 300-iteration
cap and had fresh residuals as large as `4.30660e-6`.  The associated `5.77451e-4` mean and
`3.52410e-6` variance differences cannot be treated as a converged N2 comparison.  Tightening the
fp32 request beyond its numerical floor is therefore not a viable repair.

Tracing the released dense baseline separates most of the original discrepancy from ORBIT's
iterative stopping rule.  TERA32 escalated its q-coordinate jitter to `1e-5` on 7 targets and
`1e-4` on 13 targets, while its function-coordinate jitter stayed at `1e-8`; TERA64 used `1e-8` for
both on every target.  Base support64 agreed with ORBIT64 to `3.14941e-10`; ORBIT64 versus TERA32
differed by `0.180658` in mean and `9.07684e-4` in variance.  Recomputing support64 with each
target's actual TERA32 function and q jitters reduced the worst-case maxima to `0.0106929` and
`3.09038e-5`, ratios of maxima of about 16.9 and 29.4.  No matched support solve re-escalated its
requested jitter.  Thus adaptive q regularization explains most, but not all, of the fp32
difference; remaining full-q fp32 assembly, rule-discarded-complement leakage, factorization, and
solve effects are not apportioned by this artifact.

F02b remains a draft and confirmatory labels remain locked.  Before freezing it, development
calibration must span dimensions, neighbourhood sizes, seeds, and rank boundaries; add
scale-normalized moment/projector errors, physical-constraint and permutation/perturbation checks,
a higher-precision reference, and a fresh residual diagnostic for released TERA's dense q solve;
and choose tolerances without using confirmatory labels.

---

## 2026-08-03 — F02 optimizer pilot: **stopped by the preregistered same-`m` gate**

The original F02 protocol did **not** pass development validation and must not be used for a
confirmatory or positive predictive claim.  Its first clean, exclusive-L40S pilot (Slurm job
2810520; replica 0, `D=12`, 20 TERA updates, seed 11, commit
`e27a0cd431d5a9d5258013a097b3c99cdb3907c8`) failed before optimizer selection or test access.
Released float32 TERA-50 and ORBIT-50, using the same fitted parameters and neighbours, differed by
`0.1805518866` in predictive mean and `0.0009048041` in latent variance, above the registered
`1e-4` gate.  The strict aggregate at BC path
`runs/f02_internal_optimizer/job-2810520/pilot-aggregate.json` (SHA-256
`fd4634f8ca84aa7e0235142118f617678dd498271e38f60b3731e37cc68b9392`) records one failed task,
zero valid tasks, `declared_subset_ready=false`, `analysis_ready=false`, and
`selected_update=null`.  The remaining 134 development jobs were not run.

A development-only diagnostic on the same corpus and fitted parameters (exclusive-L40S job
2810525, commit `d81e873ee293d60192977c6f62e8ae9d8030cd00`) localized the discrepancy.  Tightening ORBIT's
float32 CG target from `1e-5` to `1e-8` did not reduce it: all solves reached the 300-iteration cap
and the worst mean difference remained `0.1807389259`.  Prediction-only float64 TERA-50 and
ORBIT-50 instead agreed to `7.1487e-10` in mean and `8.1199e-14` in latent variance, while released
TERA itself changed by `0.1806576252` in mean and `0.0009076875` in variance between float32 and
float64.  The diagnostic artifact is
`runs/f02_same_m_diagnostic/replica0-d12-steps20-seed11-d81e873.json` (SHA-256
`50c0a5d0b9e9ec7610c6f0718a951c956bf7288ae885ae963348490a50765d95`).

Every local scaled difference matrix retained numerical rank 6 in float32, with reported condition
numbers from `2.59e7` to `4.74e8`.  This is consistent with the six center-of-mass and total-momentum
constraints in the `D=12` physical system and with instability in released TERA's redundant
`m^2`-coordinate float32 Cholesky, rather than an ORBIT stopping-tolerance error.  It is strong
diagnostic evidence, not yet an independent-oracle proof: fp32 SVD rank truncation remains a smaller
alternative explanation to test explicitly.

No tolerance will be relaxed and the frozen F02 protocol will not be silently rewritten.  Any
follow-up must have a new protocol identifier, preserve released fp32 TERA as an operational
baseline, add an independently implemented float64 dense support-space oracle, separately gate
support64 against ORBIT64 and ORBIT32 against ORBIT64, and compare larger-`m` ORBIT against ORBIT-50
at the same dtype.  Confirmatory labels remain locked until that revised protocol, thresholds, full
development selection, global recipe, and one-release authorization are frozen.

---

## 2026-08-03 — F02 confirmatory N-body corpus: **frozen and integrity-ready**

This is a data/provenance milestone, not a predictive result.  The preregistered F02 generator ran
from clean commit `a7d2a103aee4bcc0c58494905c6799266cb06187`, tree
`745af6587fa9fc7d26c6f1236e994a8208c43c98`, with TERA submodule `b2382e10...` as BC Slurm array
job 2810370.  It produced all 65 fixed-mass corpora: 15 development units (three replicas by five
state dimensions) and 50 untouched confirmatory units (ten replicas by five dimensions).

The strict catalog at BC path `runs/f02_nbody_data/job-2810370/catalog.json` reloaded every
NPZ/metadata/SHA bundle and reports 65 valid, 0 missing, 0 failed, 0 invalid, and 0 unexpected tasks.
All 65 semantic dataset hashes are unique.  Commit, tree, submodule, source hashes, source manifest,
repository root, and Slurm array provenance agree across tasks.  The catalog SHA-256 is
`2dee429bdaf50cc78cb40ba8b038f7e4731bb07127927c55607b528c1db66942`.

Generating the confirmatory corpora does not unlock their labels.  F02 test scoring remains blocked
until the optimizer budgets and dimension-specific ORBIT neighbour schedule have been selected only
from development validation data, serialized into a committed frozen recipe, and accepted by the
test-locked runner.  An independent gate audit showed that a recipe covering only one corpus would
still permit sequential test peeking.  The current runner therefore hard-disables confirmatory test
execution even after that per-bundle scaffold validates; release additionally requires one global
50-corpus task recipe, a single pinned analysis commit/hash, and a catalog-level one-release ledger.

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
