# Research log

The single narrative doc.  Historical numbers below were reported from
`experiments/nbody_benchmark.py`, but the referenced `runs/nbody_benchmark.json` artifact is absent
from the current checkout.  They are therefore not independently reproducible here.  Retractions and
scope downgrades are reported in place, loudly, naming the check that failed.

---

## 2026-08-04 — ORBIT-G30 paper N-body v4: **registered claim falsified**

Commit `076315efdeef4492897651515eaeeed95e8dd863` (tree
`992ee04b7048845b409de58ee722c80d8350abc6`) was deployed directly to BC and evaluated on the
complete 12-task paper-style N-body grid. Data array job 2814140 generated the four released
pair-loop datasets; benchmark array job 2814144 crossed particle counts 4/6/8/10 with seeds 6535,
8830, and 92357. Every task completed on a nonexclusive shared `short` node with one task, exactly
8 CPUs, no GPU, and thread pools fixed at 8. Wall time is descriptive only and is excluded from all
claims.

All four datasets contain the expected 9500 post-filter rows and dimensions 24/36/48/60. Their
SHA-256 digests are recorded in `releases/paper_nbody_v4_076315e.json`. The 12 task artifacts and
aggregate are complete and hash-locked; the aggregate at BC path
`runs/paper_nbody_v4/aggregate.json` has SHA-256
`ef4c6cddb9fd749415acafd00592eeafac204de6ba0a1a145a668ec09fd52cd6`.

Three-seed means are:

| particles | arm | value RMSE | value NLL | gradient RMSE | metric gate | resource gate |
|---:|:---|---:|---:|---:|:---:|:---:|
| 4 | TERA-20 | 0.010765173 | -2.513771 | 1.534284 | — | — |
| 4 | ORBIT-G30 | 0.010765257 | -2.517588 | 1.466608 | fail | pass |
| 6 | TERA-20 | 0.081240496 | -1.846455 | 2.160431 | — | — |
| 6 | ORBIT-G30 | 0.081240468 | -1.848878 | 2.135605 | pass | fail |
| 8 | TERA-20 | 0.264908031 | 0.814563 | 2.144300 | — | — |
| 8 | ORBIT-G30 | 0.264908031 | 0.813456 | 2.137720 | pass | fail |
| 10 | TERA-20 | 0.165832798 | 3.093693 | 3.313166 | — | — |
| 10 | ORBIT-G30 | 0.165833002 | 3.093718 | 3.312027 | fail | fail |

Thus `beats_TERA_under_registered_rule=false`: n=4 misses value RMSE by `8.34e-8`; n=10 misses
value RMSE by `2.04e-7` and NLL by `2.48e-5`; and the registered maximum-flop proxy fails for every
n=6/8/10 task. All state proxies pass, all primal and adjoint solves converge, and same-m TERA-20 /
ORBIT-20 agreement remains numerical-control quality. The paper's N-body table reports value and
gradient RMSE; value NLL here is the user-requested additional calibration diagnostic, computed with
predictive observation variance including fitted value-noise variance.

Post-result diagnosis must not revise v4. The guard currently computes complete value-plus-gradient
ORBIT-20 and ORBIT-30 predictions before selecting one. Removing the unused adjoint is worthwhile
but not sufficient under the existing conservative maximum proxy: a component-summary upper-bound
recalculation remains above TERA for every n=6/8/10 task. Any changed solver tolerance, resource
formula, candidate, or success rule therefore requires a new protocol and independent confirmation.

---

## 2026-08-04 — ORBIT-G30 dual guard: **refit-per-seed development robustness, formal data untouched**

The excluded two-particle study was repeated with an independent one-epoch released-TERA fit for
each of seeds 6535, 8830, and 92357, followed by complete 950-target float64 `m=20` and `m=30`
predictions and implicit mean gradients. At trust radius `0.02`, candidate-minus-`m=20` deltas were:

- value RMSE: `-3.51e-6`, `-2.10e-8`, and `-8.70e-8`;
- observation-variance value NLL: `-0.00701`, `-0.00685`, and `-0.00689`; and
- gradient RMSE: `-0.0892`, `-0.1001`, and `-0.0985`.

Every trust radius in the prespecified local sensitivity range `0.005, 0.01, 0.015, 0.02, 0.025,
0.03` improved all three metrics on every refitted seed. Thus `0.02` lies inside a common robust
interval rather than being an isolated optimum. The existing development NPZ uses the earlier
vectorized generator; its arrays differed from the pinned pair-loop version by at most about
`8.8e-14` in the direct full-dataset comparison, and it remains development-only.

A more principled nested-posterior innovation was also audited. Under fixed GP parameters,
`mean_30-mean_20` has conditional variance `variance_20-variance_30`; its standardized magnitude was
strongly correlated (`0.911` to `0.996`) with the raw expansion's squared-value-error change and
identified the catastrophic points. Used alone, however, it still slightly worsened value RMSE on
all three seeds, so that tempting gradient/NLL-only result was rejected. The frozen dual guard keeps
the `0.02` trust radius and additionally requires nonincreasing latent variance plus a two-sided
Bonferroni posterior-innovation check at family-wise `alpha=0.01`. These extra conditions changed no
development selection, but add a model-derived fail-closed check for the formal data.

Before any formal result existed, the success rule was also simplified to the paper's reporting
unit: lower three-seed mean value RMSE, value NLL, and gradient RMSE for every particle count, with
mean and standard deviation reported. Per-seed joint wins remain visible diagnostics but are no
longer an extra non-paper claim gate. Convergence and analytic resource matching still apply to
every task.

No 4/6/8/10-particle dataset was generated or read, no Slurm job was submitted, and wall time was
not used as resource evidence.

---

## 2026-08-04 — ORBIT-G30 guarded expansion: **preregistered from excluded n=2 development data**

The raw `m=30` expansion is not a credible standalone candidate: on the complete 950-target excluded
two-particle split for seed 6535, rare local failures increased value RMSE from `0.000322888` at
`m=20` to `0.0424413`, even though gradient RMSE improved from `0.791531` to `0.638605`. This failure
motivated a label-free trust guard rather than suppressing the diagnostic.

`ORBIT-G30` computes both local conditionals and accepts `m=30` only when
`|mean_30-mean_20|/sqrt(latent_variance_20) <= 0.02`; otherwise it returns the complete `m=20`
conditional. The threshold was selected and frozen on the excluded two-particle development setting,
before any reported 4/6/8/10-particle test set was generated or read. Using one fixed learned-parameter
snapshot across all three development splits, the guarded arm improved all three metrics on each
split. The candidate-minus-base deltas for seeds 6535, 8830, and 92357 respectively were:

- value RMSE: `-3.19e-6`, `-1.84e-8`, `-3.82e-8`;
- observation-variance value NLL: `-0.00702`, `-0.00685`, `-0.00694`; and
- gradient RMSE: `-0.0891`, `-0.1001`, `-0.0941`.

This is candidate-development evidence, not a paper benchmark result: n=2 is excluded from the
reported table, the parameter snapshot was not refit per development split, and no statistical claim
is made. Formal accounting pays for both conditionals, summing their operation proxies and using
their sequential maximum state. The temporary exact-order dataset matched the pinned upstream
generator on a direct bitwise small-case check; its full NPZ SHA-256 was
`74fe4131ae31da7158e384ddb5027a8170f957601b643b662dc563741de48efd`. No Slurm job was submitted.

---

## 2026-08-04 — Paper N-body v2 full-gradient and NLL path: **implemented locally, not submitted**

The paper-aligned runner now reports both primary N-body metrics, value RMSE and full-gradient RMSE,
plus the preregistered repository diagnostic value NLL. Value NLL is computed from predictive
observation variance, including the fitted value-noise variance; no gradient NLL is claimed because
the benchmark does not construct a full gradient predictive covariance. Both methods define the
gradient as the derivative of their scalar posterior mean with nearest-neighbour membership held
piecewise constant.

ORBIT differentiates its local conditional with an implicit adjoint system. Its primal and adjoint
CG solves run without retaining their iteration tapes, and both solves must meet the fixed `1e-10`
fresh-residual tolerance. Actual matvec and preconditioner counts from both solves are charged to the
resource proxy, together with fixed conservative factor-four state and reverse-pass allowances. The
registered win rule now requires lower value RMSE, value NLL, and gradient RMSE on every seed task
and every dataset mean, in addition to convergence and resource matching.

The compact local suite passed 28 tests. A two-target, 64-training-row, `m=20`, float64 smoke on the
non-reported two-particle data found a maximum ORBIT implicit-gradient versus TERA difference of
`7.58e-9`; both adjoint solves converged. This checks implementation consistency only, not predictive
superiority. No paper dataset array was run and no Slurm job was submitted or cancelled.

---

## 2026-08-04 — Paper N-body TERA/ORBIT runner: **implemented locally, not submitted**

The compact paper-aligned benchmark is now executable as 12 tasks: the authors' three seeds crossed
with the four reported N-body sizes. Each task uses the paper's normalize-before-split 90/10 data
path, fits released TERA once at its native `m=20` configuration, and evaluates `TERA-20`, the
same-neighbour `ORBIT-20` control, and the fixed `ORBIT-30` resource-expansion hypothesis on all 950
test rows in float64 prediction arithmetic. The aggregate requires lower RMSE and NLL for ORBIT-30
on every seed task and every dataset mean, plus convergence and analytic state/flop resource matches;
it makes no wall-clock, gradient, or significance claim.

Both data generation (4 tasks) and benchmark execution (12 tasks) now have nonexclusive shared-node,
8-CPU, no-GPU Slurm entries. A live read-only cluster check showed `select/cons_tres` with
`CR_CORE_MEMORY` and `short: OverSubscribe=NO`, so the entries request neither `--exclusive` nor
`--oversubscribe`. The two pure benchmark tests and one aggregate test pass under an
available Python 3.10 scientific environment; 24 existing TERA/ORBIT adapter tests also pass. The
default login Python is 3.9 and cannot import the repository's existing `dataclass(slots=True)` code,
so it was not used as test evidence. Only the two-particle smoke dataset exists in this checkout;
the four reported paper datasets must be generated before execution. No Slurm job was submitted or
cancelled.

A two-target, 64-training-row integration smoke on the non-reported two-particle data exposed why
the solve tolerance is part of the scientific protocol: at `1e-5`, same-`m` TERA/ORBIT means differed
by about `6.68e-4`; at `1e-10`, the maximum mean and latent-variance differences fell to about
`8.68e-11` and `2.78e-16`, respectively, with both ORBIT arms converged. The benchmark therefore
uses `1e-10`; its extra iterations count against ORBIT's analytic resource proxy.

---

## 2026-08-04 — Predictive evaluation simplified to the paper benchmark

The active predictive evaluation now follows the DSoftKI paper's toy N-body benchmark: reported
particle counts `4, 6, 8, 10`; 10,000 generated rows followed by the paper's 5% gradient-norm filter;
a 90/10 train/test split; three released-script seeds; and per-dataset value/gradient RMSE summarized
as mean and standard deviation. Shared-node timing is descriptive only and cannot be compared with
the paper's GPU timing.

The pending custom F02b 45-fit/122-probe calibration workflow is superseded and must not be
submitted. Its committed artifacts remain as historical numerical diagnostics. Future benchmark
tests are limited to small data/split and result/metric smoke checks; kernel and ORBIT correctness
remain covered by their existing unit tests. No Slurm job was submitted or cancelled for this
change. See `docs/PAPER_NBODY_BENCHMARK.md`.

---

## 2026-08-04 — Strict F02b 122-slot probe intake: **implemented and locally verified, not run on a cohort**

The task index advanced to v2 before any corpus run so it also binds the actual sorted runtime
package record and Slurm array/job/node identities.  The filesystem aggregate now enumerates all 122
identity-derived task directories and all 24,400 target filenames without globbing.  No-follow
descriptors, single-read hashing, regular-file and hard-link checks, directory stability checks, and
the target/index/envelope validation chain precede cohort admission.  Expected launch, fit-catalog,
fit-payload, and deployed-source identities are supplied explicitly rather than learned from the
first successful task.

The catalog retains every invalid or missing slot, requires one common runtime record, and audits
the registered scheduler topology: tasks 0–119 share one array identity, tasks 120 and 121 use two
additional pairwise-distinct allocations, and all 122 element job IDs are unique.  It may mark a
complete structural cohort `analysis_ready=true`, but hard-codes `freeze_ready=false` and can be
published only by exclusive atomic creation outside the input root.  The compact test populated one
complete 200-target task and left the other 121 registered slots missing; the valid slot was accepted
and the catalog correctly remained not analysis-ready.  No corpus, labels, GPU, Slurm job,
runtime/cost comparison, or confirmatory replica was accessed.  Threshold discovery and the
authorized shared-CPU runner remain pending.

---

## 2026-08-04 — Canonical F02b 200-target task index: **implemented and locally verified**

Each probe task now has a small canonical index over exactly 100 target positions times the two
registered ORBIT dtypes.  Target files use identity-derived names, remain separate from the index,
and are bound by raw SHA-256.  Before emission, the builder parses every target, matches fp32/fp64
target and neighbour identities, requires one common source arm and main N0 grid/stratum identity,
and enforces the registered support64, full-q, and stress counts.  It also binds the selected fit
payload/catalog, launch manifest, probe deployment, and the shared 8-CPU public execution envelope.
Schema v2 also binds the actual sorted runtime package record and Slurm array/job/node identities.

One complete synthetic resource-sweep task exercised all 200 target records and the index/envelope
pair; removing one target was the single registered rejection check.  This remains integrity and
mechanism evidence, not corpus accuracy, runtime, cost, or superiority evidence.  No corpus, labels,
GPU, Slurm job, or confirmatory replica was accessed.  Threshold workflow and the authorized runner
remain pending.

---

## 2026-08-04 — F02b full-q/stress target artifact integration: **implemented and locally verified**

The immutable target artifact schema is now v2.  A CPU-float64 ORBIT target can carry matching
support64, released full-q, and rank-boundary stress evidence without weakening their separate
identities.  Full-q attachment requires the same task, target, main N0 grid/stratum hashes, selected
rank, and 50-neighbour rows; its four arm names and all represented-system/RHS digests are retained.
Stress attachment requires the same task/source arm/target, the registered `m=D-5`, exact tolerance
and iteration cap, CPU-float64 projectors, and its separate neighbour, geometry-grid, stratum, and
stress-binding hashes.  Canonical extraction copies all nested dictionaries and tensors before
hashing, while mismatched stress identity is rejected.

One existing synthetic full-q fixture and the single compact stress fixture exercised the new
positive paths; the latter also covered the identity rejection.  This is artifact integrity and
numerical-mechanism evidence only.  No corpus, labels, GPU, Slurm job, runtime/cost comparison, or
confirmatory replica was accessed.  Strict 122-slot intake, threshold discovery, and the authorized
shared-CPU runner remain pending.

---

## 2026-08-04 — Registered F02b rank-boundary stress suite: **implemented and locally verified, not executed on corpus**

The work-plan schema is now v3 with hash
`11b3dd9863cbd010eb50e95f4f4a5941080eb10186731a34f0625dd9fd5b6586`; it supersedes the earlier
v2 hash before any probe corpus run.  The plan now binds the stress computation to exact promotion
of the source-fp32 arm onto CPU float64, the source-selected neighbour rows and absolute cutoff, a
zero-start `1e-10` solve with a formula-derived iteration cap, four deterministic SHA-counter
Rademacher probes, reverse-neighbour permutation, adjacent-pair pi/4 support rotation, exact-zero
ambient augmentation, and a native-fp64 cutoff comparison.

The registered executor scans all 100 label-free N0 geometries only for eligible repeat-zero,
seed-11 reference tasks and runs the five checks on the single worst target.  Its matrix-free full-q
operator diagnoses support/complement action without allocating the dense q system; the q RHS and
conditioned observations are assembled independently and differenced against the ORBIT support map.
A single synthetic registered fixture (`m=7`, rank 6, target 73) converged with relative residual
about `3.32e-11`; the two direct-to-support-map maximum differences were about `7.77e-16` and
`7.99e-15`.
Permutation, support rotation, and exact-zero augmentation preserved the posterior moments within
the compact fixture assertions.  This is numerical mechanism evidence only, not corpus accuracy,
runtime, cost, or superiority evidence.  No corpus, evaluation label, GPU, Slurm job, or
confirmatory replica was accessed.  Artifact integration, strict intake, threshold discovery, and
authorized shared-CPU calibration remain pending.

---

## 2026-08-04 — Registered F02b full-q precision ladder: **implemented and locally verified, not executed on corpus**

The released TERA `m=50` diagnostic now calls the pinned private one-target predictor and intercepts
its actual function and q-coordinate Cholesky inputs, selected jitters, and factors.  A separate
recomposition using the pinned kernel primitives must reproduce both captured matrices bit-for-bit
before the four registered arms run: native fp32/fp32, complete fp32 system promoted to fp64,
native quantized-input fp64/fp64, and complete fp64 system cast to fp32.  Each successful solve is
checked against its own represented system and the native fp64 canonical system.  The artifact-ready
result also records `H`, q, cross, unconditional and Schur discrepancies; exact Frobenius solve and
Cholesky backward errors; actual function/q jitters; raw moments; fixed-neighbour identities; and
support/support-complement matrix, RHS, conditional-observation, and solution norms.  Loss of
positive definiteness after a registered cast is preserved as scientific output rather than repaired
with an unregistered jitter.

The committed synthetic registered-shape test exercised an actual `2500 x 2500` q system at
`m=50`, selected target 94, and source rank 6 using the shared, thread-limited 8-CPU local policy.
The fp32 q jitter escalated to approximately `1e-3`, while native fp64 accepted `1e-8`.  Promoting
the complete fp32 system before solving reduced its own relative residual from about `5.28e-7` to
`1.69e-15`, but its residual against the canonical fp64 system remained about `6.58e-6` versus
`6.53e-6` for the fp32 solve.  Thus solve precision was not the main discrepancy in this fixture;
assembly and/or selected jitter dominated.  The native-fp64 system cast to fp32 was not positive
definite and is recorded as a failed fourth arm.  These are synthetic numerical mechanism results,
not corpus accuracy, cost, or superiority evidence.  No corpus, evaluation labels, GPU, Slurm job,
or confirmatory replica was accessed.  Stress execution, artifact integration, strict intake,
threshold discovery, and authorized shared-CPU calibration remain pending.

---

## 2026-08-04 — F02b support64 and target artifacts: **implemented and locally verified, not executed**

The independent dense support-space oracle now accepts a registered absolute rank cutoff.  Its
default standalone rule is unchanged, while the F02b path must use the exact cutoff produced by the
source-fp32 N0 SVD with strict `s > cutoff`.  The oracle still performs its own SVD and dense
Cholesky in CPU float64, records both operational and native-fp64 cutoff/rank, and does not import
ORBIT.  The registered adapter accepts only the exact-promoted fp64 arm and geometry-selected strata,
reuses the pinned fp32 neighbour rows, revalidates the arm/geometry/strata after the dense solve, and
persists both ambient and q-coordinate support projectors.  On the synthetic registered fixture,
support64 agrees with the independently tightened `1e-12` ORBIT64 solve within the predeclared test
tolerance; this is a correctness control, not predictive evidence.

Completed ORBIT targets, optionally paired with matching support64 evidence, can now be copied
immediately into canonical JSON bytes and bound by raw SHA-256.  The artifact includes represented
operator inputs, fresh/replayed residual evidence, certificates, rank and Cholesky diagnostics,
support64 spectra and projectors, and exact source/strata hashes.  Later mutation of execution
tensors or dictionaries cannot alter the immutable bytes.  Noncanonical JSON, duplicate keys,
nonfinite tensors/NaNs, mismatched support identities, and inconsistent saved residuals are rejected.
The focused oracle/execution/artifact suite passes 60 tests on a shared, thread-limited 8-CPU local
environment, and the full local suite passes (`713 passed, 1 skipped`).  No corpus, labels, GPU, or
Slurm job were accessed; full-q/stress execution, strict 122-slot intake, threshold discovery, and
an authorized runner remain pending.

---

## 2026-08-04 — F02b resource policy v2: **shared 8-CPU only, locally verified, not submitted**

The earlier unexecuted F02b v1 exclusive-L40S deployment is superseded.  The active calibration ID
is now `F02B_NUMERICAL_CALIBRATION_v2`: fit and probe work must use one shared `short`-partition task
with exactly eight requested CPUs, 64 GiB host memory, an eight-hour limit, array concurrency one,
and zero requested or visible GPUs.  Both the launcher and Python runner independently require
`OverSubscribe=OK`, reject any GPU TRES or exclusive allocation, and bind the observed policy into
the execution envelope.  CPU math-library thread controls are fixed at eight.  Shared-node runtime
remains diagnostic only and cannot support performance or cost claims.

Changing the fit device from CUDA to CPU is a registered numerical-coordinate change, not a
scheduler-only edit.  The v2 fit, probe, and combined task hashes are respectively
`7272f823...5a39`, `98a44d16...4eca`, and `0ead06b0...1f4`; the CPU fit-recipe hash is
`cc4a891a...44bb`, and the probe work-plan hash is `7cfefba0...5813`.  Execution-envelope, fit
payload, fit-catalog, probe-work-plan, and probe-launch-manifest schemas were all advanced to v2, so
no v1 fit artifact can enter the new cohort.  The focused resource/fit/probe contract suite passes
284 tests, and the full shared 8-CPU local suite passes (`703 passed, 1 skipped`).  No Slurm job was
submitted or cancelled during this migration.

---

## 2026-08-04 — F02b ORBIT probe execution: **implemented and locally verified, not executed**

The pure numerical probe executor now has a two-phase, label-free geometry path.  A source-fp32
factory privately snapshots the training tensors, the 100 public evaluation coordinates and
identities, the frozen fit parameters, and the authoritative work plan; it recomputes the pinned
vendor KNN rows rather than accepting caller-supplied neighbours.  A domain-separated source digest
binds every training/evaluation tensor, parameter, neighbour identity, and plan.  Phase-boundary
checks recompute that full content digest, in addition to checking Tensor versions, so `.data` or
NumPy mutations cannot run under stale provenance.  The fp64 arm can only be obtained by exact
promotion and is verified by reconstructing and hashing its unique binary32 source.

All 100 primary targets first undergo one direct-SVD source-fp32 geometry scan without evaluation
labels.  Each record binds the original standardized differences, singular spectrum, coordinates,
rank boundary, native and operational cutoffs, neighbour identities, and source arm.  Only the
complete ordered population can select the registered worst/median/best strata; the resulting
rank-grid and selection digests travel with every target execution and determine whether that target
receives the full tolerance sweep.  A caller can no longer freely request a stratum sweep.

The fp32 target build consumes the exact authenticated geometry object from that scan and therefore
does not perform a second SVD.  Its paired fp64 build performs one native-fp64 SVD but uses the exact
absolute cutoff frozen by the source-fp32 SVD, rather than recomputing `smax*max(D,m)*eps32` from a
different spectrum.  The artifact-facing rank record separately reports the operational and native
strict ranks; `basis_exact=false` remains expected when the fixed fp32 rule discards a mode that is
resolvable in fp64.  The generic public ORBIT builder always derives geometry from its own inputs;
precomputed geometry is confined to the private registered path after difference/digest validation.

Each target/dtype builds one reusable system and runs registered tolerances as independent
zero-start solves.  The production `1e-5` object is reused from that sequence, and nonconvergence is
retained as scientific output.  The canonical matrix-free action and residual are the solver's final
fresh `A(x)` and exact represented `b-A(x)`.  An immediate independent replay is persisted
separately with max-absolute and 2-norm differences; it is not required to be bitwise identical on
CUDA and does not replace the canonical residual in the conditional backward-error diagnostic.

Static checks and 89 focused execution/operator/reusable-system tests passed.  The full repository
suite passed on a shared, thread-limited 8-CPU environment (`703 passed, 1 skipped`).  Shared-host
runtime is not performance or cost evidence.  No corpus, protected evaluation label, GPU, Slurm job,
or confirmatory replica was accessed.  The authorization-aware runner, immutable artifact emission,
support64/full-q/stress arms, strict intake, threshold discovery, and protected evaluation remain
unimplemented; F02b is not frozen and no predictive superiority claim follows from this milestone.

---

## 2026-08-03 — F02b ORBIT reusable-system evidence: **implemented and locally verified, not executed**

The probe-facing ORBIT numerical core now separates one-time local-system construction from
independent zero-start solves.  `build_local_value_system` fixes the direct-SVD geometry, full
distance Gram, actual Kff jitter and Cholesky factor, structured Schur operator, observations, and
default preconditioner once per target/dtype.  `solve_local_value_system` can reuse those exact
objects across the registered tolerance sweep without warm starts or implicit result caching, so the
production `1e-5` result can be the sweep result itself rather than an untracked duplicate solve.
The historical `predict_local_value` interface delegates to this split and retains its legacy mean
evaluation order.

Every returned CG result now binds the final source-dtype `A(x)` action, fresh residual, distinct
recursive residual, requested tolerance, cap, termination reason, fresh-check count, replacement
count, and exact matvec/preconditioner counts.  A recursive threshold crossing that fails a fresh
check restarts from the fresh residual; reaching the cap is always reported as `maximum_iterations`.
The post-update recurrence now fails closed on a nonfinite or nonpositive new preconditioned residual.

The trusted GP builder records an analytic selected-support eigenvalue lower bound from the declared
gradient-noise model and transformed q-jitter.  It also records a matrix-free block-row/Frobenius
operator-norm upper bound conditional on the PSD kernel and positive function-noise-plus-jitter floor.
The realised-observation functional `h=z-QKff^-1 y` gives the nominal exact-arithmetic mean solve
bound `||h|| ||r||/lambda0`; the legacy and functional means and their signed reassociation delta are
both retained.  These bounds cover only the represented selected-support iterative solve and are
explicitly source-dtype, non-directed-rounding diagnostics with `floating_point_rigorous=false`.

Local verification passed 110 focused ORBIT/F02-wrapper tests and the full repository suite
(`668 passed, 1 skipped`).  The new dense audit spans 24 RBF/Matérn, scalar/ARD, three-noise-model,
and full/truncated-rank combinations; all analytic lower bounds stayed below the stored dense
minimum eigenvalue and all asserted upper bounds covered its spectral norm.  Six zero-noise-floor
cases correctly made the solve certificates unavailable.  These shared-host tests are correctness
checks only, not performance evidence.  No corpus, GPU, Slurm allocation, fit artifact, protected
label, or confirmatory replica was accessed.

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
