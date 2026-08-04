# F02b numerical calibration matrix

Calibration ID: `F02B_NUMERICAL_CALIBRATION_v2`

Status: **PREDECLARED DEVELOPMENT-ONLY DESIGN — not an F02b protocol freeze, not a numerical-gate
pass, and not authorization to read a confirmatory label.**

This document fixes the population, target-selection rules, diagnostics, and failure policy for the
first F02b numerical calibration.  This document and the complete implementation must both be
committed before any calibration job is submitted.  Every task must bind the exact clean source
commit and the literal fit/probe matrix hashes exported by `cluster/f02b_calibration_grid.py`.

The immutable grid and pure metric functions are only a pre-registration scaffold.  They do not
authorize submission.  Fit/probe runners, strict schemas and aggregators, the high-precision fixture,
and Slurm recipes must all be committed, tested, and shown to reject incomplete provenance before the
first array is launched.

Implementation checkpoint: the public execution-envelope contract, training-only fit runner,
strict 45-slot fit aggregator, exact shared 8-CPU fit launcher, and independent arbitrary-precision
RBF fixture are now present.  The label-free probe foundation is also present: it selects exactly the
100 registered development-validation coordinate rows without touching `E` or `F`, freezes pinned
vendor neighbours only in source fp32, and expands all 122 task records into the domain-separated
probe-core work-plan hash
`11b3dd9863cbd010eb50e95f4f4a5941080eb10186731a34f0625dd9fd5b6586`.  That hash also covers the
development replicas, exact evaluation design, neighbour/tie policy, rank rule, strata, stress
registry, and four-arm full-q registry.  The neighbour boundary rejects short training populations,
train/evaluation source overlap, nonfinite fp32 scaling or distances, and every selected or boundary
distance tie; the X-only selector is not by itself corpus authorization.
The launch-manifest contract separately binds the earlier fit deployment and the later probe
deployment, the fit catalog raw/integrity/cohort hashes, the work-plan hash, and the fixed three-array
topology `0-119%1`, `120-120%1`, `121-121%1`.  This manifest is an input contract, not a launcher or
submission authorization.  Its builders and validators reload private canonical policy bytes and
recompute the scheduler and probe-plan domains on every call; mutable public presentation objects are
not trust roots.  Fit and probe deployments may have different source trees and dependency locks,
but must share the TERA gitlink and catalog-generation commit/tree.  Dense diagnostics recompute the
exact represented-system residual and normwise backward error.  Matrix-free diagnostics instead
label their inputs as caller claims and
return a conditional backward-error interval: an asserted operator-norm upper bound gives the lower
endpoint, while `A(x)=b-r` gives the upper endpoint.  No generic matrix-free backward-error pass
field exists.  Cholesky diagnostics require an exactly lower-triangular, positive-diagonal factor and
report relative factorization residuals.  Every diagnostic records dtype/device evidence and uses
the fixed dtype-tiny floor.

The fit aggregator reads each canonical payload/envelope pair once
through no-follow descriptors, rejects any link alias or unregistered path, binds an explicit expected
source/tree/gitlink/dependency/catalog deployment, requires one Slurm array with unique per-task job
evidence, and can never set `freeze_ready=true`.  The fit runner itself validates live Slurm and
process-visible hardware before any corpus read or training-tensor allocation, then uses one private
byte-identical snapshot for bundle authorization and loading rather than reopening mutable input paths.  The
arbitrary-precision fixture certifies posterior moments only for a caller-supplied support basis and
coordinates; support construction, rank, and cutoff remain N0 obligations.  The audited reusable
ORBIT system and authenticated two-phase N0/ORBIT32/ORBIT64 execution path are present and locally
tested.  The independent support64 adapter now consumes only the exact-promoted CPU-float64 arm,
registered neighbours, selected N0 strata, and the frozen absolute source-fp32 cutoff.  Completed
target evidence can be copied immediately into immutable canonical bytes with a raw SHA-256.  The
registered full-q adapter now intercepts the pinned released one-target path, authenticates its
rebuilt intermediates against the actual function and q Cholesky inputs, and executes the four
registered precision arms on CPU.  The registered stress adapter now fixes its exact-promoted
CPU-float64 solve, source-fp32 neighbours and cutoff, zero start, tolerance and iteration cap,
hash-derived probes, reverse permutation, adjacent Givens support rotation, zero-coordinate
augmentation, and native-fp64 cutoff comparison.  Full-q/stress artifact integration, strict
122-slot intake and aggregate, discovery/locked-holdout threshold workflow, and the probe Slurm
runner remain incomplete.  This checkpoint therefore still authorizes no Slurm submission and no
confirmatory access.

The earlier v3 artifact from job 2810629 motivated the strata and fault tests but is excluded from
fitting numerical thresholds.  In particular, its observed errors may not be rounded upward and
adopted as acceptance thresholds.

## Scientific boundary

This calibration asks whether the fixed source-fp32 support rule, support64 oracle, and ORBIT
implementations can be separated from known numerical faults on development data.  It does not:

- select an optimizer-update budget or a resource neighbourhood;
- compare predictive scores or claim that ORBIT beats another method;
- revive the terminated original F02 135-task grid;
- certify released TERA32 as a mathematical oracle; or
- open, tensorize, predict, or score any replica in `101..110`.

All fitted models use the historical 20-update configuration solely to span realistic learned
parameters.  Once F02b later selects a final optimizer budget and neighbourhood schedule, the final
configuration must be re-audited under frozen numerical thresholds.  This calibration cannot stand
in for that audit.

## Immutable task population

### Fit matrix: 45 tasks

The primary fit population is the Cartesian product, in replica/particle/seed order:

```text
replica       = 0, 1, 2
n_particles   = 2, 4, 6, 8, 10
D             = 6 * n_particles = 12, 24, 36, 48, 60
optimizer seed = 11, 29, 47
train_steps   = 20
training_m    = 20
kernel        = RBF
```

Every other fit option follows the registered v2 recipe: CPU float32 released TERA training, batch
size 256, learning rate 0.01, fixed graph, isotropic
lengthscale initialized at 1, learned outputscale/value-noise/gradient-noise/lengthscale, and no
weight decay.  A fit artifact stores only strict JSON parameters and provenance; no pickle or
executable model state is accepted.

The matrix is one third of the terminated F02 optimizer grid because it contains only one fixed
update count.  It produces no `selected_update` field and cannot be consumed as optimizer-selection
evidence.

### Probe matrix: 122 tasks

Probe tasks reuse a fit artifact from a strict, complete 45-fit catalog:

1. all 45 fits receive one `m=50`, `repeat_id=0` reference probe;
2. the 15 seed-11 fits additionally receive `m={20,75,100,150,200}` resource-sweep probes; and
3. the `(replica=0,D=12,seed=11,m=50)` probe is replayed with `repeat_id=1,2` in distinct shared
   allocations.

The seed-11 large-`m` sweep is a sparse numerical calibration, not proof that seed and neighbourhood
size do not interact.  Before an F02b freeze, every finally selected resource `m(D)` must pass the
already-frozen gates for all 45 replica/dimension/seed cells on the complete development validation
design.

Each non-replay `m=50` probe also performs geometry-only scans at
`m={D-7,D-6,D-5}`.  These points straddle the physical transition
`r_phys=min(m,D-6)`; `m=D-5` is the first point with a rule-discarded q direction.  Geometry-only
work never fits a new model and never uses target values.

The canonical fit and probe records, their exact ordering, and their SHA-256 hashes are source
constants.  Negative indices, duplicate identities, a caller-supplied scientific coordinate, or a
matrix hash mismatch are structural failures.

```text
fit matrix SHA-256     7272f823a2bfc0f52cbfc2e27ae3a56b2f668e3ca2abff054de9209cd2fa5a39
probe matrix SHA-256   98a44d167f6a34d3e94dcffd026d030e56bcf30f77a0bd43810f16b311e54eca
combined SHA-256       0ead06b0e2f6de24c49f4bf6f999f90690ff1fb82be3585cc212bdd11fd411f4
```

## Development rows and label-independent strata

N0 geometry runs over all 100 primary validation inputs: five fixed time indices
`{0,25,50,74,99}` in each of 20 held-out development trajectories.  Posterior comparisons use a
small, predeclared subset because support64 is cubic in `m*r`.

For each target and `m`, let

```text
cutoff = s1 * max(D,m) * eps(float32)
r      = min(m,D-6)
g+     = log2(s_r / cutoff)
g-     = log2(cutoff / s_(r+1))
guard  = min(g+,g-)
```

An absent retained or discarded side has no finite guard contribution.  Negative guard means that
the expected boundary is already on the wrong side of the fixed cutoff.  Targets are sorted by
`(guard, target_source_index)`; no value, gradient, prediction, residual, or score participates.  For
the even population of 100 targets, “median” below means the upper median at zero-based sorted index
`100//2=50`.

- `m=50`: support64 uses the worst, median, and best targets.
- seed-11 extra prediction `m`: support64 uses the worst and best targets.
- seed-11 `m=D-5`: the worst target additionally receives the support/complement, permutation,
  support-rotation, exact-zero-augmentation, and discarded-mode-leakage stress suite.

Missing strata, duplicate source identities, a changed target order, or an N0 failure remains in the
artifact and makes the aggregate not freeze-ready.  A failed or expensive `m=200` probe may not be
silently removed after execution.  Each fit and probe task uses one shared node, one task, exactly
eight requested CPU cores, zero requested and zero visible GPUs, 64 GiB host memory, and an
eight-hour wall-time limit on `short`.  The live Slurm record must report `OverSubscribe=OK`; any
exclusive allocation or GPU TRES is a structural failure.  `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` are fixed at eight.  Arrays run at maximum
concurrency one, so the registered deployment requests at most eight CPUs at a time;
reproducibility replays use distinct allocations.  OOM or timeout at the registered limit is a
calibration failure.  Removing `m=200` requires a new pre-run design version that also removes it
from future F02b selection.

## N0 geometry and physical constraints

The operational rank rule remains `source-fp32-smax-maxshape-eps-v1` with strict `s > cutoff`.
Every geometry target records the complete spectrum, cutoff, selected and physical rank,
`s_r/cutoff`, `cutoff/s_(r+1)`, both log2 guards, `s_r/s_(r+1)`, absolute and fractional discarded
squared-singular-value energy, and native-fp32 and promoted-fp64 rank classifications.  Energy is
reported both at the operational selected rank and the physical expected boundary.  Rank equality
alone is insufficient.  For an exactly zero spectrum, `cutoff=0`, strict selected rank zero, and
JSON-null undefined ratios/guards are the only valid convention; a zero cutoff with any positive
singular value is a structural failure.

The six physical constraints are evaluated in standardized coordinates.  Canonical constraint
metrics take the already represented source-fp32 standardized differences, promote those values
exactly to float64 without recomputing them, and use CPU float64 linear algebra with persisted
float64 masses and spans.  CUDA, float32 accumulation, and TF32 are not admissible for this gate.  For
position coordinate `q_(i,a)`, the constraint coefficient is `mass_i * x_span[q_(i,a)]`; for momentum
coordinate
`p_(i,a)`, it is `x_span[p_(i,a)]`.  For `R=C Delta`, the sole scale-normalized constraint residual is

```text
scale_2 = ||C||_2 ||Delta||_2
floor   = finfo(float64).tiny
eta_2   = ||R||_2 / max(scale_2, floor)
```

No task-supplied or post-hoc floor is accepted.  A first-order matrix-product roundoff estimate is
reported separately using the fixed analytic diagnostic

```text
eps     = finfo(float64).eps
gamma_D = D*eps / (1-D*eps)
Bhat_2  = gamma_D || |C| |Delta| ||_2
rho_2   = ||R||_2 / max(Bhat_2, floor)
eta_exc = max(||R||_2-Bhat_2, 0) / max(scale_2, floor).
```

`Bhat_2` is an estimate, not a directed-rounding upper bound: `rho_2 <= 1` is never by itself an
acceptance certificate or a contribution to `U_j`.  The same vector-2-norm construction is retained
per constraint row, together with raw residuals and the Frobenius residual as a descriptive statistic.
Applying a raw-coordinate constraint matrix directly to standardized differences, mixing matrix
norms inside `eta_2`, or changing dtype/device is a test failure, not an alternative convention.

The geometry fault suite covers exact rank zero, repeated columns, exact linear dependencies,
cutoff equality and one-ULP perturbations on both sides, fixed neighbour permutations, and exact-zero
augmentation.  Augmentation may not change `max(D,m)` unless the original cutoff is explicitly held
fixed as a representation-only test.  Physical SO(3) rotations are not substituted for support
rotations because coordinatewise normalization does not generally preserve that symmetry.

## N1 and N2 metrics

N1 compares support64 with ORBIT64 on the fixed source-fp32 support.  N2 compares ORBIT32 with
ORBIT64 using identical fp32-selected neighbours, rank cutoff, parameters, target order, requested
tolerance, and iteration cap.  Every N1/N2 comparison metric uses CPU float64: already represented
float32 outputs are promoted exactly, already represented float64 outputs are transferred without
downcast, and no reference is ever reduced to candidate precision.  Metric helpers reject any other
comparison dtype/device.  Artifacts record each arm's source dtype plus comparison dtype/device and
matmul policy.
`basis_exact=false` in promoted fp64 is expected when discarded modes are resolvable by native fp64;
it limits the certificate to the selected support and is not by itself an N1 failure.

For a reference mean `mu*`, raw latent variance `v*`, candidate `(mu,v)`, learned outputscale `k**`,
and learned value-noise variance `sigma_f`, every target reports:

```text
absolute mean error                 |mu-mu*|
reference-scaled mean error         |mu-mu*| / max(1,|mu*|)
prior-scaled mean error             |mu-mu*| / sqrt(k**)
absolute variance error             |v-v*|
noise-scaled variance error         |v-v*| / max(sigma_f,|v*|)
prior-scaled variance error         |v-v*| / k**
```

All raw variances must remain finite and strictly positive without clipping.  The aggregate reports
family-wise maxima, not means or favorable quantiles.

For equal-rank q-support projectors `P*` and `P`, report max-absolute error,
`||P-P*||_2`, and `||P-P*||_F/sqrt(2r)`, plus symmetry and idempotence spectral errors for each
projector and each trace's absolute deviation from the declared rank.  A trace/rank or N0-rank
mismatch remains a failure even when a difference norm happens to be small.  Basis-vector equality is
never tested because signs and rotations within a support are non-identifiable.  The metric API must
receive both N0 strict-selected ranks and reject a mismatch before computing the rank-normalized
projector statistic; the runner retains that mismatch as `scientific_status=failed`.

Dense solves report both a relative residual and the normwise backward error using the fixed,
non-tunable floor of the declared residual-computation dtype:

```text
floor = finfo(residual_compute_dtype).tiny
relative residual = ||b-Ax||_2 / max(||b||_2, floor)
backward error = ||b-Ax||_2 / max(||A||_2 ||x||_2 + ||b||_2, floor).
```

Native recursive and fresh residuals retain their source dtype.  A separate CPU-float64 recomputation
from exactly promoted represented `A`, `b`, and `x` is canonical wherever dense materialization is
registered.  In a matrix-free case, let `r` be the runner's immediately recomputed residual and let
`U` be its analytic asserted upper bound on `||A||_2`.  The pure metric reports only the conditional
interval

```text
lower = ||r||_2 / max(U ||x||_2 + ||b||_2, floor)
upper = ||r||_2 / max(||b-r||_2 + ||b||_2, floor).
```

The first quantity is a lower bound on the exact normwise backward error, so it may never be used as
a small-error acceptance certificate.  The second is an upper bound conditional on the claim
`r=b-A(x)`.  The metric helper cannot prove either freshness or the asserted operator bound: the
future runner must compute both next to the operator application, bind their provenance, reject
`U||x||_2 < ||b-r||_2`, and use only the upper endpoint for a one-sided accuracy gate.  No observed
RHS magnitude may be used to choose a different floor.  This source-dtype interval is not a
directed-rounding certificate: the runner must also persist its actual `A(x)` action and the later
correctness bound must include an explicit arithmetic-roundoff margin.

A Cholesky diagnostic accepts only an exactly lower-triangular factor with strictly positive
diagonal and reports the spectral and Frobenius relative factorization residuals of `A-L L^T`.

ORBIT additionally records fresh rather than only recursive residuals and its residual-to-moment and
expected-KL diagnostics.  Those bounds apply only inside the selected support.

## Solver sweep

All probe targets run the matched fp32/fp64 production-candidate request `1e-5`.  Strata targets use
the following predeclared sweep:

```text
shared fp32/fp64 = 1e-3, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 1e-8
fp64 only        = 1e-9, 1e-10, 1e-11, 1e-12
max iterations   = min(4*m*r, 4096)
```

Requested tolerance is not convergence.  A comparison at one tolerance is eligible only when both
solvers record `converged=true` and fresh residual at or below that tolerance.  Production
tolerance also requires that tightening one registered level changes each moment by less than one
quarter of its proposed error budget.  Nonconvergence is scientific output, not a missing task.

## Released full-q diagnostic

Released TERA remains an operational diagnostic only at `m<=50`.  On the three `m=50` strata, the
probe captures actual function and q jitter and decomposes:

- native fp32 assembly plus fp32 solve;
- fp32-assembled matrix/RHS promoted and solved in fp64;
- native quantized-input fp64 assembly plus fp64 solve; and
- fp64 assembly cast to fp32 plus fp32 solve.

Each solution is checked against its own matrix and the canonical fp64 matrix.  The report records
`H`, q/cross/Schur assembly discrepancies, Cholesky backward error, solve backward error, and the
support/support-complement block, RHS, observation, and solution norms.  Full-q dense decomposition
is prohibited above `m=50`; larger probes use support projection and fixed random block probes.

The implemented adapter captures both native paths' actual adaptive jitters and requires exact
equality between independently recomposed intermediates and the matrices presented to the pinned
released Cholesky calls.  The two cross-precision solve arms operate on the complete already-jittered
represented systems: they neither reassemble nor add another jitter.  A cast system that loses
positive definiteness is retained as a factorization failure with no fabricated solution.  Large
q-system solve and factorization backward errors use explicitly labelled exact Frobenius-norm
residuals, avoiding unrelated spectral decompositions; every cross-arm tensor comparison and
support/complement norm remains CPU float64.

Matching TERA32's selected jitter defines a diagnostic arm, not the N1 reference.  Small residual
against a self-assembled fp32 matrix does not establish that its assembly is correct.

## Independent oracle and invariance suite

The existing float64 autograd/full-q tests remain necessary but are not the final higher-precision
reference.  Calibration adds small RBF fixtures whose fp32 inputs are converted as exact dyadic
numbers into at least 160-bit arithmetic.  Kernel blocks, projected observations, joint covariance,
solve, and posterior moments are recomputed independently and repeated at 256 bits; agreement must
stabilize before comparison with support64.  `numpy.longdouble` is not accepted as arbitrary
precision.

Real-data sentinel strata are also rerun through CPU float64 linear algebra.  Fixed neighbour
permutations must transform projectors as `Pi^T P Pi`; retained-support rotations must transform
coordinates, observations, noise, and the q-jitter metric together; and exact-zero augmentation must
leave rank and moments invariant within proposed bounds.

The implemented stress executor is restricted to the registered repeat-zero, seed-11 reference
tasks at `m=D-5` and selects only the worst geometry target from the complete ordered 100-target N0
scan.  It promotes the bound source-fp32 inputs exactly to CPU float64 while holding the original
absolute rank cutoff.  Four deterministic SHA-counter Rademacher probes diagnose the complete q
support/complement action without materializing the q system.  The q-space RHS and conditioned
observations are also assembled directly and differenced against the ORBIT support map.  The
remaining checks apply the frozen reverse-neighbour permutation, adjacent-pair pi/4 support rotation,
one exact zero ambient coordinate/gradient, and a native-fp64 cutoff comparison.  Nonconvergence and
added native modes remain recorded scientific output rather than triggering an unregistered repair.

## Threshold proposal and locked holdout

Replicas 0 and 1 are the discovery population.  Replica 2 is a locked development holdout: it may
be executed in the same array, but its numerical fields are not inspected until the discovery
report has serialized proposed thresholds and its own source/hash provenance.

For each metric `j`, discovery first constructs a correctness upper bound `U_j` from independent
high-precision error, projector perturbation, assembly/solve backward error, solver certificate, and
cross-allocation variation.  `U_j` is not the observed final moment maximum.  The proposed frozen
threshold is

```text
T_j = 4 * U_j.
```

A threshold can advance to review only if:

- every discovery observation is at most `U_j`;
- the locked replica-2 maximum is at most `T_j/2`; failure may not enlarge `T_j`;
- every registered material fault has error at least `10*T_j` on a metric intended to detect it;
- all identities, ranks, raw variances, convergence conditions, and condition envelopes pass; and
- the three sentinel allocations remain inside the same proposed envelope.

The final numerical gate is `max_i(error_ij/T_j) <= 1`.  No bootstrap, per-target confidence
interval, average, or multiple-testing label is used to disguise a deterministic implementation
gate.  If correct and faulty regimes do not have the registered separation, F02b remains blocked.

ORBIT32 may be proposed only if every N2, projector, convergence, variance, repeatability, and
condition-envelope requirement passes with unused margin.  Otherwise the conservative candidate is
ORBIT64 and all resource accounting must be recomputed for float64.  Mixed precision would be a
third implementation requiring its own calibration ID.

## Artifacts, provenance, and failure policy

Fit and probe stages use separate strict aggregators that enumerate the declared population rather
than globbing successful directories.  The fit catalog is accepted only with 45/45 valid tasks; a
probe must bind the exact fit artifact SHA-256.  The final aggregate is accepted only with 122/122
accounted tasks and no unexpected directory.

Each task binds the calibration ID and matrix hashes, task index and stable coordinates, corpus and
catalog identities/hashes, exact evaluation source rows, fixed-neighbour source rows, fit artifact
hash, clean commit/tree, TERA gitlink, dependency hashes, runtime packages, and shared CPU-only
Slurm allocation evidence.  Output paths are identity-derived and never overwritten.

The three hashes live in a non-recursive execution envelope outside the canonical task record being
hashed.  The scaffold CLI record alone is not this envelope and cannot be submitted directly.

Missing, duplicate, unexpected, malformed, nonfinite, clipped, OOM, timed-out, provenance-mismatched,
or exclusive/GPU-bearing/resource-mismatched tasks make `analysis_ready=false`.  Numerical gate
failures remain valid artifacts
with `scientific_status=failed`; they are not converted into structural absence.  Aggregation may
propose thresholds but must always emit `freeze_ready=false` until a separately reviewed F02b
protocol binds the report hashes and explicitly changes status.
