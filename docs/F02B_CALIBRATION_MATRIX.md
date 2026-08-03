# F02b numerical calibration matrix

Calibration ID: `F02B_NUMERICAL_CALIBRATION_v1`

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

Every other fit option is the exact `InternalTaskConfig` default used by the v3 diagnostic:
float32 released TERA training, batch size 256, learning rate 0.01, fixed graph, isotropic
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
3. the `(replica=0,D=12,seed=11,m=50)` probe is replayed with `repeat_id=1,2` in separate exclusive
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
fit matrix SHA-256     e53cabcb788e9383431b4a6b50bc6631499d9acf3f338ea659d654d76e24513e
probe matrix SHA-256   b729e755300fb997a18c07bf0cff185a1e60a7ed884355f95127cdd2f36aae7c
combined SHA-256       d81aee9b479adf437abd7f44782e4688227d3361458d686302c976bda5150114
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
silently removed after execution.  Each fit and probe task receives one exclusive node, one requested
L40S whose runtime reports at least `48,000,000,000` device-memory bytes (marketed 48 GB), 16 CPU
cores, 64 GiB host memory, and an eight-hour wall-time limit.  Arrays run at maximum concurrency one;
reproducibility replays use distinct allocations.  A task may use either the `short` or
`interactivegpu` scheduling partition because partition is not a scientific coordinate, but it must
record the partition and resolved GPU model.  OOM or timeout at the registered limit is a calibration
failure.  Removing `m=200` requires a new pre-run design version that also removes it from future
F02b selection.

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

Dense and iterative solves report both a relative residual and the normwise backward error using the
fixed, non-tunable floor of the declared residual-computation dtype:

```text
floor = finfo(residual_compute_dtype).tiny
relative residual = ||b-Ax||_2 / max(||b||_2, floor)
backward error = ||b-Ax||_2 / max(||A||_2 ||x||_2 + ||b||_2, floor).
```

Native recursive and fresh residuals retain their source dtype.  A separate CPU-float64 recomputation
from exactly promoted represented `A`, `b`, and `x` is canonical wherever dense materialization is
registered; matrix-free cases must record the predeclared operator-norm upper bound used in place of
`||A||_2`.  No observed RHS magnitude may be used to choose a different floor.

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
hash, clean commit/tree, TERA gitlink, dependency hashes, runtime packages, and exclusive Slurm
allocation evidence.  Output paths are identity-derived and never overwritten.

The three hashes live in a non-recursive execution envelope outside the canonical task record being
hashed.  The scaffold CLI record alone is not this envelope and cannot be submitted directly.

Missing, duplicate, unexpected, malformed, nonfinite, clipped, OOM, timed-out, provenance-mismatched,
or nonexclusive tasks make `analysis_ready=false`.  Numerical gate failures remain valid artifacts
with `scientific_status=failed`; they are not converted into structural absence.  Aggregation may
propose thresholds but must always emit `freeze_ready=false` until a separately reviewed F02b
protocol binds the report hashes and explicitly changes status.
