# F02b N-body protocol

> **Superseded for predictive benchmarking (2026-08-04).** Use the
> [paper-aligned N-body benchmark](PAPER_NBODY_BENCHMARK.md). The custom F02b gate design below is
> historical and does not authorize job submission.

Status: **DRAFT — not preregistered, not executable, and not permission to access any
confirmatory label.**

F02b is a new, versioned experiment proposal. It is not an in-place relaxation of
`F02_NBODY_PROTOCOL.md`, and this draft does not alter that frozen document. Numerical tolerances,
singular-gap acceptance criteria, invariance tolerances, the production dtype, and predictive claim
gates are deliberately unset here. They must be calibrated on development-only evidence and frozen
in a reviewed F02b preregistration before any confirmatory label can be read. The operational rank
cutoff itself is not `TBD`: this draft registers the current oracle's source-fp32 rule below. That
numerical rule does not certify that its discarded directions are strict algebraic zeros.

## Why the original F02 stopped

The original F02 preregistration required ORBIT-50 to agree with released TERA-50 at the same
neighbourhood in float32. That control failed on the first development pilot. The correct protocol
action is termination, not deletion of the control or post-hoc widening of its tolerance.

Consequently:

- original F02 ended at its pilot and must not advance to the 135-task optimizer-selection grid;
- no F02 optimizer-update budget or dimension-to-neighbour schedule was selected;
- no original-F02 recipe or release ledger may authorize a confirmatory run;
- results from a future experiment must be labelled F02b and bound to a new protocol identifier,
  source release, recipe, and one-release ledger; and
- the historical F02 failure remains reported even if F02b later succeeds.

### Recorded evidence

**Slurm job 2810520 — preregistered F02 pilot failure.** The task was development replica `0`,
`n_particles=2`, `D=12`, 20 optimizer updates, seed `11`, prediction `m=50`, on the registered
one-time validation design. It ran from source commit
`e27a0cd431d5a9d5258013a097b3c99cdb3907c8`, TERA gitlink
`b2382e10a045abca3d653ad58c4a2a9c1ca73458`, on an exclusively verified L40S allocation. The
same-m control observed

- maximum absolute mean difference `0.18055188655853271`; and
- maximum absolute latent-variance difference `0.0009048040956258774`.

Both exceeded the original F02 historical tolerance `0.0001`. The task exited `1`; the strict pilot
aggregate recorded one failed task, `analysis_ready=false`, and `selected_update=null`. The pilot
aggregate SHA-256 is `fd4634f8ca84aa7e0235142118f617678dd498271e38f60b3731e37cc68b9392`.

**Slurm job 2810525 — development-only diagnosis.** This task completed from commit
`d81e873ee293d60192977c6f62e8ae9d8030cd00`. Its diagnostic JSON SHA-256 is
`50c0a5d0b9e9ec7610c6f0718a951c956bf7288ae885ae963348490a50765d95`. On the same `D=12`,
`m=50`, seed-11 task:

- the default ORBIT32 solves converged for all 20 targets, retained rank `6`, and reported its basis
  complete under the then-used native-float32 numerical cutoff; this was a numerical designation,
  not a claim of algebraic exactness. Its maximum freshly recomputed relative residual was
  `9.509325536782853e-06`, yet the TERA32/ORBIT32 differences reproduced job 2810520;
- tightening the requested ORBIT32 CG tolerance to `1e-8` reached the 300-iteration cap without
  convergence, while the discrepancies remained essentially unchanged: maximum absolute mean
  `0.1807389259338379` and latent variance `0.0009041596204042435`;
- in float64 prediction using the same learned parameters, released TERA64 and ORBIT64 agreed to
  maximum absolute mean `7.148650560395708e-10` and latent variance
  `8.119893646352239e-14`; and
- released TERA32 itself differed from released TERA64 by maximum absolute mean
  `0.18065762519836426` and latent variance `0.0009076874703168869`.

All 20 float32 local spectra retained rank `6`, matching `min(m,D-6)` for this task. The evidence
rules out ordinary ORBIT CG termination error as an explanation and demonstrates material dtype
dependence in the released TERA path on the rank-constrained system. It does **not** uniquely
apportion every error or prove that ORBIT32 is accurate; that missing inference motivates the
independent oracle and cross-dtype gates below.

## F02b question and claim boundary

F02b asks two ordered questions:

1. Can ORBIT reproduce an independently implemented supported-subspace conditional, and can the
   chosen production dtype reproduce the float64 ORBIT result?
2. After those numerical gates pass, does spending ORBIT's saved local-prediction resource on a
   larger neighbourhood improve prediction relative to ORBIT-50 in the **same dtype**, with all
   learned parameters held fixed?

Question 1 is a prerequisite for Question 2. A failed numerical gate terminates F02b before any
larger-neighbour predictive claim. F02b must not call released TERA32 a mathematical oracle, and it
must not attribute every difference between released TERA32 and ORBIT-resource to neighbourhood
size.

This draft does not set predictive superiority thresholds. A final F02b preregistration must either
restate and justify them or explicitly replace them using development-only calibration, before
confirmatory release.

## Registered numerical roles

### `TERA-released-fp32`: operational baseline

`TERA-released-fp32` means the released TERA prediction implementation run in float32, including its
released q-coordinate construction, dense factorization path, and adaptive jitter behaviour. It is
the operational baseline for the question “what does the released implementation produce under its
documented execution recipe?”

It is **not** the same-m correctness oracle in F02b. Its mean, raw latent variance, selected jitter,
factorization status, and float32-to-float64 discrepancy must still be recorded. Comparisons against
it are released-code comparisons, not proofs that either implementation equals the ideal supported
conditional.

### `support64`: independent supported-subspace oracle

`support64` is a dense float64 reference implementation for one fixed target, neighbour set,
kernel/noise parameters, and regularizer semantics. It must:

- round every numeric input to its released float32 representation and only then promote it to
  float64;
- form the scaled difference matrix from those quantized inputs, compute its SVD in float64, and
  retain exactly the singular directions satisfying

  ```text
  singular_value > smax * max(D, m) * eps(float32)
  ```

  (`source-fp32-smax-maxshape-eps-v1`, with multiplier `1.0`);
- call the excluded complement **rule-discarded numerical modes**, not mathematically null or
  strictly algebraically zero directions;
- form and solve the supported conditional covariance densely in float64;
- report the numerical supported rank, cutoff, singular/eigen spectra, discarded singular-value
  energy, solve residual, mean, and unclipped latent variance;
- preserve the released q-coordinate regularizer after its exact transformation to the support;
  changing the regularizer would define a different conditional; and
- use an implementation and dense linear-algebra path independent of ORBIT's structured operator,
  PCG solver, preconditioner, and prediction routine.

Thus, `support64` names the precision of the dense construction and solve. It does not mean that the
source geometry bypasses float32 quantization, that the retained rank is an exact algebraic rank, or
that every direction below the operational cutoff is physically absent. The physical expected-rank
audit is a separate gate below; `support64` must fail that gate rather than silently relabel a
numerically small singular direction as an exact zero.

The oracle may share the written mathematical specification, test fixtures, and frozen inputs with
ORBIT, but it must not call ORBIT internals to obtain its geometry, operator products, solution, or
posterior moments. Small analytic and higher-precision fixtures must audit `support64` itself before
it can serve as an oracle.

### `ORBIT64` and `ORBIT32`

`ORBIT64` is the structured ORBIT predictor evaluated in float64 on the same frozen inputs,
neighbours, parameters, source-fp32 quantization, fixed rank rule, noise semantics, and transformed
regularizer as `support64`. It tests the structured realization separately from production-dtype
roundoff.

`ORBIT32` is the corresponding float32 path. F02b does not assume in advance that it is acceptable.
The calibration and gates below determine whether float32 may be registered as the production dtype
or whether F02b must use float64 throughout its ORBIT comparisons.

## Numerical gates before predictive analysis

The final protocol must freeze comparison metrics, aggregation rules, and calibrated thresholds.
This draft specifies the direction and scope of the gates but intentionally leaves every new
tolerance as `TBD`. The source-fp32 numerical-rank rule above is already fixed and is not one of
those tolerances.

### Gate N0: physical expected rank and numerical boundary agree

The generated fixed-mass N-body states satisfy three mass-centred position constraints and three
total-momentum constraints. For every registered nondegenerate target neighbourhood, the independent
physical expectation is

```text
r_phys = min(m, D - 6).
```

For each target, N0 must independently compare `r_phys` with the numerical rank `r_num` selected by
the fixed source-fp32 rule. The audit must record the complete ordered spectrum, `smax`, the fixed
cutoff, `r_num`, `r_phys`, the singular values immediately on both sides of the expected boundary
(when those indices exist), and scale-free measures of their separation from the cutoff and from
each other. The final preregistration must freeze a development-calibrated minimum acceptable
singular-value gap or equivalent boundary-separation criterion; that acceptance criterion is `TBD`
in this draft, but the cutoff formula is not.

N0 passes only when all three statements agree: `r_num == r_phys`, the fixed cutoff places the
expected retained and discarded boundary singular values on their respective sides, and the frozen
gap criterion certifies that the boundary is stable. Any inconsistency among the cutoff placement,
the singular-value gap, and the physical expected rank is a gate failure, even if posterior moments
from two implementations happen to agree. The implementation must not force `r_num` to `r_phys`,
change the cutoff after seeing the target, or describe rule-discarded numerical modes as strict
algebraic zeros. A degenerate neighbourhood that violates the registered physical expectation is
also a failure for the registered corpus unless a degeneracy policy was explicitly calibrated and
frozen before confirmatory access.

### Gate N1: `support64` approximately equals `ORBIT64`

After N0 passes, for every registered development diagnostic target and every prediction `m`
relevant to F02b, compare `support64` with `ORBIT64` using at least:

- maximum per-target absolute and scale-normalized mean error;
- maximum per-target absolute and scale-normalized raw latent-variance error;
- exact equality of the rank selected by the fixed source-fp32 rule and agreement of the resulting
  registered support projector;
- independent dense and freshly recomputed structured residuals; and
- positivity and finiteness of every unclipped raw latent variance.

The threshold values and the definition of “scale-normalized” are `TBD after calibration`. Failure
of N1 is an implementation or specification failure. It cannot be repaired by comparing predictive
scores, changing `m`, or falling back to released TERA32.

### Gate N2: `ORBIT32` approximately equals `ORBIT64`

Using identical neighbours, learned parameters, source-fp32 inputs, fixed rank rule, stopping
semantics, and target order, compare ORBIT32 with ORBIT64 on the same mean, raw latent-variance,
residual, rank, and positivity fields. Rank equality is exact; numerical thresholds for the other
fields are `TBD after calibration`.

If N2 fails, float32 is not an admissible production dtype for F02b. The protocol may be revised and
re-frozen to use ORBIT64, or the float32 implementation may be repaired and recalibrated, but the
gate may not be weakened after confirmatory information is available.

`TERA-released-fp32` versus `support64`, ORBIT32, and ORBIT64 remains a mandatory diagnostic table,
not an N1/N2 pass condition. This makes the released numerical discrepancy visible without treating
it as the reference truth.

## Development geometry diagnostics

Before F02b is frozen, development-only diagnostics must cover all development dimensions and all
candidate prediction neighbourhood sizes. They must record, without silently deleting failures:

- all N0 evidence, including the native-float32 and source-fp32-then-float64 singular spectra, the
  physical expected rank, fixed cutoff, observed numerical rank, and boundary gap. Any spectrum
  computed from unquantized float64 geometry is a separately labelled sensitivity diagnostic and
  must not define the operational support;
- residuals of the six physical affine constraints and rank stability under neighbour permutation;
- the numerical q-support projector `P` and rule-discarded-complement projector `I-P`; norms of
  observations, conditional cross-covariances, right-hand sides, and solutions in both components;
  support/complement covariance blocks; and any amplification introduced by adaptive q-coordinate
  jitter;
- the posterior change when modes discarded by the registered numerical rule are removed. A
  material dependence on those modes is a failure, not evidence that they were strict algebraic
  zeros or an invitation to adopt an alternative posterior definition;
- **support-rotation invariance:** applying registered orthogonal rotations within the retained
  support, with all coordinates and observations transformed consistently, must leave the posterior
  moments invariant up to calibrated numerical error; and
- **exact-zero augmentation invariance:** padding the representation with controlled redundant or
  analytically exact-zero q directions must not change the numerical supported rank or posterior
  moments up to calibrated numerical error. This fixture does not reclassify ordinary
  rule-discarded modes as exact zeros.

Rotation and augmentation here are representation-level tests, not new physical data augmentation
or additional training observations. Their tolerances are also `TBD after calibration`.

## Larger-neighbour mechanism

Once N1 and N2 pass and one ORBIT production dtype has been frozen, the larger-neighbour mechanism
is evaluated as

```text
ORBIT-resource versus ORBIT-50, in the same registered dtype.
```

Both arms must use the same fitted TERA parameters, kernel/noise semantics, support/rank policy,
solver policy, target rows, and metric implementation. The only intended scientific difference is
the prediction neighbourhood size selected under the frozen analytic resource rule. This comparison
isolates the statistical effect of buying more neighbours with ORBIT's saved local-prediction
resource.

`TERA-released-fp32` remains an operational released-code baseline and must be reported. However, an
ORBIT-resource versus TERA-released-fp32 difference combines a neighbourhood change with the known
released-fp32 numerical-path difference. It cannot, by itself, establish the larger-`m` mechanism.

Optimizer-update selection, the ORBIT resource envelope, external baselines, uncertainty, and final
predictive claim gates must be explicitly re-registered for F02b. They are not inherited merely by
linking to the terminated F02 document.

## Calibration required before thresholds can be preregistered

No new numerical-agreement, singular-gap-acceptance, or invariance threshold is chosen in this
draft. The source-fp32 rank cutoff formula remains fixed as specified above. A development-only
calibration report must be completed, reviewed, and hash-bound before changing this document from
`DRAFT`. At minimum it must:

1. Declare a calibration task matrix before running it, spanning development replicas, all five
   state dimensions, representative well- and ill-conditioned targets, optimizer seeds, and every
   prediction `m` used by numerical or resource selection.
2. Audit `support64` on analytic synthetic cases and small cases evaluated with a genuinely
   higher-accuracy dense reference. The reference construction and precision must be independent of
   both production implementations.
3. Measure absolute and scale-normalized errors separately for means and raw latent variances,
   including near-zero variances. Any denominator or scale floor must be fixed from calibration
   rules, not by clipping failed confirmatory outputs.
4. Exercise N0 around the known `min(m,D-6)` boundary, including cutoff placement, singular-value
   gaps, rule-discarded-complement leakage, neighbourhood permutations, support rotations, and
   exact-zero augmentations. It must propose the gap criterion without changing the fixed rank rule.
5. Separate operator/discretization error, iterative-solve error, and dtype roundoff by including
   dense support64 solves, ORBIT64 residual sweeps, and ORBIT32 residual sweeps.
6. Check reproducibility across repeated exclusive L40S runs and a registered independent float64
   dense environment. Runtime is descriptive; numerical agreement is the calibration target.
7. Predeclare whether each gate uses a maximum, simultaneous bound, or another family-wise rule;
   per-target failures may not be hidden by an average.
8. Produce proposed N1, N2, singular-gap, rotation, augmentation, residual, and variance-validity
   thresholds with a numerical justification and safety margin. Thresholds must distinguish
   expected roundoff from the order-`1e-1` mean discrepancy observed in job 2810525, not merely make
   existing jobs pass. No proposed threshold may override an N0 expected-rank mismatch.
9. Freeze the calibrated thresholds, metric formulas, calibration artifact hashes, and failure policy
   before optimizer-budget or neighbour-schedule selection is rerun under F02b.

If calibration does not exhibit a stable separation between correct numerical variation and
implementation failures, F02b remains blocked rather than adopting an arbitrary tolerance.

## Confirmatory-label lock and future freeze

All labels in confirmatory replicas `101..110` remain inaccessible while this protocol is a draft.
“Labels” includes Hamiltonian values, gradients, train/validation/test target arrays, target-derived
normalization quantities, and any derived score. No confirmatory corpus may be fitted, tuned,
predicted, or scored during calibration. Only already-frozen catalog-level identities, paths, and
cryptographic hashes may be audited; bundle metadata sidecars and NPZ payloads remain unopened.

Before a first confirmatory label access, one reviewed F02b freeze must bind at least:

- the final F02b protocol and distinct experiment/protocol identifiers;
- the fixed source-fp32 rank rule, the N0 expected-rank and singular-gap gate, calibrated numerical
  thresholds, and their immutable development-only calibration report;
- the independent `support64` implementation and its oracle-validation tests;
- the registered ORBIT production dtype and passed N1/N2/invariance evidence;
- exact optimizer-selection and resource-selection rules and completed development selections;
- the final per-dimension `m` schedule, external-baseline configurations, seeds, and complete task
  matrix;
- source commit/tree, dependency locks, TERA gitlink, catalog and bundle hashes, and analysis-source
  hashes; and
- a new global recipe plus protected one-release ledger that explicitly names F02b.

The confirmatory runner must remain hard-disabled until that freeze is committed and authorized.
Neither job 2810520, job 2810525, this draft, nor any original-F02 recipe constitutes release
permission.
