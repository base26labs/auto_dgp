# Paper-aligned N-body benchmark

Status: **active predictive benchmark protocol**.

This benchmark follows the toy N-body experiment in Appendix C.1 of Huang (2026),
*Scaling Gaussian Process Regression with Full Derivative Observations*, rather than the custom
F02b calibration matrix. The earlier F02/F02b work remains numerical-development history; it is not
required before running this benchmark and its pending arrays must not be submitted.

## Data and splits

- Report the four paper settings `n_particles = {4, 6, 8, 10}`, with state dimensions
  `D = {24, 36, 48, 60}`. The two-particle dataset is smoke-test only.
- Generate 10,000 samples from 100 DOP853 trajectories using `G=1`, Plummer softening `0.1`,
  positions from `N(0, 2^2)`, mass-scaled momenta from `N(0, 0.5^2)`, and masses uniform on
  `[0.5, 2.0]`.
- Remove the 5% of rows with the largest gradient norms, leaving 9,500 rows per setting.
- Use a random 90/10 train/test split with no validation split. Run the three seeds used by the
  authors' released benchmark scripts: `6535`, `8830`, and `92357`.
- Apply the paper normalization: inputs to the unit hypercube, centered energies and a shared
  energy/gradient scale, with the gradient chain-rule correction. Matching the authors' released
  implementation, normalization is computed before the random split. The exact transformation and
  split indices must be stored with each result.
- Pin generation to the released pair-loop numerical order in DSoftKI commit
  `286234baa0dd6be225bbfb1bdbb416687ea70654`, `data/get_nbody.py` blob
  `32f23c8c0f7263ef03026d4a3d34920ea3364cdc`. The generated NPZ embeds both identities and the
  benchmark loader rejects missing or mismatched provenance. This matters because a vectorized
  force reduction changes floating summation order and may diverge along a chaotic trajectory.

## Comparison and reporting

Run the released TERA baseline and ORBIT on identical training rows, test rows, fitted kernel state,
and prediction targets. Method-specific approximation sizes remain explicit; they must not be
silently equated with DSoftKI's 512 interpolation points.

The registered comparison is deliberately small:

- `TERA-20`: released TERA training and dense prediction at its native `m=20`;
- `ORBIT-20`: a same-neighbour control using the exact same learned state; and
- `ORBIT-G30`: a guarded expansion that computes both `m=20` and `m=30`.

The expansion is accepted only when all three label-free conditions hold:

1. the trust-radius condition `|mean_30 - mean_20| / sqrt(latent_variance_20) <= 0.02`;
2. posterior nesting, `latent_variance_30 <= latent_variance_20`, up to
   `128 * eps64 * max(|variance_20|, |variance_30|, 1)` roundoff; and
3. the standardized nested-posterior innovation
   `|mean_30 - mean_20| / sqrt(latent_variance_20 - latent_variance_30)` does not exceed the
   two-sided Bonferroni threshold with family-wise `alpha=0.01` over the 950 targets.

For fixed GP parameters and genuinely nested observations, the conditional mean innovation has
variance `latent_variance_20 - latent_variance_30`; the third condition is therefore a model-based
posterior-consistency check, not a label score. The branch is held piecewise constant when reporting
the selected scalar posterior's gradient. The `0.02` trust radius was fixed using the excluded
two-particle development dataset before reading any reported paper test set. A refit-per-seed
sensitivity check found that every threshold from `0.005` through `0.03` improved value RMSE, value
NLL, and gradient RMSE on all three excluded development splits. Adding the nesting/innovation guard
did not change any selected development target, but makes the formal rule fail closed if the expected
nested-posterior identity breaks. The unguarded `m=30` result remains a diagnostic, not an assessment
arm.

Following the paper's mean-and-standard-deviation reporting across three seeds, the candidate
succeeds only if `ORBIT-G30` has lower mean value RMSE, observation-variance value NLL, and gradient
RMSE than `TERA-20` for each particle count. Per-seed joint wins remain a stricter diagnostic but are
not an additional claim gate. All primal and adjoint solves must converge, and the candidate's
maximum per-target structured state and counted-operation proxies must remain within the
corresponding `TERA-20` full-value-gradient dense envelopes. The TERA envelope starts from its
`m^4` reduced covariance and `m^6/3` leading Cholesky terms, then applies the same conservative
factor-four reverse-pass allowance used for ORBIT's stored state and implicit pullback. This is a
simple deterministic decision rule, not a statistical-significance claim. Both `m` values and the
guard are fixed before reading any reported test set. Float64 ORBIT solves use a fixed `1e-10`
relative residual tolerance; all primal and adjoint iterations for both conditionals are charged to
the operation proxy. Peak structured state is the sequential maximum, while operation counts are
summed. These remain analytic safety proxies rather than measured hardware cost.

The primary table reports value RMSE and gradient RMSE for each particle count as mean and standard
deviation over the three seeds. Both gradients are defined as the derivative of the scalar posterior
mean with nearest-neighbour membership held piecewise constant. TERA uses its released-compatible
batched derivative path. ORBIT uses one matrix-free implicit adjoint solve, avoiding an autograd tape
through all CG iterations. Value NLL is additionally reported as a secondary repository diagnostic,
using predictive observation variance; it is part of the registered repository decision even though
it is not part of the paper's N-body primary table. A gradient NLL is not reported because neither
arm constructs the required full gradient predictive covariance.

Record training seconds per epoch and prediction time for diagnostics. Because these runs use shared
CPU nodes while the paper timed a single GPU, wall-clock values must not be used for cross-paper
performance or cost claims. No custom acceptance thresholds, 122-task calibration grid, or protected
holdout release is needed; the reported three-seed scores are the benchmark result.

## Compute and tests

Each scheduled task uses one nonexclusive `short`-partition allocation, one task, exactly 8 CPUs, no
GPU, and no exclusive allocation. The cluster's `select/cons_tres` configuration shares nodes by
allocated cores and memory; do not request CPU oversubscription. Set the common BLAS/OpenMP thread
controls to 8. Use only a deterministic
data/split smoke test and a result-schema/metric smoke test; numerical kernel and ORBIT unit tests
remain separate from the benchmark.

The implementation is:

- `cluster/paper_nbody_data.sbatch`: four shared-CPU dataset-generation tasks;
- `cluster/paper_nbody_benchmark.sbatch`: the exact 12-task particle/seed grid;
- `experiments/paper_nbody_benchmark.py`: one task and one immutable JSON result; and
- `experiments/paper_nbody_aggregate.py`: complete-grid aggregation and the fixed decision rule.

These files define runnable commands but do not authorize `sbatch`. The current checkout must be
committed and deployed with its environment before any array is submitted, and the ignored `runs/`
directory must already exist so Slurm can open its log paths.

The launch order, after separate authorization, is intentionally short: verify a clean deployed
commit and the repository virtual environment; create `runs/`; submit the four-task data array;
confirm all four arrays have 9,500 rows and the pinned generator metadata; submit the 12-task
benchmark array; then run `python -m experiments.paper_nbody_aggregate`. The benchmark loader and
aggregate both fail closed on provenance, guard, NLL-variance, convergence, and resource-accounting
drift. No step should use an exclusive allocation.

Reference: <https://arxiv.org/abs/2505.09134> and the authors' released benchmark scripts at
<https://github.com/base26labs/dsoftki_gp>.
