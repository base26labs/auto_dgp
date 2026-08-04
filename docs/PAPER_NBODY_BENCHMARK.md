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

## Comparison and reporting

Run the released TERA baseline and ORBIT on identical training rows, test rows, fitted kernel state,
and prediction targets. Method-specific approximation sizes remain explicit; they must not be
silently equated with DSoftKI's 512 interpolation points.

The primary table reports value RMSE and gradient RMSE for each particle count as mean and standard
deviation over the three seeds. If a method does not yet expose a valid full-gradient prediction,
mark that cell unsupported rather than substituting a different metric. Value NLL may be retained as
a secondary repository diagnostic, but it is not part of the paper's N-body primary table.

Record training seconds per epoch and prediction time for diagnostics. Because these runs use shared
CPU nodes while the paper timed a single GPU, wall-clock values must not be used for cross-paper
performance or cost claims. No custom acceptance thresholds, 122-task calibration grid, or protected
holdout release is needed; the reported three-seed scores are the benchmark result.

## Compute and tests

Each scheduled task uses one shared `short`-partition node, one task, exactly 8 CPUs, no GPU, and no
exclusive allocation. Set the common BLAS/OpenMP thread controls to 8. Use only a deterministic
data/split smoke test and a result-schema/metric smoke test; numerical kernel and ORBIT unit tests
remain separate from the benchmark.

Reference: <https://arxiv.org/abs/2505.09134> and the authors' released benchmark scripts at
<https://github.com/base26labs/dsoftki_gp>.
