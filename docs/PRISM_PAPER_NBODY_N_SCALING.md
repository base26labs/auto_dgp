# PRISM-GP N-scaling benchmark

Status: **frozen before N-scaling execution**.

The independent PRISM confirmation varied particle count at fixed corpus size, giving the
dimension sweep `D={24,36,48,60}` recorded in
`releases/paper_nbody_prism_seed43_b66f60e.json`. This secondary benchmark complements it by
holding the hardest reported state dimension fixed at `D=60` (`n_particles=10`) and varying the
number of training points.

## Grid and data

- Use the already generated seed-43, 9,500-row DSoftKI-compatible N-body corpus.
- Retain the paper's normalization-before-split rule and split seeds `6535`, `8830`, and `92357`.
- Keep the complete 950-row test split fixed within each seed.
- Use nested prefixes of the randomized training split at `N={1000,2000,4000,8550}`.
- Refit released TERA separately for every `(seed, N)` task, then evaluate TERA-20 and the frozen
  PRISM-GP-30/16 predictor from exactly the same learned parameters.

This corpus has already been evaluated by the D-scaling confirmation. The N sweep is therefore a
secondary scaling benchmark, not a new independent confirmation, and no method setting may be
changed in response to its labels.

## Metrics and reporting

For each N, report population mean and standard deviation across the three paper split seeds for
value RMSE, observation-variance value NLL, and gradient RMSE. Also report paired PRISM-minus-TERA
deltas. The existing Pareto rule—value-RMSE mean delta at most `1e-4`, strictly lower mean NLL and
gradient RMSE, and per-task solve/resource gates—is retained as a descriptive consistency check.
No statistical-significance claim is made.

`cluster/paper_nbody_prism_n_benchmark.sbatch` defines the exact 12-task grid. Every task uses one
shared `short` node allocation, one task, exactly 8 CPUs, no GPU, no exclusive allocation, and no
oversubscription. Shared-node wall clock is descriptive only and is never cost or performance
evidence. Aggregate with `python -m experiments.paper_nbody_prism_n_aggregate`.
