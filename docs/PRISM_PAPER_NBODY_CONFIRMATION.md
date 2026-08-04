# PRISM-GP paper N-body confirmation

Status: **frozen before independent corpus generation**.

## Framework

PRISM-GP-30/16 means *precision-ranked iterative structured marginals*. It shares the released
TERA fit and its native `m=20` local conditional, then considers an `m=30` conditional using the
numerical rank implied by the source float32 arrays. The expansion is eligible only when that rank
is at most 16, and is selected only when its mean shift is within `0.025` base posterior standard
deviations and its latent variance does not increase beyond float64 roundoff. Only the selected
branch is differentiated, using one implicit adjoint while reusing its primal solve.

This is motivated by DSoftKI's control of directional complexity: PRISM retains local spectral
directions supported by the precision of the observed arrays instead of treating float32
quantization modes as new float64 signal. Prediction uses float64, CG relative tolerance `1e-10`,
and at most 4096 iterations.

## Development evidence and frozen rule

The seed-42 paper corpus is development-only. Across the paper split seeds `6535`, `8830`, and
`92357`, its paired PRISM-minus-TERA mean deltas were:

| particles | value RMSE | value NLL | gradient RMSE |
|---:|---:|---:|---:|
| 4 | +8.81e-7 | -4.086e-3 | -7.452e-2 |
| 6 | -2.06e-8 | -2.671e-3 | -2.722e-2 |
| 8 | -2.58e-8 | -1.381e-3 | -7.703e-3 |
| 10 | +2.56e-7 | -5.666e-5 | -9.925e-4 |

All 12 development tasks passed the deterministic TERA-20 state and counted-operation envelopes
and the solve-residual gate. These results select the framework but cannot confirm it.

The independent decision is made separately for each particle count from the population mean over
the same three paper split seeds:

1. mean value RMSE delta is at most `1e-4` (noninferiority);
2. mean observation-variance value NLL delta is strictly below zero;
3. mean gradient RMSE delta is strictly below zero;
4. every solve converges at the registered tolerance; and
5. state and maximum counted-operation proxies remain within TERA-20 on every task.

The RMSE margin is one tenth of the paper table's `0.001` display unit and was frozen before the
independent corpus existed. This is a deterministic Pareto claim, not a significance claim. The
paper's primary N-body table reports value and gradient RMSE; NLL is included as the requested
repository diagnostic using predictive observation variance.

## Independent benchmark

Generate a separate corpus with generator seed `43`, retaining the paper's pinned pair-loop DOP853
generator, 10,000 raw rows, 95th-percentile gradient filter, 9,500 retained rows, preprocessing,
90/10 split, and split seeds `6535`, `8830`, and `92357`. The seed-42 development files must not be
read by the confirmation runner. Each task refits released TERA and evaluates TERA-20 and the frozen
PRISM candidate from the same learned parameters.

`cluster/paper_nbody_prism_confirm_data.sbatch` is the four-corpus generator and
`cluster/paper_nbody_prism_confirm.sbatch` is the 12-task benchmark. Both require a shared `short`
node, one task, exactly 8 CPUs, no GPU, no exclusive allocation, and no oversubscription. Shared-node
wall clock is descriptive only and is never cost or performance evidence. Aggregate with
`python -m experiments.paper_nbody_prism_aggregate`.
