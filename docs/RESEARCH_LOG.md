# Research log

The single narrative doc. Every reported number must be backed by committed raw evidence. Retractions
are reported in place, loudly, naming the check that failed.

---

## F01 — SPARK varying-particle/varying-dimension comparison

### Preregistered protocol

SPARK is a physics-informed model, not a generic GP. It is given the particle count and spatial
dimension and assumes a separable quadratic kinetic term plus a shared, pair-additive radial
potential. The position/momentum block, kinetic coefficients, relative masses, and radial law are
learned from training inputs and gradients; true masses, force-law constants, and test labels are
not exposed to the model. This disclosed prior closely matches the simulator and must remain part of
any result claim.

F01 freezes the Cartesian product `n_particles in {2, 4, 10}` and `spatial_dims in {1, 2, 3}`.
Each of the nine configurations uses three independently generated fixed-mass systems. Complete
trajectories provide 1,500 training and 500 test rows per system, with no trajectory shared across
roles and no target-based filtering. Both arms use the same rows. SPARK uses rank 128, hybrid-log
inducing radii, lengthscale multiplier 1, and value/gradient noise `1e-3`. TERA uses its RBF kernel,
20 neighbors, 20 epochs, and float32 implementation.

The source-frozen summary reports standardized value RMSE, gradient RMSE, and raw Gaussian value
NLL as mean plus population standard deviation over three systems. SPARK passes only if all three
means are lower than TERA in every `(n_particles, spatial_dims)` configuration. Runtime and memory
are diagnostic because shared-node wall time is not a hardware-normalized cost measure.

### Results

Each metric cell is `SPARK / TERA` (mean ± population standard deviation). State dimension is
`D = 2nd`. NLL is the unmodified Gaussian value NLL on the standardized target scale.

| n | d | D | Value RMSE | Gradient RMSE | Raw NLL | Cell gate |
|---:|---:|---:|---:|---:|---:|:---:|
| 2 | 1 | 4 | 0.0102±0.013 / 1.41±1.2 | 0.159±0.15 / 105±91 | -4.84±0.22 / 1.21e4±1.5e4 | pass |
| 2 | 2 | 8 | 0.109±0.078 / 2.57±1.7 | 2.98±3.8 / 46.1±30 | 32.5±53 / 1.33e4±1.9e4 | pass |
| 2 | 3 | 12 | 0.0532±0.074 / 1.25±0.064 | 2.84±4.0 / 8.85±5.5 | 95.5±146 / 713±715 | pass |
| 4 | 1 | 8 | 0.0141±0.012 / 12.8±11 | 0.0806±0.074 / 77.7±23 | 100±137 / 4.56e4±2.33e4 | pass |
| 4 | 2 | 16 | 0.0199±0.027 / 7.38±3.1 | 0.347±0.30 / 48.6±9.2 | -4.42±1.4 / 2.16e3±1.56e3 | pass |
| 4 | 3 | 24 | 0.0104±0.0083 / 2.78±1.0 | 0.736±0.90 / 11.9±7.0 | 14.7±26 / 8.90±5.0 | **fail** |
| 10 | 1 | 20 | 0.00407±0.0033 / 13.6±5.1 | 0.0673±0.0080 / 106±40 | -0.752±5.7 / 835±708 | pass |
| 10 | 2 | 40 | 0.00278±0.0012 / 5.37±1.3 | 0.415±0.18 / 32.6±6.4 | -4.04±0.90 / 16.4±9.1 | pass |
| 10 | 3 | 60 | 0.00170±0.00072 / 1.11±0.18 | 0.159±0.025 / 6.28±1.2 | -4.53±0.86 / 1.59±0.10 | pass |

The preregistered overall gate is **not met**. SPARK has lower mean value and gradient RMSE in 9/9
cells and lower mean NLL in 8/9, but its `(n=4,d=3)` NLL is worse despite much lower RMSE. Several
other SPARK NLL cells also have large between-system spread. The honest finding is therefore strong
in-class point-prediction accuracy with an unresolved calibration failure, not a universal win.

All 27 generation and 54 benchmark tasks completed with exit code zero on shared eight-CPU Slurm
allocations. Across the 27 jobs per arm, median in-process CPU time was 0.850 s for SPARK and
1,425.685 s for TERA; median peak process RSS was 609.3 MiB and 3,401.7 MiB, respectively. These are
diagnostics only: shared-node contention produced a 79-minute TERA straggler, so wall time and these
resource observations are not a normalized cost benchmark.

The run is bound to source commit `a177883a6751372ce5fa8818b0442944f3cafa4e` and TERA commit
`b2382e10a045abca3d653ad58c4a2a9c1ca73458`. The 54 raw records, canonical summary, dataset hashes,
and Slurm ledgers are committed under `evidence/f01_spark_nd/`.
