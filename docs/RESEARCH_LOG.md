# Research log

The single narrative doc. Every reported number must be backed by committed raw evidence. Retractions
are reported in place, loudly, naming the check that failed.

---

## F01 — SPARK varying-particle/varying-dimension comparison

### Preregistration; results pending

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

The source-frozen summary will report standardized value RMSE, gradient RMSE, and raw Gaussian value
NLL as mean plus population standard deviation over three systems. SPARK passes only if all three
means are lower than TERA in every `(n_particles, spatial_dims)` configuration. Runtime and memory
are diagnostic because shared-node wall time is not a hardware-normalized cost measure. No F01 data
or result has been generated at this point.
