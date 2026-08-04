"""Finite-feature Bayesian shared-radial GP and full SPARK reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gp.spark.radial import (
    RadialNystromBasis,
    build_state_design,
    nystrom_value_residual_variance,
    pair_geometry,
    pair_indices,
    pair_weights_from_masses,
    pooled_radial_median_scale,
)
from gp.spark.residual import (
    HamiltonianResidualization,
    PotentialResidualTransform,
    position_residual_inputs,
    prepare_hamiltonian_residualization,
    reconstruct_position_prediction,
)
from gp.spark.structure import (
    DiagonalQuadraticMean,
    HamiltonianSplit,
    infer_relative_masses,
)


@dataclass(frozen=True)
class PreparedSpark:
    """Fit-role-only quantities shared by the radial candidate ladder."""

    residualization: HamiltonianResidualization
    relative_masses: torch.Tensor
    pairs: torch.Tensor
    pair_weights: torch.Tensor
    base_lengthscale: torch.Tensor
    radial_epsilon: float

    @property
    def split(self) -> HamiltonianSplit:
        return self.residualization.split

    @property
    def kinetic_mean(self) -> DiagonalQuadraticMean:
        return self.residualization.kinetic_mean

    @property
    def residual_transform(self) -> PotentialResidualTransform:
        return self.residualization.residual_transform

    @property
    def position_q(self) -> torch.Tensor:
        return self.residualization.position_q

    @property
    def potential_values(self) -> torch.Tensor:
        return self.residualization.potential_values

    @property
    def potential_gradients(self) -> torch.Tensor:
        return self.residualization.potential_gradients

    @property
    def position_offset(self) -> torch.Tensor:
        return self.residualization.position_offset


def prepare_spark(
    X: torch.Tensor,
    value: torch.Tensor,
    gradient: torch.Tensor,
    trajectory_id: torch.Tensor | None,
    *,
    n_particles: int,
    spatial_dims: int,
    coordinate_offset: torch.Tensor | None = None,
    frozen_split: HamiltonianSplit | None = None,
    radial_epsilon: float | None = None,
    stage: str = "selection",
    source_row_ids: torch.Tensor | None = None,
) -> PreparedSpark:
    """Discover/refit the split and form normalized potential training observations.

    ``coordinate_offset`` is the dimensionless ``x_offset / x_scale`` from the outer
    fit-only scaling. It is added back before pair distances are computed.
    """

    residualization = prepare_hamiltonian_residualization(
        X,
        value,
        gradient,
        trajectory_id,
        n_particles=n_particles,
        spatial_dims=spatial_dims,
        coordinate_offset=coordinate_offset,
        frozen_split=frozen_split,
        stage=stage,
        source_row_ids=source_row_ids,
    )
    return prepare_spark_from_residualization(
        residualization,
        radial_epsilon=radial_epsilon,
    )


def prepare_spark_from_residualization(
    residualization: HamiltonianResidualization,
    *,
    radial_epsilon: float | None = None,
) -> PreparedSpark:
    """Attach SPARK's pair schema to one already-frozen shared residualization."""

    n_particles = residualization.split.n_particles
    spatial_dims = residualization.split.spatial_dims
    relative_masses = infer_relative_masses(
        residualization.kinetic_mean,
        n_particles=n_particles,
        spatial_dims=spatial_dims,
    )
    pairs = pair_indices(n_particles, device=residualization.position_q.device)
    weights = pair_weights_from_masses(relative_masses, pairs=pairs)

    if radial_epsilon is None:
        radial_epsilon = float(torch.finfo(residualization.position_q.dtype).eps ** 0.5 * 1e-2)
    if radial_epsilon <= 0:
        raise ValueError("radial_epsilon must be positive")
    base_lengthscale = pooled_radial_median_scale(
        residualization.position_q,
        n_particles=n_particles,
        spatial_dims=spatial_dims,
        radial_epsilon=radial_epsilon,
    )
    return PreparedSpark(
        residualization=residualization,
        relative_masses=relative_masses,
        pairs=pairs,
        pair_weights=weights,
        base_lengthscale=base_lengthscale,
        radial_epsilon=radial_epsilon,
    )


@dataclass(frozen=True)
class BayesianRadialFeatureGP:
    """Bayesian linear posterior over a shared whitened radial feature basis."""

    basis: RadialNystromBasis
    pair_weights: torch.Tensor
    pairs: torch.Tensor
    n_particles: int
    spatial_dims: int
    posterior_mean: torch.Tensor
    precision_cholesky: torch.Tensor
    value_noise: float
    gradient_noise: float
    radial_epsilon: float

    @classmethod
    def fit(
        cls,
        q: torch.Tensor,
        value: torch.Tensor,
        gradient: torch.Tensor,
        *,
        basis: RadialNystromBasis,
        pair_weights: torch.Tensor,
        pairs: torch.Tensor,
        n_particles: int,
        spatial_dims: int,
        value_noise: float = 1e-3,
        gradient_noise: float = 1e-3,
        radial_epsilon: float | None = None,
        point_chunk: int = 256,
    ) -> BayesianRadialFeatureGP:
        """Accumulate and solve the finite-feature Bayesian normal equations."""

        expected_gradient_shape = (q.shape[0], n_particles * spatial_dims)
        if value.shape != (q.shape[0],) or gradient.shape != expected_gradient_shape:
            raise ValueError("value or gradient has an invalid shape")
        if q.device != basis.inducing_radii.device or q.dtype != basis.inducing_radii.dtype:
            raise ValueError("training tensors and basis must have the same device and dtype")
        if value_noise <= 0 or gradient_noise <= 0:
            raise ValueError("observation noises must be positive")
        if point_chunk <= 0:
            raise ValueError("point_chunk must be positive")
        if radial_epsilon is None:
            radial_epsilon = float(torch.finfo(q.dtype).eps ** 0.5 * 1e-2)

        precision = torch.eye(basis.rank, device=q.device, dtype=q.dtype)
        right_hand_side = q.new_zeros(basis.rank)
        for start in range(0, q.shape[0], point_chunk):
            stop = min(start + point_chunk, q.shape[0])
            value_design, gradient_design = build_state_design(
                q[start:stop],
                basis,
                pair_weights,
                n_particles=n_particles,
                spatial_dims=spatial_dims,
                pairs=pairs,
                radial_epsilon=radial_epsilon,
            )
            flat_gradient_design = gradient_design.reshape(-1, basis.rank)
            flat_gradient_target = gradient[start:stop].reshape(-1)
            precision = precision + value_design.T @ value_design / value_noise
            precision = precision + flat_gradient_design.T @ flat_gradient_design / gradient_noise
            right_hand_side = (
                right_hand_side
                + value_design.T @ value[start:stop] / value_noise
                + flat_gradient_design.T @ flat_gradient_target / gradient_noise
            )

        precision = 0.5 * (precision + precision.T)
        precision_cholesky, info = torch.linalg.cholesky_ex(precision)
        if int(info) != 0:
            raise RuntimeError("Bayesian feature precision is not positive definite")
        posterior_mean = torch.cholesky_solve(
            right_hand_side[:, None],
            precision_cholesky,
        )[:, 0]
        if not bool(torch.isfinite(posterior_mean).all()):
            raise RuntimeError("Bayesian feature solve produced non-finite coefficients")
        return cls(
            basis=basis,
            pair_weights=pair_weights,
            pairs=pairs,
            n_particles=n_particles,
            spatial_dims=spatial_dims,
            posterior_mean=posterior_mean,
            precision_cholesky=precision_cholesky,
            value_noise=value_noise,
            gradient_noise=gradient_noise,
            radial_epsilon=radial_epsilon,
        )

    def predict(
        self,
        q: torch.Tensor,
        *,
        point_chunk: int = 256,
        include_nystrom_residual: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return latent value mean, Cartesian gradient mean, and value variance.

        The default variance adds an exact-prior diagonal correction to the Bayesian
        finite-feature posterior. Disable it only for controlled feature-model audits.
        """

        if q.device != self.posterior_mean.device or q.dtype != self.posterior_mean.dtype:
            raise ValueError("q and fitted model must have the same device and dtype")
        if point_chunk <= 0:
            raise ValueError("point_chunk must be positive")
        values = []
        gradients = []
        variances = []
        for start in range(0, q.shape[0], point_chunk):
            stop = min(start + point_chunk, q.shape[0])
            value_design, gradient_design = build_state_design(
                q[start:stop],
                self.basis,
                self.pair_weights,
                n_particles=self.n_particles,
                spatial_dims=self.spatial_dims,
                pairs=self.pairs,
                radial_epsilon=self.radial_epsilon,
            )
            values.append(value_design @ self.posterior_mean)
            gradients.append(torch.einsum("ndm,m->nd", gradient_design, self.posterior_mean))
            whitened = torch.linalg.solve_triangular(
                self.precision_cholesky,
                value_design.T,
                upper=False,
            )
            variance = whitened.square().sum(dim=0)
            if include_nystrom_residual:
                geometry = pair_geometry(
                    q[start:stop],
                    n_particles=self.n_particles,
                    spatial_dims=self.spatial_dims,
                    pairs=self.pairs,
                    radial_epsilon=self.radial_epsilon,
                )
                variance = variance + nystrom_value_residual_variance(
                    geometry.radii,
                    self.basis,
                    self.pair_weights,
                    value_design=value_design,
                )
            variances.append(variance)
        return torch.cat(values), torch.cat(gradients), torch.cat(variances)


@dataclass(frozen=True)
class SparkModel:
    """Full standardized energy and gradient predictor."""

    residualization: HamiltonianResidualization
    radial_gp: BayesianRadialFeatureGP

    @property
    def split(self) -> HamiltonianSplit:
        return self.residualization.split

    def predict(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return value mean, full-state gradient mean, and latent value variance."""

        q = position_residual_inputs(self.residualization, X)
        potential_value, potential_gradient, potential_variance = self.radial_gp.predict(q)
        reconstructed = reconstruct_position_prediction(
            self.residualization,
            X,
            potential_value,
            potential_gradient,
            potential_variance,
        )
        return reconstructed.value, reconstructed.gradient, reconstructed.variance


def fit_spark(
    prepared: PreparedSpark,
    *,
    lengthscale_multiplier: float = 0.5,
    rank: int = 128,
    inducing_strategy: str = "quantile",
    value_noise: float = 1e-3,
    gradient_noise: float = 1e-3,
    point_chunk: int = 256,
) -> SparkModel:
    """Fit one candidate from a prepared fit-role-only split-pair problem."""

    if lengthscale_multiplier <= 0:
        raise ValueError("lengthscale_multiplier must be positive")
    geometry = pair_geometry(
        prepared.position_q,
        n_particles=prepared.split.n_particles,
        spatial_dims=prepared.split.spatial_dims,
        pairs=prepared.pairs,
        radial_epsilon=prepared.radial_epsilon,
    )
    basis = RadialNystromBasis.from_radii(
        geometry.radii,
        lengthscale=prepared.base_lengthscale * lengthscale_multiplier,
        rank=rank,
        selection_strategy=inducing_strategy,
    )
    radial_gp = BayesianRadialFeatureGP.fit(
        prepared.position_q,
        prepared.potential_values,
        prepared.potential_gradients,
        basis=basis,
        pair_weights=prepared.pair_weights,
        pairs=prepared.pairs,
        n_particles=prepared.split.n_particles,
        spatial_dims=prepared.split.spatial_dims,
        value_noise=value_noise,
        gradient_noise=gradient_noise,
        radial_epsilon=prepared.radial_epsilon,
        point_chunk=point_chunk,
    )
    return SparkModel(
        residualization=prepared.residualization,
        radial_gp=radial_gp,
    )
