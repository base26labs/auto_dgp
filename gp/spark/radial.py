"""Whitened one-dimensional Matérn features pulled back through pair distances."""

from __future__ import annotations

from dataclasses import dataclass

import torch

_SQRT5 = 5.0**0.5


def pair_indices(
    n_particles: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return lexicographically ordered unordered particle pairs, shape ``(E, 2)``."""

    if n_particles < 2:
        raise ValueError("at least two particles are required")
    particles = torch.arange(n_particles, device=device)
    return torch.combinations(particles, r=2)


@dataclass(frozen=True)
class PairGeometry:
    """Pair radii and their derivatives with respect to the first endpoint."""

    pairs: torch.Tensor
    radii: torch.Tensor
    unit_vectors: torch.Tensor
    radial_epsilon: float


def pair_geometry(
    q: torch.Tensor,
    *,
    n_particles: int,
    spatial_dims: int,
    pairs: torch.Tensor | None = None,
    radial_epsilon: float | None = None,
) -> PairGeometry:
    """Return ``r_ab`` and ``d r_ab / d q_a`` for each state and pair.

    The numerical radius is ``sqrt(||q_a-q_b||^2 + epsilon^2)``. Consequently the
    returned unit vectors are its exact derivatives even for coincident particles.
    """

    if q.ndim != 2 or not q.is_floating_point():
        raise ValueError("q must be a two-dimensional floating-point tensor")
    if q.shape[1] != n_particles * spatial_dims:
        raise ValueError("q shape is inconsistent with the particle schema")
    if not bool(torch.isfinite(q).all()):
        raise ValueError("q must be finite")
    if radial_epsilon is None:
        radial_epsilon = float(torch.finfo(q.dtype).eps ** 0.5 * 1e-2)
    if radial_epsilon <= 0:
        raise ValueError("radial_epsilon must be positive")

    pairs = pair_indices(n_particles, device=q.device) if pairs is None else pairs
    pairs = torch.as_tensor(pairs, device=q.device, dtype=torch.long)
    expected_pairs = n_particles * (n_particles - 1) // 2
    if pairs.shape != (expected_pairs, 2):
        raise ValueError(f"pairs must have shape ({expected_pairs}, 2)")
    if bool((pairs < 0).any()) or bool((pairs >= n_particles).any()):
        raise ValueError("pairs contain an invalid particle index")

    positions = q.reshape(q.shape[0], n_particles, spatial_dims)
    displacement = positions[:, pairs[:, 0]] - positions[:, pairs[:, 1]]
    radii = torch.sqrt(displacement.square().sum(dim=-1) + radial_epsilon**2)
    return PairGeometry(
        pairs=pairs,
        radii=radii,
        unit_vectors=displacement / radii[..., None],
        radial_epsilon=radial_epsilon,
    )


def pair_weights_from_masses(
    relative_masses: torch.Tensor,
    *,
    pairs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return positive L1-normalized pair products from inferred relative masses."""

    if relative_masses.ndim != 1 or relative_masses.numel() < 2:
        raise ValueError("relative_masses must be a vector with at least two entries")
    if not relative_masses.is_floating_point():
        raise TypeError("relative_masses must use a floating-point dtype")
    if not bool(torch.isfinite(relative_masses).all()) or bool((relative_masses <= 0).any()):
        raise ValueError("relative_masses must be finite and positive")
    if pairs is None:
        pairs = pair_indices(relative_masses.numel(), device=relative_masses.device)
    pairs = torch.as_tensor(pairs, device=relative_masses.device, dtype=torch.long)
    weights = relative_masses[pairs[:, 0]] * relative_masses[pairs[:, 1]]
    return weights / weights.sum()


def _matern52_kernel_and_second_derivative(
    first: torch.Tensor,
    second: torch.Tensor,
    lengthscale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``k(first, second)`` and its derivative with respect to ``second``."""

    difference = first[:, None] - second[None, :]
    scaled_radius = difference.abs() / lengthscale
    w = _SQRT5 * scaled_radius
    exponential = torch.exp(-w)
    kernel = (1 + w + w.square() / 3) * exponential
    derivative = (5.0 / 3.0) * (1 + w) * exponential * difference / lengthscale.square()
    return kernel, derivative


@dataclass(frozen=True)
class RadialNystromBasis:
    """Whitened Matérn-5/2 inducing features and their analytic derivatives."""

    inducing_radii: torch.Tensor
    lengthscale: torch.Tensor
    inducing_cholesky: torch.Tensor
    jitter: float
    selection_strategy: str

    @property
    def rank(self) -> int:
        return self.inducing_radii.numel()

    @classmethod
    def from_radii(
        cls,
        radii: torch.Tensor,
        *,
        lengthscale: float | torch.Tensor,
        rank: int = 128,
        selection_strategy: str = "quantile",
        jitter: float | None = None,
    ) -> RadialNystromBasis:
        """Choose deterministic inducing radii and whiten their kernel.

        ``quantile`` preserves the empirical radius density. ``hybrid_log`` devotes
        half of the sites to log-spaced coverage over the observed fit support, which
        protects sparsely sampled close-encounter radii without using any held-out
        inputs or targets.
        """

        if not radii.is_floating_point() or radii.numel() == 0:
            raise ValueError("radii must be a non-empty floating-point tensor")
        flattened = radii.reshape(-1)
        if not bool(torch.isfinite(flattened).all()) or bool((flattened <= 0).any()):
            raise ValueError("radii must be finite and positive")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if selection_strategy not in {"quantile", "hybrid_log"}:
            raise ValueError(f"unknown inducing selection strategy: {selection_strategy}")
        lengthscale_tensor = torch.as_tensor(
            lengthscale,
            device=radii.device,
            dtype=radii.dtype,
        )
        if lengthscale_tensor.ndim != 0:
            raise ValueError("lengthscale must be scalar")
        if not bool(torch.isfinite(lengthscale_tensor)) or float(lengthscale_tensor) <= 0:
            raise ValueError("lengthscale must be finite and positive")

        target_rank = min(rank, flattened.numel())
        if selection_strategy == "quantile" or target_rank < 3:
            quantiles = torch.linspace(0, 1, target_rank, device=radii.device, dtype=radii.dtype)
            inducing = torch.quantile(flattened, quantiles)
        else:
            # Both component grids include the endpoints. Request two extra total
            # sites so their union ordinarily retains ``target_rank`` unique values.
            quantile_count = target_rank // 2 + 1
            support_count = target_rank - quantile_count + 2
            quantiles = torch.linspace(
                0,
                1,
                quantile_count,
                device=radii.device,
                dtype=radii.dtype,
            )
            empirical = torch.quantile(flattened, quantiles)
            support = torch.exp(
                torch.linspace(
                    flattened.min().log(),
                    flattened.max().log(),
                    support_count,
                    device=radii.device,
                    dtype=radii.dtype,
                )
            )
            support = torch.cat((flattened.min()[None], support[1:-1], flattened.max()[None]))
            inducing = torch.cat((empirical, support))
        inducing = torch.unique(inducing, sorted=True)
        if inducing.numel() == 0:
            raise RuntimeError("no distinct inducing radii were available")

        covariance, _ = _matern52_kernel_and_second_derivative(
            inducing,
            inducing,
            lengthscale_tensor,
        )
        covariance = 0.5 * (covariance + covariance.T)
        scale = covariance.diag().abs().max().clamp_min(1)
        if jitter is None:
            eps = torch.finfo(radii.dtype).eps
            jitter_tensor = scale * max(64 * eps, eps**0.5 * 1e-2)
        else:
            if jitter <= 0:
                raise ValueError("jitter must be positive")
            jitter_tensor = scale * jitter
        eye = torch.eye(inducing.numel(), device=radii.device, dtype=radii.dtype)
        for _ in range(8):
            factor, info = torch.linalg.cholesky_ex(covariance + jitter_tensor * eye)
            if int(info) == 0:
                break
            jitter_tensor = jitter_tensor * 10
        else:
            raise RuntimeError("inducing covariance is not numerically positive definite")
        return cls(
            inducing_radii=inducing,
            lengthscale=lengthscale_tensor,
            inducing_cholesky=factor,
            jitter=float(jitter_tensor),
            selection_strategy=selection_strategy,
        )

    def evaluate(self, radii: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``phi(r)`` and ``d phi(r) / dr`` with a trailing rank dimension."""

        if radii.device != self.inducing_radii.device or radii.dtype != self.inducing_radii.dtype:
            raise ValueError("radii and basis must have the same device and dtype")
        if not bool(torch.isfinite(radii).all()) or bool((radii <= 0).any()):
            raise ValueError("radii must be finite and positive")
        original_shape = radii.shape
        flattened = radii.reshape(-1)
        covariance, derivative = _matern52_kernel_and_second_derivative(
            self.inducing_radii,
            flattened,
            self.lengthscale,
        )
        features = torch.linalg.solve_triangular(
            self.inducing_cholesky,
            covariance,
            upper=False,
        ).T
        feature_derivatives = torch.linalg.solve_triangular(
            self.inducing_cholesky,
            derivative,
            upper=False,
        ).T
        return (
            features.reshape(*original_shape, self.rank),
            feature_derivatives.reshape(*original_shape, self.rank),
        )


def build_state_design(
    q: torch.Tensor,
    basis: RadialNystromBasis,
    pair_weights: torch.Tensor,
    *,
    n_particles: int,
    spatial_dims: int,
    pairs: torch.Tensor | None = None,
    radial_epsilon: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return energy and Cartesian-gradient feature rows.

    Shapes are ``(N, M)`` and ``(N, P*d, M)``. The latter is the analytic
    derivative of every energy-design column with respect to every coordinate.
    """

    geometry = pair_geometry(
        q,
        n_particles=n_particles,
        spatial_dims=spatial_dims,
        pairs=pairs,
        radial_epsilon=radial_epsilon,
    )
    pair_weights = torch.as_tensor(pair_weights, device=q.device, dtype=q.dtype)
    if pair_weights.shape != (geometry.pairs.shape[0],):
        raise ValueError(f"pair_weights must have shape ({geometry.pairs.shape[0]},)")
    if not bool(torch.isfinite(pair_weights).all()) or bool((pair_weights <= 0).any()):
        raise ValueError("pair_weights must be finite and positive")

    features, derivatives = basis.evaluate(geometry.radii)
    value_design = torch.einsum("e,nem->nm", pair_weights, features)
    gradient_design = q.new_zeros(
        q.shape[0],
        n_particles,
        spatial_dims,
        basis.rank,
    )
    for edge, (first, second) in enumerate(geometry.pairs.tolist()):
        contribution = (
            pair_weights[edge]
            * geometry.unit_vectors[:, edge, :, None]
            * derivatives[:, edge, None, :]
        )
        gradient_design[:, first] += contribution
        gradient_design[:, second] -= contribution
    return value_design, gradient_design.reshape(q.shape[0], n_particles * spatial_dims, basis.rank)


def nystrom_value_residual_variance(
    radii: torch.Tensor,
    basis: RadialNystromBasis,
    pair_weights: torch.Tensor,
    *,
    value_design: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the nonnegative exact-prior minus Nyström-prior value diagonal.

    All pairs evaluate one shared latent radial function, so the exact state prior
    diagonal contains cross-pair terms. Adding this residual to the finite-feature
    posterior avoids spuriously vanishing variance away from the inducing radii.
    """

    if radii.ndim != 2 or radii.device != basis.inducing_radii.device:
        raise ValueError("radii must have shape (N, E) on the basis device")
    if radii.dtype != basis.inducing_radii.dtype:
        raise ValueError("radii and basis must have the same dtype")
    pair_weights = torch.as_tensor(pair_weights, device=radii.device, dtype=radii.dtype)
    if pair_weights.shape != (radii.shape[1],):
        raise ValueError(f"pair_weights must have shape ({radii.shape[1]},)")
    difference = radii[:, :, None] - radii[:, None, :]
    w = _SQRT5 * difference.abs() / basis.lengthscale
    exact_pair_covariance = (1 + w + w.square() / 3) * torch.exp(-w)
    exact_prior = torch.einsum(
        "e,nef,f->n",
        pair_weights,
        exact_pair_covariance,
        pair_weights,
    )
    if value_design is None:
        features, _ = basis.evaluate(radii)
        value_design = torch.einsum("e,nem->nm", pair_weights, features)
    if value_design.shape != (radii.shape[0], basis.rank):
        raise ValueError("value_design has an invalid shape")
    feature_prior = value_design.square().sum(dim=1)
    return (exact_prior - feature_prior).clamp_min(0)


def pooled_radial_median_scale(
    q: torch.Tensor,
    *,
    n_particles: int,
    spatial_dims: int,
    max_points: int = 512,
    radial_epsilon: float | None = None,
) -> torch.Tensor:
    """Return the median of per-edge median nonzero fit-radius differences."""

    if max_points < 2:
        raise ValueError("max_points must be at least two")
    sample = q[: min(q.shape[0], max_points)]
    geometry = pair_geometry(
        sample,
        n_particles=n_particles,
        spatial_dims=spatial_dims,
        radial_epsilon=radial_epsilon,
    )
    edge_medians = []
    for edge in range(geometry.radii.shape[1]):
        distances = torch.pdist(geometry.radii[:, edge : edge + 1])
        distances = distances[distances > 0]
        if distances.numel():
            edge_medians.append(distances.median())
    if not edge_medians:
        raise RuntimeError("fit pair radii do not define a positive lengthscale")
    return torch.stack(edge_medians).median()
