"""Fit-only Hamiltonian block discovery and an integrable kinetic mean."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _validate_observations(X: torch.Tensor, gradient: torch.Tensor) -> None:
    if X.ndim != 2 or not X.is_floating_point():
        raise ValueError("X must be a two-dimensional floating-point tensor")
    if gradient.shape != X.shape:
        raise ValueError("gradient must have the same shape as X")
    if gradient.device != X.device or gradient.dtype != X.dtype:
        raise ValueError("gradient and X must have the same device and dtype")
    if X.shape[0] < 2:
        raise ValueError("at least two observations are required")
    if not bool(torch.isfinite(X).all()) or not bool(torch.isfinite(gradient).all()):
        raise ValueError("X and gradient must be finite")


def _fit_affine_columns(
    X: torch.Tensor,
    gradient: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = X[:, indices]
    targets = gradient[:, indices]
    coordinate_mean = coordinates.mean(dim=0)
    target_mean = targets.mean(dim=0)
    centered_coordinates = coordinates - coordinate_mean
    centered_targets = targets - target_mean
    denominator = centered_coordinates.square().sum(dim=0)
    scale = coordinates.square().mean(dim=0).clamp_min(1).sqrt()
    minimum = torch.finfo(X.dtype).eps * X.shape[0] * scale.square()
    if bool((denominator <= minimum).any()):
        raise ValueError("candidate kinetic coordinates must vary on the fit rows")
    slopes = (centered_coordinates * centered_targets).sum(dim=0) / denominator
    intercepts = target_mean - slopes * coordinate_mean
    return slopes, intercepts


def _trajectory_folds(
    trajectory_id: torch.Tensor,
    folds: int,
) -> list[torch.Tensor]:
    groups = torch.unique(trajectory_id, sorted=True)
    fold_count = min(folds, groups.numel())
    if fold_count < 2:
        raise ValueError("structure discovery requires at least two trajectory groups")
    assignments = torch.arange(groups.numel(), device=groups.device) % fold_count
    return [groups[assignments == fold] for fold in range(fold_count)]


def _cross_validated_affine_score(
    X: torch.Tensor,
    gradient: torch.Tensor,
    trajectory_id: torch.Tensor,
    indices: torch.Tensor,
    folds: int,
) -> float:
    squared_error = X.new_zeros(())
    squared_reference = X.new_zeros(())
    for validation_groups in _trajectory_folds(trajectory_id, folds):
        validation = torch.isin(trajectory_id, validation_groups)
        training = ~validation
        if int(training.sum()) < 2 or not bool(validation.any()):
            raise ValueError("every structure-discovery fold needs train and validation rows")
        slopes, intercepts = _fit_affine_columns(X[training], gradient[training], indices)
        prediction = X[validation][:, indices] * slopes + intercepts
        target = gradient[validation][:, indices]
        reference_mean = gradient[training][:, indices].mean(dim=0)
        squared_error = squared_error + (prediction - target).square().sum()
        squared_reference = squared_reference + (target - reference_mean).square().sum()
    tiny = torch.finfo(X.dtype).tiny
    return float(torch.sqrt(squared_error / squared_reference.clamp_min(tiny)))


@dataclass(frozen=True)
class HamiltonianSplit:
    """A data-selected kinetic block and its schema-consistent complement."""

    kinetic_indices: torch.Tensor
    position_indices: torch.Tensor
    candidate_scores: tuple[float, float]
    selected_block: int
    n_particles: int
    spatial_dims: int

    @property
    def state_dimension(self) -> int:
        return 2 * self.n_particles * self.spatial_dims


def infer_hamiltonian_split(
    X: torch.Tensor,
    gradient: torch.Tensor,
    trajectory_id: torch.Tensor,
    *,
    n_particles: int,
    spatial_dims: int,
    cv_folds: int = 3,
    max_relative_error: float = 1e-3,
    min_score_ratio: float = 20.0,
    min_slope: float = 0.0,
) -> HamiltonianSplit:
    """Select which equal state block has fit-generalizing affine gradients.

    Only ``X``, ``gradient``, and trajectory membership are inspected. Particle masses,
    values, force-law constants, and any non-fit role are absent from this API.
    """

    _validate_observations(X, gradient)
    if n_particles <= 0 or spatial_dims <= 0:
        raise ValueError("n_particles and spatial_dims must be positive")
    if X.shape[1] != 2 * n_particles * spatial_dims:
        raise ValueError("X does not contain two schema-consistent particle blocks")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least two")
    if max_relative_error <= 0 or min_score_ratio <= 1:
        raise ValueError("structure thresholds must be positive and discriminating")

    trajectory_id = torch.as_tensor(trajectory_id, device=X.device)
    if trajectory_id.ndim != 1 or trajectory_id.numel() != X.shape[0]:
        raise ValueError("trajectory_id must have one entry per row")

    block_width = n_particles * spatial_dims
    candidates = (
        torch.arange(block_width, device=X.device),
        torch.arange(block_width, 2 * block_width, device=X.device),
    )
    scores = tuple(
        _cross_validated_affine_score(
            X,
            gradient,
            trajectory_id,
            indices,
            cv_folds,
        )
        for indices in candidates
    )
    selected = 0 if scores[0] < scores[1] else 1
    selected_score = scores[selected]
    alternative_score = scores[1 - selected]
    ratio = alternative_score / max(selected_score, torch.finfo(X.dtype).tiny)
    if selected_score > max_relative_error or ratio < min_score_ratio:
        raise RuntimeError(
            "no unambiguous affine-gradient block: "
            f"scores={scores}, required_error<={max_relative_error:g}, "
            f"required_ratio>={min_score_ratio:g}"
        )

    slopes, _ = _fit_affine_columns(X, gradient, candidates[selected])
    if bool((slopes <= min_slope).any()):
        raise RuntimeError("selected affine-gradient block does not have positive slopes")
    return HamiltonianSplit(
        kinetic_indices=candidates[selected],
        position_indices=candidates[1 - selected],
        candidate_scores=scores,
        selected_block=selected,
        n_particles=n_particles,
        spatial_dims=spatial_dims,
    )


@dataclass(frozen=True)
class DiagonalQuadraticMean:
    """Integrable mean whose selected gradients are ``slope * x + intercept``."""

    indices: torch.Tensor
    slopes: torch.Tensor
    intercepts: torch.Tensor
    input_dimension: int

    def value(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim != 2 or X.shape[1] != self.input_dimension:
            raise ValueError(f"X must have shape (N, {self.input_dimension})")
        selected = X[:, self.indices]
        return (0.5 * self.slopes * selected.square() + self.intercepts * selected).sum(dim=1)

    def gradient(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim != 2 or X.shape[1] != self.input_dimension:
            raise ValueError(f"X must have shape (N, {self.input_dimension})")
        result = X.new_zeros(X.shape)
        result[:, self.indices] = X[:, self.indices] * self.slopes + self.intercepts
        return result


def fit_diagonal_quadratic_mean(
    X: torch.Tensor,
    gradient: torch.Tensor,
    kinetic_indices: torch.Tensor,
) -> DiagonalQuadraticMean:
    """Fit the integrable kinetic mean on the supplied training role."""

    _validate_observations(X, gradient)
    kinetic_indices = torch.as_tensor(kinetic_indices, device=X.device, dtype=torch.long)
    if kinetic_indices.ndim != 1 or kinetic_indices.numel() == 0:
        raise ValueError("kinetic_indices must be a non-empty vector")
    slopes, intercepts = _fit_affine_columns(X, gradient, kinetic_indices)
    return DiagonalQuadraticMean(
        indices=kinetic_indices,
        slopes=slopes,
        intercepts=intercepts,
        input_dimension=X.shape[1],
    )


def infer_relative_masses(
    kinetic_mean: DiagonalQuadraticMean,
    *,
    n_particles: int,
    spatial_dims: int,
    max_axis_relative_deviation: float = 0.1,
) -> torch.Tensor:
    """Infer positive relative masses from inverse per-particle kinetic slopes."""

    expected = n_particles * spatial_dims
    if kinetic_mean.slopes.numel() != expected:
        raise ValueError(f"kinetic mean must contain {expected} particle-coordinate slopes")
    if max_axis_relative_deviation <= 0:
        raise ValueError("max_axis_relative_deviation must be positive")
    slopes = kinetic_mean.slopes.reshape(n_particles, spatial_dims)
    if not bool(torch.isfinite(slopes).all()) or bool((slopes <= 0).any()):
        raise RuntimeError("relative masses require finite positive kinetic slopes")
    particle_slopes = slopes.median(dim=1).values
    relative_deviation = (slopes / particle_slopes[:, None] - 1).abs()
    if float(relative_deviation.max()) > max_axis_relative_deviation:
        raise RuntimeError("kinetic slopes are inconsistent across a particle's axes")
    masses = particle_slopes.reciprocal()
    return masses / masses.mean()
