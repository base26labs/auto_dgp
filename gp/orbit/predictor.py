"""Local scalar prediction with the ORBIT reduced-gradient operator.

This is deliberately a prediction-first research prototype.  It consumes fixed
kernel/noise parameters and evaluates the same local Gaussian conditional as
TERA, expressed in the orthonormal target subspace.  Hyperparameter learning
through iterative solves is a separate experiment and is not silently claimed
by this module.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from gp.orbit.operator import (
    CGResult,
    LocalGeometry,
    OrthonormalReducedOperator,
    PosteriorCertificate,
    ReducedKroneckerPreconditioner,
    build_local_geometry_from_differences,
    compute_posterior_certificate,
    solve_reduced_cg,
)

_SQRT5 = math.sqrt(5.0)


@dataclass(frozen=True)
class LocalPrediction:
    mean: torch.Tensor
    variance: torch.Tensor
    rank: int
    basis_is_exact: bool
    finite_precision_variance_correction: torch.Tensor
    solve: CGResult
    certificate: PosteriorCertificate
    functional_mean: torch.Tensor | None = None
    mean_reassociation_delta: torch.Tensor | None = None


@dataclass(frozen=True)
class LocalValueGradientPrediction:
    """One local value prediction and the gradient of its posterior mean."""

    prediction: LocalPrediction
    mean_gradient: torch.Tensor
    adjoint_solve: CGResult


@dataclass(frozen=True)
class MarginalPredictions:
    mean: torch.Tensor
    variance: torch.Tensor
    ranks: torch.Tensor
    iterations: torch.Tensor
    operator_matvecs: torch.Tensor
    preconditioner_applications: torch.Tensor
    relative_residuals: torch.Tensor
    converged: torch.Tensor
    variance_error_upper_bounds: torch.Tensor
    expected_kl_upper_bounds: torch.Tensor
    exact_arithmetic_certified: torch.Tensor
    floating_point_rigorous: torch.Tensor
    basis_exact: torch.Tensor
    finite_precision_variance_corrections: torch.Tensor
    functional_mean: torch.Tensor | None = None
    mean_reassociation_deltas: torch.Tensor | None = None
    mean_error_upper_bounds: torch.Tensor | None = None
    conditional_observation_norms: torch.Tensor | None = None
    mean_solve_certified: torch.Tensor | None = None
    mean_gradient: torch.Tensor | None = None
    adjoint_iterations: torch.Tensor | None = None
    adjoint_operator_matvecs: torch.Tensor | None = None
    adjoint_preconditioner_applications: torch.Tensor | None = None
    adjoint_relative_residuals: torch.Tensor | None = None
    adjoint_converged: torch.Tensor | None = None


@dataclass(frozen=True)
class LocalValueSystem:
    """Reusable selected-support conditional for independent zero-start solves.

    The object deliberately caches construction and the default preconditioner,
    but never caches a solve.  A tolerance sweep can therefore reuse exactly
    the same geometry, factor, operator, observations, and preconditioner while
    keeping every CG trajectory independent and zero-started.

    The dataclass is frozen, but PyTorch tensors are not deeply immutable.
    Solvers do not modify this state; callers must likewise avoid in-place
    mutation while reusing it.
    """

    geometry: LocalGeometry
    function_cholesky: torch.Tensor
    function_system_matrix: torch.Tensor
    function_jitter_requested: float
    function_jitter_used: float
    function_jitter_attempts: int
    operator: OrthonormalReducedOperator | None
    preconditioner: ReducedKroneckerPreconditioner | None
    conditional_cross: torch.Tensor
    function_weights: torch.Tensor
    conditional_value_variance: torch.Tensor
    value_condition: torch.Tensor
    orthonormal_observations: torch.Tensor
    base_mean: torch.Tensor
    conditional_observation_functional: torch.Tensor
    operator_eigenvalue_lower_bound: float
    operator_lower_bound_provenance: str
    operator_norm_upper_bound: float | None
    operator_norm_upper_bound_provenance: str

    @property
    def rhs(self) -> torch.Tensor:
        """Alias for the represented reduced-system right-hand side."""

        return self.conditional_cross


@dataclass(frozen=True)
class _CholeskyWithJitter:
    factor: torch.Tensor
    system_matrix: torch.Tensor
    jitter_used: float
    attempts: int


def _scaled_difference(
    first: torch.Tensor,
    second: torch.Tensor,
    lengthscale: torch.Tensor,
) -> torch.Tensor:
    difference = first[:, None, :] - second[None, :, :]
    if lengthscale.numel() == 1:
        return difference / lengthscale.reshape(1, 1, 1)
    return difference / lengthscale.reshape(1, 1, -1)


def _function_covariance(
    first: torch.Tensor,
    second: torch.Tensor,
    lengthscale: torch.Tensor,
    outputscale: torch.Tensor,
    kernel: str,
) -> torch.Tensor:
    difference = _scaled_difference(first, second, lengthscale)
    distance2 = (difference * difference).sum(dim=-1).clamp_min(0.0)
    if kernel == "rbf":
        return outputscale * torch.exp(-0.5 * distance2)
    if kernel == "matern52":
        radius = torch.sqrt(distance2.clamp_min(1e-12))
        scaled = _SQRT5 * radius
        return outputscale * (1.0 + scaled + scaled * scaled / 3.0) * torch.exp(-scaled)
    raise ValueError(f"unknown kernel: {kernel}")


def _gradient_scalars(
    distance2: torch.Tensor,
    outputscale: torch.Tensor,
    kernel: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kernel == "rbf":
        alpha = outputscale * torch.exp(-0.5 * distance2)
        return alpha, -alpha
    if kernel == "matern52":
        scaled = _SQRT5 * torch.sqrt(distance2.clamp_min(1e-12))
        exponential = torch.exp(-scaled)
        alpha = (5.0 / 3.0) * outputscale * (1.0 + scaled) * exponential
        beta = -(25.0 / 3.0) * outputscale * exponential
        return alpha, beta
    raise ValueError(f"unknown kernel: {kernel}")


def _cholesky_with_jitter(
    matrix: torch.Tensor,
    *,
    initial_jitter: float,
    maximum_jitter: float = 1e-1,
) -> torch.Tensor:
    """Compatibility wrapper returning only the accepted Cholesky factor."""

    return _cholesky_with_jitter_details(
        matrix,
        initial_jitter=initial_jitter,
        maximum_jitter=maximum_jitter,
    ).factor


def _cholesky_with_jitter_details(
    matrix: torch.Tensor,
    *,
    initial_jitter: float,
    maximum_jitter: float = 1e-1,
) -> _CholeskyWithJitter:
    """Factor a matrix and retain the source-dtype jitter actually accepted."""

    if initial_jitter < 0.0:
        raise ValueError("initial_jitter must be non-negative")
    identity = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    jitter = initial_jitter
    attempts = 0
    while jitter <= maximum_jitter:
        attempts += 1
        source_jitter = matrix.new_tensor(jitter)
        system_matrix = matrix + source_jitter * identity
        factor, info = torch.linalg.cholesky_ex(system_matrix)
        if int(info.max()) == 0:
            return _CholeskyWithJitter(
                factor=factor,
                system_matrix=system_matrix,
                jitter_used=float(source_jitter),
                attempts=attempts,
            )
        jitter = torch.finfo(matrix.dtype).eps if jitter == 0.0 else jitter * 10.0
    raise RuntimeError(f"Cholesky failed through jitter={maximum_jitter}")


def _projected_noise_gram(
    raw_differences: torch.Tensor,
    lengthscale: torch.Tensor,
    model: str,
) -> torch.Tensor:
    if model == "iid":
        result = raw_differences.T @ raw_differences
    elif model == "scaled":
        lengthscale2 = lengthscale * lengthscale
        if lengthscale2.numel() == 1:
            scaled = raw_differences * lengthscale2.reshape(1, 1)
        else:
            scaled = raw_differences * lengthscale2.reshape(-1, 1)
        result = raw_differences.T @ scaled
    elif model == "metric_matched":
        lengthscale2 = lengthscale * lengthscale
        if lengthscale2.numel() == 1:
            scaled = raw_differences / lengthscale2.reshape(1, 1)
        else:
            scaled = raw_differences / lengthscale2.reshape(-1, 1)
        result = raw_differences.T @ scaled
    else:
        raise ValueError(
            "gradient_noise_model must be 'iid', 'scaled' (vendor TERA), or 'metric_matched'"
        )
    return 0.5 * (result + result.T)


def _require_finite_nonnegative_scalar(value: torch.Tensor, name: str) -> None:
    if value.numel() != 1 or not bool(torch.isfinite(value).item()) or float(value) < 0.0:
        raise ValueError(f"{name} must be a finite non-negative scalar")


def _trusted_operator_lower_bound(
    geometry: LocalGeometry,
    lengthscale: torch.Tensor,
    gradient_noise: torch.Tensor,
    gradient_noise_model: str,
    reduced_jitter: float,
) -> tuple[float, str]:
    """Return the builder-specific represented-system noise lower bound.

    This uses the known relationship between raw and scaled differences; it is
    not a claim available to an arbitrarily constructed reduced operator.
    Values are evaluated in the source dtype without directed rounding.
    """

    minimum_lengthscale2 = lengthscale.square().min()
    if gradient_noise_model == "iid":
        gradient_component = gradient_noise * minimum_lengthscale2
        noise_provenance = "gradient_noise*min(lengthscale^2)"
    elif gradient_noise_model == "scaled":
        gradient_component = gradient_noise * minimum_lengthscale2.square()
        noise_provenance = "gradient_noise*min(lengthscale^4)"
    elif gradient_noise_model == "metric_matched":
        gradient_component = gradient_noise
        noise_provenance = "gradient_noise"
    else:  # Kept fail-closed even though _projected_noise_gram also validates.
        raise ValueError(f"unknown gradient noise model: {gradient_noise_model}")

    q_jitter = geometry.eigenvalues.new_tensor(reduced_jitter)
    q_jitter_component = q_jitter / geometry.eigenvalues.max()
    lower_bound = float(gradient_component + q_jitter_component)
    provenance = (
        f"trusted_gp_builder:{noise_provenance}+"
        "reduced_jitter/max_scaled_difference_eigenvalue;"
        "source_dtype_not_directed_rounding"
    )
    return lower_bound, provenance


def _trusted_operator_norm_upper_bound(
    operator: OrthonormalReducedOperator,
    *,
    function_covariance_floor: torch.Tensor,
) -> tuple[float | None, str]:
    """Bound the represented operator without materializing its dense matrix.

    The block-row and Frobenius triangle inequalities are generic.  Bounding
    ``K_ff^-1`` by the positive value-noise-plus-jitter floor additionally uses
    the trusted builder's positive-semidefinite kernel assumption.
    """

    floor = float(function_covariance_floor)
    provenance = (
        "trusted_gp_builder:block_row_plus_q_frobenius_over_function_floor;"
        "source_dtype_not_directed_rounding"
    )
    if not math.isfinite(floor) or floor <= 0.0:
        return None, provenance + ";unavailable_nonpositive_function_floor"

    coordinate_gram = operator.coordinates @ operator.coordinates.T
    coordinate_diagonal = torch.diagonal(coordinate_gram)
    pair_distance2 = (
        coordinate_diagonal[:, None] + coordinate_diagonal[None, :] - 2.0 * coordinate_gram
    ).clamp_min(0.0)
    block_norms = operator.alpha.abs() + operator.beta.abs() * pair_distance2
    unconditional_bound = block_norms.sum(dim=1).max()
    unconditional_bound = unconditional_bound + torch.linalg.norm(
        operator.gradient_noise,
        ord="fro",
    )
    q_frobenius_squared = (operator.alpha.square() * pair_distance2).sum()
    bound = unconditional_bound + q_frobenius_squared / function_covariance_floor
    bound = bound + abs(operator.jitter)
    result = float(bound)
    if not math.isfinite(result) or result <= 0.0:
        return None, provenance + ";unavailable_nonpositive_or_nonfinite_result"
    return result, provenance


def _build_local_value_system_impl(
    x_condition: torch.Tensor,
    value_condition: torch.Tensor,
    gradient_condition: torch.Tensor,
    x_target: torch.Tensor,
    *,
    lengthscale: torch.Tensor,
    outputscale: float | torch.Tensor,
    value_noise_variance: float | torch.Tensor,
    gradient_noise_variance: float | torch.Tensor,
    kernel: str,
    gradient_noise_model: str = "iid",
    rank: int | None = None,
    relative_rank_tolerance: float | None = None,
    rank_epsilon: float | torch.Tensor | None = None,
    absolute_rank_cutoff: float | torch.Tensor | None = None,
    precomputed_geometry: LocalGeometry | None = None,
    function_jitter: float = 1e-8,
    reduced_jitter: float = 1e-8,
    build_preconditioner: bool = True,
) -> LocalValueSystem:
    """Internal builder; precomputed geometry is trusted only after caller binding."""

    if x_condition.ndim != 2:
        raise ValueError("x_condition must have shape (m, d)")
    if x_target.shape != (1, x_condition.shape[1]):
        raise ValueError("x_target must have shape (1, d)")
    m, dimension = x_condition.shape
    if value_condition.shape != (m,) or gradient_condition.shape != (m, dimension):
        raise ValueError("conditioning observation shapes do not match x_condition")
    value_condition_snapshot = value_condition.clone()

    device, dtype = x_condition.device, x_condition.dtype
    lengthscale = torch.as_tensor(lengthscale, device=device, dtype=dtype).reshape(-1)
    if lengthscale.numel() not in (1, dimension) or not bool(
        torch.isfinite(lengthscale).all().item()
    ):
        raise ValueError("lengthscale must be one finite scalar or one value per dimension")
    if bool((lengthscale <= 0.0).any().item()):
        raise ValueError("lengthscale must be positive")
    outputscale = torch.as_tensor(outputscale, device=device, dtype=dtype)
    value_noise = torch.as_tensor(value_noise_variance, device=device, dtype=dtype)
    gradient_noise = torch.as_tensor(gradient_noise_variance, device=device, dtype=dtype)
    _require_finite_nonnegative_scalar(outputscale, "outputscale")
    _require_finite_nonnegative_scalar(value_noise, "value_noise_variance")
    _require_finite_nonnegative_scalar(gradient_noise, "gradient_noise_variance")
    if not math.isfinite(function_jitter) or function_jitter < 0.0:
        raise ValueError("function_jitter must be finite and non-negative")
    if not math.isfinite(reduced_jitter) or reduced_jitter < 0.0:
        raise ValueError("reduced_jitter must be finite and non-negative")

    raw_differences = (x_condition - x_target).T.contiguous()
    if lengthscale.numel() == 1:
        scaled_differences = raw_differences / lengthscale.reshape(1, 1)
    else:
        scaled_differences = raw_differences / lengthscale.reshape(-1, 1)
    if precomputed_geometry is None:
        geometry = build_local_geometry_from_differences(
            scaled_differences,
            rank=rank,
            relative_tolerance=relative_rank_tolerance,
            rank_epsilon=rank_epsilon,
            absolute_singular_value_cutoff=absolute_rank_cutoff,
        )
    else:
        if any(
            value is not None
            for value in (
                rank,
                relative_rank_tolerance,
                rank_epsilon,
                absolute_rank_cutoff,
            )
        ):
            raise ValueError(
                "precomputed_geometry is mutually exclusive with rank selection arguments"
            )
        if type(precomputed_geometry) is not LocalGeometry:
            raise TypeError("precomputed_geometry must be an exact LocalGeometry")
        geometry = precomputed_geometry
        geometry_tensors = (
            geometry.coordinates,
            geometry.q_to_z,
            geometry.eigenvalues,
            geometry.discarded_eigenvalue_sum,
        )
        if any(value.dtype != dtype or value.device != device for value in geometry_tensors):
            raise ValueError("precomputed geometry must match the input dtype and device")
        if any(not bool(torch.isfinite(value).all().item()) for value in geometry_tensors):
            raise ValueError("precomputed geometry tensors must be finite")
        if geometry.coordinates.shape[0] != m or geometry.q_to_z.shape != (
            m,
            geometry.rank,
        ):
            raise ValueError("precomputed geometry has incompatible coordinate shapes")
        if geometry.eigenvalues.shape != (geometry.rank,):
            raise ValueError("precomputed geometry eigenvalues have an incompatible shape")
        if geometry.discarded_eigenvalue_sum.shape != ():
            raise ValueError("precomputed discarded eigenvalue sum must be scalar")
        if (
            bool((geometry.eigenvalues <= 0.0).any().item())
            or float(geometry.discarded_eigenvalue_sum) < 0.0
        ):
            raise ValueError("precomputed geometry spectrum must be nonnegative")
    # Kernel coefficients continue to use the full geometry in approximate-rank
    # mode; only the represented observation basis is truncated.
    gram = scaled_differences.T @ scaled_differences

    function_covariance = _function_covariance(
        x_condition,
        x_condition,
        lengthscale,
        outputscale,
        kernel,
    )
    function_covariance = 0.5 * (function_covariance + function_covariance.T)
    function_covariance = function_covariance + value_noise * torch.eye(
        m,
        device=device,
        dtype=dtype,
    )
    factorization = _cholesky_with_jitter_details(
        function_covariance,
        initial_jitter=function_jitter,
    )
    function_cholesky = factorization.factor
    target_function_covariance = _function_covariance(
        x_condition,
        x_target,
        lengthscale,
        outputscale,
        kernel,
    ).squeeze(1)
    function_weights = torch.cholesky_solve(
        target_function_covariance.unsqueeze(1),
        function_cholesky,
    ).squeeze(1)
    conditional_value_variance = outputscale - torch.dot(
        target_function_covariance,
        function_weights,
    )
    base_mean = torch.dot(function_weights, value_condition_snapshot)

    if geometry.rank == 0:
        empty = x_condition.new_empty((0,))
        return LocalValueSystem(
            geometry=geometry,
            function_cholesky=function_cholesky,
            function_system_matrix=factorization.system_matrix,
            function_jitter_requested=function_jitter,
            function_jitter_used=factorization.jitter_used,
            function_jitter_attempts=factorization.attempts,
            operator=None,
            preconditioner=None,
            conditional_cross=empty,
            function_weights=function_weights,
            conditional_value_variance=conditional_value_variance,
            value_condition=value_condition_snapshot,
            orthonormal_observations=empty.clone(),
            base_mean=base_mean,
            conditional_observation_functional=empty.clone(),
            operator_eigenvalue_lower_bound=math.inf,
            operator_lower_bound_provenance="rank_zero_no_reduced_system",
            operator_norm_upper_bound=0.0,
            operator_norm_upper_bound_provenance="rank_zero_no_reduced_system",
        )

    diagonal = torch.diagonal(gram)
    pair_distance2 = diagonal[:, None] + diagonal[None, :] - 2.0 * gram
    pair_alpha, pair_beta = _gradient_scalars(
        pair_distance2.clamp_min(0.0),
        outputscale,
        kernel,
    )
    target_alpha, _ = _gradient_scalars(diagonal, outputscale, kernel)

    noise_gram = _projected_noise_gram(
        raw_differences,
        lengthscale,
        gradient_noise_model,
    )
    orthonormal_noise = geometry.q_to_z.T @ noise_gram @ geometry.q_to_z
    orthonormal_noise = gradient_noise * orthonormal_noise
    # TERA's q-coordinate epsilon transforms to epsilon T.T T in z coordinates.
    orthonormal_noise = orthonormal_noise + reduced_jitter * (geometry.q_to_z.T @ geometry.q_to_z)
    operator = OrthonormalReducedOperator(
        geometry.coordinates,
        pair_alpha,
        pair_beta,
        function_cholesky,
        orthonormal_noise,
        jitter=0.0,
    )
    conditional_cross = operator.conditional_cross(target_alpha, function_weights)
    preconditioner = ReducedKroneckerPreconditioner(operator) if build_preconditioner else None

    projected_observations = gradient_condition @ raw_differences
    orthonormal_observations = (projected_observations @ geometry.q_to_z).reshape(-1)
    value_observation_weights = torch.cholesky_solve(
        value_condition_snapshot.unsqueeze(1),
        function_cholesky,
    ).squeeze(1)
    conditional_observation_functional = orthonormal_observations - operator.q_matmul(
        value_observation_weights
    )
    lower_bound, lower_provenance = _trusted_operator_lower_bound(
        geometry,
        lengthscale,
        gradient_noise,
        gradient_noise_model,
        reduced_jitter,
    )
    function_covariance_floor = value_noise + value_noise.new_tensor(factorization.jitter_used)
    upper_bound, upper_provenance = _trusted_operator_norm_upper_bound(
        operator,
        function_covariance_floor=function_covariance_floor,
    )
    return LocalValueSystem(
        geometry=geometry,
        function_cholesky=function_cholesky,
        function_system_matrix=factorization.system_matrix,
        function_jitter_requested=function_jitter,
        function_jitter_used=factorization.jitter_used,
        function_jitter_attempts=factorization.attempts,
        operator=operator,
        preconditioner=preconditioner,
        conditional_cross=conditional_cross,
        function_weights=function_weights,
        conditional_value_variance=conditional_value_variance,
        value_condition=value_condition_snapshot,
        orthonormal_observations=orthonormal_observations,
        base_mean=base_mean,
        conditional_observation_functional=conditional_observation_functional,
        operator_eigenvalue_lower_bound=lower_bound,
        operator_lower_bound_provenance=lower_provenance,
        operator_norm_upper_bound=upper_bound,
        operator_norm_upper_bound_provenance=upper_provenance,
    )


def build_local_value_system(
    x_condition: torch.Tensor,
    value_condition: torch.Tensor,
    gradient_condition: torch.Tensor,
    x_target: torch.Tensor,
    *,
    lengthscale: torch.Tensor,
    outputscale: float | torch.Tensor,
    value_noise_variance: float | torch.Tensor,
    gradient_noise_variance: float | torch.Tensor,
    kernel: str,
    gradient_noise_model: str = "iid",
    rank: int | None = None,
    relative_rank_tolerance: float | None = None,
    rank_epsilon: float | torch.Tensor | None = None,
    absolute_rank_cutoff: float | torch.Tensor | None = None,
    function_jitter: float = 1e-8,
    reduced_jitter: float = 1e-8,
    build_preconditioner: bool = True,
) -> LocalValueSystem:
    """Build one local represented system for reuse across CG tolerances.

    Geometry is always derived from this call's conditioning inputs.  The sole
    trusted precomputed-geometry path is private to the registered calibration
    executor, which binds the geometry to its source differences and digest.
    """

    return _build_local_value_system_impl(
        x_condition,
        value_condition,
        gradient_condition,
        x_target,
        lengthscale=lengthscale,
        outputscale=outputscale,
        value_noise_variance=value_noise_variance,
        gradient_noise_variance=gradient_noise_variance,
        kernel=kernel,
        gradient_noise_model=gradient_noise_model,
        rank=rank,
        relative_rank_tolerance=relative_rank_tolerance,
        rank_epsilon=rank_epsilon,
        absolute_rank_cutoff=absolute_rank_cutoff,
        precomputed_geometry=None,
        function_jitter=function_jitter,
        reduced_jitter=reduced_jitter,
        build_preconditioner=build_preconditioner,
    )


def _build_local_value_system_from_registered_geometry(
    x_condition: torch.Tensor,
    value_condition: torch.Tensor,
    gradient_condition: torch.Tensor,
    x_target: torch.Tensor,
    *,
    lengthscale: torch.Tensor,
    outputscale: float | torch.Tensor,
    value_noise_variance: float | torch.Tensor,
    gradient_noise_variance: float | torch.Tensor,
    kernel: str,
    gradient_noise_model: str,
    precomputed_geometry: LocalGeometry,
    function_jitter: float = 1e-8,
    reduced_jitter: float = 1e-8,
    build_preconditioner: bool = True,
) -> LocalValueSystem:
    """Consume geometry already authenticated by the registered probe executor."""

    return _build_local_value_system_impl(
        x_condition,
        value_condition,
        gradient_condition,
        x_target,
        lengthscale=lengthscale,
        outputscale=outputscale,
        value_noise_variance=value_noise_variance,
        gradient_noise_variance=gradient_noise_variance,
        kernel=kernel,
        gradient_noise_model=gradient_noise_model,
        rank=None,
        relative_rank_tolerance=None,
        rank_epsilon=None,
        absolute_rank_cutoff=None,
        precomputed_geometry=precomputed_geometry,
        function_jitter=function_jitter,
        reduced_jitter=reduced_jitter,
        build_preconditioner=build_preconditioner,
    )


def solve_local_value_system(
    system: LocalValueSystem,
    *,
    tolerance: float = 1e-6,
    max_iterations: int | None = None,
    use_preconditioner: bool = True,
    preconditioner: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> LocalPrediction:
    """Run one independent zero-start solve against a reusable local system."""

    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if max_iterations is not None and max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if not use_preconditioner and preconditioner is not None:
        raise ValueError("preconditioner requires use_preconditioner=True")

    if system.geometry.rank == 0:
        empty = system.conditional_cross.clone()
        solve = CGResult(
            solution=empty.clone(),
            residual=empty.clone(),
            iterations=0,
            operator_matvecs=0,
            preconditioner_applications=0,
            relative_residual=0.0,
            residual_norm=0.0,
            rhs_norm=0.0,
            converged=True,
            recursive_residual=empty.clone(),
            recursive_relative_residual=0.0,
            recursive_residual_norm=0.0,
            operator_action=empty.clone(),
            requested_tolerance=tolerance,
            max_iterations=0 if max_iterations is None else max_iterations,
            termination_reason="zero_rhs",
            residual_is_fresh=True,
            fresh_check_count=0,
            residual_replacement_count=0,
            operator_norm_upper_bound=0.0,
        )
        zero = system.base_mean.new_zeros(())
        certificate = PosteriorCertificate(
            variance_error_upper_bound=0.0,
            expected_kl_upper_bound=0.0,
            operator_eigenvalue_lower_bound=math.inf,
            exact_arithmetic_certified=system.geometry.is_exact,
            solve_certified=True,
            basis_is_exact=system.geometry.is_exact,
            floating_point_rigorous=False,
            mean_error_upper_bound=0.0,
            conditional_observation_norm=0.0,
            mean_solve_certified=True,
            operator_lower_bound_provenance=system.operator_lower_bound_provenance,
        )
        return LocalPrediction(
            mean=system.base_mean,
            functional_mean=system.base_mean,
            mean_reassociation_delta=zero,
            variance=system.conditional_value_variance,
            rank=0,
            basis_is_exact=system.geometry.is_exact,
            finite_precision_variance_correction=zero,
            solve=solve,
            certificate=certificate,
        )

    if system.operator is None:
        raise ValueError("nonzero-rank LocalValueSystem must contain an operator")
    selected_preconditioner: Callable[[torch.Tensor], torch.Tensor] | None = None
    if use_preconditioner:
        selected_preconditioner = (
            preconditioner if preconditioner is not None else system.preconditioner
        )
        if selected_preconditioner is None:
            raise ValueError(
                "system has no cached preconditioner; supply one explicitly or rebuild "
                "with build_preconditioner=True"
            )
    solve = solve_reduced_cg(
        system.operator,
        system.conditional_cross,
        tolerance=tolerance,
        max_iterations=max_iterations,
        preconditioner=selected_preconditioner,
        operator_norm_upper_bound=system.operator_norm_upper_bound,
    )

    q_t_weights = system.operator.q_t_matmul(solve.solution)
    corrected_function_weights = system.function_weights - torch.cholesky_solve(
        q_t_weights.unsqueeze(1),
        system.function_cholesky,
    ).squeeze(1)
    mean = torch.dot(corrected_function_weights, system.value_condition)
    mean = mean + torch.dot(solve.solution, system.orthonormal_observations)
    functional_mean = system.base_mean + torch.dot(
        solve.solution,
        system.conditional_observation_functional,
    )
    mean_reassociation_delta = mean - functional_mean

    variance_correction = -torch.dot(solve.solution, solve.residual)
    variance = system.conditional_value_variance - torch.dot(
        solve.solution,
        system.conditional_cross,
    )
    variance = variance + variance_correction
    conditional_observation_norm = float(
        torch.linalg.norm(system.conditional_observation_functional)
    )
    certificate = compute_posterior_certificate(
        system.operator,
        solve,
        variance,
        basis_is_exact=system.geometry.is_exact,
        conditional_observation_norm=conditional_observation_norm,
        operator_eigenvalue_lower_bound=system.operator_eigenvalue_lower_bound,
        operator_lower_bound_provenance=system.operator_lower_bound_provenance,
        rhs=system.conditional_cross,
    )
    return LocalPrediction(
        mean=mean,
        functional_mean=functional_mean,
        mean_reassociation_delta=mean_reassociation_delta,
        variance=variance,
        rank=system.geometry.rank,
        basis_is_exact=system.geometry.is_exact,
        finite_precision_variance_correction=variance_correction,
        solve=solve,
        certificate=certificate,
    )


def predict_local_value(
    x_condition: torch.Tensor,
    value_condition: torch.Tensor,
    gradient_condition: torch.Tensor,
    x_target: torch.Tensor,
    *,
    lengthscale: torch.Tensor,
    outputscale: float | torch.Tensor,
    value_noise_variance: float | torch.Tensor,
    gradient_noise_variance: float | torch.Tensor,
    kernel: str,
    gradient_noise_model: str = "iid",
    rank: int | None = None,
    relative_rank_tolerance: float | None = None,
    rank_epsilon: float | torch.Tensor | None = None,
    absolute_rank_cutoff: float | torch.Tensor | None = None,
    cg_tolerance: float = 1e-6,
    cg_max_iterations: int | None = None,
    use_preconditioner: bool = True,
    function_jitter: float = 1e-8,
    reduced_jitter: float = 1e-8,
) -> LocalPrediction:
    """Build and solve one target conditional with the historical interface."""

    system = build_local_value_system(
        x_condition,
        value_condition,
        gradient_condition,
        x_target,
        lengthscale=lengthscale,
        outputscale=outputscale,
        value_noise_variance=value_noise_variance,
        gradient_noise_variance=gradient_noise_variance,
        kernel=kernel,
        gradient_noise_model=gradient_noise_model,
        rank=rank,
        relative_rank_tolerance=relative_rank_tolerance,
        rank_epsilon=rank_epsilon,
        absolute_rank_cutoff=absolute_rank_cutoff,
        function_jitter=function_jitter,
        reduced_jitter=reduced_jitter,
        build_preconditioner=use_preconditioner,
    )
    return solve_local_value_system(
        system,
        tolerance=cg_tolerance,
        max_iterations=cg_max_iterations,
        use_preconditioner=use_preconditioner,
    )


def predict_local_value_and_mean_gradient(
    x_condition: torch.Tensor,
    value_condition: torch.Tensor,
    gradient_condition: torch.Tensor,
    x_target: torch.Tensor,
    *,
    lengthscale: torch.Tensor,
    outputscale: float | torch.Tensor,
    value_noise_variance: float | torch.Tensor,
    gradient_noise_variance: float | torch.Tensor,
    kernel: str,
    gradient_noise_model: str = "iid",
    rank: int | None = None,
    relative_rank_tolerance: float | None = None,
    rank_epsilon: float | torch.Tensor | None = None,
    absolute_rank_cutoff: float | torch.Tensor | None = None,
    cg_tolerance: float = 1e-6,
    cg_max_iterations: int | None = None,
    use_preconditioner: bool = True,
    function_jitter: float = 1e-8,
    reduced_jitter: float = 1e-8,
) -> LocalValueGradientPrediction:
    """Predict a value and differentiate its local posterior mean implicitly.

    The primal and adjoint CG solves run without an autograd tape.  The returned
    gradient uses ``A w = b`` and ``A.T u = h`` to differentiate
    ``base_mean + w.T h`` as ``d(base_mean) + w.T dh + u.T(db - dA w)``.
    Neighbour selection is intentionally outside this function and therefore
    treated as piecewise constant, matching TERA's gradient benchmark.
    """

    with torch.enable_grad():
        differentiable_target = x_target.detach().clone().requires_grad_(True)
        system = build_local_value_system(
            x_condition,
            value_condition,
            gradient_condition,
            differentiable_target,
            lengthscale=lengthscale,
            outputscale=outputscale,
            value_noise_variance=value_noise_variance,
            gradient_noise_variance=gradient_noise_variance,
            kernel=kernel,
            gradient_noise_model=gradient_noise_model,
            rank=rank,
            relative_rank_tolerance=relative_rank_tolerance,
            rank_epsilon=rank_epsilon,
            absolute_rank_cutoff=absolute_rank_cutoff,
            function_jitter=function_jitter,
            reduced_jitter=reduced_jitter,
            build_preconditioner=use_preconditioner,
        )
        with torch.no_grad():
            prediction = solve_local_value_system(
                system,
                tolerance=cg_tolerance,
                max_iterations=cg_max_iterations,
                use_preconditioner=use_preconditioner,
            )
            if system.geometry.rank == 0:
                adjoint_solve = prediction.solve
            else:
                if system.operator is None:
                    raise RuntimeError("nonzero-rank system is missing its operator")
                preconditioner = system.preconditioner if use_preconditioner else None
                adjoint_solve = solve_reduced_cg(
                    system.operator,
                    system.conditional_observation_functional,
                    tolerance=cg_tolerance,
                    max_iterations=cg_max_iterations,
                    preconditioner=preconditioner,
                    operator_norm_upper_bound=system.operator_norm_upper_bound,
                )

        if system.geometry.rank == 0:
            differentiable_mean = system.base_mean
        else:
            if system.operator is None:
                raise RuntimeError("nonzero-rank system is missing its operator")
            primal = prediction.solve.solution.detach()
            adjoint = adjoint_solve.solution.detach()
            primal_residual_expression = system.conditional_cross - system.operator.matmul(primal)
            differentiable_mean = system.base_mean
            differentiable_mean = differentiable_mean + torch.dot(
                primal,
                system.conditional_observation_functional,
            )
            differentiable_mean = differentiable_mean + torch.dot(
                adjoint,
                primal_residual_expression,
            )
        mean_gradient = torch.autograd.grad(
            differentiable_mean,
            differentiable_target,
        )[0].squeeze(0)

    return LocalValueGradientPrediction(
        prediction=prediction,
        mean_gradient=mean_gradient.detach(),
        adjoint_solve=adjoint_solve,
    )


def predict_marginal_values(
    x_train: torch.Tensor,
    value_train: torch.Tensor,
    gradient_train: torch.Tensor,
    x_eval: torch.Tensor,
    *,
    m: int,
    lengthscale: torch.Tensor,
    outputscale: float | torch.Tensor,
    value_noise_variance: float | torch.Tensor,
    gradient_noise_variance: float | torch.Tensor,
    kernel: str,
    gradient_noise_model: str = "iid",
    rank: int | None = None,
    relative_rank_tolerance: float | None = None,
    neighbour_indices: torch.Tensor | None = None,
    rank_epsilon: float | torch.Tensor | None = None,
    cg_tolerance: float = 1e-6,
    cg_max_iterations: int | None = None,
    use_preconditioner: bool = True,
    function_jitter: float = 1e-8,
    reduced_jitter: float = 1e-8,
    include_mean_gradient: bool = False,
) -> MarginalPredictions:
    """Predict independent local marginals using nearest neighbours.

    ``neighbour_indices`` and ``rank_epsilon`` expose a fail-closed path for
    numerical comparisons that must hold neighbour identity and the SVD rank
    rule fixed across dtypes.  When omitted, the historical behaviour is
    unchanged: neighbours are selected in the input dtype and the rank cutoff
    uses that dtype's machine epsilon.
    """

    if m <= 0:
        raise ValueError("m must be positive")
    lengthscale = torch.as_tensor(
        lengthscale,
        device=x_train.device,
        dtype=x_train.dtype,
    ).reshape(-1)
    if lengthscale.numel() == 1:
        train_scaled = x_train / lengthscale.reshape(1, 1)
        eval_scaled = x_eval / lengthscale.reshape(1, 1)
    else:
        train_scaled = x_train / lengthscale.reshape(1, -1)
        eval_scaled = x_eval / lengthscale.reshape(1, -1)
    count = min(m, x_train.shape[0])
    if neighbour_indices is None:
        neighbours = torch.topk(
            torch.cdist(eval_scaled, train_scaled),
            k=count,
            largest=False,
        ).indices
    else:
        if not isinstance(neighbour_indices, torch.Tensor):
            raise TypeError("neighbour_indices must be a torch.Tensor")
        if neighbour_indices.dtype != torch.long:
            raise TypeError("neighbour_indices must have dtype torch.long")
        if neighbour_indices.device != x_train.device:
            raise ValueError("neighbour_indices must be on the training-data device")
        expected_shape = (x_eval.shape[0], count)
        if neighbour_indices.shape != expected_shape:
            raise ValueError(f"neighbour_indices must have shape {expected_shape}")
        if neighbour_indices.numel() > 0:
            if bool((neighbour_indices < 0).any().item()) or bool(
                (neighbour_indices >= x_train.shape[0]).any().item()
            ):
                raise ValueError("neighbour_indices contains an out-of-range row")
            if count > 1:
                ordered = torch.sort(neighbour_indices, dim=1).values
                if bool((ordered[:, 1:] == ordered[:, :-1]).any().item()):
                    raise ValueError("each neighbour_indices row must contain unique rows")
        neighbours = neighbour_indices

    predictions = []
    mean_gradients = []
    adjoint_solves = []
    for target, indices in zip(x_eval, neighbours, strict=True):
        prediction_kwargs = {
            "lengthscale": lengthscale,
            "outputscale": outputscale,
            "value_noise_variance": value_noise_variance,
            "gradient_noise_variance": gradient_noise_variance,
            "kernel": kernel,
            "gradient_noise_model": gradient_noise_model,
            "rank": rank,
            "relative_rank_tolerance": relative_rank_tolerance,
            "rank_epsilon": rank_epsilon,
            "cg_tolerance": cg_tolerance,
            "cg_max_iterations": cg_max_iterations,
            "use_preconditioner": use_preconditioner,
            "function_jitter": function_jitter,
            "reduced_jitter": reduced_jitter,
        }
        if include_mean_gradient:
            result = predict_local_value_and_mean_gradient(
                x_train[indices],
                value_train[indices],
                gradient_train[indices],
                target.unsqueeze(0),
                **prediction_kwargs,
            )
            predictions.append(result.prediction)
            mean_gradients.append(result.mean_gradient)
            adjoint_solves.append(result.adjoint_solve)
        else:
            predictions.append(
                predict_local_value(
                    x_train[indices],
                    value_train[indices],
                    gradient_train[indices],
                    target.unsqueeze(0),
                    **prediction_kwargs,
                )
            )

    return MarginalPredictions(
        mean=torch.stack([prediction.mean for prediction in predictions]),
        functional_mean=torch.stack([prediction.functional_mean for prediction in predictions]),
        mean_reassociation_deltas=torch.stack(
            [prediction.mean_reassociation_delta for prediction in predictions]
        ),
        variance=torch.stack([prediction.variance for prediction in predictions]),
        ranks=torch.tensor([prediction.rank for prediction in predictions], device=x_train.device),
        iterations=torch.tensor(
            [prediction.solve.iterations for prediction in predictions],
            device=x_train.device,
        ),
        operator_matvecs=torch.tensor(
            [prediction.solve.operator_matvecs for prediction in predictions],
            device=x_train.device,
        ),
        preconditioner_applications=torch.tensor(
            [prediction.solve.preconditioner_applications for prediction in predictions],
            device=x_train.device,
        ),
        relative_residuals=torch.tensor(
            [prediction.solve.relative_residual for prediction in predictions],
            device=x_train.device,
            dtype=x_train.dtype,
        ),
        converged=torch.tensor(
            [prediction.solve.converged for prediction in predictions],
            device=x_train.device,
        ),
        variance_error_upper_bounds=torch.tensor(
            [prediction.certificate.variance_error_upper_bound for prediction in predictions],
            device=x_train.device,
            dtype=x_train.dtype,
        ),
        mean_error_upper_bounds=torch.tensor(
            [prediction.certificate.mean_error_upper_bound for prediction in predictions],
            device=x_train.device,
            dtype=x_train.dtype,
        ),
        conditional_observation_norms=torch.tensor(
            [prediction.certificate.conditional_observation_norm for prediction in predictions],
            device=x_train.device,
            dtype=x_train.dtype,
        ),
        mean_solve_certified=torch.tensor(
            [prediction.certificate.mean_solve_certified for prediction in predictions],
            device=x_train.device,
        ),
        expected_kl_upper_bounds=torch.tensor(
            [prediction.certificate.expected_kl_upper_bound for prediction in predictions],
            device=x_train.device,
            dtype=x_train.dtype,
        ),
        exact_arithmetic_certified=torch.tensor(
            [prediction.certificate.exact_arithmetic_certified for prediction in predictions],
            device=x_train.device,
        ),
        floating_point_rigorous=torch.tensor(
            [prediction.certificate.floating_point_rigorous for prediction in predictions],
            device=x_train.device,
        ),
        basis_exact=torch.tensor(
            [prediction.basis_is_exact for prediction in predictions],
            device=x_train.device,
        ),
        finite_precision_variance_corrections=torch.stack(
            [prediction.finite_precision_variance_correction for prediction in predictions]
        ),
        mean_gradient=torch.stack(mean_gradients) if include_mean_gradient else None,
        adjoint_iterations=(
            torch.tensor(
                [solve.iterations for solve in adjoint_solves],
                device=x_train.device,
            )
            if include_mean_gradient
            else None
        ),
        adjoint_operator_matvecs=(
            torch.tensor(
                [solve.operator_matvecs for solve in adjoint_solves],
                device=x_train.device,
            )
            if include_mean_gradient
            else None
        ),
        adjoint_preconditioner_applications=(
            torch.tensor(
                [solve.preconditioner_applications for solve in adjoint_solves],
                device=x_train.device,
            )
            if include_mean_gradient
            else None
        ),
        adjoint_relative_residuals=(
            torch.tensor(
                [solve.relative_residual for solve in adjoint_solves],
                device=x_train.device,
                dtype=x_train.dtype,
            )
            if include_mean_gradient
            else None
        ),
        adjoint_converged=(
            torch.tensor(
                [solve.converged for solve in adjoint_solves],
                device=x_train.device,
            )
            if include_mean_gradient
            else None
        ),
    )
