"""Local scalar prediction with the ORBIT reduced-gradient operator.

This is deliberately a prediction-first research prototype.  It consumes fixed
kernel/noise parameters and evaluates the same local Gaussian conditional as
TERA, expressed in the orthonormal target subspace.  Hyperparameter learning
through iterative solves is a separate experiment and is not silently claimed
by this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from gp.orbit.operator import (
    CGResult,
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
    if initial_jitter < 0.0:
        raise ValueError("initial_jitter must be non-negative")
    identity = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    jitter = initial_jitter
    while jitter <= maximum_jitter:
        factor, info = torch.linalg.cholesky_ex(matrix + jitter * identity)
        if int(info.max()) == 0:
            return factor
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
    cg_tolerance: float = 1e-6,
    cg_max_iterations: int | None = None,
    use_preconditioner: bool = True,
    function_jitter: float = 1e-8,
    reduced_jitter: float = 1e-8,
) -> LocalPrediction:
    """Evaluate one target conditional from fixed local observations."""

    if x_target.shape != (1, x_condition.shape[1]):
        raise ValueError("x_target must have shape (1, d)")
    m, dimension = x_condition.shape
    if value_condition.shape != (m,) or gradient_condition.shape != (m, dimension):
        raise ValueError("conditioning observation shapes do not match x_condition")

    device, dtype = x_condition.device, x_condition.dtype
    lengthscale = torch.as_tensor(lengthscale, device=device, dtype=dtype).reshape(-1)
    outputscale = torch.as_tensor(outputscale, device=device, dtype=dtype)
    value_noise = torch.as_tensor(value_noise_variance, device=device, dtype=dtype)
    gradient_noise = torch.as_tensor(gradient_noise_variance, device=device, dtype=dtype)

    raw_differences = (x_condition - x_target).T.contiguous()
    if lengthscale.numel() == 1:
        scaled_differences = raw_differences / lengthscale.reshape(1, 1)
    else:
        scaled_differences = raw_differences / lengthscale.reshape(-1, 1)
    geometry = build_local_geometry_from_differences(
        scaled_differences,
        rank=rank,
        relative_tolerance=relative_rank_tolerance,
    )
    # Kernel coefficients must use the full geometry even in the separately
    # labelled approximate-rank mode.  Reconstructing this Gram matrix from
    # truncated coordinates would silently change the kernel distances as
    # well as the projection basis.
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
    function_cholesky = _cholesky_with_jitter(
        function_covariance,
        initial_jitter=function_jitter,
    )
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

    if geometry.rank == 0:
        solve = CGResult(
            solution=x_condition.new_empty((0,)),
            residual=x_condition.new_empty((0,)),
            iterations=0,
            operator_matvecs=0,
            preconditioner_applications=0,
            relative_residual=0.0,
            residual_norm=0.0,
            rhs_norm=0.0,
            converged=True,
        )
        certificate = PosteriorCertificate(
            variance_error_upper_bound=0.0,
            expected_kl_upper_bound=0.0,
            operator_eigenvalue_lower_bound=math.inf,
            exact_arithmetic_certified=geometry.is_exact,
            solve_certified=True,
            basis_is_exact=geometry.is_exact,
            floating_point_rigorous=False,
        )
        return LocalPrediction(
            mean=torch.dot(function_weights, value_condition),
            variance=conditional_value_variance,
            rank=0,
            basis_is_exact=geometry.is_exact,
            finite_precision_variance_correction=x_condition.new_zeros(()),
            solve=solve,
            certificate=certificate,
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
    # TERA regularizes in its nonorthogonal q coordinates.  Under
    # z = q @ q_to_z, epsilon I_q becomes epsilon q_to_z.T q_to_z.
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
    preconditioner = ReducedKroneckerPreconditioner(operator) if use_preconditioner else None
    solve = solve_reduced_cg(
        operator,
        conditional_cross,
        tolerance=cg_tolerance,
        max_iterations=cg_max_iterations,
        preconditioner=preconditioner,
    )

    projected_observations = gradient_condition @ raw_differences
    orthonormal_observations = projected_observations @ geometry.q_to_z
    q_t_weights = operator.q_t_matmul(solve.solution)
    corrected_function_weights = function_weights - torch.cholesky_solve(
        q_t_weights.unsqueeze(1),
        function_cholesky,
    ).squeeze(1)
    mean = torch.dot(corrected_function_weights, value_condition)
    mean = mean + torch.dot(solve.solution, orthonormal_observations.reshape(-1))

    # In exact arithmetic this energy form is conservative for any stored
    # iterate: v(w)-v(w*) = ||w-w*||_K^2.  It removes dependence on exact
    # Galerkin orthogonality, although the evaluated scalar remains subject to
    # ordinary floating-point roundoff.  For an ideal Galerkin iterate the
    # correction is zero.
    variance_correction = -torch.dot(solve.solution, solve.residual)
    variance = conditional_value_variance - torch.dot(solve.solution, conditional_cross)
    variance = variance + variance_correction
    certificate = compute_posterior_certificate(
        operator,
        solve,
        variance,
        basis_is_exact=geometry.is_exact,
    )
    return LocalPrediction(
        mean=mean,
        variance=variance,
        rank=geometry.rank,
        basis_is_exact=geometry.is_exact,
        finite_precision_variance_correction=variance_correction,
        solve=solve,
        certificate=certificate,
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
    cg_tolerance: float = 1e-6,
    cg_max_iterations: int | None = None,
    use_preconditioner: bool = True,
    function_jitter: float = 1e-8,
    reduced_jitter: float = 1e-8,
) -> MarginalPredictions:
    """Predict independent local marginals using nearest neighbours."""

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
    neighbours = torch.topk(
        torch.cdist(eval_scaled, train_scaled),
        k=count,
        largest=False,
    ).indices

    predictions = []
    for target, indices in zip(x_eval, neighbours, strict=True):
        predictions.append(
            predict_local_value(
                x_train[indices],
                value_train[indices],
                gradient_train[indices],
                target.unsqueeze(0),
                lengthscale=lengthscale,
                outputscale=outputscale,
                value_noise_variance=value_noise_variance,
                gradient_noise_variance=gradient_noise_variance,
                kernel=kernel,
                gradient_noise_model=gradient_noise_model,
                rank=rank,
                relative_rank_tolerance=relative_rank_tolerance,
                cg_tolerance=cg_tolerance,
                cg_max_iterations=cg_max_iterations,
                use_preconditioner=use_preconditioner,
                function_jitter=function_jitter,
                reduced_jitter=reduced_jitter,
            )
        )

    return MarginalPredictions(
        mean=torch.stack([prediction.mean for prediction in predictions]),
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
    )
