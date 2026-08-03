"""Independent dense support-space oracle for one F02 TERA conditional.

The oracle is diagnostic, not a production predictor.  It deliberately does
not import ORBIT's operator or iterative solver.  Instead it constructs the
full ``(m*r) x (m*r)`` conditional covariance in an explicit rank-revealing
support basis and solves it once with a dense Cholesky factorization.

All numeric inputs are first rounded to the released float32 representation
and only then promoted to float64.  The support rank is selected with the fixed
source-precision rule

    singular_value > largest * max(d, m) * eps(float32).

This makes the oracle a stable solve of the information present in the
released input tensors, rather than a subtly different full-float64 dataset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

SOURCE_DTYPE = torch.float32
COMPUTE_DTYPE = torch.float64
RANK_RULE_NAME = "source-fp32-smax-maxshape-eps-v1"
RANK_TOLERANCE_MULTIPLIER = 1.0
MAX_SUPPORT_RELATIVE_SOLVE_RESIDUAL = 1e-10
_SQRT5 = math.sqrt(5.0)


class DenseSupportOracleError(RuntimeError):
    """Raised when the dense oracle cannot certify a finite positive result."""


@dataclass(frozen=True, slots=True)
class DenseSupportDiagnostics:
    """Rank, spectrum, jitter, and direct-solve evidence for one target."""

    source_quantization_dtype: str
    compute_dtype: str
    rank_rule_name: str
    rank_tolerance_multiplier: float
    source_machine_epsilon: float
    rank_threshold: float
    numerical_rank: int
    maximum_rank: int
    singular_values: tuple[float, ...]
    discarded_singular_value_energy: float
    support_system_dimension: int
    function_jitter_requested: float
    function_jitter_used: float
    function_cholesky_attempts: int
    function_spectrum_before_jitter: tuple[float, ...]
    function_spectrum_after_jitter: tuple[float, ...]
    support_coordinate_jitter: float
    support_coordinate_jitter_spectrum: tuple[float, ...]
    support_spectrum_before_jitter: tuple[float, ...]
    support_spectrum_after_jitter: tuple[float, ...]
    support_condition_number_after_jitter: float
    support_cholesky_attempts: int
    support_relative_solve_residual: float
    support_relative_solve_residual_tolerance: float


@dataclass(frozen=True, slots=True)
class DenseSupportPrediction:
    """One scalar prediction and its independently materialized diagnostics.

    ``support_basis`` is the retained ambient left-singular basis ``U_r``;
    ``support_coordinates`` is ``V_r S_r`` in TERA's ``m`` q coordinates;
    and ``tera_to_support`` is ``V_r S_r^-1``.
    """

    mean: torch.Tensor
    latent_variance: torch.Tensor
    value_only_conditional_variance: torch.Tensor
    gradient_variance_reduction: torch.Tensor
    support_basis: torch.Tensor
    support_coordinates: torch.Tensor
    tera_to_support: torch.Tensor
    diagnostics: DenseSupportDiagnostics


def _quantized_float64(value: Any, *, device: torch.device) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, device=device).detach()
    except (TypeError, ValueError) as error:
        raise DenseSupportOracleError("numeric input cannot be converted to a tensor") from error
    if tensor.is_complex():
        raise DenseSupportOracleError("complex numeric inputs are not supported")
    tensor = tensor.to(dtype=SOURCE_DTYPE).to(dtype=COMPUTE_DTYPE)
    if not bool(torch.isfinite(tensor).all().item()):
        raise DenseSupportOracleError(
            "all numeric inputs must remain finite after fp32 quantization"
        )
    return tensor


def _positive_scalar(value: Any, label: str, *, device: torch.device) -> torch.Tensor:
    tensor = _quantized_float64(value, device=device)
    if tensor.numel() != 1 or float(tensor) <= 0.0:
        raise DenseSupportOracleError(f"{label} must be a finite positive scalar")
    return tensor.reshape(())


def _nonnegative_scalar(value: Any, label: str, *, device: torch.device) -> torch.Tensor:
    tensor = _quantized_float64(value, device=device)
    if tensor.numel() != 1 or float(tensor) < 0.0:
        raise DenseSupportOracleError(f"{label} must be a finite non-negative scalar")
    return tensor.reshape(())


def _spectrum(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.numel() == 0:
        return matrix.new_empty((0,))
    symmetric = 0.5 * (matrix + matrix.T)
    values = torch.linalg.eigvalsh(symmetric)
    if not bool(torch.isfinite(values).all().item()):
        raise DenseSupportOracleError("eigenspectrum contains a nonfinite value")
    return values


def _as_float_tuple(values: torch.Tensor) -> tuple[float, ...]:
    return tuple(float(value) for value in values.detach().cpu().tolist())


def _function_covariance(
    first: torch.Tensor,
    second: torch.Tensor,
    lengthscale: torch.Tensor,
    outputscale: torch.Tensor,
    kernel: str,
) -> torch.Tensor:
    difference = first[:, None, :] - second[None, :, :]
    if lengthscale.numel() == 1:
        difference = difference / lengthscale.reshape(1, 1, 1)
    else:
        difference = difference / lengthscale.reshape(1, 1, -1)
    distance2 = (difference * difference).sum(dim=-1).clamp_min(0.0)
    if kernel == "rbf":
        return outputscale * torch.exp(-0.5 * distance2)
    if kernel == "matern52":
        radius = torch.sqrt(distance2.clamp_min(1e-12))
        scaled = _SQRT5 * radius
        return outputscale * (1.0 + scaled + scaled * scaled / 3.0) * torch.exp(-scaled)
    raise DenseSupportOracleError("kernel must be 'rbf' or 'matern52'")


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
    raise DenseSupportOracleError("kernel must be 'rbf' or 'matern52'")


def _projected_noise_gram(
    raw_differences: torch.Tensor,
    lengthscale: torch.Tensor,
    model: str,
) -> torch.Tensor:
    if model == "iid":
        gram = raw_differences.T @ raw_differences
    elif model == "scaled":
        lengthscale2 = lengthscale * lengthscale
        if lengthscale2.numel() == 1:
            weighted = raw_differences * lengthscale2.reshape(1, 1)
        else:
            weighted = raw_differences * lengthscale2.reshape(-1, 1)
        gram = raw_differences.T @ weighted
    else:
        raise DenseSupportOracleError("gradient_noise_model must be 'iid' or 'scaled'")
    return 0.5 * (gram + gram.T)


def _cholesky_with_recorded_jitter(
    matrix: torch.Tensor,
    *,
    requested_jitter: float,
    maximum_jitter: float,
) -> tuple[torch.Tensor, float, int, torch.Tensor, torch.Tensor]:
    if requested_jitter < 0.0 or maximum_jitter < requested_jitter:
        raise DenseSupportOracleError("function jitter bounds are invalid")
    before = _spectrum(matrix)
    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    jitter = requested_jitter
    attempts = 0
    while jitter <= maximum_jitter:
        attempts += 1
        candidate = matrix + matrix.new_tensor(jitter) * identity
        factor, info = torch.linalg.cholesky_ex(candidate)
        if int(info.max().item()) == 0:
            return factor, jitter, attempts, before, _spectrum(candidate)
        jitter = float(torch.finfo(matrix.dtype).eps) if jitter == 0.0 else 10.0 * jitter
    raise DenseSupportOracleError(
        f"function covariance Cholesky failed through jitter={maximum_jitter}"
    )


def _validate_and_quantize_inputs(
    x_condition: torch.Tensor,
    value_condition: torch.Tensor,
    gradient_condition: torch.Tensor,
    x_target: torch.Tensor,
    *,
    lengthscale: torch.Tensor,
    outputscale: float | torch.Tensor,
    value_noise_variance: float | torch.Tensor,
    gradient_noise_variance: float | torch.Tensor,
    function_jitter: float,
    support_coordinate_jitter: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
    float,
]:
    if not isinstance(x_condition, torch.Tensor) or x_condition.ndim != 2:
        raise DenseSupportOracleError("x_condition must be a two-dimensional tensor")
    if x_condition.shape[0] == 0 or x_condition.shape[1] == 0:
        raise DenseSupportOracleError("x_condition must have nonzero m and dimension")
    m, dimension = x_condition.shape
    if value_condition.shape != (m,):
        raise DenseSupportOracleError("value_condition must have shape (m,)")
    if gradient_condition.shape != (m, dimension):
        raise DenseSupportOracleError("gradient_condition must have shape (m, d)")
    if x_target.shape != (1, dimension):
        raise DenseSupportOracleError("x_target must have shape (1, d)")
    device = x_condition.device
    if any(
        value.device != device
        for value in (value_condition, gradient_condition, x_target)
        if isinstance(value, torch.Tensor)
    ):
        raise DenseSupportOracleError("all observation tensors must use one device")

    x64 = _quantized_float64(x_condition, device=device)
    values64 = _quantized_float64(value_condition, device=device)
    gradients64 = _quantized_float64(gradient_condition, device=device)
    target64 = _quantized_float64(x_target, device=device)
    lengthscale64 = _quantized_float64(lengthscale, device=device).reshape(-1)
    if lengthscale64.numel() not in {1, dimension} or bool((lengthscale64 <= 0.0).any()):
        raise DenseSupportOracleError("lengthscale must contain one or d positive entries")
    outputscale64 = _positive_scalar(outputscale, "outputscale", device=device)
    value_noise64 = _nonnegative_scalar(
        value_noise_variance,
        "value_noise_variance",
        device=device,
    )
    gradient_noise64 = _nonnegative_scalar(
        gradient_noise_variance,
        "gradient_noise_variance",
        device=device,
    )
    function_jitter64 = _nonnegative_scalar(
        function_jitter,
        "function_jitter",
        device=device,
    )
    support_jitter64 = _positive_scalar(
        support_coordinate_jitter,
        "support_coordinate_jitter",
        device=device,
    )
    return (
        x64,
        values64,
        gradients64,
        target64,
        lengthscale64,
        outputscale64,
        value_noise64,
        gradient_noise64,
        float(function_jitter64),
        float(support_jitter64),
    )


def predict_local_dense_support(
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
    function_jitter: float = 1e-8,
    maximum_function_jitter: float = 1e-1,
    support_coordinate_jitter: float = 1e-8,
) -> DenseSupportPrediction:
    """Evaluate one TERA local conditional via an explicit dense support solve.

    The support-coordinate jitter is the image of TERA's fixed ``epsilon I``
    under the rank-revealing change of coordinates.  It is never adaptively
    increased: a failed support Cholesky is a failed diagnostic, not a silently
    changed posterior.
    """

    if kernel not in {"rbf", "matern52"}:
        raise DenseSupportOracleError("kernel must be 'rbf' or 'matern52'")
    if gradient_noise_model not in {"iid", "scaled"}:
        raise DenseSupportOracleError("gradient_noise_model must be 'iid' or 'scaled'")
    (
        x64,
        values64,
        gradients64,
        target64,
        lengthscale64,
        outputscale64,
        value_noise64,
        gradient_noise64,
        function_jitter64,
        support_jitter64,
    ) = _validate_and_quantize_inputs(
        x_condition,
        value_condition,
        gradient_condition,
        x_target,
        lengthscale=lengthscale,
        outputscale=outputscale,
        value_noise_variance=value_noise_variance,
        gradient_noise_variance=gradient_noise_variance,
        function_jitter=function_jitter,
        support_coordinate_jitter=support_coordinate_jitter,
    )
    maximum_function_jitter64 = float(
        _nonnegative_scalar(
            maximum_function_jitter,
            "maximum_function_jitter",
            device=x64.device,
        )
    )
    m, dimension = x64.shape
    raw_differences = (x64 - target64).T.contiguous()
    if lengthscale64.numel() == 1:
        scaled_differences = raw_differences / lengthscale64.reshape(1, 1)
    else:
        scaled_differences = raw_differences / lengthscale64.reshape(-1, 1)

    left, singular_values, right_transpose = torch.linalg.svd(
        scaled_differences,
        full_matrices=False,
    )
    maximum_rank = min(dimension, m)
    largest = singular_values[0]
    source_epsilon = float(torch.finfo(SOURCE_DTYPE).eps)
    rank_threshold_tensor = largest * max(dimension, m) * source_epsilon * RANK_TOLERANCE_MULTIPLIER
    keep = singular_values > rank_threshold_tensor
    numerical_rank = int(keep.sum().item())
    retained_left = left[:, keep]
    retained_singular = singular_values[keep]
    retained_right = right_transpose[keep].T
    coordinates = retained_right * retained_singular.unsqueeze(0)
    tera_to_support = retained_right / retained_singular.unsqueeze(0)
    discarded_energy = float((singular_values[~keep] ** 2).sum())

    function_covariance = _function_covariance(
        x64,
        x64,
        lengthscale64,
        outputscale64,
        kernel,
    )
    function_covariance = 0.5 * (function_covariance + function_covariance.T)
    function_covariance = function_covariance + value_noise64 * torch.eye(
        m,
        dtype=COMPUTE_DTYPE,
        device=x64.device,
    )
    (
        function_factor,
        function_jitter_used,
        function_attempts,
        function_spectrum_before,
        function_spectrum_after,
    ) = _cholesky_with_recorded_jitter(
        function_covariance,
        requested_jitter=function_jitter64,
        maximum_jitter=maximum_function_jitter64,
    )
    target_function_covariance = _function_covariance(
        x64,
        target64,
        lengthscale64,
        outputscale64,
        kernel,
    ).squeeze(1)
    function_weights = torch.cholesky_solve(
        target_function_covariance.unsqueeze(1),
        function_factor,
    ).squeeze(1)
    value_only_variance = outputscale64 - torch.dot(
        target_function_covariance,
        function_weights,
    )

    if numerical_rank == 0:
        mean = torch.dot(function_weights, values64)
        if (
            not bool(torch.isfinite(mean).item())
            or not bool(torch.isfinite(value_only_variance).item())
            or float(value_only_variance) <= 0.0
        ):
            raise DenseSupportOracleError("rank-zero oracle produced invalid predictive moments")
        diagnostics = DenseSupportDiagnostics(
            source_quantization_dtype="float32",
            compute_dtype="float64",
            rank_rule_name=RANK_RULE_NAME,
            rank_tolerance_multiplier=RANK_TOLERANCE_MULTIPLIER,
            source_machine_epsilon=source_epsilon,
            rank_threshold=float(rank_threshold_tensor),
            numerical_rank=0,
            maximum_rank=maximum_rank,
            singular_values=_as_float_tuple(singular_values),
            discarded_singular_value_energy=discarded_energy,
            support_system_dimension=0,
            function_jitter_requested=function_jitter64,
            function_jitter_used=function_jitter_used,
            function_cholesky_attempts=function_attempts,
            function_spectrum_before_jitter=_as_float_tuple(function_spectrum_before),
            function_spectrum_after_jitter=_as_float_tuple(function_spectrum_after),
            support_coordinate_jitter=support_jitter64,
            support_coordinate_jitter_spectrum=(),
            support_spectrum_before_jitter=(),
            support_spectrum_after_jitter=(),
            support_condition_number_after_jitter=1.0,
            support_cholesky_attempts=0,
            support_relative_solve_residual=0.0,
            support_relative_solve_residual_tolerance=MAX_SUPPORT_RELATIVE_SOLVE_RESIDUAL,
        )
        return DenseSupportPrediction(
            mean=mean,
            latent_variance=value_only_variance,
            value_only_conditional_variance=value_only_variance,
            gradient_variance_reduction=value_only_variance.new_zeros(()),
            support_basis=retained_left,
            support_coordinates=coordinates,
            tera_to_support=tera_to_support,
            diagnostics=diagnostics,
        )

    gram = scaled_differences.T @ scaled_differences
    diagonal = torch.diagonal(gram)
    pair_distance2 = (diagonal[:, None] + diagonal[None, :] - 2.0 * gram).clamp_min(0.0)
    pair_alpha, pair_beta = _gradient_scalars(
        pair_distance2,
        outputscale64,
        kernel,
    )
    target_alpha, _ = _gradient_scalars(diagonal, outputscale64, kernel)

    pair_coordinates = coordinates[:, None, :] - coordinates[None, :, :]
    identity_rank = torch.eye(
        numerical_rank,
        dtype=COMPUTE_DTYPE,
        device=x64.device,
    )
    blocks = pair_alpha[:, :, None, None] * identity_rank
    blocks = blocks + pair_beta[:, :, None, None] * (
        pair_coordinates[:, :, :, None] * pair_coordinates[:, :, None, :]
    )
    noise_gram = _projected_noise_gram(
        raw_differences,
        lengthscale64,
        gradient_noise_model,
    )
    support_noise = gradient_noise64 * (tera_to_support.T @ noise_gram @ tera_to_support)
    support_noise = 0.5 * (support_noise + support_noise.T)
    diagonal_blocks = torch.arange(m, device=x64.device)
    blocks[diagonal_blocks, diagonal_blocks] = (
        blocks[diagonal_blocks, diagonal_blocks] + support_noise
    )
    unconditional = blocks.permute(0, 2, 1, 3).reshape(
        m * numerical_rank,
        m * numerical_rank,
    )

    value_cross = (
        (-pair_alpha[:, :, None] * pair_coordinates).permute(0, 2, 1).reshape(m * numerical_rank, m)
    )
    function_inverse_cross = torch.cholesky_solve(
        value_cross.T,
        function_factor,
    )
    support_covariance = unconditional - value_cross @ function_inverse_cross
    support_covariance = 0.5 * (support_covariance + support_covariance.T)

    support_jitter_metric = support_jitter64 * (tera_to_support.T @ tera_to_support)
    support_jitter_metric = 0.5 * (support_jitter_metric + support_jitter_metric.T)
    support_jitter_matrix = torch.kron(
        torch.eye(m, dtype=COMPUTE_DTYPE, device=x64.device),
        support_jitter_metric.contiguous(),
    )
    jittered_support_covariance = support_covariance + support_jitter_matrix
    spectrum_before = _spectrum(support_covariance)
    spectrum_after = _spectrum(jittered_support_covariance)
    factor, info = torch.linalg.cholesky_ex(jittered_support_covariance)
    if int(info.max().item()) != 0:
        raise DenseSupportOracleError(
            "support covariance failed its single fixed-jitter Cholesky attempt"
        )
    if float(spectrum_after[0]) <= 0.0:
        raise DenseSupportOracleError("jittered support covariance is not positive definite")

    conditional_cross = (-target_alpha[:, None] * coordinates).reshape(
        -1
    ) - value_cross @ function_weights
    support_weights = torch.cholesky_solve(
        conditional_cross.unsqueeze(1),
        factor,
    ).squeeze(1)
    projected_observations = gradients64 @ raw_differences
    support_observations = projected_observations @ tera_to_support
    corrected_function_weights = function_weights - torch.cholesky_solve(
        (value_cross.T @ support_weights).unsqueeze(1),
        function_factor,
    ).squeeze(1)
    mean = torch.dot(corrected_function_weights, values64)
    mean = mean + torch.dot(support_weights, support_observations.reshape(-1))
    gradient_reduction = torch.dot(support_weights, conditional_cross)
    latent_variance = value_only_variance - gradient_reduction

    residual = conditional_cross - jittered_support_covariance @ support_weights
    rhs_norm = torch.linalg.norm(conditional_cross)
    relative_residual = float(
        torch.linalg.norm(residual) / rhs_norm.clamp_min(torch.finfo(COMPUTE_DTYPE).tiny)
    )
    if (
        not bool(torch.isfinite(mean).item())
        or not bool(torch.isfinite(latent_variance).item())
        or not bool(torch.isfinite(value_only_variance).item())
        or not bool(torch.isfinite(gradient_reduction).item())
        or float(latent_variance) <= 0.0
        or float(value_only_variance) <= 0.0
        or float(gradient_reduction) < -100.0 * torch.finfo(COMPUTE_DTYPE).eps
        or not math.isfinite(relative_residual)
        or relative_residual > MAX_SUPPORT_RELATIVE_SOLVE_RESIDUAL
    ):
        raise DenseSupportOracleError("dense support solve produced invalid predictive diagnostics")

    jitter_spectrum = _spectrum(support_jitter_metric)
    condition_number = float(spectrum_after[-1] / spectrum_after[0])
    if not math.isfinite(condition_number):
        raise DenseSupportOracleError("jittered support condition number is nonfinite")
    diagnostics = DenseSupportDiagnostics(
        source_quantization_dtype="float32",
        compute_dtype="float64",
        rank_rule_name=RANK_RULE_NAME,
        rank_tolerance_multiplier=RANK_TOLERANCE_MULTIPLIER,
        source_machine_epsilon=source_epsilon,
        rank_threshold=float(rank_threshold_tensor),
        numerical_rank=numerical_rank,
        maximum_rank=maximum_rank,
        singular_values=_as_float_tuple(singular_values),
        discarded_singular_value_energy=discarded_energy,
        support_system_dimension=m * numerical_rank,
        function_jitter_requested=function_jitter64,
        function_jitter_used=function_jitter_used,
        function_cholesky_attempts=function_attempts,
        function_spectrum_before_jitter=_as_float_tuple(function_spectrum_before),
        function_spectrum_after_jitter=_as_float_tuple(function_spectrum_after),
        support_coordinate_jitter=support_jitter64,
        support_coordinate_jitter_spectrum=_as_float_tuple(jitter_spectrum),
        support_spectrum_before_jitter=_as_float_tuple(spectrum_before),
        support_spectrum_after_jitter=_as_float_tuple(spectrum_after),
        support_condition_number_after_jitter=condition_number,
        support_cholesky_attempts=1,
        support_relative_solve_residual=relative_residual,
        support_relative_solve_residual_tolerance=MAX_SUPPORT_RELATIVE_SOLVE_RESIDUAL,
    )
    return DenseSupportPrediction(
        mean=mean,
        latent_variance=latent_variance,
        value_only_conditional_variance=value_only_variance,
        gradient_variance_reduction=gradient_reduction,
        support_basis=retained_left,
        support_coordinates=coordinates,
        tera_to_support=tera_to_support,
        diagnostics=diagnostics,
    )
