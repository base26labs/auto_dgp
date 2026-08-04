"""Precision-aware, direction-budgeted guarded ORBIT prediction.

The method uses a larger local neighbourhood only when its source-precision
numerical rank fits a fixed direction budget.  It evaluates both scalar
conditionals, applies a label-free trust and variance guard, and differentiates
only the selected conditional while reusing its primal solve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from gp.orbit.operator import CGResult
from gp.orbit.predictor import (
    LocalPrediction,
    build_local_value_system,
    differentiate_solved_local_value_system,
    solve_local_value_system,
)


@dataclass(frozen=True)
class BudgetedGuardedMarginals:
    """Predictions plus paired scalar and selected-adjoint diagnostics."""

    mean: torch.Tensor
    variance: torch.Tensor
    mean_gradient: torch.Tensor
    use_expanded: torch.Tensor
    expanded_eligible: torch.Tensor
    variance_is_nested: torch.Tensor
    normalized_mean_shift: torch.Tensor
    selected_m: torch.Tensor
    base_ranks: torch.Tensor
    expanded_ranks: torch.Tensor
    base_iterations: torch.Tensor
    base_operator_matvecs: torch.Tensor
    base_preconditioner_applications: torch.Tensor
    base_relative_residuals: torch.Tensor
    base_converged: torch.Tensor
    expanded_iterations: torch.Tensor
    expanded_operator_matvecs: torch.Tensor
    expanded_preconditioner_applications: torch.Tensor
    expanded_relative_residuals: torch.Tensor
    expanded_converged: torch.Tensor
    selected_adjoint_iterations: torch.Tensor
    selected_adjoint_operator_matvecs: torch.Tensor
    selected_adjoint_preconditioner_applications: torch.Tensor
    selected_adjoint_relative_residuals: torch.Tensor
    selected_adjoint_converged: torch.Tensor


def _validate_inputs(
    x_train: torch.Tensor,
    value_train: torch.Tensor,
    gradient_train: torch.Tensor,
    x_eval: torch.Tensor,
    *,
    base_m: int,
    expanded_m: int,
    maximum_expanded_rank: int,
    trust_radius_sigma: float,
    rank_epsilon: float | torch.Tensor,
) -> torch.Tensor:
    if x_train.ndim != 2 or x_eval.ndim != 2 or x_train.shape[1] != x_eval.shape[1]:
        raise ValueError("x_train and x_eval must be two-dimensional with matching width")
    if value_train.shape != (x_train.shape[0],) or gradient_train.shape != x_train.shape:
        raise ValueError("training value and gradient shapes must match x_train")
    tensors = (x_train, value_train, gradient_train, x_eval)
    if any(item.dtype != x_train.dtype or item.device != x_train.device for item in tensors):
        raise TypeError("all input tensors must use the same dtype and device")
    if not x_train.is_floating_point() or any(
        not bool(torch.isfinite(item).all()) for item in tensors
    ):
        raise ValueError("all inputs must be finite floating tensors")
    if not 0 < base_m < expanded_m <= x_train.shape[0]:
        raise ValueError("neighbour counts must satisfy 0 < base_m < expanded_m <= n_train")
    if not 0 < maximum_expanded_rank <= min(expanded_m, x_train.shape[1]):
        raise ValueError("maximum_expanded_rank exceeds the expanded local geometry")
    if not math.isfinite(trust_radius_sigma) or trust_radius_sigma <= 0.0:
        raise ValueError("trust_radius_sigma must be finite and positive")
    epsilon = torch.as_tensor(rank_epsilon, dtype=x_train.dtype, device=x_train.device)
    if epsilon.numel() != 1 or not bool(torch.isfinite(epsilon)):
        raise ValueError("rank_epsilon must be a finite scalar")
    if not 0.0 < float(epsilon) < 1.0:
        raise ValueError("rank_epsilon must lie in (0, 1)")
    return epsilon.reshape(())


def _solve_record(
    result: LocalPrediction | None,
) -> tuple[int, int, int, float, bool]:
    if result is None:
        return 0, 0, 0, math.nan, True
    solve = result.solve
    return (
        solve.iterations,
        solve.operator_matvecs,
        solve.preconditioner_applications,
        solve.relative_residual,
        solve.converged,
    )


def _adjoint_record(solve: CGResult) -> tuple[int, int, int, float, bool]:
    return (
        solve.iterations,
        solve.operator_matvecs,
        solve.preconditioner_applications,
        solve.relative_residual,
        solve.converged,
    )


def predict_budgeted_guarded_marginals(
    x_train: torch.Tensor,
    value_train: torch.Tensor,
    gradient_train: torch.Tensor,
    x_eval: torch.Tensor,
    *,
    base_m: int,
    expanded_m: int,
    maximum_expanded_rank: int,
    trust_radius_sigma: float,
    rank_epsilon: float | torch.Tensor,
    lengthscale: torch.Tensor,
    outputscale: float | torch.Tensor,
    value_noise_variance: float | torch.Tensor,
    gradient_noise_variance: float | torch.Tensor,
    kernel: str,
    gradient_noise_model: str = "iid",
    cg_tolerance: float = 1e-6,
    cg_max_iterations: int | None = None,
    use_preconditioner: bool = True,
    function_jitter: float = 1e-8,
    reduced_jitter: float = 1e-8,
    variance_roundoff_multiplier: float = 128.0,
) -> BudgetedGuardedMarginals:
    """Predict with a source-precision rank gate and one selected adjoint.

    Neighbour membership, rank eligibility, and the guard branch are treated as
    piecewise constant.  The expanded primal solve is skipped when its natural
    source-precision rank exceeds ``maximum_expanded_rank``.
    """

    epsilon = _validate_inputs(
        x_train,
        value_train,
        gradient_train,
        x_eval,
        base_m=base_m,
        expanded_m=expanded_m,
        maximum_expanded_rank=maximum_expanded_rank,
        trust_radius_sigma=trust_radius_sigma,
        rank_epsilon=rank_epsilon,
    )
    if not math.isfinite(variance_roundoff_multiplier) or variance_roundoff_multiplier <= 0.0:
        raise ValueError("variance_roundoff_multiplier must be finite and positive")
    lengthscale = torch.as_tensor(
        lengthscale,
        dtype=x_train.dtype,
        device=x_train.device,
    ).reshape(-1)
    if lengthscale.numel() not in {1, x_train.shape[1]}:
        raise ValueError("lengthscale must be scalar or match the input dimension")
    if not bool(torch.isfinite(lengthscale).all()) or bool((lengthscale <= 0.0).any()):
        raise ValueError("lengthscale must be finite and positive")
    if lengthscale.numel() == 1:
        train_scaled = x_train / lengthscale.reshape(1, 1)
        eval_scaled = x_eval / lengthscale.reshape(1, 1)
    else:
        train_scaled = x_train / lengthscale.reshape(1, -1)
        eval_scaled = x_eval / lengthscale.reshape(1, -1)
    neighbours = torch.topk(
        torch.cdist(eval_scaled, train_scaled),
        k=expanded_m,
        largest=False,
        sorted=True,
    ).indices

    predictions: list[LocalPrediction] = []
    gradients: list[torch.Tensor] = []
    uses: list[bool] = []
    eligibilities: list[bool] = []
    nesting: list[bool] = []
    shifts: list[float] = []
    base_ranks: list[int] = []
    expanded_ranks: list[int] = []
    base_records: list[tuple[int, int, int, float, bool]] = []
    expanded_records: list[tuple[int, int, int, float, bool]] = []
    adjoint_records: list[tuple[int, int, int, float, bool]] = []

    common = {
        "lengthscale": lengthscale,
        "outputscale": outputscale,
        "value_noise_variance": value_noise_variance,
        "gradient_noise_variance": gradient_noise_variance,
        "kernel": kernel,
        "gradient_noise_model": gradient_noise_model,
        "function_jitter": function_jitter,
        "reduced_jitter": reduced_jitter,
        "build_preconditioner": use_preconditioner,
    }
    for target, expanded_indices in zip(x_eval, neighbours, strict=True):
        base_indices = expanded_indices[:base_m]
        with torch.enable_grad():
            differentiable_target = target.detach().clone().unsqueeze(0).requires_grad_(True)
            base_system = build_local_value_system(
                x_train[base_indices],
                value_train[base_indices],
                gradient_train[base_indices],
                differentiable_target,
                **common,
            )
            expanded_system = build_local_value_system(
                x_train[expanded_indices],
                value_train[expanded_indices],
                gradient_train[expanded_indices],
                differentiable_target,
                rank_epsilon=epsilon,
                **common,
            )
            eligible = expanded_system.geometry.rank <= maximum_expanded_rank
            with torch.no_grad():
                base_prediction = solve_local_value_system(
                    base_system,
                    tolerance=cg_tolerance,
                    max_iterations=cg_max_iterations,
                    use_preconditioner=use_preconditioner,
                )
                expanded_prediction = None
                if eligible:
                    expanded_prediction = solve_local_value_system(
                        expanded_system,
                        tolerance=cg_tolerance,
                        max_iterations=cg_max_iterations,
                        use_preconditioner=use_preconditioner,
                    )

                use_expanded = False
                variance_is_nested = False
                normalized_shift = math.inf
                if expanded_prediction is not None:
                    variance_scale = torch.maximum(
                        base_prediction.variance.abs(),
                        expanded_prediction.variance.abs(),
                    ).clamp_min(1.0)
                    roundoff_floor = (
                        variance_roundoff_multiplier
                        * torch.finfo(variance_scale.dtype).eps
                        * variance_scale
                    )
                    variance_is_nested = bool(
                        expanded_prediction.variance
                        <= base_prediction.variance + roundoff_floor
                    )
                    normalized_shift = float(
                        (expanded_prediction.mean - base_prediction.mean).abs()
                        / torch.sqrt(base_prediction.variance)
                    )
                    use_expanded = (
                        variance_is_nested and normalized_shift <= trust_radius_sigma
                    )

            selected_system = expanded_system if use_expanded else base_system
            selected_prediction = expanded_prediction if use_expanded else base_prediction
            if selected_prediction is None:  # pragma: no cover - exhaustive branch guard
                raise RuntimeError("selected prediction is unavailable")
            differentiated = differentiate_solved_local_value_system(
                selected_system,
                selected_prediction,
                differentiable_target,
                cg_tolerance=cg_tolerance,
                cg_max_iterations=cg_max_iterations,
                use_preconditioner=use_preconditioner,
            )

        predictions.append(selected_prediction)
        gradients.append(differentiated.mean_gradient)
        uses.append(use_expanded)
        eligibilities.append(eligible)
        nesting.append(variance_is_nested)
        shifts.append(normalized_shift)
        base_ranks.append(base_system.geometry.rank)
        expanded_ranks.append(expanded_system.geometry.rank)
        base_records.append(_solve_record(base_prediction))
        expanded_records.append(_solve_record(expanded_prediction))
        adjoint_records.append(_adjoint_record(differentiated.adjoint_solve))

    device = x_train.device
    integer = torch.long
    real = x_train.dtype
    base_record_tensor = torch.tensor(base_records, device=device)
    expanded_record_tensor = torch.tensor(expanded_records, device=device)
    adjoint_record_tensor = torch.tensor(adjoint_records, device=device)
    return BudgetedGuardedMarginals(
        mean=torch.stack([item.mean for item in predictions]),
        variance=torch.stack([item.variance for item in predictions]),
        mean_gradient=torch.stack(gradients),
        use_expanded=torch.tensor(uses, device=device, dtype=torch.bool),
        expanded_eligible=torch.tensor(eligibilities, device=device, dtype=torch.bool),
        variance_is_nested=torch.tensor(nesting, device=device, dtype=torch.bool),
        normalized_mean_shift=torch.tensor(shifts, device=device, dtype=real),
        selected_m=torch.tensor(
            [expanded_m if use else base_m for use in uses],
            device=device,
            dtype=integer,
        ),
        base_ranks=torch.tensor(base_ranks, device=device, dtype=integer),
        expanded_ranks=torch.tensor(expanded_ranks, device=device, dtype=integer),
        base_iterations=base_record_tensor[:, 0].to(integer),
        base_operator_matvecs=base_record_tensor[:, 1].to(integer),
        base_preconditioner_applications=base_record_tensor[:, 2].to(integer),
        base_relative_residuals=base_record_tensor[:, 3].to(real),
        base_converged=base_record_tensor[:, 4].to(torch.bool),
        expanded_iterations=expanded_record_tensor[:, 0].to(integer),
        expanded_operator_matvecs=expanded_record_tensor[:, 1].to(integer),
        expanded_preconditioner_applications=expanded_record_tensor[:, 2].to(integer),
        expanded_relative_residuals=expanded_record_tensor[:, 3].to(real),
        expanded_converged=expanded_record_tensor[:, 4].to(torch.bool),
        selected_adjoint_iterations=adjoint_record_tensor[:, 0].to(integer),
        selected_adjoint_operator_matvecs=adjoint_record_tensor[:, 1].to(integer),
        selected_adjoint_preconditioner_applications=adjoint_record_tensor[:, 2].to(integer),
        selected_adjoint_relative_residuals=adjoint_record_tensor[:, 3].to(real),
        selected_adjoint_converged=adjoint_record_tensor[:, 4].to(torch.bool),
    )
