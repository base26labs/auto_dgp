"""Spectral-bound contracts for reusable ORBIT local-value systems."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from gp.orbit import (
    build_local_value_system,
    compute_posterior_certificate,
    solve_local_value_system,
)

_KERNELS = ("rbf", "matern52")
_LENGTHSCALES = ("scalar", "ard")
_NOISE_MODELS = ("iid", "scaled", "metric_matched")
_RANKS = ("full", "truncated")


def _build_system(
    *,
    kernel: str,
    lengthscale_kind: str,
    gradient_noise_model: str,
    rank_kind: str,
    gradient_noise_variance: float = 0.04,
    reduced_jitter: float = 2e-9,
):
    dtype = torch.float64
    m, dimension = 4, 5
    seed = (
        2401
        + 101 * _KERNELS.index(kernel)
        + 29 * _LENGTHSCALES.index(lengthscale_kind)
        + 7 * _NOISE_MODELS.index(gradient_noise_model)
        + _RANKS.index(rank_kind)
    )
    generator = torch.Generator().manual_seed(seed)
    x_target = torch.randn(1, dimension, generator=generator, dtype=dtype)
    x_condition = x_target + torch.randn(
        m,
        dimension,
        generator=generator,
        dtype=dtype,
    )
    values = torch.randn(m, generator=generator, dtype=dtype)
    gradients = torch.randn(m, dimension, generator=generator, dtype=dtype)
    lengthscale = (
        torch.tensor([1.15], dtype=dtype)
        if lengthscale_kind == "scalar"
        else torch.linspace(0.65, 1.45, dimension, dtype=dtype)
    )
    rank = None if rank_kind == "full" else 2
    return build_local_value_system(
        x_condition,
        values,
        gradients,
        x_target,
        lengthscale=lengthscale,
        outputscale=1.3,
        value_noise_variance=0.07,
        gradient_noise_variance=gradient_noise_variance,
        kernel=kernel,
        gradient_noise_model=gradient_noise_model,
        rank=rank,
        function_jitter=3e-10,
        reduced_jitter=reduced_jitter,
    )


@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("lengthscale_kind", _LENGTHSCALES)
@pytest.mark.parametrize("gradient_noise_model", _NOISE_MODELS)
@pytest.mark.parametrize("rank_kind", _RANKS)
def test_builder_bounds_enclose_dense_reduced_operator(
    kernel: str,
    lengthscale_kind: str,
    gradient_noise_model: str,
    rank_kind: str,
) -> None:
    system = _build_system(
        kernel=kernel,
        lengthscale_kind=lengthscale_kind,
        gradient_noise_model=gradient_noise_model,
        rank_kind=rank_kind,
    )

    assert system.operator is not None
    dense = system.operator.dense()
    minimum_eigenvalue = float(torch.linalg.eigvalsh(dense).min())
    operator_norm = float(torch.linalg.matrix_norm(dense, ord=2))
    rounding_margin = 2e-12 * max(1.0, operator_norm)

    assert system.geometry.rank == (4 if rank_kind == "full" else 2)
    assert system.geometry.is_exact is (rank_kind == "full")
    assert system.operator_eigenvalue_lower_bound > 0.0
    assert system.operator_eigenvalue_lower_bound <= minimum_eigenvalue + rounding_margin
    assert system.operator_norm_upper_bound is not None
    assert system.operator_norm_upper_bound + rounding_margin >= operator_norm

    assert system.operator_lower_bound_provenance.startswith("trusted_gp_builder:")
    assert "source_dtype_not_directed_rounding" in system.operator_lower_bound_provenance
    assert system.operator_norm_upper_bound_provenance.startswith("trusted_gp_builder:")
    assert "source_dtype_not_directed_rounding" in system.operator_norm_upper_bound_provenance
    assert "unavailable" not in system.operator_norm_upper_bound_provenance

    prediction = solve_local_value_system(
        system,
        tolerance=1e-11,
        max_iterations=100,
        use_preconditioner=True,
    )
    assert prediction.solve.operator_norm_upper_bound == system.operator_norm_upper_bound
    assert (
        prediction.certificate.operator_eigenvalue_lower_bound
        == system.operator_eigenvalue_lower_bound
    )
    assert (
        prediction.certificate.operator_lower_bound_provenance
        == system.operator_lower_bound_provenance
    )
    assert prediction.certificate.bound_scope == "selected_support_represented_system"
    assert not prediction.certificate.floating_point_rigorous


@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("gradient_noise_model", _NOISE_MODELS)
def test_zero_noise_floor_makes_mean_certificate_fail_closed(
    kernel: str,
    gradient_noise_model: str,
) -> None:
    system = _build_system(
        kernel=kernel,
        lengthscale_kind="ard",
        gradient_noise_model=gradient_noise_model,
        rank_kind="truncated",
        gradient_noise_variance=0.0,
        reduced_jitter=0.0,
    )
    prediction = solve_local_value_system(
        system,
        tolerance=1e-12,
        max_iterations=100,
        use_preconditioner=True,
    )

    assert system.operator_eigenvalue_lower_bound == 0.0
    assert prediction.certificate.operator_eigenvalue_lower_bound == 0.0
    assert math.isinf(prediction.certificate.mean_error_upper_bound)
    assert not prediction.certificate.mean_solve_certified
    assert math.isinf(prediction.certificate.variance_error_upper_bound)
    assert math.isinf(prediction.certificate.expected_kl_upper_bound)
    assert not prediction.certificate.solve_certified
    assert not prediction.certificate.exact_arithmetic_certified
    assert not prediction.certificate.floating_point_rigorous
    assert (
        prediction.certificate.operator_lower_bound_provenance
        == system.operator_lower_bound_provenance
    )
    assert "source_dtype_not_directed_rounding" in system.operator_lower_bound_provenance


def test_certificate_rejects_infinite_claimed_lower_bound() -> None:
    system = _build_system(
        kernel="rbf",
        lengthscale_kind="scalar",
        gradient_noise_model="iid",
        rank_kind="full",
    )
    prediction = solve_local_value_system(
        system,
        tolerance=1e-12,
        max_iterations=1,
        use_preconditioner=False,
    )

    with pytest.raises(ValueError, match="finite and non-negative"):
        compute_posterior_certificate(
            system.operator,
            prediction.solve,
            prediction.variance,
            operator_eigenvalue_lower_bound=math.inf,
        )


def test_certificate_fails_closed_without_fresh_residual_evidence() -> None:
    system = _build_system(
        kernel="matern52",
        lengthscale_kind="ard",
        gradient_noise_model="metric_matched",
        rank_kind="truncated",
    )
    prediction = solve_local_value_system(
        system,
        tolerance=1e-12,
        max_iterations=1,
        use_preconditioner=False,
    )
    claimed_recursive = replace(
        prediction.solve,
        residual_is_fresh=False,
        operator_action=None,
        residual_norm=0.0,
    )
    certificate = compute_posterior_certificate(
        system.operator,
        claimed_recursive,
        prediction.variance,
        conditional_observation_norm=float(
            torch.linalg.norm(system.conditional_observation_functional)
        ),
        operator_eigenvalue_lower_bound=system.operator_eigenvalue_lower_bound,
    )

    assert math.isinf(certificate.mean_error_upper_bound)
    assert math.isinf(certificate.variance_error_upper_bound)
    assert not certificate.mean_solve_certified
    assert not certificate.solve_certified
    assert not certificate.exact_arithmetic_certified


def test_certificate_fails_closed_when_residual_is_not_bound_to_action() -> None:
    system = _build_system(
        kernel="rbf",
        lengthscale_kind="ard",
        gradient_noise_model="iid",
        rank_kind="truncated",
    )
    prediction = solve_local_value_system(
        system,
        tolerance=1e-12,
        max_iterations=1,
        use_preconditioner=False,
    )
    mismatched = replace(
        prediction.solve,
        residual=prediction.solve.residual + 1.0,
    )
    certificate = compute_posterior_certificate(
        system.operator,
        mismatched,
        prediction.variance,
        conditional_observation_norm=float(
            torch.linalg.norm(system.conditional_observation_functional)
        ),
        operator_eigenvalue_lower_bound=system.operator_eigenvalue_lower_bound,
        rhs=system.conditional_cross,
    )

    assert math.isinf(certificate.mean_error_upper_bound)
    assert math.isinf(certificate.variance_error_upper_bound)
    assert not certificate.mean_solve_certified
    assert not certificate.solve_certified
