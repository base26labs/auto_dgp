"""Focused contracts for reusable ORBIT local-value systems."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import pytest
import torch

from gp.orbit import build_local_geometry_from_differences
from gp.orbit.predictor import (
    LocalPrediction,
    LocalValueSystem,
    MarginalPredictions,
    _build_local_value_system_from_registered_geometry,
    build_local_value_system,
    predict_local_value,
    predict_local_value_and_mean_gradient,
    solve_local_value_system,
)


def test_prediction_dataclasses_preserve_historical_positional_prefixes() -> None:
    assert [field.name for field in fields(LocalPrediction)][:7] == [
        "mean",
        "variance",
        "rank",
        "basis_is_exact",
        "finite_precision_variance_correction",
        "solve",
        "certificate",
    ]
    assert [field.name for field in fields(MarginalPredictions)][:14] == [
        "mean",
        "variance",
        "ranks",
        "iterations",
        "operator_matvecs",
        "preconditioner_applications",
        "relative_residuals",
        "converged",
        "variance_error_upper_bounds",
        "expected_kl_upper_bounds",
        "exact_arithmetic_certified",
        "floating_point_rigorous",
        "basis_exact",
        "finite_precision_variance_corrections",
    ]


def _nonzero_case(*, seed: int = 1901) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    dtype = torch.float64
    m, dimension = 5, 7
    x_target = torch.randn(1, dimension, generator=generator, dtype=dtype)
    x_condition = x_target + torch.randn(
        m,
        dimension,
        generator=generator,
        dtype=dtype,
    )
    values = torch.randn(m, generator=generator, dtype=dtype)
    gradients = torch.randn(m, dimension, generator=generator, dtype=dtype)
    lengthscale = torch.linspace(0.7, 1.6, dimension, dtype=dtype)
    return x_condition, values, gradients, x_target, lengthscale


def _build_system(*, seed: int = 1901, function_jitter: float = 1e-8) -> LocalValueSystem:
    x_condition, values, gradients, x_target, lengthscale = _nonzero_case(seed=seed)
    return build_local_value_system(
        x_condition,
        values,
        gradients,
        x_target,
        lengthscale=lengthscale,
        outputscale=1.4,
        value_noise_variance=0.06,
        gradient_noise_variance=0.04,
        kernel="matern52",
        gradient_noise_model="metric_matched",
        function_jitter=function_jitter,
        reduced_jitter=1e-8,
    )


def test_build_reuses_precomputed_geometry_without_a_second_svd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_condition, values, gradients, x_target, lengthscale = _nonzero_case(seed=1902)
    scaled_differences = (x_condition - x_target).T / lengthscale.reshape(-1, 1)
    geometry = build_local_geometry_from_differences(scaled_differences)
    reference = build_local_value_system(
        x_condition,
        values,
        gradients,
        x_target,
        lengthscale=lengthscale,
        outputscale=1.4,
        value_noise_variance=0.06,
        gradient_noise_variance=0.04,
        kernel="matern52",
        gradient_noise_model="metric_matched",
    )

    def forbidden_svd(*args, **kwargs):
        raise AssertionError("precomputed geometry must suppress a second SVD")

    monkeypatch.setattr(torch.linalg, "svd", forbidden_svd)
    system = _build_local_value_system_from_registered_geometry(
        x_condition,
        values,
        gradients,
        x_target,
        lengthscale=lengthscale,
        outputscale=1.4,
        value_noise_variance=0.06,
        gradient_noise_variance=0.04,
        kernel="matern52",
        gradient_noise_model="metric_matched",
        precomputed_geometry=geometry,
    )

    assert system.geometry is geometry
    reference_prediction = solve_local_value_system(
        reference,
        tolerance=1e-10,
        max_iterations=200,
    )
    precomputed_prediction = solve_local_value_system(
        system,
        tolerance=1e-10,
        max_iterations=200,
    )
    torch.testing.assert_close(
        precomputed_prediction.mean,
        reference_prediction.mean,
        rtol=2e-12,
        atol=2e-12,
    )
    torch.testing.assert_close(
        precomputed_prediction.variance,
        reference_prediction.variance,
        rtol=2e-12,
        atol=2e-12,
    )


def test_public_builder_does_not_accept_precomputed_geometry() -> None:
    x_condition, values, gradients, x_target, lengthscale = _nonzero_case(seed=1904)
    scaled_differences = (x_condition - x_target).T / lengthscale.reshape(-1, 1)
    geometry = build_local_geometry_from_differences(scaled_differences)

    with pytest.raises(TypeError, match="precomputed_geometry"):
        build_local_value_system(
            x_condition,
            values,
            gradients,
            x_target,
            lengthscale=lengthscale,
            outputscale=1.4,
            value_noise_variance=0.06,
            gradient_noise_variance=0.04,
            kernel="matern52",
            gradient_noise_model="metric_matched",
            precomputed_geometry=geometry,
        )


def _tensor_snapshot(root: object) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Snapshot every tensor reachable through the reusable system's public state."""

    snapshot: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    visited: set[int] = set()

    def visit(value: Any, path: str) -> None:
        if isinstance(value, torch.Tensor):
            snapshot[path] = (value, value.detach().clone())
            return
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                visit(getattr(value, field.name), f"{path}.{field.name}")
            return
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{path}[{key!r}]")
            return
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
            return
        if value.__class__.__module__.startswith("gp.orbit") and hasattr(value, "__dict__"):
            for name, item in vars(value).items():
                visit(item, f"{path}.{name}")

    visit(root, "system")
    return snapshot


def test_predict_local_value_delegates_to_build_then_solve() -> None:
    x_condition, values, gradients, x_target, lengthscale = _nonzero_case(seed=1903)
    model_kwargs = {
        "lengthscale": lengthscale,
        "outputscale": 1.25,
        "value_noise_variance": 0.05,
        "gradient_noise_variance": 0.03,
        "kernel": "rbf",
        "gradient_noise_model": "metric_matched",
        "function_jitter": 2e-9,
        "reduced_jitter": 3e-9,
    }
    system = build_local_value_system(
        x_condition,
        values,
        gradients,
        x_target,
        **model_kwargs,
    )
    reusable = solve_local_value_system(
        system,
        tolerance=1e-11,
        max_iterations=300,
        use_preconditioner=True,
    )
    public = predict_local_value(
        x_condition,
        values,
        gradients,
        x_target,
        **model_kwargs,
        cg_tolerance=1e-11,
        cg_max_iterations=300,
        use_preconditioner=True,
    )

    torch.testing.assert_close(public.mean, reusable.mean, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(public.functional_mean, reusable.functional_mean)
    torch.testing.assert_close(public.mean_reassociation_delta, reusable.mean_reassociation_delta)
    torch.testing.assert_close(public.variance, reusable.variance, rtol=2e-12, atol=2e-12)
    assert public.rank == reusable.rank == system.geometry.rank
    assert public.solve.iterations == reusable.solve.iterations
    assert public.solve.operator_matvecs == reusable.solve.operator_matvecs
    assert public.solve.preconditioner_applications == reusable.solve.preconditioner_applications
    assert public.solve.termination_reason == reusable.solve.termination_reason
    assert public.certificate == reusable.certificate


def test_implicit_mean_gradient_matches_direct_autograd() -> None:
    x_condition, values, gradients, x_target, lengthscale = _nonzero_case(seed=1905)
    model_kwargs = {
        "lengthscale": lengthscale,
        "outputscale": 1.25,
        "value_noise_variance": 0.05,
        "gradient_noise_variance": 0.03,
        "kernel": "rbf",
        "gradient_noise_model": "iid",
        "cg_tolerance": 1e-12,
        "cg_max_iterations": 300,
    }
    differentiable_target = x_target.clone().requires_grad_(True)
    direct = predict_local_value(
        x_condition,
        values,
        gradients,
        differentiable_target,
        **model_kwargs,
    )
    direct_gradient = torch.autograd.grad(direct.functional_mean, differentiable_target)[0].squeeze(
        0
    )

    implicit = predict_local_value_and_mean_gradient(
        x_condition,
        values,
        gradients,
        x_target,
        **model_kwargs,
    )

    assert implicit.prediction.solve.converged
    assert implicit.adjoint_solve.converged
    torch.testing.assert_close(implicit.prediction.mean, direct.mean, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(implicit.mean_gradient, direct_gradient, rtol=2e-6, atol=2e-7)


def test_one_system_supports_multiple_tolerances_without_mutating_tensor_state() -> None:
    system = _build_system(seed=1911)
    operator = system.operator
    cached_preconditioner = system.preconditioner
    snapshot = _tensor_snapshot(system)

    loose = solve_local_value_system(
        system,
        tolerance=1e-3,
        max_iterations=300,
        use_preconditioner=True,
    )
    tight = solve_local_value_system(
        system,
        tolerance=1e-11,
        max_iterations=300,
        use_preconditioner=True,
    )

    assert system.operator is operator
    assert system.preconditioner is cached_preconditioner
    assert loose.solve.requested_tolerance == pytest.approx(1e-3)
    assert tight.solve.requested_tolerance == pytest.approx(1e-11)
    assert loose.solve.iterations <= tight.solve.iterations
    assert tight.solve.relative_residual <= loose.solve.relative_residual
    after = _tensor_snapshot(system)
    assert after.keys() == snapshot.keys()
    for path, (tensor, before) in snapshot.items():
        assert after[path][0].data_ptr() == tensor.data_ptr()
        torch.testing.assert_close(after[path][0], before, rtol=0.0, atol=0.0, msg=path)


def test_conditional_observation_is_value_conditioned_gradient_functional() -> None:
    system = _build_system(seed=1917)
    _, values, _, _, _ = _nonzero_case(seed=1917)
    value_observation_weights = torch.cholesky_solve(
        values.unsqueeze(1),
        system.function_cholesky,
    ).squeeze(1)
    expected = system.orthonormal_observations.reshape(-1) - system.operator.q_matmul(
        value_observation_weights
    )

    torch.testing.assert_close(
        system.conditional_observation_functional,
        expected,
        rtol=2e-13,
        atol=2e-13,
    )
    torch.testing.assert_close(
        system.base_mean,
        torch.dot(system.function_weights, values),
        rtol=2e-13,
        atol=2e-13,
    )


def test_local_prediction_keeps_legacy_mean_and_reports_reassociation() -> None:
    system = _build_system(seed=1921)
    _, values, _, _, _ = _nonzero_case(seed=1921)
    prediction = solve_local_value_system(
        system,
        tolerance=1e-6,
        max_iterations=2,
        use_preconditioner=False,
    )
    q_t_weights = system.operator.q_t_matmul(prediction.solve.solution)
    corrected_function_weights = system.function_weights - torch.cholesky_solve(
        q_t_weights.unsqueeze(1),
        system.function_cholesky,
    ).squeeze(1)
    legacy_mean = torch.dot(corrected_function_weights, values) + torch.dot(
        prediction.solve.solution,
        system.orthonormal_observations.reshape(-1),
    )
    functional_mean = system.base_mean + torch.dot(
        prediction.solve.solution,
        system.conditional_observation_functional,
    )

    torch.testing.assert_close(prediction.mean, legacy_mean, rtol=0.0, atol=0.0)
    torch.testing.assert_close(prediction.functional_mean, functional_mean, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        prediction.mean_reassociation_delta,
        prediction.mean - prediction.functional_mean,
        rtol=0.0,
        atol=0.0,
    )


def test_represented_dense_exact_mean_error_is_bounded_by_fresh_residual() -> None:
    system = _build_system(seed=1931)
    prediction = solve_local_value_system(
        system,
        tolerance=1e-14,
        max_iterations=1,
        use_preconditioner=False,
    )
    exact_weights = torch.linalg.solve(system.operator.dense(), system.conditional_cross)
    exact_mean = system.base_mean + torch.dot(
        exact_weights,
        system.conditional_observation_functional,
    )
    represented_error = abs(float(prediction.functional_mean - exact_mean))
    expected_bound = (
        float(torch.linalg.norm(system.conditional_observation_functional))
        * prediction.solve.residual_norm
        / prediction.certificate.operator_eigenvalue_lower_bound
    )

    assert prediction.solve.residual_is_fresh
    assert prediction.certificate.mean_error_upper_bound == pytest.approx(
        expected_bound,
        rel=5e-13,
        abs=5e-15,
    )
    assert prediction.certificate.conditional_observation_norm == pytest.approx(
        float(torch.linalg.norm(system.conditional_observation_functional)),
        rel=5e-13,
        abs=5e-15,
    )
    assert represented_error <= prediction.certificate.mean_error_upper_bound * (1.0 + 2e-12)
    assert not prediction.certificate.floating_point_rigorous


def test_build_records_requested_and_escalated_function_jitter() -> None:
    dtype = torch.float64
    x_condition = torch.zeros(3, 2, dtype=dtype)
    x_target = torch.zeros(1, 2, dtype=dtype)
    system = build_local_value_system(
        x_condition,
        torch.tensor([0.2, -0.1, 0.3], dtype=dtype),
        torch.zeros(3, 2, dtype=dtype),
        x_target,
        lengthscale=torch.ones(1, dtype=dtype),
        outputscale=0.0,
        value_noise_variance=0.0,
        gradient_noise_variance=0.1,
        kernel="rbf",
        function_jitter=0.0,
    )

    assert system.function_jitter_requested == 0.0
    assert system.function_jitter_used == pytest.approx(torch.finfo(dtype).eps)
    assert system.function_jitter_attempts == 2
    assert torch.isfinite(system.function_cholesky).all()
    torch.testing.assert_close(
        system.function_cholesky @ system.function_cholesky.T,
        system.function_system_matrix,
        rtol=2e-15,
        atol=2e-15,
    )


def test_rank_zero_system_is_reusable_value_only_conditional() -> None:
    dtype = torch.float64
    x_condition = torch.zeros(4, 3, dtype=dtype)
    x_target = torch.zeros(1, 3, dtype=dtype)
    values = torch.tensor([0.2, -0.1, 0.4, 0.7], dtype=dtype)
    system = build_local_value_system(
        x_condition,
        values,
        torch.randn(4, 3, generator=torch.Generator().manual_seed(1941), dtype=dtype),
        x_target,
        lengthscale=torch.tensor([1.3], dtype=dtype),
        outputscale=1.2,
        value_noise_variance=0.05,
        gradient_noise_variance=0.02,
        kernel="matern52",
    )
    prediction = solve_local_value_system(
        system,
        tolerance=1e-8,
        max_iterations=20,
        use_preconditioner=False,
    )

    assert system.geometry.rank == prediction.rank == 0
    assert system.conditional_cross.numel() == 0
    assert system.conditional_observation_functional.numel() == 0
    assert prediction.solve.iterations == 0
    assert prediction.solve.termination_reason == "zero_rhs"
    assert prediction.solve.converged
    torch.testing.assert_close(prediction.mean, system.base_mean)
    torch.testing.assert_close(prediction.functional_mean, system.base_mean)
    torch.testing.assert_close(prediction.mean_reassociation_delta, torch.zeros((), dtype=dtype))
    assert prediction.certificate.mean_error_upper_bound == 0.0
    assert prediction.certificate.conditional_observation_norm == 0.0


def test_explicit_preconditioner_requires_opt_in_and_is_used_verbatim() -> None:
    system = _build_system(seed=1951)

    class CountingIdentity:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, value: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return value

    rejected = CountingIdentity()
    with pytest.raises(ValueError, match="preconditioner.*use_preconditioner"):
        solve_local_value_system(
            system,
            tolerance=1e-7,
            max_iterations=3,
            use_preconditioner=False,
            preconditioner=rejected,
        )
    assert rejected.calls == 0

    unpreconditioned = solve_local_value_system(
        system,
        tolerance=1e-7,
        max_iterations=3,
        use_preconditioner=False,
    )
    assert unpreconditioned.solve.preconditioner_applications == 0

    explicit = CountingIdentity()
    explicitly_preconditioned = solve_local_value_system(
        system,
        tolerance=1e-7,
        max_iterations=3,
        use_preconditioner=True,
        preconditioner=explicit,
    )
    assert explicit.calls > 0
    assert explicitly_preconditioned.solve.preconditioner_applications == explicit.calls

    cached = solve_local_value_system(
        system,
        tolerance=1e-7,
        max_iterations=3,
        use_preconditioner=True,
    )
    assert cached.solve.preconditioner_applications > 0


def test_build_snapshots_values_against_caller_in_place_mutation() -> None:
    x_condition, values, gradients, x_target, lengthscale = _nonzero_case(seed=1961)
    original_values = values.clone()
    system = build_local_value_system(
        x_condition,
        values,
        gradients,
        x_target,
        lengthscale=lengthscale,
        outputscale=1.2,
        value_noise_variance=0.05,
        gradient_noise_variance=0.03,
        kernel="rbf",
    )
    before = solve_local_value_system(
        system,
        tolerance=1e-8,
        max_iterations=100,
    )
    values.add_(1000.0)
    after = solve_local_value_system(
        system,
        tolerance=1e-8,
        max_iterations=100,
    )

    assert system.value_condition.data_ptr() != values.data_ptr()
    torch.testing.assert_close(system.value_condition, original_values, rtol=0.0, atol=0.0)
    torch.testing.assert_close(after.mean, before.mean, rtol=0.0, atol=0.0)
    torch.testing.assert_close(after.functional_mean, before.functional_mean, rtol=0.0, atol=0.0)
    assert after.certificate == before.certificate


def test_missing_cached_preconditioner_requires_explicit_replacement() -> None:
    x_condition, values, gradients, x_target, lengthscale = _nonzero_case(seed=1971)
    system = build_local_value_system(
        x_condition,
        values,
        gradients,
        x_target,
        lengthscale=lengthscale,
        outputscale=1.2,
        value_noise_variance=0.05,
        gradient_noise_variance=0.03,
        kernel="rbf",
        build_preconditioner=False,
    )

    assert system.preconditioner is None
    with pytest.raises(ValueError, match="no cached preconditioner"):
        solve_local_value_system(system, tolerance=1e-6, max_iterations=10)


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), 0.0, -1.0])
def test_reusable_solve_rejects_invalid_tolerance(tolerance: float) -> None:
    system = _build_system(seed=1981)
    with pytest.raises(ValueError, match="finite and positive"):
        solve_local_value_system(system, tolerance=tolerance)
