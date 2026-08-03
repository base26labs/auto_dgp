from __future__ import annotations

import inspect
import math

import pytest
import torch

from experiments import f02_support_dense_oracle
from experiments.f02_internal_models import (
    FrozenTERAParameters,
    TensorConfirmatorySplit,
    build_released_tera_predictor,
)
from experiments.f02_support_dense_oracle import (
    MAX_SUPPORT_RELATIVE_SOLVE_RESIDUAL,
    RANK_RULE_NAME,
    DenseSupportOracleError,
    predict_local_dense_support,
)
from gp.orbit.predictor import predict_local_value


def _quantized64(value: torch.Tensor | float) -> torch.Tensor:
    return torch.as_tensor(value).to(dtype=torch.float32).to(dtype=torch.float64)


def _full_rank_fixture(seed: int = 3):
    generator = torch.Generator().manual_seed(seed)
    m, dimension = 5, 7
    return (
        torch.randn(m, dimension, generator=generator, dtype=torch.float64),
        torch.randn(m, generator=generator, dtype=torch.float64),
        torch.randn(m, dimension, generator=generator, dtype=torch.float64),
        torch.randn(1, dimension, generator=generator, dtype=torch.float64),
    )


def _oracle_kwargs(kernel: str = "rbf") -> dict[str, object]:
    return {
        "lengthscale": torch.tensor([0.8], dtype=torch.float64),
        "outputscale": 1.2,
        "value_noise_variance": 0.03,
        "gradient_noise_variance": 0.04,
        "kernel": kernel,
        "gradient_noise_model": "iid",
        "function_jitter": 1e-8,
        "support_coordinate_jitter": 1e-8,
    }


def _released_split(
    x: torch.Tensor,
    values: torch.Tensor,
    gradients: torch.Tensor,
) -> TensorConfirmatorySplit:
    rows = x.shape[0]
    indices = torch.arange(rows, dtype=torch.long, device=x.device)
    return TensorConfirmatorySplit(
        name="oracle-test",
        source_indices=indices,
        X=x,
        value=values,
        gradient=gradients,
        trajectory_id=torch.zeros(rows, dtype=torch.long, device=x.device),
        time_index=indices.clone(),
        time_value=indices.to(dtype=x.dtype),
    )


def _independent_kernel_value(
    first: torch.Tensor,
    second: torch.Tensor,
    lengthscale: torch.Tensor,
    outputscale: torch.Tensor,
    kernel: str,
) -> torch.Tensor:
    """Kernel value without using the oracle, ORBIT, or released TERA helpers."""

    scaled_difference = (first - second) / lengthscale
    distance2 = torch.dot(scaled_difference, scaled_difference)
    if kernel == "rbf":
        return outputscale * torch.exp(-0.5 * distance2)
    if kernel == "matern52":
        scaled_radius = torch.sqrt(5.0 * distance2)
        return (
            outputscale
            * (1.0 + scaled_radius + scaled_radius.square() / 3.0)
            * torch.exp(-scaled_radius)
        )
    raise ValueError(f"unknown kernel: {kernel}")


def _independent_kernel_derivatives(
    first: torch.Tensor,
    second: torch.Tensor,
    lengthscale: torch.Tensor,
    outputscale: torch.Tensor,
    kernel: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return d/d(second) k and d(first)d(second) k independently."""

    if torch.equal(first, second):
        diagonal_scale = outputscale if kernel == "rbf" else (5.0 / 3.0) * outputscale
        return torch.zeros_like(first), torch.diag(diagonal_scale / lengthscale.square())

    first_leaf = first.detach().clone().requires_grad_(True)
    second_leaf = second.detach().clone().requires_grad_(True)
    covariance = _independent_kernel_value(
        first_leaf,
        second_leaf,
        lengthscale,
        outputscale,
        kernel,
    )
    first_gradient, second_gradient = torch.autograd.grad(
        covariance,
        (first_leaf, second_leaf),
        create_graph=True,
    )
    mixed = torch.stack(
        [
            torch.autograd.grad(
                first_gradient[index],
                second_leaf,
                retain_graph=True,
            )[0]
            for index in range(first.numel())
        ]
    )
    return second_gradient.detach(), mixed.detach()


def _independent_full_q_conditional(
    x_condition: torch.Tensor,
    values: torch.Tensor,
    gradients: torch.Tensor,
    x_target: torch.Tensor,
    *,
    lengthscale: torch.Tensor,
    outputscale: torch.Tensor,
    value_noise_variance: float,
    gradient_noise_variance: float,
    kernel: str,
    gradient_noise_model: str,
    function_jitter: float,
    q_coordinate_jitter: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the full joint Gaussian system from kernel autodiff blocks."""

    m, dimension = x_condition.shape
    target = x_target.squeeze(0)
    raw_differences = (x_condition - target).T
    function_covariance = torch.empty(m, m, dtype=x_condition.dtype)
    function_q_covariance = torch.empty(m, m * m, dtype=x_condition.dtype)
    q_blocks = torch.empty(m, m, m, m, dtype=x_condition.dtype)
    target_function_covariance = torch.empty(m, dtype=x_condition.dtype)
    target_q_covariance = torch.empty(m, m, dtype=x_condition.dtype)

    for first_index in range(m):
        target_function_covariance[first_index] = _independent_kernel_value(
            target,
            x_condition[first_index],
            lengthscale,
            outputscale,
            kernel,
        )
        target_gradient, _ = _independent_kernel_derivatives(
            target,
            x_condition[first_index],
            lengthscale,
            outputscale,
            kernel,
        )
        target_q_covariance[first_index] = target_gradient @ raw_differences

        for second_index in range(m):
            function_covariance[first_index, second_index] = _independent_kernel_value(
                x_condition[first_index],
                x_condition[second_index],
                lengthscale,
                outputscale,
                kernel,
            )
            second_gradient, _ = _independent_kernel_derivatives(
                x_condition[first_index],
                x_condition[second_index],
                lengthscale,
                outputscale,
                kernel,
            )
            column_start = second_index * m
            function_q_covariance[
                first_index,
                column_start : column_start + m,
            ] = second_gradient @ raw_differences

            _, mixed_gradient_covariance = _independent_kernel_derivatives(
                x_condition[first_index],
                x_condition[second_index],
                lengthscale,
                outputscale,
                kernel,
            )
            q_blocks[first_index, second_index] = (
                raw_differences.T @ mixed_gradient_covariance @ raw_differences
            )

    if gradient_noise_model == "iid":
        ambient_noise_metric = torch.eye(dimension, dtype=x_condition.dtype)
    elif gradient_noise_model == "scaled":
        ambient_noise_metric = torch.diag(lengthscale.square())
    else:
        raise ValueError(f"unknown gradient noise model: {gradient_noise_model}")
    projected_noise = raw_differences.T @ ambient_noise_metric @ raw_differences
    diagonal = torch.arange(m)
    q_blocks[diagonal, diagonal] = (
        q_blocks[diagonal, diagonal] + gradient_noise_variance * projected_noise
    )

    identity_m = torch.eye(m, dtype=x_condition.dtype)
    function_covariance = (
        function_covariance + (value_noise_variance + function_jitter) * identity_m
    )
    q_covariance = q_blocks.permute(0, 2, 1, 3).reshape(m * m, m * m)
    q_covariance = q_covariance + q_coordinate_jitter * torch.eye(
        m * m,
        dtype=x_condition.dtype,
    )
    observation_covariance = torch.cat(
        [
            torch.cat([function_covariance, function_q_covariance], dim=1),
            torch.cat([function_q_covariance.T, q_covariance], dim=1),
        ],
        dim=0,
    )
    observations = torch.cat([values, (gradients @ raw_differences).reshape(-1)])
    target_cross = torch.cat([target_function_covariance, target_q_covariance.reshape(-1)])
    mean = target_cross @ torch.linalg.solve(observation_covariance, observations)
    variance = outputscale - target_cross @ torch.linalg.solve(
        observation_covariance,
        target_cross,
    )
    return mean, variance


@pytest.mark.parametrize("kernel", ["rbf", "matern52"])
@pytest.mark.parametrize("gradient_noise_model", ["iid", "scaled"])
@pytest.mark.parametrize("q_coordinate_jitter", [1e-8, 1e-6, 1e-4])
def test_rank_deficient_ard_oracle_matches_independent_autodiff_full_q(
    kernel: str,
    gradient_noise_model: str,
    q_coordinate_jitter: float,
) -> None:
    """Regress the support transform against a separately assembled joint GP."""

    seed = (
        815
        + 101 * ["rbf", "matern52"].index(kernel)
        + 17 * ["iid", "scaled"].index(gradient_noise_model)
        + 3 * [1e-8, 1e-6, 1e-4].index(q_coordinate_jitter)
    )
    generator = torch.Generator().manual_seed(seed)
    m, dimension, intrinsic_rank = 5, 7, 3
    intrinsic = torch.randn(
        m,
        intrinsic_rank,
        generator=generator,
        dtype=torch.float64,
    )
    target_intrinsic = torch.randn(
        1,
        intrinsic_rank,
        generator=generator,
        dtype=torch.float64,
    )
    fixed = torch.tensor([0.2, -0.4, 0.7, 0.9], dtype=torch.float64)
    x = _quantized64(torch.cat([intrinsic, fixed.expand(m, -1)], dim=1))
    target = _quantized64(torch.cat([target_intrinsic, fixed.reshape(1, -1)], dim=1))
    values = _quantized64(torch.randn(m, generator=generator, dtype=torch.float64))
    gradients = _quantized64(torch.randn(m, dimension, generator=generator, dtype=torch.float64))
    lengthscale = _quantized64(torch.linspace(0.65, 1.7, dimension))
    outputscale = float(_quantized64(1.4))
    value_noise = float(_quantized64(0.07))
    gradient_noise = float(_quantized64(0.025))
    function_jitter = float(_quantized64(1e-8))
    q_jitter = float(_quantized64(q_coordinate_jitter))

    oracle = predict_local_dense_support(
        x,
        values,
        gradients,
        target,
        lengthscale=lengthscale,
        outputscale=outputscale,
        value_noise_variance=value_noise,
        gradient_noise_variance=gradient_noise,
        kernel=kernel,
        gradient_noise_model=gradient_noise_model,
        function_jitter=function_jitter,
        support_coordinate_jitter=q_jitter,
    )
    expected_mean, expected_variance = _independent_full_q_conditional(
        x,
        values,
        gradients,
        target,
        lengthscale=lengthscale,
        outputscale=torch.tensor(outputscale, dtype=torch.float64),
        value_noise_variance=value_noise,
        gradient_noise_variance=gradient_noise,
        kernel=kernel,
        gradient_noise_model=gradient_noise_model,
        function_jitter=function_jitter,
        q_coordinate_jitter=q_jitter,
    )

    assert oracle.diagnostics.numerical_rank == intrinsic_rank
    assert oracle.diagnostics.support_system_dimension == m * intrinsic_rank
    assert math.isclose(
        oracle.diagnostics.support_coordinate_jitter,
        q_jitter,
        rel_tol=0.0,
        abs_tol=0.0,
    )
    transformed_q_jitter = q_jitter * (oracle.tera_to_support.T @ oracle.tera_to_support)
    torch.testing.assert_close(
        torch.tensor(
            oracle.diagnostics.support_coordinate_jitter_spectrum,
            dtype=torch.float64,
        ),
        torch.linalg.eigvalsh(transformed_q_jitter),
        rtol=2e-12,
        atol=2e-14,
    )

    # The independent reference uses exact analytic r=0 Matérn values and
    # Hessians.  The released/ORBIT-compatible path clamps r to 1e-12 before
    # evaluating Matérn-5/2 scalars, producing a stable O(1e-12) sensitivity.
    # RBF has no corresponding diagonal-clamp discrepancy.
    tolerance = 5e-12 if kernel == "matern52" else 2e-12
    torch.testing.assert_close(
        oracle.mean,
        expected_mean,
        rtol=5e-11,
        atol=tolerance,
    )
    torch.testing.assert_close(
        oracle.latent_variance,
        expected_variance,
        rtol=5e-11,
        atol=tolerance,
    )


def test_rank_zero_matern_ard_scaled_noise_reduces_to_function_only() -> None:
    generator = torch.Generator().manual_seed(2401)
    m, dimension = 4, 3
    target = _quantized64(torch.tensor([[0.2, -0.5, 0.8]], dtype=torch.float64))
    x = target.expand(m, -1).clone()
    values = _quantized64(torch.randn(m, generator=generator, dtype=torch.float64))
    gradients = _quantized64(torch.randn(m, dimension, generator=generator, dtype=torch.float64))
    lengthscale = _quantized64(torch.tensor([0.7, 1.1, 1.8]))
    outputscale = float(_quantized64(1.3))
    value_noise = float(_quantized64(0.02))
    gradient_noise = float(_quantized64(0.04))
    function_jitter = float(_quantized64(1e-8))
    q_jitter = float(_quantized64(1e-6))

    oracle = predict_local_dense_support(
        x,
        values,
        gradients,
        target,
        lengthscale=lengthscale,
        outputscale=outputscale,
        value_noise_variance=value_noise,
        gradient_noise_variance=gradient_noise,
        kernel="matern52",
        gradient_noise_model="scaled",
        function_jitter=function_jitter,
        support_coordinate_jitter=q_jitter,
    )
    expected_mean, expected_variance = _independent_full_q_conditional(
        x,
        values,
        gradients,
        target,
        lengthscale=lengthscale,
        outputscale=torch.tensor(outputscale, dtype=torch.float64),
        value_noise_variance=value_noise,
        gradient_noise_variance=gradient_noise,
        kernel="matern52",
        gradient_noise_model="scaled",
        function_jitter=function_jitter,
        q_coordinate_jitter=q_jitter,
    )

    assert oracle.diagnostics.numerical_rank == 0
    assert oracle.diagnostics.support_system_dimension == 0
    assert oracle.diagnostics.support_cholesky_attempts == 0
    assert oracle.support_basis.shape == (dimension, 0)
    assert oracle.support_coordinates.shape == (m, 0)
    assert oracle.tera_to_support.shape == (m, 0)
    torch.testing.assert_close(
        oracle.gradient_variance_reduction,
        torch.zeros((), dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        oracle.latent_variance,
        oracle.value_only_conditional_variance,
        rtol=0.0,
        atol=0.0,
    )
    # Same documented Matérn r=0 clamp sensitivity as the parameterized
    # full-q regression above; no projected gradient can contribute at rank 0.
    torch.testing.assert_close(
        oracle.mean,
        expected_mean,
        rtol=5e-11,
        atol=5e-12,
    )
    torch.testing.assert_close(
        oracle.latent_variance,
        expected_variance,
        rtol=5e-11,
        atol=5e-12,
    )

    changed_gradients = gradients + gradients.new_tensor(1000.0)
    repeated = predict_local_dense_support(
        x,
        values,
        changed_gradients,
        target,
        lengthscale=lengthscale,
        outputscale=outputscale,
        value_noise_variance=value_noise,
        gradient_noise_variance=gradient_noise,
        kernel="matern52",
        gradient_noise_model="scaled",
        function_jitter=function_jitter,
        support_coordinate_jitter=q_jitter,
    )
    torch.testing.assert_close(oracle.mean, repeated.mean, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        oracle.latent_variance,
        repeated.latent_variance,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("kernel", ["rbf", "matern52"])
def test_dense_support_oracle_matches_orbit64_without_using_orbit_solver(kernel):
    x, values, gradients, target = _full_rank_fixture()
    kwargs = _oracle_kwargs(kernel)
    oracle = predict_local_dense_support(x, values, gradients, target, **kwargs)

    orbit = predict_local_value(
        _quantized64(x),
        _quantized64(values),
        _quantized64(gradients),
        _quantized64(target),
        lengthscale=_quantized64(kwargs["lengthscale"]),
        outputscale=float(_quantized64(kwargs["outputscale"])),
        value_noise_variance=float(_quantized64(kwargs["value_noise_variance"])),
        gradient_noise_variance=float(_quantized64(kwargs["gradient_noise_variance"])),
        kernel=kernel,
        gradient_noise_model="iid",
        cg_tolerance=1e-12,
        cg_max_iterations=1_000,
        use_preconditioner=False,
        function_jitter=float(_quantized64(kwargs["function_jitter"])),
        reduced_jitter=float(_quantized64(kwargs["support_coordinate_jitter"])),
    )
    assert orbit.solve.converged
    torch.testing.assert_close(oracle.mean, orbit.mean, rtol=2e-10, atol=2e-10)
    torch.testing.assert_close(
        oracle.latent_variance,
        orbit.variance,
        rtol=2e-10,
        atol=2e-10,
    )
    assert oracle.diagnostics.rank_rule_name == RANK_RULE_NAME
    assert oracle.diagnostics.support_system_dimension == 25
    assert oracle.diagnostics.support_cholesky_attempts == 1
    assert oracle.diagnostics.support_relative_solve_residual < 1e-12
    assert (
        oracle.diagnostics.support_relative_solve_residual_tolerance
        == MAX_SUPPORT_RELATIVE_SOLVE_RESIDUAL
    )

    source = inspect.getsource(f02_support_dense_oracle)
    assert "gp.orbit" not in source
    assert "solve_reduced_cg" not in source


def test_weak_source_modes_use_the_same_fixed_cutoff_in_oracle_and_orbit64() -> None:
    singular_values = torch.tensor(
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 1e-7, 9e-8, 8e-8, 7e-8, 6e-8, 5e-8],
        dtype=torch.float32,
    )
    x = torch.diag(singular_values).to(torch.float64)
    target = torch.zeros(1, 12, dtype=torch.float64)
    generator = torch.Generator().manual_seed(911)
    values = torch.randn(12, generator=generator, dtype=torch.float32).to(torch.float64)
    gradients = torch.randn(12, 12, generator=generator, dtype=torch.float32).to(torch.float64)
    kwargs = _oracle_kwargs("rbf")

    oracle = predict_local_dense_support(x, values, gradients, target, **kwargs)
    orbit_fixed = predict_local_value(
        x,
        values,
        gradients,
        target,
        lengthscale=_quantized64(kwargs["lengthscale"]),
        outputscale=float(_quantized64(kwargs["outputscale"])),
        value_noise_variance=float(_quantized64(kwargs["value_noise_variance"])),
        gradient_noise_variance=float(_quantized64(kwargs["gradient_noise_variance"])),
        kernel="rbf",
        rank_epsilon=torch.finfo(torch.float32).eps,
        cg_tolerance=1e-12,
        cg_max_iterations=2_000,
        use_preconditioner=False,
        function_jitter=float(_quantized64(kwargs["function_jitter"])),
        reduced_jitter=float(_quantized64(kwargs["support_coordinate_jitter"])),
    )
    orbit_native = predict_local_value(
        x,
        values,
        gradients,
        target,
        lengthscale=_quantized64(kwargs["lengthscale"]),
        outputscale=float(_quantized64(kwargs["outputscale"])),
        value_noise_variance=float(_quantized64(kwargs["value_noise_variance"])),
        gradient_noise_variance=float(_quantized64(kwargs["gradient_noise_variance"])),
        kernel="rbf",
        cg_tolerance=1e-10,
        cg_max_iterations=2_000,
        use_preconditioner=False,
        function_jitter=float(_quantized64(kwargs["function_jitter"])),
        reduced_jitter=float(_quantized64(kwargs["support_coordinate_jitter"])),
    )

    assert oracle.diagnostics.numerical_rank == 6
    assert orbit_fixed.rank == 6
    assert not orbit_fixed.basis_is_exact
    assert orbit_native.rank == 12
    assert orbit_fixed.solve.converged
    torch.testing.assert_close(oracle.mean, orbit_fixed.mean, rtol=2e-9, atol=2e-9)
    torch.testing.assert_close(
        oracle.latent_variance,
        orbit_fixed.variance,
        rtol=2e-9,
        atol=2e-9,
    )


def test_full_q_dense_support_oracle_matches_direct_released_tera64():
    raw_x, raw_values, raw_gradients, raw_target = _full_rank_fixture(seed=41)
    x = _quantized64(raw_x)
    values = _quantized64(raw_values)
    gradients = _quantized64(raw_gradients)
    target = _quantized64(raw_target)
    kwargs = {
        name: (_quantized64(value) if not isinstance(value, str) else value)
        for name, value in _oracle_kwargs().items()
    }
    parameters = FrozenTERAParameters(
        lengthscale=kwargs["lengthscale"],
        outputscale=float(kwargs["outputscale"]),
        sigma_f=float(kwargs["value_noise_variance"]),
        sigma_g=float(kwargs["gradient_noise_variance"]),
        kernel="rbf",
    )

    oracle = predict_local_dense_support(
        x,
        values,
        gradients,
        target,
        **kwargs,
    )
    predictor = build_released_tera_predictor(
        _released_split(x, values, gradients),
        parameters,
        m=x.shape[0],
    )
    released = predictor.predict_f_marginals(target)

    assert oracle.diagnostics.numerical_rank == x.shape[0]
    assert oracle.diagnostics.support_system_dimension == x.shape[0] ** 2
    assert released.var[0] > torch.finfo(torch.float64).eps
    torch.testing.assert_close(oracle.mean, released.mean[0], rtol=2e-10, atol=2e-10)
    torch.testing.assert_close(
        oracle.latent_variance,
        released.var[0],
        rtol=2e-10,
        atol=2e-10,
    )


def test_rank_deficient_support_transform_matches_released_full_q_tera64():
    generator = torch.Generator().manual_seed(57)
    m, dimension, intrinsic_rank = 5, 6, 2
    intrinsic = torch.randn(m, intrinsic_rank, generator=generator, dtype=torch.float64)
    target_intrinsic = torch.randn(
        1,
        intrinsic_rank,
        generator=generator,
        dtype=torch.float64,
    )
    fixed = torch.tensor([0.4, -0.7, 0.2, 0.5], dtype=torch.float64)
    x = _quantized64(torch.cat([intrinsic, fixed.expand(m, -1)], dim=1))
    target = _quantized64(torch.cat([target_intrinsic, fixed.reshape(1, -1)], dim=1))
    values = _quantized64(torch.randn(m, generator=generator, dtype=torch.float64))
    gradients = _quantized64(torch.randn(m, dimension, generator=generator, dtype=torch.float64))
    kwargs = {
        name: (_quantized64(value) if not isinstance(value, str) else value)
        for name, value in _oracle_kwargs().items()
    }
    parameters = FrozenTERAParameters(
        lengthscale=kwargs["lengthscale"],
        outputscale=float(kwargs["outputscale"]),
        sigma_f=float(kwargs["value_noise_variance"]),
        sigma_g=float(kwargs["gradient_noise_variance"]),
        kernel="rbf",
    )

    oracle = predict_local_dense_support(
        x,
        values,
        gradients,
        target,
        **kwargs,
    )
    predictor = build_released_tera_predictor(
        _released_split(x, values, gradients),
        parameters,
        m=m,
    )
    released = predictor.predict_f_marginals(target)

    assert oracle.diagnostics.numerical_rank == intrinsic_rank
    assert oracle.diagnostics.support_system_dimension == m * intrinsic_rank
    assert oracle.support_coordinates.shape == (m, intrinsic_rank)
    assert released.var[0] > torch.finfo(torch.float64).eps
    torch.testing.assert_close(oracle.mean, released.mean[0], rtol=2e-9, atol=2e-9)
    torch.testing.assert_close(
        oracle.latent_variance,
        released.var[0],
        rtol=2e-9,
        atol=2e-9,
    )


def test_known_constraint_subspace_produces_explicit_low_rank_dense_system():
    generator = torch.Generator().manual_seed(17)
    m, dimension, intrinsic_rank = 7, 8, 2
    intrinsic = torch.randn(m, intrinsic_rank, generator=generator, dtype=torch.float64)
    target_intrinsic = torch.randn(
        1,
        intrinsic_rank,
        generator=generator,
        dtype=torch.float64,
    )
    fixed = torch.tensor([0.4, -0.7, 0.2, 0.5, -0.3, 0.9], dtype=torch.float64)
    x = torch.cat([intrinsic, fixed.expand(m, -1)], dim=1)
    target = torch.cat([target_intrinsic, fixed.reshape(1, -1)], dim=1)
    values = torch.randn(m, generator=generator, dtype=torch.float64)
    gradients = torch.randn(m, dimension, generator=generator, dtype=torch.float64)

    prediction = predict_local_dense_support(
        x,
        values,
        gradients,
        target,
        **_oracle_kwargs(),
    )
    diagnostics = prediction.diagnostics
    assert diagnostics.maximum_rank == m
    assert diagnostics.numerical_rank == intrinsic_rank
    assert diagnostics.support_system_dimension == m * intrinsic_rank
    assert len(diagnostics.singular_values) == m
    assert diagnostics.singular_values[intrinsic_rank] <= diagnostics.rank_threshold
    assert len(diagnostics.support_spectrum_before_jitter) == m * intrinsic_rank
    assert len(diagnostics.support_spectrum_after_jitter) == m * intrinsic_rank
    assert len(diagnostics.support_coordinate_jitter_spectrum) == intrinsic_rank
    assert diagnostics.support_spectrum_after_jitter[0] > 0.0
    assert torch.isfinite(prediction.mean)
    assert prediction.latent_variance > 0.0


def test_prediction_is_repeatable_and_invariant_to_ambient_rotation():
    x, values, gradients, target = _full_rank_fixture(seed=21)
    first = predict_local_dense_support(
        x,
        values,
        gradients,
        target,
        **_oracle_kwargs(),
    )
    repeated = predict_local_dense_support(
        x,
        values,
        gradients,
        target,
        **_oracle_kwargs(),
    )
    torch.testing.assert_close(first.mean, repeated.mean, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        first.latent_variance,
        repeated.latent_variance,
        rtol=0.0,
        atol=0.0,
    )
    assert first.diagnostics == repeated.diagnostics

    explicitly_quantized = predict_local_dense_support(
        _quantized64(x),
        _quantized64(values),
        _quantized64(gradients),
        _quantized64(target),
        lengthscale=_quantized64(0.8).reshape(1),
        outputscale=_quantized64(1.2),
        value_noise_variance=_quantized64(0.03),
        gradient_noise_variance=_quantized64(0.04),
        kernel="rbf",
        gradient_noise_model="iid",
        function_jitter=float(_quantized64(1e-8)),
        support_coordinate_jitter=float(_quantized64(1e-8)),
    )
    torch.testing.assert_close(first.mean, explicitly_quantized.mean, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        first.latent_variance,
        explicitly_quantized.latent_variance,
        rtol=0.0,
        atol=0.0,
    )

    generator = torch.Generator().manual_seed(99)
    rotation, _ = torch.linalg.qr(
        torch.randn(x.shape[1], x.shape[1], generator=generator, dtype=torch.float64)
    )
    rotated = predict_local_dense_support(
        x @ rotation,
        values,
        gradients @ rotation,
        target @ rotation,
        **_oracle_kwargs(),
    )
    # Rotation is followed by the mandated fp32 quantization, so invariance is
    # expected to source-precision accuracy rather than bitwise equality.
    torch.testing.assert_close(first.mean, rotated.mean, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        first.latent_variance,
        rotated.latent_variance,
        rtol=2e-6,
        atol=2e-6,
    )
    assert first.diagnostics.numerical_rank == rotated.diagnostics.numerical_rank


def test_prediction_is_invariant_to_exactly_null_ambient_augmentation():
    x, values, gradients, target = _full_rank_fixture(seed=32)
    original = predict_local_dense_support(
        x,
        values,
        gradients,
        target,
        **_oracle_kwargs(),
    )
    generator = torch.Generator().manual_seed(77)
    extra_gradient_components = torch.randn(
        x.shape[0],
        3,
        generator=generator,
        dtype=torch.float64,
    )
    augmented = predict_local_dense_support(
        torch.cat([x, torch.zeros(x.shape[0], 3, dtype=x.dtype)], dim=1),
        values,
        torch.cat([gradients, extra_gradient_components], dim=1),
        torch.cat([target, torch.zeros(1, 3, dtype=target.dtype)], dim=1),
        **_oracle_kwargs(),
    )
    torch.testing.assert_close(original.mean, augmented.mean, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(
        original.latent_variance,
        augmented.latent_variance,
        rtol=2e-12,
        atol=2e-12,
    )
    assert original.diagnostics.numerical_rank == augmented.diagnostics.numerical_rank


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"outputscale": float("nan")}, "finite after fp32 quantization"),
        ({"support_coordinate_jitter": 0.0}, "finite positive scalar"),
        ({"gradient_noise_variance": -1e-3}, "finite non-negative scalar"),
        (
            {"function_jitter": 1e-3, "maximum_function_jitter": 1e-4},
            "jitter bounds",
        ),
    ],
)
def test_oracle_fails_closed_on_invalid_numeric_contract(change, message):
    x, values, gradients, target = _full_rank_fixture()
    kwargs = {**_oracle_kwargs(), **change}
    with pytest.raises(DenseSupportOracleError, match=message):
        predict_local_dense_support(x, values, gradients, target, **kwargs)


def test_oracle_rejects_shape_mismatch_instead_of_broadcasting():
    x, values, gradients, target = _full_rank_fixture()
    with pytest.raises(DenseSupportOracleError, match="gradient_condition must have shape"):
        predict_local_dense_support(
            x,
            values,
            gradients[:, :-1],
            target,
            **_oracle_kwargs(),
        )
