"""Correctness tests for the ORBIT reduced conditional operator."""

import pytest
import torch

from gp.orbit import (
    OrthonormalReducedOperator,
    ReducedKroneckerPreconditioner,
    build_local_geometry,
    build_local_geometry_from_differences,
    compute_posterior_certificate,
    solve_reduced_cg,
)
from gp.orbit.predictor import _cholesky_with_jitter, _projected_noise_gram, predict_local_value


def _fixture(m: int = 5, d: int = 8, *, seed: int = 2):
    generator = torch.Generator().manual_seed(seed)
    dtype = torch.float64
    delta = torch.randn(d, m, generator=generator, dtype=dtype)
    gram = delta.T @ delta
    geometry = build_local_geometry(gram)
    coordinates = geometry.coordinates

    pair = coordinates[:, None, :] - coordinates[None, :, :]
    distance2 = (pair * pair).sum(dim=-1)
    alpha = torch.exp(-0.5 * distance2)
    beta = -alpha
    kff = alpha + 0.05 * torch.eye(m, dtype=dtype)
    lff = torch.linalg.cholesky(kff)
    noise = 0.02 * torch.eye(geometry.rank, dtype=dtype)
    operator = OrthonormalReducedOperator(
        coordinates,
        alpha,
        beta,
        lff,
        noise,
        jitter=1e-9,
    )
    return delta, geometry, operator


def _explicit_blocks(operator: OrthonormalReducedOperator) -> torch.Tensor:
    m, rank = operator.m, operator.rank
    coords = operator.coordinates
    pair = coords[:, None, :] - coords[None, :, :]
    identity = torch.eye(rank, dtype=coords.dtype)
    blocks = operator.alpha[:, :, None, None] * identity
    blocks = blocks + operator.beta[:, :, None, None] * (pair[:, :, :, None] * pair[:, :, None, :])
    diag = torch.arange(m)
    blocks[diag, diag] = blocks[diag, diag] + operator.gradient_noise
    g0 = blocks.permute(0, 2, 1, 3).reshape(m * rank, m * rank)

    q = (-operator.alpha[:, :, None] * pair).permute(0, 2, 1).reshape(m * rank, m)
    kff_inv_qt = torch.cholesky_solve(q.T, operator.function_cholesky)
    result = g0 - q @ kff_inv_qt
    result = result + operator.jitter * torch.eye(m * rank, dtype=coords.dtype)
    return 0.5 * (result + result.T)


def test_local_geometry_is_exact_change_of_coordinates():
    delta, geometry, _ = _fixture()
    torch.testing.assert_close(
        geometry.coordinates @ geometry.coordinates.T,
        delta.T @ delta,
        rtol=1e-11,
        atol=1e-11,
    )

    gradients = torch.randn(5, delta.shape[0], dtype=delta.dtype)
    tera_q = gradients @ delta
    reduced = tera_q @ geometry.q_to_z
    reconstructed_q = reduced @ geometry.coordinates.T
    torch.testing.assert_close(reconstructed_q, tera_q, rtol=1e-11, atol=1e-11)


def test_rank_deficient_geometry_removes_only_redundancy():
    generator = torch.Generator().manual_seed(8)
    base = torch.randn(3, 6, generator=generator, dtype=torch.float64)
    delta = torch.cat([base, 2.0 * base], dim=0)
    geometry = build_local_geometry(delta.T @ delta)
    assert geometry.rank == 3

    gradients = torch.randn(4, delta.shape[0], generator=generator, dtype=delta.dtype)
    tera_q = gradients @ delta
    reduced = tera_q @ geometry.q_to_z
    torch.testing.assert_close(
        reduced @ geometry.coordinates.T,
        tera_q,
        rtol=1e-10,
        atol=1e-10,
    )


def test_matrix_free_operator_matches_explicit_dense_blocks():
    _, _, operator = _fixture()
    expected = _explicit_blocks(operator)
    torch.testing.assert_close(operator.dense(), expected, rtol=1e-11, atol=1e-11)


def test_orthonormal_conditional_is_equivalent_to_tera_coordinates():
    """The basis change must preserve TERA's mean and variance corrections."""

    _, _, operator = _fixture(m=6, d=9, seed=14)
    m, rank = operator.m, operator.rank
    assert rank == m
    coordinates = operator.coordinates
    gram = coordinates @ coordinates.T
    pair_gram = gram[:, None, :] - gram[None, :, :]

    identity_m = torch.eye(m, dtype=coordinates.dtype)
    basis_map = torch.kron(identity_m, coordinates.contiguous())

    pair_coordinates = coordinates[:, None, :] - coordinates[None, :, :]
    q_orth = (-operator.alpha[:, :, None] * pair_coordinates).permute(0, 2, 1)
    q_orth = q_orth.reshape(m * rank, m)
    q_tera = (-operator.alpha[:, :, None] * pair_gram).permute(0, 2, 1)
    q_tera = q_tera.reshape(m * m, m)
    torch.testing.assert_close(basis_map @ q_orth, q_tera, rtol=1e-10, atol=1e-10)

    orth_blocks = operator.alpha[:, :, None, None] * torch.eye(
        rank,
        dtype=coordinates.dtype,
    )
    orth_blocks = orth_blocks + operator.beta[:, :, None, None] * (
        pair_coordinates[:, :, :, None] * pair_coordinates[:, :, None, :]
    )
    diag = torch.arange(m)
    orth_blocks[diag, diag] = orth_blocks[diag, diag] + operator.gradient_noise
    g_orth = orth_blocks.permute(0, 2, 1, 3).reshape(m * rank, m * rank)

    tera_blocks = operator.alpha[:, :, None, None] * gram
    tera_blocks = tera_blocks + operator.beta[:, :, None, None] * (
        pair_gram[:, :, :, None] * pair_gram[:, :, None, :]
    )
    projected_noise = coordinates @ operator.gradient_noise @ coordinates.T
    tera_blocks[diag, diag] = tera_blocks[diag, diag] + projected_noise
    g_tera = tera_blocks.permute(0, 2, 1, 3).reshape(m * m, m * m)
    torch.testing.assert_close(
        basis_map @ g_orth @ basis_map.T,
        g_tera,
        rtol=1e-9,
        atol=1e-9,
    )

    kff_inv_q_orth = torch.cholesky_solve(q_orth.T, operator.function_cholesky)
    kff_inv_q_tera = torch.cholesky_solve(q_tera.T, operator.function_cholesky)
    k_orth = g_orth - q_orth @ kff_inv_q_orth
    k_tera = g_tera - q_tera @ kff_inv_q_tera

    generator = torch.Generator().manual_seed(31)
    target_alpha = torch.rand(m, generator=generator, dtype=coordinates.dtype)
    function_weights = torch.randn(m, generator=generator, dtype=coordinates.dtype)
    cross_orth = operator.conditional_cross(target_alpha, function_weights)
    cross_tera = basis_map @ cross_orth
    observation_orth = torch.randn(m * rank, generator=generator, dtype=coordinates.dtype)
    observation_tera = basis_map @ observation_orth

    mean_orth = torch.dot(cross_orth, torch.linalg.solve(k_orth, observation_orth))
    mean_tera = torch.dot(cross_tera, torch.linalg.solve(k_tera, observation_tera))
    variance_orth = torch.dot(cross_orth, torch.linalg.solve(k_orth, cross_orth))
    variance_tera = torch.dot(cross_tera, torch.linalg.solve(k_tera, cross_tera))
    torch.testing.assert_close(mean_orth, mean_tera, rtol=2e-8, atol=2e-8)
    torch.testing.assert_close(variance_orth, variance_tera, rtol=2e-8, atol=2e-8)


@pytest.mark.parametrize("preconditioned", [False, True])
def test_reduced_cg_matches_dense_solve(preconditioned: bool):
    _, _, operator = _fixture(m=6, d=9, seed=12)
    generator = torch.Generator().manual_seed(19)
    rhs = torch.randn(operator.size, generator=generator, dtype=torch.float64)
    preconditioner = ReducedKroneckerPreconditioner(operator) if preconditioned else None
    result = solve_reduced_cg(
        operator,
        rhs,
        tolerance=1e-10,
        max_iterations=4 * operator.size,
        preconditioner=preconditioner,
    )
    expected = torch.linalg.solve(_explicit_blocks(operator), rhs)
    assert result.converged
    assert result.relative_residual < 1e-10
    torch.testing.assert_close(result.solution, expected, rtol=2e-8, atol=2e-8)


def test_approximate_rank_mode_reports_discarded_geometry():
    delta, _, _ = _fixture(m=6, d=9)
    exact = build_local_geometry(delta.T @ delta)
    truncated = build_local_geometry(delta.T @ delta, rank=3)
    assert exact.rank == 6
    assert exact.is_exact
    assert truncated.rank == 3
    assert not truncated.is_exact
    assert float(truncated.discarded_eigenvalue_sum) > 0.0


@pytest.mark.parametrize(
    ("dtype", "small_singular_value"),
    [(torch.float64, 1e-8), (torch.float32, 1e-4)],
)
def test_direct_svd_geometry_retains_resolvable_small_modes(dtype, small_singular_value):
    differences = torch.diag(torch.tensor([1.0, small_singular_value], dtype=dtype))
    geometry = build_local_geometry_from_differences(differences)
    assert geometry.rank == 2
    assert geometry.is_exact
    torch.testing.assert_close(
        geometry.coordinates @ geometry.coordinates.T,
        differences.T @ differences,
    )


def test_residual_certificate_bounds_scalar_posterior_error():
    """An early PCG iterate is a valid conservative Gaussian conditional."""

    _, _, operator = _fixture(m=7, d=10, seed=71)
    generator = torch.Generator().manual_seed(83)
    cross = torch.randn(operator.size, generator=generator, dtype=torch.float64)
    solve = solve_reduced_cg(
        operator,
        cross,
        tolerance=1e-14,
        max_iterations=2,
        preconditioner=ReducedKroneckerPreconditioner(operator),
    )
    dense = _explicit_blocks(operator)
    exact_weights = torch.linalg.solve(dense, cross)

    prior_variance = torch.tensor(100.0, dtype=torch.float64)
    conservative_variance = (
        prior_variance
        - torch.dot(cross, solve.solution)
        - torch.dot(solve.solution, solve.residual)
    )
    exact_variance = prior_variance - torch.dot(cross, exact_weights)
    energy_error = torch.dot(
        solve.solution - exact_weights,
        dense @ (solve.solution - exact_weights),
    )
    torch.testing.assert_close(
        conservative_variance - exact_variance,
        energy_error,
        rtol=1e-9,
        atol=1e-9,
    )

    certificate = compute_posterior_certificate(operator, solve, conservative_variance)
    assert certificate.exact_arithmetic_certified
    assert not certificate.floating_point_rigorous
    assert float(energy_error) <= certificate.variance_error_upper_bound
    expected_kl = 0.5 * torch.log(conservative_variance / exact_variance)
    assert float(expected_kl) <= certificate.expected_kl_upper_bound

    truncated_scope = compute_posterior_certificate(
        operator,
        solve,
        conservative_variance,
        basis_is_exact=False,
    )
    assert truncated_scope.solve_certified
    assert not truncated_scope.exact_arithmetic_certified
    assert not truncated_scope.floating_point_rigorous


def test_certificate_explicitly_disclaims_floating_point_roundoff():
    """A recomputed fp32 residual is still not an interval enclosure."""

    coordinates = torch.tensor([[0.7]], dtype=torch.float32)
    alpha = torch.ones(1, 1, dtype=torch.float32)
    function_cholesky = torch.linalg.cholesky(torch.tensor([[1.05]], dtype=torch.float32))
    operator = OrthonormalReducedOperator(
        coordinates,
        alpha,
        -alpha,
        function_cholesky,
        100.0 * torch.eye(1, dtype=torch.float32),
        jitter=0.0,
    )
    rhs = torch.tensor([1.5409960746765137], dtype=torch.float32)
    solve = solve_reduced_cg(
        operator,
        rhs,
        tolerance=1e-12,
        max_iterations=1,
    )
    prior_variance = torch.tensor(0.5, dtype=torch.float32)
    reported_variance = (
        prior_variance - torch.dot(rhs, solve.solution) - torch.dot(solve.solution, solve.residual)
    )
    certificate = compute_posterior_certificate(operator, solve, reported_variance)

    # The represented scalar operator is exactly 101 in real arithmetic.  A
    # float64 diagnostic exposes a tiny rounding violation of the residual
    # bound, so the API must not label the computed result IEEE-rigorous.
    exact_weight = rhs.double() / 101.0
    energy_error = 101.0 * torch.sum((solve.solution.double() - exact_weight) ** 2)
    assert float(energy_error) > certificate.variance_error_upper_bound
    assert certificate.exact_arithmetic_certified
    assert not certificate.floating_point_rigorous


def test_zero_rank_geometry_falls_back_to_value_only_conditional():
    dtype = torch.float64
    m, dimension = 4, 3
    x_condition = torch.zeros(m, dimension, dtype=dtype)
    x_target = torch.zeros(1, dimension, dtype=dtype)
    values = torch.tensor([0.2, -0.1, 0.4, 0.7], dtype=dtype)
    gradients = torch.randn(m, dimension, generator=torch.Generator().manual_seed(9), dtype=dtype)
    prediction = predict_local_value(
        x_condition,
        values,
        gradients,
        x_target,
        lengthscale=torch.tensor([1.3], dtype=dtype),
        outputscale=1.2,
        value_noise_variance=0.05,
        gradient_noise_variance=0.02,
        kernel="matern52",
    )

    kff = 1.2 * torch.ones(m, m, dtype=dtype) + 0.05 * torch.eye(m, dtype=dtype)
    lff = torch.linalg.cholesky(kff + 1e-8 * torch.eye(m, dtype=dtype))
    cross = 1.2 * torch.ones(m, dtype=dtype)
    weights = torch.cholesky_solve(cross[:, None], lff).squeeze(1)
    torch.testing.assert_close(prediction.mean, torch.dot(weights, values))
    torch.testing.assert_close(prediction.variance, 1.2 - torch.dot(cross, weights))
    assert prediction.rank == 0
    assert prediction.basis_is_exact
    assert prediction.solve.converged
    assert prediction.certificate.exact_arithmetic_certified
    assert not prediction.certificate.floating_point_rigorous


def test_zero_initial_cholesky_jitter_escalates_instead_of_looping():
    singular = torch.ones(3, 3, dtype=torch.float64)
    factor = _cholesky_with_jitter(singular, initial_jitter=0.0)
    assert torch.isfinite(factor).all()


def test_metric_matched_noise_is_identity_in_orthonormal_coordinates():
    generator = torch.Generator().manual_seed(103)
    raw = torch.randn(5, 4, generator=generator, dtype=torch.float64)
    lengthscale = torch.tensor([0.7, 1.2, 1.8, 2.3, 0.9], dtype=torch.float64)
    scaled = raw / lengthscale[:, None]
    geometry = build_local_geometry(scaled.T @ scaled)
    projected_noise = _projected_noise_gram(raw, lengthscale, "metric_matched")
    orthonormal_noise = geometry.q_to_z.T @ projected_noise @ geometry.q_to_z
    torch.testing.assert_close(
        orthonormal_noise,
        torch.eye(geometry.rank, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-10,
    )


def test_local_prediction_matches_direct_tera_conditional():
    generator = torch.Generator().manual_seed(47)
    dtype = torch.float64
    m, dimension = 5, 9
    x_condition = torch.randn(m, dimension, generator=generator, dtype=dtype)
    x_target = torch.randn(1, dimension, generator=generator, dtype=dtype)
    values = torch.randn(m, generator=generator, dtype=dtype)
    gradients = torch.randn(m, dimension, generator=generator, dtype=dtype)
    lengthscale = torch.tensor([1.7], dtype=dtype)
    outputscale = torch.tensor(1.3, dtype=dtype)
    value_noise = 0.04
    gradient_noise = 0.03

    prediction = predict_local_value(
        x_condition,
        values,
        gradients,
        x_target,
        lengthscale=lengthscale,
        outputscale=outputscale,
        value_noise_variance=value_noise,
        gradient_noise_variance=gradient_noise,
        kernel="rbf",
        cg_tolerance=1e-11,
        cg_max_iterations=4 * m * m,
        reduced_jitter=0.0,
    )
    assert prediction.solve.converged

    raw_delta = (x_condition - x_target).T
    scaled_delta = raw_delta / lengthscale
    gram = scaled_delta.T @ scaled_delta
    columns = gram.T
    pair_difference = x_condition[:, None, :] - x_condition[None, :, :]
    pair_distance2 = ((pair_difference / lengthscale) ** 2).sum(dim=-1)
    alpha = outputscale * torch.exp(-0.5 * pair_distance2)
    beta = -alpha
    target_alpha = outputscale * torch.exp(-0.5 * torch.diagonal(gram))

    kff = alpha + value_noise * torch.eye(m, dtype=dtype)
    lff = torch.linalg.cholesky(kff + 1e-8 * torch.eye(m, dtype=dtype))
    kfc = outputscale * torch.exp(-0.5 * torch.diagonal(gram))
    beta_star = torch.cholesky_solve(kfc[:, None], lff).squeeze(1)

    pair_columns = columns[:, None, :] - columns[None, :, :]
    bar_k = (-target_alpha[:, None] * columns).reshape(m * m)
    q_cross = (-alpha[:, :, None] * pair_columns).permute(0, 2, 1).reshape(m * m, m)
    blocks = alpha[:, :, None, None] * gram
    blocks = blocks + beta[:, :, None, None] * (
        pair_columns[:, :, :, None] * pair_columns[:, :, None, :]
    )
    diag = torch.arange(m)
    blocks[diag, diag] = blocks[diag, diag] + gradient_noise * (raw_delta.T @ raw_delta)
    g0 = blocks.permute(0, 2, 1, 3).reshape(m * m, m * m)
    q_kff_inv = torch.cholesky_solve(q_cross.T, lff)
    k_delta = g0 - q_cross @ q_kff_inv
    k_delta = 0.5 * (k_delta + k_delta.T)
    conditional_cross = bar_k - q_cross @ beta_star
    weights = torch.linalg.solve(k_delta, conditional_cross)

    function_weights = beta_star - torch.cholesky_solve(
        (q_cross.T @ weights)[:, None],
        lff,
    ).squeeze(1)
    projected_observations = (gradients @ raw_delta).reshape(-1)
    expected_mean = torch.dot(function_weights, values) + torch.dot(
        weights,
        projected_observations,
    )
    expected_variance = outputscale - torch.dot(kfc, beta_star)
    expected_variance = expected_variance - torch.dot(weights, conditional_cross)

    torch.testing.assert_close(prediction.mean, expected_mean, rtol=2e-8, atol=2e-8)
    torch.testing.assert_close(prediction.variance, expected_variance, rtol=2e-8, atol=2e-8)


def _independent_kernel_value(
    first: torch.Tensor,
    second: torch.Tensor,
    lengthscale: torch.Tensor,
    outputscale: torch.Tensor,
    kernel: str,
) -> torch.Tensor:
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
    """Return d/d(second) k and d(first)d(second) k without ORBIT formulas."""

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


def _independent_dense_tera_q_conditional(
    x_condition: torch.Tensor,
    values: torch.Tensor,
    gradients: torch.Tensor,
    x_target: torch.Tensor,
    *,
    lengthscale: torch.Tensor,
    outputscale: torch.Tensor,
    value_noise: float,
    gradient_noise: float,
    kernel: str,
    gradient_noise_model: str,
    function_jitter: float = 1e-8,
    q_jitter: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble and solve TERA's full q-coordinate Gaussian conditional."""

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
    elif gradient_noise_model == "metric_matched":
        ambient_noise_metric = torch.diag(lengthscale.square().reciprocal())
    else:
        raise ValueError(f"unknown gradient noise model: {gradient_noise_model}")
    projected_noise = raw_differences.T @ ambient_noise_metric @ raw_differences
    diagonal = torch.arange(m)
    q_blocks[diagonal, diagonal] = q_blocks[diagonal, diagonal] + gradient_noise * projected_noise

    identity_m = torch.eye(m, dtype=x_condition.dtype)
    function_covariance = function_covariance + (value_noise + function_jitter) * identity_m
    q_covariance = q_blocks.permute(0, 2, 1, 3).reshape(m * m, m * m)
    # TERA's regularizer lives in the original, nonorthogonal q coordinates.
    q_covariance = q_covariance + q_jitter * torch.eye(m * m, dtype=x_condition.dtype)
    observation_covariance = torch.cat(
        [
            torch.cat([function_covariance, function_q_covariance], dim=1),
            torch.cat([function_q_covariance.T, q_covariance], dim=1),
        ],
        dim=0,
    )

    q_observations = (gradients @ raw_differences).reshape(-1)
    observations = torch.cat([values, q_observations])
    target_cross = torch.cat([target_function_covariance, target_q_covariance.reshape(-1)])
    mean = target_cross @ torch.linalg.solve(observation_covariance, observations)
    variance = outputscale - target_cross @ torch.linalg.solve(
        observation_covariance,
        target_cross,
    )
    return mean, variance


def _dense_q_regression_case(
    geometry_kind: str,
    *,
    seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    dtype = torch.float64
    dimension, m = 7, 5
    generator = torch.Generator().manual_seed(seed)
    if geometry_kind == "generic_full_rank":
        raw_differences = torch.randn(
            dimension,
            m,
            generator=generator,
            dtype=dtype,
        )
        expected_rank = m
    elif geometry_kind == "mixed_orientation_rank_deficient":
        ambient_basis, _ = torch.linalg.qr(
            torch.randn(
                dimension,
                3,
                generator=generator,
                dtype=dtype,
            )
        )
        mixed_coordinates = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.8, -0.6],
                [0.0, 1.0, 0.0, -0.5, 0.9],
                [0.0, 0.0, 1.0, 0.4, -0.7],
            ],
            dtype=dtype,
        )
        raw_differences = ambient_basis @ mixed_coordinates
        expected_rank = 3
    else:
        raise ValueError(f"unknown geometry kind: {geometry_kind}")

    x_target = torch.zeros(1, dimension, dtype=dtype)
    x_condition = raw_differences.T.contiguous()
    values = torch.randn(m, generator=generator, dtype=dtype)
    gradients = torch.randn(m, dimension, generator=generator, dtype=dtype)
    lengthscale = torch.linspace(0.65, 1.85, dimension, dtype=dtype)
    return x_condition, values, gradients, x_target, lengthscale, expected_rank


@pytest.mark.parametrize("kernel", ["rbf", "matern52"])
@pytest.mark.parametrize("gradient_noise_model", ["iid", "scaled", "metric_matched"])
@pytest.mark.parametrize(
    "geometry_kind",
    ["generic_full_rank", "mixed_orientation_rank_deficient"],
)
def test_predict_local_value_matches_independent_dense_q_conditional(
    kernel: str,
    gradient_noise_model: str,
    geometry_kind: str,
) -> None:
    seed = (
        1701
        + 101 * ["rbf", "matern52"].index(kernel)
        + 17 * ["iid", "scaled", "metric_matched"].index(gradient_noise_model)
        + ["generic_full_rank", "mixed_orientation_rank_deficient"].index(geometry_kind)
    )
    (
        x_condition,
        values,
        gradients,
        x_target,
        lengthscale,
        expected_rank,
    ) = _dense_q_regression_case(geometry_kind, seed=seed)
    outputscale = torch.tensor(1.4, dtype=torch.float64)
    value_noise = 0.07
    gradient_noise = 0.025

    # Leave function_jitter and reduced_jitter unset to regress their public defaults.
    prediction = predict_local_value(
        x_condition,
        values,
        gradients,
        x_target,
        lengthscale=lengthscale,
        outputscale=outputscale,
        value_noise_variance=value_noise,
        gradient_noise_variance=gradient_noise,
        kernel=kernel,
        gradient_noise_model=gradient_noise_model,
        cg_tolerance=1e-12,
        cg_max_iterations=8 * x_condition.shape[0] ** 2,
    )
    expected_mean, expected_variance = _independent_dense_tera_q_conditional(
        x_condition,
        values,
        gradients,
        x_target,
        lengthscale=lengthscale,
        outputscale=outputscale,
        value_noise=value_noise,
        gradient_noise=gradient_noise,
        kernel=kernel,
        gradient_noise_model=gradient_noise_model,
    )

    assert prediction.rank == expected_rank
    assert prediction.basis_is_exact
    assert prediction.solve.converged
    torch.testing.assert_close(prediction.mean, expected_mean, rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(
        prediction.variance,
        expected_variance,
        rtol=2e-9,
        atol=2e-10,
    )
