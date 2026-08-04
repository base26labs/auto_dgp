"""Synthetic correctness checks for SPARK-GP."""

import pytest
import torch

from gp.spark.model import (
    BayesianRadialFeatureGP,
    fit_spark,
    prepare_spark,
    prepare_spark_from_residualization,
)
from gp.spark.radial import (
    RadialNystromBasis,
    build_state_design,
    pair_geometry,
    pair_indices,
    pair_weights_from_masses,
)
from gp.spark.residual import (
    position_residual_inputs,
    position_residual_training,
    prepare_hamiltonian_residualization,
    reconstruct_position_prediction,
    reconstruct_position_prediction_suffix,
    residualization_digest,
    residualization_from_numpy,
    residualization_to_numpy,
)
from gp.spark.structure import (
    fit_diagonal_quadratic_mean,
    infer_hamiltonian_split,
    infer_relative_masses,
)

DTYPE = torch.float64


def _shared_pair_potential(
    q: torch.Tensor,
    masses: torch.Tensor,
    *,
    softening: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic common radial law and its Cartesian gradient."""

    n, n_particles, spatial_dims = q.shape
    value = q.new_zeros(n)
    gradient = q.new_zeros(n, n_particles, spatial_dims)
    for first in range(n_particles):
        for second in range(first + 1, n_particles):
            difference = q[:, first] - q[:, second]
            squared_radius = difference.square().sum(dim=1)
            weight = masses[first] * masses[second]
            value -= weight / torch.sqrt(squared_radius + softening**2)
            coefficient = weight / (squared_radius + softening**2).pow(1.5)
            contribution = coefficient[:, None] * difference
            gradient[:, first] += contribution
            gradient[:, second] -= contribution
    return value, gradient.reshape(n, -1)


def test_swapped_block_recovery_and_relative_mass_inference():
    generator = torch.Generator().manual_seed(17)
    n_particles, spatial_dims = 3, 2
    block_width = n_particles * spatial_dims
    n = 96
    masses = torch.tensor([0.7, 1.3, 2.0], dtype=DTYPE)
    q = torch.randn(n, n_particles, spatial_dims, generator=generator, dtype=DTYPE)
    p = torch.randn(n, n_particles, spatial_dims, generator=generator, dtype=DTYPE)
    q = q + torch.tensor([[-1.5, 0.2], [0.4, -0.8], [1.1, 0.7]], dtype=DTYPE)

    slopes = masses.reciprocal().repeat_interleave(spatial_dims)
    intercepts = torch.linspace(-0.2, 0.3, block_width, dtype=DTYPE)
    flat_p = p.reshape(n, -1)
    kinetic_value = (0.5 * slopes * flat_p.square() + intercepts * flat_p).sum(dim=1)
    kinetic_gradient = flat_p * slopes + intercepts
    potential_value, potential_gradient = _shared_pair_potential(q, masses)

    # Deliberately put kinetic coordinates first: discovery must not assume [q, p].
    X = torch.cat((flat_p, q.reshape(n, -1)), dim=1)
    gradient = torch.cat((kinetic_gradient, potential_gradient), dim=1)
    value = kinetic_value + potential_value
    trajectory_id = torch.arange(n) // 8

    split = infer_hamiltonian_split(
        X,
        gradient,
        trajectory_id,
        n_particles=n_particles,
        spatial_dims=spatial_dims,
    )
    assert split.selected_block == 0
    assert torch.equal(split.kinetic_indices, torch.arange(block_width))
    assert split.candidate_scores[0] < 1e-12
    assert split.candidate_scores[1] > 1e-2

    kinetic = fit_diagonal_quadratic_mean(X, gradient, split.kinetic_indices)
    inferred = infer_relative_masses(
        kinetic,
        n_particles=n_particles,
        spatial_dims=spatial_dims,
    )
    expected = masses / masses.mean()
    assert torch.allclose(kinetic.slopes, slopes, rtol=1e-12, atol=1e-12)
    assert torch.allclose(inferred, expected, rtol=1e-12, atol=1e-12)

    prepared = prepare_spark(
        X,
        value,
        gradient,
        trajectory_id,
        n_particles=n_particles,
        spatial_dims=spatial_dims,
    )
    assert prepared.split.selected_block == 0
    assert torch.allclose(prepared.relative_masses, expected, rtol=1e-12, atol=1e-12)
    assert torch.allclose(prepared.pair_weights.sum(), torch.tensor(1.0, dtype=DTYPE))

    model = fit_spark(
        prepared,
        lengthscale_multiplier=0.5,
        rank=16,
        value_noise=1e-3,
        gradient_noise=1e-3,
        point_chunk=24,
    )
    assert model.residualization is prepared.residualization
    assert residualization_digest(model.residualization) == residualization_digest(
        prepared.residualization
    )
    predicted_value, predicted_gradient, predicted_variance = model.predict(X[:5])
    assert predicted_value.shape == (5,)
    assert predicted_gradient.shape == (5, 2 * block_width)
    assert predicted_variance.shape == (5,)
    assert torch.all(torch.isfinite(predicted_value))
    assert torch.all(torch.isfinite(predicted_gradient))
    assert torch.all(predicted_variance > 0)


def test_position_residual_control_uses_same_problem_and_reconstructs_full_state():
    generator = torch.Generator().manual_seed(19)
    n_particles, spatial_dims, n = 2, 1, 48
    masses = torch.tensor([0.8, 1.6], dtype=DTYPE)
    q = torch.randn(n, n_particles, generator=generator, dtype=DTYPE)
    q += torch.tensor([-1.0, 1.0], dtype=DTYPE)
    p = torch.randn(n, n_particles, generator=generator, dtype=DTYPE)
    potential_value, potential_gradient = _shared_pair_potential(
        q[:, :, None],
        masses,
    )
    kinetic_value = 0.5 * (p.square() / masses).sum(dim=1)
    kinetic_gradient = p / masses
    X = torch.cat((q, p), dim=1)
    value = kinetic_value + potential_value
    gradient = torch.cat((potential_gradient, kinetic_gradient), dim=1)
    trajectory_id = torch.arange(n) // 8

    prepared = prepare_spark(
        X,
        value,
        gradient,
        trajectory_id,
        n_particles=n_particles,
        spatial_dims=spatial_dims,
    )
    residual = position_residual_training(prepared)
    query_q = position_residual_inputs(prepared, X)

    assert torch.equal(residual.X, prepared.position_q)
    assert torch.equal(residual.value, prepared.potential_values)
    assert torch.equal(residual.gradient, prepared.potential_gradients)
    assert torch.equal(query_q, prepared.position_q)

    normalized_value = prepared.residual_transform.normalize_value(potential_value)
    normalized_gradient = prepared.residual_transform.normalize_gradient(potential_gradient)
    normalized_variance = torch.linspace(0.1, 0.4, n, dtype=DTYPE)
    reconstructed = reconstruct_position_prediction(
        prepared,
        X,
        normalized_value,
        normalized_gradient,
        normalized_variance,
    )

    assert torch.allclose(reconstructed.value, value, rtol=1e-12, atol=1e-12)
    assert torch.allclose(reconstructed.gradient, gradient, rtol=1e-12, atol=1e-12)
    assert torch.allclose(
        reconstructed.variance,
        prepared.residual_transform.scale.square() * normalized_variance,
    )


def test_final_residualization_freezes_fit_split_and_tracks_exact_source_rows():
    generator = torch.Generator().manual_seed(21)
    n_particles, spatial_dims, n = 2, 1, 64
    masses = torch.tensor([0.7, 1.4], dtype=DTYPE)
    q = torch.randn(n, n_particles, generator=generator, dtype=DTYPE)
    q += torch.tensor([-1.2, 1.1], dtype=DTYPE)
    p = torch.randn(n, n_particles, generator=generator, dtype=DTYPE)
    potential_value, potential_gradient = _shared_pair_potential(q[:, :, None], masses)
    value = 0.5 * (p.square() / masses).sum(dim=1) + potential_value
    gradient = torch.cat((potential_gradient, p / masses), dim=1)
    X = torch.cat((q, p), dim=1)
    trajectory_id = torch.arange(n) // 8

    selection_rows = torch.arange(48)
    selection_state = prepare_hamiltonian_residualization(
        X[:48],
        value[:48],
        gradient[:48],
        trajectory_id[:48],
        n_particles=n_particles,
        spatial_dims=spatial_dims,
        stage="selection",
        source_row_ids=selection_rows,
    )
    final_rows = torch.arange(n)
    final_state = prepare_hamiltonian_residualization(
        X,
        value,
        gradient,
        None,
        n_particles=n_particles,
        spatial_dims=spatial_dims,
        frozen_split=selection_state.split,
        stage="final",
        source_row_ids=final_rows,
    )

    assert selection_state.stage == "selection"
    assert final_state.stage == "final"
    assert selection_state.source_row_ids_sha256 != final_state.source_row_ids_sha256
    assert torch.equal(final_state.split.kinetic_indices, selection_state.split.kinetic_indices)
    assert torch.equal(final_state.split.position_indices, selection_state.split.position_indices)
    assert residualization_digest(selection_state) != residualization_digest(final_state)
    assert not hasattr(final_state, "pairs")

    serialized = residualization_to_numpy(final_state, prefix="final_")
    restored = residualization_from_numpy(serialized, prefix="final_")
    assert residualization_digest(restored) == residualization_digest(final_state)
    assert torch.equal(restored.position_q, final_state.position_q)
    assert torch.equal(restored.potential_values, final_state.potential_values)
    assert torch.equal(restored.potential_gradients, final_state.potential_gradients)
    attached = prepare_spark_from_residualization(restored)
    assert attached.residualization is restored
    assert residualization_digest(attached.residualization) == residualization_digest(final_state)

    gradient_start = 13
    potential_value_prediction = final_state.potential_values
    potential_gradient_prediction = final_state.potential_gradients[gradient_start:]
    potential_variance = torch.ones_like(potential_value_prediction)
    reconstructed = reconstruct_position_prediction_suffix(
        final_state,
        X,
        potential_value_prediction,
        potential_gradient_prediction,
        potential_variance,
        gradient_start=gradient_start,
    )
    assert reconstructed.gradient_start == gradient_start
    assert torch.allclose(reconstructed.value, value, rtol=1e-12, atol=1e-12)
    assert torch.allclose(
        reconstructed.gradient,
        gradient[gradient_start:],
        rtol=1e-12,
        atol=1e-12,
    )


def test_residualization_rejects_duplicate_source_rows():
    X = torch.randn(16, 4, dtype=DTYPE)
    value = X.square().sum(dim=1)
    gradient = 2 * X
    trajectory_id = torch.arange(16) // 4

    with pytest.raises(ValueError, match="source_row_ids must be unique"):
        prepare_hamiltonian_residualization(
            X,
            value,
            gradient,
            trajectory_id,
            n_particles=2,
            spatial_dims=1,
            source_row_ids=torch.zeros(16, dtype=torch.long),
        )


def test_cartesian_design_is_finite_difference_derivative_of_energy_design():
    generator = torch.Generator().manual_seed(23)
    n, n_particles, spatial_dims = 4, 3, 2
    q = torch.randn(n, n_particles * spatial_dims, generator=generator, dtype=DTYPE)
    q = q + torch.tensor([-1.0, 0.2, 0.5, -0.7, 1.4, 0.9], dtype=DTYPE)
    pairs = pair_indices(n_particles)
    masses = torch.tensor([0.8, 1.1, 1.7], dtype=DTYPE)
    weights = pair_weights_from_masses(masses, pairs=pairs)
    geometry = pair_geometry(
        q,
        n_particles=n_particles,
        spatial_dims=spatial_dims,
        pairs=pairs,
        radial_epsilon=1e-10,
    )
    basis = RadialNystromBasis.from_radii(
        geometry.radii,
        lengthscale=0.75,
        rank=8,
    )
    value_design, gradient_design = build_state_design(
        q,
        basis,
        weights,
        n_particles=n_particles,
        spatial_dims=spatial_dims,
        pairs=pairs,
        radial_epsilon=1e-10,
    )

    step = 1e-6
    for coordinate in range(n_particles * spatial_dims):
        plus = q.clone()
        minus = q.clone()
        plus[:, coordinate] += step
        minus[:, coordinate] -= step
        plus_design, _ = build_state_design(
            plus,
            basis,
            weights,
            n_particles=n_particles,
            spatial_dims=spatial_dims,
            pairs=pairs,
            radial_epsilon=1e-10,
        )
        minus_design, _ = build_state_design(
            minus,
            basis,
            weights,
            n_particles=n_particles,
            spatial_dims=spatial_dims,
            pairs=pairs,
            radial_epsilon=1e-10,
        )
        finite_difference = (plus_design - minus_design) / (2 * step)
        assert torch.allclose(
            gradient_design[:, coordinate],
            finite_difference,
            rtol=2e-5,
            atol=2e-6,
        )

    assert torch.all(torch.isfinite(value_design))
    # Pair-distance features enforce zero net force feature-by-feature.
    reshaped = gradient_design.reshape(n, n_particles, spatial_dims, basis.rank)
    assert torch.allclose(reshaped.sum(dim=1), torch.zeros_like(reshaped[:, 0]), atol=1e-12)


def test_hybrid_log_inducing_sites_cover_sparse_lower_fit_support():
    dense_bulk = torch.linspace(1.0, 2.0, 100, dtype=DTYPE)
    sparse_tail = torch.tensor([0.1, 0.4], dtype=DTYPE)
    radii = torch.cat((sparse_tail, dense_bulk))

    quantile = RadialNystromBasis.from_radii(
        radii,
        lengthscale=0.5,
        rank=16,
        selection_strategy="quantile",
    )
    hybrid = RadialNystromBasis.from_radii(
        radii,
        lengthscale=0.5,
        rank=16,
        selection_strategy="hybrid_log",
    )

    assert quantile.selection_strategy == "quantile"
    assert hybrid.selection_strategy == "hybrid_log"
    assert hybrid.rank == 16
    assert torch.equal(hybrid.inducing_radii[[0, -1]], torch.tensor([0.1, 2.0], dtype=DTYPE))
    assert int((hybrid.inducing_radii < 1.0).sum()) > int((quantile.inducing_radii < 1.0).sum())


def test_unknown_inducing_strategy_fails_closed():
    radii = torch.linspace(0.2, 1.2, 8, dtype=DTYPE)

    with pytest.raises(ValueError, match="unknown inducing selection strategy"):
        RadialNystromBasis.from_radii(
            radii,
            lengthscale=0.4,
            rank=4,
            selection_strategy="held_out_optimized",
        )


def test_feature_posterior_matches_direct_dense_bayesian_linear_regression():
    generator = torch.Generator().manual_seed(31)
    n, n_particles, spatial_dims = 9, 3, 1
    q = torch.randn(n, n_particles, generator=generator, dtype=DTYPE)
    q = q + torch.tensor([-1.2, 0.1, 1.4], dtype=DTYPE)
    pairs = pair_indices(n_particles)
    weights = pair_weights_from_masses(
        torch.tensor([0.9, 1.4, 1.8], dtype=DTYPE),
        pairs=pairs,
    )
    geometry = pair_geometry(
        q,
        n_particles=n_particles,
        spatial_dims=spatial_dims,
        pairs=pairs,
        radial_epsilon=1e-10,
    )
    basis = RadialNystromBasis.from_radii(
        geometry.radii,
        lengthscale=0.6,
        rank=7,
    )
    value_design, gradient_design = build_state_design(
        q,
        basis,
        weights,
        n_particles=n_particles,
        spatial_dims=spatial_dims,
        pairs=pairs,
        radial_epsilon=1e-10,
    )
    true_coefficients = torch.linspace(-0.7, 0.9, basis.rank, dtype=DTYPE)
    value = value_design @ true_coefficients + 0.01 * torch.sin(torch.arange(n, dtype=DTYPE))
    gradient = torch.einsum("ndm,m->nd", gradient_design, true_coefficients)
    gradient = gradient + 0.005 * torch.cos(torch.arange(n * n_particles, dtype=DTYPE)).reshape(
        n, n_particles
    )
    value_noise, gradient_noise = 0.07, 0.11

    fitted = BayesianRadialFeatureGP.fit(
        q,
        value,
        gradient,
        basis=basis,
        pair_weights=weights,
        pairs=pairs,
        n_particles=n_particles,
        spatial_dims=spatial_dims,
        value_noise=value_noise,
        gradient_noise=gradient_noise,
        radial_epsilon=1e-10,
        point_chunk=3,
    )

    flat_gradient_design = gradient_design.reshape(-1, basis.rank)
    precision = torch.eye(basis.rank, dtype=DTYPE)
    precision += value_design.T @ value_design / value_noise
    precision += flat_gradient_design.T @ flat_gradient_design / gradient_noise
    right_hand_side = value_design.T @ value / value_noise
    right_hand_side += flat_gradient_design.T @ gradient.reshape(-1) / gradient_noise
    expected_mean = torch.linalg.solve(precision, right_hand_side)
    assert torch.allclose(fitted.posterior_mean, expected_mean, rtol=2e-11, atol=2e-11)

    qstar = torch.tensor([[-0.8, 0.4, 1.7], [-1.5, -0.2, 0.9]], dtype=DTYPE)
    test_value_design, test_gradient_design = build_state_design(
        qstar,
        basis,
        weights,
        n_particles=n_particles,
        spatial_dims=spatial_dims,
        pairs=pairs,
        radial_epsilon=1e-10,
    )
    actual_value, actual_gradient, actual_variance = fitted.predict(
        qstar,
        point_chunk=1,
        include_nystrom_residual=False,
    )
    expected_value = test_value_design @ expected_mean
    expected_gradient = torch.einsum("ndm,m->nd", test_gradient_design, expected_mean)
    expected_variance = (
        test_value_design * torch.linalg.solve(precision, test_value_design.T).T
    ).sum(dim=1)
    assert torch.allclose(actual_value, expected_value, rtol=2e-11, atol=2e-11)
    assert torch.allclose(actual_gradient, expected_gradient, rtol=2e-11, atol=2e-11)
    assert torch.allclose(actual_variance, expected_variance, rtol=2e-11, atol=2e-11)
    assert torch.all(actual_variance > 0)

    _, _, sparse_variance = fitted.predict(qstar, point_chunk=1)
    assert torch.all(sparse_variance >= actual_variance)


def test_structure_discovery_fails_closed_when_both_blocks_are_affine():
    generator = torch.Generator().manual_seed(41)
    first = torch.randn(48, 2, generator=generator, dtype=DTYPE)
    X = torch.cat((first, first.clone()), dim=1)
    gradient = 2.5 * X - 0.3
    trajectory_id = torch.arange(48) // 8

    with pytest.raises(RuntimeError, match="no unambiguous affine-gradient block"):
        infer_hamiltonian_split(
            X,
            gradient,
            trajectory_id,
            n_particles=2,
            spatial_dims=1,
        )


def test_pair_design_is_particle_permutation_equivariant():
    generator = torch.Generator().manual_seed(43)
    n, n_particles, spatial_dims = 5, 4, 2
    q = torch.randn(n, n_particles, spatial_dims, generator=generator, dtype=DTYPE)
    q = q + torch.arange(n_particles, dtype=DTYPE)[None, :, None]
    masses = torch.tensor([0.7, 1.1, 1.6, 2.2], dtype=DTYPE)
    flat_q = q.reshape(n, -1)
    pairs = pair_indices(n_particles)
    weights = pair_weights_from_masses(masses, pairs=pairs)
    geometry = pair_geometry(
        flat_q,
        n_particles=n_particles,
        spatial_dims=spatial_dims,
        pairs=pairs,
        radial_epsilon=1e-10,
    )
    basis = RadialNystromBasis.from_radii(
        geometry.radii,
        lengthscale=0.8,
        rank=12,
    )
    value, gradient = build_state_design(
        flat_q,
        basis,
        weights,
        n_particles=n_particles,
        spatial_dims=spatial_dims,
        pairs=pairs,
        radial_epsilon=1e-10,
    )

    permutation = torch.tensor([2, 0, 3, 1])
    permuted_q = q[:, permutation].reshape(n, -1)
    permuted_pairs = pair_indices(n_particles)
    permuted_weights = pair_weights_from_masses(
        masses[permutation],
        pairs=permuted_pairs,
    )
    permuted_value, permuted_gradient = build_state_design(
        permuted_q,
        basis,
        permuted_weights,
        n_particles=n_particles,
        spatial_dims=spatial_dims,
        pairs=permuted_pairs,
        radial_epsilon=1e-10,
    )

    assert torch.allclose(permuted_value, value, rtol=1e-11, atol=1e-11)
    expected_gradient = gradient.reshape(n, n_particles, spatial_dims, basis.rank)[:, permutation]
    assert torch.allclose(
        permuted_gradient.reshape(n, n_particles, spatial_dims, basis.rank),
        expected_gradient,
        rtol=1e-11,
        atol=1e-11,
    )


def test_three_dimensional_pair_features_have_zero_torque():
    generator = torch.Generator().manual_seed(47)
    n, n_particles, spatial_dims = 4, 4, 3
    q = torch.randn(n, n_particles, spatial_dims, generator=generator, dtype=DTYPE)
    q = q + torch.arange(n_particles, dtype=DTYPE)[None, :, None]
    flat_q = q.reshape(n, -1)
    pairs = pair_indices(n_particles)
    weights = pair_weights_from_masses(
        torch.tensor([0.8, 1.0, 1.4, 1.9], dtype=DTYPE),
        pairs=pairs,
    )
    geometry = pair_geometry(
        flat_q,
        n_particles=n_particles,
        spatial_dims=spatial_dims,
        pairs=pairs,
        radial_epsilon=1e-10,
    )
    basis = RadialNystromBasis.from_radii(
        geometry.radii,
        lengthscale=0.9,
        rank=10,
    )
    _, gradient = build_state_design(
        flat_q,
        basis,
        weights,
        n_particles=n_particles,
        spatial_dims=spatial_dims,
        pairs=pairs,
        radial_epsilon=1e-10,
    )
    gradient = gradient.reshape(n, n_particles, spatial_dims, basis.rank)
    torque = torch.cross(q[..., None].expand_as(gradient), gradient, dim=2).sum(dim=1)

    assert torch.allclose(torque, torch.zeros_like(torque), atol=2e-11)
