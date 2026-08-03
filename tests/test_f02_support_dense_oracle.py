from __future__ import annotations

import inspect

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
