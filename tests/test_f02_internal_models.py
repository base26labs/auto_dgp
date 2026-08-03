from __future__ import annotations

import numpy as np
import pytest
import torch

from data.generate_nbody_confirmatory import TrainNormalization
from data.load_nbody_confirmatory import (
    PreparedConfirmatoryBundle,
    PreparedConfirmatoryDataset,
    PreparedConfirmatorySplit,
)
from experiments.f02_internal_models import (
    TensorConfirmatorySplit,
    build_released_tera_predictor,
    fit_released_tera,
    freeze_tera_parameters,
    predict_orbit,
    predict_released_tera,
    predict_value_only_local_gp,
    prepared_split_to_tensors,
)


@pytest.fixture
def tiny_tensor_split() -> TensorConfirmatorySplit:
    dtype = torch.float64
    X = torch.tensor(
        [
            [-0.8, 0.2, 0.5, -0.1],
            [-0.4, -0.6, 0.1, 0.7],
            [0.0, 0.4, -0.7, 0.3],
            [0.3, -0.2, 0.8, -0.5],
            [0.7, 0.6, -0.3, 0.2],
            [1.0, -0.5, 0.4, 0.9],
        ],
        dtype=dtype,
    )
    value = torch.sin(X[:, 0]) + 0.2 * X[:, 1] ** 2 - 0.1 * X[:, 2] * X[:, 3]
    gradient = torch.stack(
        [
            torch.cos(X[:, 0]),
            0.4 * X[:, 1],
            -0.1 * X[:, 3],
            -0.1 * X[:, 2],
        ],
        dim=1,
    )
    n = X.shape[0]
    return TensorConfirmatorySplit(
        name="tiny",
        source_indices=torch.tensor([9, 2, 7, 1, 8, 3]),
        X=X,
        value=value,
        gradient=gradient,
        trajectory_id=torch.tensor([3, 1, 3, 1, 2, 2]),
        time_index=torch.tensor([0, 0, 1, 1, 0, 1]),
        time_value=torch.arange(n, dtype=dtype) / 10.0,
    )


def _prepared_split(name: str, offset: float) -> PreparedConfirmatorySplit:
    return PreparedConfirmatorySplit(
        name=name,
        source_indices=np.array([5, 1, 4], dtype=np.int64),
        X=np.array([[offset, 1.0], [offset + 2.0, 3.0], [offset + 4.0, 5.0]]),
        E=np.array([10.0 + offset, 20.0 + offset, 30.0 + offset]),
        F=np.array([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]]),
        trajectory_id=np.array([7, 5, 7], dtype=np.int64),
        time_index=np.array([2, 0, 1], dtype=np.int64),
        time_value=np.array([0.2, 0.0, 0.1]),
    )


def test_prepared_bundle_conversion_preserves_every_row_and_order() -> None:
    train = _prepared_split("train", 0.0)
    prepared = PreparedConfirmatoryDataset(
        train=train,
        validation=_prepared_split("validation", 10.0),
        test=_prepared_split("test", 20.0),
        normalization=TrainNormalization(
            x_min=np.zeros(2),
            x_span=np.ones(2),
            energy_mean=0.0,
            energy_std=1.0,
            gradient_scale=np.ones(2),
        ),
        masses=np.ones(1),
    )
    # The converter reads only the already-validated prepared view.
    bundle = PreparedConfirmatoryBundle(loaded=None, prepared=prepared)

    converted = prepared_split_to_tensors(bundle, "train", dtype=torch.float64)

    np.testing.assert_array_equal(converted.source_indices.numpy(), train.source_indices)
    np.testing.assert_array_equal(converted.X.numpy(), train.X)
    np.testing.assert_array_equal(converted.value.numpy(), train.E)
    np.testing.assert_array_equal(converted.gradient.numpy(), train.F)
    np.testing.assert_array_equal(converted.trajectory_id.numpy(), train.trajectory_id)
    np.testing.assert_array_equal(converted.time_index.numpy(), train.time_index)
    np.testing.assert_array_equal(converted.time_value.numpy(), train.time_value)


@pytest.fixture
def zero_step_fit(tiny_tensor_split):
    model = fit_released_tera(
        tiny_tensor_split,
        training_m=2,
        train_epochs=0,
        kernel="rbf",
        lengthscale=0.85,
        outputscale=1.3,
        sigma_f=0.02,
        sigma_g=0.03,
    )
    return model, freeze_tera_parameters(model)


def test_frozen_parameters_reconstruct_released_predictor_at_arbitrary_m(
    tiny_tensor_split,
    zero_step_fit,
) -> None:
    model, parameters = zero_step_fit
    predictor = build_released_tera_predictor(tiny_tensor_split, parameters, m=4)

    assert model.m == 2
    assert predictor.m == 4
    assert parameters.sigma_f == pytest.approx(0.02)
    assert parameters.sigma_g == pytest.approx(0.03)
    assert predictor.data.sigma_f == pytest.approx(np.sqrt(parameters.sigma_f))
    assert predictor.data.sigma_g == pytest.approx(parameters.sigma_g)
    assert not parameters.lengthscale.requires_grad


def test_fit_exposes_exact_update_budget_and_rejects_ambiguous_budget(
    tiny_tensor_split,
) -> None:
    model = fit_released_tera(
        tiny_tensor_split,
        training_m=2,
        train_steps=1,
        train_epochs=0,
        batch_size=2,
        lengthscale=0.85,
        outputscale=1.3,
        sigma_f=0.02,
        sigma_g=0.03,
    )
    assert model.train_steps == 1
    assert model.train_epochs == 0

    with pytest.raises(ValueError, match="at most one"):
        fit_released_tera(
            tiny_tensor_split,
            training_m=2,
            train_steps=1,
            train_epochs=1,
        )


def test_same_prediction_m_released_tera_and_orbit_agree_in_float64(
    tiny_tensor_split,
    zero_step_fit,
) -> None:
    _, parameters = zero_step_fit
    x_eval = torch.tensor(
        [[-0.15, 0.05, 0.2, -0.3], [0.55, 0.1, -0.05, 0.45]],
        dtype=torch.float64,
    )

    tera = predict_released_tera(tiny_tensor_split, x_eval, parameters, m=4)
    orbit = predict_orbit(
        tiny_tensor_split,
        x_eval,
        parameters,
        m=4,
        cg_tolerance=1e-12,
        cg_max_iterations=128,
        use_preconditioner=False,
    )

    assert bool(orbit.details.converged.all())
    torch.testing.assert_close(orbit.mean, tera.mean, rtol=2e-8, atol=2e-8)
    torch.testing.assert_close(
        orbit.latent_variance,
        tera.latent_variance,
        rtol=2e-8,
        atol=2e-8,
    )


def test_observation_variance_adds_scalar_noise_exactly_once(
    tiny_tensor_split,
    zero_step_fit,
) -> None:
    _, parameters = zero_step_fit
    x_eval = tiny_tensor_split.X[:2] + 0.07
    predictions = (
        predict_released_tera(tiny_tensor_split, x_eval, parameters, m=3),
        predict_orbit(
            tiny_tensor_split,
            x_eval,
            parameters,
            m=3,
            cg_tolerance=1e-11,
            cg_max_iterations=96,
        ),
        predict_value_only_local_gp(tiny_tensor_split, x_eval, parameters, m=3),
    )

    for prediction in predictions:
        expected = prediction.latent_variance + parameters.sigma_f
        torch.testing.assert_close(
            prediction.observation_variance,
            expected,
            rtol=0.0,
            atol=0.0,
        )


def test_value_only_local_gp_shapes_and_positive_variances(
    tiny_tensor_split,
    zero_step_fit,
) -> None:
    _, parameters = zero_step_fit
    x_eval = tiny_tensor_split.X[[1, 4]] + 0.03

    prediction = predict_value_only_local_gp(
        tiny_tensor_split,
        x_eval,
        parameters,
        m=4,
    )

    assert prediction.mean.shape == (2,)
    assert prediction.latent_variance.shape == (2,)
    assert prediction.observation_variance.shape == (2,)
    assert bool(torch.isfinite(prediction.mean).all())
    assert bool((prediction.latent_variance > 0.0).all())
    assert bool((prediction.observation_variance > prediction.latent_variance).all())
