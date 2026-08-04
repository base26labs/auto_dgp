from __future__ import annotations

import torch

from gp.orbit.budgeted import predict_budgeted_guarded_marginals
from gp.orbit.predictor import predict_marginal_values


def _case() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(7)
    x_train = torch.randn(12, 3, generator=generator, dtype=torch.float64)
    value = torch.sin(x_train).sum(dim=1)
    gradient = torch.cos(x_train)
    x_eval = torch.randn(2, 3, generator=generator, dtype=torch.float64)
    return x_train, value, gradient, x_eval


def _model_kwargs() -> dict:
    return {
        "lengthscale": torch.tensor([1.0], dtype=torch.float64),
        "outputscale": 1.2,
        "value_noise_variance": 0.01,
        "gradient_noise_variance": 0.02,
        "kernel": "rbf",
        "cg_tolerance": 1e-10,
        "cg_max_iterations": 300,
    }


def test_eligible_expansion_matches_direct_prediction() -> None:
    x_train, value, gradient, x_eval = _case()
    epsilon = torch.finfo(torch.float32).eps
    result = predict_budgeted_guarded_marginals(
        x_train,
        value,
        gradient,
        x_eval,
        base_m=3,
        expanded_m=5,
        maximum_expanded_rank=3,
        trust_radius_sigma=1e6,
        rank_epsilon=epsilon,
        **_model_kwargs(),
    )
    direct = predict_marginal_values(
        x_train,
        value,
        gradient,
        x_eval,
        m=5,
        rank_epsilon=epsilon,
        include_mean_gradient=True,
        **_model_kwargs(),
    )

    assert bool(result.use_expanded.all())
    assert bool(result.expanded_eligible.all())
    assert bool(result.variance_is_nested.all())
    torch.testing.assert_close(result.mean, direct.mean)
    torch.testing.assert_close(result.variance, direct.variance)
    torch.testing.assert_close(result.mean_gradient, direct.mean_gradient)
    assert bool(result.selected_adjoint_converged.all())


def test_ineligible_expansion_skips_primal_and_reuses_base_branch() -> None:
    x_train, value, gradient, x_eval = _case()
    result = predict_budgeted_guarded_marginals(
        x_train,
        value,
        gradient,
        x_eval,
        base_m=3,
        expanded_m=5,
        maximum_expanded_rank=1,
        trust_radius_sigma=1e6,
        rank_epsilon=torch.finfo(torch.float32).eps,
        **_model_kwargs(),
    )
    direct = predict_marginal_values(
        x_train,
        value,
        gradient,
        x_eval,
        m=3,
        include_mean_gradient=True,
        **_model_kwargs(),
    )

    assert not bool(result.expanded_eligible.any())
    assert not bool(result.use_expanded.any())
    assert bool((result.expanded_iterations == 0).all())
    assert bool(torch.isnan(result.expanded_relative_residuals).all())
    torch.testing.assert_close(result.mean, direct.mean)
    torch.testing.assert_close(result.variance, direct.variance)
    torch.testing.assert_close(result.mean_gradient, direct.mean_gradient)
