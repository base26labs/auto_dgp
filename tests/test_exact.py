"""The base CG solver must reproduce the dense exact solve, and prediction shapes must be correct."""

import pytest
import torch

from gp.cg import solve_cg
from gp.common import noise_vector, stack_targets
from gp.exact import gaussian_nll, predict_grad, predict_value, solve_exact
from gp.kernels import MaternDerivKernel, RBFDerivKernel

KERNELS = [RBFDerivKernel, MaternDerivKernel]


def _problem(cls, N=12, D=3, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(N, D, dtype=torch.float64)
    y = stack_targets(torch.randn(N, dtype=torch.float64), torch.randn(N, D, dtype=torch.float64))
    lam = noise_vector(N, D, 1e-2, 1e-2, X.device, X.dtype)
    kernel = cls(torch.tensor(0.7, dtype=torch.float64), torch.tensor(1.3, dtype=torch.float64))
    return X, y, lam, kernel


# ============================================================================
# CG vs dense exact solve
# ============================================================================


@pytest.mark.parametrize("cls", KERNELS)
def test_cg_matches_dense_solve(cls):
    X, y, lam, kernel = _problem(cls)
    alpha_exact, cond = solve_exact(X, y, lam, kernel)
    alpha_cg = solve_cg(X, y, lam, kernel, max_iter=500, tol=1e-12)
    assert cond < 1e6  # sanity: well-conditioned toy problem
    assert torch.allclose(alpha_cg, alpha_exact, atol=1e-8), (alpha_cg - alpha_exact).abs().max()


@pytest.mark.parametrize("cls", KERNELS)
def test_cg_residual_is_small(cls):
    X, y, lam, kernel = _problem(cls)
    alpha = solve_cg(X, y, lam, kernel, max_iter=500, tol=1e-10)
    resid = kernel.mvm(X, X, alpha) + lam * alpha - y
    rel = (torch.linalg.norm(resid) / torch.linalg.norm(y)).item()
    assert rel < 1e-8, rel


# ============================================================================
# Prediction and metric
# ============================================================================


@pytest.mark.parametrize("cls", KERNELS)
def test_prediction_shapes(cls):
    X, y, lam, kernel = _problem(cls)
    alpha, _ = solve_exact(X, y, lam, kernel)
    Xstar = torch.randn(5, X.shape[1], dtype=torch.float64)
    assert predict_value(Xstar, X, alpha, kernel).shape == (5,)
    assert predict_grad(Xstar, X, alpha, kernel).shape == (5, X.shape[1])


def test_gaussian_nll_matches_closed_form():
    torch.manual_seed(0)
    y = torch.randn(50, dtype=torch.float64)
    mean = torch.randn(50, dtype=torch.float64)
    var = torch.rand(50, dtype=torch.float64) + 0.5
    got = gaussian_nll(y, mean, var)
    expected = (0.5 * torch.log(2 * torch.pi * var) + 0.5 * (y - mean) ** 2 / var).mean().item()
    assert abs(got - expected) < 1e-12
