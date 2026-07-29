"""Exact dense solve (ground-truth oracle, small N) and posterior mean prediction.

The oracle the iterative methods are tested against."""

import torch

from gp.kernels import DerivKernel

__all__ = [
    "solve_exact",
    "predict_value",
    "predict_grad",
    "gaussian_nll",
]


# ============================================================================
# Dense exact solve (ground-truth oracle, small N)
# ============================================================================


def solve_exact(
    X: torch.Tensor,
    y_full: torch.Tensor,
    lam: torch.Tensor,
    kernel: DerivKernel,
    compute_cond: bool = True,
) -> tuple[torch.Tensor, float | None]:
    """Form the full (N(D+1))^2 system and solve. Returns (alpha, cond). cond is the condition number of
    the dual Hessian K+Lambda; set compute_cond=False to skip the eigendecomposition."""
    A = kernel.full(X) + torch.diag(lam)
    alpha = torch.linalg.solve(A, y_full)
    if not compute_cond:
        return alpha, None
    evals = torch.linalg.eigvalsh(A)
    return alpha, (evals.max() / evals.clamp_min(1e-30).min()).item()


# ============================================================================
# Posterior mean prediction (value and gradient)
# ============================================================================


def predict_value(
    Xstar: torch.Tensor,
    X: torch.Tensor,
    alpha: torch.Tensor,
    kernel: DerivKernel,
    col_chunk: int = 4096,
) -> torch.Tensor:
    """Posterior-mean function value at Xstar: value-row cross-cov @ alpha."""
    return kernel.mvm(Xstar, X, alpha, col_chunk=col_chunk, q_grad=False)


def predict_grad(
    Xstar: torch.Tensor,
    X: torch.Tensor,
    alpha: torch.Tensor,
    kernel: DerivKernel,
    col_chunk: int = 4096,
) -> torch.Tensor:
    """Posterior-mean gradient at Xstar (rows d_a f(x*)). Returns (nstar, D)."""
    D = X.shape[1]
    full = kernel.mvm(Xstar, X, alpha, col_chunk=col_chunk, q_grad=True).reshape(
        Xstar.shape[0], 1 + D
    )
    return full[:, 1:]


# ============================================================================
# Calibration metric
# ============================================================================


def gaussian_nll(
    y: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
) -> float:
    """Mean Gaussian negative log-likelihood (nats)."""
    var = var.clamp_min(1e-12)
    return (0.5 * torch.log(2 * torch.pi * var) + 0.5 * (y - mean) ** 2 / var).mean().item()
