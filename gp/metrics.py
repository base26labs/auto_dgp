"""Metrics for derivative GP regression."""

from collections.abc import Callable

import torch

# ============================================================================
# Pointwise error metrics
# ============================================================================


def rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def mae(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    return torch.mean((pred - target).abs()).item()


# ============================================================================
# Linear-system / dual-weight diagnostics
# ============================================================================


def rel_residual(
    A_mv: Callable[[torch.Tensor], torch.Tensor],
    alpha: torch.Tensor,
    y_full: torch.Tensor,
) -> float:
    """||(K+Lambda)alpha - y|| / ||y||  -- how well the linear system is solved."""
    return (torch.linalg.norm(A_mv(alpha) - y_full) / torch.linalg.norm(y_full)).item()


def alpha_error(
    alpha: torch.Tensor,
    alpha_star: torch.Tensor,
    A_mv: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, float]:
    """Distance to the exact dual weights.

    Returns L2 relative error, and (if A_mv given) the K-norm relative error
    ||alpha-alpha*||_{K+L} / ||alpha*||_{K+L}, which is the quantity SDD's dual
    objective actually controls (and bounds the sup-norm function error).
    """
    l2 = (torch.linalg.norm(alpha - alpha_star) / torch.linalg.norm(alpha_star)).item()
    if A_mv is None:
        return {"alpha_rel_l2": l2}
    d = alpha - alpha_star
    knorm = torch.sqrt(torch.dot(d, A_mv(d)).clamp_min(0))
    ref = torch.sqrt(torch.dot(alpha_star, A_mv(alpha_star)).clamp_min(1e-30))
    return {"alpha_rel_l2": l2, "alpha_rel_Knorm": (knorm / ref).item()}
