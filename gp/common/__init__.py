"""Shared primitives: assembling the stacked [value, grad...] targets and the noise diagonal
for the derivative dual system (K+Lambda) alpha = y."""

import torch

# ============================================================================
# Target assembly
# ============================================================================


def stack_targets(
    values: torch.Tensor,
    grads: torch.Tensor,
) -> torch.Tensor:
    """Stack [value, grad...] per point into a flat vector of length N*(D+1)."""
    N, D = grads.shape
    return torch.cat([values.reshape(N, 1), grads.reshape(N, D)], dim=1).reshape(-1)


def noise_vector(
    N: int,
    D: int,
    noise: float,
    deriv_noise: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Lambda diagonal, length N*(D+1), per-point [noise, deriv_noise*D]."""
    blk = torch.empty(1 + D, device=device, dtype=dtype)
    blk[0] = noise
    blk[1:] = deriv_noise
    return blk.repeat(N)
