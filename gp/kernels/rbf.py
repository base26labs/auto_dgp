"""Gaussian (RBF / squared-exponential) derivative kernel.

k(x,x') = s * exp(-1/2 * sum_d (x_d-x'_d)^2 / l_d^2). In the shared (K0, A, B) parameterisation
(gp/kernels/base.py) the Gaussian is the special case A = K0, B = -K0. Its infinite smoothness is a
liability for conditioning -- prefer Matern-5/2 -- but it is the classical baseline.
"""

import torch

from gp.kernels.base import DerivKernel


class RBFDerivKernel(DerivKernel):
    def scalars(
        self,
        dist2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        K0 = self.outputscale * torch.exp(-0.5 * dist2)
        return K0, K0, -K0

    def _grad_var_coef(self) -> torch.Tensor:
        # grad self-variance = s / l^2  ->  coef s
        return self.outputscale
