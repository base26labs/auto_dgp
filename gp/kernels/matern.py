"""Matern-5/2 derivative kernel.

Twice differentiable (the minimum smoothness for a well-defined grad-grad block) AND better conditioned
than the Gaussian (polynomial, not super-exponential, spectral decay) -> typically fewer CG iterations.
With w = sqrt(5) r and E = e^{-w} (r = scaled distance):

    K0 = s(1 + w + w^2/3) E,     A = (5s/3)(1 + w) E,     B = -(25s/3) E.

(The Gaussian is the A=K0, B=-K0 limit.) Should be the default derivative kernel.
"""

import torch

from gp.kernels.base import DerivKernel

_SQRT5 = 5.0**0.5


class MaternDerivKernel(DerivKernel):
    def scalars(
        self,
        dist2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = torch.sqrt(
            dist2.clamp_min(1e-12)
        )  # eps floor: sqrt has infinite grad at 0 -> inf*0=NaN when autograd flows through the
        #   kernel. Off-diagonal (dist2>1e-12) unchanged; the diagonal's lengthscale-gradient is 0
        #   anyway, so the value is unaffected.
        w = _SQRT5 * r
        E = torch.exp(-w)
        s = self.outputscale
        K0 = s * (1.0 + w + w * w / 3.0) * E
        A = (5.0 / 3.0) * s * (1.0 + w) * E
        B = -(25.0 / 3.0) * s * E
        return K0, A, B

    def _grad_var_coef(self) -> torch.Tensor:
        # grad self-variance = (5/3) s / l^2  ->  coef (5/3) s
        return (5.0 / 3.0) * self.outputscale
