"""Base (unpreconditioned) conjugate-gradient solver for the derivative dual (K+Lambda) alpha = y.

Matrix-free: uses the factored kernel MVM (O(N^2 D)). This is the plain CG baseline -- no
preconditioner. The relative residual test is in the original (K+Lambda) norm.
"""

from collections.abc import Callable

import torch

from gp.kernels import DerivKernel

__all__ = ["solve_cg"]


def solve_cg(
    X: torch.Tensor,
    y_full: torch.Tensor,
    lam: torch.Tensor,
    kernel: DerivKernel,
    max_iter: int = 500,
    tol: float = 1e-6,
    col_chunk: int = 4096,
    callback: Callable[[int, torch.Tensor, float], None] | None = None,
) -> torch.Tensor:
    # The only place the operator (K+Lambda) is touched: matrix-free. kernel.mvm is the factored
    # O(N^2 D) derivative-kernel matvec (gp/kernels/base.py) -- it never forms the D x D grad-grad
    # block, let alone the N(D+1) x N(D+1) matrix. lam is the noise diagonal in the same interleaved
    # per-point [value, grad...] order as v.
    def A_mv(v: torch.Tensor) -> torch.Tensor:
        return kernel.mvm(X, X, v, col_chunk=col_chunk) + lam * v

    x = torch.zeros_like(y_full)
    r = y_full - A_mv(x)
    p = r.clone()
    rz_old = torch.dot(r, r)
    b_norm = torch.linalg.norm(y_full)
    for i in range(max_iter):
        Ap = A_mv(p)
        alpha = rz_old / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rel = (torch.linalg.norm(r) / b_norm).item()
        if callback is not None:
            callback(i + 1, x, rel)
        if rel < tol:
            break
        rz_new = torch.dot(r, r)
        p = r + (rz_new / rz_old) * p
        rz_old = rz_new
    return x
