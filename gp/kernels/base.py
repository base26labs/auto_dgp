"""Derivative-kernel base class: the shared factored matvec, dense assembly, and diagonal.

Every stationary derivative kernel in this project has the SAME rank structure: three
scalar functions of the scaled squared distance -- K0 (value-value), A (value-grad weight and grad-grad
identity part), B (grad-grad outer-product part) -- with

    Cov(f,     f')      =  K0
    Cov(f,    d_b f')   =   A * dl2_b
    Cov(d_a f,  f')     =  -A * dl2_a
    Cov(d_a f, d_b f')  =   A * delta_ab/l_a^2 + B * dl2_a dl2_b .

The Gaussian (RBF) kernel is the special case A = K0, B = -K0. Because of this shared structure, the
O(N^2 D) factored matvec (never forming the O(D^2) grad-grad block) can be written ONCE here; a concrete
kernel supplies only ``scalars(dist2) -> (K0, A, B)`` and the value/gradient self-variance for the
diagonal. This is the DRY core: RBF and Matern-5/2 are ~10 lines each.
"""

from __future__ import annotations

import torch

# ============================================================================
# DerivKernel base class: shared factored matvec, dense assembly, diagonal
# ============================================================================


class DerivKernel:
    """Base class. Subclasses implement ``scalars`` and ``_grad_var_coef``."""

    def __init__(
        self,
        lengthscale: torch.Tensor,
        outputscale: torch.Tensor,
    ) -> None:
        self.lengthscale = lengthscale
        self.outputscale = outputscale

    # ---- kernel-specific (override) -------------------------------------
    def scalars(
        self,
        dist2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (K0, A, B) evaluated at the scaled squared distance ``dist2 = sum_d (x_d-x'_d)^2/l_d^2``."""
        raise NotImplementedError

    def _grad_var_coef(self) -> torch.Tensor:
        """Coefficient c such that the gradient self-variance is c * (1/l_a^2). (RBF: s; Matern-5/2: 5s/3.)"""
        raise NotImplementedError

    # ---- shared ----------------------------------------------------------
    def _ls(self, D: int) -> torch.Tensor:
        l = self.lengthscale.reshape(-1)
        return l.expand(D) if l.numel() == 1 else l

    def diag(self, X: torch.Tensor) -> torch.Tensor:
        """Diagonal of the full derivative kernel, length N*(D+1): [value var, grad var_0, ...] per point."""
        N, D = X.shape
        inv_l2 = 1.0 / (self._ls(D) ** 2)
        blk = torch.cat([self.outputscale.reshape(1), self._grad_var_coef() * inv_l2])
        return blk.repeat(N)

    def full(self, X: torch.Tensor) -> torch.Tensor:
        """Dense N(D+1) x N(D+1) derivative kernel (small N only)."""
        return block_kernel(X, X, self)

    def mvm(
        self,
        Xq: torch.Tensor,
        Xall: torch.Tensor,
        vec: torch.Tensor,
        col_chunk: int = 4096,
        q_grad: bool = True,
    ) -> torch.Tensor:
        """Factored derivative-kernel matvec: block rows of (K @ vec) for queries Xq against columns Xall.

        O(N^2 D), never forming the O(D^2) grad-grad block. With p = v_g/l^2 and m_ij = Xq_i.p_j - qj_j,
        the accumulators (below) reduce to the RBF form when A=K0, B=-K0. ``vec`` has length Nall*(D+1);
        returns nq*(1+D) if q_grad else nq. Validated against the dense assembly in tests/.
        """
        N, D = Xall.shape
        nq = Xq.shape[0]
        inv_l2 = 1.0 / (self._ls(D) ** 2)
        rl = self._ls(D).reciprocal()

        V = vec.reshape(N, 1 + D)
        v0, vg = V[:, 0], V[:, 1:]
        p = vg * inv_l2
        qj = (Xall * p).sum(-1)

        Xq2, Xall2 = Xq * rl, Xall * rl
        sq_q, sq_c = (Xq2 * Xq2).sum(-1), (Xall2 * Xall2).sum(-1)

        SK = Xq.new_zeros(nq)  # sum_j K0_ij v0_j
        SAm = Xq.new_zeros(nq)  # sum_j A_ij m_ij
        if q_grad:
            SAp = Xq.new_zeros(nq, D)  # sum_j A_ij p_ja
            SA0 = Xq.new_zeros(nq)  # sum_j A_ij v0_j
            SBm = Xq.new_zeros(nq)  # sum_j B_ij m_ij
            T = Xq.new_zeros(nq, D)  # sum_j (A_ij v0_j - B_ij m_ij) Xall_ja
        for s in range(0, N, col_chunk):
            e = min(s + col_chunk, N)
            dist2 = sq_q[:, None] + sq_c[None, s:e] - 2 * (Xq2 @ Xall2[s:e].T)
            K0, A, B = self.scalars(dist2)
            m = (Xq @ p[s:e].T) - qj[None, s:e]
            SK += (K0 * v0[None, s:e]).sum(1)
            SAm += (A * m).sum(1)
            if q_grad:
                SAp += A @ p[s:e]
                SA0 += (A * v0[None, s:e]).sum(1)
                SBm += (B * m).sum(1)
                T += (A * v0[None, s:e] - B * m) @ Xall[s:e]
        o0 = SK + SAm
        if not q_grad:
            return o0
        og = SAp + inv_l2[None, :] * (Xq * (SBm - SA0)[:, None] + T)
        return torch.cat([o0[:, None], og], dim=1).reshape(-1)


# ============================================================================
# Dense block assembly (validation and small blocks)
# ============================================================================


def block_kernel(
    X1: torch.Tensor,
    X2: torch.Tensor,
    kernel: DerivKernel | torch.Tensor,
    outputscale: torch.Tensor | bool | None = None,
    x1_grad: bool = True,
    x2_grad: bool = True,
) -> torch.Tensor:
    """Dense derivative-kernel block between point sets. Rows/cols in per-point block order
    [value, grad_0..grad_{D-1}]. For validation and small blocks (e.g. Schwarz block factors).

    Accepts either form: block_kernel(X1, X2, kernel[, x1_grad, x2_grad]) with a DerivKernel, or the legacy
    block_kernel(X1, X2, lengthscale, outputscale[, x1_grad, x2_grad]) which builds an RBF kernel."""
    if not isinstance(kernel, DerivKernel):
        # legacy (lengthscale, outputscale) signature -> RBF block. Imported here (not at module top)
        # to break the gp.kernels.base <-> gp.kernels.rbf import cycle.
        from gp.kernels.rbf import RBFDerivKernel

        kernel = RBFDerivKernel(kernel, outputscale)
    elif outputscale is not None:
        x1_grad, x2_grad = outputscale, x1_grad  # DerivKernel + positional grad flags
    n1, D = X1.shape
    n2 = X2.shape[0]
    inv_l2 = 1.0 / (kernel._ls(D) ** 2)
    diff = X1[:, None, :] - X2[None, :, :]
    dl2 = diff * inv_l2
    dist2 = (diff * dl2).sum(-1)
    K0, A, B = kernel.scalars(dist2)

    r1 = 1 + D if x1_grad else 1
    r2 = 1 + D if x2_grad else 1
    out = X1.new_zeros((n1, r1, n2, r2))
    out[:, 0, :, 0] = K0
    if x2_grad:
        out[:, 0, :, 1:] = A[:, :, None] * dl2
    if x1_grad:
        out[:, 1:, :, 0] = (-A[:, :, None] * dl2).permute(0, 2, 1)
    if x1_grad and x2_grad:
        gg = A[:, :, None, None] * torch.diag(inv_l2)[None, None, :, :] + B[:, :, None, None] * (
            dl2[:, :, :, None] * dl2[:, :, None, :]
        )
        out[:, 1:, :, 1:] = gg.permute(0, 2, 1, 3)
    return out.reshape(n1 * r1, n2 * r2)
