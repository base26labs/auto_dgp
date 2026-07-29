"""The factored/blocked derivative-kernel matvec must equal the dense assembly.

This is the load-bearing correctness test for `gp/kernels`: `DerivKernel.mvm` computes `(K @ vec)`
without ever forming the O(D^2) grad-grad block, so it is only trustworthy if it agrees with the
explicit dense `block_kernel` assembly. fp64 for a tight tolerance.
"""

import pytest
import torch

from gp.kernels import MaternDerivKernel, RBFDerivKernel, block_kernel

KERNELS = [RBFDerivKernel, MaternDerivKernel]


def _kernel(cls, D):
    ls = torch.linspace(0.5, 1.5, D, dtype=torch.float64)  # ARD lengthscales
    return cls(ls, torch.tensor(1.3, dtype=torch.float64))


# ============================================================================
# Factored matvec (DerivKernel.mvm) vs dense block_kernel
# ============================================================================


@pytest.mark.parametrize("cls", KERNELS)
def test_mvm_value_and_grad_matches_dense(cls):
    torch.manual_seed(0)
    nq, nall, D = 7, 11, 4
    Xq = torch.randn(nq, D, dtype=torch.float64)
    Xall = torch.randn(nall, D, dtype=torch.float64)
    vec = torch.randn(nall * (1 + D), dtype=torch.float64)
    kernel = _kernel(cls, D)

    got = kernel.mvm(Xq, Xall, vec, q_grad=True)
    dense = block_kernel(Xq, Xall, kernel) @ vec
    assert torch.allclose(got, dense, atol=1e-10), (got - dense).abs().max()


@pytest.mark.parametrize("cls", KERNELS)
def test_mvm_value_only_matches_dense(cls):
    torch.manual_seed(1)
    nq, nall, D = 5, 9, 3
    Xq = torch.randn(nq, D, dtype=torch.float64)
    Xall = torch.randn(nall, D, dtype=torch.float64)
    vec = torch.randn(nall * (1 + D), dtype=torch.float64)
    kernel = _kernel(cls, D)

    got = kernel.mvm(Xq, Xall, vec, q_grad=False)
    dense = block_kernel(Xq, Xall, kernel, x1_grad=False) @ vec
    assert torch.allclose(got, dense, atol=1e-10), (got - dense).abs().max()


@pytest.mark.parametrize("cls", KERNELS)
def test_mvm_chunking_is_invariant(cls):
    """Column chunking must not change the result."""
    torch.manual_seed(2)
    n, D = 13, 3
    X = torch.randn(n, D, dtype=torch.float64)
    vec = torch.randn(n * (1 + D), dtype=torch.float64)
    kernel = _kernel(cls, D)
    a = kernel.mvm(X, X, vec, col_chunk=4096)
    b = kernel.mvm(X, X, vec, col_chunk=4)
    assert torch.allclose(a, b, atol=1e-11)


# ============================================================================
# Kernel properties
# ============================================================================


@pytest.mark.parametrize("cls", KERNELS)
def test_full_is_symmetric(cls):
    torch.manual_seed(5)
    n, D = 8, 3
    X = torch.randn(n, D, dtype=torch.float64)
    A = _kernel(cls, D).full(X)
    assert torch.allclose(A, A.T, atol=1e-10)


@pytest.mark.parametrize("cls", KERNELS)
def test_diag_matches_full_diagonal(cls):
    torch.manual_seed(6)
    n, D = 8, 4
    X = torch.randn(n, D, dtype=torch.float64)
    kernel = _kernel(cls, D)
    assert torch.allclose(kernel.diag(X), torch.diag(kernel.full(X)), atol=1e-10)
