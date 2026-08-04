"""Focused tests for the evidence returned by the reduced ORBIT CG solve."""

from __future__ import annotations

import math

import pytest
import torch

from gp.orbit.operator import solve_reduced_cg


class _DenseOperator:
    """Small public-API-compatible operator with exact call accounting."""

    def __init__(self, matrix: torch.Tensor) -> None:
        self.matrix = matrix
        self.size = matrix.shape[0]
        self.calls = 0

    def matmul(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.matrix @ value


def _assert_fresh_residual_identity(
    result,
    rhs: torch.Tensor,
    *,
    atol: float = 1e-14,
) -> None:
    """The reported residual must be tied to the reported final A(x)."""

    assert result.residual_is_fresh
    torch.testing.assert_close(
        result.operator_action + result.residual,
        rhs,
        rtol=0.0,
        atol=atol,
    )
    assert result.residual_norm == pytest.approx(float(torch.linalg.norm(result.residual)))
    assert result.recursive_residual_norm == pytest.approx(
        float(torch.linalg.norm(result.recursive_residual))
    )
    expected_relative = result.residual_norm / result.rhs_norm if result.rhs_norm else 0.0
    expected_recursive_relative = (
        result.recursive_residual_norm / result.rhs_norm if result.rhs_norm else 0.0
    )
    assert result.relative_residual == pytest.approx(expected_relative)
    assert result.recursive_relative_residual == pytest.approx(expected_recursive_relative)


def test_converged_result_binds_fresh_operator_action_and_solver_request() -> None:
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    rhs = torch.tensor([1.0, 2.0], dtype=torch.float64)
    operator = _DenseOperator(matrix)

    result = solve_reduced_cg(
        operator,
        rhs,
        tolerance=1e-13,
        max_iterations=4,
        operator_norm_upper_bound=5.1,
    )

    assert result.converged
    assert result.termination_reason == "converged_fresh_residual"
    assert result.requested_tolerance == 1e-13
    assert result.max_iterations == 4
    assert result.operator_norm_upper_bound == 5.1
    assert result.fresh_check_count == 1
    assert result.residual_replacement_count == 0
    assert result.operator_matvecs == operator.calls
    assert result.operator_matvecs == result.iterations + result.fresh_check_count
    torch.testing.assert_close(result.operator_action, matrix @ result.solution)
    _assert_fresh_residual_identity(result, rhs)


def test_zero_rhs_has_complete_zero_cost_fresh_evidence() -> None:
    operator = _DenseOperator(torch.eye(3, dtype=torch.float64))
    rhs = torch.zeros(3, dtype=torch.float64)

    result = solve_reduced_cg(
        operator,
        rhs,
        tolerance=2e-7,
        max_iterations=7,
        operator_norm_upper_bound=1.0,
    )

    assert result.converged
    assert result.termination_reason == "zero_rhs"
    assert result.iterations == 0
    assert result.operator_matvecs == operator.calls == 0
    assert result.preconditioner_applications == 0
    assert result.fresh_check_count == 0
    assert result.residual_replacement_count == 0
    assert result.requested_tolerance == 2e-7
    assert result.max_iterations == 7
    assert result.operator_norm_upper_bound == 1.0
    torch.testing.assert_close(result.operator_action, torch.zeros_like(rhs))
    torch.testing.assert_close(result.recursive_residual, torch.zeros_like(rhs))
    _assert_fresh_residual_identity(result, rhs, atol=0.0)


def test_iteration_cap_returns_a_final_fresh_check() -> None:
    matrix = torch.diag(torch.tensor([1.0, 4.0], dtype=torch.float64))
    rhs = torch.ones(2, dtype=torch.float64)
    operator = _DenseOperator(matrix)

    result = solve_reduced_cg(
        operator,
        rhs,
        tolerance=1e-16,
        max_iterations=1,
    )

    assert not result.converged
    assert result.termination_reason == "maximum_iterations"
    assert result.iterations == result.max_iterations == 1
    assert result.fresh_check_count == 1
    assert result.residual_replacement_count == 0
    assert result.operator_matvecs == operator.calls == 2
    torch.testing.assert_close(result.operator_action, matrix @ result.solution)
    _assert_fresh_residual_identity(result, rhs)


def test_iteration_cap_does_not_apply_an_unused_next_preconditioner() -> None:
    operator = _DenseOperator(torch.diag(torch.tensor([1.0, 4.0], dtype=torch.float64)))
    calls = 0

    def identity(value: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return value

    result = solve_reduced_cg(
        operator,
        torch.ones(2, dtype=torch.float64),
        tolerance=1e-16,
        max_iterations=1,
        preconditioner=identity,
    )

    assert result.termination_reason == "maximum_iterations"
    assert result.preconditioner_applications == calls == 1


@pytest.mark.parametrize("tolerance", [math.nan, math.inf, -math.inf, 0.0, -1.0])
def test_nonfinite_or_nonpositive_tolerance_is_rejected(tolerance: float) -> None:
    operator = _DenseOperator(torch.eye(2, dtype=torch.float64))

    with pytest.raises(ValueError, match="finite and positive"):
        solve_reduced_cg(operator, torch.ones(2, dtype=torch.float64), tolerance=tolerance)

    assert operator.calls == 0


@pytest.mark.parametrize("bound", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_invalid_operator_norm_upper_bound_is_rejected(bound: float) -> None:
    operator = _DenseOperator(torch.eye(2, dtype=torch.float64))

    with pytest.raises(ValueError, match="finite and positive"):
        solve_reduced_cg(
            operator,
            torch.ones(2, dtype=torch.float64),
            operator_norm_upper_bound=bound,
        )

    assert operator.calls == 0


def test_recursive_stop_restarts_from_fresh_residual_and_counts_replacement() -> None:
    class _StatefulScalarOperator:
        size = 1

        def __init__(self) -> None:
            self.calls = 0

        def matmul(self, value: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            # The first recursive residual is zero, but its fresh check uses a
            # deliberately drifted action.  The solver must replace and restart.
            multiplier = (1.0, 0.0, 1.0, 0.5)[self.calls - 1]
            return value * multiplier

    operator = _StatefulScalarOperator()
    preconditioner_calls = 0

    def identity_preconditioner(value: torch.Tensor) -> torch.Tensor:
        nonlocal preconditioner_calls
        preconditioner_calls += 1
        return value

    rhs = torch.ones(1, dtype=torch.float64)
    result = solve_reduced_cg(
        operator,
        rhs,
        tolerance=1e-12,
        max_iterations=2,
        preconditioner=identity_preconditioner,
    )

    assert result.converged
    assert result.termination_reason == "converged_fresh_residual"
    assert result.iterations == 2
    assert result.fresh_check_count == 2
    assert result.residual_replacement_count == 1
    assert result.operator_matvecs == operator.calls == 4
    assert result.preconditioner_applications == preconditioner_calls == 2
    _assert_fresh_residual_identity(result, rhs, atol=0.0)


def test_fresh_and_recursive_residuals_remain_distinct_at_a_drifted_cap() -> None:
    class _DriftedScalarOperator:
        size = 1

        def __init__(self) -> None:
            self.calls = 0

        def matmul(self, value: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            multiplier = (1.0, 0.0)[self.calls - 1]
            return value * multiplier

    operator = _DriftedScalarOperator()
    rhs = torch.ones(1, dtype=torch.float64)
    result = solve_reduced_cg(
        operator,
        rhs,
        tolerance=1e-12,
        max_iterations=1,
    )

    assert not result.converged
    assert result.termination_reason == "maximum_iterations"
    assert result.fresh_check_count == 1
    assert result.residual_replacement_count == 0
    assert result.operator_matvecs == operator.calls == 2
    assert result.recursive_residual_norm == 0.0
    assert result.recursive_relative_residual == 0.0
    assert result.residual_norm == 1.0
    assert result.relative_residual == 1.0
    _assert_fresh_residual_identity(result, rhs, atol=0.0)
