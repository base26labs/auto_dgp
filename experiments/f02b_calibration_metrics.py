"""Pure numerical metrics for development-only F02b calibration.

The functions in this module do not read labels, files, or mutable global
state.  They validate tensor precision and geometry explicitly and return only
JSON-safe Python scalars, lists, dictionaries, and ``None``.  Invalid inputs
raise instead of being clipped, cast, or silently omitted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import torch

_SUPPORTED_DTYPES = {torch.float32, torch.float64}
_CONSTRAINT_LABELS = (
    "position_x",
    "position_y",
    "position_z",
    "momentum_x",
    "momentum_y",
    "momentum_z",
)


class CalibrationMetricInputError(ValueError):
    """Raised when a calibration metric cannot be computed fail-closed."""


def _require_real_tensor(
    value: object,
    label: str,
    *,
    ndim: int | None = None,
    nonempty: bool = True,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise CalibrationMetricInputError(f"{label} must be a torch.Tensor")
    if value.dtype not in _SUPPORTED_DTYPES:
        raise CalibrationMetricInputError(f"{label} must have dtype float32 or float64")
    if ndim is not None and value.ndim != ndim:
        raise CalibrationMetricInputError(f"{label} must be {ndim}-dimensional")
    if nonempty and value.numel() == 0:
        raise CalibrationMetricInputError(f"{label} must be nonempty")
    if not bool(torch.isfinite(value).all().item()):
        raise CalibrationMetricInputError(f"{label} must contain only finite values")
    return value


def _require_same_tensor_contract(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    reference_label: str,
    candidate_label: str,
) -> None:
    if reference.shape != candidate.shape:
        raise CalibrationMetricInputError(
            f"{reference_label} and {candidate_label} must have identical shapes"
        )
    if reference.dtype != candidate.dtype:
        raise CalibrationMetricInputError(
            f"{reference_label} and {candidate_label} must have identical dtypes"
        )
    if reference.device != candidate.device:
        raise CalibrationMetricInputError(
            f"{reference_label} and {candidate_label} must be on the same device"
        )


def _require_canonical_comparison_tensor(value: torch.Tensor, label: str) -> None:
    if value.dtype != torch.float64 or value.device.type != "cpu":
        raise CalibrationMetricInputError(
            f"{label} must use the canonical CPU float64 comparison representation"
        )


def _positive_scalar(
    value: float | torch.Tensor,
    label: str,
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.dtype != like.dtype:
            raise CalibrationMetricInputError(f"{label} must have dtype {like.dtype}")
        if value.device != like.device:
            raise CalibrationMetricInputError(f"{label} must be on device {like.device}")
        if value.numel() != 1:
            raise CalibrationMetricInputError(f"{label} must be a scalar")
        scalar = value.reshape(())
    else:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise CalibrationMetricInputError(f"{label} must be a real scalar")
        scalar = like.new_tensor(float(value))
    if not bool(torch.isfinite(scalar).item()) or float(scalar) <= 0.0:
        raise CalibrationMetricInputError(f"{label} must be finite and strictly positive")
    return scalar


def _nonnegative_scalar(
    value: float | torch.Tensor,
    label: str,
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.dtype != like.dtype:
            raise CalibrationMetricInputError(f"{label} must have dtype {like.dtype}")
        if value.device != like.device:
            raise CalibrationMetricInputError(f"{label} must be on device {like.device}")
        if value.numel() != 1:
            raise CalibrationMetricInputError(f"{label} must be a scalar")
        scalar = value.reshape(())
    else:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise CalibrationMetricInputError(f"{label} must be a real scalar")
        scalar = like.new_tensor(float(value))
    if not bool(torch.isfinite(scalar).item()) or float(scalar) < 0.0:
        raise CalibrationMetricInputError(f"{label} must be finite and nonnegative")
    return scalar


def _finite_float(value: torch.Tensor | float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise CalibrationMetricInputError(f"{label} is nonfinite")
    return result


def _finite_float_list(value: torch.Tensor, label: str) -> list[float]:
    if not bool(torch.isfinite(value).all().item()):
        raise CalibrationMetricInputError(f"{label} is nonfinite")
    return [float(item) for item in value.detach().cpu().tolist()]


def _require_explicit_compute_dtype(
    compute_dtype: object,
    label: str,
    tensors: Sequence[tuple[torch.Tensor, str]],
) -> torch.dtype:
    if compute_dtype not in (torch.float32, torch.float64):
        raise CalibrationMetricInputError(f"{label} must be torch.float32 or torch.float64")
    for tensor, tensor_label in tensors:
        if tensor.dtype != compute_dtype:
            raise CalibrationMetricInputError(
                f"{label} must equal the actual dtype of {tensor_label}"
            )
    return compute_dtype


def _require_same_dtype_and_device(
    tensors: Sequence[tuple[torch.Tensor, str]],
) -> None:
    reference, reference_label = tensors[0]
    for candidate, candidate_label in tensors[1:]:
        if candidate.dtype != reference.dtype:
            raise CalibrationMetricInputError(
                f"{reference_label} and {candidate_label} must have identical dtypes"
            )
        if candidate.device != reference.device:
            raise CalibrationMetricInputError(
                f"{reference_label} and {candidate_label} must be on the same device"
            )


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _stable_vector_norm_2(value: torch.Tensor, label: str) -> torch.Tensor:
    """Compute a scaled Euclidean norm without avoidable square underflow."""

    scale = torch.max(torch.abs(value))
    if float(scale) == 0.0:
        return value.new_zeros(())
    scaled = value / scale
    norm = scale * torch.sqrt(torch.sum(scaled * scaled))
    _finite_float(norm, label)
    return norm


def _dense_solve_error_record(
    residual: torch.Tensor,
    rhs: torch.Tensor,
    solution: torch.Tensor,
    operator_norm: torch.Tensor,
    *,
    residual_compute_dtype: torch.dtype,
) -> dict[str, Any]:
    residual_norm = _stable_vector_norm_2(residual, "residual_norm_2")
    rhs_norm = _stable_vector_norm_2(rhs, "rhs_norm_2")
    solution_norm = _stable_vector_norm_2(solution, "solution_norm_2")
    scalars = (
        (residual_norm, "residual_norm_2"),
        (rhs_norm, "rhs_norm_2"),
        (solution_norm, "solution_norm_2"),
        (operator_norm, "operator_norm_2"),
    )
    for value, label in scalars:
        _finite_float(value, label)

    floor = torch.finfo(residual_compute_dtype).tiny
    floor_tensor = residual.new_tensor(floor)
    relative_denominator = torch.maximum(rhs_norm, floor_tensor)
    backward_scale = operator_norm * solution_norm + rhs_norm
    _finite_float(backward_scale, "backward_error_scale")
    backward_denominator = torch.maximum(backward_scale, floor_tensor)
    relative_residual = residual_norm / relative_denominator
    backward_error = residual_norm / backward_denominator
    return {
        "system_dimension": int(rhs.numel()),
        "residual_compute_dtype": _dtype_name(residual_compute_dtype),
        "residual_compute_device": str(residual.device),
        "normalization_floor": floor,
        "residual_source": "recomputed_as_b_minus_A_matmul_x",
        "operator_norm_source": "dense_matrix_spectral_norm",
        "operator_norm_is_upper_bound": False,
        "operator_norm_2": _finite_float(operator_norm, "operator_norm_2"),
        "solution_norm_2": _finite_float(solution_norm, "solution_norm_2"),
        "rhs_norm_2": _finite_float(rhs_norm, "rhs_norm_2"),
        "residual_norm_2": _finite_float(residual_norm, "residual_norm_2"),
        "relative_residual_denominator": _finite_float(
            relative_denominator,
            "relative_residual_denominator",
        ),
        "backward_error_denominator": _finite_float(
            backward_denominator,
            "backward_error_denominator",
        ),
        "relative_residual": _finite_float(relative_residual, "relative_residual"),
        "normwise_backward_error": _finite_float(
            backward_error,
            "normwise_backward_error",
        ),
    }


def dense_solve_error_metrics(
    A: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    *,
    residual_compute_dtype: torch.dtype,
) -> dict[str, Any]:
    """Recompute dense linear-solve residual and normwise backward error.

    No input is cast or moved.  ``residual_compute_dtype`` is an explicit
    assertion about the represented system and must exactly match every input.
    """

    A = _require_real_tensor(A, "A", ndim=2)
    b = _require_real_tensor(b, "b", ndim=1)
    x = _require_real_tensor(x, "x", ndim=1)
    _require_same_dtype_and_device(((A, "A"), (b, "b"), (x, "x")))
    _require_explicit_compute_dtype(
        residual_compute_dtype,
        "residual_compute_dtype",
        ((A, "A"), (b, "b"), (x, "x")),
    )
    if A.shape[0] != A.shape[1]:
        raise CalibrationMetricInputError("A must be square")
    dimension = A.shape[0]
    if b.shape != (dimension,) or x.shape != (dimension,):
        raise CalibrationMetricInputError("b and x must have shape (A.shape[0],)")

    residual = b - A @ x
    if not bool(torch.isfinite(residual).all().item()):
        raise CalibrationMetricInputError("dense residual recomputation produced a nonfinite value")
    operator_norm = torch.linalg.matrix_norm(A, ord=2)
    result = _dense_solve_error_record(
        residual,
        b,
        x,
        operator_norm,
        residual_compute_dtype=residual_compute_dtype,
    )
    result["dense_operator_spectral_norm"] = result["operator_norm_2"]
    return result


def dense_solve_frobenius_error_metrics(
    A: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    *,
    residual_compute_dtype: torch.dtype,
) -> dict[str, Any]:
    """Recompute a dense solve's residual and Frobenius-norm backward error.

    This is the registered large full-q variant.  It retains exact dense
    residual arithmetic but uses ``||A||_F`` as the matrix norm, avoiding an
    unnecessary cubic-time SVD of a matrix that has already required a dense
    Cholesky factorization.  The returned field names and provenance make the
    norm choice explicit; this function never labels the Frobenius norm as a
    spectral norm.
    """

    A = _require_real_tensor(A, "A", ndim=2)
    b = _require_real_tensor(b, "b", ndim=1)
    x = _require_real_tensor(x, "x", ndim=1)
    _require_same_dtype_and_device(((A, "A"), (b, "b"), (x, "x")))
    _require_explicit_compute_dtype(
        residual_compute_dtype,
        "residual_compute_dtype",
        ((A, "A"), (b, "b"), (x, "x")),
    )
    if A.shape[0] != A.shape[1]:
        raise CalibrationMetricInputError("A must be square")
    dimension = A.shape[0]
    if b.shape != (dimension,) or x.shape != (dimension,):
        raise CalibrationMetricInputError("b and x must have shape (A.shape[0],)")

    residual = b - A @ x
    if not bool(torch.isfinite(residual).all().item()):
        raise CalibrationMetricInputError(
            "dense residual recomputation produced a nonfinite value"
        )
    operator_norm = torch.linalg.matrix_norm(A, ord="fro")
    result = _dense_solve_error_record(
        residual,
        b,
        x,
        operator_norm,
        residual_compute_dtype=residual_compute_dtype,
    )
    result["operator_norm_source"] = "dense_matrix_frobenius_norm"
    result["backward_error_matrix_norm"] = "frobenius"
    result["operator_frobenius_norm"] = result.pop("operator_norm_2")
    return result


def matrix_free_solve_error_metrics(
    residual: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    *,
    operator_norm_upper_bound: float | torch.Tensor,
    residual_compute_dtype: torch.dtype,
) -> dict[str, Any]:
    """Bound solve error from caller-claimed matrix-free evidence.

    ``residual`` is only a caller claim that it equals ``b - A(x)``; this
    function cannot establish its freshness or independently apply ``A``.
    Likewise, ``operator_norm_upper_bound`` is an externally asserted bound,
    not a norm computed or verified here.  Consequently the quantity formed
    with that upper bound is only a *lower* bound on the exact normwise
    backward error, never a generic backward-error certificate.

    Conditional on the residual claim, ``A(x) = b - residual`` supplies a
    lower bound on ``||A||_2`` and hence an upper bound on the exact normwise
    backward error.  A fixed dtype ``tiny`` floor handles zero and subnormal
    normalization scales.  For exactly zero ``x``, the residual claim must
    satisfy ``b - residual == 0`` exactly.
    """

    residual = _require_real_tensor(residual, "residual", ndim=1)
    b = _require_real_tensor(b, "b", ndim=1)
    x = _require_real_tensor(x, "x", ndim=1)
    _require_same_tensor_contract(residual, b, "residual", "b")
    _require_same_tensor_contract(residual, x, "residual", "x")
    _require_explicit_compute_dtype(
        residual_compute_dtype,
        "residual_compute_dtype",
        ((residual, "residual"), (b, "b"), (x, "x")),
    )
    operator_norm_upper = _positive_scalar(
        operator_norm_upper_bound,
        "operator_norm_upper_bound",
        like=residual,
    )

    claimed_operator_action = b - residual
    if not bool(torch.isfinite(claimed_operator_action).all().item()):
        raise CalibrationMetricInputError(
            "b - residual produced a nonfinite claimed operator action"
        )
    solution_is_exact_zero = int(torch.count_nonzero(x).item()) == 0
    if solution_is_exact_zero and not torch.equal(b, residual):
        raise CalibrationMetricInputError(
            "caller-claimed residual is inconsistent with exactly zero x: "
            "b - residual must be exactly zero"
        )

    residual_norm = _stable_vector_norm_2(residual, "residual_norm_2")
    rhs_norm = _stable_vector_norm_2(b, "rhs_norm_2")
    solution_norm = _stable_vector_norm_2(x, "solution_norm_2")
    claimed_action_norm = _stable_vector_norm_2(
        claimed_operator_action,
        "claimed_operator_action_norm_2",
    )
    for value, label in (
        (residual_norm, "residual_norm_2"),
        (rhs_norm, "rhs_norm_2"),
        (solution_norm, "solution_norm_2"),
        (claimed_action_norm, "claimed_operator_action_norm_2"),
        (operator_norm_upper, "externally_asserted_operator_norm_upper_bound"),
    ):
        _finite_float(value, label)

    floor = torch.finfo(residual_compute_dtype).tiny
    floor_tensor = residual.new_tensor(floor)
    relative_denominator = torch.maximum(rhs_norm, floor_tensor)
    operator_lower_denominator = torch.maximum(solution_norm, floor_tensor)
    operator_norm_lower = claimed_action_norm / operator_lower_denominator

    asserted_action_upper = operator_norm_upper * solution_norm
    _finite_float(asserted_action_upper, "asserted_operator_action_norm_upper_bound")
    if bool((asserted_action_upper < claimed_action_norm).item()):
        raise CalibrationMetricInputError(
            "externally asserted operator norm upper bound contradicts the "
            "caller-claimed operator action"
        )

    lower_bound_scale = asserted_action_upper + rhs_norm
    upper_bound_scale = claimed_action_norm + rhs_norm
    for value, label in (
        (lower_bound_scale, "exact_backward_error_lower_bound_scale"),
        (upper_bound_scale, "exact_backward_error_upper_bound_scale"),
        (operator_norm_lower, "operator_norm_lower_bound_from_claimed_action"),
    ):
        _finite_float(value, label)
    lower_bound_denominator = torch.maximum(lower_bound_scale, floor_tensor)
    upper_bound_denominator = torch.maximum(upper_bound_scale, floor_tensor)
    relative_residual = residual_norm / relative_denominator
    backward_error_lower = residual_norm / lower_bound_denominator
    backward_error_upper = residual_norm / upper_bound_denominator

    return {
        "system_dimension": int(b.numel()),
        "residual_compute_dtype": _dtype_name(residual_compute_dtype),
        "residual_compute_device": str(residual.device),
        "normalization_floor": floor,
        "residual_provenance": "caller_claimed",
        "operator_norm_upper_bound_provenance": "externally_asserted",
        "exact_normwise_backward_error_lower_bound_condition": (
            "caller_claimed_residual_and_externally_asserted_operator_norm_upper_bound"
        ),
        "exact_normwise_backward_error_upper_bound_condition": "caller_claimed_residual",
        "externally_asserted_operator_norm_upper_bound": _finite_float(
            operator_norm_upper,
            "externally_asserted_operator_norm_upper_bound",
        ),
        "solution_is_exact_zero": solution_is_exact_zero,
        "solution_norm_2": _finite_float(solution_norm, "solution_norm_2"),
        "rhs_norm_2": _finite_float(rhs_norm, "rhs_norm_2"),
        "residual_norm_2": _finite_float(residual_norm, "residual_norm_2"),
        "claimed_operator_action_norm_2": _finite_float(
            claimed_action_norm,
            "claimed_operator_action_norm_2",
        ),
        "relative_residual_denominator": _finite_float(
            relative_denominator,
            "relative_residual_denominator",
        ),
        "operator_norm_lower_bound_denominator": _finite_float(
            operator_lower_denominator,
            "operator_norm_lower_bound_denominator",
        ),
        "operator_norm_lower_bound_from_claimed_action": _finite_float(
            operator_norm_lower,
            "operator_norm_lower_bound_from_claimed_action",
        ),
        "exact_normwise_backward_error_lower_bound_denominator": _finite_float(
            lower_bound_denominator,
            "exact_normwise_backward_error_lower_bound_denominator",
        ),
        "exact_normwise_backward_error_upper_bound_denominator": _finite_float(
            upper_bound_denominator,
            "exact_normwise_backward_error_upper_bound_denominator",
        ),
        "relative_residual": _finite_float(relative_residual, "relative_residual"),
        "exact_normwise_backward_error_lower_bound": _finite_float(
            backward_error_lower,
            "exact_normwise_backward_error_lower_bound",
        ),
        "exact_normwise_backward_error_upper_bound": _finite_float(
            backward_error_upper,
            "exact_normwise_backward_error_upper_bound",
        ),
    }


def cholesky_backward_error_metrics(
    A: torch.Tensor,
    L: torch.Tensor,
    *,
    compute_dtype: torch.dtype,
) -> dict[str, Any]:
    """Report the relative factorization residual for a Cholesky factor.

    ``L`` must be exactly lower triangular with a strictly positive diagonal;
    inputs that are merely square roots of ``A`` are not Cholesky factors.
    The factorization residual is exactly ``A - L @ L.T`` in the
    caller-declared input dtype and device.  Inaccurate valid-form factors are
    measured rather than repaired.
    """

    A = _require_real_tensor(A, "A", ndim=2)
    L = _require_real_tensor(L, "L", ndim=2)
    _require_same_tensor_contract(A, L, "A", "L")
    _require_explicit_compute_dtype(
        compute_dtype,
        "compute_dtype",
        ((A, "A"), (L, "L")),
    )
    if A.shape[0] != A.shape[1]:
        raise CalibrationMetricInputError("A and L must be square")
    if int(torch.count_nonzero(torch.triu(L, diagonal=1)).item()) != 0:
        raise CalibrationMetricInputError("L must be exactly lower triangular")
    if bool((torch.diagonal(L) <= 0.0).any().item()):
        raise CalibrationMetricInputError("L diagonal must be strictly positive")

    residual = A - L @ L.T
    if not bool(torch.isfinite(residual).all().item()):
        raise CalibrationMetricInputError(
            "Cholesky residual recomputation produced a nonfinite value"
        )
    matrix_spectral_norm = torch.linalg.matrix_norm(A, ord=2)
    matrix_frobenius_norm = torch.linalg.matrix_norm(A, ord="fro")
    residual_spectral_norm = torch.linalg.matrix_norm(residual, ord=2)
    residual_frobenius_norm = torch.linalg.matrix_norm(residual, ord="fro")
    for value, label in (
        (matrix_spectral_norm, "matrix_spectral_norm"),
        (matrix_frobenius_norm, "matrix_frobenius_norm"),
        (residual_spectral_norm, "residual_spectral_norm"),
        (residual_frobenius_norm, "residual_frobenius_norm"),
    ):
        _finite_float(value, label)

    floor = torch.finfo(compute_dtype).tiny
    floor_tensor = A.new_tensor(floor)
    spectral_denominator = torch.maximum(matrix_spectral_norm, floor_tensor)
    frobenius_denominator = torch.maximum(matrix_frobenius_norm, floor_tensor)
    spectral_relative_residual = residual_spectral_norm / spectral_denominator
    frobenius_relative_residual = residual_frobenius_norm / frobenius_denominator
    return {
        "matrix_dimension": int(A.shape[0]),
        "compute_dtype": _dtype_name(compute_dtype),
        "compute_device": str(A.device),
        "normalization_floor": floor,
        "factor_contract": "exactly_lower_triangular_with_strictly_positive_diagonal",
        "residual_kind": "cholesky_factorization_residual",
        "residual_definition": "A_minus_L_matmul_L_transpose",
        "matrix_spectral_norm": _finite_float(
            matrix_spectral_norm,
            "matrix_spectral_norm",
        ),
        "matrix_frobenius_norm": _finite_float(
            matrix_frobenius_norm,
            "matrix_frobenius_norm",
        ),
        "residual_spectral_norm": _finite_float(
            residual_spectral_norm,
            "residual_spectral_norm",
        ),
        "residual_frobenius_norm": _finite_float(
            residual_frobenius_norm,
            "residual_frobenius_norm",
        ),
        "spectral_relative_factorization_residual_denominator": _finite_float(
            spectral_denominator,
            "spectral_relative_factorization_residual_denominator",
        ),
        "frobenius_relative_factorization_residual_denominator": _finite_float(
            frobenius_denominator,
            "frobenius_relative_factorization_residual_denominator",
        ),
        "spectral_relative_factorization_residual": _finite_float(
            spectral_relative_residual,
            "spectral_relative_factorization_residual",
        ),
        "frobenius_relative_factorization_residual": _finite_float(
            frobenius_relative_residual,
            "frobenius_relative_factorization_residual",
        ),
    }


def cholesky_frobenius_backward_error_metrics(
    A: torch.Tensor,
    L: torch.Tensor,
    *,
    compute_dtype: torch.dtype,
) -> dict[str, Any]:
    """Report exact Frobenius relative residual for a large Cholesky factor.

    The factor contract is identical to :func:`cholesky_backward_error_metrics`.
    Only the cubic-time spectral-norm calculations are omitted; the exact
    ``A - L @ L.T`` residual and both Frobenius norms are still recomputed in
    the declared source dtype.
    """

    A = _require_real_tensor(A, "A", ndim=2)
    L = _require_real_tensor(L, "L", ndim=2)
    _require_same_tensor_contract(A, L, "A", "L")
    _require_explicit_compute_dtype(
        compute_dtype,
        "compute_dtype",
        ((A, "A"), (L, "L")),
    )
    if A.shape[0] != A.shape[1]:
        raise CalibrationMetricInputError("A and L must be square")
    if int(torch.count_nonzero(torch.triu(L, diagonal=1)).item()) != 0:
        raise CalibrationMetricInputError("L must be exactly lower triangular")
    if bool((torch.diagonal(L) <= 0.0).any().item()):
        raise CalibrationMetricInputError("L diagonal must be strictly positive")

    residual = A - L @ L.T
    if not bool(torch.isfinite(residual).all().item()):
        raise CalibrationMetricInputError(
            "Cholesky residual recomputation produced a nonfinite value"
        )
    matrix_norm = torch.linalg.matrix_norm(A, ord="fro")
    residual_norm = torch.linalg.matrix_norm(residual, ord="fro")
    floor = torch.finfo(compute_dtype).tiny
    denominator = torch.maximum(matrix_norm, A.new_tensor(floor))
    relative_residual = residual_norm / denominator
    return {
        "matrix_dimension": int(A.shape[0]),
        "compute_dtype": _dtype_name(compute_dtype),
        "compute_device": str(A.device),
        "normalization_floor": floor,
        "factor_contract": "exactly_lower_triangular_with_strictly_positive_diagonal",
        "residual_kind": "cholesky_factorization_residual",
        "residual_definition": "A_minus_L_matmul_L_transpose",
        "reported_matrix_norm": "frobenius",
        "spectral_metrics_computed": False,
        "matrix_frobenius_norm": _finite_float(matrix_norm, "matrix_frobenius_norm"),
        "residual_frobenius_norm": _finite_float(
            residual_norm,
            "residual_frobenius_norm",
        ),
        "frobenius_relative_factorization_residual_denominator": _finite_float(
            denominator,
            "frobenius_relative_factorization_residual_denominator",
        ),
        "frobenius_relative_factorization_residual": _finite_float(
            relative_residual,
            "frobenius_relative_factorization_residual",
        ),
    }


def moment_error_metrics(
    reference_mean: torch.Tensor,
    candidate_mean: torch.Tensor,
    reference_variance: torch.Tensor,
    candidate_variance: torch.Tensor,
    *,
    outputscale: float | torch.Tensor,
    sigma_f: float | torch.Tensor,
) -> dict[str, Any]:
    """Return per-target absolute and fixed-scale moment errors.

    Inputs are the already represented arm outputs, exactly promoted or moved to
    the canonical CPU float64 comparison representation.  Means are in canonical
    train-standardized scalar units.  ``sigma_f`` is
    the value-noise *variance*, so it has the same units as the two raw latent
    variance tensors.  Both reference and candidate raw variances must remain
    finite and strictly positive; this function never clips them.
    """

    reference_mean = _require_real_tensor(reference_mean, "reference_mean", ndim=1)
    candidate_mean = _require_real_tensor(candidate_mean, "candidate_mean", ndim=1)
    reference_variance = _require_real_tensor(
        reference_variance,
        "reference_variance",
        ndim=1,
    )
    candidate_variance = _require_real_tensor(
        candidate_variance,
        "candidate_variance",
        ndim=1,
    )
    tensors = (
        (reference_mean, candidate_mean, "reference_mean", "candidate_mean"),
        (reference_mean, reference_variance, "reference_mean", "reference_variance"),
        (reference_mean, candidate_variance, "reference_mean", "candidate_variance"),
    )
    for first, second, first_label, second_label in tensors:
        _require_same_tensor_contract(first, second, first_label, second_label)
    _require_canonical_comparison_tensor(reference_mean, "moment tensors")
    if bool((reference_variance <= 0.0).any().item()):
        raise CalibrationMetricInputError("reference_variance must be strictly positive")
    if bool((candidate_variance <= 0.0).any().item()):
        raise CalibrationMetricInputError("candidate_variance must be strictly positive")

    outputscale_tensor = _positive_scalar(outputscale, "outputscale", like=reference_mean)
    sigma_f_tensor = _positive_scalar(sigma_f, "sigma_f", like=reference_mean)
    absolute_mean = torch.abs(candidate_mean - reference_mean)
    absolute_variance = torch.abs(candidate_variance - reference_variance)
    metrics = {
        "absolute_mean": absolute_mean,
        "mean_over_max_one_abs_reference": absolute_mean
        / torch.maximum(torch.ones_like(reference_mean), torch.abs(reference_mean)),
        "mean_over_sqrt_outputscale": absolute_mean / torch.sqrt(outputscale_tensor),
        "absolute_variance": absolute_variance,
        "variance_over_max_sigma_f_abs_reference_variance": absolute_variance
        / torch.maximum(
            sigma_f_tensor.expand_as(reference_variance), torch.abs(reference_variance)
        ),
        "variance_over_outputscale": absolute_variance / outputscale_tensor,
    }
    per_target = {name: _finite_float_list(values, name) for name, values in metrics.items()}
    maxima = {name: _finite_float(values.max(), f"max_{name}") for name, values in metrics.items()}
    return {
        "n_targets": int(reference_mean.numel()),
        "comparison_dtype": "float64",
        "comparison_device": "cpu",
        "outputscale": _finite_float(outputscale_tensor, "outputscale"),
        "sigma_f": _finite_float(sigma_f_tensor, "sigma_f"),
        "per_target": per_target,
        "max": maxima,
    }


def projector_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    reference_rank: int,
    candidate_rank: int,
) -> dict[str, Any]:
    """Compare equal-size projector matrices without comparing basis vectors.

    Inputs use the canonical CPU float64 comparison representation, and the
    independently supplied N0 ranks must match.  For positive rank, the
    normalized Frobenius difference is
    ``||reference-candidate||_F / sqrt(2*rank)``.  That statistic is undefined
    at rank zero, so the JSON result contains ``None`` while retaining the raw
    Frobenius and spectral differences.
    """

    reference = _require_real_tensor(reference, "reference", ndim=2)
    candidate = _require_real_tensor(candidate, "candidate", ndim=2)
    _require_same_tensor_contract(reference, candidate, "reference", "candidate")
    _require_canonical_comparison_tensor(reference, "projector tensors")
    if reference.shape[0] != reference.shape[1]:
        raise CalibrationMetricInputError("projectors must be square")
    ranks = (reference_rank, candidate_rank)
    if any(isinstance(rank, bool) or not isinstance(rank, Integral) for rank in ranks):
        raise CalibrationMetricInputError("projector ranks must be integers")
    reference_rank, candidate_rank = (int(rank) for rank in ranks)
    if not all(0 <= rank <= reference.shape[0] for rank in ranks):
        raise CalibrationMetricInputError(
            "projector ranks must lie between zero and projector size"
        )
    if reference_rank != candidate_rank:
        raise CalibrationMetricInputError(
            "reference and candidate projector ranks must match before comparison"
        )
    rank = reference_rank

    difference = candidate - reference
    reference_symmetry = reference - reference.T
    candidate_symmetry = candidate - candidate.T
    reference_idempotence = reference @ reference - reference
    candidate_idempotence = candidate @ candidate - candidate
    derived = (
        difference,
        reference_symmetry,
        candidate_symmetry,
        reference_idempotence,
        candidate_idempotence,
    )
    if not all(bool(torch.isfinite(value).all().item()) for value in derived):
        raise CalibrationMetricInputError("projector metric arithmetic produced a nonfinite value")

    frobenius = torch.linalg.matrix_norm(difference, ord="fro")
    normalized_frobenius = None
    if rank > 0:
        normalized_frobenius = _finite_float(
            frobenius / math.sqrt(2.0 * rank),
            "difference_frobenius_normalized",
        )
    return {
        "size": int(reference.shape[0]),
        "comparison_dtype": "float64",
        "comparison_device": "cpu",
        "rank": rank,
        "reference_rank": reference_rank,
        "candidate_rank": candidate_rank,
        "maxabs": _finite_float(torch.max(torch.abs(difference)), "maxabs"),
        "difference_spectral_norm": _finite_float(
            torch.linalg.matrix_norm(difference, ord=2),
            "difference_spectral_norm",
        ),
        "difference_frobenius_norm": _finite_float(
            frobenius,
            "difference_frobenius_norm",
        ),
        "difference_frobenius_normalized": normalized_frobenius,
        "reference_symmetry_spectral_error": _finite_float(
            torch.linalg.matrix_norm(reference_symmetry, ord=2),
            "reference_symmetry_spectral_error",
        ),
        "candidate_symmetry_spectral_error": _finite_float(
            torch.linalg.matrix_norm(candidate_symmetry, ord=2),
            "candidate_symmetry_spectral_error",
        ),
        "reference_idempotence_spectral_error": _finite_float(
            torch.linalg.matrix_norm(reference_idempotence, ord=2),
            "reference_idempotence_spectral_error",
        ),
        "candidate_idempotence_spectral_error": _finite_float(
            torch.linalg.matrix_norm(candidate_idempotence, ord=2),
            "candidate_idempotence_spectral_error",
        ),
        "reference_trace": _finite_float(torch.trace(reference), "reference_trace"),
        "candidate_trace": _finite_float(torch.trace(candidate), "candidate_trace"),
        "reference_trace_error_from_rank": _finite_float(
            torch.abs(torch.trace(reference) - rank),
            "reference_trace_error_from_rank",
        ),
        "candidate_trace_error_from_rank": _finite_float(
            torch.abs(torch.trace(candidate) - rank),
            "candidate_trace_error_from_rank",
        ),
    }


def _finite_ratio_or_none(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def _finite_log2_ratio_or_none(numerator: float, denominator: float) -> float | None:
    if numerator <= 0.0 or denominator <= 0.0:
        return None
    value = math.log2(numerator) - math.log2(denominator)
    return value if math.isfinite(value) else None


def rank_boundary_metrics(
    singular_values: torch.Tensor,
    *,
    cutoff: float | torch.Tensor,
    expected_rank: int,
) -> dict[str, Any]:
    """Evaluate a strict numerical-rank rule at a physical rank boundary.

    Singular values must be nonnegative and sorted in descending order.  A
    value exactly equal to ``cutoff`` is discarded.  Missing retained/dropped
    sides are represented by JSON ``null``.  Exact-zero boundary values also
    produce ``null`` for ratios whose mathematical value is infinite, rather
    than emitting non-JSON ``Infinity``.
    """

    singular_values = _require_real_tensor(singular_values, "singular_values", ndim=1)
    if bool((singular_values < 0.0).any().item()):
        raise CalibrationMetricInputError("singular_values must be nonnegative")
    if singular_values.numel() > 1 and bool(
        (singular_values[:-1] < singular_values[1:]).any().item()
    ):
        raise CalibrationMetricInputError("singular_values must be sorted descending")
    if isinstance(expected_rank, bool) or not isinstance(expected_rank, Integral):
        raise CalibrationMetricInputError("expected_rank must be an integer")
    expected_rank = int(expected_rank)
    if not 0 <= expected_rank <= singular_values.numel():
        raise CalibrationMetricInputError(
            "expected_rank must lie between zero and the number of singular values"
        )
    cutoff_tensor = _nonnegative_scalar(cutoff, "cutoff", like=singular_values)
    cutoff_value = _finite_float(cutoff_tensor, "cutoff")
    values = [float(value) for value in singular_values.detach().cpu().tolist()]
    zero_spectrum = values[0] == 0.0
    if cutoff_value == 0.0 and not zero_spectrum:
        raise CalibrationMetricInputError(
            "cutoff may be zero only when the complete singular spectrum is zero"
        )
    selected_rank = int((singular_values > cutoff_tensor).sum().item())

    squared_values = singular_values.to(dtype=torch.float64).square()
    total_energy_tensor = squared_values.sum()
    selected_discarded_energy_tensor = squared_values[selected_rank:].sum()
    expected_discarded_energy_tensor = squared_values[expected_rank:].sum()
    if not all(
        bool(torch.isfinite(value).item())
        for value in (
            total_energy_tensor,
            selected_discarded_energy_tensor,
            expected_discarded_energy_tensor,
        )
    ):
        raise CalibrationMetricInputError(
            "singular-value energy arithmetic produced a nonfinite value"
        )
    total_energy = float(total_energy_tensor)
    selected_discarded_energy = float(selected_discarded_energy_tensor)
    expected_discarded_energy = float(expected_discarded_energy_tensor)
    selected_discarded_fraction = (
        selected_discarded_energy / total_energy if total_energy > 0.0 else None
    )
    expected_discarded_fraction = (
        expected_discarded_energy / total_energy if total_energy > 0.0 else None
    )

    keep_value = values[expected_rank - 1] if expected_rank > 0 else None
    drop_value = values[expected_rank] if expected_rank < len(values) else None
    keep_ratio = None if keep_value is None else _finite_ratio_or_none(keep_value, cutoff_value)
    drop_ratio = None if drop_value is None else _finite_ratio_or_none(cutoff_value, drop_value)
    keep_guard = (
        None if keep_value is None else _finite_log2_ratio_or_none(keep_value, cutoff_value)
    )
    drop_guard = (
        None if drop_value is None else _finite_log2_ratio_or_none(cutoff_value, drop_value)
    )
    boundary_ratio = (
        None
        if keep_value is None or drop_value is None
        else _finite_ratio_or_none(keep_value, drop_value)
    )
    finite_guards = [guard for guard in (keep_guard, drop_guard) if guard is not None]
    minimum_guard = min(finite_guards) if finite_guards else None
    return {
        "n_singular_values": len(values),
        "cutoff": cutoff_value,
        "zero_spectrum": zero_spectrum,
        "expected_rank": expected_rank,
        "strict_selected_rank": selected_rank,
        "rank_matches_expected": selected_rank == expected_rank,
        "keep_singular_value": keep_value,
        "drop_singular_value": drop_value,
        "keep_over_cutoff_ratio": keep_ratio,
        "cutoff_over_drop_ratio": drop_ratio,
        "log2_keep_guard_bits": keep_guard,
        "log2_drop_guard_bits": drop_guard,
        "minimum_log2_guard_bits": minimum_guard,
        "boundary_keep_over_drop_ratio": boundary_ratio,
        "drop_is_exact_zero": drop_value == 0.0 if drop_value is not None else None,
        "total_singular_value_energy": total_energy,
        "selected_discarded_singular_value_energy": selected_discarded_energy,
        "selected_discarded_energy_fraction": selected_discarded_fraction,
        "expected_discarded_singular_value_energy": expected_discarded_energy,
        "expected_discarded_energy_fraction": expected_discarded_fraction,
    }


def nbody_physical_constraint_residuals(
    standardized_differences: torch.Tensor,
    masses: torch.Tensor,
    x_span: torch.Tensor,
) -> dict[str, Any]:
    """Audit the six affine N-body constraints in standardized coordinates.

    ``standardized_differences`` has shape ``(D, m)`` and uses the state layout
    ``[q_1,...,q_n,p_1,...,p_n]`` with three Cartesian coordinates per
    particle.  If ``x_raw = x_min + x_span*x_standardized``, the three
    mass-centred position rows have coefficients ``mass_i*x_span_q[i, axis]``;
    the three total-momentum rows have coefficients ``x_span_p[i, axis]``.
    Inputs must already be exact source-value promotions on CPU float64.  Per-row
    residuals refer to these six physical constraint rows.  The reported
    first-order roundoff quantity is an estimate, not a directed upper bound.
    """

    standardized_differences = _require_real_tensor(
        standardized_differences,
        "standardized_differences",
        ndim=2,
    )
    masses = _require_real_tensor(masses, "masses", ndim=1)
    x_span = _require_real_tensor(x_span, "x_span", ndim=1)
    if standardized_differences.dtype != torch.float64:
        raise CalibrationMetricInputError(
            "canonical physical-constraint metrics require exact promotion to float64"
        )
    if standardized_differences.device.type != "cpu":
        raise CalibrationMetricInputError(
            "canonical physical-constraint metrics require CPU linear algebra"
        )
    for tensor, label in ((masses, "masses"), (x_span, "x_span")):
        if tensor.dtype != standardized_differences.dtype:
            raise CalibrationMetricInputError(
                f"{label} must have dtype {standardized_differences.dtype}"
            )
        if tensor.device != standardized_differences.device:
            raise CalibrationMetricInputError(
                f"{label} must be on device {standardized_differences.device}"
            )
    if bool((masses <= 0.0).any().item()):
        raise CalibrationMetricInputError("masses must be strictly positive")
    if bool((x_span <= 0.0).any().item()):
        raise CalibrationMetricInputError("x_span must be strictly positive")
    particles = int(masses.numel())
    expected_dimension = 6 * particles
    if x_span.shape != (expected_dimension,):
        raise CalibrationMetricInputError("x_span must have shape (6 * n_particles,)")
    if standardized_differences.shape[0] != expected_dimension:
        raise CalibrationMetricInputError(
            "standardized_differences first dimension must equal 6 * n_particles"
        )

    spans = x_span.reshape(2, particles, 3)
    constraint_matrix = standardized_differences.new_zeros((6, expected_dimension))
    position_columns = torch.arange(3 * particles, device=x_span.device).reshape(particles, 3)
    momentum_columns = position_columns + 3 * particles
    axes = torch.arange(3, device=x_span.device)
    constraint_matrix[axes[:, None], position_columns.T] = (masses[:, None] * spans[0]).T
    constraint_matrix[(axes + 3)[:, None], momentum_columns.T] = spans[1].T

    residual = constraint_matrix @ standardized_differences
    if not bool(torch.isfinite(residual).all().item()):
        raise CalibrationMetricInputError(
            "constraint residual arithmetic produced a nonfinite value"
        )
    difference_spectral_norm = torch.linalg.matrix_norm(standardized_differences, ord=2)
    constraint_spectral_norm = torch.linalg.matrix_norm(constraint_matrix, ord=2)
    residual_spectral_norm = torch.linalg.matrix_norm(residual, ord=2)
    row_norms = torch.linalg.vector_norm(constraint_matrix, dim=1)
    row_residual_norms = torch.linalg.vector_norm(residual, dim=1)
    tiny = torch.finfo(standardized_differences.dtype).tiny
    floor = standardized_differences.new_tensor(tiny)
    global_scale = constraint_spectral_norm * difference_spectral_norm
    global_denominator = torch.maximum(global_scale, floor)
    row_scales = row_norms * difference_spectral_norm
    row_denominators = torch.maximum(row_scales, floor.expand_as(row_scales))
    per_row_normalized = row_residual_norms / row_denominators
    global_normalized = residual_spectral_norm / global_denominator

    dimension_epsilon = expected_dimension * torch.finfo(standardized_differences.dtype).eps
    if dimension_epsilon >= 1.0:
        raise CalibrationMetricInputError("matrix-product roundoff factor is undefined")
    matrix_product_gamma = dimension_epsilon / (1.0 - dimension_epsilon)
    absolute_product = torch.abs(constraint_matrix) @ torch.abs(standardized_differences)
    roundoff_estimate_matrix = matrix_product_gamma * absolute_product
    if not bool(torch.isfinite(roundoff_estimate_matrix).all().item()):
        raise CalibrationMetricInputError("roundoff-estimate arithmetic produced a nonfinite value")
    roundoff_estimate_spectral = torch.linalg.matrix_norm(roundoff_estimate_matrix, ord=2)
    row_roundoff_estimates = torch.linalg.vector_norm(roundoff_estimate_matrix, dim=1)
    roundoff_denominator = torch.maximum(roundoff_estimate_spectral, floor)
    row_roundoff_denominators = torch.maximum(
        row_roundoff_estimates,
        floor.expand_as(row_roundoff_estimates),
    )
    residual_over_roundoff_estimate = residual_spectral_norm / roundoff_denominator
    per_row_over_roundoff_estimate = row_residual_norms / row_roundoff_denominators
    normalized_excess_over_roundoff_estimate = (
        torch.clamp_min(residual_spectral_norm - roundoff_estimate_spectral, 0.0)
        / global_denominator
    )
    per_row_normalized_excess = (
        torch.clamp_min(
            row_residual_norms - row_roundoff_estimates,
            0.0,
        )
        / row_denominators
    )
    global_residual_frobenius = torch.linalg.matrix_norm(residual, ord="fro")
    return {
        "n_particles": particles,
        "dimension": expected_dimension,
        "n_differences": int(standardized_differences.shape[1]),
        "compute_dtype": "float64",
        "device_type": "cpu",
        "constraint_labels": list(_CONSTRAINT_LABELS),
        "constraint_matrix": [
            [float(value) for value in row] for row in constraint_matrix.detach().cpu().tolist()
        ],
        "difference_spectral_norm": _finite_float(
            difference_spectral_norm,
            "difference_spectral_norm",
        ),
        "constraint_spectral_norm": _finite_float(
            constraint_spectral_norm,
            "constraint_spectral_norm",
        ),
        "normalization_scale_spectral": _finite_float(
            global_scale,
            "normalization_scale_spectral",
        ),
        "normalization_floor": float(tiny),
        "matrix_product_gamma": matrix_product_gamma,
        "per_row_residual_norm": _finite_float_list(
            row_residual_norms,
            "per_row_residual_norm",
        ),
        "per_row_normalized_residual": _finite_float_list(
            per_row_normalized,
            "per_row_normalized_residual",
        ),
        "per_row_roundoff_estimate": _finite_float_list(
            row_roundoff_estimates,
            "per_row_roundoff_estimate",
        ),
        "per_row_residual_over_roundoff_estimate": _finite_float_list(
            per_row_over_roundoff_estimate,
            "per_row_residual_over_roundoff_estimate",
        ),
        "per_row_normalized_excess_over_roundoff_estimate": _finite_float_list(
            per_row_normalized_excess,
            "per_row_normalized_excess_over_roundoff_estimate",
        ),
        "global_residual_frobenius": _finite_float(
            global_residual_frobenius,
            "global_residual_frobenius",
        ),
        "global_residual_spectral": _finite_float(
            residual_spectral_norm,
            "global_residual_spectral",
        ),
        "global_normalized_residual": _finite_float(
            global_normalized,
            "global_normalized_residual",
        ),
        "roundoff_estimate_spectral": _finite_float(
            roundoff_estimate_spectral,
            "roundoff_estimate_spectral",
        ),
        "residual_over_roundoff_estimate": _finite_float(
            residual_over_roundoff_estimate,
            "residual_over_roundoff_estimate",
        ),
        "normalized_excess_over_roundoff_estimate": _finite_float(
            normalized_excess_over_roundoff_estimate,
            "normalized_excess_over_roundoff_estimate",
        ),
    }


def _record_guard(record: Mapping[str, object], position: int) -> tuple[int, float]:
    source_index = record.get("target_source_index")
    if isinstance(source_index, bool) or not isinstance(source_index, Integral):
        raise CalibrationMetricInputError(
            f"geometry record {position} target_source_index must be an integer"
        )
    if int(source_index) < 0:
        raise CalibrationMetricInputError(
            f"geometry record {position} target_source_index must be nonnegative"
        )
    direct_guard = record.get("minimum_log2_guard_bits")
    if direct_guard is None:
        side_guards = (
            record.get("log2_keep_guard_bits"),
            record.get("log2_drop_guard_bits"),
        )
        guards: list[float] = []
        for value in side_guards:
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise CalibrationMetricInputError(
                    f"geometry record {position} side guard must be a real scalar or null"
                )
            numeric = float(value)
            if not math.isfinite(numeric):
                raise CalibrationMetricInputError(
                    f"geometry record {position} side guard must be finite"
                )
            guards.append(numeric)
        if not guards:
            raise CalibrationMetricInputError(
                f"geometry record {position} has no finite rank-boundary guard"
            )
        direct_guard = min(guards)
    if isinstance(direct_guard, bool) or not isinstance(direct_guard, Real):
        raise CalibrationMetricInputError(
            f"geometry record {position} minimum guard must be a real scalar"
        )
    guard = float(direct_guard)
    if not math.isfinite(guard):
        raise CalibrationMetricInputError(
            f"geometry record {position} minimum guard must be finite"
        )
    provided_side_guards = []
    for value in (
        record.get("log2_keep_guard_bits"),
        record.get("log2_drop_guard_bits"),
    ):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise CalibrationMetricInputError(
                    f"geometry record {position} side guard must be a real scalar or null"
                )
            numeric = float(value)
            if not math.isfinite(numeric):
                raise CalibrationMetricInputError(
                    f"geometry record {position} side guard must be finite"
                )
            provided_side_guards.append(numeric)
    if provided_side_guards and guard != min(provided_side_guards):
        raise CalibrationMetricInputError(
            f"geometry record {position} minimum guard is inconsistent with its side guards"
        )
    return int(source_index), guard


def select_geometry_strata(
    records: Sequence[Mapping[str, object]],
    *,
    count: int,
) -> dict[str, Any]:
    """Select deterministic worst/median/best geometry without using labels.

    Records are ordered by their minimum retained/dropped log2 guard, with
    ``target_source_index`` as the only tie-break.  Count two selects
    worst/best; count three also selects the upper median of the complete
    sorted record set.
    """

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise CalibrationMetricInputError("records must be a sequence of mappings")
    if isinstance(count, bool) or not isinstance(count, Integral) or int(count) not in {2, 3}:
        raise CalibrationMetricInputError("count must be either 2 or 3")
    count = int(count)
    if len(records) < count:
        raise CalibrationMetricInputError("records must contain at least count entries")

    ranked: list[tuple[float, int]] = []
    seen_indices: set[int] = set()
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise CalibrationMetricInputError(f"geometry record {position} must be a mapping")
        source_index, guard = _record_guard(record, position)
        if source_index in seen_indices:
            raise CalibrationMetricInputError("target_source_index values must be unique")
        seen_indices.add(source_index)
        ranked.append((guard, source_index))
    ranked.sort(key=lambda item: (item[0], item[1]))

    if count == 2:
        positions = (0, len(ranked) - 1)
        names = ("worst", "best")
    else:
        positions = (0, len(ranked) // 2, len(ranked) - 1)
        names = ("worst", "median", "best")
    selected = [
        {
            "stratum": name,
            "target_source_index": ranked[position][1],
            "minimum_log2_guard_bits": ranked[position][0],
        }
        for name, position in zip(names, positions, strict=True)
    ]
    return {
        "available_count": len(ranked),
        "selected_count": count,
        "selection_rule": (
            "ascending minimum log2 guard; target_source_index tie-break; upper median"
        ),
        "selected": selected,
    }
