"""Registered four-arm full-q precision diagnostic for F02b.

The released TERA predictor represents one local derivative conditional by an
``(m * m) x (m * m)`` Schur system.  This module is deliberately diagnostic:
it is allowed only for the registered ``m=50`` geometry strata and only on
CPU.  It calls the pinned released one-target path to capture the *actual*
function and q-coordinate Cholesky inputs, factors, and selected jitters.  The
intermediate tensors are then rebuilt with the pinned vendor primitives and
must match those captured matrices exactly before any precision comparison is
reported.

Four registered arms separate assembly and solve precision:

* native fp32 assembly and fp32 solve;
* the complete fp32 represented system promoted to fp64 before its solve;
* native fp64 assembly from exact-promoted source-fp32 inputs and fp64 solve;
* the complete native-fp64 represented system cast to fp32 before its solve.

All cross-arm comparisons and support/complement decompositions use CPU
float64.  This diagnostic is not the N1 correctness reference; support64 and
the later arbitrary-precision fixtures retain that role.
"""

from __future__ import annotations

import hashlib
import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from experiments.f02_internal_models import build_released_tera_predictor
from experiments.f02b_calibration_metrics import (
    cholesky_backward_error_metrics,
    cholesky_frobenius_backward_error_metrics,
    dense_solve_frobenius_error_metrics,
)
from experiments.f02b_calibration_probe_execution import (
    SOURCE_DTYPE,
    ProbeExecutionEvidenceError,
    ProbeExecutionInputError,
    RegisteredOrbitArmInputs,
    RegisteredOrbitStrata,
    RegisteredSourceGeometry,
    _validate_registered_strata,
    _validate_source_geometry_reference,
    _validate_target_position,
)

FULL_Q_M = 50
FULL_Q_ARM_NAMES = (
    "native_fp32_assembly_fp32_solve",
    "fp32_assembly_promoted_fp64_solve",
    "native_quantized_input_fp64_assembly_fp64_solve",
    "fp64_assembly_cast_fp32_solve",
)
_FULL_Q_CONSTRUCTION_TOKEN = object()


@dataclass(frozen=True, slots=True)
class _CapturedCholesky:
    base_matrix: torch.Tensor
    system_matrix: torch.Tensor
    factor: torch.Tensor
    jitter_used: float
    attempts: int


@dataclass(frozen=True, slots=True)
class _ReleasedCapture:
    mean: torch.Tensor
    clipped_variance: torch.Tensor
    function: _CapturedCholesky
    q_coordinate: _CapturedCholesky


@dataclass(frozen=True, slots=True)
class FullQAssembly:
    """Materialized released TERA intermediates for one native dtype."""

    dtype: torch.dtype
    H: torch.Tensor
    q: torch.Tensor
    function_covariance: torch.Tensor
    function_system: torch.Tensor
    function_factor: torch.Tensor
    target_function_cross: torch.Tensor
    value_gradient_cross: torch.Tensor
    unconditional_q_covariance: torch.Tensor
    schur_covariance: torch.Tensor
    q_system: torch.Tensor
    q_factor: torch.Tensor
    conditional_cross: torch.Tensor
    raw_q_observations: torch.Tensor
    conditional_observations: torch.Tensor
    function_weights: torch.Tensor
    value_only_variance: torch.Tensor
    base_mean: torch.Tensor
    function_jitter_used: float
    function_cholesky_attempts: int
    q_jitter_used: float
    q_cholesky_attempts: int
    released_mean: torch.Tensor
    released_clipped_variance: torch.Tensor


@dataclass(frozen=True, slots=True)
class FullQArmExecution:
    """Compact evidence for one registered assembly/solve precision arm."""

    name: str
    assembly_dtype: torch.dtype
    solve_dtype: torch.dtype
    system_provenance: str
    represented_system_sha256: str
    represented_rhs_sha256: str
    function_jitter_used: float
    q_jitter_used: float
    function_cholesky_error: dict[str, Any]
    factorization_succeeded: bool
    factorization_failure_reason: str | None
    q_cholesky_error: dict[str, Any] | None
    own_system_solve_error: dict[str, Any] | None
    canonical_fp64_system_solve_error: dict[str, Any] | None
    assembly_discrepancies_from_native_fp64: dict[str, dict[str, Any]]
    support_decomposition: dict[str, Any] | None
    mean: float | None
    raw_latent_variance: float | None
    released_native_equivalence: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class FullQTargetExecution:
    """The four registered full-q arms for one selected ``m=50`` target."""

    task_index: int
    source_arm_binding_sha256: str
    source_rank_reference_sha256: str
    source_rank_grid_sha256: str
    strata_selection_sha256: str
    target_position: int
    target_source_index: int
    neighbour_positions: torch.Tensor
    neighbour_source_indices: torch.Tensor
    m: int
    q_system_dimension: int
    support_rank: int
    support_projector_sha256: str
    arms: tuple[FullQArmExecution, ...]
    canonical_arm_name: str
    diagnostic_role: str
    _construction_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._construction_token is not _FULL_Q_CONSTRUCTION_TOKEN:
            raise ProbeExecutionInputError(
                "FullQTargetExecution must be created by the registered executor"
            )
        if tuple(arm.name for arm in self.arms) != FULL_Q_ARM_NAMES:
            raise ProbeExecutionEvidenceError(
                "full-q execution does not contain the four registered arms in order"
            )


def _snapshot_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone().contiguous()


def _finite_scalar(value: torch.Tensor | float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ProbeExecutionEvidenceError(f"{label} is nonfinite")
    return result


def _tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().contiguous().cpu()
    hasher = hashlib.sha256()
    hasher.update(str(value.dtype).encode("ascii"))
    hasher.update(repr(tuple(value.shape)).encode("ascii"))
    hasher.update(value.numpy().tobytes(order="C"))
    return hasher.hexdigest()


def _recording_cholesky(trace: list[_CapturedCholesky]) -> Callable[..., torch.Tensor]:
    def cholesky(
        matrix: torch.Tensor,
        jitter0: float = 1e-8,
        jitter_max: float = 1e-1,
    ) -> torch.Tensor:
        jitter = jitter0
        attempts = 0
        identity = torch.eye(
            matrix.shape[-1],
            dtype=matrix.dtype,
            device=matrix.device,
        )
        while True:
            attempts += 1
            represented_jitter = matrix.new_tensor(jitter)
            system = matrix + represented_jitter * identity
            try:
                factor = torch.linalg.cholesky(system)
            except Exception:
                jitter *= 10.0
                if jitter > jitter_max:
                    raise
                continue
            trace.append(
                _CapturedCholesky(
                    base_matrix=_snapshot_tensor(matrix),
                    system_matrix=_snapshot_tensor(system),
                    factor=_snapshot_tensor(factor),
                    jitter_used=float(represented_jitter),
                    attempts=attempts,
                )
            )
            return factor

    return cholesky


def _capture_released_target(
    arm: RegisteredOrbitArmInputs,
    *,
    target_position: int,
) -> _ReleasedCapture:
    """Run the pinned private one-target path and capture its two factors."""

    from gp import tera as released_tera_wrapper

    if not released_tera_wrapper._VENDOR_SRC:  # pragma: no cover - import guard
        raise ProbeExecutionEvidenceError("pinned released TERA source is unavailable")

    import gp_sim_kl.models.common as released_common
    import md22_regression.models.tera as released_tera
    from gp_sim_kl.utils import scale_inputs

    original_function = released_common.cholesky_with_jitter
    original_q = released_tera.cholesky_with_jitter
    expected_signature = ("K", "jitter0", "jitter_max")
    if tuple(inspect.signature(original_function).parameters) != expected_signature:
        raise ProbeExecutionEvidenceError(
            "pinned released function Cholesky API changed"
        )
    if tuple(inspect.signature(original_q).parameters) != expected_signature:
        raise ProbeExecutionEvidenceError("pinned released q Cholesky API changed")

    predictor = build_released_tera_predictor(
        arm.train,
        arm.parameters,
        m=FULL_Q_M,
    )
    predict_one = getattr(predictor, "_predict_one", None)
    if predict_one is None or tuple(inspect.signature(predict_one).parameters) != (
        "x_eval",
        "x_eval_scaled",
        "idx",
    ):
        raise ProbeExecutionEvidenceError("pinned released _predict_one API changed")

    function_trace: list[_CapturedCholesky] = []
    q_trace: list[_CapturedCholesky] = []
    target = arm.evaluation.X[target_position].unsqueeze(0)
    lengthscale = arm.parameters.lengthscale.to(
        dtype=arm.train.X.dtype,
        device=arm.train.X.device,
    )
    target_scaled = scale_inputs(target, lengthscale)
    neighbours = arm.fixed_neighbours.positions[target_position]
    released_common.cholesky_with_jitter = _recording_cholesky(function_trace)
    released_tera.cholesky_with_jitter = _recording_cholesky(q_trace)
    try:
        with torch.inference_mode():
            mean, variance = predict_one(
                x_eval=target,
                x_eval_scaled=target_scaled,
                idx=neighbours,
            )
    finally:
        released_common.cholesky_with_jitter = original_function
        released_tera.cholesky_with_jitter = original_q
    if len(function_trace) != 1 or len(q_trace) != 1:
        raise ProbeExecutionEvidenceError(
            "released full-q target must make exactly one function and one q Cholesky call"
        )
    if mean.shape != torch.Size([]) or variance.shape != torch.Size([]):
        raise ProbeExecutionEvidenceError("released full-q moments must be scalars")
    if not bool(torch.isfinite(mean).item()) or not bool(torch.isfinite(variance).item()):
        raise ProbeExecutionEvidenceError("released full-q moments are nonfinite")
    return _ReleasedCapture(
        mean=_snapshot_tensor(mean),
        clipped_variance=_snapshot_tensor(variance),
        function=function_trace[0],
        q_coordinate=q_trace[0],
    )


def _assemble_from_released_primitives(
    arm: RegisteredOrbitArmInputs,
    *,
    target_position: int,
    capture: _ReleasedCapture,
) -> FullQAssembly:
    """Rebuild every intermediate and authenticate it against the capture."""

    import md22_regression.models.tera as released_tera

    neighbours = arm.fixed_neighbours.positions[target_position]
    x_condition = arm.train.X[neighbours]
    values = arm.train.value[neighbours]
    gradients = arm.train.gradient[neighbours]
    target = arm.evaluation.X[target_position].unsqueeze(0)
    dtype = x_condition.dtype
    device = x_condition.device
    m = int(x_condition.shape[0])
    lengthscale = arm.parameters.lengthscale.to(dtype=dtype, device=device)
    outputscale = torch.as_tensor(
        arm.parameters.outputscale,
        dtype=dtype,
        device=device,
    )

    if lengthscale.numel() == 1:
        x_scaled = x_condition / lengthscale.reshape(1)
        target_scaled = target / lengthscale.reshape(1)
    else:
        x_scaled = x_condition / lengthscale.view(1, -1)
        target_scaled = target / lengthscale.view(1, -1)
    raw_differences = (x_condition - target).T.contiguous()
    scaled_differences = (x_scaled - target_scaled).T.contiguous()
    H = scaled_differences.T @ scaled_differences
    columns = H.T.contiguous()
    q = columns[:, None, :] - columns[None, :, :]

    radii_target = torch.diagonal(H, 0)
    target_alpha = released_tera._alpha_from_r(
        radii_target,
        arm.parameters.kernel,
        outputscale,
    )
    neighbour_differences = x_scaled[:, None, :] - x_scaled[None, :, :]
    radii_neighbours = (neighbour_differences * neighbour_differences).sum(dim=-1)
    pair_alpha = released_tera._alpha_from_r(
        radii_neighbours,
        arm.parameters.kernel,
        outputscale,
    )
    pair_beta = released_tera._beta_from_r(
        radii_neighbours,
        arm.parameters.kernel,
        outputscale,
    )

    target_q_cross = ((-target_alpha[:, None]) * columns).reshape(m * m).contiguous()
    value_gradient_cross = (
        ((-pair_alpha[:, :, None]) * q)
        .permute(0, 2, 1)
        .reshape(m * m, m)
        .contiguous()
    )
    blocks = pair_alpha[:, :, None, None] * H.view(1, 1, m, m)
    blocks = blocks + pair_beta[:, :, None, None] * (
        q[:, :, :, None] * q[:, :, None, :]
    )
    if arm.parameters.sigma_g > 0.0:
        noise_gram = released_tera._projected_gradient_noise_gram(
            raw_differences,
            lengthscale,
            arm.parameters.gradient_noise_model,
        )
        diagonal_indices = torch.arange(m, device=device)
        blocks[diagonal_indices, diagonal_indices] = (
            blocks[diagonal_indices, diagonal_indices]
            + torch.as_tensor(arm.parameters.sigma_g, dtype=dtype, device=device)
            * noise_gram
        )
    unconditional = (
        blocks.permute(0, 2, 1, 3).reshape(m * m, m * m).contiguous()
    )

    function_covariance = released_tera.function_covariance(
        x_condition,
        x_condition,
        lengthscale,
        arm.parameters.outputscale,
        arm.parameters.kernel,
    )
    function_covariance = 0.5 * (function_covariance + function_covariance.T)
    released_sigma_f = math.sqrt(arm.parameters.sigma_f)
    if released_sigma_f > 0.0:
        function_covariance = function_covariance + (released_sigma_f**2) * torch.eye(
            m,
            dtype=dtype,
            device=device,
        )
    if not torch.equal(function_covariance, capture.function.base_matrix):
        difference = torch.max(torch.abs(function_covariance - capture.function.base_matrix))
        raise ProbeExecutionEvidenceError(
            "rebuilt function covariance does not exactly match released TERA "
            f"(maxabs={float(difference)})"
        )

    target_function_cross = released_tera.function_covariance(
        x_condition,
        target,
        lengthscale,
        arm.parameters.outputscale,
        arm.parameters.kernel,
    ).squeeze(-1)
    joint_rhs = torch.cat(
        [target_function_cross[:, None], value_gradient_cross.T],
        dim=1,
    )
    joint_solution = torch.cholesky_solve(joint_rhs, capture.function.factor)
    function_weights = joint_solution[:, 0]
    function_inverse_cross = joint_solution[:, 1:]
    value_only_variance = outputscale - torch.dot(
        target_function_cross,
        function_weights,
    )
    conditional_cross = target_q_cross - value_gradient_cross @ function_weights
    schur = unconditional - value_gradient_cross @ function_inverse_cross
    schur = 0.5 * (schur + schur.T)
    if not torch.equal(schur, capture.q_coordinate.base_matrix):
        difference = torch.max(torch.abs(schur - capture.q_coordinate.base_matrix))
        raise ProbeExecutionEvidenceError(
            "rebuilt q Schur covariance does not exactly match released TERA "
            f"(maxabs={float(difference)})"
        )

    raw_q_observations = gradients @ raw_differences
    alpha_values = torch.cholesky_solve(
        values.unsqueeze(1),
        capture.function.factor,
    ).squeeze(1)
    conditional_observations = (
        raw_q_observations.reshape(-1) - value_gradient_cross @ alpha_values
    )
    base_mean = torch.dot(function_weights, values)

    return FullQAssembly(
        dtype=dtype,
        H=_snapshot_tensor(H),
        q=_snapshot_tensor(q),
        function_covariance=_snapshot_tensor(function_covariance),
        function_system=_snapshot_tensor(capture.function.system_matrix),
        function_factor=_snapshot_tensor(capture.function.factor),
        target_function_cross=_snapshot_tensor(target_function_cross),
        value_gradient_cross=_snapshot_tensor(value_gradient_cross),
        unconditional_q_covariance=_snapshot_tensor(unconditional),
        schur_covariance=_snapshot_tensor(schur),
        q_system=_snapshot_tensor(capture.q_coordinate.system_matrix),
        q_factor=_snapshot_tensor(capture.q_coordinate.factor),
        conditional_cross=_snapshot_tensor(conditional_cross),
        raw_q_observations=_snapshot_tensor(raw_q_observations.reshape(-1)),
        conditional_observations=_snapshot_tensor(conditional_observations),
        function_weights=_snapshot_tensor(function_weights),
        value_only_variance=_snapshot_tensor(value_only_variance),
        base_mean=_snapshot_tensor(base_mean),
        function_jitter_used=capture.function.jitter_used,
        function_cholesky_attempts=capture.function.attempts,
        q_jitter_used=capture.q_coordinate.jitter_used,
        q_cholesky_attempts=capture.q_coordinate.attempts,
        released_mean=_snapshot_tensor(capture.mean),
        released_clipped_variance=_snapshot_tensor(capture.clipped_variance),
    )


def _canonical64(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=torch.float64).contiguous()


def _discrepancy(
    reference64: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, Any]:
    reference64 = _canonical64(reference64)
    candidate64 = _canonical64(candidate)
    if reference64.shape != candidate64.shape:
        raise ProbeExecutionEvidenceError("assembly comparison shapes do not match")
    difference = candidate64 - reference64
    reference_norm = torch.linalg.vector_norm(reference64.reshape(-1))
    candidate_norm = torch.linalg.vector_norm(candidate64.reshape(-1))
    difference_norm = torch.linalg.vector_norm(difference.reshape(-1))
    floor = reference64.new_tensor(torch.finfo(torch.float64).tiny)
    denominator = torch.maximum(reference_norm, floor)
    return {
        "shape": list(reference64.shape),
        "comparison_dtype": "float64",
        "comparison_device": "cpu",
        "reference_norm_frobenius": _finite_scalar(reference_norm, "reference norm"),
        "candidate_norm_frobenius": _finite_scalar(candidate_norm, "candidate norm"),
        "difference_norm_frobenius": _finite_scalar(difference_norm, "difference norm"),
        "difference_relative_frobenius": _finite_scalar(
            difference_norm / denominator,
            "relative difference",
        ),
        "difference_maxabs": _finite_scalar(
            torch.max(torch.abs(difference)),
            "maximum absolute difference",
        ),
    }


def _assembly_discrepancies(
    canonical: FullQAssembly,
    candidate: FullQAssembly,
    *,
    cast_dtype: torch.dtype | None = None,
) -> dict[str, dict[str, Any]]:
    fields = {
        "H": (canonical.H, candidate.H),
        "q": (canonical.q, candidate.q),
        "function_covariance": (
            canonical.function_covariance,
            candidate.function_covariance,
        ),
        "function_system_after_selected_jitter": (
            canonical.function_system,
            candidate.function_system,
        ),
        "target_function_cross": (
            canonical.target_function_cross,
            candidate.target_function_cross,
        ),
        "value_gradient_cross_Q": (
            canonical.value_gradient_cross,
            candidate.value_gradient_cross,
        ),
        "unconditional_q_covariance_G0": (
            canonical.unconditional_q_covariance,
            candidate.unconditional_q_covariance,
        ),
        "schur_covariance_before_q_jitter": (
            canonical.schur_covariance,
            candidate.schur_covariance,
        ),
        "q_system_after_selected_jitter": (
            canonical.q_system,
            candidate.q_system,
        ),
        "conditional_cross_rhs": (
            canonical.conditional_cross,
            candidate.conditional_cross,
        ),
        "conditional_observations": (
            canonical.conditional_observations,
            candidate.conditional_observations,
        ),
    }
    if cast_dtype is not None:
        fields = {
            name: (reference, value.to(dtype=cast_dtype))
            for name, (reference, value) in fields.items()
        }
    return {
        name: _discrepancy(reference, value)
        for name, (reference, value) in fields.items()
    }


def _vector_decomposition(
    value64: torch.Tensor,
    projector64: torch.Tensor,
    *,
    m: int,
) -> dict[str, float]:
    matrix = value64.reshape(m, m)
    support = matrix @ projector64
    complement = matrix - support
    total_norm = torch.linalg.vector_norm(matrix)
    support_norm = torch.linalg.vector_norm(support)
    complement_norm = torch.linalg.vector_norm(complement)
    floor = value64.new_tensor(torch.finfo(torch.float64).tiny)
    denominator = torch.maximum(total_norm, floor)
    return {
        "total_norm_2": _finite_scalar(total_norm, "total vector norm"),
        "support_norm_2": _finite_scalar(support_norm, "support vector norm"),
        "complement_norm_2": _finite_scalar(
            complement_norm,
            "complement vector norm",
        ),
        "support_fraction": _finite_scalar(
            support_norm / denominator,
            "support vector fraction",
        ),
        "complement_fraction": _finite_scalar(
            complement_norm / denominator,
            "complement vector fraction",
        ),
    }


def _support_decomposition(
    system: torch.Tensor,
    rhs: torch.Tensor,
    raw_observations: torch.Tensor,
    observations: torch.Tensor,
    solution: torch.Tensor,
    projector: torch.Tensor,
    *,
    m: int,
) -> dict[str, Any]:
    system64 = _canonical64(system)
    rhs64 = _canonical64(rhs)
    raw_observations64 = _canonical64(raw_observations)
    observations64 = _canonical64(observations)
    solution64 = _canonical64(solution)
    projector64 = _canonical64(projector)
    if projector64.shape != (m, m):
        raise ProbeExecutionEvidenceError("q support projector has the wrong shape")
    symmetry_error = torch.max(torch.abs(projector64 - projector64.T))
    idempotence_error = torch.max(
        torch.abs(projector64 @ projector64 - projector64)
    )

    blocks = system64.reshape(m, m, m, m)
    support_left = torch.einsum("ac,icjb->iajb", projector64, blocks)
    complement_left = blocks - support_left
    support_support = torch.einsum(
        "iajc,cb->iajb",
        support_left,
        projector64,
    )
    support_complement = support_left - support_support
    complement_support = torch.einsum(
        "iajc,cb->iajb",
        complement_left,
        projector64,
    )
    complement_complement = complement_left - complement_support
    total_norm = torch.linalg.vector_norm(blocks)
    floor = system64.new_tensor(torch.finfo(torch.float64).tiny)
    denominator = torch.maximum(total_norm, floor)

    matrix_norms: dict[str, float] = {
        "total_frobenius_norm": _finite_scalar(total_norm, "total matrix norm")
    }
    for name, value in (
        ("support_support", support_support),
        ("support_complement", support_complement),
        ("complement_support", complement_support),
        ("complement_complement", complement_complement),
    ):
        norm = torch.linalg.vector_norm(value)
        matrix_norms[f"{name}_frobenius_norm"] = _finite_scalar(
            norm,
            f"{name} matrix norm",
        )
        matrix_norms[f"{name}_fraction"] = _finite_scalar(
            norm / denominator,
            f"{name} matrix fraction",
        )
    return {
        "comparison_dtype": "float64",
        "comparison_device": "cpu",
        "projector_symmetry_maxabs": _finite_scalar(
            symmetry_error,
            "projector symmetry error",
        ),
        "projector_idempotence_maxabs": _finite_scalar(
            idempotence_error,
            "projector idempotence error",
        ),
        "matrix_blocks": matrix_norms,
        "rhs": _vector_decomposition(rhs64, projector64, m=m),
        "raw_q_observations": _vector_decomposition(
            raw_observations64,
            projector64,
            m=m,
        ),
        "conditional_observations": _vector_decomposition(
            observations64,
            projector64,
            m=m,
        ),
        "solution": _vector_decomposition(solution64, projector64, m=m),
    }


def _solve_arm(
    *,
    name: str,
    assembly: FullQAssembly,
    canonical: FullQAssembly,
    system: torch.Tensor,
    rhs: torch.Tensor,
    raw_observations: torch.Tensor,
    observations: torch.Tensor,
    base_mean: torch.Tensor,
    value_only_variance: torch.Tensor,
    factor: torch.Tensor | None,
    assembly_dtype: torch.dtype,
    system_provenance: str,
    assembly_discrepancies: dict[str, dict[str, Any]],
    projector: torch.Tensor,
    released_native: FullQAssembly | None,
) -> FullQArmExecution:
    solve_dtype = system.dtype
    function_factor_error = cholesky_backward_error_metrics(
        assembly.function_system,
        assembly.function_factor,
        compute_dtype=assembly.dtype,
    )
    if factor is None:
        factor, info = torch.linalg.cholesky_ex(system)
        if int(info.max().item()) != 0:
            return FullQArmExecution(
                name=name,
                assembly_dtype=assembly_dtype,
                solve_dtype=solve_dtype,
                system_provenance=system_provenance,
                represented_system_sha256=_tensor_sha256(system),
                represented_rhs_sha256=_tensor_sha256(rhs),
                function_jitter_used=assembly.function_jitter_used,
                q_jitter_used=assembly.q_jitter_used,
                function_cholesky_error=function_factor_error,
                factorization_succeeded=False,
                factorization_failure_reason=(
                    "fixed_represented_q_system_not_positive_definite_in_solve_dtype"
                ),
                q_cholesky_error=None,
                own_system_solve_error=None,
                canonical_fp64_system_solve_error=None,
                assembly_discrepancies_from_native_fp64=assembly_discrepancies,
                support_decomposition=None,
                mean=None,
                raw_latent_variance=None,
                released_native_equivalence=None,
            )
    if factor.dtype != solve_dtype or factor.device != system.device:
        raise ProbeExecutionEvidenceError("full-q factor does not match its represented system")
    solution = torch.cholesky_solve(rhs.unsqueeze(1), factor).squeeze(1)
    mean = base_mean + torch.dot(solution, observations)
    raw_variance = value_only_variance - torch.dot(solution, rhs)
    for value, label in ((solution, "solution"), (mean, "mean"), (raw_variance, "variance")):
        if not bool(torch.isfinite(value).all().item()):
            raise ProbeExecutionEvidenceError(f"full-q {name} {label} is nonfinite")

    own_error = dense_solve_frobenius_error_metrics(
        system,
        rhs,
        solution,
        residual_compute_dtype=solve_dtype,
    )
    canonical_solution = _canonical64(solution)
    canonical_error = dense_solve_frobenius_error_metrics(
        canonical.q_system,
        canonical.conditional_cross,
        canonical_solution,
        residual_compute_dtype=torch.float64,
    )
    q_factor_error = cholesky_frobenius_backward_error_metrics(
        system,
        factor,
        compute_dtype=solve_dtype,
    )
    projector_m = int(projector.shape[0])

    native_equivalence = None
    if released_native is not None:
        released_mean = released_native.released_mean
        released_variance = released_native.released_clipped_variance
        clipped_raw = torch.clamp(raw_variance, min=torch.finfo(solve_dtype).eps)
        native_equivalence = {
            "released_mean": _finite_scalar(released_mean, "released mean"),
            "reassociated_mean": _finite_scalar(mean, "reassociated mean"),
            "mean_absolute_difference": _finite_scalar(
                torch.abs(mean - released_mean),
                "released mean difference",
            ),
            "released_clipped_variance": _finite_scalar(
                released_variance,
                "released variance",
            ),
            "recomputed_raw_variance": _finite_scalar(
                raw_variance,
                "raw variance",
            ),
            "recomputed_released_clipped_variance": _finite_scalar(
                clipped_raw,
                "recomputed clipped variance",
            ),
            "clipped_variance_absolute_difference": _finite_scalar(
                torch.abs(clipped_raw - released_variance),
                "released variance difference",
            ),
            "released_variance_floor": float(torch.finfo(solve_dtype).eps),
            "released_variance_floor_active": bool(
                released_variance == torch.finfo(solve_dtype).eps
            ),
        }

    return FullQArmExecution(
        name=name,
        assembly_dtype=assembly_dtype,
        solve_dtype=solve_dtype,
        system_provenance=system_provenance,
        represented_system_sha256=_tensor_sha256(system),
        represented_rhs_sha256=_tensor_sha256(rhs),
        function_jitter_used=assembly.function_jitter_used,
        q_jitter_used=assembly.q_jitter_used,
        function_cholesky_error=function_factor_error,
        factorization_succeeded=True,
        factorization_failure_reason=None,
        q_cholesky_error=q_factor_error,
        own_system_solve_error=own_error,
        canonical_fp64_system_solve_error=canonical_error,
        assembly_discrepancies_from_native_fp64=assembly_discrepancies,
        support_decomposition=_support_decomposition(
            system,
            rhs,
            raw_observations,
            observations,
            solution,
            projector,
            m=projector_m,
        ),
        mean=_finite_scalar(mean, f"{name} mean"),
        raw_latent_variance=_finite_scalar(raw_variance, f"{name} variance"),
        released_native_equivalence=native_equivalence,
    )


def _execute_four_precision_arms(
    assembly32: FullQAssembly,
    assembly64: FullQAssembly,
    projector64: torch.Tensor,
) -> tuple[FullQArmExecution, ...]:
    """Execute the registered precision ladder on already authenticated assemblies."""

    if assembly32.dtype != torch.float32 or assembly64.dtype != torch.float64:
        raise ProbeExecutionInputError(
            "the four-arm ladder requires native fp32 and native fp64 assemblies"
        )
    m = int(assembly32.H.shape[0])
    if assembly32.H.shape != (m, m) or assembly64.H.shape != (m, m):
        raise ProbeExecutionEvidenceError("native full-q assembly dimensions disagree")
    if assembly32.q_system.shape != (m * m, m * m) or (
        assembly64.q_system.shape != (m * m, m * m)
    ):
        raise ProbeExecutionEvidenceError("native full-q Schur dimensions disagree")
    if projector64.shape != (m, m):
        raise ProbeExecutionEvidenceError("full-q projector does not match assembly m")

    discrepancies32 = _assembly_discrepancies(assembly64, assembly32)
    discrepancies64 = _assembly_discrepancies(assembly64, assembly64)
    discrepancies64_cast32 = _assembly_discrepancies(
        assembly64,
        assembly64,
        cast_dtype=torch.float32,
    )

    promoted32_system = assembly32.q_system.to(dtype=torch.float64)
    promoted32_rhs = assembly32.conditional_cross.to(dtype=torch.float64)
    promoted32_raw_observations = assembly32.raw_q_observations.to(dtype=torch.float64)
    promoted32_observations = assembly32.conditional_observations.to(
        dtype=torch.float64
    )
    promoted32_base_mean = assembly32.base_mean.to(dtype=torch.float64)
    promoted32_value_variance = assembly32.value_only_variance.to(dtype=torch.float64)
    cast64_system = assembly64.q_system.to(dtype=torch.float32)
    cast64_rhs = assembly64.conditional_cross.to(dtype=torch.float32)
    cast64_raw_observations = assembly64.raw_q_observations.to(dtype=torch.float32)
    cast64_observations = assembly64.conditional_observations.to(dtype=torch.float32)
    cast64_base_mean = assembly64.base_mean.to(dtype=torch.float32)
    cast64_value_variance = assembly64.value_only_variance.to(dtype=torch.float32)

    return (
        _solve_arm(
            name=FULL_Q_ARM_NAMES[0],
            assembly=assembly32,
            canonical=assembly64,
            system=assembly32.q_system,
            rhs=assembly32.conditional_cross,
            raw_observations=assembly32.raw_q_observations,
            observations=assembly32.conditional_observations,
            base_mean=assembly32.base_mean,
            value_only_variance=assembly32.value_only_variance,
            factor=assembly32.q_factor,
            assembly_dtype=torch.float32,
            system_provenance="captured_released_native_fp32_jittered_q_system",
            assembly_discrepancies=discrepancies32,
            projector=projector64,
            released_native=assembly32,
        ),
        _solve_arm(
            name=FULL_Q_ARM_NAMES[1],
            assembly=assembly32,
            canonical=assembly64,
            system=promoted32_system,
            rhs=promoted32_rhs,
            raw_observations=promoted32_raw_observations,
            observations=promoted32_observations,
            base_mean=promoted32_base_mean,
            value_only_variance=promoted32_value_variance,
            factor=None,
            assembly_dtype=torch.float32,
            system_provenance=(
                "complete_released_fp32_jittered_q_system_exactly_promoted_to_fp64"
            ),
            assembly_discrepancies=discrepancies32,
            projector=projector64,
            released_native=None,
        ),
        _solve_arm(
            name=FULL_Q_ARM_NAMES[2],
            assembly=assembly64,
            canonical=assembly64,
            system=assembly64.q_system,
            rhs=assembly64.conditional_cross,
            raw_observations=assembly64.raw_q_observations,
            observations=assembly64.conditional_observations,
            base_mean=assembly64.base_mean,
            value_only_variance=assembly64.value_only_variance,
            factor=assembly64.q_factor,
            assembly_dtype=torch.float64,
            system_provenance=(
                "captured_released_native_fp64_from_exact_promoted_fp32_inputs"
            ),
            assembly_discrepancies=discrepancies64,
            projector=projector64,
            released_native=assembly64,
        ),
        _solve_arm(
            name=FULL_Q_ARM_NAMES[3],
            assembly=assembly64,
            canonical=assembly64,
            system=cast64_system,
            rhs=cast64_rhs,
            raw_observations=cast64_raw_observations,
            observations=cast64_observations,
            base_mean=cast64_base_mean,
            value_only_variance=cast64_value_variance,
            factor=None,
            assembly_dtype=torch.float64,
            system_provenance=(
                "complete_released_native_fp64_jittered_q_system_cast_to_fp32"
            ),
            assembly_discrepancies=discrepancies64_cast32,
            projector=projector64,
            released_native=None,
        ),
    )


def _validate_full_q_entry(
    source_arm: RegisteredOrbitArmInputs,
    promoted_arm: RegisteredOrbitArmInputs,
    source_geometry: RegisteredSourceGeometry,
    strata: RegisteredOrbitStrata,
) -> tuple[int, float, int, str]:
    if type(source_arm) is not RegisteredOrbitArmInputs or type(
        promoted_arm
    ) is not RegisteredOrbitArmInputs:
        raise ProbeExecutionInputError("full-q requires registered fp32 and fp64 arms")
    source_arm.assert_unchanged()
    promoted_arm.assert_unchanged()
    if source_arm.train.X.dtype != SOURCE_DTYPE or source_arm.train.X.device.type != "cpu":
        raise ProbeExecutionInputError("full-q source arm must be CPU float32")
    if promoted_arm.train.X.dtype != torch.float64 or promoted_arm.train.X.device.type != "cpu":
        raise ProbeExecutionInputError("full-q promoted arm must be CPU float64")
    if (
        source_arm.source_arm_binding_sha256
        != promoted_arm.source_arm_binding_sha256
        or source_arm.work_plan != promoted_arm.work_plan
        or promoted_arm.binding_kind != "exact_promotion_of_bound_source_fp32_arm"
    ):
        raise ProbeExecutionInputError(
            "full-q fp64 arm must be the exact promotion of the bound fp32 arm"
        )
    plan = source_arm.work_plan
    if (
        not plan.run_full_q
        or plan.production_m != FULL_Q_M
        or plan.full_q_m != FULL_Q_M
    ):
        raise ProbeExecutionInputError("full-q is registered only at m=50")
    if type(source_geometry) is not RegisteredSourceGeometry:
        raise ProbeExecutionInputError("full-q requires registered N0 source geometry")
    position = _validate_target_position(
        source_geometry.target_position,
        source_arm.evaluation.X.shape[0],
    )
    cutoff, selected_rank, reference_sha256 = _validate_source_geometry_reference(
        source_arm,
        source_geometry,
        target_position=position,
    )
    promoted_cutoff, promoted_rank, promoted_reference = (
        _validate_source_geometry_reference(
            promoted_arm,
            source_geometry,
            target_position=position,
        )
    )
    if (promoted_cutoff, promoted_rank, promoted_reference) != (
        cutoff,
        selected_rank,
        reference_sha256,
    ):
        raise ProbeExecutionEvidenceError("fp32 and fp64 arms disagree on bound N0 evidence")
    if not _validate_registered_strata(source_arm, source_geometry, strata):
        raise ProbeExecutionInputError(
            "full-q is registered only for geometry-selected m=50 strata"
        )
    if not _validate_registered_strata(promoted_arm, source_geometry, strata):
        raise ProbeExecutionEvidenceError("promoted arm changed the registered stratum role")
    if not torch.equal(
        source_arm.fixed_neighbours.positions[position],
        promoted_arm.fixed_neighbours.positions[position],
    ) or not torch.equal(
        source_arm.fixed_neighbours.source_indices[position],
        promoted_arm.fixed_neighbours.source_indices[position],
    ):
        raise ProbeExecutionEvidenceError("full-q arms do not share exact neighbour identities")
    return position, cutoff, selected_rank, reference_sha256


def execute_registered_full_q_target(
    source_arm: RegisteredOrbitArmInputs,
    promoted_arm: RegisteredOrbitArmInputs,
    source_geometry: RegisteredSourceGeometry,
    strata: RegisteredOrbitStrata,
) -> FullQTargetExecution:
    """Execute the registered four-arm released full-q diagnostic.

    This call performs no file, catalog, label, environment, or scheduler I/O.
    Authorization and immutable artifact emission remain downstream runner
    responsibilities.
    """

    position, cutoff, selected_rank, source_reference = _validate_full_q_entry(
        source_arm,
        promoted_arm,
        source_geometry,
        strata,
    )
    del cutoff  # The full-q diagnostic uses the projector, not a rank re-selection.

    with torch.inference_mode():
        capture32 = _capture_released_target(source_arm, target_position=position)
        assembly32 = _assemble_from_released_primitives(
            source_arm,
            target_position=position,
            capture=capture32,
        )
        capture64 = _capture_released_target(promoted_arm, target_position=position)
        assembly64 = _assemble_from_released_primitives(
            promoted_arm,
            target_position=position,
            capture=capture64,
        )

        projector32 = (
            source_geometry.geometry.coordinates
            @ source_geometry.geometry.q_to_z.T
        )
        projector64 = _canonical64(projector32)
        if projector64.shape != (FULL_Q_M, FULL_Q_M):
            raise ProbeExecutionEvidenceError(
                "source N0 geometry did not produce the registered m=50 q projector"
            )

        arms = _execute_four_precision_arms(
            assembly32,
            assembly64,
            projector64,
        )

    source_arm.assert_unchanged()
    promoted_arm.assert_unchanged()
    post_position, _, post_rank, post_reference = _validate_full_q_entry(
        source_arm,
        promoted_arm,
        source_geometry,
        strata,
    )
    if (post_position, post_rank, post_reference) != (
        position,
        selected_rank,
        source_reference,
    ):
        raise ProbeExecutionEvidenceError(
            "registered full-q identities changed during execution"
        )

    neighbours = source_arm.fixed_neighbours.positions[position]
    neighbour_sources = source_arm.fixed_neighbours.source_indices[position]
    return FullQTargetExecution(
        task_index=source_arm.work_plan.task_index,
        source_arm_binding_sha256=source_arm.source_arm_binding_sha256,
        source_rank_reference_sha256=source_reference,
        source_rank_grid_sha256=strata.source_rank_grid_sha256,
        strata_selection_sha256=strata.selection_sha256,
        target_position=position,
        target_source_index=int(source_arm.evaluation.source_indices[position]),
        neighbour_positions=_snapshot_tensor(neighbours),
        neighbour_source_indices=_snapshot_tensor(neighbour_sources),
        m=FULL_Q_M,
        q_system_dimension=FULL_Q_M * FULL_Q_M,
        support_rank=selected_rank,
        support_projector_sha256=_tensor_sha256(projector64),
        arms=arms,
        canonical_arm_name=FULL_Q_ARM_NAMES[2],
        diagnostic_role=(
            "released_TERA_precision_decomposition_not_N1_correctness_reference"
        ),
        _construction_token=_FULL_Q_CONSTRUCTION_TOKEN,
    )


__all__ = [
    "FULL_Q_ARM_NAMES",
    "FULL_Q_M",
    "FullQArmExecution",
    "FullQAssembly",
    "FullQTargetExecution",
    "execute_registered_full_q_target",
]
