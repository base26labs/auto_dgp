"""Immutable canonical target artifacts for registered F02b probe execution.

Execution dataclasses are frozen, but their tensors and nested dictionaries are
not deeply immutable.  This module is the immediate downstream boundary: it
copies every retained numerical field to canonical JSON bytes and binds those
bytes by SHA-256.  It performs no filesystem, corpus, label, scheduler, or
network access.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import torch

from cluster.f02b_calibration_grid import CALIBRATION_ID, probe_task_for_index
from experiments.f02b_calibration_contract import (
    CalibrationContractError,
    canonical_json_bytes,
    parse_strict_json_bytes,
)
from experiments.f02b_calibration_full_q import (
    FULL_Q_ARM_NAMES,
    FULL_Q_M,
    FullQTargetExecution,
)
from experiments.f02b_calibration_probe_core import (
    PRIMARY_EVALUATION_ROW_COUNT,
    STRESS_TOLERANCE,
)
from experiments.f02b_calibration_probe_execution import (
    OrbitTargetExecution,
    Support64TargetExecution,
)
from experiments.f02b_calibration_stress import StressTargetExecution

PROBE_TARGET_ARTIFACT_SCHEMA_VERSION = "f02b_calibration_probe_target_artifact_v2"
PROBE_TARGET_ARTIFACT_TYPE = "f02b_registered_numerical_target"
_TOP_LEVEL_FIELDS = {
    "artifact_type",
    "calibration_id",
    "full_q",
    "orbit",
    "schema_version",
    "source_arm_binding_sha256",
    "source_rank_grid_sha256",
    "source_rank_reference_sha256",
    "strata_selection_sha256",
    "stress",
    "support64",
    "target_position",
    "target_source_index",
    "task_index",
    "task_record",
}
_FULL_Q_FIELDS = {
    "arms",
    "canonical_arm_name",
    "diagnostic_role",
    "m",
    "neighbour_positions",
    "neighbour_source_indices",
    "q_system_dimension",
    "support_projector_sha256",
    "support_rank",
}
_STRESS_FIELDS = {
    "base_solve",
    "m",
    "max_iterations",
    "native_fp64_q_projector",
    "neighbour_positions",
    "neighbour_source_indices",
    "requested_tolerance",
    "selected_rank",
    "source_q_projector",
    "source_rank_grid_sha256",
    "source_rank_reference_sha256",
    "strata_selection_sha256",
    "stress_binding_sha256",
    "tests",
}


class ProbeTargetArtifactError(ValueError):
    """Raised when target evidence cannot form one strict canonical artifact."""


@dataclass(frozen=True, slots=True)
class CanonicalProbeTargetArtifact:
    """Immutable canonical bytes and their raw SHA-256 identity."""

    payload_bytes: bytes
    payload_sha256: str

    def __post_init__(self) -> None:
        if type(self.payload_bytes) is not bytes:
            raise ProbeTargetArtifactError("artifact payload must be immutable bytes")
        observed = hashlib.sha256(self.payload_bytes).hexdigest()
        if self.payload_sha256 != observed:
            raise ProbeTargetArtifactError("artifact payload SHA-256 is mismatched")


def _validate_sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProbeTargetArtifactError(f"{label} must be a lowercase SHA-256")
    return value


def _tensor_payload(value: torch.Tensor) -> dict[str, Any]:
    if value.is_complex():
        raise ProbeTargetArtifactError("complex tensors are not admissible in probe artifacts")
    detached = value.detach().cpu().contiguous()
    if not bool(torch.isfinite(detached).all().item()):
        raise ProbeTargetArtifactError("probe artifact tensors must be finite")
    return {
        "dtype": str(detached.dtype).removeprefix("torch."),
        "shape": list(detached.shape),
        "source_device": str(value.device),
        "values": detached.tolist(),
    }


def _json_value(value: Any) -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if math.isnan(value):
            raise ProbeTargetArtifactError("NaN is not admissible in probe artifacts")
        if math.isinf(value):
            return {
                "special_float": (
                    "positive_infinity" if value > 0.0 else "negative_infinity"
                )
            }
        return value
    if isinstance(value, torch.Tensor):
        return _tensor_payload(value)
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ProbeTargetArtifactError("probe artifact mappings require string keys")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise ProbeTargetArtifactError(
        f"unsupported probe artifact value type: {type(value).__name__}"
    )


def _operator_payload(execution: OrbitTargetExecution) -> dict[str, Any]:
    operator = execution.system.operator
    if operator is None:
        return {"present": False}
    return {
        "present": True,
        "coordinates": _tensor_payload(operator.coordinates),
        "alpha": _tensor_payload(operator.alpha),
        "beta": _tensor_payload(operator.beta),
        "function_cholesky": _tensor_payload(operator.function_cholesky),
        "gradient_noise": _tensor_payload(operator.gradient_noise),
        "jitter": operator.jitter,
    }


def _orbit_payload(execution: OrbitTargetExecution) -> dict[str, Any]:
    _ = execution.production_solve
    if not execution.solves:
        raise ProbeTargetArtifactError("ORBIT target execution contains no solves")
    tolerances = [evidence.requested_tolerance for evidence in execution.solves]
    if len(set(tolerances)) != len(tolerances):
        raise ProbeTargetArtifactError("ORBIT target execution repeats a tolerance")
    solve_payloads: list[dict[str, Any]] = []
    for evidence in execution.solves:
        solve = evidence.prediction.solve
        if (
            solve.requested_tolerance != evidence.requested_tolerance
            or not isinstance(solve.operator_action, torch.Tensor)
            or not torch.equal(evidence.verified_operator_action, solve.operator_action)
            or not torch.equal(evidence.verified_residual, solve.residual)
        ):
            raise ProbeTargetArtifactError(
                "ORBIT solve evidence is inconsistent at artifact extraction"
            )
        solve_payloads.append(_json_value(evidence))
    system = execution.system
    return {
        "compute_dtype": str(execution.compute_dtype).removeprefix("torch."),
        "function_cholesky_error": _json_value(execution.function_cholesky_error),
        "include_stratum_sweep": execution.include_stratum_sweep,
        "neighbour_positions": _tensor_payload(execution.neighbour_positions),
        "neighbour_source_indices": _tensor_payload(execution.neighbour_source_indices),
        "production_tolerance": execution.production_tolerance,
        "rank_boundary": _json_value(execution.rank_boundary),
        "solves": solve_payloads,
        "system": {
            "base_mean": _tensor_payload(system.base_mean),
            "conditional_cross": _tensor_payload(system.conditional_cross),
            "conditional_observation_functional": _tensor_payload(
                system.conditional_observation_functional
            ),
            "conditional_value_variance": _tensor_payload(
                system.conditional_value_variance
            ),
            "function_cholesky": _tensor_payload(system.function_cholesky),
            "function_jitter_attempts": system.function_jitter_attempts,
            "function_jitter_requested": system.function_jitter_requested,
            "function_jitter_used": system.function_jitter_used,
            "function_system_matrix": _tensor_payload(system.function_system_matrix),
            "function_weights": _tensor_payload(system.function_weights),
            "geometry": _json_value(system.geometry),
            "operator": _operator_payload(execution),
            "operator_eigenvalue_lower_bound": system.operator_eigenvalue_lower_bound,
            "operator_lower_bound_provenance": system.operator_lower_bound_provenance,
            "operator_norm_upper_bound": system.operator_norm_upper_bound,
            "operator_norm_upper_bound_provenance": (
                system.operator_norm_upper_bound_provenance
            ),
            "orthonormal_observations": _tensor_payload(
                system.orthonormal_observations
            ),
            "value_condition": _tensor_payload(system.value_condition),
        },
    }


def _support64_payload(execution: Support64TargetExecution) -> dict[str, Any]:
    return {
        "ambient_scaled_difference_support_projector": _tensor_payload(
            execution.ambient_scaled_difference_support_projector
        ),
        "compute_dtype": str(execution.compute_dtype).removeprefix("torch."),
        "cutoff_provenance": execution.cutoff_provenance,
        "neighbour_positions": _tensor_payload(execution.neighbour_positions),
        "neighbour_source_indices": _tensor_payload(execution.neighbour_source_indices),
        "prediction": _json_value(execution.prediction),
        "q_coordinate_support_projector": _tensor_payload(
            execution.q_coordinate_support_projector
        ),
        "rank_boundary": _json_value(execution.rank_boundary),
    }


def _full_q_payload(execution: FullQTargetExecution) -> dict[str, Any]:
    return {
        "arms": [_json_value(arm) for arm in execution.arms],
        "canonical_arm_name": execution.canonical_arm_name,
        "diagnostic_role": execution.diagnostic_role,
        "m": execution.m,
        "neighbour_positions": _tensor_payload(execution.neighbour_positions),
        "neighbour_source_indices": _tensor_payload(
            execution.neighbour_source_indices
        ),
        "q_system_dimension": execution.q_system_dimension,
        "support_projector_sha256": execution.support_projector_sha256,
        "support_rank": execution.support_rank,
    }


def _stress_payload(execution: StressTargetExecution) -> dict[str, Any]:
    return {
        "base_solve": _json_value(execution.base_solve),
        "m": execution.m,
        "max_iterations": execution.max_iterations,
        "native_fp64_q_projector": _tensor_payload(
            execution.native_fp64_q_projector
        ),
        "neighbour_positions": _tensor_payload(execution.neighbour_positions),
        "neighbour_source_indices": _tensor_payload(
            execution.neighbour_source_indices
        ),
        "requested_tolerance": execution.requested_tolerance,
        "selected_rank": execution.selected_rank,
        "source_q_projector": _tensor_payload(execution.source_q_projector),
        "source_rank_grid_sha256": execution.source_rank_grid_sha256,
        "source_rank_reference_sha256": execution.source_rank_reference_sha256,
        "strata_selection_sha256": execution.strata_selection_sha256,
        "stress_binding_sha256": execution.stress_binding_sha256,
        "tests": _json_value(execution.tests),
    }


def _validate_matching_support64(
    orbit: OrbitTargetExecution,
    support64: Support64TargetExecution,
) -> None:
    scalar_fields = (
        "task_index",
        "source_arm_binding_sha256",
        "source_rank_reference_sha256",
        "source_rank_grid_sha256",
        "strata_selection_sha256",
        "target_position",
        "target_source_index",
    )
    if any(getattr(orbit, name) != getattr(support64, name) for name in scalar_fields):
        raise ProbeTargetArtifactError("support64 identity does not match ORBIT target evidence")
    if (
        orbit.compute_dtype != torch.float64
        or support64.compute_dtype != torch.float64
        or not orbit.include_stratum_sweep
    ):
        raise ProbeTargetArtifactError(
            "support64 may accompany only a selected CPU-float64 ORBIT target"
        )
    if not torch.equal(orbit.neighbour_positions, support64.neighbour_positions) or not torch.equal(
        orbit.neighbour_source_indices,
        support64.neighbour_source_indices,
    ):
        raise ProbeTargetArtifactError("support64 neighbours do not match ORBIT evidence")


def _validate_matching_full_q(
    orbit: OrbitTargetExecution,
    full_q: FullQTargetExecution,
) -> None:
    scalar_fields = (
        "task_index",
        "source_arm_binding_sha256",
        "source_rank_reference_sha256",
        "source_rank_grid_sha256",
        "strata_selection_sha256",
        "target_position",
        "target_source_index",
    )
    if any(getattr(orbit, name) != getattr(full_q, name) for name in scalar_fields):
        raise ProbeTargetArtifactError("full-q identity does not match ORBIT target evidence")
    if (
        orbit.compute_dtype != torch.float64
        or not orbit.include_stratum_sweep
        or full_q.m != FULL_Q_M
        or full_q.q_system_dimension != FULL_Q_M * FULL_Q_M
        or full_q.support_rank != orbit.system.geometry.rank
        or full_q.canonical_arm_name != FULL_Q_ARM_NAMES[2]
        or tuple(arm.name for arm in full_q.arms) != FULL_Q_ARM_NAMES
    ):
        raise ProbeTargetArtifactError(
            "full-q may accompany only its selected CPU-float64 ORBIT target"
        )
    if not torch.equal(orbit.neighbour_positions, full_q.neighbour_positions) or not torch.equal(
        orbit.neighbour_source_indices,
        full_q.neighbour_source_indices,
    ):
        raise ProbeTargetArtifactError("full-q neighbours do not match ORBIT evidence")
    _validate_sha256(full_q.support_projector_sha256, "full_q.support_projector_sha256")
    for index, arm in enumerate(full_q.arms):
        _validate_sha256(
            arm.represented_system_sha256,
            f"full_q.arms[{index}].represented_system_sha256",
        )
        _validate_sha256(
            arm.represented_rhs_sha256,
            f"full_q.arms[{index}].represented_rhs_sha256",
        )


def _validate_matching_stress(
    orbit: OrbitTargetExecution,
    stress: StressTargetExecution,
) -> None:
    scalar_fields = (
        "task_index",
        "source_arm_binding_sha256",
        "target_position",
        "target_source_index",
    )
    if any(getattr(orbit, name) != getattr(stress, name) for name in scalar_fields):
        raise ProbeTargetArtifactError("stress identity does not match ORBIT target evidence")
    task = probe_task_for_index(orbit.task_index)
    task_record = task.as_record()
    expected_iterations = min(
        4 * stress.m * min(stress.m, task_record["D"] - 6),
        4096,
    )
    if (
        orbit.compute_dtype != torch.float64
        or task.stress_m != stress.m
        or stress.requested_tolerance != STRESS_TOLERANCE
        or stress.max_iterations != expected_iterations
        or stress.base_solve.get("rank") != stress.selected_rank
        or stress.base_solve.get("requested_tolerance") != STRESS_TOLERANCE
        or stress.base_solve.get("max_iterations") != expected_iterations
        or stress.neighbour_positions.dtype != torch.long
        or stress.neighbour_source_indices.dtype != torch.long
        or stress.neighbour_positions.device.type != "cpu"
        or stress.neighbour_source_indices.device.type != "cpu"
        or stress.neighbour_positions.shape != (stress.m,)
        or stress.neighbour_source_indices.shape != (stress.m,)
        or stress.source_q_projector.dtype != torch.float64
        or stress.native_fp64_q_projector.dtype != torch.float64
        or stress.source_q_projector.device.type != "cpu"
        or stress.native_fp64_q_projector.device.type != "cpu"
        or stress.source_q_projector.shape != (stress.m, stress.m)
        or stress.native_fp64_q_projector.shape != (stress.m, stress.m)
    ):
        raise ProbeTargetArtifactError(
            "stress may accompany only its registered CPU-float64 ORBIT target"
        )
    for name in (
        "stress_binding_sha256",
        "source_rank_reference_sha256",
        "source_rank_grid_sha256",
        "strata_selection_sha256",
    ):
        _validate_sha256(getattr(stress, name), f"stress.{name}")


def build_canonical_probe_target_artifact(
    orbit: OrbitTargetExecution,
    *,
    support64: Support64TargetExecution | None = None,
    full_q: FullQTargetExecution | None = None,
    stress: StressTargetExecution | None = None,
) -> CanonicalProbeTargetArtifact:
    """Copy one target execution into immutable, hash-bound canonical bytes."""

    if type(orbit) is not OrbitTargetExecution:
        raise ProbeTargetArtifactError("orbit must be an exact OrbitTargetExecution")
    if (
        type(orbit.target_position) is not int
        or not 0 <= orbit.target_position < PRIMARY_EVALUATION_ROW_COUNT
        or type(orbit.target_source_index) is not int
        or orbit.target_source_index < 0
    ):
        raise ProbeTargetArtifactError("ORBIT target identity is invalid")
    if (
        orbit.neighbour_positions.dtype != torch.long
        or orbit.neighbour_source_indices.dtype != torch.long
        or orbit.neighbour_positions.ndim != 1
        or orbit.neighbour_source_indices.shape != orbit.neighbour_positions.shape
    ):
        raise ProbeTargetArtifactError("ORBIT neighbour identity is malformed")
    for name in (
        "source_arm_binding_sha256",
        "source_rank_reference_sha256",
        "source_rank_grid_sha256",
        "strata_selection_sha256",
    ):
        _validate_sha256(getattr(orbit, name), f"orbit.{name}")
    if support64 is not None:
        if type(support64) is not Support64TargetExecution:
            raise ProbeTargetArtifactError(
                "support64 must be an exact Support64TargetExecution"
            )
        _validate_matching_support64(orbit, support64)
    if full_q is not None:
        if type(full_q) is not FullQTargetExecution:
            raise ProbeTargetArtifactError("full_q must be an exact FullQTargetExecution")
        _validate_matching_full_q(orbit, full_q)
    if stress is not None:
        if type(stress) is not StressTargetExecution:
            raise ProbeTargetArtifactError("stress must be an exact StressTargetExecution")
        _validate_matching_stress(orbit, stress)
    try:
        task_record = probe_task_for_index(orbit.task_index).as_record()
    except (IndexError, ValueError) as error:
        raise ProbeTargetArtifactError("ORBIT task index is outside the frozen grid") from error
    payload = {
        "schema_version": PROBE_TARGET_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": PROBE_TARGET_ARTIFACT_TYPE,
        "calibration_id": CALIBRATION_ID,
        "task_index": orbit.task_index,
        "task_record": task_record,
        "target_position": orbit.target_position,
        "target_source_index": orbit.target_source_index,
        "source_arm_binding_sha256": orbit.source_arm_binding_sha256,
        "source_rank_reference_sha256": orbit.source_rank_reference_sha256,
        "source_rank_grid_sha256": orbit.source_rank_grid_sha256,
        "strata_selection_sha256": orbit.strata_selection_sha256,
        "orbit": _orbit_payload(orbit),
        "support64": None if support64 is None else _support64_payload(support64),
        "full_q": None if full_q is None else _full_q_payload(full_q),
        "stress": None if stress is None else _stress_payload(stress),
    }
    try:
        payload_bytes = canonical_json_bytes(payload)
    except CalibrationContractError as error:
        raise ProbeTargetArtifactError("probe target payload is not strict canonical JSON") from error
    return CanonicalProbeTargetArtifact(
        payload_bytes=payload_bytes,
        payload_sha256=hashlib.sha256(payload_bytes).hexdigest(),
    )


def parse_canonical_probe_target_artifact(
    payload_bytes: bytes,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Parse and minimally validate immutable canonical target bytes."""

    if type(payload_bytes) is not bytes:
        raise ProbeTargetArtifactError("artifact parser requires immutable bytes")
    observed_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    if expected_sha256 is not None and (
        _validate_sha256(expected_sha256, "expected_sha256") != observed_sha256
    ):
        raise ProbeTargetArtifactError("artifact raw SHA-256 is mismatched")
    try:
        parsed = parse_strict_json_bytes(payload_bytes)
        if canonical_json_bytes(parsed) != payload_bytes:
            raise ProbeTargetArtifactError("artifact bytes are not canonically encoded")
    except CalibrationContractError as error:
        raise ProbeTargetArtifactError("artifact bytes are not strict canonical JSON") from error
    if not isinstance(parsed, dict) or set(parsed) != _TOP_LEVEL_FIELDS:
        raise ProbeTargetArtifactError("artifact top-level fields are mismatched")
    if (
        parsed["schema_version"] != PROBE_TARGET_ARTIFACT_SCHEMA_VERSION
        or parsed["artifact_type"] != PROBE_TARGET_ARTIFACT_TYPE
        or parsed["calibration_id"] != CALIBRATION_ID
        or type(parsed["task_index"]) is not int
    ):
        raise ProbeTargetArtifactError("artifact schema or calibration identity is invalid")
    try:
        expected_task = probe_task_for_index(parsed["task_index"]).as_record()
    except (IndexError, ValueError) as error:
        raise ProbeTargetArtifactError("artifact task index is outside the frozen grid") from error
    if parsed["task_record"] != expected_task:
        raise ProbeTargetArtifactError("artifact task record is mismatched")
    if (
        type(parsed["target_position"]) is not int
        or not 0 <= parsed["target_position"] < PRIMARY_EVALUATION_ROW_COUNT
        or type(parsed["target_source_index"]) is not int
        or parsed["target_source_index"] < 0
    ):
        raise ProbeTargetArtifactError("artifact target identity is invalid")
    for name in (
        "source_arm_binding_sha256",
        "source_rank_reference_sha256",
        "source_rank_grid_sha256",
        "strata_selection_sha256",
    ):
        _validate_sha256(parsed[name], name)
    if not isinstance(parsed["orbit"], dict) or any(
        parsed[name] is not None and not isinstance(parsed[name], dict)
        for name in ("support64", "full_q", "stress")
    ):
        raise ProbeTargetArtifactError("artifact numerical arm records are malformed")
    if parsed["full_q"] is not None:
        full_q = parsed["full_q"]
        if set(full_q) != _FULL_Q_FIELDS or (
            parsed["orbit"].get("compute_dtype") != "float64"
            or parsed["orbit"].get("include_stratum_sweep") is not True
            or full_q["m"] != FULL_Q_M
            or full_q["q_system_dimension"] != FULL_Q_M * FULL_Q_M
            or full_q["canonical_arm_name"] != FULL_Q_ARM_NAMES[2]
            or not isinstance(full_q["arms"], list)
            or tuple(arm.get("name") for arm in full_q["arms"] if isinstance(arm, dict))
            != FULL_Q_ARM_NAMES
        ):
            raise ProbeTargetArtifactError("artifact full-q record is malformed")
        _validate_sha256(
            full_q["support_projector_sha256"],
            "full_q.support_projector_sha256",
        )
    if parsed["stress"] is not None:
        stress = parsed["stress"]
        if set(stress) != _STRESS_FIELDS:
            raise ProbeTargetArtifactError("artifact stress record is malformed")
        if (
            type(stress["m"]) is not int
            or type(stress["selected_rank"]) is not int
            or type(stress["max_iterations"]) is not int
            or type(stress["requested_tolerance"]) is not float
        ):
            raise ProbeTargetArtifactError("artifact stress record is malformed")
        expected_iterations = min(
            4
            * stress["m"]
            * min(stress["m"], parsed["task_record"]["D"] - 6),
            4096,
        )
        if (
            parsed["orbit"].get("compute_dtype") != "float64"
            or stress["m"] != parsed["task_record"]["stress_m"]
            or stress["requested_tolerance"] != STRESS_TOLERANCE
            or stress["max_iterations"] != expected_iterations
            or not isinstance(stress["base_solve"], dict)
            or stress["base_solve"].get("rank") != stress["selected_rank"]
            or stress["base_solve"].get("requested_tolerance")
            != STRESS_TOLERANCE
            or stress["base_solve"].get("max_iterations") != expected_iterations
            or not isinstance(stress["tests"], dict)
            or set(stress["tests"])
            != {
                "support_complement",
                "permutation",
                "support_rotation",
                "exact_zero_augmentation",
                "discarded_mode_leakage",
            }
        ):
            raise ProbeTargetArtifactError("artifact stress record is malformed")
        for name in (
            "stress_binding_sha256",
            "source_rank_reference_sha256",
            "source_rank_grid_sha256",
            "strata_selection_sha256",
        ):
            _validate_sha256(stress[name], f"stress.{name}")
    return parsed


__all__ = [
    "CanonicalProbeTargetArtifact",
    "PROBE_TARGET_ARTIFACT_SCHEMA_VERSION",
    "PROBE_TARGET_ARTIFACT_TYPE",
    "ProbeTargetArtifactError",
    "build_canonical_probe_target_artifact",
    "parse_canonical_probe_target_artifact",
]
