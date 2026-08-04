"""Public, label-free execution-envelope contract for F02b calibration.

This module binds scheduler-visible resources and numeric-payload identities to
the immutable development-only task grids.  It is deliberately a pure API: it
does not read paths, corpora, payloads, labels, environment variables, Slurm,
or mutable runtime state.  Numeric payloads remain separate raw-byte artifacts;
only their canonical path identity and SHA-256 enter an envelope.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from cluster.f02b_calibration_grid import (
    CALIBRATION_ID,
    CALIBRATION_MATRIX_SHA256,
    DEVELOPMENT_REPLICAS,
    FIT_TASK_COUNT,
    FIT_TASK_MATRIX_SHA256,
    FIT_TASKS,
    PROBE_TASK_COUNT,
    PROBE_TASK_MATRIX_SHA256,
    PROBE_TASKS,
)

EXECUTION_ENVELOPE_SCHEMA_VERSION = "f02b_calibration_execution_envelope_v1"
TASK_ROLES = ("fit", "probe")

MINIMUM_GPU_MEMORY_BYTES = 48_000_000_000
MINIMUM_HOST_MEMORY_BYTES = 64 * 1024**3
WALLTIME_SECONDS = 8 * 60 * 60
F02_CATALOG_SHA256 = "2dee429bdaf50cc78cb40ba8b038f7e4731bb07127927c55607b528c1db66942"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class CalibrationContractError(ValueError):
    """Raised when an execution envelope is not exactly the frozen contract."""


class _FrozenJSONDict(dict[str, Any]):
    """A JSON-serializable dictionary whose public instance cannot be mutated."""

    @staticmethod
    def _immutable(*_args: object, **_kwargs: object) -> None:
        raise TypeError("frozen JSON contract constants cannot be mutated")

    __delitem__ = _immutable
    __ior__ = _immutable
    __setitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


# Seed is intentionally absent: the frozen task record is its sole authority.
# Prediction-neighbour, CG, jitter, and prediction-only fields are intentionally
# absent because this recipe describes only the shared fit.
FIT_RECIPE = _FrozenJSONDict(
    {
        "train_steps": 20,
        "train_epochs": 0,
        "training_m": 20,
        "kernel": "rbf",
        "outputscale": 1.0,
        "sigma_f": 1e-3,
        "sigma_g": 1e-3,
        "lengthscale": 1.0,
        "lengthscale_init": "median",
        "lengthscale_init_max_points": 2048,
        "use_ard": False,
        "batch_size": 256,
        "lr": 0.01,
        "weight_decay": 0.0,
        "graph_refresh_epochs": 0,
        "learn_lengthscale": True,
        "learn_outputscale": True,
        "learn_sigma_f": True,
        "learn_sigma_g": True,
        "min_sigma_f": 1e-6,
        "min_sigma_g": 0.0,
        "dtype": "float32",
        "device": "cuda",
    }
)
FIT_RECIPE_SHA256 = "3aaf4ba3172c5fa9a8878d321570385efb31d6050330268ebbf1399ec9fad421"
EXCLUDED_FIT_RECIPE_FIELDS = frozenset(
    {
        "seed",
        "candidate_m",
        "cg_tolerance",
        "cg_max_iterations",
        "use_preconditioner",
        "function_jitter",
        "reduced_jitter",
        "prediction_m",
        "prediction_only",
    }
)

RESOURCE_CONTRACT = _FrozenJSONDict(
    {
        "exclusive_node": True,
        "requested_gpu_count": 1,
        "required_gpu_model": "NVIDIA L40S",
        "minimum_gpu_memory_bytes": MINIMUM_GPU_MEMORY_BYTES,
        "requested_cpus_per_task": 16,
        "minimum_host_memory_bytes": MINIMUM_HOST_MEMORY_BYTES,
        "requested_walltime_seconds": WALLTIME_SECONDS,
        "array_concurrency": 1,
        "allowed_partitions": ("short", "interactivegpu"),
    }
)

NUMERICAL_POLICY = _FrozenJSONDict(
    {
        "source_dtype": "float32",
        "source_device": "cuda",
        "canonical_comparison_dtype": "float64",
        "canonical_comparison_device": "cpu",
        "physical_compute_dtype": "float64",
        "physical_compute_device": "cpu",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
    }
)

MATRIX_HASHES = _FrozenJSONDict(
    {
        "fit_task_matrix_sha256": FIT_TASK_MATRIX_SHA256,
        "probe_task_matrix_sha256": PROBE_TASK_MATRIX_SHA256,
        "calibration_matrix_sha256": CALIBRATION_MATRIX_SHA256,
    }
)

CATALOG_IDENTITY = _FrozenJSONDict(
    {
        "catalog_sha256": F02_CATALOG_SHA256,
        "binding_scope": "identity-only-no-payload-access",
    }
)

_RUNTIME_ALLOCATION_KEYS = {
    "exclusive_node",
    "requested_gpu_count",
    "visible_gpu_count",
    "visible_gpu_models",
    "visible_gpu_memory_bytes",
    "requested_cpus_per_task",
    "available_cpu_count",
    "available_host_memory_bytes",
    "requested_walltime_seconds",
    "walltime_limit_seconds",
    "array_concurrency",
    "partition",
}
_PAYLOAD_BINDING_KEYS = {
    "task_role",
    "task_index",
    "numeric_payload_path",
    "numeric_payload_sha256",
}
_CANONICAL_RECORD_KEYS = {
    "schema_version",
    "calibration_id",
    "task_role",
    "task_index",
    "task_record",
    "matrix_hashes",
    "catalog_identity",
    "fit_recipe",
    "fit_recipe_sha256",
    "resource_contract",
    "numerical_policy",
    "runtime_allocation",
    "payload_binding",
}
_ENVELOPE_KEYS = {"canonical_record", "canonical_record_sha256"}


def _validate_json_value(
    value: Any,
    *,
    label: str = "$",
    active: set[int] | None = None,
) -> None:
    """Reject values outside finite JSON plus immutable tuple arrays."""

    if active is None:
        active = set()
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise CalibrationContractError(f"{label} contains a nonfinite number")
        return
    if isinstance(value, dict):
        identity = id(value)
        if identity in active:
            raise CalibrationContractError(f"{label} contains a recursive object")
        active.add(identity)
        try:
            for key, item in value.items():
                if type(key) is not str:
                    raise CalibrationContractError(f"{label} contains a non-string object key")
                _validate_json_value(item, label=f"{label}.{key}", active=active)
        finally:
            active.remove(identity)
        return
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise CalibrationContractError(f"{label} contains a recursive array")
        active.add(identity)
        try:
            for index, item in enumerate(value):
                _validate_json_value(item, label=f"{label}[{index}]", active=active)
        finally:
            active.remove(identity)
        return
    raise CalibrationContractError(f"{label} is not a strict JSON value")


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic finite JSON bytes without accepting implicit coercions."""

    _validate_json_value(value)
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CalibrationContractError("value is not canonical finite JSON") from error


def canonical_sha256(value: Any) -> str:
    """Hash the canonical strict-JSON representation of ``value``."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def parse_strict_json_bytes(value: bytes) -> Any:
    """Parse in-memory JSON bytes while rejecting duplicate keys and constants."""

    if type(value) is not bytes:
        raise CalibrationContractError("strict JSON input must be bytes")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise CalibrationContractError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    def reject_constant(token: str) -> None:
        raise CalibrationContractError(f"nonfinite JSON constant: {token}")

    try:
        parsed = json.loads(
            value.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except CalibrationContractError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CalibrationContractError("invalid strict JSON bytes") from error
    canonical_json_bytes(parsed)
    return parsed


def _fresh_json(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value))


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CalibrationContractError(f"{label} must be an object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise CalibrationContractError(
            f"{label} fields mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _require_plain_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise CalibrationContractError(f"{label} must be an integer, not bool or another type")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise CalibrationContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_canonical_match(value: Any, expected: Any, label: str) -> None:
    if canonical_json_bytes(value) != canonical_json_bytes(expected):
        raise CalibrationContractError(f"{label} does not match the frozen contract")


def _validate_task_role(value: Any) -> str:
    if type(value) is not str or value not in TASK_ROLES:
        raise CalibrationContractError(f"task_role must be one of {TASK_ROLES}")
    return value


def _task_count(task_role: str) -> int:
    return FIT_TASK_COUNT if task_role == "fit" else PROBE_TASK_COUNT


def _validate_task_index(task_role: str, value: Any) -> int:
    task_index = _require_plain_int(value, "task_index")
    count = _task_count(task_role)
    if not 0 <= task_index < count:
        raise CalibrationContractError(
            f"{task_role} task_index {task_index} is outside [0, {count})"
        )
    return task_index


def _expected_task_record(task_role: str, task_index: int) -> dict[str, Any]:
    task = FIT_TASKS[task_index] if task_role == "fit" else PROBE_TASKS[task_index]
    return task.as_record()


def validate_fit_recipe(value: Any) -> dict[str, Any]:
    """Validate and return a fresh copy of the prediction-free fit recipe."""

    recipe = _require_mapping(value, "fit_recipe")
    _require_exact_keys(recipe, set(FIT_RECIPE), "fit_recipe")
    if set(recipe) & EXCLUDED_FIT_RECIPE_FIELDS:
        raise CalibrationContractError("fit_recipe contains a prediction-only or task-owned field")
    _require_canonical_match(recipe, FIT_RECIPE, "fit_recipe")
    if canonical_sha256(recipe) != FIT_RECIPE_SHA256:
        raise CalibrationContractError("fit_recipe SHA-256 is not frozen")
    return _fresh_json(recipe)


def validate_resource_contract(value: Any) -> dict[str, Any]:
    """Validate an embedded copy of the immutable requested-resource contract."""

    contract = _require_mapping(value, "resource_contract")
    _require_exact_keys(contract, set(RESOURCE_CONTRACT), "resource_contract")
    _require_canonical_match(contract, RESOURCE_CONTRACT, "resource_contract")
    return _fresh_json(contract)


def validate_runtime_allocation(value: Any) -> dict[str, Any]:
    """Validate one concrete allocation against the immutable resource contract."""

    canonical_json_bytes(value)
    allocation = _require_mapping(value, "runtime_allocation")
    _require_exact_keys(allocation, _RUNTIME_ALLOCATION_KEYS, "runtime_allocation")
    if allocation["exclusive_node"] is not True:
        raise CalibrationContractError("runtime allocation must verify an exclusive node")
    integer_fields = (
        "requested_gpu_count",
        "visible_gpu_count",
        "requested_cpus_per_task",
        "available_cpu_count",
        "available_host_memory_bytes",
        "requested_walltime_seconds",
        "walltime_limit_seconds",
        "array_concurrency",
    )
    for field in integer_fields:
        _require_plain_int(allocation[field], f"runtime_allocation.{field}")
    if allocation["requested_gpu_count"] != RESOURCE_CONTRACT["requested_gpu_count"]:
        raise CalibrationContractError("runtime allocation must record a one-GPU request")
    visible_count = allocation["visible_gpu_count"]
    if visible_count < 1:
        raise CalibrationContractError("runtime allocation must expose at least one GPU")
    models = allocation["visible_gpu_models"]
    memory_values = allocation["visible_gpu_memory_bytes"]
    if not isinstance(models, list) or len(models) != visible_count:
        raise CalibrationContractError("visible GPU models do not match visible_gpu_count")
    if not isinstance(memory_values, list) or len(memory_values) != visible_count:
        raise CalibrationContractError("visible GPU memory does not match visible_gpu_count")
    if any(model != RESOURCE_CONTRACT["required_gpu_model"] for model in models):
        raise CalibrationContractError("every visible runtime GPU must be NVIDIA L40S")
    for index, memory_bytes in enumerate(memory_values):
        _require_plain_int(memory_bytes, f"runtime_allocation.visible_gpu_memory_bytes[{index}]")
        if memory_bytes < MINIMUM_GPU_MEMORY_BYTES:
            raise CalibrationContractError("a visible runtime GPU has insufficient memory")
    if allocation["requested_cpus_per_task"] != RESOURCE_CONTRACT["requested_cpus_per_task"]:
        raise CalibrationContractError("runtime allocation must record a 16-CPU task request")
    if allocation["available_cpu_count"] < RESOURCE_CONTRACT["requested_cpus_per_task"]:
        raise CalibrationContractError("runtime allocation exposes fewer than 16 CPUs")
    if allocation["available_host_memory_bytes"] < MINIMUM_HOST_MEMORY_BYTES:
        raise CalibrationContractError("runtime allocation has less than 64 GiB host memory")
    if (
        allocation["requested_walltime_seconds"] != WALLTIME_SECONDS
        or allocation["walltime_limit_seconds"] != WALLTIME_SECONDS
    ):
        raise CalibrationContractError(
            "runtime allocation walltime request and limit must both be exactly eight hours"
        )
    if allocation["array_concurrency"] != RESOURCE_CONTRACT["array_concurrency"]:
        raise CalibrationContractError("runtime allocation array concurrency must be one")
    if allocation["partition"] not in RESOURCE_CONTRACT["allowed_partitions"]:
        raise CalibrationContractError("runtime allocation partition is not registered")
    return _fresh_json(allocation)


def _validate_numeric_payload_path(value: Any) -> str:
    if (
        type(value) is not str
        or not value
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise CalibrationContractError(
            "numeric_payload_path must be a canonical relative POSIX path"
        )
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or value != pure.as_posix() or not pure.name:
        raise CalibrationContractError(
            "numeric_payload_path must be a canonical relative POSIX path"
        )
    return value


def build_payload_binding(
    task_role: str,
    task_index: int,
    *,
    numeric_payload_path: str,
    numeric_payload_sha256: str,
) -> dict[str, Any]:
    """Build a path/hash identity without reading or embedding numeric payload bytes."""

    role = _validate_task_role(task_role)
    index = _validate_task_index(role, task_index)
    return {
        "task_role": role,
        "task_index": index,
        "numeric_payload_path": _validate_numeric_payload_path(numeric_payload_path),
        "numeric_payload_sha256": _require_sha256(
            numeric_payload_sha256,
            "numeric_payload_sha256",
        ),
    }


def validate_payload_binding(
    value: Any,
    *,
    expected_task_role: str | None = None,
    expected_task_index: int | None = None,
) -> dict[str, Any]:
    """Strictly validate a numeric payload identity without opening its path."""

    canonical_json_bytes(value)
    binding = _require_mapping(value, "payload_binding")
    _require_exact_keys(binding, _PAYLOAD_BINDING_KEYS, "payload_binding")
    normalized = build_payload_binding(
        binding["task_role"],
        binding["task_index"],
        numeric_payload_path=binding["numeric_payload_path"],
        numeric_payload_sha256=binding["numeric_payload_sha256"],
    )
    if expected_task_role is not None:
        role = _validate_task_role(expected_task_role)
        if normalized["task_role"] != role:
            raise CalibrationContractError("payload_binding task role is mismatched")
    if expected_task_index is not None:
        role = normalized["task_role"] if expected_task_role is None else expected_task_role
        index = _validate_task_index(role, expected_task_index)
        if normalized["task_index"] != index:
            raise CalibrationContractError("payload_binding task index is mismatched")
    return normalized


def verify_numeric_payload_bytes(
    payload_binding: Any,
    payload_bytes: bytes,
    *,
    expected_task_role: str | None = None,
    expected_task_index: int | None = None,
) -> dict[str, Any]:
    """Verify raw bytes against a binding without parsing their numeric contents."""

    binding = validate_payload_binding(
        payload_binding,
        expected_task_role=expected_task_role,
        expected_task_index=expected_task_index,
    )
    if type(payload_bytes) is not bytes:
        raise CalibrationContractError("numeric payload must be supplied as raw bytes")
    if hashlib.sha256(payload_bytes).hexdigest() != binding["numeric_payload_sha256"]:
        raise CalibrationContractError("numeric payload raw-bytes SHA-256 is mismatched")
    return binding


def _canonical_record(
    task_role: str,
    task_index: int,
    *,
    runtime_allocation: Any,
    payload_binding: Any,
) -> dict[str, Any]:
    allocation = validate_runtime_allocation(runtime_allocation)
    binding = validate_payload_binding(
        payload_binding,
        expected_task_role=task_role,
        expected_task_index=task_index,
    )
    return {
        "schema_version": EXECUTION_ENVELOPE_SCHEMA_VERSION,
        "calibration_id": CALIBRATION_ID,
        "task_role": task_role,
        "task_index": task_index,
        "task_record": _fresh_json(_expected_task_record(task_role, task_index)),
        "matrix_hashes": _fresh_json(MATRIX_HASHES),
        "catalog_identity": _fresh_json(CATALOG_IDENTITY),
        "fit_recipe": _fresh_json(FIT_RECIPE),
        "fit_recipe_sha256": FIT_RECIPE_SHA256,
        "resource_contract": _fresh_json(RESOURCE_CONTRACT),
        "numerical_policy": _fresh_json(NUMERICAL_POLICY),
        "runtime_allocation": allocation,
        "payload_binding": binding,
    }


def build_execution_envelope(
    task_role: str,
    task_index: int,
    *,
    runtime_allocation: Any,
    payload_binding: Any,
) -> dict[str, Any]:
    """Build a deterministic fit or probe envelope from public identities only."""

    role = _validate_task_role(task_role)
    index = _validate_task_index(role, task_index)
    record = _canonical_record(
        role,
        index,
        runtime_allocation=runtime_allocation,
        payload_binding=payload_binding,
    )
    return {
        "canonical_record": record,
        # Kept outside ``record`` to avoid a recursive/self-referential hash.
        "canonical_record_sha256": canonical_sha256(record),
    }


def build_fit_execution_envelope(
    task_index: int,
    *,
    runtime_allocation: Any,
    payload_binding: Any,
) -> dict[str, Any]:
    """Build one envelope for a frozen fit-task index."""

    return build_execution_envelope(
        "fit",
        task_index,
        runtime_allocation=runtime_allocation,
        payload_binding=payload_binding,
    )


def build_probe_execution_envelope(
    task_index: int,
    *,
    runtime_allocation: Any,
    payload_binding: Any,
) -> dict[str, Any]:
    """Build one envelope for a frozen probe-task index."""

    return build_execution_envelope(
        "probe",
        task_index,
        runtime_allocation=runtime_allocation,
        payload_binding=payload_binding,
    )


def validate_execution_envelope(
    value: Any,
    *,
    expected_task_role: str | None = None,
    expected_task_index: int | None = None,
) -> dict[str, Any]:
    """Validate every envelope field and return a detached canonical copy."""

    canonical_json_bytes(value)
    envelope = _require_mapping(value, "execution envelope")
    _require_exact_keys(envelope, _ENVELOPE_KEYS, "execution envelope")
    record = _require_mapping(envelope["canonical_record"], "canonical_record")
    _require_exact_keys(record, _CANONICAL_RECORD_KEYS, "canonical_record")
    _require_sha256(envelope["canonical_record_sha256"], "canonical_record_sha256")
    if envelope["canonical_record_sha256"] != canonical_sha256(record):
        raise CalibrationContractError("canonical_record SHA-256 is mismatched")
    if record["schema_version"] != EXECUTION_ENVELOPE_SCHEMA_VERSION:
        raise CalibrationContractError("execution-envelope schema version is unsupported")
    if record["calibration_id"] != CALIBRATION_ID:
        raise CalibrationContractError("calibration ID is mismatched")

    role = _validate_task_role(record["task_role"])
    index = _validate_task_index(role, record["task_index"])
    if expected_task_role is not None and role != _validate_task_role(expected_task_role):
        raise CalibrationContractError("execution-envelope task role is mismatched")
    if expected_task_index is not None:
        expected_role = role if expected_task_role is None else expected_task_role
        if index != _validate_task_index(expected_role, expected_task_index):
            raise CalibrationContractError("execution-envelope task index is mismatched")

    task_record = _require_mapping(record["task_record"], "task_record")
    expected_task_record = _expected_task_record(role, index)
    _require_exact_keys(task_record, set(expected_task_record), "task_record")
    replica = _require_plain_int(task_record.get("replica"), "task_record.replica")
    if replica not in DEVELOPMENT_REPLICAS:
        raise CalibrationContractError("confirmatory replicas are forbidden in calibration")
    _require_canonical_match(task_record, expected_task_record, "task_record")
    if role == "probe" and task_record["role"] != PROBE_TASKS[index].role:
        raise CalibrationContractError("probe role is mismatched")

    _require_canonical_match(record["matrix_hashes"], MATRIX_HASHES, "matrix_hashes")
    _require_canonical_match(record["catalog_identity"], CATALOG_IDENTITY, "catalog_identity")
    validate_fit_recipe(record["fit_recipe"])
    if record["fit_recipe_sha256"] != FIT_RECIPE_SHA256:
        raise CalibrationContractError("fit_recipe_sha256 is mismatched")
    validate_resource_contract(record["resource_contract"])
    _require_canonical_match(record["numerical_policy"], NUMERICAL_POLICY, "numerical_policy")
    allocation = validate_runtime_allocation(record["runtime_allocation"])
    binding = validate_payload_binding(
        record["payload_binding"],
        expected_task_role=role,
        expected_task_index=index,
    )

    expected_record = _canonical_record(
        role,
        index,
        runtime_allocation=allocation,
        payload_binding=binding,
    )
    _require_canonical_match(record, expected_record, "canonical_record")
    return _fresh_json(envelope)


def validate_fit_execution_envelope(
    value: Any,
    *,
    expected_task_index: int | None = None,
) -> dict[str, Any]:
    """Validate an envelope as a fit task."""

    return validate_execution_envelope(
        value,
        expected_task_role="fit",
        expected_task_index=expected_task_index,
    )


def validate_probe_execution_envelope(
    value: Any,
    *,
    expected_task_index: int | None = None,
) -> dict[str, Any]:
    """Validate an envelope as a probe task."""

    return validate_execution_envelope(
        value,
        expected_task_role="probe",
        expected_task_index=expected_task_index,
    )


if FIT_TASK_COUNT != 45 or PROBE_TASK_COUNT != 122:
    raise RuntimeError("F02b calibration contract requires the frozen 45/122 task matrices")
if canonical_sha256(FIT_RECIPE) != FIT_RECIPE_SHA256:
    raise RuntimeError("F02b fit recipe does not match its frozen SHA-256")
if set(FIT_RECIPE) & EXCLUDED_FIT_RECIPE_FIELDS:
    raise RuntimeError("F02b fit recipe contains task-owned or prediction-only fields")


__all__ = [
    "CALIBRATION_ID",
    "CALIBRATION_MATRIX_SHA256",
    "CATALOG_IDENTITY",
    "CalibrationContractError",
    "EXECUTION_ENVELOPE_SCHEMA_VERSION",
    "EXCLUDED_FIT_RECIPE_FIELDS",
    "F02_CATALOG_SHA256",
    "FIT_RECIPE",
    "FIT_RECIPE_SHA256",
    "FIT_TASK_COUNT",
    "FIT_TASK_MATRIX_SHA256",
    "MATRIX_HASHES",
    "MINIMUM_GPU_MEMORY_BYTES",
    "MINIMUM_HOST_MEMORY_BYTES",
    "NUMERICAL_POLICY",
    "PROBE_TASK_COUNT",
    "PROBE_TASK_MATRIX_SHA256",
    "RESOURCE_CONTRACT",
    "TASK_ROLES",
    "WALLTIME_SECONDS",
    "build_execution_envelope",
    "build_fit_execution_envelope",
    "build_payload_binding",
    "build_probe_execution_envelope",
    "canonical_json_bytes",
    "canonical_sha256",
    "parse_strict_json_bytes",
    "validate_execution_envelope",
    "validate_fit_execution_envelope",
    "validate_fit_recipe",
    "validate_payload_binding",
    "validate_probe_execution_envelope",
    "validate_resource_contract",
    "validate_runtime_allocation",
    "verify_numeric_payload_bytes",
]
