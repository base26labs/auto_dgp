"""Pure-JSON launch contract for the F02b calibration probe stage.

The manifest binds deployment identities, immutable input/output locations,
the completed fit-stage catalog, and the exact three-submission scheduler
topology.  It contains no probe scientific coordinates: those remain owned
solely by ``cluster.f02b_calibration_grid``.

This module performs no filesystem or scheduler access.  Filesystem identity,
live allocation evidence, and task execution belong to the future probe
preflight and runner, which must verify this manifest's canonical raw bytes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any

from cluster.f02b_calibration_grid import CALIBRATION_ID, PROBE_TASK_COUNT
from experiments.f02b_calibration_contract import (
    FIT_RECIPE_SHA256,
    CalibrationContractError,
    canonical_json_bytes,
    canonical_sha256,
    parse_strict_json_bytes,
)
from experiments.f02b_calibration_fit_aggregate import (
    CalibrationFitAggregateError,
    validate_expected_deployment,
)
from experiments.f02b_calibration_probe_core import canonical_probe_work_plan_payload

PROBE_LAUNCH_MANIFEST_SCHEMA_VERSION = "f02b_calibration_probe_launch_manifest_v2"
PROBE_LAUNCH_MANIFEST_TYPE = "f02b_calibration_probe_launch"

# These independent canonical byte literals are the authority used by builders
# and validators.  In particular, they are not snapshots made from the public
# dict subclasses in f02b_calibration_contract: dict.__setitem__ can bypass
# those subclasses' overridden mutators.  Each use parses a fresh plain-JSON
# value from these immutable bytes.
_MATRIX_HASHES_CANONICAL_BYTES = (
    b'{"calibration_matrix_sha256":"0ead06b0e2f6de24c49f4bf6f999f90690ff1fb82be3585cc212bdd11fd411f4",'
    b'"fit_task_matrix_sha256":"7272f823a2bfc0f52cbfc2e27ae3a56b2f668e3ca2abff054de9209cd2fa5a39",'
    b'"probe_task_matrix_sha256":"98a44d167f6a34d3e94dcffd026d030e56bcf30f77a0bd43810f16b311e54eca"}'
)
_RESOURCE_CONTRACT_CANONICAL_BYTES = (
    b'{"allowed_partitions":["short"],"array_concurrency":1,"exclusive_node":false,'
    b'"minimum_gpu_memory_bytes":0,"minimum_host_memory_bytes":68719476736,'
    b'"requested_cpus_per_task":8,"requested_gpu_count":0,'
    b'"requested_walltime_seconds":28800,"required_gpu_model":null}'
)
_NUMERICAL_POLICY_CANONICAL_BYTES = (
    b'{"canonical_comparison_device":"cpu","canonical_comparison_dtype":"float64",'
    b'"cuda_matmul_allow_tf32":false,"cudnn_allow_tf32":false,'
    b'"float32_matmul_precision":"highest","physical_compute_device":"cpu",'
    b'"physical_compute_dtype":"float64","source_device":"cpu",'
    b'"source_dtype":"float32"}'
)
_SUBMISSION_IDENTITY_POLICY_CANONICAL_BYTES = (
    b'{"require_pairwise_distinct":true,"scheduler_identity_field":"array_job_id"}'
)
_SUBMISSION_PLAN_CANONICAL_BYTES = (
    b'[{"array_concurrency":1,"array_spec":"0-119%1","array_task_count":120,'
    b'"array_task_max":119,"array_task_min":0,"array_task_step":1,'
    b'"submission_identity":"primary-0-119","submission_role":"primary"},'
    b'{"array_concurrency":1,"array_spec":"120-120%1","array_task_count":1,'
    b'"array_task_max":120,"array_task_min":120,"array_task_step":1,'
    b'"submission_identity":"replay-1-120","submission_role":"replay-1"},'
    b'{"array_concurrency":1,"array_spec":"121-121%1","array_task_count":1,'
    b'"array_task_max":121,"array_task_min":121,"array_task_step":1,'
    b'"submission_identity":"replay-2-121","submission_role":"replay-2"}]'
)

_SCHEDULER_PLAN_SHA256_LITERAL = "561449486237f4ac7c681909ee98dea2eed81166ce4c7a1bcca6071540215b6f"
_PROBE_WORK_PLAN_SHA256_LITERAL = "11b3dd9863cbd010eb50e95f4f4a5941080eb10186731a34f0625dd9fd5b6586"
SCHEDULER_PLAN_SHA256 = _SCHEDULER_PLAN_SHA256_LITERAL

_MANIFEST_PAYLOAD_FIELDS = (
    "schema_version",
    "manifest_type",
    "calibration_id",
    "matrix_hashes",
    "resource_contract",
    "numerical_policy",
    "fit_recipe_sha256",
    "probe_work_plan_sha256",
    "scheduler_plan_sha256",
    "probe_deployment",
    "expected_fit_deployment",
    "paths",
    "fit_catalog_identity",
    "submission_identity_policy",
    "submission_plan",
)
_MANIFEST_FIELDS = {*_MANIFEST_PAYLOAD_FIELDS, "integrity"}
_PATH_FIELDS = {
    "data_root",
    "fit_stage_root",
    "fit_catalog_path",
    "probe_output_root",
    "data_catalog_path",
}
_FIT_CATALOG_IDENTITY_FIELDS = {
    "raw_sha256",
    "integrity_payload_sha256",
    "cohort_identity_sha256",
}
_SUBMISSION_PLAN_FIELDS = {
    "submission_identity",
    "submission_role",
    "array_spec",
    "array_task_min",
    "array_task_max",
    "array_task_count",
    "array_task_step",
    "array_concurrency",
}
_SUBMISSION_IDENTITY_POLICY_FIELDS = {
    "scheduler_identity_field",
    "require_pairwise_distinct",
}
_INTEGRITY_FIELDS = {"algorithm", "covered_fields", "payload_sha256"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_FORBIDDEN_SCIENCE_FIELDS = {
    "candidate_m",
    "fit_task_index",
    "kernel",
    "m",
    "n_dims",
    "n_particles",
    "repeat_id",
    "replica",
    "seed",
    "task_index",
    "tolerance",
    "train_steps",
    "training_m",
}


class ProbeLaunchManifestError(ValueError):
    """Raised when a probe launch manifest violates the frozen JSON contract."""


def _load_canonical_literal(value: bytes, label: str) -> Any:
    """Return a fresh plain-JSON value from one immutable canonical literal."""

    try:
        parsed = parse_strict_json_bytes(value)
        if canonical_json_bytes(parsed) != value:
            raise RuntimeError(f"{label} is not encoded as canonical JSON")
    except CalibrationContractError as error:
        raise RuntimeError(f"{label} is not a valid strict-JSON literal") from error
    return parsed


def _fresh_json(value: Any, label: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (CalibrationContractError, TypeError, ValueError) as error:
        raise ProbeLaunchManifestError(f"{label} must be finite strict JSON") from error


def _require_object(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProbeLaunchManifestError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ProbeLaunchManifestError(
            f"{label} fields mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return value


def _require_plain_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ProbeLaunchManifestError(f"{label} must be an integer, not bool")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ProbeLaunchManifestError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _strict_match(value: Any, expected: Any, label: str) -> None:
    try:
        matches = canonical_json_bytes(value) == canonical_json_bytes(expected)
    except CalibrationContractError as error:
        raise ProbeLaunchManifestError(f"{label} is not finite strict JSON") from error
    if not matches:
        raise ProbeLaunchManifestError(f"{label} does not match the frozen contract")


def _reject_science_fields(value: Any, label: str = "manifest") -> None:
    if isinstance(value, dict):
        forbidden = set(value) & _FORBIDDEN_SCIENCE_FIELDS
        if forbidden:
            raise ProbeLaunchManifestError(
                f"{label} contains forbidden scientific fields: {sorted(forbidden)}"
            )
        for name, item in value.items():
            _reject_science_fields(item, f"{label}.{name}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_science_fields(item, f"{label}[{index}]")


def _canonical_absolute_path(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value.startswith("//")
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ProbeLaunchManifestError(f"{label} must be a canonical absolute POSIX path")
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or path == PurePosixPath("/")
        or ".." in path.parts
        or value != path.as_posix()
    ):
        raise ProbeLaunchManifestError(f"{label} must be a canonical absolute POSIX path")
    return value


def _is_within(child: PurePosixPath, parent: PurePosixPath) -> bool:
    return child == parent or parent in child.parents


def _validate_paths(value: Any) -> dict[str, str]:
    paths = _require_object(value, _PATH_FIELDS, "paths")
    normalized = {
        name: _canonical_absolute_path(paths[name], f"paths.{name}")
        for name in sorted(_PATH_FIELDS)
    }
    if len(set(normalized.values())) != len(normalized):
        raise ProbeLaunchManifestError("manifest paths must not alias one another")
    pure = {name: PurePosixPath(path) for name, path in normalized.items()}
    for name in ("fit_catalog_path", "data_catalog_path"):
        if pure[name].suffix != ".json":
            raise ProbeLaunchManifestError(f"paths.{name} must name a JSON file")
    if _is_within(pure["fit_catalog_path"], pure["fit_stage_root"]):
        raise ProbeLaunchManifestError("fit catalog must be outside the fit-stage root")
    output = pure["probe_output_root"]
    for name in ("data_root", "fit_stage_root"):
        source = pure[name]
        if _is_within(output, source) or _is_within(source, output):
            raise ProbeLaunchManifestError(f"probe output root must not overlap paths.{name}")
    for name in ("fit_catalog_path", "data_catalog_path"):
        if _is_within(pure[name], output):
            raise ProbeLaunchManifestError(f"paths.{name} must not be inside the probe output root")
    return normalized


def _validate_fit_catalog_identity(value: Any) -> dict[str, str]:
    identity = _require_object(
        value,
        _FIT_CATALOG_IDENTITY_FIELDS,
        "fit_catalog_identity",
    )
    return {
        name: _require_sha256(identity[name], f"fit_catalog_identity.{name}")
        for name in sorted(_FIT_CATALOG_IDENTITY_FIELDS)
    }


def _normalize_submission_plan(
    value: Any,
    *,
    expected_count: int,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != expected_count:
        raise ProbeLaunchManifestError("submission_plan must contain exactly three entries")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        entry = _require_object(
            raw,
            _SUBMISSION_PLAN_FIELDS,
            f"submission_plan[{index}]",
        )
        for field in (
            "array_task_min",
            "array_task_max",
            "array_task_count",
            "array_task_step",
            "array_concurrency",
        ):
            _require_plain_int(entry[field], f"submission_plan[{index}].{field}")
        normalized.append(_fresh_json(entry, f"submission_plan[{index}]"))
    identities = [entry["submission_identity"] for entry in normalized]
    if len(identities) != len(set(identities)):
        raise ProbeLaunchManifestError("submission identities must be pairwise distinct")
    return normalized


def _validate_scheduler_topology(plan: list[dict[str, Any]]) -> None:
    covered: list[int] = []
    for entry in plan:
        minimum = entry["array_task_min"]
        maximum = entry["array_task_max"]
        step = entry["array_task_step"]
        count = entry["array_task_count"]
        concurrency = entry["array_concurrency"]
        if step != 1 or minimum < 0 or maximum < minimum or concurrency != 1:
            raise ProbeLaunchManifestError("submission_plan contains an invalid array range")
        expected = list(range(minimum, maximum + 1, step))
        if count != len(expected):
            raise ProbeLaunchManifestError("submission_plan count does not match its range")
        if entry["array_spec"] != f"{minimum}-{maximum}%{concurrency}":
            raise ProbeLaunchManifestError("submission_plan array spec is inconsistent")
        covered.extend(expected)
    if covered != list(range(PROBE_TASK_COUNT)) or len(covered) != len(set(covered)):
        raise ProbeLaunchManifestError("submission_plan must cover each probe task exactly once")


def _load_scheduler_contract() -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Load and re-hash the authoritative scheduler-plan domain."""

    try:
        policy = _require_object(
            _load_canonical_literal(
                _SUBMISSION_IDENTITY_POLICY_CANONICAL_BYTES,
                "submission identity policy literal",
            ),
            _SUBMISSION_IDENTITY_POLICY_FIELDS,
            "submission_identity_policy literal",
        )
        if policy != {
            "scheduler_identity_field": "array_job_id",
            "require_pairwise_distinct": True,
        }:
            raise ProbeLaunchManifestError("submission identity policy literal is invalid")
        raw_plan = _load_canonical_literal(
            _SUBMISSION_PLAN_CANONICAL_BYTES,
            "submission plan literal",
        )
        plan = _normalize_submission_plan(raw_plan, expected_count=3)
        _validate_scheduler_topology(plan)
    except ProbeLaunchManifestError as error:
        raise RuntimeError("frozen scheduler-plan literal is invalid") from error
    domain = {
        "submission_identity_policy": policy,
        "submission_plan": plan,
    }
    observed = canonical_sha256(domain)
    if observed != _SCHEDULER_PLAN_SHA256_LITERAL:
        raise RuntimeError("frozen scheduler-plan domain SHA-256 is inconsistent")
    return policy, plan, observed


def _load_envelope_contracts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    matrix_hashes = _load_canonical_literal(
        _MATRIX_HASHES_CANONICAL_BYTES,
        "matrix-hashes literal",
    )
    resource_contract = _load_canonical_literal(
        _RESOURCE_CONTRACT_CANONICAL_BYTES,
        "resource-contract literal",
    )
    numerical_policy = _load_canonical_literal(
        _NUMERICAL_POLICY_CANONICAL_BYTES,
        "numerical-policy literal",
    )
    if not all(
        type(value) is dict for value in (matrix_hashes, resource_contract, numerical_policy)
    ):
        raise RuntimeError("frozen envelope-contract literals must be JSON objects")
    return matrix_hashes, resource_contract, numerical_policy


def _recompute_probe_work_plan_sha256() -> str:
    """Re-hash the current domain-separated core plan against its literal."""

    try:
        observed = canonical_sha256(canonical_probe_work_plan_payload())
    except (CalibrationContractError, TypeError, ValueError) as error:
        raise RuntimeError("probe work-plan payload is not canonical strict JSON") from error
    if observed != _PROBE_WORK_PLAN_SHA256_LITERAL:
        raise RuntimeError("probe work-plan payload SHA-256 is inconsistent")
    return observed


def _validate_cross_deployment_identity(deployments: Mapping[str, Mapping[str, Any]]) -> None:
    probe = deployments["probe_deployment"]
    expected_fit = deployments["expected_fit_deployment"]
    for field in (
        "tera_gitlink",
        "catalog_generation_commit",
        "catalog_generation_tree",
    ):
        if probe[field] != expected_fit[field]:
            raise ProbeLaunchManifestError(f"probe and expected-fit deployments must share {field}")


def _manifest_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    return {field: document[field] for field in _MANIFEST_PAYLOAD_FIELDS}


def validate_probe_launch_manifest(value: Any) -> dict[str, Any]:
    """Strictly validate and detach a probe launch manifest JSON value."""

    matrix_hashes, resource_contract, numerical_policy = _load_envelope_contracts()
    expected_policy, expected_plan, scheduler_plan_sha256 = _load_scheduler_contract()
    probe_work_plan_sha256 = _recompute_probe_work_plan_sha256()
    manifest = _fresh_json(value, "probe launch manifest")
    _reject_science_fields(manifest)
    document = _require_object(manifest, _MANIFEST_FIELDS, "probe launch manifest")
    if document["schema_version"] != PROBE_LAUNCH_MANIFEST_SCHEMA_VERSION:
        raise ProbeLaunchManifestError("probe launch manifest schema version is unsupported")
    if document["manifest_type"] != PROBE_LAUNCH_MANIFEST_TYPE:
        raise ProbeLaunchManifestError("probe launch manifest type is unsupported")
    if document["calibration_id"] != CALIBRATION_ID:
        raise ProbeLaunchManifestError("probe launch manifest calibration ID is mismatched")
    _strict_match(document["matrix_hashes"], matrix_hashes, "matrix_hashes")
    _strict_match(document["resource_contract"], resource_contract, "resource_contract")
    _strict_match(document["numerical_policy"], numerical_policy, "numerical_policy")
    if document["fit_recipe_sha256"] != FIT_RECIPE_SHA256:
        raise ProbeLaunchManifestError("fit recipe SHA-256 is mismatched")
    if document["probe_work_plan_sha256"] != probe_work_plan_sha256:
        raise ProbeLaunchManifestError("probe work-plan SHA-256 is mismatched")
    if document["scheduler_plan_sha256"] != scheduler_plan_sha256:
        raise ProbeLaunchManifestError("scheduler plan SHA-256 is mismatched")
    deployments: dict[str, dict[str, Any]] = {}
    for field in ("probe_deployment", "expected_fit_deployment"):
        try:
            deployments[field] = validate_expected_deployment(document[field])
        except CalibrationFitAggregateError as error:
            raise ProbeLaunchManifestError(f"{field} is invalid") from error
    _validate_cross_deployment_identity(deployments)
    paths = _validate_paths(document["paths"])
    fit_catalog_identity = _validate_fit_catalog_identity(document["fit_catalog_identity"])
    policy = _require_object(
        document["submission_identity_policy"],
        _SUBMISSION_IDENTITY_POLICY_FIELDS,
        "submission_identity_policy",
    )
    _strict_match(policy, expected_policy, "submission_identity_policy")
    plan = _normalize_submission_plan(
        document["submission_plan"],
        expected_count=len(expected_plan),
    )
    _validate_scheduler_topology(plan)
    _strict_match(plan, expected_plan, "submission_plan")
    integrity = _require_object(document["integrity"], _INTEGRITY_FIELDS, "integrity")
    if integrity["algorithm"] != "sha256":
        raise ProbeLaunchManifestError("manifest integrity algorithm must be sha256")
    if integrity["covered_fields"] != list(_MANIFEST_PAYLOAD_FIELDS):
        raise ProbeLaunchManifestError("manifest integrity covered_fields are mismatched")
    digest = _require_sha256(integrity["payload_sha256"], "integrity.payload_sha256")
    if digest != canonical_sha256(_manifest_payload(document)):
        raise ProbeLaunchManifestError("manifest integrity SHA-256 is mismatched")
    normalized = {
        **document,
        **deployments,
        "matrix_hashes": matrix_hashes,
        "resource_contract": resource_contract,
        "numerical_policy": numerical_policy,
        "paths": paths,
        "fit_catalog_identity": fit_catalog_identity,
        "submission_identity_policy": expected_policy,
        "submission_plan": expected_plan,
    }
    return _fresh_json(normalized, "validated probe launch manifest")


def build_probe_launch_manifest(
    *,
    probe_deployment: Mapping[str, Any],
    expected_fit_deployment: Mapping[str, Any],
    data_root: str,
    fit_stage_root: str,
    fit_catalog_path: str,
    probe_output_root: str,
    data_catalog_path: str,
    fit_catalog_raw_sha256: str,
    fit_catalog_integrity_payload_sha256: str,
    fit_catalog_cohort_identity_sha256: str,
) -> dict[str, Any]:
    """Build the unique canonical manifest for one frozen probe launch cohort."""

    matrix_hashes, resource_contract, numerical_policy = _load_envelope_contracts()
    submission_identity_policy, submission_plan, scheduler_plan_sha256 = _load_scheduler_contract()
    probe_work_plan_sha256 = _recompute_probe_work_plan_sha256()
    deployments: dict[str, dict[str, Any]] = {}
    for field, value in (
        ("probe_deployment", probe_deployment),
        ("expected_fit_deployment", expected_fit_deployment),
    ):
        try:
            deployments[field] = validate_expected_deployment(value)
        except CalibrationFitAggregateError as error:
            raise ProbeLaunchManifestError(f"{field} is invalid") from error
    _validate_cross_deployment_identity(deployments)
    document: dict[str, Any] = {
        "schema_version": PROBE_LAUNCH_MANIFEST_SCHEMA_VERSION,
        "manifest_type": PROBE_LAUNCH_MANIFEST_TYPE,
        "calibration_id": CALIBRATION_ID,
        "matrix_hashes": matrix_hashes,
        "resource_contract": resource_contract,
        "numerical_policy": numerical_policy,
        "fit_recipe_sha256": FIT_RECIPE_SHA256,
        "probe_work_plan_sha256": probe_work_plan_sha256,
        "scheduler_plan_sha256": scheduler_plan_sha256,
        **deployments,
        "paths": {
            "data_root": data_root,
            "fit_stage_root": fit_stage_root,
            "fit_catalog_path": fit_catalog_path,
            "probe_output_root": probe_output_root,
            "data_catalog_path": data_catalog_path,
        },
        "fit_catalog_identity": {
            "raw_sha256": fit_catalog_raw_sha256,
            "integrity_payload_sha256": fit_catalog_integrity_payload_sha256,
            "cohort_identity_sha256": fit_catalog_cohort_identity_sha256,
        },
        "submission_identity_policy": submission_identity_policy,
        "submission_plan": submission_plan,
    }
    document["integrity"] = {
        "algorithm": "sha256",
        "covered_fields": list(_MANIFEST_PAYLOAD_FIELDS),
        "payload_sha256": canonical_sha256(_manifest_payload(document)),
    }
    return validate_probe_launch_manifest(document)


def build_probe_launch_manifest_bytes(**kwargs: Any) -> bytes:
    """Build and encode a canonical manifest without a presentation newline."""

    return canonical_json_bytes(build_probe_launch_manifest(**kwargs))


def parse_probe_launch_manifest_bytes(value: bytes) -> dict[str, Any]:
    """Parse canonical raw bytes with duplicate-key and nonfinite rejection."""

    try:
        parsed = parse_strict_json_bytes(value)
    except CalibrationContractError as error:
        raise ProbeLaunchManifestError("probe launch manifest bytes are not strict JSON") from error
    manifest = validate_probe_launch_manifest(parsed)
    if value != canonical_json_bytes(manifest):
        raise ProbeLaunchManifestError("probe launch manifest bytes are not canonical JSON")
    return manifest


def _build_public_scheduler_views() -> tuple[
    Mapping[str, Any],
    tuple[Mapping[str, Any], ...],
]:
    """Build read-only presentation views, never used as validation authority."""

    policy, plan, _digest = _load_scheduler_contract()
    return MappingProxyType(policy), tuple(MappingProxyType(entry) for entry in plan)


SUBMISSION_IDENTITY_POLICY, SUBMISSION_PLAN = _build_public_scheduler_views()


__all__ = [
    "PROBE_LAUNCH_MANIFEST_SCHEMA_VERSION",
    "PROBE_LAUNCH_MANIFEST_TYPE",
    "SCHEDULER_PLAN_SHA256",
    "SUBMISSION_IDENTITY_POLICY",
    "SUBMISSION_PLAN",
    "ProbeLaunchManifestError",
    "build_probe_launch_manifest",
    "build_probe_launch_manifest_bytes",
    "parse_probe_launch_manifest_bytes",
    "validate_probe_launch_manifest",
]
