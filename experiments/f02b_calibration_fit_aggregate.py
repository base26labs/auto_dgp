"""Strict 45/45 aggregation of development-only F02b fit artifacts.

The aggregator accepts only the 45 identity-derived task directories emitted by
``f02b_calibration_fit``.  It opens the run root and every task directory with
``O_NOFOLLOW``, reads each regular file once, hashes those same bytes, rejects
noncanonical JSON, and validates the public payload/envelope pair.  A failure
still produces a complete, integrity-protected 45-slot catalog, but such a
catalog is never analysis-ready and is never freeze-ready.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from cluster.f02b_calibration_grid import (
    CALIBRATION_ID,
    FIT_TASK_COUNT,
    FIT_TASKS,
)
from experiments.f02_design import TRAIN_TIME_INDICES
from experiments.f02b_calibration_contract import (
    CATALOG_IDENTITY,
    F02_CATALOG_SHA256,
    FIT_RECIPE,
    FIT_RECIPE_SHA256,
    MATRIX_HASHES,
    CalibrationContractError,
    canonical_json_bytes,
    canonical_sha256,
    parse_strict_json_bytes,
    validate_fit_execution_envelope,
    verify_numeric_payload_bytes,
)
from experiments.f02b_calibration_fit import (
    FIT_PAYLOAD_SCHEMA_VERSION,
    FIT_PAYLOAD_TYPE,
    CalibrationFitError,
    fit_artifact_paths,
    validate_fit_numeric_payload,
)

FIT_CATALOG_SCHEMA_VERSION = "f02b_calibration_fit_catalog_v1"
FIT_CATALOG_TYPE = "f02b_development_fit_stage_catalog"

_HEX40_LENGTH = 40
_SHA256_LENGTH = 64
_PAYLOAD_KEYS = {
    "artifact_type",
    "capabilities",
    "data_access",
    "fit_recipe",
    "fit_recipe_sha256",
    "input_identity",
    "parameters",
    "provenance",
    "schema_version",
    "task_index",
    "task_record",
    "task_role",
}
_DATA_ACCESS_KEYS = {
    "fit_callable_splits",
    "integrity_loading",
    "prediction_or_scoring_performed",
    "tensorized_splits",
    "training_time_indices",
}
_EXPECTED_DATA_ACCESS = {
    "integrity_loading": "authoritative_complete_bundle",
    "tensorized_splits": ["train"],
    "fit_callable_splits": ["train"],
    "training_time_indices": list(TRAIN_TIME_INDICES),
    "prediction_or_scoring_performed": False,
}
_INPUT_IDENTITY_KEYS = {"bundle", "catalog"}
_INPUT_CATALOG_KEYS = {
    "generation_git_commit",
    "generation_git_tree",
    "path",
    "schema_version",
    "sha256",
}
_BUNDLE_KEYS = {
    "D",
    "catalog_bundle_task_index",
    "dataset_content_sha256",
    "dataset_path",
    "file_sha256",
    "generator_config",
    "manifest_path",
    "metadata_path",
    "n_dims",
    "n_particles",
    "phase",
    "replica",
    "sha256_manifest_file_sha256",
}
_PARAMETER_KEYS = {
    "gradient_noise_model",
    "kernel",
    "lengthscale",
    "outputscale",
    "sigma_f",
    "sigma_g",
}
_PROVENANCE_KEYS = {
    "dependencies",
    "runtime",
    "runtime_numerics",
    "scheduler",
    "source",
}
_SOURCE_KEYS = {"commit", "repo_root", "status_porcelain", "tera_gitlink", "tree"}
_DEPENDENCY_KEYS = {"pyproject.toml", "uv.lock"}
_RUNTIME_KEYS = {"packages", "platform", "python_executable", "python_version"}
_SCHEDULER_KEYS = {
    "array_job_id",
    "array_task_id",
    "exclusive_verification_mode",
    "job_id",
    "node_list",
}
_EXPECTED_DEPLOYMENT_KEYS = {
    "catalog_generation_commit",
    "catalog_generation_tree",
    "catalog_sha256",
    "pyproject_sha256",
    "source_commit",
    "source_tree",
    "tera_gitlink",
    "uv_lock_sha256",
}
_TASK_RESULT_KEYS = {
    "artifact_directory",
    "envelope_path",
    "envelope_raw_sha256",
    "envelope_record_sha256",
    "errors",
    "payload_canonical_sha256",
    "payload_path",
    "payload_raw_sha256",
    "status",
    "task_index",
    "task_record",
}
_HASH_RESULT_FIELDS = (
    "payload_raw_sha256",
    "payload_canonical_sha256",
    "envelope_raw_sha256",
    "envelope_record_sha256",
)
_FAILURE_KEYS = {"code", "message", "path"}
_LABEL_POLICY = {
    "scope": "development_only",
    "validation_labels_exposed": False,
    "test_labels_exposed": False,
    "prediction_or_scoring_in_catalog": False,
}
_CATALOG_PAYLOAD_FIELDS = (
    "schema_version",
    "catalog_type",
    "calibration_id",
    "stage",
    "matrix_hashes",
    "catalog_identity",
    "fit_recipe_sha256",
    "expected_deployment",
    "expected_task_count",
    "valid_task_count",
    "invalid_task_count",
    "unexpected_path_count",
    "structural_failure_count",
    "cohort_identity_sha256",
    "tasks",
    "structural_failures",
    "label_policy",
    "analysis_ready",
    "freeze_ready",
)
_CATALOG_KEYS = {*_CATALOG_PAYLOAD_FIELDS, "integrity"}
_CATALOG_INTEGRITY_KEYS = {"algorithm", "covered_fields", "payload_sha256"}
_OPEN_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_OPEN_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


class CalibrationFitAggregateError(RuntimeError):
    """Raised for invalid aggregation configuration or catalog publication."""


@dataclass(frozen=True, slots=True)
class ExpectedFitStagePaths:
    """The sole directory and file paths authorized for one fit task."""

    task_index: int
    directory: Path
    payload_path: Path
    envelope_path: Path

    @property
    def payload_relative_path(self) -> str:
        return f"{self.directory.name}/{self.payload_path.name}"

    @property
    def envelope_relative_path(self) -> str:
        return f"{self.directory.name}/{self.envelope_path.name}"


@dataclass(frozen=True, slots=True)
class _RawFile:
    data: bytes
    sha256: str
    identity: tuple[int, int]


def expected_fit_stage_paths(input_root: str | Path) -> tuple[ExpectedFitStagePaths, ...]:
    """Enumerate the exact 45 runner-derived path pairs without globbing."""

    records: list[ExpectedFitStagePaths] = []
    for task in FIT_TASKS:
        paths = fit_artifact_paths(input_root, task.task_index)
        records.append(
            ExpectedFitStagePaths(
                task_index=task.task_index,
                directory=paths.directory,
                payload_path=paths.numeric_payload,
                envelope_path=paths.execution_envelope,
            )
        )
    relative_paths = [
        relative
        for record in records
        for relative in (
            record.directory.name,
            record.payload_relative_path,
            record.envelope_relative_path,
        )
    ]
    if len(records) != FIT_TASK_COUNT:
        raise CalibrationFitAggregateError("fit-stage path matrix is not exactly 45 tasks")
    if len(relative_paths) != len(set(relative_paths)):
        raise CalibrationFitAggregateError("fit-stage path matrix contains duplicate paths")
    return tuple(records)


def _require_object(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CalibrationFitAggregateError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise CalibrationFitAggregateError(
            f"{label} fields mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return value


def _require_plain_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise CalibrationFitAggregateError(f"{label} must be an integer, not bool")
    return value


def _require_digest(value: Any, length: int, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        kind = "git object ID" if length == _HEX40_LENGTH else "SHA-256"
        raise CalibrationFitAggregateError(f"{label} must be a lowercase {kind}")
    return value


def _require_sha256_or_none(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _require_digest(value, _SHA256_LENGTH, label)


def _strict_match(value: Any, expected: Any, label: str) -> None:
    try:
        matched = canonical_json_bytes(value) == canonical_json_bytes(expected)
    except CalibrationContractError as error:
        raise CalibrationFitAggregateError(f"{label} is not finite strict JSON") from error
    if not matched:
        raise CalibrationFitAggregateError(f"{label} does not match the frozen identity")


def _require_canonical_absolute_path(value: Any, label: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise CalibrationFitAggregateError(f"{label} must be a canonical absolute POSIX path")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise CalibrationFitAggregateError(f"{label} must be a canonical absolute POSIX path")
    return value


def validate_expected_deployment(value: Any) -> dict[str, Any]:
    """Validate the infrastructure-supplied deployed code/catalog identity."""

    expected = _require_object(value, _EXPECTED_DEPLOYMENT_KEYS, "expected deployment")
    for field in (
        "source_commit",
        "source_tree",
        "tera_gitlink",
        "catalog_generation_commit",
        "catalog_generation_tree",
    ):
        _require_digest(expected[field], _HEX40_LENGTH, f"expected deployment {field}")
    for field in ("pyproject_sha256", "uv_lock_sha256", "catalog_sha256"):
        _require_digest(expected[field], _SHA256_LENGTH, f"expected deployment {field}")
    if expected["catalog_sha256"] != F02_CATALOG_SHA256:
        raise CalibrationFitAggregateError("expected deployment catalog SHA-256 is not frozen")
    return json.loads(canonical_json_bytes(expected))


def _failure(code: str, path: str, message: str) -> dict[str, str]:
    if not all(type(item) is str and item for item in (code, path, message)):
        raise CalibrationFitAggregateError("failure records require nonempty text")
    return {"code": code, "path": path, "message": message}


def _empty_task_result(paths: ExpectedFitStagePaths) -> dict[str, Any]:
    return {
        "task_index": paths.task_index,
        "task_record": FIT_TASKS[paths.task_index].as_record(),
        "artifact_directory": paths.directory.name,
        "payload_path": paths.payload_relative_path,
        "envelope_path": paths.envelope_relative_path,
        "status": "invalid",
        "payload_raw_sha256": None,
        "payload_canonical_sha256": None,
        "envelope_raw_sha256": None,
        "envelope_record_sha256": None,
        "errors": [],
    }


def _add_task_error(
    result: dict[str, Any],
    code: str,
    path: str,
    message: str,
) -> None:
    failure = _failure(code, path, message)
    if failure not in result["errors"]:
        result["errors"].append(failure)
    result["status"] = "invalid"


def _scandir_names(directory_fd: int, label: str) -> set[str]:
    try:
        with os.scandir(directory_fd) as entries:
            return {entry.name for entry in entries}
    except OSError as error:
        raise CalibrationFitAggregateError(f"cannot enumerate {label}") from error


def _same_stat(left: os.stat_result, right: os.stat_result) -> bool:
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    return all(getattr(left, field) == getattr(right, field) for field in fields)


def _read_regular_file_once(directory_fd: int, name: str, display_path: str) -> _RawFile:
    """Read one no-follow regular file and hash exactly the bytes parsed later."""

    try:
        descriptor = os.open(name, _OPEN_FILE_FLAGS, dir_fd=directory_fd)
    except FileNotFoundError as error:
        raise CalibrationFitAggregateError("required file is missing") from error
    except OSError as error:
        raise CalibrationFitAggregateError(
            "required path cannot be opened without following"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CalibrationFitAggregateError("required path is not a regular file")
        if before.st_nlink != 1:
            raise CalibrationFitAggregateError("required file must have exactly one hard link")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_state = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        raise CalibrationFitAggregateError("required file changed while reading") from error
    if not _same_stat(before, after) or not _same_stat(after, path_state):
        raise CalibrationFitAggregateError("required file changed while reading")
    data = b"".join(chunks)
    if len(data) != before.st_size:
        raise CalibrationFitAggregateError("required file size changed while reading")
    if not display_path:
        raise CalibrationFitAggregateError("required file path identity is empty")
    return _RawFile(
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        identity=(before.st_dev, before.st_ino),
    )


def _validate_explicit_payload_shape(value: Any, task_index: int) -> dict[str, Any]:
    """Check the runner's complete public wire shape before semantic validation."""

    payload = _require_object(value, _PAYLOAD_KEYS, "fit numeric payload")
    if payload["schema_version"] != FIT_PAYLOAD_SCHEMA_VERSION:
        raise CalibrationFitAggregateError("fit payload schema version is unsupported")
    if payload["artifact_type"] != FIT_PAYLOAD_TYPE or payload["task_role"] != "fit":
        raise CalibrationFitAggregateError("fit payload role/type is mismatched")
    if payload["task_index"] != task_index:
        raise CalibrationFitAggregateError("fit payload is in the wrong task slot")
    _strict_match(payload["task_record"], FIT_TASKS[task_index].as_record(), "fit task record")
    recipe = _require_object(payload["fit_recipe"], set(FIT_RECIPE), "fit recipe")
    _strict_match(recipe, FIT_RECIPE, "fit recipe")
    if payload["fit_recipe_sha256"] != FIT_RECIPE_SHA256:
        raise CalibrationFitAggregateError("fit recipe SHA-256 is mismatched")
    access = _require_object(payload["data_access"], _DATA_ACCESS_KEYS, "fit data access")
    _strict_match(access, _EXPECTED_DATA_ACCESS, "fit data access")
    identity = _require_object(payload["input_identity"], _INPUT_IDENTITY_KEYS, "input identity")
    catalog = _require_object(identity["catalog"], _INPUT_CATALOG_KEYS, "input catalog")
    if catalog["sha256"] != F02_CATALOG_SHA256:
        raise CalibrationFitAggregateError("fit payload catalog SHA-256 is mismatched")
    _require_canonical_absolute_path(catalog["path"], "input catalog path")
    bundle = _require_object(identity["bundle"], _BUNDLE_KEYS, "input bundle")
    for field in ("dataset_path", "metadata_path", "manifest_path"):
        _require_canonical_absolute_path(bundle[field], f"input bundle {field}")
    if not isinstance(bundle["file_sha256"], dict):
        raise CalibrationFitAggregateError("input bundle file hashes must be an object")
    _require_object(payload["parameters"], _PARAMETER_KEYS, "fit parameters")
    provenance = _require_object(payload["provenance"], _PROVENANCE_KEYS, "fit provenance")
    source = _require_object(provenance["source"], _SOURCE_KEYS, "fit source provenance")
    _require_canonical_absolute_path(source["repo_root"], "fit source repository path")
    dependencies = _require_object(provenance["dependencies"], _DEPENDENCY_KEYS, "fit dependencies")
    for name, record in dependencies.items():
        _require_object(record, {"sha256"}, f"fit dependency {name}")
    runtime = _require_object(provenance["runtime"], _RUNTIME_KEYS, "fit runtime")
    if not isinstance(runtime["packages"], list):
        raise CalibrationFitAggregateError("fit runtime packages must be an array")
    for position, package in enumerate(runtime["packages"]):
        _require_object(package, {"name", "version"}, f"fit runtime package {position}")
    _require_object(provenance["scheduler"], _SCHEDULER_KEYS, "fit scheduler")
    if payload["capabilities"] != ["fit_parameters_for_f02b_numerical_calibration"]:
        raise CalibrationFitAggregateError("fit payload capabilities are mismatched")
    return payload


def _parse_canonical_json(raw: _RawFile, label: str) -> Any:
    try:
        value = parse_strict_json_bytes(raw.data)
        canonical = canonical_json_bytes(value)
    except CalibrationContractError as error:
        raise CalibrationFitAggregateError(f"{label} is not strict finite JSON") from error
    if raw.data != canonical:
        raise CalibrationFitAggregateError(f"{label} bytes are not canonical JSON")
    return value


def _validate_payload_bytes(raw: _RawFile, task_index: int) -> dict[str, Any]:
    parsed = _parse_canonical_json(raw, "fit numeric payload")
    _validate_explicit_payload_shape(parsed, task_index)
    try:
        payload = validate_fit_numeric_payload(parsed)
    except CalibrationFitError as error:
        raise CalibrationFitAggregateError("fit payload fails runner validation") from error
    if payload["task_index"] != task_index:
        raise CalibrationFitAggregateError("fit payload task index changed during validation")
    return payload


def _validate_envelope_bytes(
    raw: _RawFile,
    payload_raw: _RawFile,
    expected: ExpectedFitStagePaths,
) -> dict[str, Any]:
    parsed = _parse_canonical_json(raw, "fit execution envelope")
    try:
        envelope = validate_fit_execution_envelope(
            parsed,
            expected_task_index=expected.task_index,
        )
        binding = verify_numeric_payload_bytes(
            envelope["canonical_record"]["payload_binding"],
            payload_raw.data,
            expected_task_role="fit",
            expected_task_index=expected.task_index,
        )
    except CalibrationContractError as error:
        raise CalibrationFitAggregateError(
            "fit envelope or raw payload binding is invalid"
        ) from error
    if binding["numeric_payload_path"] != expected.payload_relative_path:
        raise CalibrationFitAggregateError("fit envelope payload path is noncanonical")
    return envelope


def _deployment_matches(payload: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    source = payload["provenance"]["source"]
    dependencies = payload["provenance"]["dependencies"]
    catalog = payload["input_identity"]["catalog"]
    observed = {
        "source_commit": source["commit"],
        "source_tree": source["tree"],
        "tera_gitlink": source["tera_gitlink"],
        "pyproject_sha256": dependencies["pyproject.toml"]["sha256"],
        "uv_lock_sha256": dependencies["uv.lock"]["sha256"],
        "catalog_generation_commit": catalog["generation_git_commit"],
        "catalog_generation_tree": catalog["generation_git_tree"],
        "catalog_sha256": catalog["sha256"],
    }
    return canonical_json_bytes(observed) == canonical_json_bytes(expected)


def _cohort_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    provenance = payload["provenance"]
    return {
        "source": provenance["source"],
        "dependencies": provenance["dependencies"],
        "runtime": provenance["runtime"],
        "runtime_numerics": provenance["runtime_numerics"],
        "catalog": payload["input_identity"]["catalog"],
        "fit_recipe": payload["fit_recipe"],
        "fit_recipe_sha256": payload["fit_recipe_sha256"],
        "data_access": payload["data_access"],
        "capabilities": payload["capabilities"],
    }


def _audit_scheduler_cohort(
    payloads: Mapping[int, Mapping[str, Any]],
    results: list[dict[str, Any]],
) -> None:
    """Require one array and unique per-task scheduler evidence."""

    array_job_groups: defaultdict[str, list[int]] = defaultdict(list)
    array_task_groups: defaultdict[int, list[int]] = defaultdict(list)
    job_groups: defaultdict[str, list[int]] = defaultdict(list)
    for task_index in sorted(payloads):
        scheduler = payloads[task_index]["provenance"]["scheduler"]
        array_job_groups[scheduler["array_job_id"]].append(task_index)
        array_task_groups[scheduler["array_task_id"]].append(task_index)
        job_groups[scheduler["job_id"]].append(task_index)

    if len(array_job_groups) > 1:
        task_groups = [sorted(task_indices) for _, task_indices in sorted(array_job_groups.items())]
        for task_index in sorted(payloads):
            _add_task_error(
                results[task_index],
                "scheduler_array_job_mismatch",
                results[task_index]["payload_path"],
                f"fit payloads come from mixed array-job task groups {task_groups}",
            )
    for array_task_id, task_indices in sorted(array_task_groups.items()):
        if len(task_indices) > 1:
            owners = sorted(task_indices)
            for task_index in owners:
                _add_task_error(
                    results[task_index],
                    "duplicate_scheduler_array_task_id",
                    results[task_index]["payload_path"],
                    f"scheduler array_task_id {array_task_id} is reused by tasks {owners}",
                )
    for job_id, task_indices in sorted(job_groups.items()):
        if len(task_indices) > 1:
            owners = sorted(task_indices)
            for task_index in owners:
                _add_task_error(
                    results[task_index],
                    "duplicate_scheduler_job_id",
                    results[task_index]["payload_path"],
                    f"scheduler job_id {job_id} is reused by tasks {owners}",
                )


def _catalog_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    return {field: document[field] for field in _CATALOG_PAYLOAD_FIELDS}


def _build_catalog(
    task_results: list[dict[str, Any]],
    structural_failures: list[dict[str, str]],
    expected_deployment: Mapping[str, Any],
    *,
    cohort_identity_sha256: str | None,
) -> dict[str, Any]:
    for result in task_results:
        result["status"] = "invalid" if result["errors"] else "valid"
    valid_count = sum(result["status"] == "valid" for result in task_results)
    task_failure_count = sum(len(result["errors"]) for result in task_results)
    analysis_ready = (
        valid_count == FIT_TASK_COUNT
        and not structural_failures
        and task_failure_count == 0
        and cohort_identity_sha256 is not None
    )
    document: dict[str, Any] = {
        "schema_version": FIT_CATALOG_SCHEMA_VERSION,
        "catalog_type": FIT_CATALOG_TYPE,
        "calibration_id": CALIBRATION_ID,
        "stage": "fit",
        "matrix_hashes": json.loads(canonical_json_bytes(MATRIX_HASHES)),
        "catalog_identity": json.loads(canonical_json_bytes(CATALOG_IDENTITY)),
        "fit_recipe_sha256": FIT_RECIPE_SHA256,
        "expected_deployment": json.loads(canonical_json_bytes(expected_deployment)),
        "expected_task_count": FIT_TASK_COUNT,
        "valid_task_count": valid_count,
        "invalid_task_count": FIT_TASK_COUNT - valid_count,
        "unexpected_path_count": sum(
            failure["code"] == "unexpected_path" for failure in structural_failures
        ),
        "structural_failure_count": len(structural_failures) + task_failure_count,
        "cohort_identity_sha256": cohort_identity_sha256,
        "tasks": task_results,
        "structural_failures": structural_failures,
        "label_policy": dict(_LABEL_POLICY),
        "analysis_ready": analysis_ready,
        "freeze_ready": False,
    }
    document["integrity"] = {
        "algorithm": "sha256",
        "covered_fields": list(_CATALOG_PAYLOAD_FIELDS),
        "payload_sha256": canonical_sha256(_catalog_payload(document)),
    }
    return validate_fit_catalog(document)


def _missing_root_catalog(
    paths: tuple[ExpectedFitStagePaths, ...],
    expected_deployment: Mapping[str, Any],
    failure: dict[str, str],
) -> dict[str, Any]:
    results = [_empty_task_result(path) for path in paths]
    for result in results:
        _add_task_error(
            result, "missing_task_directory", result["artifact_directory"], failure["message"]
        )
    return _build_catalog(
        results,
        [failure],
        expected_deployment,
        cohort_identity_sha256=None,
    )


def aggregate_fit_stage(
    input_root: str | Path,
    *,
    expected_deployment: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate exactly 45 artifacts under one no-follow run-root descriptor."""

    deployment = validate_expected_deployment(expected_deployment)
    root_path = Path(input_root)
    paths = expected_fit_stage_paths(root_path)
    try:
        root_fd = os.open(root_path, _OPEN_DIRECTORY_FLAGS)
    except FileNotFoundError:
        return _missing_root_catalog(
            paths,
            deployment,
            _failure("missing_input_root", str(root_path), "fit-stage input root is missing"),
        )
    except OSError:
        return _missing_root_catalog(
            paths,
            deployment,
            _failure(
                "invalid_input_root",
                str(root_path),
                "fit-stage input root is not a no-follow regular directory",
            ),
        )

    results = [_empty_task_result(path) for path in paths]
    structural_failures: list[dict[str, str]] = []
    payloads: dict[int, dict[str, Any]] = {}
    file_identities: defaultdict[tuple[int, int], list[tuple[int, str]]] = defaultdict(list)
    payload_hash_tasks: defaultdict[str, list[int]] = defaultdict(list)
    envelope_hash_tasks: defaultdict[str, list[int]] = defaultdict(list)
    try:
        root_before = os.fstat(root_fd)
        root_names_before = _scandir_names(root_fd, "fit-stage input root")
        expected_directories = {path.directory.name for path in paths}
        for name in sorted(root_names_before - expected_directories):
            structural_failures.append(
                _failure("unexpected_path", name, "input root contains an unregistered entry")
            )

        for expected, result in zip(paths, results, strict=True):
            directory_name = expected.directory.name
            try:
                directory_fd = os.open(directory_name, _OPEN_DIRECTORY_FLAGS, dir_fd=root_fd)
            except FileNotFoundError:
                _add_task_error(
                    result,
                    "missing_task_directory",
                    directory_name,
                    "required task directory is missing",
                )
                continue
            except OSError:
                _add_task_error(
                    result,
                    "invalid_task_directory",
                    directory_name,
                    "required task path is not a no-follow regular directory",
                )
                continue
            try:
                directory_before = os.fstat(directory_fd)
                names_before = _scandir_names(directory_fd, directory_name)
                expected_names = {expected.payload_path.name, expected.envelope_path.name}
                for name in sorted(names_before - expected_names):
                    relative = f"{directory_name}/{name}"
                    structural_failures.append(
                        _failure(
                            "unexpected_path",
                            relative,
                            "task directory contains an unregistered entry",
                        )
                    )
                    _add_task_error(
                        result,
                        "unexpected_path",
                        relative,
                        "task directory contains an unregistered entry",
                    )

                raw_payload: _RawFile | None = None
                raw_envelope: _RawFile | None = None
                try:
                    raw_payload = _read_regular_file_once(
                        directory_fd,
                        expected.payload_path.name,
                        expected.payload_relative_path,
                    )
                    result["payload_raw_sha256"] = raw_payload.sha256
                    file_identities[raw_payload.identity].append((expected.task_index, "payload"))
                    payload_hash_tasks[raw_payload.sha256].append(expected.task_index)
                except CalibrationFitAggregateError as error:
                    code = (
                        "missing_payload"
                        if expected.payload_path.name not in names_before
                        else "invalid_payload_file"
                    )
                    _add_task_error(result, code, expected.payload_relative_path, str(error))
                try:
                    raw_envelope = _read_regular_file_once(
                        directory_fd,
                        expected.envelope_path.name,
                        expected.envelope_relative_path,
                    )
                    result["envelope_raw_sha256"] = raw_envelope.sha256
                    file_identities[raw_envelope.identity].append((expected.task_index, "envelope"))
                    envelope_hash_tasks[raw_envelope.sha256].append(expected.task_index)
                except CalibrationFitAggregateError as error:
                    code = (
                        "missing_envelope"
                        if expected.envelope_path.name not in names_before
                        else "invalid_envelope_file"
                    )
                    _add_task_error(result, code, expected.envelope_relative_path, str(error))

                if raw_payload is not None:
                    try:
                        payload = _validate_payload_bytes(raw_payload, expected.task_index)
                        payloads[expected.task_index] = payload
                        result["payload_canonical_sha256"] = canonical_sha256(payload)
                        if not _deployment_matches(payload, deployment):
                            _add_task_error(
                                result,
                                "deployment_identity_mismatch",
                                expected.payload_relative_path,
                                "payload code/dependency/catalog identity differs from deployment",
                            )
                    except CalibrationFitAggregateError as error:
                        _add_task_error(
                            result,
                            "malformed_payload",
                            expected.payload_relative_path,
                            str(error),
                        )
                if raw_envelope is not None and raw_payload is not None:
                    try:
                        envelope = _validate_envelope_bytes(raw_envelope, raw_payload, expected)
                        result["envelope_record_sha256"] = envelope["canonical_record_sha256"]
                        if expected.task_index in payloads:
                            _strict_match(
                                envelope["canonical_record"]["task_record"],
                                payloads[expected.task_index]["task_record"],
                                "payload/envelope task identity",
                            )
                            _strict_match(
                                envelope["canonical_record"]["fit_recipe"],
                                payloads[expected.task_index]["fit_recipe"],
                                "payload/envelope fit recipe",
                            )
                    except CalibrationFitAggregateError as error:
                        _add_task_error(
                            result,
                            "malformed_envelope",
                            expected.envelope_relative_path,
                            str(error),
                        )
                elif raw_envelope is not None:
                    _add_task_error(
                        result,
                        "unverifiable_envelope_binding",
                        expected.envelope_relative_path,
                        "payload bytes are unavailable for envelope hash verification",
                    )

                names_after = _scandir_names(directory_fd, directory_name)
                try:
                    directory_path_state = os.stat(
                        directory_name,
                        dir_fd=root_fd,
                        follow_symlinks=False,
                    )
                except OSError:
                    directory_path_state = None
                if (
                    names_after != names_before
                    or directory_path_state is None
                    or not _same_stat(directory_before, directory_path_state)
                ):
                    _add_task_error(
                        result,
                        "task_directory_changed",
                        directory_name,
                        "task directory changed during aggregation",
                    )
            finally:
                os.close(directory_fd)

        for identity, owners in file_identities.items():
            if len(owners) <= 1:
                continue
            owner_text = ",".join(f"{index}:{kind}" for index, kind in owners)
            for task_index, kind in owners:
                result = results[task_index]
                path = result["payload_path"] if kind == "payload" else result["envelope_path"]
                _add_task_error(
                    result,
                    "duplicate_file_identity",
                    path,
                    f"input inode is shared by {owner_text}; identity={identity}",
                )
        for digest, task_indices in payload_hash_tasks.items():
            if len(task_indices) > 1:
                for task_index in task_indices:
                    _add_task_error(
                        results[task_index],
                        "duplicate_payload_bytes",
                        results[task_index]["payload_path"],
                        f"payload bytes are duplicated across task slots {task_indices}: {digest}",
                    )
        for digest, task_indices in envelope_hash_tasks.items():
            if len(task_indices) > 1:
                for task_index in task_indices:
                    _add_task_error(
                        results[task_index],
                        "duplicate_envelope_bytes",
                        results[task_index]["envelope_path"],
                        f"envelope bytes are duplicated across task slots {task_indices}: {digest}",
                    )

        cohort_groups: defaultdict[str, list[int]] = defaultdict(list)
        bundle_groups: defaultdict[str, defaultdict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        bundle_paths: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
        bundle_contents: defaultdict[str, set[str]] = defaultdict(set)
        for task_index, payload in payloads.items():
            cohort_groups[canonical_sha256(_cohort_identity(payload))].append(task_index)
            bundle = payload["input_identity"]["bundle"]
            stem = FIT_TASKS[task_index].dataset_stem
            bundle_groups[stem][canonical_sha256(bundle)].append(task_index)
            for field in ("dataset_path", "metadata_path", "manifest_path"):
                bundle_paths[(field, bundle[field])].append(task_index)
            bundle_contents[bundle["dataset_content_sha256"]].add(stem)
        if len(cohort_groups) > 1:
            for task_indices in cohort_groups.values():
                for task_index in task_indices:
                    _add_task_error(
                        results[task_index],
                        "cohort_identity_mismatch",
                        results[task_index]["payload_path"],
                        "source, dependency, runtime, or catalog cohort identity is mixed",
                    )
        for stem, identities in bundle_groups.items():
            if len(identities) > 1:
                for task_indices in identities.values():
                    for task_index in task_indices:
                        _add_task_error(
                            results[task_index],
                            "bundle_identity_mismatch",
                            results[task_index]["payload_path"],
                            f"seed-replicated bundle identity is inconsistent for {stem}",
                        )
        for (field, path), task_indices in bundle_paths.items():
            stems = {FIT_TASKS[index].dataset_stem for index in task_indices}
            if len(stems) > 1:
                for task_index in task_indices:
                    _add_task_error(
                        results[task_index],
                        "duplicate_bundle_path",
                        results[task_index]["payload_path"],
                        f"{field} is reused by distinct bundle identities: {path}",
                    )
        for digest, stems in bundle_contents.items():
            if len(stems) > 1:
                for task_index, payload in payloads.items():
                    if payload["input_identity"]["bundle"]["dataset_content_sha256"] == digest:
                        _add_task_error(
                            results[task_index],
                            "duplicate_dataset_content",
                            results[task_index]["payload_path"],
                            f"dataset content hash is reused by bundle identities {sorted(stems)}",
                        )

        _audit_scheduler_cohort(payloads, results)

        root_names_after = _scandir_names(root_fd, "fit-stage input root")
        try:
            root_path_state = root_path.stat(follow_symlinks=False)
        except OSError:
            root_path_state = None
        if (
            root_names_after != root_names_before
            or root_path_state is None
            or not _same_stat(root_before, root_path_state)
        ):
            structural_failures.append(
                _failure(
                    "input_root_changed",
                    str(root_path),
                    "fit-stage input root changed during aggregation",
                )
            )
    finally:
        os.close(root_fd)

    cohort_sha = next(iter(cohort_groups)) if len(cohort_groups) == 1 else None
    return _build_catalog(
        results,
        structural_failures,
        deployment,
        cohort_identity_sha256=cohort_sha,
    )


def validate_fit_catalog(value: Any) -> dict[str, Any]:
    """Strictly validate a complete success or failure fit-stage catalog."""

    try:
        canonical_json_bytes(value)
    except CalibrationContractError as error:
        raise CalibrationFitAggregateError("fit catalog is not finite strict JSON") from error
    document = _require_object(value, _CATALOG_KEYS, "fit catalog")
    if document["schema_version"] != FIT_CATALOG_SCHEMA_VERSION:
        raise CalibrationFitAggregateError("fit catalog schema version is unsupported")
    if document["catalog_type"] != FIT_CATALOG_TYPE:
        raise CalibrationFitAggregateError("fit catalog type is unsupported")
    if document["calibration_id"] != CALIBRATION_ID or document["stage"] != "fit":
        raise CalibrationFitAggregateError("fit catalog identity or stage is mismatched")
    _strict_match(document["matrix_hashes"], MATRIX_HASHES, "catalog matrix hashes")
    _strict_match(document["catalog_identity"], CATALOG_IDENTITY, "catalog identity")
    if document["fit_recipe_sha256"] != FIT_RECIPE_SHA256:
        raise CalibrationFitAggregateError("fit catalog recipe SHA-256 is mismatched")
    validate_expected_deployment(document["expected_deployment"])
    if document["expected_task_count"] != FIT_TASK_COUNT:
        raise CalibrationFitAggregateError("fit catalog expected task count is not 45")

    tasks = document["tasks"]
    if not isinstance(tasks, list) or len(tasks) != FIT_TASK_COUNT:
        raise CalibrationFitAggregateError("fit catalog must retain exactly 45 ordered tasks")
    canonical_paths = expected_fit_stage_paths(".")
    valid_count = 0
    task_failure_count = 0
    for task_index, raw_result in enumerate(tasks):
        result = _require_object(raw_result, _TASK_RESULT_KEYS, f"fit catalog task {task_index}")
        if _require_plain_int(result["task_index"], "fit catalog task index") != task_index:
            raise CalibrationFitAggregateError(
                "fit catalog task indices are reordered or duplicated"
            )
        _strict_match(result["task_record"], FIT_TASKS[task_index].as_record(), "catalog task")
        expected = canonical_paths[task_index]
        expected_strings = {
            "artifact_directory": expected.directory.name,
            "payload_path": expected.payload_relative_path,
            "envelope_path": expected.envelope_relative_path,
        }
        if any(result[field] != path for field, path in expected_strings.items()):
            raise CalibrationFitAggregateError("fit catalog contains a noncanonical task path")
        for field in _HASH_RESULT_FIELDS:
            _require_sha256_or_none(result[field], f"fit catalog task {task_index}.{field}")
        errors = result["errors"]
        if not isinstance(errors, list):
            raise CalibrationFitAggregateError("fit catalog task errors must be an array")
        for failure in errors:
            item = _require_object(failure, _FAILURE_KEYS, "fit catalog task failure")
            if any(type(item[field]) is not str or not item[field] for field in _FAILURE_KEYS):
                raise CalibrationFitAggregateError("fit catalog task failure fields must be text")
        task_failure_count += len(errors)
        if result["status"] == "valid":
            if errors or any(result[field] is None for field in _HASH_RESULT_FIELDS):
                raise CalibrationFitAggregateError("valid fit catalog task lacks complete hashes")
            if result["payload_raw_sha256"] != result["payload_canonical_sha256"]:
                raise CalibrationFitAggregateError("valid payload raw and canonical hashes differ")
            valid_count += 1
        elif result["status"] == "invalid":
            if not errors:
                raise CalibrationFitAggregateError("invalid fit catalog task must retain a failure")
        else:
            raise CalibrationFitAggregateError("fit catalog task status is unsupported")

    failures = document["structural_failures"]
    if not isinstance(failures, list):
        raise CalibrationFitAggregateError("fit catalog structural failures must be an array")
    for failure in failures:
        item = _require_object(failure, _FAILURE_KEYS, "fit catalog structural failure")
        if any(type(item[field]) is not str or not item[field] for field in _FAILURE_KEYS):
            raise CalibrationFitAggregateError("fit catalog structural failure fields must be text")
    unexpected_count = sum(failure["code"] == "unexpected_path" for failure in failures)
    expected_structural_count = len(failures) + task_failure_count
    count_checks = {
        "valid_task_count": valid_count,
        "invalid_task_count": FIT_TASK_COUNT - valid_count,
        "unexpected_path_count": unexpected_count,
        "structural_failure_count": expected_structural_count,
    }
    for field, expected_count in count_checks.items():
        if type(document[field]) is not int or document[field] != expected_count:
            raise CalibrationFitAggregateError(
                "fit catalog failure/count accounting is inconsistent"
            )
    cohort_sha = _require_sha256_or_none(
        document["cohort_identity_sha256"], "cohort identity SHA-256"
    )
    expected_ready = (
        valid_count == FIT_TASK_COUNT
        and not failures
        and task_failure_count == 0
        and cohort_sha is not None
    )
    if type(document["analysis_ready"]) is not bool or document["analysis_ready"] != expected_ready:
        raise CalibrationFitAggregateError("fit catalog analysis_ready is inconsistent")
    if document["freeze_ready"] is not False:
        raise CalibrationFitAggregateError("fit-stage catalogs must always have freeze_ready=false")
    _strict_match(document["label_policy"], _LABEL_POLICY, "fit catalog label policy")
    integrity = _require_object(document["integrity"], _CATALOG_INTEGRITY_KEYS, "catalog integrity")
    if integrity["algorithm"] != "sha256":
        raise CalibrationFitAggregateError("fit catalog integrity algorithm must be sha256")
    if integrity["covered_fields"] != list(_CATALOG_PAYLOAD_FIELDS):
        raise CalibrationFitAggregateError("fit catalog integrity covered fields are mismatched")
    if integrity["payload_sha256"] != canonical_sha256(_catalog_payload(document)):
        raise CalibrationFitAggregateError("fit catalog payload SHA-256 is mismatched")
    return json.loads(canonical_json_bytes(document))


def _open_output_directory(path: Path) -> int:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return os.open(path, _OPEN_DIRECTORY_FLAGS)
    except OSError as error:
        raise CalibrationFitAggregateError("cannot open a no-follow output directory") from error


def write_fit_catalog_exclusive(output_path: str | Path, catalog: Any) -> Path:
    """Atomically publish canonical catalog bytes without replacing any entry."""

    validated = validate_fit_catalog(catalog)
    destination = Path(output_path)
    if not destination.name or destination.name in {".", ".."}:
        raise CalibrationFitAggregateError("fit catalog output must name a file")
    encoded = canonical_json_bytes(validated)
    directory_fd = _open_output_directory(destination.parent)
    temporary_name: str | None = None
    descriptor: int | None = None
    published = False
    try:
        for _ in range(100):
            candidate = f".{destination.name}.{secrets.token_hex(12)}.tmp"
            try:
                descriptor = os.open(
                    candidate,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=directory_fd,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if descriptor is None or temporary_name is None:
            raise CalibrationFitAggregateError("cannot reserve a temporary catalog output")
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written == 0:
                raise CalibrationFitAggregateError("catalog temporary write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(
                temporary_name,
                destination.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise CalibrationFitAggregateError(
                f"refusing to overwrite fit catalog output: {destination}"
            ) from error
        published = True
        os.unlink(temporary_name, dir_fd=directory_fd)
        temporary_name = None
        os.fsync(directory_fd)
    except CalibrationFitAggregateError:
        raise
    except OSError as error:
        state = "after publication" if published else "before publication"
        raise CalibrationFitAggregateError(
            f"cannot atomically write fit catalog ({state}): {destination}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except OSError:
                pass
        os.close(directory_fd)
    return destination


def aggregate_and_write_fit_stage(
    input_root: str | Path,
    output_path: str | Path,
    *,
    expected_deployment: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    """Build and exclusively publish a success or structural-failure catalog."""

    resolved_input = Path(input_root).resolve()
    resolved_output = Path(output_path).resolve()
    try:
        resolved_output.relative_to(resolved_input)
    except ValueError:
        pass
    else:
        raise CalibrationFitAggregateError(
            "fit catalog output must be outside the aggregated input root"
        )
    catalog = aggregate_fit_stage(input_root, expected_deployment=expected_deployment)
    return catalog, write_fit_catalog_exclusive(output_path, catalog)


__all__ = [
    "CalibrationFitAggregateError",
    "ExpectedFitStagePaths",
    "FIT_CATALOG_SCHEMA_VERSION",
    "FIT_CATALOG_TYPE",
    "aggregate_and_write_fit_stage",
    "aggregate_fit_stage",
    "expected_fit_stage_paths",
    "validate_expected_deployment",
    "validate_fit_catalog",
    "write_fit_catalog_exclusive",
]
