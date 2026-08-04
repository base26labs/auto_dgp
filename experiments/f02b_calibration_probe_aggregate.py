"""Strict filesystem intake for all 122 registered F02b probe tasks.

The aggregate enumerates the frozen task/dtype/target population without
globbing.  It opens directories and files without following links, reads every
accepted file once, verifies the canonical task-index/envelope/target chain,
and retains invalid or missing slots as explicit structural failures.  The
result may become analysis-ready but can never freeze F02b by itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cluster.f02b_calibration_grid import (
    CALIBRATION_ID,
    FIT_TASK_COUNT,
    PROBE_TASK_COUNT,
    PROBE_TASK_MATRIX_SHA256,
    PROBE_TASKS,
)
from experiments.f02b_calibration_contract import (
    CalibrationContractError,
    canonical_json_bytes,
    canonical_sha256,
)
from experiments.f02b_calibration_fit_aggregate import (
    CalibrationFitAggregateError,
    validate_expected_deployment,
)
from experiments.f02b_calibration_probe_core import PROBE_WORK_PLAN_SHA256
from experiments.f02b_calibration_probe_task_artifact import (
    PROBE_TARGET_ARTIFACT_COUNT,
    PROBE_TASK_INDEX_SCHEMA_VERSION,
    CanonicalProbeTaskArtifact,
    IndexedProbeTarget,
    ProbeTaskArtifactError,
    ProbeTaskArtifactPaths,
    probe_task_artifact_paths,
    validate_probe_task_artifact_pair,
)

PROBE_CATALOG_SCHEMA_VERSION = "f02b_calibration_probe_catalog_v1"
PROBE_CATALOG_TYPE = "f02b_development_probe_stage_catalog"
_EXPECTED_CONTEXT_FIELDS = {
    "fit_catalog_cohort_identity_sha256",
    "fit_catalog_integrity_payload_sha256",
    "fit_catalog_raw_sha256",
    "fit_payloads",
    "launch_manifest_raw_sha256",
    "probe_deployment",
}
_FIT_PAYLOAD_FIELDS = {"fit_payload_raw_sha256", "fit_task_index"}
_TASK_RESULT_FIELDS = {
    "artifact_directory",
    "envelope_raw_sha256",
    "envelope_record_sha256",
    "errors",
    "index_raw_sha256",
    "job_identity",
    "source_arm_binding_sha256",
    "status",
    "target_artifact_count",
    "target_bytes_sha256",
    "target_total_bytes",
    "task_index",
    "task_record",
}
_JOB_IDENTITY_FIELDS = {
    "array_job_id",
    "array_task_id",
    "job_id",
    "node_list",
}
_FAILURE_FIELDS = {"code", "message", "path"}
_SCHEDULER_AUDIT_FIELDS = {
    "array_job_ids",
    "complete",
    "job_id_count",
    "passed",
    "policy",
}
_LABEL_POLICY = {
    "scope": "development_only",
    "evaluation_labels_exposed": False,
    "confirmatory_replicas_exposed": False,
    "predictive_scoring_performed": False,
}
_SCHEDULER_POLICY = {
    "primary_task_range": [0, 119],
    "replay_task_indices": [120, 121],
    "three_pairwise_distinct_array_job_ids": True,
    "globally_unique_job_ids": True,
}
_CATALOG_PAYLOAD_FIELDS = (
    "schema_version",
    "catalog_type",
    "calibration_id",
    "stage",
    "probe_task_matrix_sha256",
    "probe_work_plan_sha256",
    "probe_task_index_schema_version",
    "expected_context",
    "expected_task_count",
    "valid_task_count",
    "invalid_task_count",
    "unexpected_path_count",
    "structural_failure_count",
    "runtime_cohort_sha256",
    "scheduler_audit",
    "tasks",
    "structural_failures",
    "label_policy",
    "analysis_ready",
    "freeze_ready",
)
_CATALOG_FIELDS = {*_CATALOG_PAYLOAD_FIELDS, "integrity"}
_INTEGRITY_FIELDS = {"algorithm", "covered_fields", "payload_sha256"}
_OPEN_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_OPEN_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
    os,
    "O_NOFOLLOW",
    0,
)


class CalibrationProbeAggregateError(RuntimeError):
    """Raised for invalid aggregate configuration or catalog validation."""


@dataclass(frozen=True, slots=True)
class ExpectedProbeStagePaths:
    task_index: int
    paths: ProbeTaskArtifactPaths
    target_filenames: tuple[str, ...]

    @property
    def expected_filenames(self) -> set[str]:
        return {
            self.paths.task_index.name,
            self.paths.execution_envelope.name,
            *self.target_filenames,
        }


@dataclass(frozen=True, slots=True)
class _RawFile:
    data: bytes
    sha256: str
    identity: tuple[int, int]


def _sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CalibrationProbeAggregateError(f"{label} must be a lowercase SHA-256")
    return value


def _plain_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise CalibrationProbeAggregateError(f"{label} must be an integer, not bool")
    return value


def _object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise CalibrationProbeAggregateError(f"{label} fields are mismatched")
    return value


def _fresh_json(value: Any, label: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (CalibrationContractError, TypeError, ValueError) as error:
        raise CalibrationProbeAggregateError(f"{label} must be finite strict JSON") from error


def validate_expected_probe_context(value: Any) -> dict[str, Any]:
    """Validate the launch/fit/deployment identities expected in every slot."""

    context = _object(
        _fresh_json(value, "expected probe context"),
        _EXPECTED_CONTEXT_FIELDS,
        "expected probe context",
    )
    for name in (
        "fit_catalog_raw_sha256",
        "fit_catalog_integrity_payload_sha256",
        "fit_catalog_cohort_identity_sha256",
        "launch_manifest_raw_sha256",
    ):
        _sha256(context[name], name)
    try:
        context["probe_deployment"] = validate_expected_deployment(
            context["probe_deployment"]
        )
    except CalibrationFitAggregateError as error:
        raise CalibrationProbeAggregateError("expected probe deployment is invalid") from error
    fit_payloads = context["fit_payloads"]
    if not isinstance(fit_payloads, list) or len(fit_payloads) != FIT_TASK_COUNT:
        raise CalibrationProbeAggregateError("expected context requires all 45 fit payloads")
    normalized: list[dict[str, Any]] = []
    for position, raw in enumerate(fit_payloads):
        record = _object(raw, _FIT_PAYLOAD_FIELDS, f"fit_payloads[{position}]")
        if _plain_int(record["fit_task_index"], "fit_task_index") != position:
            raise CalibrationProbeAggregateError("fit payload indexes must be ordered 0 through 44")
        normalized.append(
            {
                "fit_task_index": position,
                "fit_payload_raw_sha256": _sha256(
                    record["fit_payload_raw_sha256"],
                    f"fit_payloads[{position}].fit_payload_raw_sha256",
                ),
            }
        )
    context["fit_payloads"] = normalized
    return _fresh_json(context, "validated expected probe context")


def expected_probe_stage_paths(input_root: str | Path) -> tuple[ExpectedProbeStagePaths, ...]:
    """Enumerate all 122 directories and 24,400 target filenames without globbing."""

    records: list[ExpectedProbeStagePaths] = []
    for task in PROBE_TASKS:
        paths = probe_task_artifact_paths(input_root, task.task_index)
        names = tuple(
            paths.target_path(position, dtype).name
            for position in range(PROBE_TARGET_ARTIFACT_COUNT // 2)
            for dtype in ("float32", "float64")
        )
        records.append(
            ExpectedProbeStagePaths(
                task_index=task.task_index,
                paths=paths,
                target_filenames=names,
            )
        )
    directory_names = [record.paths.directory.name for record in records]
    if len(records) != PROBE_TASK_COUNT or len(directory_names) != len(set(directory_names)):
        raise CalibrationProbeAggregateError("probe-stage path registry is not exactly 122 tasks")
    if any(
        len(record.expected_filenames) != PROBE_TARGET_ARTIFACT_COUNT + 2
        for record in records
    ):
        raise CalibrationProbeAggregateError("probe-stage target filenames are not unique")
    return tuple(records)


def _failure(code: str, path: str, message: str) -> dict[str, str]:
    if not all(type(value) is str and value for value in (code, path, message)):
        raise CalibrationProbeAggregateError("failure records require nonempty text")
    return {"code": code, "path": path, "message": message}


def _empty_task_result(record: ExpectedProbeStagePaths) -> dict[str, Any]:
    return {
        "task_index": record.task_index,
        "task_record": PROBE_TASKS[record.task_index].as_record(),
        "artifact_directory": record.paths.directory.name,
        "status": "invalid",
        "errors": [],
        "index_raw_sha256": None,
        "envelope_raw_sha256": None,
        "envelope_record_sha256": None,
        "source_arm_binding_sha256": None,
        "target_artifact_count": 0,
        "target_total_bytes": 0,
        "target_bytes_sha256": None,
        "job_identity": None,
    }


def _same_stat(left: os.stat_result, right: os.stat_result) -> bool:
    return all(
        getattr(left, name) == getattr(right, name)
        for name in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    )


def _scandir_names(directory_fd: int, label: str) -> set[str]:
    try:
        with os.scandir(directory_fd) as entries:
            return {entry.name for entry in entries}
    except OSError as error:
        raise CalibrationProbeAggregateError(f"cannot enumerate {label}") from error


def _read_regular_file_once(directory_fd: int, name: str) -> _RawFile:
    try:
        descriptor = os.open(name, _OPEN_FILE_FLAGS, dir_fd=directory_fd)
    except FileNotFoundError as error:
        raise CalibrationProbeAggregateError("required file is missing") from error
    except OSError as error:
        raise CalibrationProbeAggregateError("required file cannot be opened no-follow") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise CalibrationProbeAggregateError(
                "required file must be regular with exactly one hard link"
            )
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
        raise CalibrationProbeAggregateError("required file changed while reading") from error
    if not _same_stat(before, after) or not _same_stat(after, path_state):
        raise CalibrationProbeAggregateError("required file changed while reading")
    data = b"".join(chunks)
    if len(data) != before.st_size:
        raise CalibrationProbeAggregateError("required file size changed while reading")
    return _RawFile(
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        identity=(before.st_dev, before.st_ino),
    )


def _context_matches(index: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    inputs = index["input_artifacts"]
    task = PROBE_TASKS[index["task_index"]]
    comparisons = {
        "fit_catalog_raw_sha256": expected["fit_catalog_raw_sha256"],
        "fit_catalog_integrity_payload_sha256": expected[
            "fit_catalog_integrity_payload_sha256"
        ],
        "fit_catalog_cohort_identity_sha256": expected[
            "fit_catalog_cohort_identity_sha256"
        ],
        "launch_manifest_raw_sha256": expected["launch_manifest_raw_sha256"],
        "fit_payload_raw_sha256": expected["fit_payloads"][task.fit_task_index][
            "fit_payload_raw_sha256"
        ],
    }
    if any(inputs[name] != value for name, value in comparisons.items()) or (
        inputs["fit_task_index"] != task.fit_task_index
        or canonical_json_bytes(index["probe_deployment"])
        != canonical_json_bytes(expected["probe_deployment"])
    ):
        raise CalibrationProbeAggregateError("probe task input cohort is mismatched")


def _target_bytes_digest(targets: tuple[IndexedProbeTarget, ...]) -> str:
    hasher = hashlib.sha256()
    for target in targets:
        hasher.update(target.filename.encode("ascii"))
        hasher.update(b"\x00")
        hasher.update(target.payload_sha256.encode("ascii"))
        hasher.update(b"\x00")
        hasher.update(str(len(target.payload_bytes)).encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _scheduler_audit(valid_results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(valid_results) != PROBE_TASK_COUNT:
        return {
            "complete": False,
            "passed": False,
            "array_job_ids": [],
            "job_id_count": len(valid_results),
            "policy": _fresh_json(_SCHEDULER_POLICY, "scheduler policy"),
        }
    identities = [result["job_identity"] for result in valid_results]
    primary_ids = {identity["array_job_id"] for identity in identities[:120]}
    replay_ids = [identities[120]["array_job_id"], identities[121]["array_job_id"]]
    array_ids = [next(iter(primary_ids))] + replay_ids if len(primary_ids) == 1 else replay_ids
    job_ids = [identity["job_id"] for identity in identities]
    passed = (
        len(primary_ids) == 1
        and len(set(array_ids)) == 3
        and len(set(job_ids)) == PROBE_TASK_COUNT
    )
    return {
        "complete": True,
        "passed": passed,
        "array_job_ids": array_ids,
        "job_id_count": len(set(job_ids)),
        "policy": _fresh_json(_SCHEDULER_POLICY, "scheduler policy"),
    }


def _catalog_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    return {field: document[field] for field in _CATALOG_PAYLOAD_FIELDS}


def _build_catalog(
    *,
    expected_context: Mapping[str, Any],
    tasks: list[dict[str, Any]],
    structural_failures: list[dict[str, str]],
    unexpected_path_count: int,
) -> dict[str, Any]:
    valid = [task for task in tasks if task["status"] == "valid"]
    runtime_records = [task.pop("_runtime") for task in tasks]
    runtime_digests = {
        canonical_sha256(runtime)
        for runtime in runtime_records
        if runtime is not None
    }
    runtime_cohort_sha256 = next(iter(runtime_digests)) if len(runtime_digests) == 1 else None
    if len(runtime_digests) > 1:
        structural_failures.append(
            _failure(
                "runtime_cohort_mismatch",
                ".",
                "valid probe tasks do not share one runtime package/environment record",
            )
        )
    scheduler = _scheduler_audit(valid)
    if scheduler["complete"] and not scheduler["passed"]:
        structural_failures.append(
            _failure(
                "scheduler_cohort_mismatch",
                ".",
                "probe tasks do not match the registered three-allocation topology",
            )
        )
    analysis_ready = (
        len(valid) == PROBE_TASK_COUNT
        and not structural_failures
        and unexpected_path_count == 0
        and scheduler["passed"]
        and runtime_cohort_sha256 is not None
    )
    document: dict[str, Any] = {
        "schema_version": PROBE_CATALOG_SCHEMA_VERSION,
        "catalog_type": PROBE_CATALOG_TYPE,
        "calibration_id": CALIBRATION_ID,
        "stage": "development_probe_numerical_evidence",
        "probe_task_matrix_sha256": PROBE_TASK_MATRIX_SHA256,
        "probe_work_plan_sha256": PROBE_WORK_PLAN_SHA256,
        "probe_task_index_schema_version": PROBE_TASK_INDEX_SCHEMA_VERSION,
        "expected_context": _fresh_json(expected_context, "expected context"),
        "expected_task_count": PROBE_TASK_COUNT,
        "valid_task_count": len(valid),
        "invalid_task_count": PROBE_TASK_COUNT - len(valid),
        "unexpected_path_count": unexpected_path_count,
        "structural_failure_count": len(structural_failures),
        "runtime_cohort_sha256": runtime_cohort_sha256,
        "scheduler_audit": scheduler,
        "tasks": tasks,
        "structural_failures": structural_failures,
        "label_policy": _fresh_json(_LABEL_POLICY, "label policy"),
        "analysis_ready": analysis_ready,
        "freeze_ready": False,
    }
    document["integrity"] = {
        "algorithm": "sha256",
        "covered_fields": list(_CATALOG_PAYLOAD_FIELDS),
        "payload_sha256": canonical_sha256(_catalog_payload(document)),
    }
    return document


def aggregate_probe_stage(
    input_root: str | Path,
    *,
    expected_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit the exact 122-slot filesystem and return a canonical catalog object."""

    context = validate_expected_probe_context(expected_context)
    root = Path(input_root)
    expected = expected_probe_stage_paths(root)
    tasks = [_empty_task_result(record) for record in expected]
    for task in tasks:
        task["_runtime"] = None
    failures: list[dict[str, str]] = []
    unexpected_count = 0
    try:
        root_fd = os.open(root, _OPEN_DIRECTORY_FLAGS)
    except FileNotFoundError:
        for record, result in zip(expected, tasks, strict=True):
            error = _failure(
                "missing_task_directory",
                record.paths.directory.name,
                "registered probe task directory is missing",
            )
            result["errors"].append(error)
            failures.append(error)
        return _build_catalog(
            expected_context=context,
            tasks=tasks,
            structural_failures=failures,
            unexpected_path_count=0,
        )
    except OSError as error:
        raise CalibrationProbeAggregateError("probe-stage root cannot be opened no-follow") from error

    seen_files: set[tuple[int, int]] = set()
    try:
        root_before = os.fstat(root_fd)
        root_names = _scandir_names(root_fd, "probe-stage root")
        expected_names = {record.paths.directory.name for record in expected}
        unexpected_root = sorted(root_names - expected_names)
        unexpected_count += len(unexpected_root)
        failures.extend(
            _failure("unexpected_root_path", name, "unregistered probe-stage root entry")
            for name in unexpected_root
        )

        for record, result in zip(expected, tasks, strict=True):
            dirname = record.paths.directory.name
            if dirname not in root_names:
                error = _failure(
                    "missing_task_directory",
                    dirname,
                    "registered probe task directory is missing",
                )
                result["errors"].append(error)
                failures.append(error)
                continue
            try:
                task_fd = os.open(dirname, _OPEN_DIRECTORY_FLAGS, dir_fd=root_fd)
            except OSError as error:
                failure = _failure(
                    "invalid_task_directory",
                    dirname,
                    f"registered task directory cannot be opened no-follow: {error}",
                )
                result["errors"].append(failure)
                failures.append(failure)
                continue
            try:
                task_before = os.fstat(task_fd)
                actual_names = _scandir_names(task_fd, dirname)
                missing = sorted(record.expected_filenames - actual_names)
                unexpected = sorted(actual_names - record.expected_filenames)
                if missing:
                    failure = _failure(
                        "missing_task_files",
                        dirname,
                        f"registered task is missing {len(missing)} files",
                    )
                    result["errors"].append(failure)
                    failures.append(failure)
                if unexpected:
                    unexpected_count += len(unexpected)
                    failure = _failure(
                        "unexpected_task_files",
                        dirname,
                        f"registered task has {len(unexpected)} unexpected entries",
                    )
                    result["errors"].append(failure)
                    failures.append(failure)
                if missing or unexpected:
                    continue

                ordered_names = (
                    record.paths.task_index.name,
                    record.paths.execution_envelope.name,
                    *record.target_filenames,
                )
                raw_files: dict[str, _RawFile] = {}
                alias_found = False
                for name in ordered_names:
                    raw = _read_regular_file_once(task_fd, name)
                    if raw.identity in seen_files:
                        alias_found = True
                    seen_files.add(raw.identity)
                    raw_files[name] = raw
                if alias_found:
                    failure = _failure(
                        "file_identity_alias",
                        dirname,
                        "probe task files alias an already consumed inode",
                    )
                    result["errors"].append(failure)
                    failures.append(failure)
                    continue

                target_records = tuple(
                    IndexedProbeTarget(
                        filename=name,
                        payload_bytes=raw_files[name].data,
                        payload_sha256=raw_files[name].sha256,
                    )
                    for name in record.target_filenames
                )
                index_raw = raw_files[record.paths.task_index.name]
                envelope_raw = raw_files[record.paths.execution_envelope.name]
                artifact = CanonicalProbeTaskArtifact(
                    index_bytes=index_raw.data,
                    index_sha256=index_raw.sha256,
                    targets=target_records,
                )
                index, envelope = validate_probe_task_artifact_pair(
                    artifact,
                    envelope_raw.data,
                    output_root=root,
                )
                _context_matches(index, context)
                scheduler = index["execution_provenance"]["scheduler"]
                task_after = os.fstat(task_fd)
                names_after = _scandir_names(task_fd, dirname)
                if not _same_stat(task_before, task_after) or names_after != actual_names:
                    raise CalibrationProbeAggregateError(
                        "probe task directory changed during aggregation"
                    )
                result.update(
                    {
                        "status": "valid",
                        "index_raw_sha256": index_raw.sha256,
                        "envelope_raw_sha256": envelope_raw.sha256,
                        "envelope_record_sha256": envelope[
                            "canonical_record_sha256"
                        ],
                        "source_arm_binding_sha256": index[
                            "source_arm_binding_sha256"
                        ],
                        "target_artifact_count": len(target_records),
                        "target_total_bytes": sum(
                            len(target.payload_bytes) for target in target_records
                        ),
                        "target_bytes_sha256": _target_bytes_digest(target_records),
                        "job_identity": {
                            name: scheduler[name]
                            for name in _JOB_IDENTITY_FIELDS
                        },
                        "_runtime": index["execution_provenance"]["runtime"],
                    }
                )
            except (CalibrationProbeAggregateError, ProbeTaskArtifactError) as error:
                failure = _failure(
                    "invalid_task_artifact",
                    dirname,
                    str(error),
                )
                result["errors"].append(failure)
                failures.append(failure)
            finally:
                os.close(task_fd)
        root_after = os.fstat(root_fd)
        root_names_after = _scandir_names(root_fd, "probe-stage root")
        if not _same_stat(root_before, root_after) or root_names_after != root_names:
            failures.append(
                _failure(
                    "input_root_changed",
                    ".",
                    "probe-stage root changed during aggregation",
                )
            )
    finally:
        os.close(root_fd)

    return _build_catalog(
        expected_context=context,
        tasks=tasks,
        structural_failures=failures,
        unexpected_path_count=unexpected_count,
    )


def validate_probe_catalog(value: Any) -> dict[str, Any]:
    """Validate an aggregate catalog before threshold discovery consumes it."""

    document = _object(
        _fresh_json(value, "probe catalog"),
        _CATALOG_FIELDS,
        "probe catalog",
    )
    if (
        document["schema_version"] != PROBE_CATALOG_SCHEMA_VERSION
        or document["catalog_type"] != PROBE_CATALOG_TYPE
        or document["calibration_id"] != CALIBRATION_ID
        or document["stage"] != "development_probe_numerical_evidence"
        or document["probe_task_matrix_sha256"] != PROBE_TASK_MATRIX_SHA256
        or document["probe_work_plan_sha256"] != PROBE_WORK_PLAN_SHA256
        or document["probe_task_index_schema_version"]
        != PROBE_TASK_INDEX_SCHEMA_VERSION
    ):
        raise CalibrationProbeAggregateError("probe catalog schema identity is invalid")
    validate_expected_probe_context(document["expected_context"])
    tasks = document["tasks"]
    if not isinstance(tasks, list) or len(tasks) != PROBE_TASK_COUNT:
        raise CalibrationProbeAggregateError("probe catalog must contain exactly 122 task slots")
    valid_count = 0
    for position, raw in enumerate(tasks):
        task = _object(raw, _TASK_RESULT_FIELDS, f"tasks[{position}]")
        expected_paths = probe_task_artifact_paths("/catalog-root", position)
        if (
            task["task_index"] != position
            or task["task_record"] != PROBE_TASKS[position].as_record()
            or task["artifact_directory"] != expected_paths.directory.name
        ):
            raise CalibrationProbeAggregateError("probe catalog task ordering is mismatched")
        if task["status"] not in {"valid", "invalid"} or not isinstance(task["errors"], list):
            raise CalibrationProbeAggregateError("probe catalog task status is invalid")
        for error_position, error in enumerate(task["errors"]):
            _object(error, _FAILURE_FIELDS, f"tasks[{position}].errors[{error_position}]")
        if task["status"] == "valid":
            if task["errors"]:
                raise CalibrationProbeAggregateError("valid probe task contains errors")
            for name in (
                "index_raw_sha256",
                "envelope_raw_sha256",
                "envelope_record_sha256",
                "source_arm_binding_sha256",
                "target_bytes_sha256",
            ):
                _sha256(task[name], f"tasks[{position}].{name}")
            if (
                task["target_artifact_count"] != PROBE_TARGET_ARTIFACT_COUNT
                or type(task["target_total_bytes"]) is not int
                or task["target_total_bytes"] <= 0
            ):
                raise CalibrationProbeAggregateError("valid probe task target counts are invalid")
            job = _object(
                task["job_identity"],
                _JOB_IDENTITY_FIELDS,
                f"tasks[{position}].job_identity",
            )
            if (
                job["array_task_id"] != position
                or any(
                    type(job[name]) is not str or not job[name]
                    for name in ("array_job_id", "job_id", "node_list")
                )
            ):
                raise CalibrationProbeAggregateError("valid probe task job identity is invalid")
            valid_count += 1
        elif not task["errors"]:
            raise CalibrationProbeAggregateError("invalid probe task lacks an error record")
    for name in (
        "expected_task_count",
        "valid_task_count",
        "invalid_task_count",
        "unexpected_path_count",
        "structural_failure_count",
    ):
        if type(document[name]) is not int or document[name] < 0:
            raise CalibrationProbeAggregateError(f"probe catalog {name} is invalid")
    if (
        document["expected_task_count"] != PROBE_TASK_COUNT
        or document["valid_task_count"] != valid_count
        or document["invalid_task_count"] != PROBE_TASK_COUNT - valid_count
        or document["freeze_ready"] is not False
        or document["label_policy"] != _LABEL_POLICY
    ):
        raise CalibrationProbeAggregateError("probe catalog counts or policy are invalid")
    failures = document["structural_failures"]
    if not isinstance(failures, list):
        raise CalibrationProbeAggregateError("probe structural failures must be an array")
    for position, failure in enumerate(failures):
        _object(failure, _FAILURE_FIELDS, f"structural_failures[{position}]")
    if document["structural_failure_count"] != len(failures):
        raise CalibrationProbeAggregateError("probe structural failure count is mismatched")
    scheduler = _object(
        document["scheduler_audit"],
        _SCHEDULER_AUDIT_FIELDS,
        "scheduler_audit",
    )
    if (
        type(scheduler["complete"]) is not bool
        or type(scheduler["passed"]) is not bool
        or type(scheduler["job_id_count"]) is not int
        or not isinstance(scheduler["array_job_ids"], list)
        or scheduler["policy"] != _SCHEDULER_POLICY
    ):
        raise CalibrationProbeAggregateError("probe scheduler audit is invalid")
    runtime_cohort_sha256 = document["runtime_cohort_sha256"]
    if runtime_cohort_sha256 is not None:
        _sha256(runtime_cohort_sha256, "runtime_cohort_sha256")
    expected_ready = (
        valid_count == PROBE_TASK_COUNT
        and len(failures) == 0
        and document["unexpected_path_count"] == 0
        and scheduler["passed"] is True
        and document["runtime_cohort_sha256"] is not None
    )
    if document["analysis_ready"] is not expected_ready:
        raise CalibrationProbeAggregateError("probe catalog analysis_ready is inconsistent")
    integrity = _object(document["integrity"], _INTEGRITY_FIELDS, "integrity")
    if (
        integrity["algorithm"] != "sha256"
        or integrity["covered_fields"] != list(_CATALOG_PAYLOAD_FIELDS)
        or integrity["payload_sha256"] != canonical_sha256(_catalog_payload(document))
    ):
        raise CalibrationProbeAggregateError("probe catalog integrity is mismatched")
    return _fresh_json(document, "validated probe catalog")


def _open_output_directory(path: Path) -> int:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return os.open(path, _OPEN_DIRECTORY_FLAGS)
    except OSError as error:
        raise CalibrationProbeAggregateError(
            "cannot open a no-follow probe-catalog output directory"
        ) from error


def write_probe_catalog_exclusive(output_path: str | Path, catalog: Any) -> Path:
    """Atomically publish canonical catalog bytes without replacing any entry."""

    validated = validate_probe_catalog(catalog)
    destination = Path(output_path)
    if not destination.name or destination.name in {".", ".."}:
        raise CalibrationProbeAggregateError("probe catalog output must name a file")
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
            raise CalibrationProbeAggregateError(
                "cannot reserve a temporary probe-catalog output"
            )
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written == 0:
                raise CalibrationProbeAggregateError(
                    "probe-catalog temporary write made no progress"
                )
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
            raise CalibrationProbeAggregateError(
                f"refusing to overwrite probe catalog output: {destination}"
            ) from error
        published = True
        os.unlink(temporary_name, dir_fd=directory_fd)
        temporary_name = None
        os.fsync(directory_fd)
    except CalibrationProbeAggregateError:
        raise
    except OSError as error:
        state = "after publication" if published else "before publication"
        raise CalibrationProbeAggregateError(
            f"cannot atomically write probe catalog ({state}): {destination}"
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


def aggregate_and_write_probe_stage(
    input_root: str | Path,
    output_path: str | Path,
    *,
    expected_context: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    """Build and exclusively publish a success or structural-failure catalog."""

    resolved_input = Path(input_root).resolve()
    resolved_output = Path(output_path).resolve()
    try:
        resolved_output.relative_to(resolved_input)
    except ValueError:
        pass
    else:
        raise CalibrationProbeAggregateError(
            "probe catalog output must be outside the aggregated input root"
        )
    catalog = aggregate_probe_stage(input_root, expected_context=expected_context)
    return catalog, write_probe_catalog_exclusive(output_path, catalog)


__all__ = [
    "CalibrationProbeAggregateError",
    "ExpectedProbeStagePaths",
    "PROBE_CATALOG_SCHEMA_VERSION",
    "PROBE_CATALOG_TYPE",
    "aggregate_and_write_probe_stage",
    "aggregate_probe_stage",
    "expected_probe_stage_paths",
    "validate_expected_probe_context",
    "validate_probe_catalog",
    "write_probe_catalog_exclusive",
]
