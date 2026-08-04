"""Canonical task indexes for the registered F02b probe stage.

One probe task owns exactly 100 target positions and two ORBIT dtypes.  The
large target records remain separate canonical files; this module builds a
small canonical index that enumerates all 200 identity-derived filenames and
binds their raw bytes.  Optional support64, full-q, and stress evidence is
validated from the target records before the index is emitted.

The module performs no filesystem, corpus, label, environment, or scheduler
I/O.  A later runner and aggregate may use the path and pair validators here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cluster.f02b_calibration_grid import (
    CALIBRATION_ID,
    PROBE_TASK_MATRIX_SHA256,
    probe_task_for_index,
)
from experiments.f02b_calibration_contract import (
    CalibrationContractError,
    build_payload_binding,
    build_probe_execution_envelope,
    canonical_json_bytes,
    parse_strict_json_bytes,
    validate_probe_execution_envelope,
    verify_numeric_payload_bytes,
)
from experiments.f02b_calibration_fit import SHARING_VERIFICATION_MODE
from experiments.f02b_calibration_fit_aggregate import (
    CalibrationFitAggregateError,
    validate_expected_deployment,
)
from experiments.f02b_calibration_probe_artifact import (
    CanonicalProbeTargetArtifact,
    ProbeTargetArtifactError,
    parse_canonical_probe_target_artifact,
)
from experiments.f02b_calibration_probe_core import (
    PRIMARY_EVALUATION_ROW_COUNT,
    PROBE_WORK_PLAN_SHA256,
    build_probe_work_plan,
)

PROBE_TASK_INDEX_SCHEMA_VERSION = "f02b_calibration_probe_task_index_v2"
PROBE_TASK_INDEX_TYPE = "f02b_registered_probe_task_index"
PROBE_TARGET_ARTIFACT_COUNT = 2 * PRIMARY_EVALUATION_ROW_COUNT
_DTYPES = ("float32", "float64")
_INDEX_FIELDS = {
    "artifact_type",
    "calibration_id",
    "data_access",
    "execution_provenance",
    "input_artifacts",
    "probe_deployment",
    "probe_work_plan_sha256",
    "schema_version",
    "source_arm_binding_sha256",
    "target_artifact_count",
    "targets",
    "task_index",
    "task_record",
}
_EXECUTION_PROVENANCE_FIELDS = {"runtime", "scheduler"}
_RUNTIME_FIELDS = {"packages", "platform", "python_executable", "python_version"}
_SCHEDULER_FIELDS = {
    "array_job_id",
    "array_task_id",
    "job_id",
    "node_list",
    "sharing_verification_mode",
}
_INPUT_FIELDS = {
    "fit_catalog_cohort_identity_sha256",
    "fit_catalog_integrity_payload_sha256",
    "fit_catalog_raw_sha256",
    "fit_payload_raw_sha256",
    "fit_task_index",
    "launch_manifest_raw_sha256",
}
_TARGET_FIELDS = {
    "compute_dtype",
    "filename",
    "full_q_present",
    "source_rank_grid_sha256",
    "source_rank_reference_sha256",
    "strata_selection_sha256",
    "stress_present",
    "support64_present",
    "target_artifact_raw_sha256",
    "target_position",
    "target_source_index",
}
_DATA_ACCESS = {
    "confirmatory_replica_accessed": False,
    "evaluation_coordinates_accessed": True,
    "evaluation_labels_accessed": False,
    "training_values_and_gradients_accessed": True,
}


class ProbeTaskArtifactError(ValueError):
    """Raised when a task index or its target set violates the registry."""


@dataclass(frozen=True, slots=True)
class ProbeTaskArtifactPaths:
    directory: Path
    task_index: Path
    execution_envelope: Path

    def target_path(self, target_position: int, compute_dtype: str) -> Path:
        return self.directory / _target_filename(target_position, compute_dtype)


@dataclass(frozen=True, slots=True)
class IndexedProbeTarget:
    filename: str
    payload_bytes: bytes
    payload_sha256: str

    def __post_init__(self) -> None:
        if type(self.payload_bytes) is not bytes:
            raise ProbeTaskArtifactError("indexed target payload must be immutable bytes")
        if hashlib.sha256(self.payload_bytes).hexdigest() != self.payload_sha256:
            raise ProbeTaskArtifactError("indexed target payload SHA-256 is mismatched")


@dataclass(frozen=True, slots=True)
class CanonicalProbeTaskArtifact:
    index_bytes: bytes
    index_sha256: str
    targets: tuple[IndexedProbeTarget, ...]

    def __post_init__(self) -> None:
        if type(self.index_bytes) is not bytes:
            raise ProbeTaskArtifactError("probe task index must be immutable bytes")
        if hashlib.sha256(self.index_bytes).hexdigest() != self.index_sha256:
            raise ProbeTaskArtifactError("probe task index SHA-256 is mismatched")
        if len(self.targets) != PROBE_TARGET_ARTIFACT_COUNT:
            raise ProbeTaskArtifactError("probe task artifact must contain 200 targets")


def _sha256(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProbeTaskArtifactError(f"{label} must be a lowercase SHA-256")
    return value


def _plain_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ProbeTaskArtifactError(f"{label} must be an integer, not bool")
    return value


def _object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ProbeTaskArtifactError(f"{label} fields are mismatched")
    return value


def _fresh_json(value: Any, label: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (CalibrationContractError, TypeError, ValueError) as error:
        raise ProbeTaskArtifactError(f"{label} must be finite strict JSON") from error


def _execution_provenance(value: Any, task_index: int) -> dict[str, Any]:
    provenance = _object(
        _fresh_json(value, "execution_provenance"),
        _EXECUTION_PROVENANCE_FIELDS,
        "execution_provenance",
    )
    runtime = _object(provenance["runtime"], _RUNTIME_FIELDS, "execution runtime")
    for name in ("platform", "python_executable", "python_version"):
        if type(runtime[name]) is not str or not runtime[name]:
            raise ProbeTaskArtifactError(f"execution runtime {name} must be nonempty text")
    packages = runtime["packages"]
    if not isinstance(packages, list) or not packages:
        raise ProbeTaskArtifactError("execution runtime packages must be a nonempty list")
    normalized_packages: list[tuple[str, str]] = []
    for position, raw in enumerate(packages):
        package = _object(raw, {"name", "version"}, f"runtime package {position}")
        if any(type(package[name]) is not str or not package[name] for name in package):
            raise ProbeTaskArtifactError("runtime package fields must be nonempty text")
        normalized_packages.append((package["name"], package["version"]))
    if normalized_packages != sorted(normalized_packages) or len(
        normalized_packages
    ) != len(set(normalized_packages)):
        raise ProbeTaskArtifactError("runtime packages must be sorted and unique")

    scheduler = _object(
        provenance["scheduler"],
        _SCHEDULER_FIELDS,
        "execution scheduler",
    )
    for name in ("array_job_id", "job_id"):
        if type(scheduler[name]) is not str or not scheduler[name].isdigit():
            raise ProbeTaskArtifactError(f"scheduler {name} must be decimal text")
    if type(scheduler["node_list"]) is not str or not scheduler["node_list"]:
        raise ProbeTaskArtifactError("scheduler node_list must be nonempty text")
    if _plain_int(scheduler["array_task_id"], "scheduler.array_task_id") != task_index:
        raise ProbeTaskArtifactError("scheduler array task does not match the probe task")
    if scheduler["sharing_verification_mode"] != SHARING_VERIFICATION_MODE:
        raise ProbeTaskArtifactError("scheduler sharing verification mode is not registered")
    return provenance


def _target_filename(target_position: int, compute_dtype: str) -> str:
    position = _plain_int(target_position, "target_position")
    if not 0 <= position < PRIMARY_EVALUATION_ROW_COUNT:
        raise ProbeTaskArtifactError("target_position is outside the registered grid")
    if compute_dtype not in _DTYPES:
        raise ProbeTaskArtifactError("compute_dtype must be float32 or float64")
    return f"target-{position:03d}-{compute_dtype}.json"


def probe_task_artifact_paths(
    output_root: str | Path,
    task_index: int,
) -> ProbeTaskArtifactPaths:
    """Derive the sole task directory and metadata paths from the frozen grid."""

    try:
        task = probe_task_for_index(task_index)
    except (IndexError, ValueError) as error:
        raise ProbeTaskArtifactError("probe task index is outside the frozen grid") from error
    record = task.as_record()
    dirname = (
        f"probe-{task.task_index:03d}-{record['dataset_stem']}-m{task.m}-"
        f"repeat{task.repeat_id}-{PROBE_TASK_MATRIX_SHA256[:12]}"
    )
    directory = Path(output_root).resolve() / dirname
    return ProbeTaskArtifactPaths(
        directory=directory,
        task_index=directory / "probe-index.json",
        execution_envelope=directory / "execution-envelope.json",
    )


def _input_artifacts(
    task_index: int,
    *,
    fit_payload_raw_sha256: str,
    fit_catalog_raw_sha256: str,
    fit_catalog_integrity_payload_sha256: str,
    fit_catalog_cohort_identity_sha256: str,
    launch_manifest_raw_sha256: str,
) -> dict[str, Any]:
    task = probe_task_for_index(task_index)
    return {
        "fit_task_index": task.fit_task_index,
        "fit_payload_raw_sha256": _sha256(
            fit_payload_raw_sha256,
            "fit_payload_raw_sha256",
        ),
        "fit_catalog_raw_sha256": _sha256(
            fit_catalog_raw_sha256,
            "fit_catalog_raw_sha256",
        ),
        "fit_catalog_integrity_payload_sha256": _sha256(
            fit_catalog_integrity_payload_sha256,
            "fit_catalog_integrity_payload_sha256",
        ),
        "fit_catalog_cohort_identity_sha256": _sha256(
            fit_catalog_cohort_identity_sha256,
            "fit_catalog_cohort_identity_sha256",
        ),
        "launch_manifest_raw_sha256": _sha256(
            launch_manifest_raw_sha256,
            "launch_manifest_raw_sha256",
        ),
    }


def _target_entry(
    task_index: int,
    artifact: CanonicalProbeTargetArtifact,
) -> tuple[dict[str, Any], IndexedProbeTarget, dict[str, Any]]:
    if type(artifact) is not CanonicalProbeTargetArtifact:
        raise ProbeTaskArtifactError(
            "targets must be exact CanonicalProbeTargetArtifact objects"
        )
    try:
        parsed = parse_canonical_probe_target_artifact(
            artifact.payload_bytes,
            expected_sha256=artifact.payload_sha256,
        )
    except ProbeTargetArtifactError as error:
        raise ProbeTaskArtifactError("target artifact is invalid") from error
    if parsed["task_index"] != task_index:
        raise ProbeTaskArtifactError("target artifact belongs to a different task")
    compute_dtype = parsed["orbit"].get("compute_dtype")
    filename = _target_filename(parsed["target_position"], compute_dtype)
    entry = {
        "compute_dtype": compute_dtype,
        "filename": filename,
        "full_q_present": parsed["full_q"] is not None,
        "source_rank_grid_sha256": parsed["source_rank_grid_sha256"],
        "source_rank_reference_sha256": parsed["source_rank_reference_sha256"],
        "strata_selection_sha256": parsed["strata_selection_sha256"],
        "stress_present": parsed["stress"] is not None,
        "support64_present": parsed["support64"] is not None,
        "target_artifact_raw_sha256": artifact.payload_sha256,
        "target_position": parsed["target_position"],
        "target_source_index": parsed["target_source_index"],
    }
    indexed = IndexedProbeTarget(
        filename=filename,
        payload_bytes=artifact.payload_bytes,
        payload_sha256=artifact.payload_sha256,
    )
    return entry, indexed, parsed


def _validate_target_population(
    task_index: int,
    parsed_targets: Mapping[tuple[int, str], dict[str, Any]],
) -> str:
    task = probe_task_for_index(task_index)
    plan = build_probe_work_plan(task)
    expected_keys = {
        (position, dtype)
        for position in range(PRIMARY_EVALUATION_ROW_COUNT)
        for dtype in _DTYPES
    }
    if set(parsed_targets) != expected_keys:
        raise ProbeTaskArtifactError(
            "target population must contain every position and dtype exactly once"
        )

    source_bindings: set[str] = set()
    main_grid_hashes: set[str] = set()
    strata_hashes: set[str] = set()
    support_positions: list[int] = []
    full_q_positions: list[int] = []
    stress_positions: list[int] = []
    target_sources: set[int] = set()
    for position in range(PRIMARY_EVALUATION_ROW_COUNT):
        arm32 = parsed_targets[(position, "float32")]
        arm64 = parsed_targets[(position, "float64")]
        for payload in (arm32, arm64):
            source_bindings.add(payload["source_arm_binding_sha256"])
            main_grid_hashes.add(payload["source_rank_grid_sha256"])
            strata_hashes.add(payload["strata_selection_sha256"])
        paired_fields = (
            "target_position",
            "target_source_index",
            "source_arm_binding_sha256",
            "source_rank_reference_sha256",
            "source_rank_grid_sha256",
            "strata_selection_sha256",
        )
        if any(arm32[name] != arm64[name] for name in paired_fields) or (
            arm32["orbit"]["neighbour_positions"]
            != arm64["orbit"]["neighbour_positions"]
            or arm32["orbit"]["neighbour_source_indices"]
            != arm64["orbit"]["neighbour_source_indices"]
            or arm32["orbit"]["include_stratum_sweep"]
            != arm64["orbit"]["include_stratum_sweep"]
        ):
            raise ProbeTaskArtifactError("paired fp32/fp64 target identities are mismatched")
        target_sources.add(arm64["target_source_index"])
        if any(arm32[name] is not None for name in ("support64", "full_q", "stress")):
            raise ProbeTaskArtifactError("fp32 target artifacts cannot carry auxiliary arms")

        selected = arm64["orbit"]["include_stratum_sweep"] is True
        if (arm64["support64"] is not None) != selected:
            raise ProbeTaskArtifactError(
                "support64 presence must exactly match the registered selected strata"
            )
        if selected:
            support_positions.append(position)
        expect_full_q = selected and plan.run_full_q
        if (arm64["full_q"] is not None) != expect_full_q:
            raise ProbeTaskArtifactError(
                "full-q presence must exactly match the registered m=50 strata"
            )
        if expect_full_q:
            full_q_positions.append(position)
        if arm64["stress"] is not None:
            stress_positions.append(position)

    if (
        len(source_bindings) != 1
        or len(main_grid_hashes) != 1
        or len(strata_hashes) != 1
        or len(target_sources) != PRIMARY_EVALUATION_ROW_COUNT
        or len(support_positions) != plan.support_target_count
        or len(full_q_positions)
        != (plan.support_target_count if plan.run_full_q else 0)
        or len(stress_positions) != plan.stress_support_target_count
    ):
        raise ProbeTaskArtifactError("target population does not match the registered work plan")
    return next(iter(source_bindings))


def build_canonical_probe_task_artifact(
    task_index: int,
    target_artifacts: Sequence[CanonicalProbeTargetArtifact],
    *,
    fit_payload_raw_sha256: str,
    fit_catalog_raw_sha256: str,
    fit_catalog_integrity_payload_sha256: str,
    fit_catalog_cohort_identity_sha256: str,
    launch_manifest_raw_sha256: str,
    probe_deployment: Mapping[str, Any],
    execution_provenance: Mapping[str, Any],
) -> CanonicalProbeTaskArtifact:
    """Build one immutable index after validating the complete 200-target set."""

    try:
        task = probe_task_for_index(task_index)
    except (IndexError, ValueError) as error:
        raise ProbeTaskArtifactError("probe task index is outside the frozen grid") from error
    if not isinstance(target_artifacts, Sequence) or isinstance(
        target_artifacts,
        (str, bytes, bytearray),
    ):
        raise ProbeTaskArtifactError("target_artifacts must be a sequence")
    if len(target_artifacts) != PROBE_TARGET_ARTIFACT_COUNT:
        raise ProbeTaskArtifactError("probe task index requires exactly 200 target artifacts")
    try:
        deployment = validate_expected_deployment(probe_deployment)
    except CalibrationFitAggregateError as error:
        raise ProbeTaskArtifactError("probe_deployment is invalid") from error

    entries_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    indexed_by_key: dict[tuple[int, str], IndexedProbeTarget] = {}
    parsed_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for artifact in target_artifacts:
        entry, indexed, parsed = _target_entry(task.task_index, artifact)
        key = (entry["target_position"], entry["compute_dtype"])
        if key in entries_by_key:
            raise ProbeTaskArtifactError("target artifact identity is duplicated")
        entries_by_key[key] = entry
        indexed_by_key[key] = indexed
        parsed_by_key[key] = parsed

    source_binding = _validate_target_population(task.task_index, parsed_by_key)
    ordered_keys = [
        (position, dtype)
        for position in range(PRIMARY_EVALUATION_ROW_COUNT)
        for dtype in _DTYPES
    ]
    input_artifacts = _input_artifacts(
        task.task_index,
        fit_payload_raw_sha256=fit_payload_raw_sha256,
        fit_catalog_raw_sha256=fit_catalog_raw_sha256,
        fit_catalog_integrity_payload_sha256=fit_catalog_integrity_payload_sha256,
        fit_catalog_cohort_identity_sha256=fit_catalog_cohort_identity_sha256,
        launch_manifest_raw_sha256=launch_manifest_raw_sha256,
    )
    payload = {
        "schema_version": PROBE_TASK_INDEX_SCHEMA_VERSION,
        "artifact_type": PROBE_TASK_INDEX_TYPE,
        "calibration_id": CALIBRATION_ID,
        "task_index": task.task_index,
        "task_record": task.as_record(),
        "probe_work_plan_sha256": PROBE_WORK_PLAN_SHA256,
        "source_arm_binding_sha256": source_binding,
        "input_artifacts": input_artifacts,
        "probe_deployment": deployment,
        "execution_provenance": _execution_provenance(
            execution_provenance,
            task.task_index,
        ),
        "target_artifact_count": PROBE_TARGET_ARTIFACT_COUNT,
        "targets": [entries_by_key[key] for key in ordered_keys],
        "data_access": _fresh_json(_DATA_ACCESS, "data_access"),
    }
    index_bytes = canonical_json_bytes(payload)
    return CanonicalProbeTaskArtifact(
        index_bytes=index_bytes,
        index_sha256=hashlib.sha256(index_bytes).hexdigest(),
        targets=tuple(indexed_by_key[key] for key in ordered_keys),
    )


def parse_canonical_probe_task_index(
    index_bytes: bytes,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Strictly parse a canonical 200-target task index."""

    if type(index_bytes) is not bytes:
        raise ProbeTaskArtifactError("probe task index parser requires immutable bytes")
    observed_sha256 = hashlib.sha256(index_bytes).hexdigest()
    if expected_sha256 is not None and _sha256(
        expected_sha256,
        "expected_sha256",
    ) != observed_sha256:
        raise ProbeTaskArtifactError("probe task index raw SHA-256 is mismatched")
    try:
        parsed = parse_strict_json_bytes(index_bytes)
        if canonical_json_bytes(parsed) != index_bytes:
            raise ProbeTaskArtifactError("probe task index is not canonically encoded")
    except CalibrationContractError as error:
        raise ProbeTaskArtifactError("probe task index is not strict canonical JSON") from error
    document = _object(parsed, _INDEX_FIELDS, "probe task index")
    if (
        document["schema_version"] != PROBE_TASK_INDEX_SCHEMA_VERSION
        or document["artifact_type"] != PROBE_TASK_INDEX_TYPE
        or document["calibration_id"] != CALIBRATION_ID
        or document["probe_work_plan_sha256"] != PROBE_WORK_PLAN_SHA256
    ):
        raise ProbeTaskArtifactError("probe task index schema identity is invalid")
    task_index = _plain_int(document["task_index"], "task_index")
    try:
        task = probe_task_for_index(task_index)
    except (IndexError, ValueError) as error:
        raise ProbeTaskArtifactError("probe task index is outside the frozen grid") from error
    if document["task_record"] != task.as_record():
        raise ProbeTaskArtifactError("probe task record is mismatched")
    _sha256(document["source_arm_binding_sha256"], "source_arm_binding_sha256")
    inputs = _object(document["input_artifacts"], _INPUT_FIELDS, "input_artifacts")
    if _plain_int(inputs["fit_task_index"], "fit_task_index") != task.fit_task_index:
        raise ProbeTaskArtifactError("fit task index is mismatched")
    for name in _INPUT_FIELDS - {"fit_task_index"}:
        _sha256(inputs[name], f"input_artifacts.{name}")
    try:
        validate_expected_deployment(document["probe_deployment"])
    except CalibrationFitAggregateError as error:
        raise ProbeTaskArtifactError("probe_deployment is invalid") from error
    _execution_provenance(document["execution_provenance"], task.task_index)
    if document["data_access"] != _DATA_ACCESS:
        raise ProbeTaskArtifactError("probe task data-access record is mismatched")
    if (
        _plain_int(document["target_artifact_count"], "target_artifact_count")
        != PROBE_TARGET_ARTIFACT_COUNT
        or not isinstance(document["targets"], list)
        or len(document["targets"]) != PROBE_TARGET_ARTIFACT_COUNT
    ):
        raise ProbeTaskArtifactError("probe task target count is mismatched")
    observed_keys: list[tuple[int, str]] = []
    entries_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for index, raw in enumerate(document["targets"]):
        entry = _object(raw, _TARGET_FIELDS, f"targets[{index}]")
        position = _plain_int(entry["target_position"], f"targets[{index}].target_position")
        dtype = entry["compute_dtype"]
        if entry["filename"] != _target_filename(position, dtype):
            raise ProbeTaskArtifactError("target filename is not identity-derived")
        if type(entry["target_source_index"]) is not int or entry["target_source_index"] < 0:
            raise ProbeTaskArtifactError("target source identity is invalid")
        for name in (
            "target_artifact_raw_sha256",
            "source_rank_reference_sha256",
            "source_rank_grid_sha256",
            "strata_selection_sha256",
        ):
            _sha256(entry[name], f"targets[{index}].{name}")
        if any(
            type(entry[name]) is not bool
            for name in ("support64_present", "full_q_present", "stress_present")
        ):
            raise ProbeTaskArtifactError("target optional-arm flags must be boolean")
        key = (position, dtype)
        observed_keys.append(key)
        entries_by_key[key] = entry
    expected_keys = [
        (position, dtype)
        for position in range(PRIMARY_EVALUATION_ROW_COUNT)
        for dtype in _DTYPES
    ]
    if observed_keys != expected_keys:
        raise ProbeTaskArtifactError("target index ordering or coverage is mismatched")
    plan = build_probe_work_plan(task)
    support_positions: list[int] = []
    full_q_positions: list[int] = []
    stress_positions: list[int] = []
    grid_hashes: set[str] = set()
    strata_hashes: set[str] = set()
    source_indices: set[int] = set()
    for position in range(PRIMARY_EVALUATION_ROW_COUNT):
        arm32 = entries_by_key[(position, "float32")]
        arm64 = entries_by_key[(position, "float64")]
        if any(
            arm32[name] != arm64[name]
            for name in (
                "target_position",
                "target_source_index",
                "source_rank_reference_sha256",
                "source_rank_grid_sha256",
                "strata_selection_sha256",
            )
        ):
            raise ProbeTaskArtifactError("indexed fp32/fp64 identities are mismatched")
        if any(
            arm32[name]
            for name in ("support64_present", "full_q_present", "stress_present")
        ):
            raise ProbeTaskArtifactError("indexed fp32 target carries an auxiliary arm")
        source_indices.add(arm64["target_source_index"])
        grid_hashes.add(arm64["source_rank_grid_sha256"])
        strata_hashes.add(arm64["strata_selection_sha256"])
        if arm64["support64_present"]:
            support_positions.append(position)
        if arm64["full_q_present"]:
            full_q_positions.append(position)
        if arm64["stress_present"]:
            stress_positions.append(position)
    if (
        len(source_indices) != PRIMARY_EVALUATION_ROW_COUNT
        or len(grid_hashes) != 1
        or len(strata_hashes) != 1
        or len(support_positions) != plan.support_target_count
        or full_q_positions != (support_positions if plan.run_full_q else [])
        or len(stress_positions) != plan.stress_support_target_count
    ):
        raise ProbeTaskArtifactError("indexed target metadata violates the work plan")
    return _fresh_json(document, "validated probe task index")


def _index_relative_path(paths: ProbeTaskArtifactPaths) -> str:
    return f"{paths.directory.name}/{paths.task_index.name}"


def build_probe_task_execution_envelope(
    task_index: int,
    artifact: CanonicalProbeTaskArtifact,
    *,
    output_root: str | Path,
    runtime_allocation: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one canonical task index to the public probe execution envelope."""

    parsed = parse_canonical_probe_task_index(
        artifact.index_bytes,
        expected_sha256=artifact.index_sha256,
    )
    if parsed["task_index"] != task_index:
        raise ProbeTaskArtifactError("task artifact and requested envelope index differ")
    paths = probe_task_artifact_paths(output_root, task_index)
    binding = build_payload_binding(
        "probe",
        task_index,
        numeric_payload_path=_index_relative_path(paths),
        numeric_payload_sha256=artifact.index_sha256,
    )
    try:
        return build_probe_execution_envelope(
            task_index,
            runtime_allocation=runtime_allocation,
            payload_binding=binding,
        )
    except CalibrationContractError as error:
        raise ProbeTaskArtifactError("probe execution envelope is invalid") from error


def validate_probe_task_artifact_pair(
    artifact: CanonicalProbeTaskArtifact,
    envelope_bytes: bytes,
    *,
    output_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the task index, all target bytes, and its public envelope."""

    if type(artifact) is not CanonicalProbeTaskArtifact or type(envelope_bytes) is not bytes:
        raise ProbeTaskArtifactError("artifact pair requires immutable canonical objects")
    index = parse_canonical_probe_task_index(
        artifact.index_bytes,
        expected_sha256=artifact.index_sha256,
    )
    task_index = index["task_index"]
    expected_targets = index["targets"]
    if len(artifact.targets) != len(expected_targets):
        raise ProbeTaskArtifactError("indexed target byte count is mismatched")
    parsed_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for expected, target in zip(expected_targets, artifact.targets, strict=True):
        if (
            target.filename != expected["filename"]
            or target.payload_sha256 != expected["target_artifact_raw_sha256"]
        ):
            raise ProbeTaskArtifactError("indexed target bytes do not match the task index")
        parsed_target = parse_canonical_probe_target_artifact(
            target.payload_bytes,
            expected_sha256=target.payload_sha256,
        )
        if (
            parsed_target["task_index"] != task_index
            or parsed_target["target_position"] != expected["target_position"]
            or parsed_target["target_source_index"] != expected["target_source_index"]
            or parsed_target["orbit"]["compute_dtype"] != expected["compute_dtype"]
            or parsed_target["source_rank_reference_sha256"]
            != expected["source_rank_reference_sha256"]
            or parsed_target["source_rank_grid_sha256"]
            != expected["source_rank_grid_sha256"]
            or parsed_target["strata_selection_sha256"]
            != expected["strata_selection_sha256"]
            or (parsed_target["support64"] is not None)
            != expected["support64_present"]
            or (parsed_target["full_q"] is not None) != expected["full_q_present"]
            or (parsed_target["stress"] is not None) != expected["stress_present"]
        ):
            raise ProbeTaskArtifactError("indexed target content identity is mismatched")
        key = (expected["target_position"], expected["compute_dtype"])
        parsed_by_key[key] = parsed_target
    source_binding = _validate_target_population(task_index, parsed_by_key)
    if source_binding != index["source_arm_binding_sha256"]:
        raise ProbeTaskArtifactError("indexed source-arm binding is mismatched")
    try:
        envelope_raw = parse_strict_json_bytes(envelope_bytes)
        envelope = validate_probe_execution_envelope(
            envelope_raw,
            expected_task_index=task_index,
        )
        if canonical_json_bytes(envelope) != envelope_bytes:
            raise ProbeTaskArtifactError("probe execution envelope is not canonical JSON")
        paths = probe_task_artifact_paths(output_root, task_index)
        binding = envelope["canonical_record"]["payload_binding"]
        if binding["numeric_payload_path"] != _index_relative_path(paths):
            raise ProbeTaskArtifactError("probe index binding path is not identity-derived")
        verify_numeric_payload_bytes(
            binding,
            artifact.index_bytes,
            expected_task_role="probe",
            expected_task_index=task_index,
        )
    except ProbeTaskArtifactError:
        raise
    except (CalibrationContractError, ProbeTargetArtifactError) as error:
        raise ProbeTaskArtifactError("probe task artifact pair is invalid") from error
    return index, envelope


__all__ = [
    "CanonicalProbeTaskArtifact",
    "IndexedProbeTarget",
    "PROBE_TARGET_ARTIFACT_COUNT",
    "PROBE_TASK_INDEX_SCHEMA_VERSION",
    "PROBE_TASK_INDEX_TYPE",
    "ProbeTaskArtifactError",
    "ProbeTaskArtifactPaths",
    "build_canonical_probe_task_artifact",
    "build_probe_task_execution_envelope",
    "parse_canonical_probe_task_index",
    "probe_task_artifact_paths",
    "validate_probe_task_artifact_pair",
]
