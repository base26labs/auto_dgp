"""Fail-closed artifact and result contract for the F02 external baselines.

This module is deliberately independent of the frozen DSoftKI source tree.  It
defines the boundary around that code: canonical F02 arrays are serialized in
three disjoint artifacts, released-recipe configuration is made explicit, and
worker output is validated before central scoring.  In particular, the model
worker receives training labels and evaluation *features*, never evaluation
labels.  The central-only label escrow is joined to predictions later by stable
source IDs.

The actual released model imports belong in a separate Slurm-only backend.  No
external model is imported or executed by this module.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import re
import zipfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from data.load_nbody_confirmatory import PreparedConfirmatorySplit
from experiments.f02_design import (
    EVALUATION_TIME_INDICES,
    OPTIMIZER_SELECTION_TIME_INDICES,
    TRAIN_TIME_INDICES,
)

ARTIFACT_SCHEMA_VERSION = "f02_external_artifact_v1"
CONFIG_SCHEMA_VERSION = "f02_external_config_v1"
RESULT_SCHEMA_VERSION = "f02_external_result_v1"
METHOD_IDS = ("dsoftki-512", "ddsvgp-512")
OPTIMIZER_SEEDS = (11, 29, 47)
OPTIMIZER_UPDATE_CANDIDATES = (20, 50, 100)
F02_DIMENSIONS = (12, 24, 36, 48, 60)
F02_REPLICAS = (0, 1, 2, *range(101, 111))
F02_PARTICLE_COUNTS = (2, 4, 6, 8, 10)
TRAIN_ROWS = 1_500
TRAIN_BATCH_SIZE = 1_024
UPDATES_PER_EPOCH = math.ceil(TRAIN_ROWS / TRAIN_BATCH_SIZE)
VENDOR_COMMIT = "b2382e10a045abca3d653ad58c4a2a9c1ca73458"
VENDOR_SOURCE_TREE = "9589281d694942291eceaaf474851f2d4d24edc9"
F02_PROTOCOL_SHA256 = "6a103772e99e953d71a13f9655faea91f532613d750610b8358b6a1cc2bb2df8"

_HEX_40 = re.compile(r"[0-9a-f]{40}")
_HEX_64 = re.compile(r"[0-9a-f]{64}")
_TRAIN_FIELDS = {
    "metadata_json",
    "X_standardized",
    "H_standardized",
    "dH_dx_standardized",
    "source_ids",
    "source_indices",
    "trajectory_ids",
    "time_indices",
}
_FEATURE_FIELDS = {
    "metadata_json",
    "X_standardized",
    "source_ids",
    "source_indices",
    "trajectory_ids",
    "time_indices",
}
_LABEL_FIELDS = {
    "metadata_json",
    "H_standardized",
    "dH_dx_standardized",
    "source_ids",
    "source_indices",
    "trajectory_ids",
    "time_indices",
}
_COMMON_METADATA_FIELDS = {
    "schema_version",
    "artifact_role",
    "identity",
    "normalization_sha256",
    "row_count",
    "dimension",
    "conventions",
    "task_design",
}
_CONVENTION_FIELDS = {
    "input",
    "scalar_target",
    "gradient_target",
    "adapter_unit_cube_transform",
    "adapter_target_standardization",
    "energy_sign_flip",
    "joint_value_gradient_scaling",
}


class ExternalAdapterError(RuntimeError):
    """Raised when an external task is incomplete or violates its contract."""


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _is_real_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def _require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ExternalAdapterError(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _require_hex(value: str, pattern: re.Pattern[str], label: str) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ExternalAdapterError(f"{label} is not a lowercase hexadecimal digest")


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ExternalAdapterError("value is not canonical-JSON compatible") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _strict_json_object(value: str, label: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ExternalAdapterError(f"duplicate JSON key in {label}: {key}")
            result[key] = item
        return result

    try:
        parsed = json.loads(
            value,
            object_pairs_hook=unique_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ExternalAdapterError(f"nonfinite JSON constant in {label}: {item}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ExternalAdapterError(f"invalid JSON in {label}") from error
    if not isinstance(parsed, dict):
        raise ExternalAdapterError(f"{label} must be a JSON object")
    return parsed


def _array_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asarray(value), allow_pickle=False)
    return buffer.getvalue()


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an NPZ with stable member order and timestamps, refusing overwrite."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw_handle = path.open("xb")
    except FileExistsError as error:
        raise ExternalAdapterError(f"refusing to overwrite artifact: {path}") from error
    try:
        with (
            raw_handle,
            zipfile.ZipFile(
                raw_handle,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            ) as archive,
        ):
            for name in sorted(arrays):
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, _array_bytes(arrays[name]), compresslevel=9)
    except Exception:
        # A partial file is invalid and should remain visible for forensic inspection.
        raise


def _frozen_array(value: Any) -> np.ndarray:
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


def _require_finite_array(value: np.ndarray, shape: tuple[int, ...], label: str) -> None:
    if value.shape != shape:
        raise ExternalAdapterError(f"{label} has shape {value.shape}, expected {shape}")
    if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
        raise ExternalAdapterError(f"{label} must be a finite floating array")


def _require_index_array(value: np.ndarray, rows: int, label: str) -> None:
    if value.shape != (rows,) or not np.issubdtype(value.dtype, np.integer):
        raise ExternalAdapterError(f"{label} must be a length-{rows} integer array")


def _require_trajectory_time_design(
    trajectory_ids: np.ndarray,
    time_indices: np.ndarray,
    expected_times: tuple[int, ...],
    expected_trajectory_count: int,
    label: str,
) -> None:
    trajectories = np.unique(trajectory_ids)
    if (
        trajectories.size != expected_trajectory_count
        or np.any(trajectories < 0)
        or len(trajectory_ids) != expected_trajectory_count * len(expected_times)
    ):
        raise ExternalAdapterError(f"{label} trajectory accounting does not match F02")
    expected = np.asarray(expected_times, dtype=np.int64)
    for trajectory in trajectories:
        observed = time_indices[trajectory_ids == trajectory]
        if not np.array_equal(observed, expected):
            raise ExternalAdapterError(
                f"{label} trajectory {int(trajectory)} does not contain the frozen time design"
            )


@dataclass(frozen=True, slots=True)
class CorpusIdentity:
    """Catalog-bound identity for one fixed-mass F02 corpus."""

    bundle_id: str
    catalog_task_index: int
    replica: int
    n_particles: int
    dimension: int
    phase: str
    dataset_content_sha256: str
    catalog_sha256: str

    def __post_init__(self) -> None:
        if not _is_plain_int(self.n_particles) or self.n_particles not in F02_PARTICLE_COUNTS:
            raise ValueError("n_particles is outside the F02 grid")
        if (
            not _is_plain_int(self.dimension)
            or self.dimension != 6 * self.n_particles
            or self.dimension not in F02_DIMENSIONS
        ):
            raise ValueError("dimension must equal 6*n_particles in the F02 grid")
        if not _is_plain_int(self.replica):
            raise ValueError("replica must be an integer")
        expected_phase = "development" if self.replica in {0, 1, 2} else "confirmatory"
        if self.replica not in F02_REPLICAS:
            raise ValueError("replica is outside the F02 grid")
        if self.phase != expected_phase:
            raise ValueError(f"replica {self.replica} requires phase {expected_phase!r}")
        expected_bundle_id = f"replica-{self.replica}-n-{self.n_particles}-d-{self.dimension}"
        if self.bundle_id != expected_bundle_id:
            raise ValueError(f"bundle_id must equal {expected_bundle_id!r}")
        expected_task_index = F02_REPLICAS.index(self.replica) * len(
            F02_PARTICLE_COUNTS
        ) + F02_PARTICLE_COUNTS.index(self.n_particles)
        if (
            not _is_plain_int(self.catalog_task_index)
            or self.catalog_task_index != expected_task_index
        ):
            raise ValueError(f"catalog_task_index must equal {expected_task_index}")
        if _HEX_64.fullmatch(self.dataset_content_sha256) is None:
            raise ValueError("dataset_content_sha256 must be a lowercase SHA-256 digest")
        if _HEX_64.fullmatch(self.catalog_sha256) is None:
            raise ValueError("catalog_sha256 must be a lowercase SHA-256 digest")


def canonical_normalization_sha256(
    *,
    x_min: np.ndarray,
    x_span: np.ndarray,
    energy_mean: float,
    energy_std: float,
    gradient_scale: np.ndarray,
) -> str:
    """Identify the exact train-derived canonical normalization without JSON rounding."""

    arrays = {
        "energy_mean": np.asarray(float(energy_mean), dtype=np.float64),
        "energy_std": np.asarray(float(energy_std), dtype=np.float64),
        "gradient_scale": np.asarray(gradient_scale),
        "x_min": np.asarray(x_min),
        "x_span": np.asarray(x_span),
    }
    digest = hashlib.sha256()
    for name in sorted(arrays):
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(_array_bytes(arrays[name]))
    return digest.hexdigest()


def _source_ids(identity: CorpusIdentity, source_indices: np.ndarray) -> np.ndarray:
    return np.asarray(
        [f"{identity.bundle_id}:{int(index)}" for index in source_indices],
        dtype=np.str_,
    )


def _artifact_metadata(
    identity: CorpusIdentity,
    normalization_sha256: str,
    role: str,
    rows: int,
    task_design: dict[str, Any],
) -> dict[str, Any]:
    _require_hex(normalization_sha256, _HEX_64, "normalization_sha256")
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_role": role,
        "identity": asdict(identity),
        "normalization_sha256": normalization_sha256,
        "row_count": rows,
        "dimension": identity.dimension,
        "task_design": task_design,
        "conventions": {
            "input": "canonical_train_min_span_standardized_x",
            "scalar_target": "standardized_positive_H_without_sign_flip",
            "gradient_target": "standardized_dH_dx_in_canonical_x_coordinates",
            "adapter_unit_cube_transform": False,
            "adapter_target_standardization": False,
            "energy_sign_flip": False,
            "joint_value_gradient_scaling": False,
        },
    }


@dataclass(frozen=True, slots=True)
class ExternalArtifactPaths:
    """Paths created together, with a deliberately narrower worker view."""

    training: Path
    evaluation_features: Path
    evaluation_labels_central_only: Path

    def worker_inputs(self) -> WorkerArtifactPaths:
        return WorkerArtifactPaths(
            training=self.training,
            evaluation_features=self.evaluation_features,
        )


@dataclass(frozen=True, slots=True)
class WorkerArtifactPaths:
    """The only artifact paths admitted at the external worker boundary."""

    training: Path
    evaluation_features: Path


def _validate_split_for_export(
    split: PreparedConfirmatorySplit,
    identity: CorpusIdentity,
) -> None:
    rows = len(split.source_indices)
    dimension = identity.dimension
    _require_finite_array(np.asarray(split.X), (rows, dimension), f"{split.name}.X")
    _require_finite_array(np.asarray(split.E), (rows,), f"{split.name}.E")
    _require_finite_array(np.asarray(split.F), (rows, dimension), f"{split.name}.F")
    for name in ("source_indices", "trajectory_id", "time_index"):
        _require_index_array(np.asarray(getattr(split, name)), rows, f"{split.name}.{name}")
    indices = np.asarray(split.source_indices)
    if rows == 0 or np.any(indices < 0) or np.any(np.diff(indices) <= 0):
        raise ExternalAdapterError(
            f"{split.name}.source_indices must be nonempty, non-negative, and strictly increasing"
        )


def write_external_artifact_bundle(
    output_directory: str | Path,
    *,
    identity: CorpusIdentity,
    normalization_sha256: str,
    training: PreparedConfirmatorySplit,
    evaluation: PreparedConfirmatorySplit,
    evaluation_design: str = "primary",
) -> ExternalArtifactPaths:
    """Write canonical train/features/central-label artifacts for one task.

    ``training`` must be the 1,500-state F02 design.  Evaluation labels are
    written to a separately named central-only artifact and are never included
    in :class:`WorkerArtifactPaths`.
    """

    if training.name != "train":
        raise ExternalAdapterError("training artifact requires split name 'train'")
    if evaluation.name not in {"validation", "test"}:
        raise ExternalAdapterError("evaluation split must be 'validation' or 'test'")
    expected_phase = "development" if evaluation.name == "validation" else "confirmatory"
    if identity.phase != expected_phase:
        raise ExternalAdapterError(
            f"evaluation split {evaluation.name!r} requires phase {expected_phase!r}"
        )
    _validate_split_for_export(training, identity)
    _validate_split_for_export(evaluation, identity)
    if len(training.source_indices) != TRAIN_ROWS:
        raise ExternalAdapterError(f"F02 external training requires exactly {TRAIN_ROWS} rows")
    if np.intersect1d(training.source_indices, evaluation.source_indices).size:
        raise ExternalAdapterError("training and evaluation source rows overlap")
    evaluation_indices = {
        "primary": EVALUATION_TIME_INDICES,
        "optimizer_selection": OPTIMIZER_SELECTION_TIME_INDICES,
    }.get(evaluation_design)
    if evaluation_indices is None:
        raise ExternalAdapterError("evaluation_design must be 'primary' or 'optimizer_selection'")
    if tuple(np.unique(training.time_index).tolist()) != TRAIN_TIME_INDICES:
        raise ExternalAdapterError("training artifact does not use the frozen F02 time design")
    if tuple(np.unique(evaluation.time_index).tolist()) != evaluation_indices:
        raise ExternalAdapterError("evaluation artifact does not use its declared F02 time design")
    _require_trajectory_time_design(
        np.asarray(training.trajectory_id),
        np.asarray(training.time_index),
        TRAIN_TIME_INDICES,
        60,
        "training",
    )
    _require_trajectory_time_design(
        np.asarray(evaluation.trajectory_id),
        np.asarray(evaluation.time_index),
        evaluation_indices,
        20,
        "evaluation",
    )

    task_design = {
        "training_time_indices": list(TRAIN_TIME_INDICES),
        "evaluation_split": evaluation.name,
        "evaluation_design": evaluation_design,
        "evaluation_time_indices": list(evaluation_indices),
    }

    output = Path(output_directory)
    train_ids = _source_ids(identity, np.asarray(training.source_indices))
    evaluation_ids = _source_ids(identity, np.asarray(evaluation.source_indices))
    common_train = {
        "source_ids": train_ids,
        "source_indices": np.asarray(training.source_indices, dtype=np.int64),
        "trajectory_ids": np.asarray(training.trajectory_id, dtype=np.int64),
        "time_indices": np.asarray(training.time_index, dtype=np.int64),
    }
    common_evaluation = {
        "source_ids": evaluation_ids,
        "source_indices": np.asarray(evaluation.source_indices, dtype=np.int64),
        "trajectory_ids": np.asarray(evaluation.trajectory_id, dtype=np.int64),
        "time_indices": np.asarray(evaluation.time_index, dtype=np.int64),
    }
    training_metadata = _artifact_metadata(
        identity,
        normalization_sha256,
        "training_with_labels",
        len(train_ids),
        task_design,
    )
    feature_metadata = _artifact_metadata(
        identity,
        normalization_sha256,
        "evaluation_features_without_labels",
        len(evaluation_ids),
        task_design,
    )
    label_metadata = _artifact_metadata(
        identity,
        normalization_sha256,
        "evaluation_labels_central_only",
        len(evaluation_ids),
        task_design,
    )

    training_path = output / "training.canonical.npz"
    feature_path = output / "evaluation.features_only.npz"
    label_path = output / "evaluation.labels.central_only.npz"
    _write_deterministic_npz(
        training_path,
        {
            "metadata_json": np.asarray(_canonical_json_bytes(training_metadata).decode()),
            "X_standardized": np.asarray(training.X),
            "H_standardized": np.asarray(training.E),
            "dH_dx_standardized": np.asarray(training.F),
            **common_train,
        },
    )
    _write_deterministic_npz(
        feature_path,
        {
            "metadata_json": np.asarray(_canonical_json_bytes(feature_metadata).decode()),
            "X_standardized": np.asarray(evaluation.X),
            **common_evaluation,
        },
    )
    _write_deterministic_npz(
        label_path,
        {
            "metadata_json": np.asarray(_canonical_json_bytes(label_metadata).decode()),
            "H_standardized": np.asarray(evaluation.E),
            "dH_dx_standardized": np.asarray(evaluation.F),
            **common_evaluation,
        },
    )
    return ExternalArtifactPaths(
        training=training_path,
        evaluation_features=feature_path,
        evaluation_labels_central_only=label_path,
    )


@dataclass(frozen=True, slots=True)
class CanonicalTrainingArtifact:
    metadata: dict[str, Any]
    X_standardized: np.ndarray
    H_standardized: np.ndarray
    dH_dx_standardized: np.ndarray
    source_ids: tuple[str, ...]
    source_indices: np.ndarray
    trajectory_ids: np.ndarray
    time_indices: np.ndarray
    artifact_sha256: str


@dataclass(frozen=True, slots=True)
class CanonicalEvaluationFeatures:
    """Label-free worker input.  Slots prevent accidental dynamic label fields."""

    metadata: dict[str, Any]
    X_standardized: np.ndarray
    source_ids: tuple[str, ...]
    source_indices: np.ndarray
    trajectory_ids: np.ndarray
    time_indices: np.ndarray
    artifact_sha256: str


@dataclass(frozen=True, slots=True)
class CentralEvaluationLabels:
    metadata: dict[str, Any]
    H_standardized: np.ndarray
    dH_dx_standardized: np.ndarray
    source_ids: tuple[str, ...]
    source_indices: np.ndarray
    trajectory_ids: np.ndarray
    time_indices: np.ndarray
    artifact_sha256: str


def _load_record(path: Path, expected_fields: set[str], label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as record:
            actual = set(record.files)
            if actual != expected_fields:
                raise ExternalAdapterError(
                    f"{label} archive fields differ: missing={sorted(expected_fields - actual)}, "
                    f"unexpected={sorted(actual - expected_fields)}"
                )
            return {name: _frozen_array(record[name]) for name in expected_fields}
    except (OSError, ValueError) as error:
        if isinstance(error, ExternalAdapterError):
            raise
        raise ExternalAdapterError(f"cannot load {label}: {path}") from error


def _parse_artifact(
    path: Path,
    expected_fields: set[str],
    expected_role: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray], tuple[str, ...]]:
    arrays = _load_record(path, expected_fields, expected_role)
    raw_metadata = arrays.pop("metadata_json")
    if raw_metadata.shape != () or not np.issubdtype(raw_metadata.dtype, np.str_):
        raise ExternalAdapterError("artifact metadata_json must be a scalar string")
    metadata = _strict_json_object(str(raw_metadata.item()), "artifact metadata_json")
    _require_exact_keys(metadata, _COMMON_METADATA_FIELDS, "artifact metadata")
    if metadata["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ExternalAdapterError("unsupported external artifact schema_version")
    if metadata["artifact_role"] != expected_role:
        raise ExternalAdapterError("external artifact role mismatch")
    try:
        identity = CorpusIdentity(**metadata["identity"])
    except (TypeError, ValueError) as error:
        raise ExternalAdapterError("invalid corpus identity in artifact metadata") from error
    _require_hex(metadata["normalization_sha256"], _HEX_64, "normalization_sha256")
    conventions = metadata["conventions"]
    if not isinstance(conventions, dict):
        raise ExternalAdapterError("artifact conventions must be an object")
    _require_exact_keys(conventions, _CONVENTION_FIELDS, "artifact conventions")
    expected_conventions = _artifact_metadata(
        identity,
        metadata["normalization_sha256"],
        expected_role,
        metadata["row_count"],
        metadata["task_design"],
    )["conventions"]
    if conventions != expected_conventions:
        raise ExternalAdapterError("artifact conventions are not canonical F02 conventions")
    rows = metadata["row_count"]
    dimension = metadata["dimension"]
    if (
        not _is_plain_int(rows)
        or rows <= 0
        or not _is_plain_int(dimension)
        or dimension != identity.dimension
    ):
        raise ExternalAdapterError("artifact row count or dimension is invalid")
    for name in ("source_indices", "trajectory_ids", "time_indices"):
        _require_index_array(arrays[name], rows, name)
    task_design = metadata["task_design"]
    if not isinstance(task_design, dict):
        raise ExternalAdapterError("artifact task_design must be an object")
    _require_exact_keys(
        task_design,
        {
            "training_time_indices",
            "evaluation_split",
            "evaluation_design",
            "evaluation_time_indices",
        },
        "artifact task_design",
    )
    expected_evaluation_indices = {
        "primary": EVALUATION_TIME_INDICES,
        "optimizer_selection": OPTIMIZER_SELECTION_TIME_INDICES,
    }.get(task_design["evaluation_design"])
    expected_split = "validation" if identity.phase == "development" else "test"
    if (
        task_design["training_time_indices"] != list(TRAIN_TIME_INDICES)
        or expected_evaluation_indices is None
        or task_design["evaluation_time_indices"] != list(expected_evaluation_indices)
        or task_design["evaluation_split"] != expected_split
        or (identity.phase == "confirmatory" and task_design["evaluation_design"] != "primary")
    ):
        raise ExternalAdapterError("artifact task_design is not a registered F02 design")
    expected_indices = (
        TRAIN_TIME_INDICES
        if expected_role == "training_with_labels"
        else expected_evaluation_indices
    )
    if tuple(np.unique(arrays["time_indices"]).tolist()) != expected_indices:
        raise ExternalAdapterError("artifact time indices do not match task_design")
    _require_trajectory_time_design(
        arrays["trajectory_ids"],
        arrays["time_indices"],
        expected_indices,
        60 if expected_role == "training_with_labels" else 20,
        expected_role,
    )
    if np.any(arrays["source_indices"] < 0) or np.any(np.diff(arrays["source_indices"]) <= 0):
        raise ExternalAdapterError("artifact source_indices are not canonical")
    raw_ids = arrays["source_ids"]
    if raw_ids.shape != (rows,) or not np.issubdtype(raw_ids.dtype, np.str_):
        raise ExternalAdapterError("artifact source_ids must be a one-dimensional string array")
    source_ids = tuple(str(value) for value in raw_ids.tolist())
    expected_ids = tuple(_source_ids(identity, arrays["source_indices"]).tolist())
    if source_ids != expected_ids or len(set(source_ids)) != rows:
        raise ExternalAdapterError("artifact source_ids are not exact, ordered, and unique")
    return metadata, arrays, source_ids


def load_training_artifact(path: str | Path) -> CanonicalTrainingArtifact:
    resolved = Path(path)
    metadata, arrays, source_ids = _parse_artifact(
        resolved,
        _TRAIN_FIELDS,
        "training_with_labels",
    )
    rows, dimension = metadata["row_count"], metadata["dimension"]
    if rows != TRAIN_ROWS:
        raise ExternalAdapterError(f"F02 external training requires exactly {TRAIN_ROWS} rows")
    _require_finite_array(arrays["X_standardized"], (rows, dimension), "X_standardized")
    _require_finite_array(arrays["H_standardized"], (rows,), "H_standardized")
    _require_finite_array(
        arrays["dH_dx_standardized"],
        (rows, dimension),
        "dH_dx_standardized",
    )
    return CanonicalTrainingArtifact(
        metadata=metadata,
        X_standardized=arrays["X_standardized"],
        H_standardized=arrays["H_standardized"],
        dH_dx_standardized=arrays["dH_dx_standardized"],
        source_ids=source_ids,
        source_indices=arrays["source_indices"],
        trajectory_ids=arrays["trajectory_ids"],
        time_indices=arrays["time_indices"],
        artifact_sha256=sha256_file(resolved),
    )


def load_evaluation_features(path: str | Path) -> CanonicalEvaluationFeatures:
    resolved = Path(path)
    metadata, arrays, source_ids = _parse_artifact(
        resolved,
        _FEATURE_FIELDS,
        "evaluation_features_without_labels",
    )
    rows, dimension = metadata["row_count"], metadata["dimension"]
    _require_finite_array(arrays["X_standardized"], (rows, dimension), "X_standardized")
    return CanonicalEvaluationFeatures(
        metadata=metadata,
        X_standardized=arrays["X_standardized"],
        source_ids=source_ids,
        source_indices=arrays["source_indices"],
        trajectory_ids=arrays["trajectory_ids"],
        time_indices=arrays["time_indices"],
        artifact_sha256=sha256_file(resolved),
    )


def load_central_evaluation_labels(path: str | Path) -> CentralEvaluationLabels:
    resolved = Path(path)
    metadata, arrays, source_ids = _parse_artifact(
        resolved,
        _LABEL_FIELDS,
        "evaluation_labels_central_only",
    )
    rows, dimension = metadata["row_count"], metadata["dimension"]
    _require_finite_array(arrays["H_standardized"], (rows,), "H_standardized")
    _require_finite_array(
        arrays["dH_dx_standardized"],
        (rows, dimension),
        "dH_dx_standardized",
    )
    return CentralEvaluationLabels(
        metadata=metadata,
        H_standardized=arrays["H_standardized"],
        dH_dx_standardized=arrays["dH_dx_standardized"],
        source_ids=source_ids,
        source_indices=arrays["source_indices"],
        trajectory_ids=arrays["trajectory_ids"],
        time_indices=arrays["time_indices"],
        artifact_sha256=sha256_file(resolved),
    )


@dataclass(frozen=True, slots=True)
class ExternalBaselineConfig:
    """One fully specified released external training opportunity."""

    method_id: str
    dimension: int
    seed: int
    selected_updates: int
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        if self.method_id not in METHOD_IDS:
            raise ValueError(f"method_id must be one of {METHOD_IDS}")
        if self.dimension not in F02_DIMENSIONS:
            raise ValueError(f"dimension must be one of {F02_DIMENSIONS}")
        if self.seed not in OPTIMIZER_SEEDS:
            raise ValueError(f"seed must be one of {OPTIMIZER_SEEDS}")
        if self.selected_updates not in OPTIMIZER_UPDATE_CANDIDATES:
            raise ValueError(f"selected_updates must be one of {OPTIMIZER_UPDATE_CANDIDATES}")
        if re.fullmatch(r"cuda(?::\d+)?", self.device) is None:
            raise ValueError("released external tasks require an explicit CUDA device")

    @property
    def epochs(self) -> int:
        return self.selected_updates // UPDATES_PER_EPOCH

    def to_payload(self) -> dict[str, Any]:
        shared_training = {
            "seed": self.seed,
            "batch_size": TRAIN_BATCH_SIZE,
            "epochs": self.epochs,
            "selected_logical_optimizer_updates": self.selected_updates,
            "expected_updates_per_epoch": UPDATES_PER_EPOCH,
            "train_rows": TRAIN_ROWS,
            "expected_states_processed": self.epochs * TRAIN_ROWS,
            "dataloader_workers": 0,
            "test_dataset_passed_to_train": False,
            "in_training_evaluation": False,
            "curve_log_every": 0,
            "wandb_enabled": False,
        }
        preprocessing = {
            "consume_canonical_arrays_directly": True,
            "adapter_unit_cube_transform": False,
            "adapter_target_standardization": False,
            "energy_sign_flip": False,
            "joint_value_gradient_scaling": False,
        }
        if self.method_id == "dsoftki-512":
            model = {
                "released_entrypoint": "gp.dsoft_ki.train.train_gp",
                "kernel_target": "RBFKernel",
                "kernel_nu_cli_field": 1.5,
                "kernel_lengthscale_initial": 1.0,
                "use_ard": False,
                "use_scale": False,
                "num_interp": 512,
                "interp_init": "kmeans",
                "fit_chunk_size": 256,
                "value_noise_initial_variance": 0.1,
                "derivative_noise_initial_variance": 0.1 * self.dimension,
                "learn_noise": True,
                "solver": "cg",
                "cg_tolerance": 1e-5,
                "max_cg_iterations": 50,
                "mll_approx": "hutchinson_fallback",
                "per_interp_T": True,
                "min_T": 5e-5,
                "learn_T": True,
                "use_qr": True,
                "embed_dim": -1,
                "hidden_dim": 64,
                "use_dot": True,
                "grad_only": False,
                "dtype": "float32",
                "device": self.device,
                "fit_device": self.device,
            }
            training = {**shared_training, "learning_rate": 0.02}
            released_reference = {
                "source": "exp/run_nbody.sh",
                "num_interp": 512,
                "batch_size": 1024,
                "epochs": 50,
                "learning_rate": 0.02,
                "value_noise_initial_variance": 0.1,
                "derivative_noise_initial_variance_rule": "0.1 * dimension",
                "use_ard": True,
                "use_scale": False,
                "dataset_standardize": True,
                "dataset_unit_cube": True,
            }
            protocol_overrides = {
                "epochs": "selected from {10,25,50} to match {20,50,100} logical updates",
                "optimizer_seeds": list(OPTIMIZER_SEEDS),
                "use_ard": False,
                "dataset_standardize": False,
                "dataset_unit_cube": False,
                "reason": "F02 isotropic-RBF and canonical-array protocol",
            }
        else:
            model = {
                "released_entrypoint": "gp.ddsvgp.train.train_gp",
                "kernel_target": "RBFKernelDirectionalGrad",
                "kernel_lengthscale_initial": 1.0,
                "use_ard": False,
                "use_scale": True,
                "num_inducing": 512,
                "induce_init": "kmeans",
                "num_directions": 2,
                "value_and_derivative_noise_initial_variance": 0.1,
                "learn_noise": True,
                "mll_type": "PLL",
                "learn_inducing_locations": True,
                "dtype": "float32",
                "device": self.device,
            }
            training = {
                **shared_training,
                "learning_rate": 0.03,
                "optimizer_step_calls_per_logical_update": 2,
                "directional_coordinates_per_logical_update": 2,
                "lr_scheduler": None,
                "scheduler_gamma": 0.1,
            }
            released_reference = {
                "source": "exp/run_nbody.sh",
                "num_inducing": 512,
                "batch_size": 1024,
                "epochs": 50,
                "learning_rate": 0.03,
                "noise_initial_variance": 0.1,
                "num_directions": 2,
                "mll_type": "PLL",
                "use_ard": False,
                "use_scale": True,
                "dataset_standardize": True,
                "dataset_unit_cube": True,
            }
            protocol_overrides = {
                "epochs": "selected from {10,25,50} to match {20,50,100} logical updates",
                "optimizer_seeds": list(OPTIMIZER_SEEDS),
                "dataset_standardize": False,
                "dataset_unit_cube": False,
                "reason": "F02 canonical-array protocol",
            }
        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "method_id": self.method_id,
            "dimension": self.dimension,
            "vendor_commit": VENDOR_COMMIT,
            "vendor_source_tree": VENDOR_SOURCE_TREE,
            "model": model,
            "training": training,
            "preprocessing": preprocessing,
            "released_nbody_reference": released_reference,
            "f02_protocol_overrides": protocol_overrides,
            "variance_contract": _variance_contract(self.method_id),
        }

    @property
    def sha256(self) -> str:
        return _json_sha256(self.to_payload())


def validate_external_config_payload(payload: dict[str, Any]) -> ExternalBaselineConfig:
    if not isinstance(payload, dict):
        raise ExternalAdapterError("external config payload must be an object")
    try:
        method_id = payload["method_id"]
        dimension = payload["dimension"]
        training = payload["training"]
        model = payload["model"]
        if not isinstance(training, dict) or not isinstance(model, dict):
            raise TypeError
        config = ExternalBaselineConfig(
            method_id=method_id,
            dimension=dimension,
            seed=training["seed"],
            selected_updates=training["selected_logical_optimizer_updates"],
            device=model["device"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ExternalAdapterError("external config payload identity is invalid") from error
    if payload != config.to_payload():
        raise ExternalAdapterError("external config differs from the frozen released/F02 recipe")
    return config


def _variance_contract(method_id: str) -> dict[str, Any]:
    if method_id == "dsoftki-512":
        return {
            "native_variance_kind": "released_noise_floored_predictive_covariance",
            "native_includes_observation_noise": "not_decomposable_noise_floor",
            "common_latent_verified": False,
            "common_latent_fixture_sha256": None,
            "eligible_for_common_latent_nll": False,
        }
    if method_id == "ddsvgp-512":
        return {
            "native_variance_kind": "released_likelihood_observation_predictive_variance",
            "native_includes_observation_noise": True,
            "common_latent_verified": False,
            "common_latent_fixture_sha256": None,
            "eligible_for_common_latent_nll": False,
        }
    raise ExternalAdapterError(f"unknown external method_id: {method_id}")


def seed_external_runtime(seed: int) -> dict[str, Any]:
    """Seed every RNG used by the released runners and enforce serial loaders."""

    if seed not in OPTIMIZER_SEEDS:
        raise ExternalAdapterError(f"seed must be one of {OPTIMIZER_SEEDS}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return {
        "python_random_seed": seed,
        "numpy_seed": seed,
        "torch_cpu_seed": seed,
        "torch_cuda_all_seed": seed,
        "dataloader_workers": 0,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }


@dataclass(frozen=True, slots=True)
class TrainingAccounting:
    epochs_completed: int
    logical_update_attempts: int
    logical_updates_applied: int
    skipped_nonfinite_updates: int
    optimizer_step_calls: int
    states_processed: int
    posterior_fit_solve_passes: int
    posterior_fit_pseudoinverse_passes: int
    directional_coordinate_draw_count: int
    directional_coordinate_sequence_sha256: str | None
    fit_seconds_descriptive: float
    fit_peak_gpu_allocated_bytes: int


def validate_training_accounting(
    config: ExternalBaselineConfig,
    accounting: TrainingAccounting,
) -> None:
    count_fields = (
        "epochs_completed",
        "logical_update_attempts",
        "logical_updates_applied",
        "skipped_nonfinite_updates",
        "optimizer_step_calls",
        "states_processed",
        "posterior_fit_solve_passes",
        "posterior_fit_pseudoinverse_passes",
        "directional_coordinate_draw_count",
        "fit_peak_gpu_allocated_bytes",
    )
    if any(
        not _is_plain_int(getattr(accounting, name)) or getattr(accounting, name) < 0
        for name in count_fields
    ):
        raise ExternalAdapterError("training accounting counts must be non-negative integers")
    if accounting.epochs_completed != config.epochs:
        raise ExternalAdapterError("training did not complete the selected epoch budget")
    if accounting.logical_update_attempts != config.selected_updates:
        raise ExternalAdapterError("logical optimizer-update attempts do not match the budget")
    if (
        accounting.logical_updates_applied + accounting.skipped_nonfinite_updates
        != accounting.logical_update_attempts
    ):
        raise ExternalAdapterError("applied/skipped optimizer update accounting does not close")
    if accounting.skipped_nonfinite_updates != 0:
        raise ExternalAdapterError("a complete external task cannot hide skipped nonfinite updates")
    if accounting.states_processed != config.epochs * TRAIN_ROWS:
        raise ExternalAdapterError("states_processed does not match complete epoch traversal")
    if (
        not _is_real_number(accounting.fit_seconds_descriptive)
        or not math.isfinite(accounting.fit_seconds_descriptive)
        or accounting.fit_seconds_descriptive < 0.0
    ):
        raise ExternalAdapterError("training timing or peak allocation is invalid")
    if config.method_id == "dsoftki-512":
        if accounting.optimizer_step_calls != config.selected_updates:
            raise ExternalAdapterError("DSoftKI must apply one Adam step per logical update")
        if accounting.posterior_fit_solve_passes != config.epochs:
            raise ExternalAdapterError("DSoftKI must record one posterior fit solve per epoch")
        if not 0 <= accounting.posterior_fit_pseudoinverse_passes <= config.epochs:
            raise ExternalAdapterError("DSoftKI pseudoinverse pass count is invalid")
        if (
            accounting.directional_coordinate_draw_count != 0
            or accounting.directional_coordinate_sequence_sha256 is not None
        ):
            raise ExternalAdapterError("DSoftKI must not report DDSVGP direction sampling")
    else:
        if accounting.optimizer_step_calls != 2 * config.selected_updates:
            raise ExternalAdapterError("DDSVGP has two optimizer.step calls per logical update")
        if accounting.posterior_fit_solve_passes != 0:
            raise ExternalAdapterError("DDSVGP must not report DSoftKI posterior fit solves")
        if accounting.posterior_fit_pseudoinverse_passes != 0:
            raise ExternalAdapterError("DDSVGP must not report pseudoinverse fit passes")
        if accounting.directional_coordinate_draw_count != 2 * config.selected_updates:
            raise ExternalAdapterError("DDSVGP direction draw count is incomplete")
        digest = accounting.directional_coordinate_sequence_sha256
        if not isinstance(digest, str) or _HEX_64.fullmatch(digest) is None:
            raise ExternalAdapterError("DDSVGP direction sequence SHA-256 is missing or invalid")


@dataclass(frozen=True, slots=True)
class ExternalPredictions:
    source_ids: tuple[str, ...]
    mean_standardized_H: np.ndarray
    native_variance_standardized_H: np.ndarray
    gradient_standardized_dH_dx: np.ndarray | None
    observation_noise_variance: float
    prediction_seconds_descriptive: float
    prediction_peak_gpu_allocated_bytes: int


class ExternalBackend(Protocol):
    """Backend boundary intentionally contains no evaluation-label parameter."""

    method_id: str
    is_test_double: bool

    def train(
        self,
        training: CanonicalTrainingArtifact,
        config: ExternalBaselineConfig,
    ) -> tuple[Any, TrainingAccounting]: ...

    def predict(
        self,
        model: Any,
        evaluation: CanonicalEvaluationFeatures,
        config: ExternalBaselineConfig,
    ) -> ExternalPredictions: ...


def _assert_exclusive_slurm_environment(environment: dict[str, str]) -> None:
    required = {
        "SLURM_JOB_ID": None,
        "F02_EXCLUSIVE_ALLOCATION_VERIFIED": "1",
        "F02_EXTERNAL_DEPENDENCIES_VERIFIED": "1",
    }
    for name, exact in required.items():
        value = environment.get(name)
        if not value or (exact is not None and value != exact):
            raise ExternalAdapterError(
                "released external backends require a dependency-verified exclusive Slurm allocation"
            )


def _same_artifact_identity(
    training: CanonicalTrainingArtifact,
    evaluation: CanonicalEvaluationFeatures,
) -> None:
    for field in ("identity", "normalization_sha256", "dimension", "task_design"):
        if training.metadata[field] != evaluation.metadata[field]:
            raise ExternalAdapterError(f"training/evaluation artifact {field} mismatch")
    if set(training.source_ids).intersection(evaluation.source_ids):
        raise ExternalAdapterError("training and evaluation source IDs overlap")


def _validate_predictions(
    config: ExternalBaselineConfig,
    evaluation: CanonicalEvaluationFeatures,
    predictions: ExternalPredictions,
) -> None:
    rows = len(evaluation.source_ids)
    if predictions.source_ids != evaluation.source_ids:
        raise ExternalAdapterError("prediction source IDs do not exactly match evaluation features")
    _require_finite_array(
        np.asarray(predictions.mean_standardized_H),
        (rows,),
        "mean_standardized_H",
    )
    native_variance = np.asarray(predictions.native_variance_standardized_H)
    _require_finite_array(native_variance, (rows,), "native_variance_standardized_H")
    if np.any(native_variance <= 0.0):
        raise ExternalAdapterError("native predictive variance must be raw, finite, and positive")
    if predictions.gradient_standardized_dH_dx is not None:
        _require_finite_array(
            np.asarray(predictions.gradient_standardized_dH_dx),
            (rows, config.dimension),
            "gradient_standardized_dH_dx",
        )
    if (
        not _is_real_number(predictions.observation_noise_variance)
        or not math.isfinite(predictions.observation_noise_variance)
        or predictions.observation_noise_variance <= 0.0
    ):
        raise ExternalAdapterError("observation noise variance must be finite and positive")
    if (
        not _is_real_number(predictions.prediction_seconds_descriptive)
        or not math.isfinite(predictions.prediction_seconds_descriptive)
        or predictions.prediction_seconds_descriptive < 0.0
        or not _is_plain_int(predictions.prediction_peak_gpu_allocated_bytes)
        or predictions.prediction_peak_gpu_allocated_bytes < 0
    ):
        raise ExternalAdapterError("prediction timing or peak allocation is invalid")


def _task_identity(
    config: ExternalBaselineConfig,
    evaluation: CanonicalEvaluationFeatures,
) -> dict[str, Any]:
    identity = CorpusIdentity(**evaluation.metadata["identity"])
    design = evaluation.metadata["task_design"]
    return {
        "method_id": config.method_id,
        "bundle_id": identity.bundle_id,
        "catalog_task_index": identity.catalog_task_index,
        "replica": identity.replica,
        "n_particles": identity.n_particles,
        "dimension": identity.dimension,
        "seed": config.seed,
        "optimizer_updates": config.selected_updates,
        "evaluation_split": design["evaluation_split"],
        "evaluation_design": design["evaluation_design"],
    }


def build_external_result(
    *,
    config: ExternalBaselineConfig,
    training: CanonicalTrainingArtifact,
    evaluation: CanonicalEvaluationFeatures,
    accounting: TrainingAccounting,
    predictions: ExternalPredictions,
    seed_report: dict[str, Any],
    repo_commit: str,
    repo_tree: str,
    dependency_lock_sha256: str,
) -> dict[str, Any]:
    """Build a prediction-only result; evaluation labels and metrics are forbidden."""

    _same_artifact_identity(training, evaluation)
    if evaluation.metadata["dimension"] != config.dimension:
        raise ExternalAdapterError("config dimension does not match artifact dimension")
    validate_training_accounting(config, accounting)
    _validate_predictions(config, evaluation, predictions)
    _require_hex(repo_commit, _HEX_40, "repo_commit")
    _require_hex(repo_tree, _HEX_40, "repo_tree")
    _require_hex(dependency_lock_sha256, _HEX_64, "dependency_lock_sha256")
    expected_seed_report = {
        "python_random_seed": config.seed,
        "numpy_seed": config.seed,
        "torch_cpu_seed": config.seed,
        "torch_cuda_all_seed": config.seed,
        "dataloader_workers": 0,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
    if seed_report != expected_seed_report:
        raise ExternalAdapterError("runtime seed report is incomplete or inconsistent")
    variance = {
        **_variance_contract(config.method_id),
        "canonical_standardized_units": True,
        "observation_noise_variance": float(predictions.observation_noise_variance),
        "native_variance_raw_finite_positive": True,
        "native_variance_floor_variance": (
            float(predictions.observation_noise_variance)
            if config.method_id == "dsoftki-512"
            else None
        ),
        "common_latent_variance": None,
    }
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete",
        "task_identity": _task_identity(config, evaluation),
        "config": config.to_payload(),
        "config_sha256": config.sha256,
        "sources": {
            "training_artifact_sha256": training.artifact_sha256,
            "evaluation_features_artifact_sha256": evaluation.artifact_sha256,
            "training_source_ids_sha256": _json_sha256(list(training.source_ids)),
            "evaluation_source_ids_sha256": _json_sha256(list(evaluation.source_ids)),
            "dataset_content_sha256": evaluation.metadata["identity"]["dataset_content_sha256"],
            "catalog_sha256": evaluation.metadata["identity"]["catalog_sha256"],
            "normalization_sha256": evaluation.metadata["normalization_sha256"],
        },
        "training_accounting": asdict(accounting),
        "runtime_seeding": seed_report,
        "predictions": {
            "source_ids": list(predictions.source_ids),
            "mean_standardized_H": np.asarray(predictions.mean_standardized_H).tolist(),
            "native_variance_standardized_H": np.asarray(
                predictions.native_variance_standardized_H
            ).tolist(),
            "gradient_standardized_dH_dx": (
                None
                if predictions.gradient_standardized_dH_dx is None
                else np.asarray(predictions.gradient_standardized_dH_dx).tolist()
            ),
        },
        "variance_semantics": variance,
        "runtime": {
            "fit_seconds_descriptive": accounting.fit_seconds_descriptive,
            "fit_peak_gpu_allocated_bytes": accounting.fit_peak_gpu_allocated_bytes,
            "prediction_seconds_descriptive": predictions.prediction_seconds_descriptive,
            "prediction_peak_gpu_allocated_bytes": (
                predictions.prediction_peak_gpu_allocated_bytes
            ),
        },
        "provenance": {
            "repo_commit": repo_commit,
            "repo_tree": repo_tree,
            "vendor_commit": VENDOR_COMMIT,
            "vendor_source_tree": VENDOR_SOURCE_TREE,
            "dependency_lock_sha256": dependency_lock_sha256,
            "f02_protocol_sha256": F02_PROTOCOL_SHA256,
        },
    }
    # This also rejects NaN/Infinity before the result reaches disk.
    _canonical_json_bytes(result)
    validate_external_result(result)
    return result


def validate_external_result(result: dict[str, Any]) -> None:
    """Strictly validate a complete, prediction-only external result."""

    root_fields = {
        "schema_version",
        "status",
        "task_identity",
        "config",
        "config_sha256",
        "sources",
        "training_accounting",
        "runtime_seeding",
        "predictions",
        "variance_semantics",
        "runtime",
        "provenance",
    }
    _require_exact_keys(result, root_fields, "external result")
    if result["schema_version"] != RESULT_SCHEMA_VERSION or result["status"] != "complete":
        raise ExternalAdapterError("external result is not a complete supported result")
    config = validate_external_config_payload(result["config"])
    if result["config_sha256"] != config.sha256:
        raise ExternalAdapterError("external result config SHA-256 mismatch")

    task = result["task_identity"]
    if not isinstance(task, dict):
        raise ExternalAdapterError("task_identity must be an object")
    _require_exact_keys(
        task,
        {
            "method_id",
            "bundle_id",
            "catalog_task_index",
            "replica",
            "n_particles",
            "dimension",
            "seed",
            "optimizer_updates",
            "evaluation_split",
            "evaluation_design",
        },
        "task_identity",
    )
    if (
        task["method_id"] != config.method_id
        or task["dimension"] != config.dimension
        or task["seed"] != config.seed
        or task["optimizer_updates"] != config.selected_updates
        or task["evaluation_design"] not in {"primary", "optimizer_selection"}
    ):
        raise ExternalAdapterError("task identity does not match external config")
    try:
        identity = CorpusIdentity(
            bundle_id=task["bundle_id"],
            catalog_task_index=task["catalog_task_index"],
            replica=task["replica"],
            n_particles=task["n_particles"],
            dimension=task["dimension"],
            phase="development" if task["evaluation_split"] == "validation" else "confirmatory",
            dataset_content_sha256=result["sources"]["dataset_content_sha256"],
            catalog_sha256=result["sources"]["catalog_sha256"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ExternalAdapterError("task/source corpus identity is invalid") from error
    expected_split = "validation" if identity.phase == "development" else "test"
    if task["evaluation_split"] != expected_split:
        raise ExternalAdapterError("evaluation split does not match corpus phase")
    if identity.phase == "confirmatory" and task["evaluation_design"] != "primary":
        raise ExternalAdapterError("confirmatory external tasks require the primary design")

    sources = result["sources"]
    if not isinstance(sources, dict):
        raise ExternalAdapterError("sources must be an object")
    _require_exact_keys(
        sources,
        {
            "training_artifact_sha256",
            "evaluation_features_artifact_sha256",
            "training_source_ids_sha256",
            "evaluation_source_ids_sha256",
            "dataset_content_sha256",
            "catalog_sha256",
            "normalization_sha256",
        },
        "sources",
    )
    for name, value in sources.items():
        _require_hex(value, _HEX_64, f"sources.{name}")

    accounting_payload = result["training_accounting"]
    if not isinstance(accounting_payload, dict):
        raise ExternalAdapterError("training_accounting must be an object")
    _require_exact_keys(
        accounting_payload,
        {field.name for field in fields(TrainingAccounting)},
        "training_accounting",
    )
    try:
        accounting = TrainingAccounting(**accounting_payload)
    except TypeError as error:
        raise ExternalAdapterError("training_accounting has invalid types") from error
    validate_training_accounting(config, accounting)

    predictions = result["predictions"]
    if not isinstance(predictions, dict):
        raise ExternalAdapterError("predictions must be an object")
    _require_exact_keys(
        predictions,
        {
            "source_ids",
            "mean_standardized_H",
            "native_variance_standardized_H",
            "gradient_standardized_dH_dx",
        },
        "predictions",
    )
    source_ids = predictions["source_ids"]
    if (
        not isinstance(source_ids, list)
        or not source_ids
        or not all(isinstance(value, str) and value for value in source_ids)
        or len(set(source_ids)) != len(source_ids)
    ):
        raise ExternalAdapterError("prediction source_ids are invalid or duplicated")
    if _json_sha256(source_ids) != sources["evaluation_source_ids_sha256"]:
        raise ExternalAdapterError("prediction source IDs do not match their bound hash")
    rows = len(source_ids)
    _require_finite_array(
        np.asarray(predictions["mean_standardized_H"]),
        (rows,),
        "predictions.mean_standardized_H",
    )
    native_variance = np.asarray(predictions["native_variance_standardized_H"])
    _require_finite_array(
        native_variance,
        (rows,),
        "predictions.native_variance_standardized_H",
    )
    if np.any(native_variance <= 0.0):
        raise ExternalAdapterError("prediction native variance is not strictly positive")
    gradient = predictions["gradient_standardized_dH_dx"]
    if gradient is not None:
        _require_finite_array(
            np.asarray(gradient),
            (rows, config.dimension),
            "predictions.gradient_standardized_dH_dx",
        )

    variance = result["variance_semantics"]
    if not isinstance(variance, dict):
        raise ExternalAdapterError("variance_semantics must be an object")
    expected_variance = {
        **_variance_contract(config.method_id),
        "canonical_standardized_units": True,
        "observation_noise_variance": variance.get("observation_noise_variance"),
        "native_variance_raw_finite_positive": True,
        "native_variance_floor_variance": (
            variance.get("observation_noise_variance")
            if config.method_id == "dsoftki-512"
            else None
        ),
        "common_latent_variance": None,
    }
    if variance != expected_variance:
        raise ExternalAdapterError(
            "variance semantics are incomplete or claim an unverified common latent variance"
        )
    observation_noise = variance["observation_noise_variance"]
    if (
        not _is_real_number(observation_noise)
        or not math.isfinite(observation_noise)
        or observation_noise <= 0.0
    ):
        raise ExternalAdapterError("observation_noise_variance must be finite and positive")

    runtime_seed = result["runtime_seeding"]
    expected_seed = {
        "python_random_seed": config.seed,
        "numpy_seed": config.seed,
        "torch_cpu_seed": config.seed,
        "torch_cuda_all_seed": config.seed,
        "dataloader_workers": 0,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
    if runtime_seed != expected_seed:
        raise ExternalAdapterError("runtime_seeding is incomplete or inconsistent")

    provenance = result["provenance"]
    if not isinstance(provenance, dict):
        raise ExternalAdapterError("provenance must be an object")
    _require_exact_keys(
        provenance,
        {
            "repo_commit",
            "repo_tree",
            "vendor_commit",
            "vendor_source_tree",
            "dependency_lock_sha256",
            "f02_protocol_sha256",
        },
        "provenance",
    )
    for name in ("repo_commit", "repo_tree", "vendor_commit", "vendor_source_tree"):
        _require_hex(provenance[name], _HEX_40, f"provenance.{name}")
    for name in ("dependency_lock_sha256", "f02_protocol_sha256"):
        _require_hex(provenance[name], _HEX_64, f"provenance.{name}")
    if (
        provenance["vendor_commit"] != VENDOR_COMMIT
        or provenance["vendor_source_tree"] != VENDOR_SOURCE_TREE
        or provenance["f02_protocol_sha256"] != F02_PROTOCOL_SHA256
    ):
        raise ExternalAdapterError(
            "external result source provenance differs from the frozen source"
        )
    runtime = result["runtime"]
    if not isinstance(runtime, dict):
        raise ExternalAdapterError("runtime must be an object")
    _require_exact_keys(
        runtime,
        {
            "fit_seconds_descriptive",
            "fit_peak_gpu_allocated_bytes",
            "prediction_seconds_descriptive",
            "prediction_peak_gpu_allocated_bytes",
        },
        "runtime",
    )
    if (
        runtime["fit_seconds_descriptive"] != accounting.fit_seconds_descriptive
        or runtime["fit_peak_gpu_allocated_bytes"] != accounting.fit_peak_gpu_allocated_bytes
    ):
        raise ExternalAdapterError("runtime fit accounting does not match training_accounting")
    for name in ("fit_seconds_descriptive", "prediction_seconds_descriptive"):
        value = runtime[name]
        if not _is_real_number(value) or not math.isfinite(value) or value < 0.0:
            raise ExternalAdapterError(f"runtime.{name} must be finite and non-negative")
    for name in (
        "fit_peak_gpu_allocated_bytes",
        "prediction_peak_gpu_allocated_bytes",
    ):
        value = runtime[name]
        if not _is_plain_int(value) or value < 0:
            raise ExternalAdapterError(f"runtime.{name} must be a non-negative integer")
    _canonical_json_bytes(result)


def save_external_result(result: dict[str, Any], path: str | Path) -> None:
    """Write one validated result atomically with exclusive-create semantics."""

    validate_external_result(result)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    try:
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise ExternalAdapterError(
            f"refusing to overwrite external result: {destination}"
        ) from error


def load_external_result(path: str | Path) -> dict[str, Any]:
    """Load a strict JSON result and re-run the full schema validation."""

    resolved = Path(path)
    try:
        result = _strict_json_object(resolved.read_text(encoding="utf-8"), "external result")
    except (OSError, UnicodeError) as error:
        raise ExternalAdapterError(f"cannot read external result: {resolved}") from error
    validate_external_result(result)
    return result


def run_external_worker(
    *,
    artifacts: WorkerArtifactPaths,
    config: ExternalBaselineConfig,
    backend: ExternalBackend,
    repo_commit: str,
    repo_tree: str,
    dependency_lock_sha256: str,
    test_only: bool = False,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run one backend through the label-isolated contract.

    Production calls are refused unless the Slurm harness explicitly attests an
    exclusive, dependency-verified allocation.  ``test_only`` is accepted only
    for a backend declaring itself a test double.
    """

    if backend.method_id != config.method_id:
        raise ExternalAdapterError("backend method_id does not match the task config")
    if test_only:
        if backend.is_test_double is not True:
            raise ExternalAdapterError("test_only execution requires an explicit test double")
    else:
        _assert_exclusive_slurm_environment(
            dict(os.environ if environment is None else environment)
        )
    training = load_training_artifact(artifacts.training)
    evaluation = load_evaluation_features(artifacts.evaluation_features)
    _same_artifact_identity(training, evaluation)
    seed_report = seed_external_runtime(config.seed)
    model, accounting = backend.train(training, config)
    predictions = backend.predict(model, evaluation, config)
    return build_external_result(
        config=config,
        training=training,
        evaluation=evaluation,
        accounting=accounting,
        predictions=predictions,
        seed_report=seed_report,
        repo_commit=repo_commit,
        repo_tree=repo_tree,
        dependency_lock_sha256=dependency_lock_sha256,
    )


def join_central_labels(
    result: dict[str, Any],
    labels: CentralEvaluationLabels,
) -> tuple[np.ndarray, np.ndarray]:
    """Central-only source-ID join performed after worker output validation."""

    validate_external_result(result)
    prediction_ids = tuple(result["predictions"]["source_ids"])
    if prediction_ids != labels.source_ids:
        raise ExternalAdapterError("prediction and central-label source IDs do not exactly match")
    task = result["task_identity"]
    identity = labels.metadata["identity"]
    if (
        task["bundle_id"] != identity["bundle_id"]
        or task["catalog_task_index"] != identity["catalog_task_index"]
        or result["sources"]["normalization_sha256"] != labels.metadata["normalization_sha256"]
    ):
        raise ExternalAdapterError("central labels do not belong to this external result")
    return labels.H_standardized, labels.dH_dx_standardized
