"""Fail-closed loading and deterministic preparation of confirmatory N-body data.

The generator writes three mutually constraining artifacts: an NPZ record, a
human-readable metadata document, and a SHA-256 manifest.  This module treats
all three as part of the dataset identity.  It never creates a new split and
never drops rows based on energy or gradient labels.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from data.generate_nbody_confirmatory import (
    SCHEMA_VERSION,
    ConfirmatoryConfig,
    ConfirmatoryDataset,
    DataIntegrityError,
    SplitManifest,
    TrainNormalization,
    validate_dataset,
    verify_sha256_manifest,
)

_SPLIT_NAMES = ("train", "validation", "test")
_REQUIRED_NPZ_FIELDS = {
    "schema_version",
    "config_json",
    "validation_json",
    "X",
    "E",
    "F",
    "masses",
    "trajectory_id",
    "time_index",
    "time_value",
    "train_indices",
    "validation_indices",
    "test_indices",
    "train_trajectory_ids",
    "validation_trajectory_ids",
    "test_trajectory_ids",
    "x_train_min",
    "x_train_span",
    "energy_train_mean",
    "energy_train_std",
    "gradient_scale",
}


@dataclass(frozen=True, slots=True)
class BundleProvenance:
    """Paths, hashes, and source documents defining one frozen bundle."""

    dataset_path: Path
    metadata_path: Path
    sha256_manifest_path: Path
    file_sha256: dict[str, str]
    metadata: dict[str, object]
    config_payload: dict[str, object]
    validation_payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class LoadedConfirmatoryBundle:
    """Validated raw data together with its preserved provenance."""

    dataset: ConfirmatoryDataset
    validation: dict[str, float | int]
    provenance: BundleProvenance


@dataclass(frozen=True, slots=True)
class PreparedConfirmatorySplit:
    """One normalized split in canonical source-row order."""

    name: str
    source_indices: np.ndarray
    X: np.ndarray
    E: np.ndarray
    F: np.ndarray
    trajectory_id: np.ndarray
    time_index: np.ndarray
    time_value: np.ndarray


@dataclass(frozen=True, slots=True)
class PreparedConfirmatoryDataset:
    """Normalized train/validation/test views without row selection."""

    train: PreparedConfirmatorySplit
    validation: PreparedConfirmatorySplit
    test: PreparedConfirmatorySplit
    normalization: TrainNormalization
    masses: np.ndarray

    def split(self, name: str) -> PreparedConfirmatorySplit:
        if name not in _SPLIT_NAMES:
            raise KeyError(name)
        return getattr(self, name)


@dataclass(frozen=True, slots=True)
class PreparedConfirmatoryBundle:
    """Prepared data retaining the exact loaded bundle and provenance."""

    loaded: LoadedConfirmatoryBundle
    prepared: PreparedConfirmatoryDataset


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DataIntegrityError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise DataIntegrityError(f"{label} must contain a JSON object")
    return value


def _parse_json_object(value: str, label: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise DataIntegrityError(f"invalid {label} in NPZ record") from error
    if not isinstance(parsed, dict):
        raise DataIntegrityError(f"{label} in NPZ record must be a JSON object")
    return parsed


def _scalar_text(record: np.lib.npyio.NpzFile, name: str) -> str:
    value = np.asarray(record[name])
    if value.shape != ():
        raise DataIntegrityError(f"{name} must be a scalar string")
    return str(value.item())


def _frozen_array(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


def _require_array_kinds(arrays: dict[str, np.ndarray]) -> None:
    floating = (
        "X",
        "E",
        "F",
        "masses",
        "time_value",
        "x_train_min",
        "x_train_span",
        "energy_train_mean",
        "energy_train_std",
        "gradient_scale",
    )
    integral = (
        "trajectory_id",
        "time_index",
        "train_indices",
        "validation_indices",
        "test_indices",
        "train_trajectory_ids",
        "validation_trajectory_ids",
        "test_trajectory_ids",
    )
    for name in floating:
        if not np.issubdtype(arrays[name].dtype, np.floating):
            raise DataIntegrityError(f"{name} must have a floating dtype")
    for name in integral:
        if arrays[name].ndim != 1 or not np.issubdtype(arrays[name].dtype, np.integer):
            raise DataIntegrityError(f"{name} must be a one-dimensional integer array")
    for name in ("energy_train_mean", "energy_train_std"):
        if arrays[name].shape != ():
            raise DataIntegrityError(f"{name} must be a scalar")


def _require_canonical_split_order(splits: SplitManifest) -> None:
    for split in _SPLIT_NAMES:
        indices = splits.indices(split)
        trajectory_ids = splits.trajectory_ids(split)
        if indices.size == 0 or trajectory_ids.size == 0:
            raise DataIntegrityError(f"{split} split must be nonempty")
        if np.any(np.diff(indices) <= 0):
            raise DataIntegrityError(f"{split} sample indices are not strictly increasing")
        if np.any(np.diff(trajectory_ids) <= 0):
            raise DataIntegrityError(f"{split} trajectory IDs are not strictly increasing")


def _load_npz_dataset(
    dataset_path: Path,
) -> tuple[
    ConfirmatoryDataset,
    dict[str, object],
    dict[str, object],
    dict[str, float | int],
]:
    try:
        with np.load(dataset_path, allow_pickle=False) as record:
            missing = _REQUIRED_NPZ_FIELDS - set(record.files)
            if missing:
                raise DataIntegrityError(f"NPZ record is missing fields: {sorted(missing)}")
            if _scalar_text(record, "schema_version") != SCHEMA_VERSION:
                raise DataIntegrityError("unsupported NPZ schema_version")
            config_payload = _parse_json_object(
                _scalar_text(record, "config_json"),
                "config_json",
            )
            validation_payload = _parse_json_object(
                _scalar_text(record, "validation_json"),
                "validation_json",
            )
            arrays = {
                name: _frozen_array(np.asarray(record[name]))
                for name in _REQUIRED_NPZ_FIELDS
                if name not in {"schema_version", "config_json", "validation_json"}
            }
    except (OSError, ValueError) as error:
        if isinstance(error, DataIntegrityError):
            raise
        raise DataIntegrityError(f"cannot load NPZ record: {dataset_path}") from error

    try:
        config = ConfirmatoryConfig(**config_payload)
        config.validate()
    except (TypeError, ValueError) as error:
        raise DataIntegrityError("invalid confirmatory config in NPZ record") from error
    if _canonical_json(config_payload) != _canonical_json(asdict(config)):
        raise DataIntegrityError("NPZ config does not exactly match the schema")

    _require_array_kinds(arrays)
    splits = SplitManifest(
        train_trajectory_ids=arrays["train_trajectory_ids"],
        validation_trajectory_ids=arrays["validation_trajectory_ids"],
        test_trajectory_ids=arrays["test_trajectory_ids"],
        train_indices=arrays["train_indices"],
        validation_indices=arrays["validation_indices"],
        test_indices=arrays["test_indices"],
    )
    _require_canonical_split_order(splits)
    normalization = TrainNormalization(
        x_min=arrays["x_train_min"],
        x_span=arrays["x_train_span"],
        energy_mean=float(np.asarray(arrays["energy_train_mean"])),
        energy_std=float(np.asarray(arrays["energy_train_std"])),
        gradient_scale=arrays["gradient_scale"],
    )
    dataset = ConfirmatoryDataset(
        config=config,
        X=arrays["X"],
        E=arrays["E"],
        F=arrays["F"],
        masses=arrays["masses"],
        trajectory_id=arrays["trajectory_id"],
        time_index=arrays["time_index"],
        time_value=arrays["time_value"],
        splits=splits,
        normalization=normalization,
    )
    validation = validate_dataset(dataset)
    if _canonical_json(validation_payload) != _canonical_json(validation):
        raise DataIntegrityError("stored validation report does not match loaded data")
    return dataset, config_payload, validation_payload, validation


def _mapping_field(parent: dict[str, object], name: str, label: str) -> dict[str, object]:
    value = parent.get(name)
    if not isinstance(value, dict):
        raise DataIntegrityError(f"metadata {label}.{name} must be an object")
    return value


def _assert_metadata_matches(
    metadata: dict[str, object],
    dataset_path: Path,
    dataset: ConfirmatoryDataset,
    config_payload: dict[str, object],
    validation_payload: dict[str, object],
) -> None:
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise DataIntegrityError("unsupported metadata schema_version")
    if metadata.get("dataset_file") != dataset_path.name:
        raise DataIntegrityError("metadata dataset_file does not match the NPZ filename")
    if _canonical_json(metadata.get("config")) != _canonical_json(config_payload):
        raise DataIntegrityError("metadata config does not match the NPZ config")
    if not np.array_equal(np.asarray(metadata.get("masses")), dataset.masses):
        raise DataIntegrityError("metadata masses do not match the NPZ masses")

    array_metadata = _mapping_field(metadata, "arrays", "root")
    for name in ("X", "E", "F"):
        descriptor = _mapping_field(array_metadata, name, "arrays")
        expected = {
            "shape": list(getattr(dataset, name).shape),
            "dtype": str(getattr(dataset, name).dtype),
        }
        if descriptor != expected:
            raise DataIntegrityError(f"metadata descriptor for {name} does not match the NPZ")

    split_metadata = _mapping_field(metadata, "splits", "root")
    for split in _SPLIT_NAMES:
        descriptor = _mapping_field(split_metadata, split, "splits")
        expected = {
            "trajectory_ids": dataset.splits.trajectory_ids(split).tolist(),
            "n_samples": int(dataset.splits.indices(split).size),
        }
        if descriptor != expected:
            raise DataIntegrityError(f"metadata {split} split does not match the NPZ")

    normalization = _mapping_field(metadata, "normalization", "root")
    expected_normalization = {
        "source_split": "train",
        "x_min": dataset.normalization.x_min.tolist(),
        "x_span": dataset.normalization.x_span.tolist(),
        "energy_mean": dataset.normalization.energy_mean,
        "energy_std": dataset.normalization.energy_std,
        "gradient_scale": dataset.normalization.gradient_scale.tolist(),
    }
    if normalization != expected_normalization:
        raise DataIntegrityError("metadata normalization does not match the NPZ")
    if _canonical_json(metadata.get("validation")) != _canonical_json(validation_payload):
        raise DataIntegrityError("metadata validation report does not match the NPZ")


def load_confirmatory_bundle(dataset_path: str | Path) -> LoadedConfirmatoryBundle:
    """Load and semantically validate one generated confirmatory bundle.

    Sidecars are inferred from ``<stem>.npz`` as ``<stem>.metadata.json`` and
    ``<stem>.sha256.json``.  Hashes are checked before any content is trusted,
    then NPZ, metadata, config, split, normalization, and validation records
    are checked against one another.
    """

    dataset_path = Path(dataset_path)
    if dataset_path.suffix != ".npz":
        raise ValueError("dataset_path must name a .npz artifact")
    metadata_path = dataset_path.with_suffix(".metadata.json")
    manifest_path = dataset_path.with_suffix(".sha256.json")

    manifest = _read_json_object(manifest_path, "SHA-256 manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise DataIntegrityError("unsupported SHA-256 manifest schema_version")
    actual_hashes = verify_sha256_manifest(manifest_path)
    expected_files = {dataset_path.name, metadata_path.name}
    if set(actual_hashes) != expected_files:
        raise DataIntegrityError("SHA-256 manifest must cover exactly the NPZ and metadata files")

    metadata = _read_json_object(metadata_path, "metadata")
    dataset, config_payload, validation_payload, validation = _load_npz_dataset(dataset_path)
    _assert_metadata_matches(
        metadata,
        dataset_path,
        dataset,
        config_payload,
        validation_payload,
    )
    return LoadedConfirmatoryBundle(
        dataset=dataset,
        validation=validation,
        provenance=BundleProvenance(
            dataset_path=dataset_path,
            metadata_path=metadata_path,
            sha256_manifest_path=manifest_path,
            file_sha256=dict(actual_hashes),
            metadata=metadata,
            config_payload=config_payload,
            validation_payload=validation_payload,
        ),
    )


def _prepare_split(dataset: ConfirmatoryDataset, name: str) -> PreparedConfirmatorySplit:
    indices = dataset.splits.indices(name)
    normalization = dataset.normalization
    X = (dataset.X[indices] - normalization.x_min) / normalization.x_span
    E = (dataset.E[indices] - normalization.energy_mean) / normalization.energy_std
    F = dataset.F[indices] * normalization.gradient_scale
    for label, value in (("X", X), ("E", E), ("F", F)):
        if not np.isfinite(value).all():
            raise DataIntegrityError(f"normalization produced nonfinite {label} in {name} split")
    return PreparedConfirmatorySplit(
        name=name,
        source_indices=_frozen_array(indices),
        X=_frozen_array(X),
        E=_frozen_array(E),
        F=_frozen_array(F),
        trajectory_id=_frozen_array(dataset.trajectory_id[indices]),
        time_index=_frozen_array(dataset.time_index[indices]),
        time_value=_frozen_array(dataset.time_value[indices]),
    )


def prepare_confirmatory_dataset(dataset: ConfirmatoryDataset) -> PreparedConfirmatoryDataset:
    """Apply persisted train-only normalization to every stored sample.

    The stored split manifests are authoritative.  Preparation preserves their
    row order and cardinality; nonfinite transformations fail rather than being
    filtered out.
    """

    _require_canonical_split_order(dataset.splits)
    prepared = PreparedConfirmatoryDataset(
        train=_prepare_split(dataset, "train"),
        validation=_prepare_split(dataset, "validation"),
        test=_prepare_split(dataset, "test"),
        normalization=dataset.normalization,
        masses=dataset.masses,
    )
    combined = np.concatenate(
        [prepared.split(split).source_indices for split in _SPLIT_NAMES]
    )
    if not np.array_equal(np.sort(combined), np.arange(dataset.X.shape[0])):
        raise DataIntegrityError("prepared splits do not preserve every source row exactly once")
    return prepared


def load_prepared_confirmatory_bundle(dataset_path: str | Path) -> PreparedConfirmatoryBundle:
    """Load, verify, and normalize a bundle in one deterministic operation."""

    loaded = load_confirmatory_bundle(dataset_path)
    return PreparedConfirmatoryBundle(
        loaded=loaded,
        prepared=prepare_confirmatory_dataset(loaded.dataset),
    )
