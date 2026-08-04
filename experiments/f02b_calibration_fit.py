"""Create strict, development-only F02b TERA fit artifacts.

The CLI resolves science coordinates only through the immutable fit-task grid.
The authoritative bundle loader may read the complete single-file bundle for
integrity validation, but this runner tensorizes and passes only the frozen
training split to the fit callable.  It performs no prediction or scoring.

Each successful task creates one identity-derived directory containing:

* a canonical JSON numeric payload with learned parameters and provenance; and
* the public F02b execution envelope binding that payload's path and raw bytes.

Neither file contains executable model state, a pickle, an optimizer-selection
decision, or authority to freeze or inspect confirmatory labels.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import re
import stat
import struct
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from cluster.check_python_environment import audit_distributions
from cluster.f02b_calibration_grid import F02bCalibrationFitTask, fit_task_for_index
from data.generate_nbody_confirmatory import ConfirmatoryConfig
from data.load_nbody_confirmatory import load_prepared_confirmatory_bundle
from experiments.f02_design import TRAIN_TIME_INDICES, select_time_indices
from experiments.f02_internal_models import (
    FrozenTERAParameters,
    TensorConfirmatorySplit,
    fit_released_tera,
    freeze_tera_parameters,
)
from experiments.f02_internal_task import (
    BundleIdentity,
    InternalTaskConfig,
    _preflight_bundle_identity,
    validate_catalog_identity,
)
from experiments.f02b_calibration_contract import (
    F02_CATALOG_SHA256,
    FIT_RECIPE,
    FIT_RECIPE_SHA256,
    FIT_TASK_MATRIX_SHA256,
    MINIMUM_HOST_MEMORY_BYTES,
    NUMERICAL_POLICY,
    WALLTIME_SECONDS,
    build_fit_execution_envelope,
    build_payload_binding,
    canonical_json_bytes,
    parse_strict_json_bytes,
    validate_fit_execution_envelope,
    validate_fit_recipe,
    validate_runtime_allocation,
    verify_numeric_payload_bytes,
)

FIT_PAYLOAD_SCHEMA_VERSION = "f02b_calibration_fit_payload_v1"
FIT_PAYLOAD_TYPE = "f02b_development_fit_parameters"
EXCLUSIVE_VERIFICATION_MODE = "scontrol_show_job_oversubscribe_exclusive_v1"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_PAYLOAD_FIELDS = {
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
_PARAMETER_FIELDS = {
    "gradient_noise_model",
    "kernel",
    "lengthscale",
    "outputscale",
    "sigma_f",
    "sigma_g",
}
_FULL_CONFIG_FIELDS = {
    "batch_size",
    "candidate_m",
    "cg_max_iterations",
    "cg_tolerance",
    "device",
    "dtype",
    "function_jitter",
    "graph_refresh_epochs",
    "kernel",
    "learn_lengthscale",
    "learn_outputscale",
    "learn_sigma_f",
    "learn_sigma_g",
    "lengthscale",
    "lengthscale_init",
    "lengthscale_init_max_points",
    "lr",
    "min_sigma_f",
    "min_sigma_g",
    "outputscale",
    "reduced_jitter",
    "seed",
    "sigma_f",
    "sigma_g",
    "train_epochs",
    "train_steps",
    "training_m",
    "use_ard",
    "use_preconditioner",
    "weight_decay",
}
_FIT_CALL_RECIPE_FIELDS = (
    "training_m",
    "train_steps",
    "train_epochs",
    "kernel",
    "outputscale",
    "sigma_f",
    "sigma_g",
    "lengthscale",
    "lengthscale_init",
    "lengthscale_init_max_points",
    "use_ard",
    "batch_size",
    "lr",
    "weight_decay",
    "graph_refresh_epochs",
    "learn_lengthscale",
    "learn_outputscale",
    "learn_sigma_f",
    "learn_sigma_g",
    "min_sigma_f",
    "min_sigma_g",
)
_CAPABILITIES = ("fit_parameters_for_f02b_numerical_calibration",)


class CalibrationFitError(RuntimeError):
    """Raised when a calibration fit or artifact fails closed."""


@dataclass(frozen=True, slots=True)
class FitInputs:
    """Validated identity evidence and the sole tensorized training split."""

    identity: Mapping[str, Any]
    train: Any


@dataclass(frozen=True, slots=True)
class FitArtifactPaths:
    """Identity-derived paths created for one successful fit task."""

    directory: Path
    numeric_payload: Path
    execution_envelope: Path


@dataclass(frozen=True, slots=True)
class _CorpusSnapshot:
    """Private byte-for-byte copies used for both authorization and loading."""

    dataset_path: Path
    catalog_path: Path


FitCallable = Callable[[Any, InternalTaskConfig], Mapping[str, Any] | FrozenTERAParameters]
IdentityProvider = Callable[
    [F02bCalibrationFitTask, InternalTaskConfig, Path, Path, Path],
    FitInputs,
]
PostflightValidator = Callable[
    [
        F02bCalibrationFitTask,
        InternalTaskConfig,
        Path,
        Path,
        Path,
        Mapping[str, Any],
    ],
    None,
]


def _fresh_json(value: Any, label: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except Exception as error:
        raise CalibrationFitError(f"{label} must be finite strict JSON") from error


def _data_access_payload() -> dict[str, Any]:
    return {
        "integrity_loading": "authoritative_complete_bundle",
        "tensorized_splits": ["train"],
        "fit_callable_splits": ["train"],
        "training_time_indices": list(TRAIN_TIME_INDICES),
        "prediction_or_scoring_performed": False,
    }


def _capabilities_payload() -> list[str]:
    return list(_CAPABILITIES)


def _strict_json_file(path: Path, label: str) -> Any:
    try:
        return parse_strict_json_bytes(path.read_bytes())
    except Exception as error:
        raise CalibrationFitError(f"cannot read strict JSON {label}: {path}") from error


def _read_regular_file_once(path: Path, label: str) -> bytes:
    """Read one stable regular-file inode without following its final path component."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:  # pragma: no cover - the registered launcher is Linux-only
        raise CalibrationFitError("this runner requires O_NOFOLLOW support")
    flags = os.O_RDONLY | os.O_CLOEXEC | nofollow
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CalibrationFitError(f"cannot snapshot required {label}: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CalibrationFitError(f"required {label} is not a regular file: {path}")
        chunks = bytearray()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.extend(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise CalibrationFitError(f"cannot snapshot required {label}: {path}") from error
    finally:
        os.close(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
        raise CalibrationFitError(f"required {label} changed while it was snapshotted: {path}")
    if len(chunks) != before.st_size:
        raise CalibrationFitError(f"required {label} size changed while it was snapshotted: {path}")
    return bytes(chunks)


def _write_private_snapshot(path: Path, content: bytes, label: str) -> None:
    """Materialize captured bytes once in a private directory with no overwrite."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags, 0o400)
        try:
            view = memoryview(content)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:  # pragma: no cover - defensive short-write guard
                    raise OSError("snapshot write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise CalibrationFitError(f"cannot materialize private {label} snapshot") from error
    if _read_regular_file_once(path, f"materialized {label}") != content:
        raise CalibrationFitError(f"private {label} snapshot is not byte-identical")


@contextmanager
def _immutable_corpus_snapshot(
    dataset_path: Path,
    catalog_path: Path,
) -> Iterator[_CorpusSnapshot]:
    """Capture corpus inputs once, then expose only private byte-identical paths."""

    dataset = dataset_path.resolve()
    catalog = catalog_path.resolve()
    metadata = dataset.with_suffix(".metadata.json")
    manifest = dataset.with_suffix(".sha256.json")
    captured = {
        "dataset": _read_regular_file_once(dataset, "bundle dataset"),
        "metadata": _read_regular_file_once(metadata, "bundle metadata"),
        "manifest": _read_regular_file_once(manifest, "bundle manifest"),
        "catalog": _read_regular_file_once(catalog, "catalog"),
    }
    for name in ("metadata", "manifest", "catalog"):
        try:
            parsed = parse_strict_json_bytes(captured[name])
        except Exception as error:
            raise CalibrationFitError(f"captured {name} must be strict JSON") from error
        if not isinstance(parsed, dict):
            raise CalibrationFitError(f"captured {name} must contain a JSON object")

    with tempfile.TemporaryDirectory(prefix="f02b-fit-input-") as temporary:
        snapshot_root = Path(temporary)
        bundle_root = snapshot_root / "bundle"
        catalog_root = snapshot_root / "catalog"
        bundle_root.mkdir(mode=0o700)
        catalog_root.mkdir(mode=0o700)
        snapshot_dataset = bundle_root / dataset.name
        snapshot_metadata = snapshot_dataset.with_suffix(".metadata.json")
        snapshot_manifest = snapshot_dataset.with_suffix(".sha256.json")
        snapshot_catalog = catalog_root / "catalog.json"
        for path, content, label in (
            (snapshot_dataset, captured["dataset"], "bundle dataset"),
            (snapshot_metadata, captured["metadata"], "bundle metadata"),
            (snapshot_manifest, captured["manifest"], "bundle manifest"),
            (snapshot_catalog, captured["catalog"], "catalog"),
        ):
            _write_private_snapshot(path, content, label)
        yield _CorpusSnapshot(
            dataset_path=snapshot_dataset,
            catalog_path=snapshot_catalog,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise CalibrationFitError(f"cannot hash required file: {path}") from error
    return digest.hexdigest()


def _require_exact_keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CalibrationFitError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise CalibrationFitError(
            f"{label} fields mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return value


def _require_text(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise CalibrationFitError(f"{label} must be a nonempty string")
    return value


def _require_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise CalibrationFitError(f"{label} must be an integer, not bool")
    return value


def _require_real(value: object, label: str) -> float:
    if type(value) not in {int, float}:
        raise CalibrationFitError(f"{label} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise CalibrationFitError(f"{label} must be finite")
    return result


def _require_binary32(value: object, label: str) -> float:
    if type(value) is not float:
        raise CalibrationFitError(f"{label} must be a JSON float")
    result = value
    if not math.isfinite(result):
        raise CalibrationFitError(f"{label} must be finite")
    if result == 0.0 and math.copysign(1.0, result) < 0.0:
        raise CalibrationFitError(f"{label} must not be negative zero")
    try:
        binary32 = struct.unpack("!f", struct.pack("!f", result))[0]
    except (OverflowError, struct.error) as error:
        raise CalibrationFitError(f"{label} is outside binary32 range") from error
    if binary32 != result:
        raise CalibrationFitError(f"{label} is not exactly binary32-representable")
    return result


def _to_binary32(value: object, label: str) -> float:
    result = _require_real(value, label)
    try:
        return struct.unpack("!f", struct.pack("!f", result))[0]
    except (OverflowError, struct.error) as error:
        raise CalibrationFitError(f"{label} is outside binary32 range") from error


def _require_sha256(value: object, label: str) -> str:
    result = _require_text(value, label)
    if _HEX64.fullmatch(result) is None:
        raise CalibrationFitError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _require_git_oid(value: object, label: str) -> str:
    result = _require_text(value, label)
    if _HEX40.fullmatch(result) is None:
        raise CalibrationFitError(f"{label} must be a full lowercase SHA-1 object ID")
    return result


def _task(task_index: int) -> F02bCalibrationFitTask:
    if type(task_index) is not int:
        raise CalibrationFitError("task index must be an integer, not bool")
    try:
        return fit_task_for_index(task_index)
    except (IndexError, TypeError) as error:
        raise CalibrationFitError("task index is outside the immutable fit grid") from error


def _fit_config(task: F02bCalibrationFitTask) -> InternalTaskConfig:
    """Adapt the public fit recipe to the complete current InternalTaskConfig."""

    recipe = validate_fit_recipe(FIT_RECIPE)
    return InternalTaskConfig(
        **recipe,
        seed=task.seed,
        # These prediction-only fields are inert during fit.  They are explicit
        # here so an upstream default change cannot silently alter this runner.
        candidate_m=(),
        cg_tolerance=1e-5,
        cg_max_iterations=None,
        use_preconditioner=True,
        function_jitter=1e-8,
        reduced_jitter=1e-8,
    )


def _config_payload(task: F02bCalibrationFitTask) -> dict[str, Any]:
    payload = _fresh_json(asdict(_fit_config(task)), "InternalTaskConfig")
    config = _require_exact_keys(payload, _FULL_CONFIG_FIELDS, "internal_task_config")
    projected = {name: config[name] for name in FIT_RECIPE}
    validate_fit_recipe(projected)
    if config["seed"] != task.seed:
        raise CalibrationFitError("InternalTaskConfig seed is not owned by the fit task")
    expected_prediction_only = {
        "candidate_m": [],
        "cg_tolerance": 1e-5,
        "cg_max_iterations": None,
        "use_preconditioner": True,
        "function_jitter": 1e-8,
        "reduced_jitter": 1e-8,
    }
    if any(config[name] != expected for name, expected in expected_prediction_only.items()):
        raise CalibrationFitError("fit-only InternalTaskConfig adapter fields changed")
    return config


def fit_config_for_task_index(task_index: int) -> InternalTaskConfig:
    """Return the complete frozen fit configuration for one grid index."""

    task = _task(task_index)
    _config_payload(task)
    return _fit_config(task)


def _expected_generator_config(task: F02bCalibrationFitTask) -> dict[str, Any]:
    return _fresh_json(
        asdict(
            ConfirmatoryConfig(
                n_particles=task.n_particles,
                n_dims=task.n_dims,
                replica=task.replica,
            )
        ),
        "generator config",
    )


def _catalog_bundle_index(task: F02bCalibrationFitTask) -> int:
    particle_counts = (2, 4, 6, 8, 10)
    try:
        particle_index = particle_counts.index(task.n_particles)
    except ValueError as error:  # pragma: no cover - grid import invariant
        raise CalibrationFitError("fit task is outside the frozen catalog coordinates") from error
    return task.replica * len(particle_counts) + particle_index


def _parameters_payload(value: Mapping[str, Any] | FrozenTERAParameters) -> dict[str, Any]:
    if isinstance(value, FrozenTERAParameters):
        if value.lengthscale.dtype != torch.float32:
            raise CalibrationFitError("frozen TERA lengthscale must have dtype float32")
        raw: Any = {
            "lengthscale": value.lengthscale.detach().cpu().reshape(-1).tolist(),
            "outputscale": value.outputscale,
            "sigma_f": value.sigma_f,
            "sigma_g": value.sigma_g,
            "kernel": value.kernel,
            "gradient_noise_model": value.gradient_noise_model,
        }
    else:
        raw = value
    payload = _fresh_json(raw, "fit parameters")
    return _require_exact_keys(payload, _PARAMETER_FIELDS, "parameters")


def _validate_parameters(value: object, config: Mapping[str, Any]) -> dict[str, Any]:
    parameters = _require_exact_keys(value, _PARAMETER_FIELDS, "parameters")
    if parameters["kernel"] != config["kernel"]:
        raise CalibrationFitError("parameter kernel does not match the frozen recipe")
    if parameters["gradient_noise_model"] != "iid":
        raise CalibrationFitError("gradient_noise_model must be exactly 'iid'")
    lengthscale = parameters["lengthscale"]
    if not isinstance(lengthscale, list) or len(lengthscale) != 1:
        raise CalibrationFitError("isotropic lengthscale must be a one-element JSON list")
    if _require_binary32(lengthscale[0], "parameters.lengthscale[0]") <= 0.0:
        raise CalibrationFitError("parameters.lengthscale[0] must be strictly positive")
    outputscale = _require_binary32(parameters["outputscale"], "parameters.outputscale")
    sigma_f = _require_binary32(parameters["sigma_f"], "parameters.sigma_f")
    sigma_g = _require_binary32(parameters["sigma_g"], "parameters.sigma_g")
    if outputscale <= 0.0:
        raise CalibrationFitError("parameters.outputscale must be strictly positive")
    if sigma_f < _to_binary32(config["min_sigma_f"], "fit_recipe.min_sigma_f"):
        raise CalibrationFitError("parameters.sigma_f is below the frozen minimum")
    if sigma_g < _to_binary32(config["min_sigma_g"], "fit_recipe.min_sigma_g"):
        raise CalibrationFitError("parameters.sigma_g is below the frozen minimum")
    return parameters


def _validate_bundle(value: object, task: F02bCalibrationFitTask) -> dict[str, Any]:
    bundle = _require_exact_keys(
        value,
        {
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
        },
        "input_identity.bundle",
    )
    dataset = Path(_require_text(bundle["dataset_path"], "bundle.dataset_path"))
    metadata = Path(_require_text(bundle["metadata_path"], "bundle.metadata_path"))
    manifest = Path(_require_text(bundle["manifest_path"], "bundle.manifest_path"))
    if not all(path.is_absolute() for path in (dataset, metadata, manifest)):
        raise CalibrationFitError("bundle paths must be absolute")
    if dataset.name != f"{task.dataset_stem}.npz":
        raise CalibrationFitError("bundle dataset path does not match the fit task")
    if metadata != dataset.with_suffix(".metadata.json"):
        raise CalibrationFitError("bundle metadata is not the dataset sidecar")
    if manifest != dataset.with_suffix(".sha256.json"):
        raise CalibrationFitError("bundle manifest is not the dataset sidecar")
    expected = {
        "replica": task.replica,
        "n_particles": task.n_particles,
        "n_dims": task.n_dims,
        "D": task.dimension,
        "phase": "development",
        "catalog_bundle_task_index": _catalog_bundle_index(task),
    }
    for name in ("replica", "n_particles", "n_dims", "D", "catalog_bundle_task_index"):
        _require_int(bundle[name], f"bundle.{name}")
    if _require_text(bundle["phase"], "bundle.phase") != "development":
        raise CalibrationFitError("confirmatory or mismatched bundle coordinates are forbidden")
    if task.replica not in (0, 1, 2) or canonical_json_bytes(
        {name: bundle[name] for name in expected}
    ) != canonical_json_bytes(expected):
        raise CalibrationFitError("confirmatory or mismatched bundle coordinates are forbidden")
    if canonical_json_bytes(bundle["generator_config"]) != canonical_json_bytes(
        _expected_generator_config(task)
    ):
        raise CalibrationFitError("bundle generator config does not match the fit task")
    hashes = _require_exact_keys(
        bundle["file_sha256"],
        {dataset.name, metadata.name},
        "bundle.file_sha256",
    )
    for name, digest in hashes.items():
        _require_sha256(digest, f"bundle.file_sha256.{name}")
    _require_sha256(bundle["sha256_manifest_file_sha256"], "bundle manifest SHA-256")
    _require_sha256(bundle["dataset_content_sha256"], "bundle content SHA-256")
    return bundle


def _validate_catalog(value: object) -> dict[str, Any]:
    catalog = _require_exact_keys(
        value,
        {"generation_git_commit", "generation_git_tree", "path", "schema_version", "sha256"},
        "input_identity.catalog",
    )
    if _require_int(catalog["schema_version"], "catalog.schema_version") != 1:
        raise CalibrationFitError("catalog schema_version must be 1")
    if not Path(_require_text(catalog["path"], "catalog.path")).is_absolute():
        raise CalibrationFitError("catalog path must be absolute")
    if _require_sha256(catalog["sha256"], "catalog.sha256") != F02_CATALOG_SHA256:
        raise CalibrationFitError("catalog SHA-256 does not match the public F02b contract")
    _require_git_oid(catalog["generation_git_commit"], "catalog generation commit")
    _require_git_oid(catalog["generation_git_tree"], "catalog generation tree")
    return catalog


def _validate_source(value: object) -> dict[str, Any]:
    source = _require_exact_keys(
        value,
        {"commit", "repo_root", "status_porcelain", "tera_gitlink", "tree"},
        "provenance.source",
    )
    if not Path(_require_text(source["repo_root"], "source.repo_root")).is_absolute():
        raise CalibrationFitError("source.repo_root must be absolute")
    _require_git_oid(source["commit"], "source.commit")
    _require_git_oid(source["tree"], "source.tree")
    _require_git_oid(source["tera_gitlink"], "source.tera_gitlink")
    if source["status_porcelain"] != []:
        raise CalibrationFitError("fit execution requires a globally clean worktree")
    return source


def _validate_dependencies(value: object) -> dict[str, Any]:
    dependencies = _require_exact_keys(
        value,
        {"pyproject.toml", "uv.lock"},
        "provenance.dependencies",
    )
    for name, raw in dependencies.items():
        record = _require_exact_keys(raw, {"sha256"}, f"dependencies.{name}")
        _require_sha256(record["sha256"], f"dependencies.{name}.sha256")
    return dependencies


def _validate_runtime(value: object) -> dict[str, Any]:
    runtime = _require_exact_keys(
        value,
        {"packages", "platform", "python_executable", "python_version"},
        "provenance.runtime",
    )
    for name in ("platform", "python_executable", "python_version"):
        _require_text(runtime[name], f"runtime.{name}")
    packages = runtime["packages"]
    if not isinstance(packages, list) or not packages:
        raise CalibrationFitError("runtime.packages must be a nonempty list")
    normalized: list[tuple[str, str]] = []
    for position, raw in enumerate(packages):
        package = _require_exact_keys(raw, {"name", "version"}, f"runtime.packages[{position}]")
        normalized.append(
            (
                _require_text(package["name"], f"runtime package {position} name"),
                _require_text(package["version"], f"runtime package {position} version"),
            )
        )
    if normalized != sorted(normalized) or len(normalized) != len(set(normalized)):
        raise CalibrationFitError("runtime packages must be sorted and unique")
    return runtime


def _validate_scheduler(value: object, task: F02bCalibrationFitTask) -> dict[str, Any]:
    scheduler = _require_exact_keys(
        value,
        {"array_job_id", "array_task_id", "exclusive_verification_mode", "job_id", "node_list"},
        "provenance.scheduler",
    )
    for name in ("array_job_id", "job_id"):
        identifier = _require_text(scheduler[name], f"scheduler.{name}")
        if re.fullmatch(r"[0-9]+", identifier) is None:
            raise CalibrationFitError(f"scheduler.{name} must be a decimal digit string")
    _require_text(scheduler["node_list"], "scheduler.node_list")
    if scheduler["exclusive_verification_mode"] != EXCLUSIVE_VERIFICATION_MODE:
        raise CalibrationFitError("scheduler exclusive verification mode is not registered")
    if _require_int(scheduler["array_task_id"], "scheduler.array_task_id") != task.task_index:
        raise CalibrationFitError("scheduler array task does not match the fit task")
    return scheduler


def _validate_identity(
    value: object,
    task: F02bCalibrationFitTask,
    *,
    dataset_path: Path | None = None,
    catalog_path: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    identity = _require_exact_keys(
        value,
        {
            "bundle",
            "catalog",
            "dependencies",
            "runtime",
            "runtime_allocation",
            "scheduler",
            "source",
        },
        "identity",
    )
    bundle = _validate_bundle(identity["bundle"], task)
    catalog = _validate_catalog(identity["catalog"])
    source = _validate_source(identity["source"])
    _validate_dependencies(identity["dependencies"])
    _validate_runtime(identity["runtime"])
    _validate_scheduler(identity["scheduler"], task)
    try:
        validate_runtime_allocation(identity["runtime_allocation"])
    except Exception as error:
        raise CalibrationFitError("runtime allocation violates the public contract") from error
    for expected, observed, label in (
        (dataset_path, Path(bundle["dataset_path"]), "dataset"),
        (catalog_path, Path(catalog["path"]), "catalog"),
        (repo_root, Path(source["repo_root"]), "repository"),
    ):
        if expected is not None and expected.resolve() != observed:
            raise CalibrationFitError(f"identity {label} path does not match the requested path")
    return identity


def build_fit_numeric_payload(
    task_index: int,
    parameters: Mapping[str, Any] | FrozenTERAParameters,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the canonical parameter-only fit result plus required provenance."""

    task = _task(task_index)
    identity_payload = _fresh_json(identity, "fit identity")
    validated_identity = _validate_identity(identity_payload, task)
    config = _config_payload(task)
    parameter_payload = _parameters_payload(parameters)
    _validate_parameters(parameter_payload, config)
    payload = {
        "schema_version": FIT_PAYLOAD_SCHEMA_VERSION,
        "artifact_type": FIT_PAYLOAD_TYPE,
        "task_role": "fit",
        "task_index": task.task_index,
        "task_record": task.as_record(),
        "fit_recipe_sha256": FIT_RECIPE_SHA256,
        "fit_recipe": _fresh_json(FIT_RECIPE, "fit recipe"),
        "data_access": _data_access_payload(),
        "input_identity": {
            "catalog": validated_identity["catalog"],
            "bundle": validated_identity["bundle"],
        },
        "parameters": parameter_payload,
        "provenance": {
            "source": validated_identity["source"],
            "dependencies": validated_identity["dependencies"],
            "runtime": validated_identity["runtime"],
            "scheduler": validated_identity["scheduler"],
            "runtime_numerics": _fresh_json(NUMERICAL_POLICY, "numerical policy"),
        },
        "capabilities": _capabilities_payload(),
    }
    return validate_fit_numeric_payload(payload)


def validate_fit_numeric_payload(value: Mapping[str, Any] | bytes | str | Path) -> dict[str, Any]:
    """Strictly validate a fit payload without trusting its filename."""

    try:
        if isinstance(value, bytes):
            raw = parse_strict_json_bytes(value)
        elif isinstance(value, (str, Path)):
            raw = _strict_json_file(Path(value), "fit numeric payload")
        else:
            raw = _fresh_json(value, "fit numeric payload")
    except CalibrationFitError:
        raise
    except Exception as error:
        raise CalibrationFitError("fit numeric payload is not strict JSON") from error
    payload = _require_exact_keys(raw, _PAYLOAD_FIELDS, "fit numeric payload")
    if payload["schema_version"] != FIT_PAYLOAD_SCHEMA_VERSION:
        raise CalibrationFitError("unsupported fit payload schema_version")
    if payload["artifact_type"] != FIT_PAYLOAD_TYPE:
        raise CalibrationFitError("unexpected fit payload type")
    if payload["task_role"] != "fit":
        raise CalibrationFitError("fit payload task_role must be 'fit'")
    task = _task(_require_int(payload["task_index"], "task_index"))
    if canonical_json_bytes(payload["task_record"]) != canonical_json_bytes(task.as_record()):
        raise CalibrationFitError("task record does not match the immutable fit grid")
    if payload["fit_recipe_sha256"] != FIT_RECIPE_SHA256:
        raise CalibrationFitError("fit recipe hash is mismatched")
    try:
        validate_fit_recipe(payload["fit_recipe"])
    except Exception as error:
        raise CalibrationFitError("fit recipe is mismatched") from error
    if canonical_json_bytes(payload["data_access"]) != canonical_json_bytes(_data_access_payload()):
        raise CalibrationFitError("train-only data-access isolation is mismatched")
    if canonical_json_bytes(payload["capabilities"]) != canonical_json_bytes(
        _capabilities_payload()
    ):
        raise CalibrationFitError("fit payload capabilities are mismatched")
    input_identity = _require_exact_keys(
        payload["input_identity"],
        {"bundle", "catalog"},
        "input_identity",
    )
    _validate_bundle(input_identity["bundle"], task)
    _validate_catalog(input_identity["catalog"])
    _validate_parameters(payload["parameters"], payload["fit_recipe"])
    provenance = _require_exact_keys(
        payload["provenance"],
        {"dependencies", "runtime", "runtime_numerics", "scheduler", "source"},
        "provenance",
    )
    _validate_source(provenance["source"])
    _validate_dependencies(provenance["dependencies"])
    _validate_runtime(provenance["runtime"])
    _validate_scheduler(provenance["scheduler"], task)
    if canonical_json_bytes(provenance["runtime_numerics"]) != canonical_json_bytes(
        NUMERICAL_POLICY
    ):
        raise CalibrationFitError("runtime numerical policy is mismatched")
    return _fresh_json(payload, "validated fit payload")


def fit_artifact_paths(output_root: str | Path, task_index: int) -> FitArtifactPaths:
    """Derive all output paths solely from the immutable fit identity."""

    task = _task(task_index)
    dirname = (
        f"fit-{task.task_index:03d}-{task.dataset_stem}-seed{task.seed}-"
        f"{FIT_TASK_MATRIX_SHA256[:12]}"
    )
    directory = Path(output_root).resolve() / dirname
    return FitArtifactPaths(
        directory=directory,
        numeric_payload=directory / "parameters.json",
        execution_envelope=directory / "execution-envelope.json",
    )


def _payload_relative_path(paths: FitArtifactPaths) -> str:
    return f"{paths.directory.name}/{paths.numeric_payload.name}"


def build_fit_artifacts(
    task_index: int,
    parameters: Mapping[str, Any] | FrozenTERAParameters,
    identity: Mapping[str, Any],
    *,
    output_root: str | Path,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    """Build canonical payload bytes and their public execution envelope."""

    payload = build_fit_numeric_payload(task_index, parameters, identity)
    payload_bytes = canonical_json_bytes(payload)
    paths = fit_artifact_paths(output_root, task_index)
    binding = build_payload_binding(
        "fit",
        task_index,
        numeric_payload_path=_payload_relative_path(paths),
        numeric_payload_sha256=hashlib.sha256(payload_bytes).hexdigest(),
    )
    allocation = _validate_identity(identity, _task(task_index))["runtime_allocation"]
    envelope = build_fit_execution_envelope(
        task_index,
        runtime_allocation=allocation,
        payload_binding=binding,
    )
    validate_fit_artifact_pair(payload_bytes, canonical_json_bytes(envelope))
    return payload, payload_bytes, envelope


def validate_fit_artifact_pair(
    numeric_payload: bytes | str | Path,
    execution_envelope: bytes | str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate payload bytes and envelope together, including the raw-byte hash."""

    try:
        payload_bytes = (
            numeric_payload
            if isinstance(numeric_payload, bytes)
            else Path(numeric_payload).read_bytes()
        )
        envelope_bytes = (
            execution_envelope
            if isinstance(execution_envelope, bytes)
            else Path(execution_envelope).read_bytes()
        )
        payload = validate_fit_numeric_payload(payload_bytes)
        envelope = validate_fit_execution_envelope(
            parse_strict_json_bytes(envelope_bytes),
            expected_task_index=payload["task_index"],
        )
        if payload_bytes != canonical_json_bytes(payload):
            raise CalibrationFitError("fit numeric payload bytes are not canonical JSON")
        if envelope_bytes != canonical_json_bytes(envelope):
            raise CalibrationFitError("fit execution envelope bytes are not canonical JSON")
        expected_paths = fit_artifact_paths(Path.cwd(), payload["task_index"])
        expected_binding_path = _payload_relative_path(expected_paths)
        binding = envelope["canonical_record"]["payload_binding"]
        if binding["numeric_payload_path"] != expected_binding_path:
            raise CalibrationFitError("fit payload binding path is not identity-derived")
        verify_numeric_payload_bytes(
            binding,
            payload_bytes,
            expected_task_role="fit",
            expected_task_index=payload["task_index"],
        )
    except CalibrationFitError:
        raise
    except Exception as error:
        raise CalibrationFitError("fit payload/envelope pair is invalid") from error
    return payload, envelope


def _run_git(repo_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise CalibrationFitError(f"git command failed: {' '.join(arguments)}") from error
    return completed.stdout.strip()


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise CalibrationFitError(f"required allocation environment is missing: {name}")
    return value


def _environment_int(name: str) -> int:
    value = _required_environment(name)
    if re.fullmatch(r"[0-9]+", value) is None:
        raise CalibrationFitError(
            f"allocation environment {name} must be a non-negative decimal integer"
        )
    return int(value)


def _run_scontrol_job_record(job_id: str) -> dict[str, str]:
    try:
        completed = subprocess.run(
            ["scontrol", "show", "job", job_id, "-o"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise CalibrationFitError("cannot query the live Slurm job with scontrol") from error
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise CalibrationFitError("scontrol must return exactly one live job record")
    fields: dict[str, str] = {}
    for token in lines[0].split():
        if "=" not in token:
            continue
        name, value = token.split("=", 1)
        if not name or name in fields:
            raise CalibrationFitError("scontrol returned malformed or duplicate fields")
        fields[name] = value
    return fields


def _observe_cuda_and_cpu_affinity() -> tuple[int, list[str], list[int]]:
    try:
        affinity = os.sched_getaffinity(0)
    except (AttributeError, OSError) as error:
        raise CalibrationFitError("cannot observe the process CPU affinity") from error
    available_cpu_count = len(affinity)
    if type(available_cpu_count) is not int or available_cpu_count <= 0:
        raise CalibrationFitError("process CPU affinity is empty")
    try:
        if not torch.cuda.is_available():
            raise CalibrationFitError("CUDA is unavailable to the fit process")
        visible_gpu_count = torch.cuda.device_count()
        if type(visible_gpu_count) is not int or visible_gpu_count < 1:
            raise CalibrationFitError("no CUDA devices are visible to the fit process")
        properties = [torch.cuda.get_device_properties(index) for index in range(visible_gpu_count)]
    except CalibrationFitError:
        raise
    except Exception as error:
        raise CalibrationFitError("cannot inspect CUDA device properties") from error
    models: list[str] = []
    memory_bytes: list[int] = []
    for index, device in enumerate(properties):
        name = getattr(device, "name", None)
        total_memory = getattr(device, "total_memory", None)
        if type(name) is not str or not name:
            raise CalibrationFitError(f"CUDA device {index} has no valid model name")
        if type(total_memory) is not int or total_memory <= 0:
            raise CalibrationFitError(f"CUDA device {index} has no valid memory size")
        models.append(name)
        memory_bytes.append(total_memory)
    return available_cpu_count, models, memory_bytes


def _observe_runtime_evidence(
    task: F02bCalibrationFitTask,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Independently bind live Slurm and process-visible hardware evidence."""

    job_id = _required_environment("SLURM_JOB_ID")
    array_job_id = _required_environment("SLURM_ARRAY_JOB_ID")
    for value, label in ((job_id, "SLURM_JOB_ID"), (array_job_id, "SLURM_ARRAY_JOB_ID")):
        if re.fullmatch(r"[0-9]+", value) is None:
            raise CalibrationFitError(f"{label} must be a decimal digit string")
    array_task_id = _environment_int("SLURM_ARRAY_TASK_ID")
    if array_task_id != task.task_index:
        raise CalibrationFitError("Slurm array task does not match the immutable fit task")
    expected_array_grid = {
        "SLURM_ARRAY_TASK_COUNT": 45,
        "SLURM_ARRAY_TASK_MIN": 0,
        "SLURM_ARRAY_TASK_MAX": 44,
        "SLURM_ARRAY_TASK_STEP": 1,
    }
    if any(_environment_int(name) != expected for name, expected in expected_array_grid.items()):
        raise CalibrationFitError("Slurm array environment is not the exact 0-44 step-1 grid")

    fields = _run_scontrol_job_record(job_id)
    exact_fields = {
        "JobId": job_id,
        "ArrayJobId": array_job_id,
        "ArrayTaskId": str(task.task_index),
        "ArrayTaskThrottle": "1",
        "OverSubscribe": "EXCLUSIVE",
        "TimeLimit": "08:00:00",
        "CPUs/Task": "16",
        "MinMemoryNode": "64G",
        "TresPerNode": "gres/gpu:l40s:1",
        "NumNodes": "1",
    }
    mismatched = [name for name, expected in exact_fields.items() if fields.get(name) != expected]
    if mismatched:
        raise CalibrationFitError(
            f"live scontrol job fields violate the F02b contract: {sorted(mismatched)}"
        )
    partition = fields.get("Partition")
    if partition not in {"short", "interactivegpu"}:
        raise CalibrationFitError("live scontrol partition is not registered")
    if partition != _required_environment("SLURM_JOB_PARTITION"):
        raise CalibrationFitError("live scontrol partition disagrees with Slurm environment")
    node_list = fields.get("NodeList")
    if not node_list or node_list == "(null)":
        raise CalibrationFitError("live scontrol job has no allocated node")
    if node_list != _required_environment("SLURM_JOB_NODELIST"):
        raise CalibrationFitError("live scontrol node list disagrees with Slurm environment")

    available_cpu_count, models, memory_bytes = _observe_cuda_and_cpu_affinity()
    allocation = {
        "exclusive_node": True,
        "requested_gpu_count": 1,
        "visible_gpu_count": len(models),
        "visible_gpu_models": models,
        "visible_gpu_memory_bytes": memory_bytes,
        "requested_cpus_per_task": 16,
        "available_cpu_count": available_cpu_count,
        "available_host_memory_bytes": MINIMUM_HOST_MEMORY_BYTES,
        "requested_walltime_seconds": WALLTIME_SECONDS,
        "walltime_limit_seconds": WALLTIME_SECONDS,
        "array_concurrency": 1,
        "partition": partition,
    }
    scheduler = {
        "job_id": fields["JobId"],
        "array_job_id": fields["ArrayJobId"],
        "array_task_id": int(fields["ArrayTaskId"]),
        "node_list": node_list,
        "exclusive_verification_mode": EXCLUSIVE_VERIFICATION_MODE,
    }
    try:
        validated_allocation = validate_runtime_allocation(allocation)
    except Exception as error:
        raise CalibrationFitError("observed runtime allocation violates the contract") from error
    return validated_allocation, _validate_scheduler(scheduler, task)


def _source_identity(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    status = _run_git(root, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
    if status:
        raise CalibrationFitError("fit execution requires a globally clean worktree")
    gitlink = _run_git(root, "rev-parse", "HEAD:gp/tera/vendor")
    if _run_git(root / "gp/tera/vendor", "rev-parse", "HEAD") != gitlink:
        raise CalibrationFitError("checked-out TERA vendor does not match the committed gitlink")
    return {
        "repo_root": str(root),
        "commit": _run_git(root, "rev-parse", "HEAD"),
        "tree": _run_git(root, "rev-parse", "HEAD^{tree}"),
        "status_porcelain": [],
        "tera_gitlink": gitlink,
    }


def _runtime_identity() -> dict[str, Any]:
    audit = audit_distributions()
    if audit.get("status") != "pass" or audit.get("issues") != []:
        raise CalibrationFitError("runtime dependency audit did not pass")
    packages = audit.get("packages")
    if not isinstance(packages, list):
        raise CalibrationFitError("runtime dependency audit did not return packages")
    runtime = {
        "python_executable": os.path.realpath(sys.executable),
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }
    return _validate_runtime(runtime)


def _dependency_identity(repo_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("pyproject.toml", "uv.lock"):
        path = repo_root / name
        content = _read_regular_file_once(path, f"dependency declaration {name}")
        result[name] = {"sha256": hashlib.sha256(content).hexdigest()}
    return _validate_dependencies(result)


def _preload_default_fit_implementation(repo_root: Path) -> None:
    """Load every repository module used by the production fit before source capture."""

    root = repo_root.resolve()
    try:
        wrapper = importlib.import_module("gp.tera")
        released_data = importlib.import_module("md22_regression.data")
        released_model = importlib.import_module("md22_regression.models.tera")
        _ = torch.optim.Adam  # Force lazy torch.optim resolution before dependency capture.
    except Exception as error:
        raise CalibrationFitError("cannot preload the default TERA fit implementation") from error

    modules = (
        (sys.modules[fit_released_tera.__module__], root, "internal fit adapter"),
        (
            sys.modules[load_prepared_confirmatory_bundle.__module__],
            root,
            "authoritative bundle loader",
        ),
        (wrapper, root / "gp" / "tera", "TERA wrapper"),
        (released_data, root / "gp" / "tera" / "vendor" / "src", "released TERA data"),
        (released_model, root / "gp" / "tera" / "vendor" / "src", "released TERA model"),
    )
    for module, expected_root, label in modules:
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str):
            raise CalibrationFitError(f"{label} has no auditable source origin")
        try:
            Path(origin).resolve().relative_to(expected_root.resolve())
        except ValueError as error:
            raise CalibrationFitError(
                f"{label} was imported outside the bound source tree"
            ) from error


def _strict_scan_identity_json(dataset_path: Path, catalog_path: Path) -> None:
    for path, label in (
        (catalog_path, "catalog"),
        (dataset_path.with_suffix(".metadata.json"), "bundle metadata"),
        (dataset_path.with_suffix(".sha256.json"), "bundle manifest"),
    ):
        if not isinstance(_strict_json_file(path, label), dict):
            raise CalibrationFitError(f"{label} must contain a JSON object")


def _loaded_bundle_matches_preflight(
    bundle: Any,
    preflight: BundleIdentity,
    loaded_dataset_path: Path,
) -> None:
    provenance = bundle.loaded.provenance
    loaded_dataset = loaded_dataset_path.resolve()
    if provenance.dataset_path.resolve() != loaded_dataset:
        raise CalibrationFitError("authoritative loader did not read the private dataset snapshot")
    if provenance.metadata_path.resolve() != loaded_dataset.with_suffix(".metadata.json"):
        raise CalibrationFitError("authoritative loader did not read the private metadata snapshot")
    if provenance.sha256_manifest_path.resolve() != loaded_dataset.with_suffix(".sha256.json"):
        raise CalibrationFitError("authoritative loader did not read the private manifest snapshot")
    if dict(sorted(provenance.file_sha256.items())) != preflight.file_sha256:
        raise CalibrationFitError("loaded bundle hashes changed after preflight")
    if provenance.config_payload != preflight.generator_config:
        raise CalibrationFitError("loaded bundle config changed after preflight")
    if _sha256_file(provenance.sha256_manifest_path) != preflight.manifest_sha256:
        raise CalibrationFitError("loaded bundle manifest changed after preflight")


def _preflight_corpus_identity(
    task: F02bCalibrationFitTask,
    dataset_path: Path,
    catalog_path: Path,
    *,
    snapshot: _CorpusSnapshot | None = None,
) -> tuple[dict[str, Any], BundleIdentity]:
    dataset = dataset_path.resolve()
    catalog = catalog_path.resolve()
    if dataset.name != f"{task.dataset_stem}.npz":
        raise CalibrationFitError("dataset filename does not match the immutable fit task")
    if snapshot is None:
        with _immutable_corpus_snapshot(dataset, catalog) as captured:
            return _preflight_corpus_identity(
                task,
                dataset,
                catalog,
                snapshot=captured,
            )
    _strict_scan_identity_json(snapshot.dataset_path, snapshot.catalog_path)
    try:
        captured_preflight = _preflight_bundle_identity(snapshot.dataset_path)
        preflight = BundleIdentity(
            dataset_path=dataset,
            file_sha256=captured_preflight.file_sha256,
            manifest_sha256=captured_preflight.manifest_sha256,
            generator_config=captured_preflight.generator_config,
        )
        authorization = validate_catalog_identity(snapshot.catalog_path, preflight)
    except Exception as error:
        raise CalibrationFitError("bundle/catalog identity validation failed") from error
    if authorization.catalog_sha256 != F02_CATALOG_SHA256:
        raise CalibrationFitError("catalog SHA-256 does not match the public F02b contract")
    entry = authorization.bundle_entry
    expected = {
        "task_index": _catalog_bundle_index(task),
        "phase": "development",
        "replica": task.replica,
        "n_particles": task.n_particles,
        "n_dims": task.n_dims,
        "D": task.dimension,
    }
    if task.replica not in (0, 1, 2) or any(
        entry.get(name) != item for name, item in expected.items()
    ):
        raise CalibrationFitError("confirmatory or mismatched catalog bundle is forbidden")
    hashes = entry.get("hashes")
    if not isinstance(hashes, dict):
        raise CalibrationFitError("catalog bundle hashes are missing")
    identity = {
        "catalog": {
            "path": str(catalog),
            "sha256": authorization.catalog_sha256,
            "schema_version": 1,
            "generation_git_commit": authorization.generation_git_commit,
            "generation_git_tree": authorization.generation_git_tree,
        },
        "bundle": {
            "dataset_path": str(dataset),
            "metadata_path": str(dataset.with_suffix(".metadata.json")),
            "manifest_path": str(dataset.with_suffix(".sha256.json")),
            "file_sha256": preflight.file_sha256,
            "sha256_manifest_file_sha256": preflight.manifest_sha256,
            "dataset_content_sha256": hashes.get("dataset_content_sha256"),
            "generator_config": preflight.generator_config,
            "catalog_bundle_task_index": entry.get("task_index"),
            "phase": entry.get("phase"),
            "replica": entry.get("replica"),
            "n_particles": entry.get("n_particles"),
            "n_dims": entry.get("n_dims"),
            "D": entry.get("D"),
        },
    }
    _validate_bundle(identity["bundle"], task)
    _validate_catalog(identity["catalog"])
    return identity, preflight


def default_postflight_validator(
    task: F02bCalibrationFitTask,
    config: InternalTaskConfig,
    dataset_path: Path,
    catalog_path: Path,
    repo_root: Path,
    expected_identity: Mapping[str, Any],
) -> None:
    """Revalidate every mutable byte identity immediately after fitting."""

    del config
    expected = _validate_identity(
        _fresh_json(expected_identity, "preflight identity"),
        task,
        dataset_path=dataset_path.resolve(),
        catalog_path=catalog_path.resolve(),
        repo_root=repo_root.resolve(),
    )
    corpus, _ = _preflight_corpus_identity(task, dataset_path, catalog_path)
    observed = {
        **corpus,
        "source": _source_identity(repo_root.resolve()),
        "dependencies": _dependency_identity(repo_root.resolve()),
    }
    expected_mutable = {
        name: expected[name] for name in ("catalog", "bundle", "source", "dependencies")
    }
    if canonical_json_bytes(observed) != canonical_json_bytes(expected_mutable):
        raise CalibrationFitError("postflight input/source identity changed during fitting")


def default_identity_provider(
    task: F02bCalibrationFitTask,
    config: InternalTaskConfig,
    dataset_path: Path,
    catalog_path: Path,
    repo_root: Path,
) -> FitInputs:
    """Validate production identities and expose only tensorized training rows."""

    dataset = dataset_path.resolve()
    catalog = catalog_path.resolve()
    root = repo_root.resolve()

    # Refuse an unregistered/shared execution before reading corpus bytes or
    # allocating the train tensors on CUDA.  Preload the production adapter
    # first so the source identity covers the code that will actually run.
    _preload_default_fit_implementation(root)
    runtime_allocation, scheduler = _observe_runtime_evidence(task)
    source = _source_identity(root)
    dependencies = _dependency_identity(root)
    runtime = _runtime_identity()

    with _immutable_corpus_snapshot(dataset, catalog) as snapshot:
        corpus_identity, preflight = _preflight_corpus_identity(
            task,
            dataset,
            catalog,
            snapshot=snapshot,
        )

        # Full-bundle loading is solely an integrity operation.  Authorization
        # and loading consume the same private bytes; only ``train`` escapes.
        bundle = load_prepared_confirmatory_bundle(snapshot.dataset_path)
        _loaded_bundle_matches_preflight(bundle, preflight, snapshot.dataset_path)
        selected_train = select_time_indices(bundle.prepared.train, TRAIN_TIME_INDICES)
        train = TensorConfirmatorySplit(
            name="train",
            source_indices=torch.as_tensor(
                selected_train.source_indices.copy(), dtype=torch.long, device=config.device
            ),
            X=torch.as_tensor(selected_train.X.copy(), dtype=torch.float32, device=config.device),
            value=torch.as_tensor(
                selected_train.E.copy(), dtype=torch.float32, device=config.device
            ),
            gradient=torch.as_tensor(
                selected_train.F.copy(), dtype=torch.float32, device=config.device
            ),
            trajectory_id=torch.as_tensor(
                selected_train.trajectory_id.copy(), dtype=torch.long, device=config.device
            ),
            time_index=torch.as_tensor(
                selected_train.time_index.copy(), dtype=torch.long, device=config.device
            ),
            time_value=torch.as_tensor(
                selected_train.time_value.copy(), dtype=torch.float32, device=config.device
            ),
        )

    identity = {
        **corpus_identity,
        "source": source,
        "dependencies": dependencies,
        "runtime": runtime,
        "runtime_allocation": runtime_allocation,
        "scheduler": scheduler,
    }
    return FitInputs(identity=_validate_identity(identity, task), train=train)


def _fit_kwargs(config: InternalTaskConfig) -> dict[str, Any]:
    if set(_FIT_CALL_RECIPE_FIELDS) != set(FIT_RECIPE) - {"dtype", "device"}:
        raise CalibrationFitError("fit callable adapter drifted from the public recipe")
    return {name: getattr(config, name) for name in _FIT_CALL_RECIPE_FIELDS}


def default_fit_callable(train: Any, config: InternalTaskConfig) -> FrozenTERAParameters:
    """Fit released TERA from the sole train-only object supplied by the runner."""

    if not isinstance(train, TensorConfirmatorySplit) or train.name != "train":
        raise CalibrationFitError("default fit callable accepts only a tensorized train split")
    model = fit_released_tera(train, seed=config.seed, **_fit_kwargs(config))
    return freeze_tera_parameters(model)


def _run_with_numerical_policy(
    fit_callable: FitCallable,
    train: Any,
    config: InternalTaskConfig,
) -> Mapping[str, Any] | FrozenTERAParameters:
    previous_precision = torch.get_float32_matmul_precision()
    previous_matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if torch.get_float32_matmul_precision() != "highest":
            raise CalibrationFitError("failed to activate highest float32 matmul precision")
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise CalibrationFitError("failed to disable CUDA TF32")
        return fit_callable(train, config)
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32


def write_fit_artifacts_exclusive(
    output_root: str | Path,
    task_index: int,
    payload_bytes: bytes,
    envelope: Mapping[str, Any],
) -> FitArtifactPaths:
    """Reserve an identity directory and write a validated pair without overwrite."""

    paths = fit_artifact_paths(output_root, task_index)
    envelope_bytes = canonical_json_bytes(envelope)
    validate_fit_artifact_pair(payload_bytes, envelope_bytes)
    reserved = False
    try:
        paths.directory.parent.mkdir(parents=True, exist_ok=True)
        paths.directory.mkdir()
        reserved = True
        with paths.numeric_payload.open("xb") as handle:
            handle.write(payload_bytes)
        with paths.execution_envelope.open("xb") as handle:
            handle.write(envelope_bytes)
        validate_fit_artifact_pair(paths.numeric_payload, paths.execution_envelope)
    except Exception as error:
        if not reserved and isinstance(error, FileExistsError):
            raise CalibrationFitError(
                f"refusing to overwrite fit artifact: {paths.directory}"
            ) from error
        cleanup_error: OSError | None = None
        if reserved:
            for path in (paths.execution_envelope, paths.numeric_payload):
                try:
                    path.unlink(missing_ok=True)
                except OSError as caught:
                    cleanup_error = caught
            try:
                paths.directory.rmdir()
            except OSError as caught:
                cleanup_error = caught
        if cleanup_error is not None:
            raise CalibrationFitError(
                f"fit artifact write failed and reserved directory cleanup was incomplete: "
                f"{paths.directory}"
            ) from cleanup_error
        if isinstance(error, CalibrationFitError):
            raise
        raise CalibrationFitError(
            f"cannot write fit artifact directory: {paths.directory}"
        ) from error
    return paths


def run_fit_task(
    task_index: int,
    *,
    dataset_path: str | Path,
    catalog_path: str | Path,
    output_root: str | Path,
    fit_callable: FitCallable = default_fit_callable,
    identity_provider: IdentityProvider = default_identity_provider,
    postflight_validator: PostflightValidator = default_postflight_validator,
    repo_root: str | Path = _REPO_ROOT,
) -> tuple[dict[str, Any], dict[str, Any], FitArtifactPaths]:
    """Run one immutable fit, then write its parameter payload and envelope."""

    task = _task(task_index)
    paths = fit_artifact_paths(output_root, task.task_index)
    if paths.directory.exists():
        raise CalibrationFitError(f"refusing to overwrite fit artifact: {paths.directory}")
    config = fit_config_for_task_index(task.task_index)
    dataset = Path(dataset_path).resolve()
    catalog = Path(catalog_path).resolve()
    root = Path(repo_root).resolve()
    inputs = identity_provider(task, config, dataset, catalog, root)
    if not isinstance(inputs, FitInputs):
        raise CalibrationFitError("identity provider must return FitInputs")
    identity = _fresh_json(inputs.identity, "fit identity")
    _validate_identity(
        identity,
        task,
        dataset_path=dataset,
        catalog_path=catalog,
        repo_root=root,
    )
    learned = _run_with_numerical_policy(fit_callable, inputs.train, config)
    postflight_validator(
        task,
        config,
        dataset,
        catalog,
        root,
        _fresh_json(identity, "postflight identity"),
    )
    payload, payload_bytes, envelope = build_fit_artifacts(
        task.task_index,
        learned,
        identity,
        output_root=output_root,
    )
    written = write_fit_artifacts_exclusive(
        output_root,
        task.task_index,
        payload_bytes,
        envelope,
    )
    return payload, envelope, written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    fit_callable: FitCallable = default_fit_callable,
    identity_provider: IdentityProvider = default_identity_provider,
    postflight_validator: PostflightValidator = default_postflight_validator,
) -> int:
    args = build_parser().parse_args(argv)
    _, _, paths = run_fit_task(
        args.task_index,
        dataset_path=args.dataset,
        catalog_path=args.catalog,
        output_root=args.output_root,
        fit_callable=fit_callable,
        identity_provider=identity_provider,
        postflight_validator=postflight_validator,
    )
    print(paths.execution_envelope)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CalibrationFitError",
    "EXCLUSIVE_VERIFICATION_MODE",
    "FIT_PAYLOAD_SCHEMA_VERSION",
    "FIT_PAYLOAD_TYPE",
    "FitArtifactPaths",
    "FitInputs",
    "build_fit_artifacts",
    "build_fit_numeric_payload",
    "build_parser",
    "default_fit_callable",
    "default_identity_provider",
    "default_postflight_validator",
    "fit_artifact_paths",
    "fit_config_for_task_index",
    "main",
    "run_fit_task",
    "validate_fit_artifact_pair",
    "validate_fit_numeric_payload",
    "write_fit_artifacts_exclusive",
]
