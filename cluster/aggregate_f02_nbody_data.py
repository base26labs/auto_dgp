"""Offline, provenance-strict cataloguing for F02 N-body data arrays.

The generator writes one fixed-mass confirmatory bundle per Slurm array task.
This aggregator does not trust those task records: it reconstructs the expected
replica-major task grid, verifies task and source provenance, and reloads every
three-file bundle through :func:`data.load_nbody_confirmatory.load_confirmatory_bundle`.
Missing, failed, invalid, and unexpected tasks remain visible in the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cluster.generate_f02_nbody import GenerationTask, generation_tasks
from data.generate_nbody_confirmatory import ConfirmatoryConfig
from data.load_nbody_confirmatory import LoadedConfirmatoryBundle, load_confirmatory_bundle

DEFAULT_REPLICAS = (0, 1, 2, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110)
DEFAULT_DEVELOPMENT_REPLICAS = (0, 1, 2)
DEFAULT_PARTICLE_COUNTS = (2, 4, 6, 8, 10)

EXPECTED_SOURCE_PATHS = frozenset(
    {
        "cluster/generate_f02_nbody.py",
        "data/generate_nbody_confirmatory.py",
        "data/load_nbody_confirmatory.py",
        "docs/F02_NBODY_PROTOCOL.md",
        "pyproject.toml",
        "uv.lock",
    }
)

_RESULT_KEYS = frozenset(
    {"schema_version", "task", "config", "artifacts", "validation"}
)
_RESULT_ARTIFACT_KEYS = frozenset(
    {"dataset", "metadata", "sha256_manifest", "file_sha256"}
)
_REQUIRED_TASK_ARTIFACTS = {
    "result": "result.json",
    "provenance": "provenance.env",
    "submodules": "git-submodules.txt",
    "source_hashes": "source-files.sha256",
    "artifact_hashes": "artifacts.sha256",
}
_ARTIFACT_HASH_NAMES = frozenset({"result.json", "source-files.sha256"})


@dataclass(slots=True)
class _LoadedTask:
    record: dict[str, Any]
    bundle: dict[str, Any] | None = None
    commit: str = ""
    tree: str = ""
    submodules: str = ""
    source_hashes: dict[str, str] | None = None
    source_manifest_sha256: str = ""
    array_job_id: str = ""
    repo_root: str = ""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _is_git_object_id(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 40:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _parse_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line or line.lstrip().startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"malformed provenance line: {line!r}")
        key, value = line.split("=", 1)
        if not key or key in result:
            raise ValueError(f"duplicate or empty provenance key: {key!r}")
        result[key] = value
    return result


def _parse_sha256sum(path: Path, *, basenames: bool) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text().splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or not _is_sha256(fields[0]):
            raise ValueError(f"malformed SHA-256 line: {line!r}")
        source = fields[1].lstrip("*")
        key = Path(source).name if basenames else Path(source).as_posix()
        if not key or key in entries:
            raise ValueError(f"duplicate or empty SHA-256 path: {key!r}")
        entries[key] = fields[0].lower()
    return entries


def _parse_csv_ints(value: str, label: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"{label} must be comma-separated integers") from error
    if not parsed:
        raise ValueError(f"{label} must not be empty")
    return parsed


def _task_record(
    path: Path,
    task: GenerationTask,
    phase: str,
    *,
    status: str,
    errors: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "task_index": task.task_index,
        "expected_task": asdict(task),
        "phase": phase,
        "path": str(path),
        "status": status,
        "exit_code": None,
        "eligible_for_catalog": False,
        "errors": list(errors),
    }


def _expected_config(
    task: GenerationTask,
    *,
    n_trajectories: int,
    steps_per_trajectory: int,
    dt: float,
    mass_seed: int,
    trajectory_seed: int,
    split_seed: int,
    validation_seed: int,
) -> ConfirmatoryConfig:
    config = ConfirmatoryConfig(
        n_particles=task.n_particles,
        n_dims=task.n_dims,
        n_trajectories=n_trajectories,
        steps_per_trajectory=steps_per_trajectory,
        dt=dt,
        replica=task.replica,
        mass_seed=mass_seed,
        trajectory_seed=trajectory_seed,
        split_seed=split_seed,
        validation_seed=validation_seed,
    )
    config.validate()
    return config


def _resolve_recorded_path(raw: Any, repo_root: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError("recorded artifact path must be a nonempty string")
    path = Path(raw)
    if not path.is_absolute():
        if not repo_root:
            raise ValueError("relative artifact path has no recorded repo_root")
        path = Path(repo_root) / path
    return path.resolve()


def _update_content_hash(digest: Any, name: str, value: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    metadata = {
        "name": name,
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "nbytes": int(array.nbytes),
    }
    header = _canonical_bytes(metadata)
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(array.tobytes(order="C"))


def _dataset_content_sha256(loaded: LoadedConfirmatoryBundle) -> str:
    """Hash semantic dataset arrays, independent of ZIP timestamps and paths."""

    dataset = loaded.dataset
    digest = hashlib.sha256()
    arrays = {
        "X": dataset.X,
        "E": dataset.E,
        "F": dataset.F,
        "masses": dataset.masses,
        "trajectory_id": dataset.trajectory_id,
        "time_index": dataset.time_index,
        "time_value": dataset.time_value,
        "train_indices": dataset.splits.train_indices,
        "validation_indices": dataset.splits.validation_indices,
        "test_indices": dataset.splits.test_indices,
        "train_trajectory_ids": dataset.splits.train_trajectory_ids,
        "validation_trajectory_ids": dataset.splits.validation_trajectory_ids,
        "test_trajectory_ids": dataset.splits.test_trajectory_ids,
        "x_train_min": dataset.normalization.x_min,
        "x_train_span": dataset.normalization.x_span,
        "energy_train_mean": np.asarray(dataset.normalization.energy_mean),
        "energy_train_std": np.asarray(dataset.normalization.energy_std),
        "gradient_scale": dataset.normalization.gradient_scale,
    }
    for name, value in arrays.items():
        _update_content_hash(digest, name, value)
    return digest.hexdigest()


def _validate_submodules(path: Path, errors: list[str]) -> str:
    lines = [line.rstrip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        errors.append("git-submodules.txt is empty")
        return ""
    if any(not line.startswith(" ") for line in lines):
        errors.append("git-submodules.txt reports a modified or uninitialized submodule")
    return "\n".join(line.strip() for line in lines)


def _validate_result_schema(result: Any, errors: list[str]) -> dict[str, Any]:
    if not isinstance(result, dict):
        errors.append("result.json must contain an object")
        return {}
    if set(result) != _RESULT_KEYS:
        errors.append(
            "result.json has the wrong top-level schema: "
            f"expected {sorted(_RESULT_KEYS)}, got {sorted(result)}"
        )
    if result.get("schema_version") != 1:
        errors.append("result.json schema_version must equal 1")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != _RESULT_ARTIFACT_KEYS:
        errors.append("result artifacts are missing or have the wrong schema")
    if not isinstance(result.get("task"), dict):
        errors.append("result task is missing or malformed")
    if not isinstance(result.get("config"), dict):
        errors.append("result config is missing or malformed")
    if not isinstance(result.get("validation"), dict):
        errors.append("result validation is missing or malformed")
    return result


def _load_task(
    path: Path,
    task: GenerationTask,
    phase: str,
    expected_config: ConfirmatoryConfig,
    *,
    replicas: Sequence[int],
    particle_counts: Sequence[int],
) -> _LoadedTask:
    if not path.exists():
        return _LoadedTask(
            _task_record(path, task, phase, status="missing")
        )
    if not path.is_dir():
        return _LoadedTask(
            _task_record(
                path,
                task,
                phase,
                status="invalid",
                errors=("task path is not a directory",),
            )
        )

    record = _task_record(path, task, phase, status="invalid")
    exit_code_path = path / "exit-code.txt"
    if not exit_code_path.is_file():
        record["errors"].append("missing exit-code.txt")
        return _LoadedTask(record)
    try:
        exit_code = int(exit_code_path.read_text().strip())
    except ValueError:
        record["errors"].append("exit-code.txt is not an integer")
        return _LoadedTask(record)
    record["exit_code"] = exit_code
    if exit_code != 0:
        record["status"] = "failed"
        record["errors"].append(f"task exited with status {exit_code}")
        return _LoadedTask(record)

    required = {name: path / filename for name, filename in _REQUIRED_TASK_ARTIFACTS.items()}
    missing = [name for name, artifact in required.items() if not artifact.is_file()]
    if missing:
        record["errors"].append(
            f"missing required artifacts: {', '.join(sorted(missing))}"
        )
        return _LoadedTask(record)

    errors: list[str] = []
    try:
        result = _validate_result_schema(_read_json(required["result"]), errors)
        provenance = _parse_env(required["provenance"])
        source_hashes = _parse_sha256sum(required["source_hashes"], basenames=False)
        artifact_hashes = _parse_sha256sum(required["artifact_hashes"], basenames=True)
        submodules = _validate_submodules(required["submodules"], errors)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        record["errors"].append(f"could not parse task artifacts: {error}")
        return _LoadedTask(record)

    if set(artifact_hashes) != _ARTIFACT_HASH_NAMES:
        errors.append("artifacts.sha256 must cover exactly result.json and source-files.sha256")
    for name in _ARTIFACT_HASH_NAMES:
        artifact_path = path / name
        if artifact_hashes.get(name) != _sha256_file(artifact_path):
            errors.append(f"artifacts.sha256 does not verify {name}")

    if set(source_hashes) != EXPECTED_SOURCE_PATHS:
        errors.append(
            "source-files.sha256 has the wrong source set: "
            f"expected {sorted(EXPECTED_SOURCE_PATHS)}, got {sorted(source_hashes)}"
        )

    commit = provenance.get("git_commit", "")
    tree = provenance.get("git_tree", "")
    git_describe = provenance.get("git_describe", "")
    if not _is_git_object_id(commit):
        errors.append("provenance git_commit is not a full Git object ID")
    if not _is_git_object_id(tree):
        errors.append("provenance git_tree is not a full Git object ID")
    if not git_describe:
        errors.append("provenance git_describe is missing")
    elif git_describe == "dirty" or git_describe.endswith("-dirty"):
        errors.append("provenance reports a dirty Git tree")

    try:
        recorded_replicas = _parse_csv_ints(provenance.get("replicas", ""), "replicas")
        recorded_particles = _parse_csv_ints(
            provenance.get("particle_counts", ""), "particle_counts"
        )
    except ValueError as error:
        errors.append(str(error))
        recorded_replicas = []
        recorded_particles = []
    if recorded_replicas != list(replicas):
        errors.append("provenance replicas do not match the expected ordered grid")
    if recorded_particles != list(particle_counts):
        errors.append("provenance particle_counts do not match the expected ordered grid")
    if provenance.get("slurm_array_task_id") != str(task.task_index):
        errors.append("provenance Slurm array task ID does not match task mapping")
    array_job_id = provenance.get("slurm_array_job_id", "")
    if not array_job_id:
        errors.append("provenance Slurm array job ID is missing")
    repo_root = provenance.get("repo_root", "")
    if not repo_root:
        errors.append("provenance repo_root is missing")
    else:
        recorded_root = Path(repo_root).resolve()
        for source, expected_hash in source_hashes.items():
            source_path = recorded_root / source
            try:
                if not source_path.is_file():
                    errors.append(f"recorded source file is unavailable: {source}")
                elif _sha256_file(source_path) != expected_hash:
                    errors.append(f"recorded source hash does not verify current file: {source}")
            except OSError as error:
                errors.append(f"could not verify recorded source file {source}: {error}")

    if result.get("task") != asdict(task):
        errors.append("result task does not exactly match the expected replica-major mapping")
    expected_config_payload = asdict(expected_config)
    if result.get("config") != expected_config_payload:
        errors.append("result config does not exactly match the frozen generator config")

    bundle_entry: dict[str, Any] | None = None
    artifacts = result.get("artifacts")
    if isinstance(artifacts, dict):
        try:
            dataset_path = _resolve_recorded_path(artifacts.get("dataset"), repo_root)
            metadata_path = _resolve_recorded_path(artifacts.get("metadata"), repo_root)
            manifest_path = _resolve_recorded_path(
                artifacts.get("sha256_manifest"), repo_root
            )
            stem = (
                f"nbody_fixedmass_n{task.n_particles}_d{task.n_dims}"
                f"_replica{task.replica}"
            )
            if dataset_path.name != f"{stem}.npz":
                errors.append("recorded dataset filename does not match the expected task stem")
            if metadata_path != dataset_path.with_suffix(".metadata.json"):
                errors.append("recorded metadata path is not the dataset sidecar")
            if manifest_path != dataset_path.with_suffix(".sha256.json"):
                errors.append("recorded SHA manifest path is not the dataset sidecar")
            if len({dataset_path.parent, metadata_path.parent, manifest_path.parent}) != 1:
                errors.append("recorded bundle files do not share one directory")

            loaded = load_confirmatory_bundle(dataset_path)
            if loaded.dataset.config != expected_config:
                errors.append("loaded bundle config does not match the expected task config")
            if loaded.provenance.dataset_path.resolve() != dataset_path:
                errors.append("loader dataset path does not match the recorded path")
            if loaded.provenance.metadata_path.resolve() != metadata_path:
                errors.append("loader metadata path does not match the recorded path")
            if loaded.provenance.sha256_manifest_path.resolve() != manifest_path:
                errors.append("loader SHA manifest path does not match the recorded path")
            if result.get("validation") != loaded.validation:
                errors.append("recorded validation report does not match the reloaded bundle")

            recorded_file_hashes = artifacts.get("file_sha256")
            if recorded_file_hashes != loaded.provenance.file_sha256:
                errors.append("recorded bundle file hashes do not match loader-verified hashes")
            if not isinstance(recorded_file_hashes, dict) or not all(
                _is_sha256(value) for value in recorded_file_hashes.values()
            ):
                errors.append("recorded bundle file hashes are malformed")

            dataset_file_sha256 = loaded.provenance.file_sha256.get(dataset_path.name)
            metadata_file_sha256 = loaded.provenance.file_sha256.get(metadata_path.name)
            if not _is_sha256(dataset_file_sha256) or not _is_sha256(metadata_file_sha256):
                errors.append("loader did not return the expected NPZ and metadata hashes")
            manifest_file_sha256 = _sha256_file(manifest_path)
            content_sha256 = _dataset_content_sha256(loaded)

            bundle_entry = {
                "task_index": task.task_index,
                "phase": phase,
                "replica": task.replica,
                "n_particles": task.n_particles,
                "n_dims": task.n_dims,
                "D": expected_config.state_dim,
                "paths": {
                    "dataset": str(dataset_path),
                    "metadata": str(metadata_path),
                    "sha256_manifest": str(manifest_path),
                },
                "hashes": {
                    "dataset_file_sha256": dataset_file_sha256,
                    "metadata_file_sha256": metadata_file_sha256,
                    "sha256_manifest_file_sha256": manifest_file_sha256,
                    "dataset_content_sha256": content_sha256,
                },
                "config": expected_config_payload,
                "validation": loaded.validation,
                "provenance": {
                    "git_commit": commit,
                    "git_tree": tree,
                    "git_describe": git_describe,
                    "submodules_sha256": _sha256_bytes(submodules.encode()),
                    "source_manifest_sha256": _sha256_file(required["source_hashes"]),
                    "source_hashes_sha256": _sha256_json(source_hashes),
                    "result_file_sha256": _sha256_file(required["result"]),
                    "slurm_array_job_id": array_job_id,
                    "slurm_array_task_id": str(task.task_index),
                    "repo_root": repo_root,
                },
                "unique_content": None,
                "eligible_for_catalog": False,
            }
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            errors.append(f"bundle verification failed: {error}")
    else:
        errors.append("result artifacts are unavailable for bundle verification")

    source_manifest_sha256 = _sha256_file(required["source_hashes"])
    if errors:
        record["errors"].extend(errors)
        return _LoadedTask(
            record=record,
            commit=commit,
            tree=tree,
            submodules=submodules,
            source_hashes=source_hashes,
            source_manifest_sha256=source_manifest_sha256,
            array_job_id=array_job_id,
            repo_root=repo_root,
        )

    if bundle_entry is None:
        record["errors"].append("bundle verification produced no catalog entry")
        return _LoadedTask(record)

    record["status"] = "valid"
    return _LoadedTask(
        record=record,
        bundle=bundle_entry,
        commit=commit,
        tree=tree,
        submodules=submodules,
        source_hashes=source_hashes,
        source_manifest_sha256=source_manifest_sha256,
        array_job_id=array_job_id,
        repo_root=repo_root,
    )
def aggregate(
    run_root: Path,
    replicas: Sequence[int] = DEFAULT_REPLICAS,
    particle_counts: Sequence[int] = DEFAULT_PARTICLE_COUNTS,
    *,
    development_replicas: Sequence[int] = DEFAULT_DEVELOPMENT_REPLICAS,
    n_dims: int = 3,
    n_trajectories: int = 100,
    steps_per_trajectory: int = 100,
    dt: float = 0.01,
    mass_seed: int = 1729,
    trajectory_seed: int = 2718,
    split_seed: int = 31415,
    validation_seed: int = 1618,
) -> dict[str, Any]:
    """Verify an expected F02 data array and return a downstream-safe catalog."""

    replicas = list(replicas)
    particle_counts = list(particle_counts)
    development_replicas = list(development_replicas)
    tasks = generation_tasks(replicas, particle_counts, n_dims=n_dims)
    if len(set(development_replicas)) != len(development_replicas):
        raise ValueError("development_replicas must not contain duplicates")
    if not set(development_replicas).issubset(replicas):
        raise ValueError("development_replicas must be a subset of replicas")
    development_set = set(development_replicas)

    loaded_tasks: list[_LoadedTask] = []
    for task in tasks:
        phase = "development" if task.replica in development_set else "confirmatory"
        config = _expected_config(
            task,
            n_trajectories=n_trajectories,
            steps_per_trajectory=steps_per_trajectory,
            dt=dt,
            mass_seed=mass_seed,
            trajectory_seed=trajectory_seed,
            split_seed=split_seed,
            validation_seed=validation_seed,
        )
        loaded_tasks.append(
            _load_task(
                run_root / f"task-{task.task_index}",
                task,
                phase,
                config,
                replicas=replicas,
                particle_counts=particle_counts,
            )
        )

    expected_paths = {Path(task.record["path"]) for task in loaded_tasks}
    unexpected_paths = (
        sorted(path for path in run_root.glob("task-*") if path not in expected_paths)
        if run_root.is_dir()
        else []
    )
    unexpected_records = [
        {
            "task_index": None,
            "expected_task": None,
            "phase": None,
            "path": str(path),
            "status": "unexpected",
            "exit_code": None,
            "eligible_for_catalog": False,
            "errors": ["path is not part of the declared replica-major task grid"],
        }
        for path in unexpected_paths
    ]

    locally_valid = [task for task in loaded_tasks if task.record["status"] == "valid"]
    provenance_values = {
        "commit": {task.commit for task in locally_valid},
        "tree": {task.tree for task in locally_valid},
        "submodules": {task.submodules for task in locally_valid},
        "source_hashes": {
            _sha256_json(task.source_hashes) for task in locally_valid
        },
        "source_manifest": {
            task.source_manifest_sha256 for task in locally_valid
        },
        "array_job_id": {task.array_job_id for task in locally_valid},
        "repo_root": {task.repo_root for task in locally_valid},
    }
    same_provenance = {
        name: bool(locally_valid) and len(values) == 1
        for name, values in provenance_values.items()
    }
    common_provenance = bool(locally_valid) and all(same_provenance.values())
    if not common_provenance:
        for task in locally_valid:
            task.record["errors"].append(
                "cross-task commit/tree/submodule/source/Slurm provenance is inconsistent"
            )

    bundles = [task.bundle for task in locally_valid if task.bundle is not None]
    content_counts: dict[str, int] = {}
    for bundle in bundles:
        content_hash = bundle["hashes"]["dataset_content_sha256"]
        content_counts[content_hash] = content_counts.get(content_hash, 0) + 1
    duplicate_hashes = sorted(
        content_hash for content_hash, count in content_counts.items() if count > 1
    )
    duplicate_set = set(duplicate_hashes)
    for task in locally_valid:
        assert task.bundle is not None
        content_hash = task.bundle["hashes"]["dataset_content_sha256"]
        unique = content_hash not in duplicate_set
        task.bundle["unique_content"] = unique
        eligible = common_provenance and unique
        task.bundle["eligible_for_catalog"] = eligible
        task.record["eligible_for_catalog"] = eligible
        if not unique:
            task.record["errors"].append(
                "dataset semantic content is duplicated across expected tasks"
            )

    all_expected_tasks_valid = len(locally_valid) == len(tasks)
    all_expected_tasks_unique = all_expected_tasks_valid and not duplicate_hashes
    overall_ready = bool(
        all_expected_tasks_valid
        and not unexpected_records
        and common_provenance
        and all_expected_tasks_unique
    )

    task_records = [task.record for task in loaded_tasks] + unexpected_records
    status_counts = {
        status: sum(record["status"] == status for record in task_records)
        for status in ("valid", "missing", "failed", "invalid", "unexpected")
    }
    phase_counts = {
        phase: sum(bundle["phase"] == phase for bundle in bundles)
        for phase in ("development", "confirmatory")
    }

    common_task = locally_valid[0] if locally_valid else None
    provenance = {
        "verified": common_provenance,
        "same_commit": same_provenance["commit"],
        "same_tree": same_provenance["tree"],
        "same_submodules": same_provenance["submodules"],
        "same_source_hashes": same_provenance["source_hashes"],
        "same_source_manifest": same_provenance["source_manifest"],
        "same_slurm_array_job": same_provenance["array_job_id"],
        "same_repo_root": same_provenance["repo_root"],
        "git_commit": common_task.commit if common_task and same_provenance["commit"] else None,
        "git_tree": common_task.tree if common_task and same_provenance["tree"] else None,
        "submodules_sha256": (
            _sha256_bytes(common_task.submodules.encode())
            if common_task and same_provenance["submodules"]
            else None
        ),
        "source_manifest_sha256": (
            common_task.source_manifest_sha256
            if common_task and same_provenance["source_manifest"]
            else None
        ),
        "source_hashes_sha256": (
            _sha256_json(common_task.source_hashes)
            if common_task and same_provenance["source_hashes"]
            else None
        ),
        "slurm_array_job_id": (
            common_task.array_job_id
            if common_task and same_provenance["array_job_id"]
            else None
        ),
        "repo_root": (
            common_task.repo_root
            if common_task and same_provenance["repo_root"]
            else None
        ),
    }

    return {
        "schema_version": 1,
        "catalog_type": "f02_nbody_confirmatory_data",
        "overall_ready": overall_ready,
        "input": {
            "run_root": str(run_root),
            "replicas": replicas,
            "development_replicas": development_replicas,
            "particle_counts": particle_counts,
            "n_dims": n_dims,
            "generation": {
                "n_trajectories": n_trajectories,
                "steps_per_trajectory": steps_per_trajectory,
                "dt": dt,
                "mass_seed": mass_seed,
                "trajectory_seed": trajectory_seed,
                "split_seed": split_seed,
                "validation_seed": validation_seed,
            },
        },
        "task_accounting": {
            "expected_task_count": len(tasks),
            "all_expected_tasks_valid": all_expected_tasks_valid,
            "all_expected_tasks_unique": all_expected_tasks_unique,
            "status_counts": status_counts,
            "tasks": task_records,
        },
        "provenance": provenance,
        "independence": {
            "verified": bool(bundles) and not duplicate_hashes,
            "candidate_bundle_count": len(bundles),
            "unique_bundle_count": sum(
                bundle["unique_content"] is True for bundle in bundles
            ),
            "duplicate_dataset_content_sha256": duplicate_hashes,
        },
        "catalog": {
            "bundle_count": len(bundles),
            "phase_counts": phase_counts,
            "bundles": bundles,
        },
    }


def _arg_csv_ints(value: str) -> list[int]:
    try:
        return _parse_csv_ints(value, "value")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_root",
        type=Path,
        help="job directory containing task-<array-index> directories",
    )
    parser.add_argument(
        "--replicas",
        type=_arg_csv_ints,
        default=list(DEFAULT_REPLICAS),
    )
    parser.add_argument(
        "--development-replicas",
        type=_arg_csv_ints,
        default=list(DEFAULT_DEVELOPMENT_REPLICAS),
    )
    parser.add_argument(
        "--particle-counts",
        type=_arg_csv_ints,
        default=list(DEFAULT_PARTICLE_COUNTS),
    )
    parser.add_argument("--n-dims", type=int, default=3)
    parser.add_argument("--n-trajectories", type=int, default=100)
    parser.add_argument("--steps-per-trajectory", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--mass-seed", type=int, default=1729)
    parser.add_argument("--trajectory-seed", type=int, default=2718)
    parser.add_argument("--split-seed", type=int, default=31415)
    parser.add_argument("--validation-seed", type=int, default=1618)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = aggregate(
        args.run_root,
        args.replicas,
        args.particle_counts,
        development_replicas=args.development_replicas,
        n_dims=args.n_dims,
        n_trajectories=args.n_trajectories,
        steps_per_trajectory=args.steps_per_trajectory,
        dt=args.dt,
        mass_seed=args.mass_seed,
        trajectory_seed=args.trajectory_seed,
        split_seed=args.split_seed,
        validation_seed=args.validation_seed,
    )
    _write_json_atomic(args.out, report)
    counts = report["task_accounting"]["status_counts"]
    print(
        f"wrote {args.out}; valid={counts['valid']} missing={counts['missing']} "
        f"failed={counts['failed']} invalid={counts['invalid']} "
        f"unexpected={counts['unexpected']}"
    )
    print(f"overall_ready={report['overall_ready']}")


if __name__ == "__main__":
    main()
