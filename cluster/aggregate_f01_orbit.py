"""Offline, provenance-strict aggregation for F01 Slurm array tasks.

The statistical unit is one independently simulated dataset, identified by the
SHA-256 fingerprint written by ``run_f01_orbit.py``.  Wall-clock measurements
are deliberately excluded from all aggregation and uncertainty calculations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCOPE = "mechanism experiment only; not a SOTA performance claim"
_CONFIG_EXCEPTIONS = {"seed", "out"}
_EXCLUSIVE_VERIFICATION_MODES = {
    "slurm_explicit",
    "slurm_no_oversubscribe_full_node_sole_job",
}


@dataclass
class _LoadedTask:
    record: dict[str, Any]
    config: dict[str, Any]
    normalized_config: dict[str, Any]
    commit: str
    tree: str
    submodules: str
    array_job_id: str
    source_manifest_sha256: str
    packages_sha256: str
    exclusive_verification_mode: str
    same_m_tolerance: float
    instances: list[dict[str, Any]]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


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
        result[key] = value
    return result


def _parse_sha256sum(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text().splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ValueError(f"malformed SHA-256 line: {line!r}")
        digest, source_path = fields
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError(f"invalid SHA-256 digest: {digest!r}") from error
        basename = Path(source_path.lstrip("*")).name
        if not basename or basename in entries:
            raise ValueError(f"duplicate or empty SHA-256 artifact name: {basename!r}")
        entries[basename] = digest
    return entries


def _as_finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} is not numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field} is not finite")
    return converted


def _dataset_hash_is_valid(dataset: dict[str, Any]) -> bool:
    declared = dataset.get("combined_sha256")
    if not isinstance(declared, str) or len(declared) != 64:
        return False
    payload = {key: value for key, value in dataset.items() if key != "combined_sha256"}
    if _sha256_json(payload) != declared:
        return False
    tensors = dataset.get("tensors")
    return bool(tensors) and all(
        isinstance(tensor, dict)
        and isinstance(tensor.get("sha256"), str)
        and len(tensor["sha256"]) == 64
        for tensor in tensors
    )


def _task_record(
    path: Path,
    task_index: int,
    expected_seed: int,
    *,
    status: str,
    errors: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "task_index": task_index,
        "expected_seed": expected_seed,
        "path": str(path),
        "status": status,
        "eligible_for_pool": False,
        "errors": list(errors),
    }


def _load_task(path: Path, task_index: int, expected_seed: int) -> _LoadedTask:
    if not path.exists():
        return _LoadedTask(
            record=_task_record(path, task_index, expected_seed, status="missing"),
            config={},
            normalized_config={},
            commit="",
            tree="",
            submodules="",
            array_job_id="",
            source_manifest_sha256="",
            packages_sha256="",
            exclusive_verification_mode="",
            same_m_tolerance=math.nan,
            instances=[],
        )
    if not path.is_dir():
        return _LoadedTask(
            record=_task_record(
                path,
                task_index,
                expected_seed,
                status="invalid",
                errors=("task path is not a directory",),
            ),
            config={},
            normalized_config={},
            commit="",
            tree="",
            submodules="",
            array_job_id="",
            source_manifest_sha256="",
            packages_sha256="",
            exclusive_verification_mode="",
            same_m_tolerance=math.nan,
            instances=[],
        )

    exit_code_path = path / "exit-code.txt"
    if exit_code_path.exists():
        try:
            exit_code = int(exit_code_path.read_text().strip())
        except ValueError:
            exit_code = None
        if exit_code is None:
            record = _task_record(
                path,
                task_index,
                expected_seed,
                status="invalid",
                errors=("exit-code.txt is not an integer",),
            )
            return _empty_loaded(record)
        if exit_code != 0:
            record = _task_record(
                path,
                task_index,
                expected_seed,
                status="failed",
                errors=(f"task exited with status {exit_code}",),
            )
            record["exit_code"] = exit_code
            return _empty_loaded(record)
    else:
        record = _task_record(
            path,
            task_index,
            expected_seed,
            status="invalid",
            errors=("missing exit-code.txt",),
        )
        return _empty_loaded(record)

    required = {
        "result": path / "result.json",
        "datasets": path / "datasets.json",
        "runtime": path / "runtime.json",
        "provenance": path / "provenance.env",
        "git_status": path / "git-status.txt",
        "submodules": path / "git-submodules.txt",
        "source_hashes": path / "source-files.sha256",
        "artifact_hashes": path / "artifacts.sha256",
    }
    missing = [name for name, required_path in required.items() if not required_path.is_file()]
    if missing:
        record = _task_record(
            path,
            task_index,
            expected_seed,
            status="invalid",
            errors=(f"missing required artifacts: {', '.join(sorted(missing))}",),
        )
        return _empty_loaded(record)

    errors: list[str] = []
    try:
        result = _read_json(required["result"])
        datasets = _read_json(required["datasets"])
        runtime = _read_json(required["runtime"])
        provenance = _parse_env(required["provenance"])
        artifact_hashes = _parse_sha256sum(required["artifact_hashes"])
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        record = _task_record(
            path,
            task_index,
            expected_seed,
            status="invalid",
            errors=(f"could not parse task artifacts: {error}",),
        )
        return _empty_loaded(record)

    for artifact_name in (
        "datasets.json",
        "runtime.json",
        "result.json",
        "source-files.sha256",
    ):
        artifact_path = path / artifact_name
        if artifact_hashes.get(artifact_name) != _sha256_bytes(artifact_path.read_bytes()):
            errors.append(f"artifacts.sha256 does not verify {artifact_name}")
    source_manifest_sha256 = _sha256_bytes(required["source_hashes"].read_bytes())

    if required["git_status"].read_text().strip():
        errors.append("git-status.txt is not clean")
    submodule_lines = [
        line.rstrip() for line in required["submodules"].read_text().splitlines() if line.strip()
    ]
    if not submodule_lines:
        errors.append("git-submodules.txt is empty")
    elif any(not line.startswith(" ") for line in submodule_lines):
        errors.append("git-submodules.txt reports a modified or uninitialized submodule")
    submodules = "\n".join(line.strip() for line in submodule_lines)

    config = result.get("config")
    preregistration = result.get("preregistration")
    cluster_provenance = result.get("cluster_provenance")
    rows = result.get("rows")
    if not isinstance(config, dict):
        errors.append("result config is missing or malformed")
        config = {}
    if not isinstance(preregistration, dict):
        errors.append("result preregistration is missing or malformed")
        preregistration = {}
    if not isinstance(cluster_provenance, dict):
        errors.append("cluster_provenance is missing or malformed")
        cluster_provenance = {}
    if not isinstance(rows, list):
        errors.append("result rows are missing or malformed")
        rows = []

    if cluster_provenance.get("exclusive_node_verified") is not True:
        errors.append("exclusive_node_verified is not true")
    exclusive_verification_mode = cluster_provenance.get("exclusive_node_verification_mode")
    if exclusive_verification_mode not in _EXCLUSIVE_VERIFICATION_MODES:
        errors.append("exclusive node verification mode is missing or unsupported")
        exclusive_verification_mode = ""
    if provenance.get("exclusive_verification_mode") != exclusive_verification_mode:
        errors.append("exclusive verification mode is not linked into task provenance")
    selected_environment = runtime.get("selected_environment")
    if not isinstance(selected_environment, dict):
        errors.append("runtime selected_environment is missing or malformed")
    elif (
        selected_environment.get("F01_SLURM_EXCLUSIVE_VERIFIED") != "1"
        or selected_environment.get("F01_SLURM_EXCLUSIVE_MODE") != exclusive_verification_mode
    ):
        errors.append("runtime environment does not verify the declared exclusive mode")
    if cluster_provenance.get("wall_time_is_inferential") is not False:
        errors.append("cluster wall_time_is_inferential is not false")
    if preregistration.get("wall_time_is_inferential") is not False:
        errors.append("preregistration wall_time_is_inferential is not false")

    commit = provenance.get("git_commit", "")
    tree = provenance.get("git_tree", "")
    if not commit:
        errors.append("git_commit is missing")
    if not tree:
        errors.append("git_tree is missing")
    if "dirty" in provenance.get("git_describe", "").lower():
        errors.append("git_describe reports a dirty worktree")
    if provenance.get("array_task_id") != str(task_index):
        errors.append("array task id does not match the expected index")
    if not cluster_provenance.get("slurm_job_id"):
        errors.append("Slurm job id is missing")
    if str(cluster_provenance.get("slurm_array_task_id")) != str(task_index):
        errors.append("result Slurm array task id does not match the expected index")
    array_job_id = str(cluster_provenance.get("slurm_array_job_id") or "")
    if not array_job_id or provenance.get("array_job_id") != array_job_id:
        errors.append("Slurm array job ids are missing or inconsistent")

    try:
        config_seed = int(config.get("seed"))
    except (TypeError, ValueError):
        config_seed = None
    try:
        manifest_seed = int(datasets.get("base_seed"))
    except (AttributeError, TypeError, ValueError):
        manifest_seed = None
    try:
        env_seed = int(provenance.get("seed", ""))
    except ValueError:
        env_seed = None
    if {config_seed, manifest_seed, env_seed} != {expected_seed}:
        errors.append("expected, config, manifest, and provenance seeds do not match")

    if _sha256_bytes(required["datasets"].read_bytes()) != cluster_provenance.get(
        "dataset_manifest_sha256"
    ):
        errors.append("dataset manifest SHA-256 does not match result provenance")
    if _sha256_bytes(required["runtime"].read_bytes()) != cluster_provenance.get(
        "runtime_manifest_sha256"
    ):
        errors.append("runtime manifest SHA-256 does not match result provenance")

    packages = runtime.get("packages") if isinstance(runtime, dict) else None
    runtime_package_hash = runtime.get("packages_sha256") if isinstance(runtime, dict) else None
    if not isinstance(packages, list) or _sha256_json(packages) != runtime_package_hash:
        errors.append("runtime package manifest SHA-256 is invalid")
    if cluster_provenance.get("packages_sha256") != runtime_package_hash:
        errors.append("package SHA-256 is not linked into result provenance")

    dataset_records = datasets.get("datasets") if isinstance(datasets, dict) else None
    repeats = config.get("repeats")
    if not isinstance(repeats, int) or repeats <= 0:
        errors.append("config repeats is not a positive integer")
        repeats = 0
    if not isinstance(dataset_records, list):
        errors.append("dataset records are missing or malformed")
        dataset_records = []
    if len(dataset_records) != repeats:
        errors.append("dataset record count does not equal config repeats")
    dataset_by_repeat: dict[int, dict[str, Any]] = {}
    for dataset in dataset_records:
        if not isinstance(dataset, dict) or not _dataset_hash_is_valid(dataset):
            errors.append("a dataset fingerprint is malformed or has an invalid hash")
            continue
        repeat = dataset.get("repeat")
        if not isinstance(repeat, int) or repeat in dataset_by_repeat:
            errors.append("dataset repeat ids are malformed or duplicated")
            continue
        dataset_by_repeat[repeat] = dataset
    if set(dataset_by_repeat) != set(range(repeats)):
        errors.append("dataset repeat ids are not complete and zero-based")

    m_values = config.get("m_values")
    tera_max_m = config.get("tera_max_m")
    if (
        not isinstance(m_values, list)
        or not m_values
        or any(not isinstance(value, int) for value in m_values)
        or not isinstance(tera_max_m, int)
    ):
        errors.append("m_values or tera_max_m is malformed")
        m_values = []
        tera_max_m = 0

    row_maps: dict[int, dict[str, dict[int, dict[str, Any]]]] = {
        repeat: {"ORBIT-exact": {}, "TERA": {}} for repeat in range(repeats)
    }
    for row in rows:
        if not isinstance(row, dict) or row.get("method") not in {"ORBIT-exact", "TERA"}:
            continue
        repeat = row.get("repeat")
        m_value = row.get("m")
        method = row.get("method")
        if not isinstance(repeat, int) or repeat not in row_maps or not isinstance(m_value, int):
            errors.append("a method row has an invalid repeat or m")
            continue
        if m_value in row_maps[repeat][method]:
            errors.append("duplicate method/repeat/m row")
            continue
        row_maps[repeat][method][m_value] = row

    expected_orbit = set(m_values)
    expected_tera = {value for value in m_values if value <= tera_max_m}
    for repeat, methods in row_maps.items():
        if set(methods["ORBIT-exact"]) != expected_orbit:
            errors.append(f"repeat {repeat} has incomplete ORBIT rows")
        if set(methods["TERA"]) != expected_tera:
            errors.append(f"repeat {repeat} has incomplete TERA rows")
        for m_value in expected_tera & set(methods["ORBIT-exact"]):
            orbit_row = methods["ORBIT-exact"][m_value]
            for field in (
                "maxabs_mean_to_same_m_tera",
                "maxabs_variance_to_same_m_tera",
            ):
                try:
                    _as_finite_number(orbit_row.get(field), field)
                except ValueError as error:
                    errors.append(str(error))
        for orbit_row in methods["ORBIT-exact"].values():
            for field in (
                "cg_converged_fraction",
                "cg_relative_residual_max",
                "variance_valid_fraction",
                "variance_min_raw",
            ):
                try:
                    _as_finite_number(orbit_row.get(field), field)
                except ValueError as error:
                    errors.append(str(error))

    try:
        tolerance = _as_finite_number(
            preregistration.get("same_m_equivalence_tolerance"),
            "same_m_equivalence_tolerance",
        )
    except ValueError as error:
        errors.append(str(error))
        tolerance = math.nan
    if tolerance < 0.0:
        errors.append("same-m equivalence tolerance is negative")

    if errors:
        record = _task_record(
            path,
            task_index,
            expected_seed,
            status="invalid",
            errors=errors,
        )
        return _empty_loaded(record)

    instances = [
        {
            "task_index": task_index,
            "base_seed": expected_seed,
            "repeat": repeat,
            "dataset_sha256": dataset_by_repeat[repeat]["combined_sha256"],
            "dataset_content_sha256": _sha256_json(
                {
                    key: value
                    for key, value in dataset_by_repeat[repeat].items()
                    if key not in {"combined_sha256", "repeat"}
                }
            ),
            "orbit": row_maps[repeat]["ORBIT-exact"],
            "tera": row_maps[repeat]["TERA"],
        }
        for repeat in range(repeats)
    ]
    normalized_config = {
        key: value for key, value in config.items() if key not in _CONFIG_EXCEPTIONS
    }
    record = _task_record(path, task_index, expected_seed, status="valid")
    record.update(
        {
            "base_seed": expected_seed,
            "repeat_count": repeats,
            "dataset_sha256": [instance["dataset_sha256"] for instance in instances],
            "git_commit": commit,
            "git_tree": tree,
            "slurm_array_job_id": array_job_id,
            "source_manifest_sha256": source_manifest_sha256,
            "packages_sha256": runtime_package_hash,
            "exclusive_verification_mode": exclusive_verification_mode,
            "normalized_config_sha256": _sha256_json(normalized_config),
        }
    )
    return _LoadedTask(
        record=record,
        config=config,
        normalized_config=normalized_config,
        commit=commit,
        tree=tree,
        submodules=submodules,
        array_job_id=array_job_id,
        source_manifest_sha256=source_manifest_sha256,
        packages_sha256=runtime_package_hash,
        exclusive_verification_mode=exclusive_verification_mode,
        same_m_tolerance=tolerance,
        instances=instances,
    )


def _empty_loaded(record: dict[str, Any]) -> _LoadedTask:
    return _LoadedTask(
        record=record,
        config={},
        normalized_config={},
        commit="",
        tree="",
        submodules="",
        array_job_id="",
        source_manifest_sha256="",
        packages_sha256="",
        exclusive_verification_mode="",
        same_m_tolerance=math.nan,
        instances=[],
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a quantile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any] | None:
    if not baseline or len(baseline) != len(candidate):
        return None
    rng = random.Random(seed)
    baseline_means: list[float] = []
    candidate_means: list[float] = []
    improvement_means: list[float] = []
    size = len(baseline)
    for _ in range(samples):
        indices = [rng.randrange(size) for _ in range(size)]
        baseline_mean = statistics.fmean(baseline[index] for index in indices)
        candidate_mean = statistics.fmean(candidate[index] for index in indices)
        baseline_means.append(baseline_mean)
        candidate_means.append(candidate_mean)
        improvement_means.append(baseline_mean - candidate_mean)
    return {
        "method": "paired nonparametric bootstrap percentile",
        "confidence_level": 0.95,
        "samples": samples,
        "seed": seed,
        "baseline_mean_kl_ci": [
            _quantile(baseline_means, 0.025),
            _quantile(baseline_means, 0.975),
        ],
        "candidate_mean_kl_ci": [
            _quantile(candidate_means, 0.025),
            _quantile(candidate_means, 0.975),
        ],
        "mean_paired_improvement_ci": [
            _quantile(improvement_means, 0.025),
            _quantile(improvement_means, 0.975),
        ],
        "resampling_unit": "independent simulated dataset",
    }


def _gate_status(has_evidence: bool, observed_pass: bool) -> str:
    if not has_evidence:
        return "insufficient"
    return "pass" if observed_pass else "fail"


def aggregate(
    run_root: Path,
    expected_seeds: Sequence[int],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260803,
) -> dict[str, Any]:
    """Aggregate expected task directories without dropping failed tasks."""

    if not expected_seeds:
        raise ValueError("expected_seeds must not be empty")
    if len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("expected_seeds must be unique")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in expected_seeds
    ):
        raise ValueError("expected_seeds must contain non-negative integers")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    loaded_tasks = [
        _load_task(
            run_root / f"seed-{seed}-task-{task_index}",
            task_index,
            seed,
        )
        for task_index, seed in enumerate(expected_seeds)
    ]
    expected_paths = {Path(task.record["path"]) for task in loaded_tasks}
    unexpected_paths = (
        sorted(path for path in run_root.glob("seed-*-task-*") if path not in expected_paths)
        if run_root.is_dir()
        else []
    )
    unexpected_records = [
        {
            "task_index": None,
            "expected_seed": None,
            "path": str(path),
            "status": "unexpected",
            "eligible_for_pool": False,
            "errors": ["directory is not part of the declared expected seed array"],
        }
        for path in unexpected_paths
    ]

    structurally_valid = [task for task in loaded_tasks if task.record["status"] == "valid"]
    dimension_values = {
        "commit": {task.commit for task in structurally_valid},
        "tree": {task.tree for task in structurally_valid},
        "submodules": {task.submodules for task in structurally_valid},
        "array_job_id": {task.array_job_id for task in structurally_valid},
        "source_manifest": {task.source_manifest_sha256 for task in structurally_valid},
        "config": {_sha256_json(task.normalized_config) for task in structurally_valid},
        "packages": {task.packages_sha256 for task in structurally_valid},
        "exclusive_verification_mode": {
            task.exclusive_verification_mode for task in structurally_valid
        },
        "same_m_tolerance": {task.same_m_tolerance for task in structurally_valid},
    }
    same_dimensions = {
        name: bool(structurally_valid) and len(values) == 1
        for name, values in dimension_values.items()
    }
    consistent_across_valid_tasks = bool(structurally_valid) and all(same_dimensions.values())
    all_expected_tasks_valid = len(structurally_valid) == len(expected_seeds)
    provenance_verified = (
        all_expected_tasks_valid and not unexpected_records and consistent_across_valid_tasks
    )

    if consistent_across_valid_tasks:
        for task in structurally_valid:
            task.record["eligible_for_pool"] = True
        candidate_instances = [
            instance for task in structurally_valid for instance in task.instances
        ]
    else:
        candidate_instances = []
        if structurally_valid:
            for task in structurally_valid:
                task.record["errors"].append(
                    "cross-task commit/tree/submodule/config/runtime provenance is inconsistent"
                )

    hash_counts: dict[str, int] = {}
    for instance in candidate_instances:
        dataset_hash = instance["dataset_content_sha256"]
        hash_counts[dataset_hash] = hash_counts.get(dataset_hash, 0) + 1
    duplicate_hashes = sorted(
        dataset_hash for dataset_hash, count in hash_counts.items() if count > 1
    )
    if duplicate_hashes:
        duplicate_set = set(duplicate_hashes)
        for task in structurally_valid:
            if any(
                instance["dataset_content_sha256"] in duplicate_set for instance in task.instances
            ):
                task.record["eligible_for_pool"] = False
                task.record["errors"].append(
                    "one or more dataset contents are duplicated across statistical units"
                )
    independent_instances = [
        instance
        for instance in candidate_instances
        if hash_counts[instance["dataset_content_sha256"]] == 1
    ]
    independence_pass = bool(candidate_instances) and not duplicate_hashes
    analysis_ready = provenance_verified and independence_pass

    if structurally_valid and consistent_across_valid_tasks:
        common_config = structurally_valid[0].config
        tolerance = structurally_valid[0].same_m_tolerance
    else:
        common_config = {}
        tolerance = math.nan

    h1_values: list[tuple[float, float]] = []
    orbit_rows: list[dict[str, Any]] = []
    for instance in independent_instances:
        orbit_rows.extend(instance["orbit"].values())
        for m_value in sorted(set(instance["orbit"]) & set(instance["tera"])):
            row = instance["orbit"][m_value]
            h1_values.append(
                (
                    _as_finite_number(
                        row.get("maxabs_mean_to_same_m_tera"),
                        "maxabs_mean_to_same_m_tera",
                    ),
                    _as_finite_number(
                        row.get("maxabs_variance_to_same_m_tera"),
                        "maxabs_variance_to_same_m_tera",
                    ),
                )
            )

    h1_dtype_eligible = common_config.get("dtype") == "float64"
    h1_has_evidence = bool(h1_values) and h1_dtype_eligible and math.isfinite(tolerance)
    h1_observed_pass = h1_has_evidence and all(
        mean_error <= tolerance and variance_error <= tolerance
        for mean_error, variance_error in h1_values
    )

    cg_tolerance = common_config.get("cg_tolerance")
    solver_values: list[tuple[float, float]] = []
    variance_values: list[tuple[float, float]] = []
    for row in orbit_rows:
        solver_values.append(
            (
                _as_finite_number(row.get("cg_converged_fraction"), "cg_converged_fraction"),
                _as_finite_number(row.get("cg_relative_residual_max"), "cg_relative_residual_max"),
            )
        )
        variance_values.append(
            (
                _as_finite_number(row.get("variance_valid_fraction"), "variance_valid_fraction"),
                _as_finite_number(row.get("variance_min_raw"), "variance_min_raw"),
            )
        )
    solver_has_evidence = (
        bool(solver_values)
        and isinstance(cg_tolerance, (int, float))
        and not isinstance(cg_tolerance, bool)
        and math.isfinite(float(cg_tolerance))
        and float(cg_tolerance) >= 0.0
    )
    solver_observed_pass = solver_has_evidence and all(
        converged == 1.0 and residual <= float(cg_tolerance)
        for converged, residual in solver_values
    )
    variance_has_evidence = bool(variance_values)
    variance_observed_pass = variance_has_evidence and all(
        valid == 1.0 and minimum > 0.0 for valid, minimum in variance_values
    )

    m_values = common_config.get("m_values") if common_config else None
    tera_max_m = common_config.get("tera_max_m") if common_config else None
    n_train = common_config.get("n_train") if common_config else None
    reference_m: int | None = None
    candidate_m: int | None = None
    if (
        isinstance(m_values, list)
        and m_values
        and all(isinstance(value, int) for value in m_values)
        and isinstance(tera_max_m, int)
    ):
        paired_m = [value for value in m_values if value <= tera_max_m]
        reference_m = max(paired_m, default=None)
        candidate_m = max(m_values)
    nontrivial_candidate = bool(
        reference_m is not None
        and candidate_m is not None
        and isinstance(n_train, int)
        and reference_m < candidate_m < n_train
    )

    baseline_kls: list[float] = []
    candidate_kls: list[float] = []
    if reference_m is not None and candidate_m is not None:
        for instance in independent_instances:
            baseline_value = instance["orbit"].get(reference_m, {}).get("avg_marginal_kl")
            candidate_value = instance["orbit"].get(candidate_m, {}).get("avg_marginal_kl")
            try:
                baseline_number = _as_finite_number(baseline_value, "baseline avg_marginal_kl")
                candidate_number = _as_finite_number(candidate_value, "candidate avg_marginal_kl")
            except ValueError:
                continue
            if baseline_number < 0.0 or candidate_number < 0.0:
                continue
            baseline_kls.append(baseline_number)
            candidate_kls.append(candidate_number)

    paired_improvements = [
        baseline - candidate
        for baseline, candidate in zip(baseline_kls, candidate_kls, strict=True)
    ]
    enough_h2_instances = len(baseline_kls) >= 3 and len(baseline_kls) == len(independent_instances)
    h2_has_evidence = nontrivial_candidate and enough_h2_instances
    baseline_mean = statistics.fmean(baseline_kls) if baseline_kls else None
    candidate_mean = statistics.fmean(candidate_kls) if candidate_kls else None
    improvement_mean = statistics.fmean(paired_improvements) if paired_improvements else None
    h2_observed_pass = bool(
        h2_has_evidence
        and baseline_mean is not None
        and candidate_mean is not None
        and candidate_mean < baseline_mean
    )
    uncertainty = _paired_bootstrap(
        baseline_kls,
        candidate_kls,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    ci_excludes_zero = bool(uncertainty and uncertainty["mean_paired_improvement_ci"][0] > 0.0)

    h1_confirmatory = analysis_ready and h1_observed_pass
    solver_confirmatory = analysis_ready and solver_observed_pass
    variance_confirmatory = analysis_ready and variance_observed_pass
    h2_confirmatory = analysis_ready and h2_observed_pass
    overall_mechanism_pass = bool(
        h1_confirmatory and solver_confirmatory and variance_confirmatory and h2_confirmatory
    )

    task_records = [task.record for task in loaded_tasks] + unexpected_records
    status_counts = {
        status: sum(record["status"] == status for record in task_records)
        for status in ("valid", "missing", "failed", "invalid", "unexpected")
    }
    provenance = {
        "verified": provenance_verified,
        "consistent_across_valid_tasks": consistent_across_valid_tasks,
        "same_commit": same_dimensions["commit"],
        "same_tree": same_dimensions["tree"],
        "same_submodules": same_dimensions["submodules"],
        "same_slurm_array_job": same_dimensions["array_job_id"],
        "same_source_manifest": same_dimensions["source_manifest"],
        "same_config_except_seed_and_output": same_dimensions["config"],
        "same_package_manifest": same_dimensions["packages"],
        "same_exclusive_verification_mode": same_dimensions["exclusive_verification_mode"],
        "same_equivalence_tolerance": same_dimensions["same_m_tolerance"],
        "commit": structurally_valid[0].commit if same_dimensions["commit"] else None,
        "tree": structurally_valid[0].tree if same_dimensions["tree"] else None,
        "slurm_array_job_id": (
            structurally_valid[0].array_job_id if same_dimensions["array_job_id"] else None
        ),
        "submodules_sha256": (
            _sha256_bytes(structurally_valid[0].submodules.encode())
            if same_dimensions["submodules"]
            else None
        ),
        "source_manifest_sha256": (
            structurally_valid[0].source_manifest_sha256
            if same_dimensions["source_manifest"]
            else None
        ),
        "normalized_config_sha256": (
            _sha256_json(structurally_valid[0].normalized_config)
            if same_dimensions["config"]
            else None
        ),
        "packages_sha256": (
            structurally_valid[0].packages_sha256 if same_dimensions["packages"] else None
        ),
        "exclusive_verification_mode": (
            structurally_valid[0].exclusive_verification_mode
            if same_dimensions["exclusive_verification_mode"]
            else None
        ),
    }

    return {
        "schema_version": 1,
        "scope": SCOPE,
        "input": {
            "run_root": str(run_root),
            "expected_seeds": list(expected_seeds),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        },
        "task_accounting": {
            "expected_task_count": len(expected_seeds),
            "all_expected_tasks_valid": all_expected_tasks_valid,
            "status_counts": status_counts,
            "tasks": task_records,
        },
        "provenance": provenance,
        "pool": {
            "analysis_ready": analysis_ready,
            "independence_pass": independence_pass,
            "candidate_dataset_instance_count": len(candidate_instances),
            "independent_dataset_instance_count": len(independent_instances),
            "duplicate_dataset_content_sha256": duplicate_hashes,
            "instances": [
                {
                    "task_index": instance["task_index"],
                    "base_seed": instance["base_seed"],
                    "repeat": instance["repeat"],
                    "dataset_sha256": instance["dataset_sha256"],
                    "dataset_content_sha256": instance["dataset_content_sha256"],
                }
                for instance in independent_instances
            ],
        },
        "gates": {
            "h1_same_m_equivalence": {
                "status": _gate_status(h1_has_evidence, h1_observed_pass),
                "observed_pass": h1_observed_pass,
                "confirmatory_pass": h1_confirmatory,
                "requires_float64": True,
                "dtype_eligible": h1_dtype_eligible,
                "tolerance": tolerance if math.isfinite(tolerance) else None,
                "paired_comparison_count": len(h1_values),
                "maxabs_mean": max((value[0] for value in h1_values), default=None),
                "maxabs_variance": max((value[1] for value in h1_values), default=None),
            },
            "solver": {
                "status": _gate_status(solver_has_evidence, solver_observed_pass),
                "observed_pass": solver_observed_pass,
                "confirmatory_pass": solver_confirmatory,
                "cg_tolerance": cg_tolerance,
                "row_count": len(solver_values),
                "max_relative_residual": max((value[1] for value in solver_values), default=None),
                "min_converged_fraction": min((value[0] for value in solver_values), default=None),
            },
            "variance": {
                "status": _gate_status(variance_has_evidence, variance_observed_pass),
                "observed_pass": variance_observed_pass,
                "confirmatory_pass": variance_confirmatory,
                "row_count": len(variance_values),
                "min_valid_fraction": min((value[0] for value in variance_values), default=None),
                "minimum_raw_variance": min((value[1] for value in variance_values), default=None),
            },
            "h2_larger_m_headroom": {
                "status": _gate_status(h2_has_evidence, h2_observed_pass),
                "observed_pass": h2_observed_pass,
                "confirmatory_pass": h2_confirmatory,
                "reference_m": reference_m,
                "candidate_m": candidate_m,
                "candidate_is_nontrivial": nontrivial_candidate,
                "requires_candidate_below_n_train": True,
                "minimum_independent_instances": 3,
                "paired_instance_count": len(baseline_kls),
                "reference_mean_kl": baseline_mean,
                "candidate_mean_kl": candidate_mean,
                "mean_paired_improvement": improvement_mean,
                "uncertainty": uncertainty,
                "bootstrap_ci_excludes_zero": ci_excludes_zero,
            },
            "overall_mechanism_pass": overall_mechanism_pass,
        },
        "timing": {
            "wall_time_is_inferential": False,
            "included_in_hypothesis_tests": False,
            "uncertainty_reported_for_wall_time": False,
            "note": (
                "seconds_descriptive and other wall-time fields are intentionally excluded; "
                "the bootstrap applies only to paired marginal-KL accuracy differences"
            ),
        },
    }


def _parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected seeds must be comma-separated integers"
        ) from error
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("expected seeds must be unique non-negative integers")
    return seeds


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate one F01 Slurm array job without using wall time inferentially"
    )
    parser.add_argument(
        "run_root",
        type=Path,
        help="job directory containing seed-<seed>-task-<index> directories",
    )
    parser.add_argument("--expected-seeds", type=_parse_seeds, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = aggregate(
        args.run_root,
        args.expected_seeds,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    _write_json_atomic(args.out, report)
    counts = report["task_accounting"]["status_counts"]
    print(
        f"wrote {args.out}; valid={counts['valid']} missing={counts['missing']} "
        f"failed={counts['failed']} invalid={counts['invalid']}"
    )
    print(f"overall_mechanism_pass={report['gates']['overall_mechanism_pass']}")


if __name__ == "__main__":
    main()
