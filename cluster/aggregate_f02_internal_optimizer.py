"""Strict aggregation and update selection for the F02 internal Slurm grid.

The aggregator enumerates a caller-declared set of array indices.  It never
discovers the analysis population by globbing successful tasks.  Missing,
failed, malformed, duplicate, and unexpected task directories remain visible
in the report.  The registered TERA update budget is selected only for the
exact, analysis-ready 135-task grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cluster.f02_internal_grid import (  # noqa: E402
    OPTIMIZER_SELECTION_TASKS,
    SEEDS,
    TASK_COUNT,
    UPDATE_BUDGETS,
    InternalOptimizerTask,
    task_for_index,
)
from data.generate_nbody_confirmatory import ConfirmatoryConfig  # noqa: E402
from experiments.f02_design import (  # noqa: E402
    OPTIMIZER_SELECTION_TIME_INDICES,
    TRAIN_TIME_INDICES,
)

AGGREGATE_SCHEMA_VERSION = "f02_internal_optimizer_aggregate_v1"
TASK_RESULT_SCHEMA_VERSION = "f02_internal_task_v1"
EXPECTED_CATALOG_SHA256 = "2dee429bdaf50cc78cb40ba8b038f7e4731bb07127927c55607b528c1db66942"
REFERENCE_M = 50
SAME_M_FLOAT32_TOLERANCE = 1e-4
TRAIN_ROWS = 60 * len(TRAIN_TIME_INDICES)
EVALUATION_ROWS = 20 * len(OPTIMIZER_SELECTION_TIME_INDICES)
FULL_TASK_INDICES = tuple(range(TASK_COUNT))
_EXCLUSIVE_MODES = {
    "slurm_explicit",
    "slurm_no_oversubscribe_full_node_sole_job",
}
_TASK_DIRECTORY_RE = re.compile(
    r"^task-(?P<index>\d+)-replica-(?P<replica>\d+)-n-(?P<particles>\d+)"
    r"-steps-(?P<steps>\d+)-seed-(?P<seed>\d+)$"
)
_TASK_INDEX_PREFIX_RE = re.compile(r"^task-(?P<index>\d+)(?:-|$)")
_HEX40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")

_RESULT_TOP_LEVEL_KEYS = {
    "schema_version",
    "status",
    "task_config",
    "training",
    "evaluation",
    "corpus",
    "frozen_parameters",
    "arms",
    "catalog",
    "provenance",
}
_REQUIRED_SOURCE_PATHS = {
    "cluster/check_python_environment.py",
    "cluster/f02_internal_grid.py",
    "cluster/f02_internal_optimizer.sbatch",
    "cluster/submit_f02_internal_optimizer.sh",
    "data/generate_nbody_confirmatory.py",
    "data/load_nbody_confirmatory.py",
    "experiments/f02_design.py",
    "experiments/f02_internal_models.py",
    "experiments/f02_internal_task.py",
    "experiments/f02_metrics.py",
    "gp/orbit/__init__.py",
    "gp/orbit/operator.py",
    "gp/orbit/predictor.py",
    "pyproject.toml",
    "uv.lock",
}
_REQUIRED_ARTIFACTS = {
    "artifacts.sha256",
    "command.txt",
    "dataset-files.sha256",
    "dependency-audit.json",
    "dependency-files.sha256",
    "dependency-packages.txt",
    "exit-code.txt",
    "finished-at.txt",
    "git-status.txt",
    "git-submodules.txt",
    "gpu-processes-after.csv",
    "gpu-processes-before.csv",
    "gpu-topology.txt",
    "gpu.csv",
    "provenance.env",
    "result.json",
    "runtime.json",
    "slurm-job.txt",
    "slurm-jobs-on-node.txt",
    "slurm-node.txt",
    "source-files.sha256",
}
_HASHED_ARTIFACTS = _REQUIRED_ARTIFACTS - {"artifacts.sha256"}


@dataclass(slots=True)
class _LoadedTask:
    record: dict[str, Any]
    result_sha256: str = ""
    tera_metrics: dict[str, Any] | None = None
    scalar_rmse: float | None = None
    commit: str = ""
    tree: str = ""
    submodules: str = ""
    tera_gitlink: str = ""
    source_manifest_sha256: str = ""
    dependency_manifest_sha256: str = ""
    dependency_audit_sha256: str = ""
    packages_sha256: str = ""
    array_job_id: str = ""
    repo_root: str = ""
    catalog_path: str = ""
    catalog_sha256: str = ""
    catalog_generation_commit: str = ""
    catalog_generation_tree: str = ""
    exclusive_mode: str = ""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _nonnegative_finite(value: Any) -> bool:
    return _finite(value) and float(value) >= 0.0


def _positive_finite(value: Any) -> bool:
    return _finite(value) and float(value) > 0.0


def _parse_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"malformed provenance line: {line!r}")
        key, value = line.split("=", 1)
        if not key or key in result:
            raise ValueError(f"duplicate or empty provenance key: {key!r}")
        result[key] = value
    return result


def _parse_sha256sum(path: Path, *, basenames: bool = False) -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or _HEX64_RE.fullmatch(fields[0]) is None:
            raise ValueError(f"malformed SHA-256 line: {line!r}")
        raw_path = fields[1].lstrip("*")
        key = Path(raw_path).name if basenames else Path(raw_path).as_posix()
        if not key or key in entries:
            raise ValueError(f"duplicate or empty SHA-256 path: {key!r}")
        entries[key] = (fields[0].lower(), raw_path)
    if not entries:
        raise ValueError("SHA-256 manifest is empty")
    return entries


def _expected_directory_name(task: InternalOptimizerTask) -> str:
    return (
        f"task-{task.task_index}-replica-{task.replica}-n-{task.n_particles}"
        f"-steps-{task.train_steps}-seed-{task.seed}"
    )


def _task_record(
    path: Path,
    task: InternalOptimizerTask,
    *,
    status: str,
    errors: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "task_index": task.task_index,
        "expected_task": task.as_record(),
        "expected_directory_name": _expected_directory_name(task),
        "path": str(path),
        "status": status,
        "exit_code": None,
        "eligible_for_declared_aggregate": False,
        "errors": list(errors),
    }


def _expected_task_config(task: InternalOptimizerTask) -> dict[str, Any]:
    return {
        "training_m": 20,
        "train_steps": task.train_steps,
        "train_epochs": 0,
        "kernel": "rbf",
        "outputscale": 1.0,
        "sigma_f": 0.001,
        "sigma_g": 0.001,
        "lengthscale": 1.0,
        "lengthscale_init": "median",
        "lengthscale_init_max_points": 2048,
        "use_ard": False,
        "seed": task.seed,
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
        "candidate_m": [],
        "cg_tolerance": 1e-5,
        "cg_max_iterations": None,
        "use_preconditioner": True,
        "function_jitter": 1e-8,
        "reduced_jitter": 1e-8,
        "dtype": "float32",
        "device": "cuda",
    }


def _expected_generator_config(task: InternalOptimizerTask) -> dict[str, Any]:
    return asdict(
        ConfirmatoryConfig(
            n_particles=task.n_particles,
            n_dims=task.n_dims,
            replica=task.replica,
        )
    )


def _validate_exact_keys(
    value: Any,
    expected: set[str],
    label: str,
    errors: list[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return {}
    if set(value) != expected:
        errors.append(
            f"{label} has the wrong keys: expected {sorted(expected)}, got {sorted(value)}"
        )
    return value


def _validate_raw_prediction_checks(
    value: Any,
    label: str,
    errors: list[str],
) -> None:
    checks = _validate_exact_keys(
        value,
        {
            "row_count",
            "mean_all_finite",
            "latent",
            "observation",
            "value_noise_variance",
            "released_variance_epsilon_floor",
            "observation_is_latent_plus_noise",
            "all_valid",
        },
        f"{label}.raw_prediction_checks",
        errors,
    )
    if checks.get("row_count") != EVALUATION_ROWS:
        errors.append(f"{label} prediction row_count must equal {EVALUATION_ROWS}")
    for flag in (
        "mean_all_finite",
        "observation_is_latent_plus_noise",
        "all_valid",
    ):
        if checks.get(flag) is not True:
            errors.append(f"{label} {flag} gate did not pass")
    for variance_name in ("latent", "observation"):
        variance = checks.get(variance_name)
        if not isinstance(variance, dict):
            errors.append(f"{label} {variance_name} variance checks are malformed")
            continue
        if set(variance) != {"all_finite", "all_positive", "minimum_raw", "maximum_raw"}:
            errors.append(f"{label} {variance_name} variance checks have the wrong schema")
        if variance.get("all_finite") is not True or variance.get("all_positive") is not True:
            errors.append(f"{label} {variance_name} variance gate did not pass")
        if not _positive_finite(variance.get("minimum_raw")):
            errors.append(f"{label} {variance_name} minimum_raw must be positive and finite")
        if not _positive_finite(variance.get("maximum_raw")):
            errors.append(f"{label} {variance_name} maximum_raw must be positive and finite")
        if _finite(variance.get("minimum_raw")) and _finite(variance.get("maximum_raw")):
            if variance["minimum_raw"] > variance["maximum_raw"]:
                errors.append(f"{label} {variance_name} variance extrema are reversed")
    if not _nonnegative_finite(checks.get("value_noise_variance")):
        errors.append(f"{label} value_noise_variance must be finite and nonnegative")


def _validate_trajectory_metrics(
    raw: Any,
    *,
    label: str,
    task: InternalOptimizerTask,
    evaluation_ids: Sequence[int],
    errors: list[str],
) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or len(raw) != EVALUATION_ROWS:
        errors.append(f"{label} must contain exactly {EVALUATION_ROWS} trajectory records")
        return []
    expected_keys = {
        "replica",
        "dimension",
        "trajectory_id",
        "n_points",
        "standardized_mse",
        "standardized_rmse",
        "gaussian_nll",
        "interval_coverage",
    }
    seen: list[int] = []
    valid: list[dict[str, Any]] = []
    for index, record in enumerate(raw):
        item = _validate_exact_keys(record, expected_keys, f"{label}[{index}]", errors)
        if not item:
            continue
        trajectory_id = item.get("trajectory_id")
        if not _is_int(trajectory_id) or trajectory_id < 0:
            errors.append(f"{label}[{index}] trajectory_id must be a nonnegative integer")
        else:
            seen.append(trajectory_id)
        if item.get("replica") != task.replica:
            errors.append(f"{label}[{index}] replica does not match the task map")
        if item.get("dimension") != 2 * task.n_particles * task.n_dims:
            errors.append(f"{label}[{index}] dimension does not match the task map")
        if item.get("n_points") != len(OPTIMIZER_SELECTION_TIME_INDICES):
            errors.append(f"{label}[{index}] n_points must equal one")
        mse = item.get("standardized_mse")
        rmse = item.get("standardized_rmse")
        if not _nonnegative_finite(mse):
            errors.append(f"{label}[{index}] standardized_mse is invalid")
        if not _nonnegative_finite(rmse):
            errors.append(f"{label}[{index}] standardized_rmse is invalid")
        if _nonnegative_finite(mse) and _nonnegative_finite(rmse):
            if float(rmse) != math.sqrt(float(mse)):
                errors.append(f"{label}[{index}] RMSE is not sqrt(MSE)")
        if not _finite(item.get("gaussian_nll")):
            errors.append(f"{label}[{index}] gaussian_nll is nonfinite")
        coverage = item.get("interval_coverage")
        if not isinstance(coverage, list) or len(coverage) != 3:
            errors.append(f"{label}[{index}] interval_coverage has the wrong schema")
        else:
            levels: list[float] = []
            for entry in coverage:
                if not isinstance(entry, dict) or set(entry) != {"level", "coverage"}:
                    errors.append(f"{label}[{index}] interval coverage entry is malformed")
                    continue
                if not _finite(entry.get("level")) or not _finite(entry.get("coverage")):
                    errors.append(f"{label}[{index}] interval coverage entry is nonfinite")
                    continue
                levels.append(float(entry["level"]))
                if not 0.0 <= float(entry["coverage"]) <= 1.0:
                    errors.append(f"{label}[{index}] interval coverage lies outside [0, 1]")
            if levels != [0.5, 0.9, 0.95]:
                errors.append(f"{label}[{index}] interval levels are not the frozen levels")
        valid.append(item)
    if len(set(seen)) != len(seen):
        errors.append(f"{label} contains duplicate trajectory ids")
    if sorted(seen) != sorted(evaluation_ids):
        errors.append(f"{label} trajectory ids do not match the evaluation source rows")
    return valid


def _validate_prediction_moments(
    raw: Any,
    *,
    label: str,
    value_noise_variance: float | None,
    errors: list[str],
) -> dict[str, list[Any]]:
    moments = _validate_exact_keys(
        raw,
        {"mean", "latent_variance", "observation_variance"},
        f"{label}.prediction_moments",
        errors,
    )
    vectors: dict[str, list[Any]] = {}
    for name in ("mean", "latent_variance", "observation_variance"):
        vector = moments.get(name)
        if not isinstance(vector, list) or len(vector) != EVALUATION_ROWS:
            errors.append(f"{label} {name} must contain exactly {EVALUATION_ROWS} values")
            continue
        if not all(_finite(item) for item in vector):
            errors.append(f"{label} {name} contains a nonfinite value")
            continue
        if name != "mean" and not all(float(item) > 0.0 for item in vector):
            errors.append(f"{label} {name} contains a nonpositive raw variance")
        vectors[name] = vector
    if value_noise_variance is not None and all(
        name in vectors for name in ("latent_variance", "observation_variance")
    ):
        for latent, observation in zip(
            vectors["latent_variance"], vectors["observation_variance"], strict=True
        ):
            if not math.isclose(
                float(observation),
                float(latent) + value_noise_variance,
                rel_tol=2e-6,
                abs_tol=2e-7,
            ):
                errors.append(f"{label} observation variance is not latent plus scalar noise")
                break
    return vectors


def _validate_orbit_solver(raw: Any, errors: list[str]) -> None:
    expected_keys = {
        "fresh_relative_residuals",
        "maximum_fresh_relative_residual",
        "converged",
        "all_converged",
        "iterations",
        "operator_matvecs",
        "preconditioner_applications",
        "ranks",
        "basis_exact",
        "exact_arithmetic_certified",
        "floating_point_rigorous",
        "variance_error_upper_bounds",
        "expected_kl_upper_bounds",
        "finite_precision_variance_corrections",
    }
    solver = _validate_exact_keys(raw, expected_keys, "ORBIT-50.solver", errors)
    vector_names = expected_keys - {"maximum_fresh_relative_residual", "all_converged"}
    vectors: dict[str, list[Any]] = {}
    for name in vector_names:
        value = solver.get(name)
        if not isinstance(value, list) or len(value) != EVALUATION_ROWS:
            errors.append(f"ORBIT-50 solver {name} must contain {EVALUATION_ROWS} values")
        else:
            vectors[name] = value
    residuals = vectors.get("fresh_relative_residuals", [])
    if residuals and not all(_nonnegative_finite(value) and value <= 1e-5 for value in residuals):
        errors.append("ORBIT-50 freshly recomputed residual gate did not pass")
    if residuals and solver.get("maximum_fresh_relative_residual") != max(residuals):
        errors.append("ORBIT-50 maximum residual does not match its residual vector")
    if solver.get("all_converged") is not True:
        errors.append("ORBIT-50 all_converged gate did not pass")
    for name in ("converged", "basis_exact"):
        if name in vectors and not all(value is True for value in vectors[name]):
            errors.append(f"ORBIT-50 {name} gate did not pass")
    for name in ("exact_arithmetic_certified", "floating_point_rigorous"):
        if name in vectors and not all(isinstance(value, bool) for value in vectors[name]):
            errors.append(f"ORBIT-50 {name} must contain booleans")
    for name in ("iterations", "operator_matvecs", "preconditioner_applications"):
        if name in vectors and not all(_is_int(value) and value >= 0 for value in vectors[name]):
            errors.append(f"ORBIT-50 {name} must contain nonnegative integers")
    if "ranks" in vectors and not all(
        _is_int(value) and 0 < value <= REFERENCE_M for value in vectors["ranks"]
    ):
        errors.append("ORBIT-50 ranks must be integers in [1, 50]")
    for name in ("variance_error_upper_bounds", "expected_kl_upper_bounds"):
        if name in vectors and not all(
            value is None or _nonnegative_finite(value) for value in vectors[name]
        ):
            errors.append(f"ORBIT-50 {name} contains an invalid bound")
    corrections = vectors.get("finite_precision_variance_corrections", [])
    if corrections and not all(_finite(value) for value in corrections):
        errors.append("ORBIT-50 finite-precision variance corrections are nonfinite")


def _orbit_matmul_flops(m: int, rank: int) -> int:
    return 10 * m * m * rank + 2 * m * rank * rank + 2 * m * m


def _preconditioner_flops(m: int, rank: int) -> int:
    return 4 * m * m * rank + 4 * m * rank * rank


def _validate_orbit_resources(raw: Any, solver: Any, errors: list[str]) -> None:
    resources = _validate_exact_keys(
        raw,
        {
            "counting_schema",
            "requested_m",
            "effective_m",
            "preconditioner_counted",
            "per_target",
            "operator_core_elements_max",
            "counted_flops_total",
            "counted_flops_mean_per_target",
        },
        "ORBIT-50.analytic_resources",
        errors,
    )
    if (
        resources.get("counting_schema") != "orbit_structured_proxy_v1"
        or resources.get("requested_m") != REFERENCE_M
        or resources.get("effective_m") != REFERENCE_M
        or resources.get("preconditioner_counted") is not True
    ):
        errors.append("ORBIT-50 analytic resource header does not match the frozen recipe")
    rows = resources.get("per_target")
    if not isinstance(rows, list) or len(rows) != EVALUATION_ROWS:
        errors.append(f"ORBIT-50 resources must contain {EVALUATION_ROWS} target records")
        return
    solver = solver if isinstance(solver, dict) else {}
    expected_row_keys = {
        "rank",
        "iterations",
        "operator_matvecs",
        "preconditioner_applications",
        "reduced_system_dimension",
        "operator_core_elements",
        "structured_operator_flops",
        "preconditioner_flops",
        "counted_flops",
    }
    counted: list[int] = []
    core: list[int] = []
    for index, row in enumerate(rows):
        item = _validate_exact_keys(
            row, expected_row_keys, f"ORBIT-50.analytic_resources.per_target[{index}]", errors
        )
        if not item or not all(_is_int(item.get(name)) for name in expected_row_keys):
            errors.append(f"ORBIT-50 resource row {index} must contain integers")
            continue
        if any(item[name] < 0 for name in expected_row_keys - {"rank"}) or item["rank"] <= 0:
            errors.append(f"ORBIT-50 resource row {index} contains an invalid count")
            continue
        rank = item["rank"]
        expected_core = 3 * REFERENCE_M**2 + 2 * REFERENCE_M * rank + rank**2 + rank
        expected_operator = item["operator_matvecs"] * _orbit_matmul_flops(REFERENCE_M, rank)
        expected_preconditioner = item["preconditioner_applications"] * _preconditioner_flops(
            REFERENCE_M, rank
        )
        expected_values = {
            "reduced_system_dimension": REFERENCE_M * rank,
            "operator_core_elements": expected_core,
            "structured_operator_flops": expected_operator,
            "preconditioner_flops": expected_preconditioner,
            "counted_flops": expected_operator + expected_preconditioner,
        }
        if any(item[name] != value for name, value in expected_values.items()):
            errors.append(f"ORBIT-50 resource row {index} does not satisfy the counting schema")
        for name in ("ranks", "iterations", "operator_matvecs", "preconditioner_applications"):
            solver_values = solver.get(name)
            row_name = "rank" if name == "ranks" else name
            if isinstance(solver_values, list) and len(solver_values) == EVALUATION_ROWS:
                if item[row_name] != solver_values[index]:
                    errors.append(f"ORBIT-50 resource row {index} disagrees with solver {name}")
        counted.append(item["counted_flops"])
        core.append(item["operator_core_elements"])
    if len(counted) == EVALUATION_ROWS:
        if resources.get("counted_flops_total") != sum(counted):
            errors.append("ORBIT-50 counted_flops_total is inconsistent")
        if resources.get("counted_flops_mean_per_target") != sum(counted) / len(counted):
            errors.append("ORBIT-50 counted_flops_mean_per_target is inconsistent")
        if resources.get("operator_core_elements_max") != max(core):
            errors.append("ORBIT-50 operator_core_elements_max is inconsistent")


def _validate_arm(
    arm: Any,
    *,
    label: str,
    family: str,
    task: InternalOptimizerTask,
    evaluation_ids: Sequence[int],
    value_noise_variance: float | None,
    errors: list[str],
) -> dict[str, Any] | None:
    if not isinstance(arm, dict):
        errors.append(f"{label} arm must be an object")
        return None
    required = {
        "label",
        "family",
        "requested_m",
        "effective_m",
        "hyperparameters_source",
        "raw_prediction_checks",
        "metrics",
        "prediction_moments",
        "prediction_seconds_descriptive",
        "prediction_peak_gpu_allocated_bytes",
    }
    if label in {"TERA-50", "ORBIT-50"}:
        required.add("analytic_resources")
    if label == "ORBIT-50":
        required.update({"solver", "same_m_agreement_to_TERA_50"})
    if label == "value-only-conditional-50":
        required.add("control_semantics")
    if set(arm) != required:
        errors.append(f"{label} arm has the wrong top-level schema")
    expected_header = {
        "label": label,
        "family": family,
        "requested_m": REFERENCE_M,
        "effective_m": REFERENCE_M,
        "hyperparameters_source": "TERA-gradient-fit",
    }
    if any(arm.get(name) != value for name, value in expected_header.items()):
        errors.append(f"{label} arm header does not match the frozen recipe")
    _validate_raw_prediction_checks(arm.get("raw_prediction_checks"), label, errors)
    moment_vectors = _validate_prediction_moments(
        arm.get("prediction_moments"),
        label=label,
        value_noise_variance=value_noise_variance,
        errors=errors,
    )
    checks = arm.get("raw_prediction_checks")
    if isinstance(checks, dict):
        for check_name, vector_name in (
            ("latent", "latent_variance"),
            ("observation", "observation_variance"),
        ):
            variance_check = checks.get(check_name)
            vector = moment_vectors.get(vector_name)
            if isinstance(variance_check, dict) and vector:
                if variance_check.get("minimum_raw") != min(vector):
                    errors.append(f"{label} {check_name} recorded minimum does not match moments")
                if variance_check.get("maximum_raw") != max(vector):
                    errors.append(f"{label} {check_name} recorded maximum does not match moments")
    if not _nonnegative_finite(arm.get("prediction_seconds_descriptive")):
        errors.append(f"{label} descriptive prediction time is invalid")
    peak = arm.get("prediction_peak_gpu_allocated_bytes")
    if not _is_int(peak) or peak < 0:
        errors.append(f"{label} GPU peak allocation must be a nonnegative integer")
    metrics = _validate_exact_keys(
        arm.get("metrics"), {"latent", "observation"}, f"{label}.metrics", errors
    )
    latent = _validate_trajectory_metrics(
        metrics.get("latent"),
        label=f"{label}.metrics.latent",
        task=task,
        evaluation_ids=evaluation_ids,
        errors=errors,
    )
    observation = _validate_trajectory_metrics(
        metrics.get("observation"),
        label=f"{label}.metrics.observation",
        task=task,
        evaluation_ids=evaluation_ids,
        errors=errors,
    )
    if len(latent) == len(observation) == EVALUATION_ROWS:
        for left, right in zip(latent, observation, strict=True):
            for name in (
                "replica",
                "dimension",
                "trajectory_id",
                "n_points",
                "standardized_mse",
                "standardized_rmse",
            ):
                if left[name] != right[name]:
                    errors.append(f"{label} latent/observation metrics disagree on {name}")
                    break
    if label == "TERA-50":
        resources = arm.get("analytic_resources")
        expected_resources = {
            "counting_schema": "tera_dense_local_v1",
            "requested_m": 50,
            "effective_m": 50,
            "explicit_reduced_covariance_elements_per_target": 50**4,
            "reduced_cholesky_leading_flops_per_target": (50**6) / 3.0,
        }
        if resources != expected_resources:
            errors.append("TERA-50 analytic resources do not match the registered reference")
        floor = arm.get("raw_prediction_checks", {}).get("released_variance_epsilon_floor")
        if (
            not isinstance(floor, dict)
            or floor.get("inactive") is not True
            or not _positive_finite(floor.get("value"))
            or floor.get("failure_policy") != "equality-to-floor-fails-before-scoring"
        ):
            errors.append("TERA-50 released variance floor gate is missing or active")
    elif label == "ORBIT-50":
        _validate_orbit_solver(arm.get("solver"), errors)
        _validate_orbit_resources(arm.get("analytic_resources"), arm.get("solver"), errors)
        agreement = _validate_exact_keys(
            arm.get("same_m_agreement_to_TERA_50"),
            {"maxabs_mean", "maxabs_latent_variance", "absolute_tolerance", "passes"},
            "ORBIT-50.same_m_agreement_to_TERA_50",
            errors,
        )
        if agreement.get("absolute_tolerance") != SAME_M_FLOAT32_TOLERANCE:
            errors.append("ORBIT-50 same-m tolerance is not the frozen float32 tolerance")
        for name in ("maxabs_mean", "maxabs_latent_variance"):
            if not _nonnegative_finite(agreement.get(name)):
                errors.append(f"ORBIT-50 same-m {name} is invalid")
            elif agreement[name] > SAME_M_FLOAT32_TOLERANCE:
                errors.append(f"ORBIT-50 same-m {name} exceeds tolerance")
        if agreement.get("passes") is not True:
            errors.append("ORBIT-50 same-m agreement gate did not pass")
        if arm.get("raw_prediction_checks", {}).get("released_variance_epsilon_floor") is not None:
            errors.append("ORBIT-50 unexpectedly reports a released TERA variance floor")
    else:
        if arm.get("raw_prediction_checks", {}).get("released_variance_epsilon_floor") is not None:
            errors.append("value-only conditional unexpectedly reports a released variance floor")
        semantics = arm.get("control_semantics")
        if not isinstance(semantics, str) or "not a standalone value-only" not in semantics:
            errors.append("value-only conditional control semantics are missing")
    return metrics if latent and observation else None


def _validate_result(
    result: Any,
    task: InternalOptimizerTask,
    errors: list[str],
) -> tuple[dict[str, Any] | None, float | None]:
    document = _validate_exact_keys(result, _RESULT_TOP_LEVEL_KEYS, "result.json", errors)
    if document.get("schema_version") != TASK_RESULT_SCHEMA_VERSION:
        errors.append("result.json has the wrong schema_version")
    if document.get("status") != "complete":
        errors.append("result.json status is not complete")

    expected_config = _expected_task_config(task)
    if document.get("task_config") != expected_config:
        errors.append("result task_config does not exactly match the frozen grid task")

    training = _validate_exact_keys(
        document.get("training"),
        {
            "split",
            "time_indices",
            "rows",
            "training_m",
            "train_steps",
            "train_epochs",
            "batch_size",
            "effective_batch_size",
            "optimizer_updates",
            "vecchia_target_factors_processed",
            "fit_seconds_descriptive",
            "fit_peak_gpu_allocated_bytes",
        },
        "result.training",
        errors,
    )
    expected_training = {
        "split": "train",
        "time_indices": list(TRAIN_TIME_INDICES),
        "rows": TRAIN_ROWS,
        "training_m": 20,
        "train_steps": task.train_steps,
        "train_epochs": 0,
        "batch_size": 256,
        "effective_batch_size": 256,
        "optimizer_updates": task.train_steps,
        "vecchia_target_factors_processed": task.train_steps * 256,
    }
    if any(training.get(name) != value for name, value in expected_training.items()):
        errors.append("result training design/update accounting does not match the frozen recipe")
    if not _nonnegative_finite(training.get("fit_seconds_descriptive")):
        errors.append("training fit_seconds_descriptive is invalid")
    peak = training.get("fit_peak_gpu_allocated_bytes")
    if not _is_int(peak) or peak < 0:
        errors.append("training GPU peak allocation must be a nonnegative integer")

    evaluation = _validate_exact_keys(
        document.get("evaluation"),
        {"split", "design", "time_indices", "test_gate"},
        "result.evaluation",
        errors,
    )
    if (
        evaluation.get("split") != "validation"
        or evaluation.get("design") != "optimizer_selection"
        or evaluation.get("time_indices") != [50]
    ):
        errors.append("result evaluation is not the registered validation/time-50 design")
    expected_gate = {
        "required": False,
        "validated": False,
        "committed_at_head": False,
        "path": None,
        "payload_sha256": None,
        "schema_version": None,
    }
    if evaluation.get("test_gate") != expected_gate:
        errors.append("optimizer-selection task unexpectedly crossed a test recipe gate")

    corpus = _validate_exact_keys(
        document.get("corpus"),
        {
            "replica",
            "dimension",
            "train_rows",
            "evaluation_rows",
            "train_source_indices",
            "evaluation_source_indices",
            "evaluation_trajectory_ids",
        },
        "result.corpus",
        errors,
    )
    dimension = 2 * task.n_particles * task.n_dims
    if (
        corpus.get("replica") != task.replica
        or corpus.get("dimension") != dimension
        or corpus.get("train_rows") != TRAIN_ROWS
        or corpus.get("evaluation_rows") != EVALUATION_ROWS
    ):
        errors.append("result corpus identity/row counts do not match the task map")
    train_sources = corpus.get("train_source_indices")
    eval_sources = corpus.get("evaluation_source_indices")
    eval_ids = corpus.get("evaluation_trajectory_ids")
    if not isinstance(train_sources, list) or len(train_sources) != TRAIN_ROWS:
        errors.append(f"train_source_indices must contain exactly {TRAIN_ROWS} rows")
        train_sources = []
    if not isinstance(eval_sources, list) or len(eval_sources) != EVALUATION_ROWS:
        errors.append(f"evaluation_source_indices must contain exactly {EVALUATION_ROWS} rows")
        eval_sources = []
    if not isinstance(eval_ids, list) or len(eval_ids) != EVALUATION_ROWS:
        errors.append(f"evaluation_trajectory_ids must contain exactly {EVALUATION_ROWS} ids")
        eval_ids = []
    for label, sources in (("train", train_sources), ("evaluation", eval_sources)):
        if not all(_is_int(value) and 0 <= value < 10_000 for value in sources):
            errors.append(f"{label} source indices must be integers in [0, 10000)")
        if len(set(sources)) != len(sources):
            errors.append(f"{label} source indices are not unique")
    if train_sources:
        train_by_trajectory: dict[int, list[int]] = {}
        for source in train_sources:
            train_by_trajectory.setdefault(source // 100, []).append(source % 100)
        if len(train_by_trajectory) != 60 or any(
            sorted(times) != list(TRAIN_TIME_INDICES) for times in train_by_trajectory.values()
        ):
            errors.append("training source ids do not encode 60 complete frozen time slices")
    if eval_sources:
        derived_eval_ids = [source // 100 for source in eval_sources]
        if any(source % 100 != 50 for source in eval_sources):
            errors.append("evaluation source ids include a time index other than 50")
        if sorted(derived_eval_ids) != sorted(eval_ids):
            errors.append("evaluation source ids and trajectory ids disagree")
    if not all(_is_int(value) and value >= 0 for value in eval_ids) or len(set(eval_ids)) != len(
        eval_ids
    ):
        errors.append("evaluation trajectory ids must be unique nonnegative integers")
    if set(train_sources) & set(eval_sources):
        errors.append("training and evaluation source ids overlap")

    frozen = document.get("frozen_parameters")
    if not isinstance(frozen, dict) or set(frozen) != {
        "kernel",
        "lengthscale",
        "outputscale",
        "sigma_f_variance",
        "sigma_g_variance",
        "gradient_noise_model",
    }:
        errors.append("frozen_parameters has the wrong schema")
        frozen = {}
    if frozen.get("kernel") != "rbf" or frozen.get("gradient_noise_model") != "iid":
        errors.append("frozen kernel/noise model does not match the registered recipe")
    lengthscale = frozen.get("lengthscale")
    if (
        not isinstance(lengthscale, list)
        or not lengthscale
        or not all(_positive_finite(value) for value in lengthscale)
    ):
        errors.append("frozen lengthscale is invalid")
    for name in ("outputscale", "sigma_f_variance"):
        if not _positive_finite(frozen.get(name)):
            errors.append(f"frozen {name} must be positive and finite")
    if not _nonnegative_finite(frozen.get("sigma_g_variance")):
        errors.append("frozen sigma_g_variance must be nonnegative and finite")
    sigma_f = (
        float(frozen["sigma_f_variance"])
        if _positive_finite(frozen.get("sigma_f_variance"))
        else None
    )

    arms = document.get("arms")
    expected_arms = {"TERA-50", "ORBIT-50", "value-only-conditional-50"}
    if not isinstance(arms, dict) or set(arms) != expected_arms:
        errors.append("optimizer-selection result has the wrong arm set; candidate_m must be empty")
        arms = {}
    tera_metrics = _validate_arm(
        arms.get("TERA-50"),
        label="TERA-50",
        family="TERA",
        task=task,
        evaluation_ids=eval_ids,
        value_noise_variance=sigma_f,
        errors=errors,
    )
    _validate_arm(
        arms.get("ORBIT-50"),
        label="ORBIT-50",
        family="ORBIT",
        task=task,
        evaluation_ids=eval_ids,
        value_noise_variance=sigma_f,
        errors=errors,
    )
    _validate_arm(
        arms.get("value-only-conditional-50"),
        label="value-only-conditional-50",
        family="value-only-conditioning-ablation",
        task=task,
        evaluation_ids=eval_ids,
        value_noise_variance=sigma_f,
        errors=errors,
    )
    tera_moments = arms.get("TERA-50", {}).get("prediction_moments", {})
    orbit_moments = arms.get("ORBIT-50", {}).get("prediction_moments", {})
    agreement = arms.get("ORBIT-50", {}).get("same_m_agreement_to_TERA_50", {})
    if all(isinstance(value, dict) for value in (tera_moments, orbit_moments, agreement)):
        for vector_name, agreement_name in (
            ("mean", "maxabs_mean"),
            ("latent_variance", "maxabs_latent_variance"),
        ):
            tera_vector = tera_moments.get(vector_name)
            orbit_vector = orbit_moments.get(vector_name)
            if (
                isinstance(tera_vector, list)
                and isinstance(orbit_vector, list)
                and len(tera_vector) == len(orbit_vector) == EVALUATION_ROWS
                and all(_finite(value) for value in (*tera_vector, *orbit_vector))
            ):
                recomputed = max(
                    abs(float(left) - float(right))
                    for left, right in zip(tera_vector, orbit_vector, strict=True)
                )
                if recomputed > SAME_M_FLOAT32_TOLERANCE:
                    errors.append(f"ORBIT-50 stored {vector_name} fails the same-m tolerance")
                recorded = agreement.get(agreement_name)
                if not _finite(recorded) or not math.isclose(
                    float(recorded), recomputed, rel_tol=2e-5, abs_tol=2e-7
                ):
                    errors.append(
                        f"ORBIT-50 {agreement_name} does not match stored prediction moments"
                    )
    scalar_rmse: float | None = None
    if tera_metrics is not None:
        latent = tera_metrics.get("latent")
        if (
            isinstance(latent, list)
            and len(latent) == EVALUATION_ROWS
            and all(
                isinstance(record, dict) and _nonnegative_finite(record.get("standardized_mse"))
                for record in latent
            )
        ):
            # Protocol order: mean trajectory MSE within this seed/corpus, then sqrt.
            scalar_rmse = math.sqrt(
                sum(float(record["standardized_mse"]) for record in latent) / len(latent)
            )
            if not math.isfinite(scalar_rmse):
                errors.append("TERA-50 scalar RMSE aggregation is nonfinite")
                scalar_rmse = None
    return tera_metrics, scalar_rmse


def _validate_manifest_files(
    entries: dict[str, tuple[str, str]],
    *,
    root: Path | None,
    label: str,
    errors: list[str],
) -> None:
    for key, (digest, raw_path) in entries.items():
        path = Path(raw_path)
        if root is not None:
            if path.is_absolute() or ".." in path.parts:
                errors.append(f"{label} contains a non-relative source path: {raw_path}")
                continue
            path = (root / path).resolve()
            try:
                path.relative_to(root.resolve())
            except ValueError:
                errors.append(f"{label} source path escapes repo_root: {raw_path}")
                continue
        if not path.is_file():
            errors.append(f"{label} recorded file is unavailable: {raw_path}")
        elif _sha256_file(path) != digest:
            errors.append(f"{label} does not verify recorded file: {key}")


def _parse_tres(record: str, field: str) -> dict[str, str]:
    match = re.search(rf"(?:^|\s){re.escape(field)}=([^\s]+)", record)
    if match is None:
        return {}
    result: dict[str, str] = {}
    for item in match.group(1).split(","):
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = value
    return result


def _validate_exclusivity(
    mode: str,
    job_record: str,
    node_record: str,
    node_jobs: str,
    array_job_id: str,
    errors: list[str],
) -> None:
    jobs = [line.strip() for line in node_jobs.splitlines() if line.strip()]
    if jobs != [array_job_id]:
        errors.append("slurm-jobs-on-node.txt does not show the array job as the sole job family")
    if mode == "slurm_explicit":
        if "OverSubscribe=EXCLUSIVE" not in job_record:
            errors.append("explicit exclusivity mode is not supported by slurm-job.txt")
    elif mode == "slurm_no_oversubscribe_full_node_sole_job":
        if "OverSubscribe=NO" not in job_record:
            errors.append("fallback exclusivity mode is not supported by slurm-job.txt")
        allocated = _parse_tres(job_record, "AllocTRES")
        configured = _parse_tres(node_record, "CfgTRES")
        for name in ("cpu", "gres/gpu"):
            if not allocated.get(name) or allocated.get(name) != configured.get(name):
                errors.append(f"fallback exclusivity does not allocate the node's full {name} TRES")
        if allocated.get("gres/gpu") == "0":
            errors.append("fallback exclusivity reports zero allocated GPUs")
    else:
        errors.append("exclusive_verification_mode is missing or unsupported")


def _parse_command(path: Path, errors: list[str]) -> tuple[list[str], dict[str, str | bool]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1 or not lines[0].startswith("command="):
        errors.append("command.txt must contain exactly one command= line")
        return [], {}
    try:
        tokens = shlex.split(lines[0].removeprefix("command="))
    except ValueError as error:
        errors.append(f"command.txt is not shell-parseable: {error}")
        return [], {}
    try:
        script_index = next(
            index
            for index, token in enumerate(tokens)
            if token.endswith("experiments/f02_internal_task.py")
        )
    except StopIteration:
        errors.append("command.txt does not invoke experiments/f02_internal_task.py")
        return tokens, {}
    tail = tokens[script_index + 1 :]
    if not tail or tail[0].startswith("--"):
        errors.append("command.txt is missing its dataset positional argument")
        return tokens, {}
    options: dict[str, str | bool] = {}
    index = 1
    while index < len(tail):
        name = tail[index]
        if not name.startswith("--") or name in options:
            errors.append(f"command.txt has a malformed or duplicate option: {name!r}")
            return tokens, options
        if name == "--use-preconditioner":
            options[name] = True
            index += 1
        elif index + 1 >= len(tail) or tail[index + 1].startswith("--"):
            errors.append(f"command.txt option has no value: {name}")
            return tokens, options
        else:
            options[name] = tail[index + 1]
            index += 2
    return [tail[0], *tokens[: script_index + 1]], options


def _validate_command(
    path: Path,
    task: InternalOptimizerTask,
    task_dir: Path,
    catalog_path: str,
    errors: list[str],
) -> None:
    prefix, options = _parse_command(path, errors)
    if not prefix:
        return
    dataset_path = Path(prefix[0])
    if dataset_path.name != f"{task.dataset_stem}.npz":
        errors.append("command dataset does not match the task-map corpus")
    expected = {
        "--catalog": catalog_path,
        "--out": str(task_dir / "result.json"),
        "--evaluation-split": "validation",
        "--evaluation-design": "optimizer_selection",
        "--training-m": "20",
        "--train-steps": str(task.train_steps),
        "--train-epochs": "0",
        "--kernel": "rbf",
        "--outputscale": "1.0",
        "--sigma-f": "0.001",
        "--sigma-g": "0.001",
        "--lengthscale": "1.0",
        "--seed": str(task.seed),
        "--batch-size": "256",
        "--lr": "0.01",
        "--weight-decay": "0.0",
        "--candidate-m": "none",
        "--cg-tolerance": "1e-5",
        "--use-preconditioner": True,
        "--function-jitter": "1e-8",
        "--reduced-jitter": "1e-8",
        "--dtype": "float32",
        "--device": "cuda",
    }
    if options != expected:
        errors.append("command.txt options do not exactly match the frozen Slurm recipe")


def _validate_dependency_audit(
    audit: Any,
    package_lines: list[str],
    errors: list[str],
) -> None:
    expected_keys = {
        "schema_version",
        "status",
        "python_executable",
        "package_count",
        "packages",
        "issues",
    }
    report = _validate_exact_keys(audit, expected_keys, "dependency-audit.json", errors)
    packages = report.get("packages")
    if (
        report.get("schema_version") != 1
        or report.get("status") != "pass"
        or report.get("issues") != []
        or not isinstance(packages, list)
        or report.get("package_count") != len(packages)
    ):
        errors.append("dependency audit did not record a clean version-1 environment")
        return
    expected_lines: list[str] = []
    for item in packages:
        if not isinstance(item, dict) or set(item) != {"name", "version"}:
            errors.append("dependency audit package record is malformed")
            continue
        name = item.get("name")
        version = item.get("version")
        if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
            errors.append("dependency audit package name/version is malformed")
            continue
        expected_lines.append(f"{name}=={version}")
    if package_lines != expected_lines or package_lines != sorted(package_lines):
        errors.append("dependency-packages.txt does not exactly match the sorted dependency audit")
    if len(set(package_lines)) != len(package_lines):
        errors.append("dependency-packages.txt contains duplicate distributions")


def _validate_runtime(runtime: Any, provenance: dict[str, str], errors: list[str]) -> None:
    expected_keys = {
        "cuda_visible_devices",
        "numpy",
        "platform",
        "python_executable",
        "python_version",
        "scipy",
        "torch",
        "torch_cuda_available",
        "torch_cuda_devices",
        "torch_cuda_runtime",
    }
    report = _validate_exact_keys(runtime, expected_keys, "runtime.json", errors)
    if report.get("torch_cuda_available") is not True:
        errors.append("runtime.json does not report CUDA availability")
    if report.get("cuda_visible_devices") != provenance.get("cuda_visible_devices"):
        errors.append("runtime CUDA_VISIBLE_DEVICES disagrees with provenance.env")
    devices = report.get("torch_cuda_devices")
    if not isinstance(devices, list) or len(devices) != 1:
        errors.append("runtime.json must expose exactly one CUDA device")
    else:
        device = devices[0]
        if not isinstance(device, dict) or set(device) != {
            "capability",
            "index",
            "name",
            "total_memory_bytes",
        }:
            errors.append("runtime CUDA device record has the wrong schema")
        elif (
            device.get("index") != 0
            or "L40S" not in str(device.get("name", "")).upper()
            or not _is_int(device.get("total_memory_bytes"))
            or device["total_memory_bytes"] <= 0
        ):
            errors.append("runtime CUDA device is not the required single L40S")
    for name in ("numpy", "scipy", "torch", "python_version", "platform", "torch_cuda_runtime"):
        if not isinstance(report.get(name), str) or not report[name]:
            errors.append(f"runtime {name} is missing")
    if report.get("python_executable") != provenance.get("python"):
        try:
            same_python = (
                Path(str(report.get("python_executable"))).resolve()
                == Path(provenance.get("python", "")).resolve()
            )
        except OSError:
            same_python = False
        if not same_python:
            errors.append("runtime and provenance Python executables disagree")


def _validate_result_provenance(
    result: dict[str, Any],
    task: InternalOptimizerTask,
    env: dict[str, str],
    source_entries: dict[str, tuple[str, str]],
    dependency_entries: dict[str, tuple[str, str]],
    dataset_entries: dict[str, tuple[str, str]],
    submodule_lines: list[str],
    runtime: dict[str, Any],
    errors: list[str],
) -> tuple[str, str]:
    provenance = _validate_exact_keys(
        result.get("provenance"),
        {"git", "data", "task_config", "dependencies", "submodules", "runtime"},
        "result.provenance",
        errors,
    )
    git = provenance.get("git")
    if not isinstance(git, dict) or set(git) != {"commit", "tree", "describe", "status_porcelain"}:
        errors.append("result provenance git record has the wrong schema")
        git = {}
    if (
        git.get("commit") != env.get("git_commit")
        or git.get("tree") != env.get("git_tree")
        or git.get("describe") != env.get("git_describe")
        or git.get("status_porcelain") != []
    ):
        errors.append("result git provenance does not match the clean Slurm provenance")
    if provenance.get("task_config") != _expected_task_config(task):
        errors.append("result provenance task_config does not match the frozen task")

    data = provenance.get("data")
    expected_data_keys = {
        "dataset_path",
        "metadata_path",
        "manifest_path",
        "file_sha256",
        "manifest_sha256",
        "generator_config",
    }
    if not isinstance(data, dict) or set(data) != expected_data_keys:
        errors.append("result data provenance has the wrong schema")
        data = {}
    if data.get("generator_config") != _expected_generator_config(task):
        errors.append("result generator config does not match the frozen corpus task")
    expected_names = {
        "dataset_path": f"{task.dataset_stem}.npz",
        "metadata_path": f"{task.dataset_stem}.metadata.json",
        "manifest_path": f"{task.dataset_stem}.sha256.json",
    }
    for name, basename in expected_names.items():
        if not isinstance(data.get(name), str) or Path(data[name]).name != basename:
            errors.append(f"result data provenance {name} does not match the task corpus")
            continue
        recorded_entry = next(
            (value for key, value in dataset_entries.items() if Path(key).name == basename),
            None,
        )
        if (
            recorded_entry is None
            or Path(data[name]).resolve() != Path(recorded_entry[1]).resolve()
        ):
            errors.append(f"result data provenance {name} path disagrees with dataset-files.sha256")
    file_hashes = data.get("file_sha256")
    if not isinstance(file_hashes, dict) or set(file_hashes) != {
        expected_names["dataset_path"],
        expected_names["metadata_path"],
    }:
        errors.append("result data file_sha256 has the wrong file set")
    else:
        for filename, digest in file_hashes.items():
            entry = next(
                (value for key, value in dataset_entries.items() if Path(key).name == filename),
                None,
            )
            if entry is None or digest != entry[0]:
                errors.append(
                    f"result data hash does not match dataset-files.sha256 for {filename}"
                )
    manifest_entry = next(
        (
            value
            for key, value in dataset_entries.items()
            if Path(key).name == expected_names["manifest_path"]
        ),
        None,
    )
    if manifest_entry is None or data.get("manifest_sha256") != manifest_entry[0]:
        errors.append("result manifest hash does not match dataset-files.sha256")

    dependencies = provenance.get("dependencies")
    if not isinstance(dependencies, dict) or set(dependencies) != {"pyproject.toml", "uv.lock"}:
        errors.append("result dependency provenance has the wrong file set")
    else:
        for name in ("pyproject.toml", "uv.lock"):
            entry = dependency_entries.get(name)
            if (
                not isinstance(dependencies[name], dict)
                or set(dependencies[name]) != {"sha256"}
                or entry is None
                or dependencies[name]["sha256"] != entry[0]
            ):
                errors.append(f"result dependency hash does not match {name}")
    submodules = provenance.get("submodules")
    if not isinstance(submodules, dict) or set(submodules) != {"status", "tera_gitlink"}:
        errors.append("result submodule provenance has the wrong schema")
        submodules = {}
    if submodules.get("status") != submodule_lines:
        errors.append("result submodule status does not match git-submodules.txt")
    if submodules.get("tera_gitlink") != env.get("tera_gitlink"):
        errors.append("result TERA gitlink does not match Slurm provenance")
    runtime_provenance = provenance.get("runtime")
    if not isinstance(runtime_provenance, dict) or set(runtime_provenance) != {
        "python",
        "platform",
        "torch",
        "numpy",
        "device",
        "dtype",
    }:
        errors.append("result runtime provenance has the wrong schema")
    elif (
        runtime_provenance.get("python") != runtime.get("python_version")
        or runtime_provenance.get("torch") != runtime.get("torch")
        or runtime_provenance.get("numpy") != runtime.get("numpy")
        or runtime_provenance.get("platform") != runtime.get("platform")
        or runtime_provenance.get("device") != "cuda"
        or runtime_provenance.get("dtype") != "float32"
    ):
        errors.append("result runtime provenance does not match runtime.json/frozen device")

    catalog = result.get("catalog")
    if not isinstance(catalog, dict) or set(catalog) != {
        "path",
        "sha256",
        "generation_git_commit",
        "generation_git_tree",
        "task_index",
    }:
        errors.append("result catalog record has the wrong schema")
        catalog = {}
    particle_position = (2, 4, 6, 8, 10).index(task.n_particles)
    expected_catalog_index = task.replica * 5 + particle_position
    if (
        catalog.get("sha256") != EXPECTED_CATALOG_SHA256
        or catalog.get("task_index") != expected_catalog_index
        or not isinstance(catalog.get("path"), str)
        or Path(catalog.get("path", "")).resolve() != Path(env.get("catalog_path", "")).resolve()
    ):
        errors.append("result catalog identity does not match the frozen catalog/task")
    for name in ("generation_git_commit", "generation_git_tree"):
        if _HEX40_RE.fullmatch(str(catalog.get(name, ""))) is None:
            errors.append(f"result catalog {name} is not a Git object id")
    return str(catalog.get("generation_git_commit", "")), str(
        catalog.get("generation_git_tree", "")
    )


def _load_task(path: Path, task: InternalOptimizerTask) -> _LoadedTask:
    if not path.exists():
        return _LoadedTask(record=_task_record(path, task, status="missing"))
    if not path.is_dir() or path.is_symlink():
        return _LoadedTask(
            record=_task_record(
                path,
                task,
                status="invalid",
                errors=("expected task path is not a real, non-symlink directory",),
            )
        )
    record = _task_record(path, task, status="invalid")
    exit_code_path = path / "exit-code.txt"
    if not exit_code_path.is_file():
        record["errors"].append("missing exit-code.txt")
        return _LoadedTask(record=record)
    try:
        exit_code = int(exit_code_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        record["errors"].append("exit-code.txt is not a single integer")
        return _LoadedTask(record=record)
    record["exit_code"] = exit_code
    if exit_code != 0:
        record["status"] = "failed"
        record["errors"].append(f"task exited with status {exit_code}")
        return _LoadedTask(record=record)
    missing = sorted(name for name in _REQUIRED_ARTIFACTS if not (path / name).is_file())
    if missing:
        record["errors"].append(f"missing required artifacts: {', '.join(missing)}")
        return _LoadedTask(record=record)

    errors: list[str] = record["errors"]
    try:
        result = _read_json(path / "result.json")
        env = _parse_env(path / "provenance.env")
        artifact_entries = _parse_sha256sum(path / "artifacts.sha256", basenames=True)
        dataset_entries = _parse_sha256sum(path / "dataset-files.sha256")
        source_entries = _parse_sha256sum(path / "source-files.sha256")
        dependency_entries = _parse_sha256sum(path / "dependency-files.sha256", basenames=True)
        dependency_audit = _read_json(path / "dependency-audit.json")
        runtime = _read_json(path / "runtime.json")
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as error:
        errors.append(f"could not parse task artifacts: {error}")
        return _LoadedTask(record=record)

    if set(artifact_entries) != _HASHED_ARTIFACTS:
        errors.append(
            "artifacts.sha256 has the wrong artifact set: "
            f"expected {sorted(_HASHED_ARTIFACTS)}, got {sorted(artifact_entries)}"
        )
    for name in _HASHED_ARTIFACTS:
        entry = artifact_entries.get(name)
        if entry is None or entry[0] != _sha256_file(path / name):
            errors.append(f"artifacts.sha256 does not verify {name}")

    if (path / "git-status.txt").read_text(encoding="utf-8").strip():
        errors.append("git-status.txt is not clean")
    raw_submodules = (path / "git-submodules.txt").read_text(encoding="utf-8").splitlines()
    submodule_lines = [line.rstrip() for line in raw_submodules if line.strip()]
    if not submodule_lines or any(not line.startswith(" ") for line in submodule_lines):
        errors.append("git-submodules.txt is empty, modified, or uninitialized")
    normalized_submodules = [line.strip() for line in submodule_lines]
    submodules = "\n".join(normalized_submodules)
    tera_line = next(
        (line for line in normalized_submodules if " gp/tera/vendor " in f" {line} "),
        "",
    )
    if not tera_line or not tera_line.startswith(env.get("tera_gitlink", "")):
        errors.append("git-submodules.txt does not verify the recorded TERA gitlink")

    required_env = {
        "captured_at_utc",
        "repo_root",
        "git_commit",
        "git_tree",
        "git_describe",
        "tera_gitlink",
        "hostname",
        "python",
        "array_job_id",
        "array_task_id",
        "replica",
        "n_particles",
        "n_dims",
        "train_steps",
        "seed",
        "dataset_stem",
        "catalog_path",
        "catalog_sha256",
        "exclusive_verification_mode",
        "slurm_job_id",
        "slurm_job_nodelist",
        "slurm_cpus_per_task",
        "cuda_visible_devices",
    }
    if set(env) != required_env:
        errors.append("provenance.env has the wrong key set")
    expected_env = {
        "array_task_id": str(task.task_index),
        "replica": str(task.replica),
        "n_particles": str(task.n_particles),
        "n_dims": str(task.n_dims),
        "train_steps": str(task.train_steps),
        "seed": str(task.seed),
        "dataset_stem": task.dataset_stem,
        "catalog_sha256": EXPECTED_CATALOG_SHA256,
        "slurm_cpus_per_task": "16",
    }
    if any(env.get(name) != value for name, value in expected_env.items()):
        errors.append("provenance.env task/config identity does not match the grid")
    for name in ("git_commit", "git_tree", "tera_gitlink"):
        if _HEX40_RE.fullmatch(env.get(name, "")) is None:
            errors.append(f"provenance.env {name} is not a Git object id")
    if "dirty" in env.get("git_describe", "").lower():
        errors.append("provenance.env git_describe reports a dirty tree")
    if not env.get("array_job_id") or not env.get("slurm_job_id") or not env.get("repo_root"):
        errors.append("provenance.env is missing Slurm/repository identity")
    run_job_match = re.fullmatch(r"job-(\d+)", path.parent.name)
    if run_job_match is not None and env.get("array_job_id") != run_job_match.group(1):
        errors.append("task array_job_id does not match the job-* run directory")

    catalog_entries = [
        entry for key, entry in dataset_entries.items() if Path(key).name == "catalog.json"
    ]
    expected_dataset_names = {
        "catalog.json",
        f"{task.dataset_stem}.npz",
        f"{task.dataset_stem}.metadata.json",
        f"{task.dataset_stem}.sha256.json",
    }
    if {Path(key).name for key in dataset_entries} != expected_dataset_names:
        errors.append("dataset-files.sha256 has the wrong catalog/corpus file set")
    if len(catalog_entries) != 1 or catalog_entries[0][0] != EXPECTED_CATALOG_SHA256:
        errors.append("dataset-files.sha256 does not authenticate the frozen catalog")
    for digest, raw_path in dataset_entries.values():
        artifact_path = Path(raw_path)
        if not artifact_path.is_file():
            errors.append(f"dataset-files.sha256 recorded file is unavailable: {raw_path}")
        elif _sha256_file(artifact_path) != digest:
            errors.append(f"dataset-files.sha256 does not verify {raw_path}")
    catalog_path = env.get("catalog_path", "")
    if catalog_entries and Path(catalog_entries[0][1]).resolve() != Path(catalog_path).resolve():
        errors.append("dataset-files.sha256 catalog path disagrees with provenance.env")

    repo_root = env.get("repo_root", "")
    source_keys = set(source_entries)
    if source_keys != _REQUIRED_SOURCE_PATHS:
        errors.append(
            "source-files.sha256 has the wrong frozen source closure: "
            f"expected {sorted(_REQUIRED_SOURCE_PATHS)}, got {sorted(source_keys)}"
        )
    _validate_manifest_files(
        source_entries,
        root=Path(repo_root) if repo_root else None,
        label="source-files.sha256",
        errors=errors,
    )
    if set(dependency_entries) != {"pyproject.toml", "uv.lock"}:
        errors.append("dependency-files.sha256 must cover exactly pyproject.toml and uv.lock")
    for name, (digest, raw_path) in dependency_entries.items():
        dependency_path = Path(raw_path)
        if not dependency_path.is_absolute():
            dependency_path = Path(repo_root) / dependency_path
        if not dependency_path.is_file() or _sha256_file(dependency_path) != digest:
            errors.append(f"dependency-files.sha256 does not verify {name}")

    package_lines = (path / "dependency-packages.txt").read_text(encoding="utf-8").splitlines()
    _validate_dependency_audit(dependency_audit, package_lines, errors)
    _validate_runtime(runtime, env, errors)
    if isinstance(dependency_audit, dict) and isinstance(runtime, dict):
        audit_python = dependency_audit.get("python_executable")
        runtime_python = runtime.get("python_executable")
        if not isinstance(audit_python, str) or not isinstance(runtime_python, str):
            errors.append("dependency/runtime Python executable identity is malformed")
        elif Path(audit_python).resolve() != Path(runtime_python).resolve():
            errors.append("dependency audit and runtime Python executables disagree")
    job_record = (path / "slurm-job.txt").read_text(encoding="utf-8")
    node_record = (path / "slurm-node.txt").read_text(encoding="utf-8")
    node_jobs = (path / "slurm-jobs-on-node.txt").read_text(encoding="utf-8")
    _validate_exclusivity(
        env.get("exclusive_verification_mode", ""),
        job_record,
        node_record,
        node_jobs,
        env.get("array_job_id", ""),
        errors,
    )
    _validate_command(path / "command.txt", task, path, catalog_path, errors)

    tera_metrics, scalar_rmse = _validate_result(result, task, errors)
    catalog_generation_commit, catalog_generation_tree = _validate_result_provenance(
        result if isinstance(result, dict) else {},
        task,
        env,
        source_entries,
        dependency_entries,
        dataset_entries,
        submodule_lines,
        runtime if isinstance(runtime, dict) else {},
        errors,
    )
    if scalar_rmse is None:
        errors.append("TERA-50 selection scalar RMSE could not be computed")

    loaded = _LoadedTask(
        record=record,
        result_sha256=_sha256_file(path / "result.json"),
        tera_metrics=tera_metrics,
        scalar_rmse=scalar_rmse,
        commit=env.get("git_commit", ""),
        tree=env.get("git_tree", ""),
        submodules=submodules,
        tera_gitlink=env.get("tera_gitlink", ""),
        source_manifest_sha256=_sha256_file(path / "source-files.sha256"),
        dependency_manifest_sha256=_sha256_file(path / "dependency-files.sha256"),
        dependency_audit_sha256=_sha256_file(path / "dependency-audit.json"),
        packages_sha256=_sha256_file(path / "dependency-packages.txt"),
        array_job_id=env.get("array_job_id", ""),
        repo_root=repo_root,
        catalog_path=str(Path(catalog_path).resolve()) if catalog_path else "",
        catalog_sha256=env.get("catalog_sha256", ""),
        catalog_generation_commit=catalog_generation_commit,
        catalog_generation_tree=catalog_generation_tree,
        exclusive_mode=env.get("exclusive_verification_mode", ""),
    )
    if errors:
        record["status"] = "invalid"
    else:
        record["status"] = "valid"
        record["result_sha256"] = loaded.result_sha256
        record["tera_50_standardized_scalar_rmse"] = scalar_rmse
    return loaded


def parse_task_indices(value: str) -> tuple[int, ...]:
    """Parse a nonempty comma/range declaration into ordered unique indices."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("expected task indices must be a nonempty declaration")
    parsed: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError("expected task indices contain an empty component")
        if re.fullmatch(r"\d+", part):
            parsed.append(int(part))
            continue
        match = re.fullmatch(r"(\d+)-(\d+)", part)
        if match is None:
            raise ValueError(f"invalid expected task index component: {part!r}")
        start, stop = map(int, match.groups())
        if stop < start:
            raise ValueError(f"descending expected task index range: {part!r}")
        parsed.extend(range(start, stop + 1))
    if any(index < 0 or index >= TASK_COUNT for index in parsed):
        raise ValueError(f"expected task indices must lie in [0, {TASK_COUNT})")
    if len(set(parsed)) != len(parsed):
        raise ValueError("expected task indices contain duplicates")
    if parsed != sorted(parsed):
        raise ValueError("expected task indices must be declared in increasing order")
    return tuple(parsed)


def _unexpected_task_directories(
    run_root: Path,
    expected_tasks: Sequence[InternalOptimizerTask],
) -> tuple[list[dict[str, Any]], list[int]]:
    expected_names = {_expected_directory_name(task) for task in expected_tasks}
    expected_indices = {task.task_index for task in expected_tasks}
    unexpected: list[dict[str, Any]] = []
    observed_indices: list[int] = []
    if not run_root.is_dir():
        return unexpected, []
    for path in sorted(run_root.iterdir(), key=lambda item: item.name):
        if not path.name.startswith("task-"):
            continue
        prefix_match = _TASK_INDEX_PREFIX_RE.match(path.name)
        parsed_index = int(prefix_match.group("index")) if prefix_match is not None else None
        if parsed_index is not None:
            observed_indices.append(parsed_index)
        if path.name in expected_names and path.is_dir() and not path.is_symlink():
            match = _TASK_DIRECTORY_RE.fullmatch(path.name)
            if match is not None:
                continue
        reason = "undeclared or malformed task directory"
        if path.name in expected_names and not path.is_dir():
            reason = "declared task path is not a directory"
        elif parsed_index in expected_indices:
            reason = "duplicate or identity-mismatched directory for a declared task index"
        unexpected.append(
            {
                "path": str(path),
                "name": path.name,
                "parsed_task_index": parsed_index,
                "reason": reason,
            }
        )
    counts = Counter(observed_indices)
    duplicates = sorted(index for index, count in counts.items() if count > 1)
    return unexpected, duplicates


def _same_value(tasks: Sequence[_LoadedTask], attribute: str) -> bool:
    values = [getattr(task, attribute) for task in tasks]
    return bool(values) and all(value == values[0] and value not in {"", None} for value in values)


def _metric_payload(loaded: _LoadedTask, task: InternalOptimizerTask) -> dict[str, Any]:
    return {
        "task_index": task.task_index,
        "replica": task.replica,
        "n_particles": task.n_particles,
        "dimension": 2 * task.n_particles * task.n_dims,
        "train_steps": task.train_steps,
        "seed": task.seed,
        "trajectory_mse_aggregation_order": "sqrt(mean(trajectory standardized_mse))",
        "standardized_scalar_rmse": loaded.scalar_rmse,
        "tera_50_metrics": loaded.tera_metrics,
    }


def _selection_metrics(
    loaded_by_index: dict[int, _LoadedTask],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    per_corpus: list[dict[str, Any]] = []
    update_summaries: list[dict[str, Any]] = []
    for updates in UPDATE_BUDGETS:
        corpus_rows: list[dict[str, Any]] = []
        for replica in (0, 1, 2):
            for n_particles in (2, 4, 6, 8, 10):
                tasks = [
                    task
                    for task in OPTIMIZER_SELECTION_TASKS
                    if task.replica == replica
                    and task.n_particles == n_particles
                    and task.train_steps == updates
                ]
                if [task.seed for task in tasks] != list(SEEDS):
                    raise RuntimeError("internal F02 task grid no longer has the frozen seed order")
                seed_rmse = [loaded_by_index[task.task_index].scalar_rmse for task in tasks]
                if any(value is None for value in seed_rmse):
                    raise RuntimeError("analysis-ready F02 task is missing scalar RMSE")
                numeric = [float(value) for value in seed_rmse if value is not None]
                # Protocol order: average the three already-rooted seed/corpus RMSE values.
                mean_seed_rmse = sum(numeric) / len(numeric)
                row = {
                    "replica": replica,
                    "n_particles": n_particles,
                    "dimension": 6 * n_particles,
                    "train_steps": updates,
                    "seed_order": list(SEEDS),
                    "seed_task_indices": [task.task_index for task in tasks],
                    "seed_standardized_scalar_rmse": numeric,
                    "mean_seed_standardized_scalar_rmse": mean_seed_rmse,
                }
                corpus_rows.append(row)
                per_corpus.append(row)
        if len(corpus_rows) != 15:
            raise RuntimeError("internal F02 update summary does not contain 15 corpora")
        # Final protocol layer: equal weight for the 3 replicas x 5 dimensions.
        macro = sum(row["mean_seed_standardized_scalar_rmse"] for row in corpus_rows) / len(
            corpus_rows
        )
        update_summaries.append(
            {
                "train_steps": updates,
                "corpus_count": 15,
                "macro_mean_standardized_scalar_rmse": macro,
            }
        )
    winner = min(
        update_summaries,
        key=lambda row: (row["macro_mean_standardized_scalar_rmse"], row["train_steps"]),
    )
    return per_corpus, update_summaries, int(winner["train_steps"])


def aggregate(
    run_root: str | Path,
    expected_task_indices: Sequence[int],
) -> dict[str, Any]:
    """Validate a declared F02 task population and select only for the full grid."""

    root = Path(run_root).resolve()
    indices = tuple(expected_task_indices)
    if not indices:
        raise ValueError("expected_task_indices must not be empty")
    if any(not _is_int(index) for index in indices):
        raise ValueError("expected_task_indices must contain integers")
    if len(set(indices)) != len(indices):
        raise ValueError("expected_task_indices contain duplicates")
    if indices != tuple(sorted(indices)):
        raise ValueError("expected_task_indices must be in increasing order")
    if any(index < 0 or index >= TASK_COUNT for index in indices):
        raise ValueError(f"expected_task_indices must lie in [0, {TASK_COUNT})")
    expected_tasks = tuple(task_for_index(index) for index in indices)
    unexpected, duplicate_indices = _unexpected_task_directories(root, expected_tasks)
    loaded_tasks = [
        _load_task(root / _expected_directory_name(task), task) for task in expected_tasks
    ]
    structurally_valid = [task for task in loaded_tasks if task.record["status"] == "valid"]

    provenance_dimensions = (
        "commit",
        "tree",
        "submodules",
        "tera_gitlink",
        "source_manifest_sha256",
        "dependency_manifest_sha256",
        "dependency_audit_sha256",
        "packages_sha256",
        "array_job_id",
        "repo_root",
        "catalog_path",
        "catalog_sha256",
        "catalog_generation_commit",
        "catalog_generation_tree",
        "exclusive_mode",
    )
    same = {name: _same_value(structurally_valid, name) for name in provenance_dimensions}
    all_expected_valid = len(structurally_valid) == len(expected_tasks)
    provenance_verified = all_expected_valid and all(same.values())
    no_unexpected = not unexpected and not duplicate_indices
    declared_subset_ready = all_expected_valid and provenance_verified and no_unexpected
    for loaded in structurally_valid:
        loaded.record["eligible_for_declared_aggregate"] = declared_subset_ready
        if not provenance_verified:
            loaded.record["errors"].append(
                "task is structurally valid but cross-task provenance is inconsistent"
            )

    counts = Counter(task.record["status"] for task in loaded_tasks)
    status_counts = {
        "valid": counts.get("valid", 0),
        "missing": counts.get("missing", 0),
        "failed": counts.get("failed", 0),
        "invalid": counts.get("invalid", 0),
        "unexpected": len(unexpected),
    }
    full_grid_requested = indices == FULL_TASK_INDICES
    analysis_ready = declared_subset_ready and full_grid_requested
    per_task_metrics = [
        _metric_payload(loaded, task)
        for task, loaded in zip(expected_tasks, loaded_tasks, strict=True)
        if loaded.record["status"] == "valid"
    ]
    per_corpus: list[dict[str, Any]] = []
    update_summaries: list[dict[str, Any]] = []
    selected_update: int | None = None
    selection_status = "not_run_incomplete_or_nonfull_declared_grid"
    if analysis_ready:
        by_index = {
            task.task_index: loaded
            for task, loaded in zip(expected_tasks, loaded_tasks, strict=True)
        }
        per_corpus, update_summaries, selected_update = _selection_metrics(by_index)
        selection_status = "complete"

    report = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "aggregate_type": "f02_internal_optimizer_selection",
        "input": {
            "run_root": str(root),
            "expected_task_indices": list(indices),
            "expected_task_count": len(indices),
            "full_grid_indices": [0, TASK_COUNT - 1],
            "full_grid_requested": full_grid_requested,
        },
        "task_accounting": {
            "all_expected_tasks_valid": all_expected_valid,
            "no_unexpected_task_directories": no_unexpected,
            "status_counts": status_counts,
            "duplicate_task_indices": duplicate_indices,
            "unexpected_task_directories": unexpected,
            "tasks": [task.record for task in loaded_tasks],
        },
        "provenance": {
            "verified": provenance_verified,
            "aggregator_source": {
                "path": "cluster/aggregate_f02_internal_optimizer.py",
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            **{f"same_{name}": value for name, value in same.items()},
            "git_commit": structurally_valid[0].commit if same["commit"] else None,
            "git_tree": structurally_valid[0].tree if same["tree"] else None,
            "tera_gitlink": (structurally_valid[0].tera_gitlink if same["tera_gitlink"] else None),
            "source_manifest_sha256": (
                structurally_valid[0].source_manifest_sha256
                if same["source_manifest_sha256"]
                else None
            ),
            "dependency_manifest_sha256": (
                structurally_valid[0].dependency_manifest_sha256
                if same["dependency_manifest_sha256"]
                else None
            ),
            "dependency_audit_sha256": (
                structurally_valid[0].dependency_audit_sha256
                if same["dependency_audit_sha256"]
                else None
            ),
            "packages_sha256": (
                structurally_valid[0].packages_sha256 if same["packages_sha256"] else None
            ),
            "slurm_array_job_id": (
                structurally_valid[0].array_job_id if same["array_job_id"] else None
            ),
            "repo_root": structurally_valid[0].repo_root if same["repo_root"] else None,
            "catalog_path": (structurally_valid[0].catalog_path if same["catalog_path"] else None),
            "catalog_sha256": (
                structurally_valid[0].catalog_sha256 if same["catalog_sha256"] else None
            ),
            "catalog_generation_git_commit": (
                structurally_valid[0].catalog_generation_commit
                if same["catalog_generation_commit"]
                else None
            ),
            "catalog_generation_git_tree": (
                structurally_valid[0].catalog_generation_tree
                if same["catalog_generation_tree"]
                else None
            ),
            "exclusive_verification_mode": (
                structurally_valid[0].exclusive_mode if same["exclusive_mode"] else None
            ),
        },
        "metrics": {
            "selection_arm": "TERA-50",
            "selection_metric": "standardized_scalar_rmse",
            "uses_nll": False,
            "aggregation_order": [
                "within each seed/corpus: sqrt(mean trajectory standardized_mse)",
                "within each replica-dimension-update: arithmetic mean across seeds 11,29,47",
                (
                    "within each update: equal-weight arithmetic mean across "
                    "15 replica-dimension corpora"
                ),
            ],
            "exact_tie_rule": (
                "choose fewer optimizer updates only when macro RMSE values are numerically equal"
            ),
            "selection_status": selection_status,
            "per_task": per_task_metrics,
            "per_corpus": per_corpus,
            "update_summaries": update_summaries,
        },
        "declared_subset_ready": declared_subset_ready,
        "analysis_ready": analysis_ready,
        "selected_update": selected_update,
    }
    # Fail here rather than emitting non-standard NaN/Infinity JSON.
    _canonical_bytes(report)
    return report


def save_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Atomically write one strict, newline-terminated JSON aggregate."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--expected-task-indices",
        required=True,
        help="explicit comma/range declaration, e.g. 0 for pilot or 0-134 for the full grid",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        indices = parse_task_indices(args.expected_task_indices)
        report = aggregate(args.run_root, indices)
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as error:
        build_parser().error(str(error))
    save_report(report, args.out)
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AGGREGATE_SCHEMA_VERSION",
    "EXPECTED_CATALOG_SHA256",
    "FULL_TASK_INDICES",
    "aggregate",
    "build_parser",
    "main",
    "parse_task_indices",
    "save_report",
]
