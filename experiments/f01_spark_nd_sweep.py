"""Preregistered SPARK-versus-TERA sweep over particle count and spatial dimension.

The Cartesian product ``n in {2, 4, 10}`` and ``d in {1, 2, 3}`` is evaluated on three
independently generated fixed-mass systems per configuration. Complete trajectories are assigned
to either 1,500 training rows or 500 test rows. SPARK receives the disclosed particle schema but
not true masses or force-law constants; TERA receives the same standardized labeled rows. Both
arms are fixed before generation. The comparison reports standardized value RMSE, gradient RMSE,
and raw Gaussian value NLL. Runtime and memory are diagnostic only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import torch

from data.get_nbody_sweep import (
    ARCHIVE_KEYS,
    DT,
    GRAVITATIONAL_CONSTANT,
    N_TRAJECTORIES,
    ROWS_PER_TRAJECTORY,
    SOFTENING,
    STEPS_PER_TRAJECTORY,
    dataset_filename,
    file_sha256,
    generate_nbody_sweep_dataset,
    save_dataset,
)
from gp.exact import gaussian_nll
from gp.metrics import rmse
from gp.spark import fit_spark, prepare_spark
from gp.tera import run_tera

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "f01_spark_nd_sweep_v1"
SUMMARY_PROTOCOL = "f01_spark_nd_sweep_summary_v1"
PARTICLE_COUNTS = (2, 4, 10)
SPATIAL_DIMS = (1, 2, 3)
REPLICATES = (0, 1, 2)
ARMS = ("spark", "tera")
TRAIN_TRAJECTORIES = 30
TEST_TRAJECTORIES = 10
TRAIN_ROWS = TRAIN_TRAJECTORIES * ROWS_PER_TRAJECTORY
TEST_ROWS = TEST_TRAJECTORIES * ROWS_PER_TRAJECTORY

SPARK_CONFIG = {
    "rank": 128,
    "inducing_strategy": "hybrid_log",
    "lengthscale_multiplier": 1.0,
    "value_noise": 1e-3,
    "gradient_noise": 1e-3,
    "point_chunk": 256,
}
TERA_CONFIG = {
    "neighbors": 20,
    "kernel": "rbf",
    "epochs": 20,
    "precision": "float32",
}
METRICS = ("value_rmse", "gradient_rmse", "raw_nll")
SOURCE_FILES = (
    "data/get_nbody.py",
    "data/get_nbody_sweep.py",
    "experiments/f01_spark_nd_sweep.py",
    "gp/exact/__init__.py",
    "gp/metrics.py",
    "gp/spark/__init__.py",
    "gp/spark/model.py",
    "gp/spark/radial.py",
    "gp/spark/residual.py",
    "gp/spark/structure.py",
    "gp/tera/__init__.py",
    "pyproject.toml",
    "uv.lock",
)


def _simulator_seed(n_particles: int, n_dims: int, replicate: int) -> int:
    return 2_026_080_500 + 100 * n_particles + 10 * n_dims + replicate


CELL_SPECS = tuple(
    {
        "cell_id": f"n{n_particles}_d{n_dims}_seed{replicate}",
        "n_particles": n_particles,
        "spatial_dims": n_dims,
        "state_dimension": 2 * n_particles * n_dims,
        "replicate": replicate,
        "simulator_seed": _simulator_seed(n_particles, n_dims, replicate),
        "dataset_name": dataset_filename(
            n_particles,
            n_dims,
            _simulator_seed(n_particles, n_dims, replicate),
        ),
    }
    for n_particles in PARTICLE_COUNTS
    for n_dims in SPATIAL_DIMS
    for replicate in REPLICATES
)
GENERATION_TASK_IDS = tuple(range(len(CELL_SPECS)))
TASK_IDS = tuple(range(len(CELL_SPECS) * len(ARMS)))


def generation_assignment(task_id: int) -> dict[str, Any]:
    if type(task_id) is not int or task_id not in GENERATION_TASK_IDS:
        raise ValueError(f"generation task ID must be one of {GENERATION_TASK_IDS}")
    return dict(CELL_SPECS[task_id])


def task_assignment(task_id: int) -> tuple[dict[str, Any], str]:
    if type(task_id) is not int or task_id not in TASK_IDS:
        raise ValueError(f"benchmark task ID must be one of {TASK_IDS}")
    cell_index, arm_index = divmod(task_id, len(ARMS))
    return dict(CELL_SPECS[cell_index]), ARMS[arm_index]


def result_relative_path(cell: Mapping[str, Any], arm: str) -> Path:
    return Path("results") / str(cell["cell_id"]) / f"{arm}.json"


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_snapshot() -> dict[str, Any]:
    """Bind results to one clean source commit and the pinned TERA submodule."""

    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("benchmark source checkout has tracked modifications")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tera_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT / "gp" / "tera" / "vendor",
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    files = {name: _sha256(ROOT / name) for name in SOURCE_FILES}
    return {"git_commit": commit, "tera_commit": tera_commit, "files": files}


def _load_dataset(path: Path, cell: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    if path.name != cell["dataset_name"] or not path.is_file() or path.is_symlink():
        raise ValueError("dataset path differs from its assigned regular file")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(ARCHIVE_KEYS):
            raise ValueError("dataset archive schema drifted")
        if int(archive["n_particles"]) != cell["n_particles"]:
            raise ValueError("dataset particle count differs from its cell")
        if int(archive["n_dims"]) != cell["spatial_dims"]:
            raise ValueError("dataset spatial dimension differs from its cell")
        if int(archive["seed"]) != cell["simulator_seed"]:
            raise ValueError("dataset seed differs from its cell")
        if int(archive["n_trajectories"]) != N_TRAJECTORIES:
            raise ValueError("dataset trajectory count drifted")
        if int(archive["rows_per_trajectory"]) != ROWS_PER_TRAJECTORY:
            raise ValueError("dataset rows per trajectory drifted")
        if int(archive["steps_per_trajectory"]) != STEPS_PER_TRAJECTORY:
            raise ValueError("dataset integration length drifted")
        if float(archive["dt"]) != DT or float(archive["G"]) != GRAVITATIONAL_CONSTANT:
            raise ValueError("dataset integration constants drifted")
        if float(archive["softening"]) != SOFTENING:
            raise ValueError("dataset softening drifted")
        X = np.asarray(archive["X"], dtype=np.float64).copy()
        value = np.asarray(archive["E"], dtype=np.float64).copy()
        gradient = np.asarray(archive["F"], dtype=np.float64).copy()
        trajectory_id = np.asarray(archive["trajectory_id"], dtype=np.int64).copy()
    return X, value, gradient, trajectory_id


def _split_and_scale(
    X: np.ndarray,
    value: np.ndarray,
    gradient: np.ndarray,
    trajectory_id: np.ndarray,
) -> dict[str, torch.Tensor]:
    train = trajectory_id < TRAIN_TRAJECTORIES
    test = (trajectory_id >= TRAIN_TRAJECTORIES) & (
        trajectory_id < TRAIN_TRAJECTORIES + TEST_TRAJECTORIES
    )
    if int(train.sum()) != TRAIN_ROWS or int(test.sum()) != TEST_ROWS:
        raise ValueError("trajectory-disjoint role counts drifted")
    if bool(np.any(train & test)):
        raise RuntimeError("train and test trajectories overlap")

    x_offset = X[train].min(axis=0)
    x_scale = max(float(np.ptp(X[train], axis=0).max()), 1e-12)
    value_mean = float(value[train].mean())
    value_scale = max(float(value[train].std(ddof=0)), 1e-8)
    scaled_X = (X - x_offset) / x_scale
    scaled_value = (value - value_mean) / value_scale
    scaled_gradient = gradient * (x_scale / value_scale)
    return {
        "train_X": torch.from_numpy(scaled_X[train]),
        "train_value": torch.from_numpy(scaled_value[train]),
        "train_gradient": torch.from_numpy(scaled_gradient[train]),
        "train_trajectory_id": torch.from_numpy(trajectory_id[train]),
        "test_X": torch.from_numpy(scaled_X[test]),
        "test_value": torch.from_numpy(scaled_value[test]),
        "test_gradient": torch.from_numpy(scaled_gradient[test]),
        "coordinate_offset": torch.from_numpy(x_offset / x_scale),
    }


def _spark_metrics(data: Mapping[str, torch.Tensor], cell: Mapping[str, Any]) -> tuple[dict, dict]:
    prepared = prepare_spark(
        data["train_X"],
        data["train_value"],
        data["train_gradient"],
        data["train_trajectory_id"],
        n_particles=int(cell["n_particles"]),
        spatial_dims=int(cell["spatial_dims"]),
        coordinate_offset=data["coordinate_offset"],
    )
    started = time.perf_counter()
    model = fit_spark(prepared, **SPARK_CONFIG)
    mean, gradient, variance = model.predict(data["test_X"])
    return (
        {
            "value_rmse": rmse(mean, data["test_value"]),
            "gradient_rmse": rmse(gradient, data["test_gradient"]),
            "raw_nll": gaussian_nll(data["test_value"], mean, variance),
        },
        {
            "configuration": dict(SPARK_CONFIG),
            "effective_rank": model.radial_gp.basis.rank,
            "fit_and_predict_wall_s_diagnostic": time.perf_counter() - started,
        },
    )


def _tera_metrics(data: Mapping[str, torch.Tensor], cell: Mapping[str, Any]) -> tuple[dict, dict]:
    tensors = [
        data[name].float()
        for name in (
            "train_X",
            "train_value",
            "train_gradient",
            "test_X",
            "test_value",
            "test_gradient",
        )
    ]
    mean, variance, gradient_rmse, fit_s = run_tera(
        *tensors,
        m=TERA_CONFIG["neighbors"],
        kernel=TERA_CONFIG["kernel"],
        train_epochs=TERA_CONFIG["epochs"],
        seed=int(cell["replicate"]),
    )
    metrics = {
        "value_rmse": rmse(mean, tensors[4]),
        "gradient_rmse": float(gradient_rmse),
        "raw_nll": gaussian_nll(tensors[4], mean, variance),
    }
    return metrics, {"configuration": dict(TERA_CONFIG), "fit_wall_s_diagnostic": fit_s}


def _runtime() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
    }


def run_task(
    task_id: int,
    data_dir: str | Path,
    *,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    cell, arm = task_assignment(task_id)
    dataset_path = Path(data_dir) / cell["dataset_name"]
    X, value, gradient, trajectory_id = _load_dataset(dataset_path, cell)
    data = _split_and_scale(X, value, gradient, trajectory_id)
    torch.manual_seed(int(cell["replicate"]))
    started_wall = time.perf_counter()
    started_usage = resource.getrusage(resource.RUSAGE_SELF)
    metrics, details = _spark_metrics(data, cell) if arm == "spark" else _tera_metrics(data, cell)
    ended_usage = resource.getrusage(resource.RUSAGE_SELF)
    if set(metrics) != set(METRICS) or not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("benchmark arm produced invalid metrics")
    return {
        "protocol": PROTOCOL,
        "complete": True,
        "task_id": task_id,
        **cell,
        "arm": arm,
        "dataset_sha256": file_sha256(dataset_path),
        "training_rows": TRAIN_ROWS,
        "test_rows": TEST_ROWS,
        "fixed_before_data": True,
        "metrics": metrics,
        "details": details,
        "resource_record": {
            "user_cpu_s": ended_usage.ru_utime - started_usage.ru_utime,
            "system_cpu_s": ended_usage.ru_stime - started_usage.ru_stime,
            "wall_s": time.perf_counter() - started_wall,
            "max_rss_kib": ended_usage.ru_maxrss,
            "interpretation": "diagnostic only; shared-node wall time is not normalized cost",
        },
        "runtime": _runtime(),
        "source_snapshot": dict(snapshot),
    }


def _load_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_bytes())
    if result.get("protocol") != PROTOCOL or result.get("complete") is not True:
        raise ValueError(f"{path} is not a complete {PROTOCOL} result")
    return result


def summarize(result_root: str | Path) -> dict[str, Any]:
    root = Path(result_root)
    grouped: dict[tuple[int, int], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    commits = set()
    for task_id in TASK_IDS:
        expected_cell, expected_arm = task_assignment(task_id)
        path = root / result_relative_path(expected_cell, expected_arm)
        result = _load_result(path)
        identity = {
            "task_id": task_id,
            "cell_id": expected_cell["cell_id"],
            "n_particles": expected_cell["n_particles"],
            "spatial_dims": expected_cell["spatial_dims"],
            "replicate": expected_cell["replicate"],
            "simulator_seed": expected_cell["simulator_seed"],
            "dataset_name": expected_cell["dataset_name"],
            "arm": expected_arm,
        }
        if any(result.get(key) != value for key, value in identity.items()):
            raise ValueError(f"{path} identity differs from its task assignment")
        if result.get("training_rows") != TRAIN_ROWS or result.get("test_rows") != TEST_ROWS:
            raise ValueError(f"{path} role counts drifted")
        metrics = result.get("metrics", {})
        if set(metrics) != set(METRICS) or not all(
            math.isfinite(float(metrics[name])) for name in METRICS
        ):
            raise ValueError(f"{path} metrics are invalid")
        commits.add(result["source_snapshot"]["git_commit"])
        grouped[(expected_cell["n_particles"], expected_cell["spatial_dims"])][expected_arm].append(
            result
        )

    configurations = {}
    all_pass = True
    for (n_particles, n_dims), arms in sorted(grouped.items()):
        if set(arms) != set(ARMS) or any(len(arms[arm]) != len(REPLICATES) for arm in ARMS):
            raise ValueError("summary does not contain three results per arm and configuration")
        for replicate in REPLICATES:
            hashes = {
                result["dataset_sha256"]
                for arm in ARMS
                for result in arms[arm]
                if result["replicate"] == replicate
            }
            if len(hashes) != 1:
                raise ValueError("paired arms used different dataset bytes")
        statistics = {}
        for arm in ARMS:
            statistics[arm] = {}
            for metric in METRICS:
                values = np.asarray([result["metrics"][metric] for result in arms[arm]])
                statistics[arm][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                }
        comparison = {
            "value_rmse_ratio": statistics["spark"]["value_rmse"]["mean"]
            / statistics["tera"]["value_rmse"]["mean"],
            "gradient_rmse_ratio": statistics["spark"]["gradient_rmse"]["mean"]
            / statistics["tera"]["gradient_rmse"]["mean"],
            "raw_nll_difference": statistics["spark"]["raw_nll"]["mean"]
            - statistics["tera"]["raw_nll"]["mean"],
        }
        passed = (
            comparison["value_rmse_ratio"] < 1.0
            and comparison["gradient_rmse_ratio"] < 1.0
            and comparison["raw_nll_difference"] < 0.0
        )
        all_pass = all_pass and passed
        configurations[f"n{n_particles}_d{n_dims}"] = {
            "n_particles": n_particles,
            "spatial_dims": n_dims,
            "state_dimension": 2 * n_particles * n_dims,
            "seed_count": len(REPLICATES),
            "metrics": statistics,
            "spark_vs_tera": comparison,
            "spark_lower_all_three_metrics": passed,
        }
    if len(commits) != 1:
        raise ValueError("results were produced by different source commits")
    return {
        "protocol": SUMMARY_PROTOCOL,
        "result_protocol": PROTOCOL,
        "complete": True,
        "task_count": len(TASK_IDS),
        "source_commit": next(iter(commits)),
        "decision_rule": (
            "SPARK three-seed mean value RMSE, gradient RMSE, and raw standardized value NLL "
            "must each be lower than TERA in every (n_particles, spatial_dims) configuration"
        ),
        "all_configurations_pass": all_pass,
        "timing_used_for_decision": False,
        "configurations": configurations,
    }


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(payload))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate-task")
    generate_parser.add_argument("--task-id", type=int, required=True)
    generate_parser.add_argument("--data-dir", type=Path, required=True)
    run_parser = commands.add_parser("run-task")
    run_parser.add_argument("--task-id", type=int, required=True)
    run_parser.add_argument("--data-dir", type=Path, required=True)
    run_parser.add_argument("--result-root", type=Path, required=True)
    summary_parser = commands.add_parser("summarize")
    summary_parser.add_argument("--result-root", type=Path, required=True)
    summary_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    thread_count = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    torch.set_num_threads(thread_count)
    if args.command == "generate-task":
        cell = generation_assignment(args.task_id)
        arrays = generate_nbody_sweep_dataset(
            n_particles=cell["n_particles"],
            n_dims=cell["spatial_dims"],
            seed=cell["simulator_seed"],
        )
        print(save_dataset(args.data_dir, arrays))
    elif args.command == "run-task":
        cell, arm = task_assignment(args.task_id)
        result = run_task(args.task_id, args.data_dir, snapshot=source_snapshot())
        path = args.result_root / result_relative_path(cell, arm)
        _write_new(path, result)
        print(path)
    else:
        summary = summarize(args.result_root)
        _write_new(args.out, summary)
        print(args.out)


if __name__ == "__main__":
    main()
