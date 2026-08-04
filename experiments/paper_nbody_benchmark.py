"""Paper-aligned N-body benchmark for released TERA and ORBIT.

The data generation, preprocessing, 90/10 split, and three random seeds follow
the toy N-body experiment in Huang (2026).  TERA retains its released native
training configuration.  ORBIT consumes the exact same fitted kernel state:

* TERA-20 is the released dense local baseline;
* ORBIT-20 is the same-neighbour numerical control; and
* ORBIT-30 is the preregistered resource-expansion hypothesis.

The benchmark evaluates the complete paper test split.  Wall-clock values are
descriptive only because execution is on shared CPU nodes.  Analytic operation
and state proxies are reported separately.  ORBIT currently exposes a scalar
conditional, so full-gradient RMSE is explicitly unsupported rather than
silently replaced by a different quantity.

Usage:
    python -m experiments.paper_nbody_benchmark --task-index 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.f02_internal_models import (
    FrozenTERAParameters,
    ScalarPrediction,
    TensorConfirmatorySplit,
    fit_released_tera,
    freeze_tera_parameters,
    predict_orbit,
    predict_released_tera,
)
from gp.exact import gaussian_nll
from gp.metrics import rmse

SCHEMA = "paper_nbody_benchmark_task_v1"
PAPER_REFERENCE = "https://arxiv.org/abs/2505.09134"
PAPER_PARTICLES = (4, 6, 8, 10)
PAPER_SEEDS = (6535, 8830, 92357)
PAPER_ROWS_AFTER_FILTER = 9500
TRAIN_FRACTION = 0.9

TERA_TRAIN_M = 20
TERA_PREDICT_M = 20
TERA_TRAIN_EPOCHS = 1
TERA_BATCH_SIZE = 256
TERA_LEARNING_RATE = 0.01
ORBIT_CANDIDATE_M = 30
ORBIT_CG_TOLERANCE = 1e-10
ORBIT_CG_MAX_ITERATIONS = 4096
PREDICTION_DTYPE = torch.float64


@dataclass(frozen=True, slots=True)
class PaperBenchmarkTask:
    task_index: int
    n_particles: int
    dimension: int
    seed: int


@dataclass(frozen=True, slots=True)
class PreparedPaperSplit:
    train: TensorConfirmatorySplit
    test: TensorConfirmatorySplit
    normalization: dict[str, Any]
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


TASKS = tuple(
    PaperBenchmarkTask(
        task_index=task_index,
        n_particles=n_particles,
        dimension=6 * n_particles,
        seed=seed,
    )
    for task_index, (seed, n_particles) in enumerate(
        (seed, n_particles) for seed in PAPER_SEEDS for n_particles in PAPER_PARTICLES
    )
)


def task_for_index(task_index: int) -> PaperBenchmarkTask:
    if isinstance(task_index, bool) or not isinstance(task_index, int):
        raise TypeError("task_index must be an integer")
    if task_index < 0 or task_index >= len(TASKS):
        raise IndexError(f"task_index must be in [0, {len(TASKS) - 1}]")
    return TASKS[task_index]


def _split_tensor(
    *,
    name: str,
    indices: torch.Tensor,
    x: torch.Tensor,
    value: torch.Tensor,
    gradient: torch.Tensor,
) -> TensorConfirmatorySplit:
    count = int(indices.numel())
    return TensorConfirmatorySplit(
        name=name,
        source_indices=indices.clone(),
        X=x[indices].contiguous(),
        value=value[indices].contiguous(),
        gradient=gradient[indices].contiguous(),
        trajectory_id=torch.full((count,), -1, dtype=torch.long),
        time_index=torch.full((count,), -1, dtype=torch.long),
        time_value=torch.full((count,), float("nan"), dtype=x.dtype),
    )


def prepare_paper_arrays(
    x_raw: np.ndarray,
    value_raw: np.ndarray,
    gradient_raw: np.ndarray,
    *,
    n_particles: int,
    seed: int,
) -> PreparedPaperSplit:
    """Apply the authors' released N-body preprocessing before the split."""

    x = torch.as_tensor(np.asarray(x_raw), dtype=torch.float32).clone()
    value = torch.as_tensor(np.asarray(value_raw), dtype=torch.float32).reshape(-1).clone()
    gradient = torch.as_tensor(np.asarray(gradient_raw), dtype=torch.float32).clone()
    expected_dimension = 6 * n_particles
    if x.ndim != 2 or x.shape[1] != expected_dimension or x.shape[0] < 2:
        raise ValueError(f"X must have shape (N, {expected_dimension}) with N >= 2")
    if value.shape != (x.shape[0],) or gradient.shape != x.shape:
        raise ValueError("E and F shapes must match X")
    if not bool(
        torch.isfinite(x).all() and torch.isfinite(value).all() and torch.isfinite(gradient).all()
    ):
        raise ValueError("paper benchmark arrays must be finite")

    minimum = x.min(dim=0).values
    ranges = x.max(dim=0).values - minimum + 1e-8
    if not bool(torch.isfinite(ranges).all()) or bool((ranges <= 0.0).any()):
        raise ValueError("paper input ranges must be finite and positive")
    x = (x - minimum) / ranges

    value_mean = value.mean()
    shared_scale = torch.cat(((value - value_mean).unsqueeze(1), gradient), dim=1).std()
    if not bool(torch.isfinite(shared_scale)) or float(shared_scale) <= 0.0:
        raise ValueError("paper energy/gradient scale must be finite and positive")
    value = (value - value_mean) / shared_scale
    gradient = gradient * ranges / shared_scale

    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(x.shape[0], generator=generator)
    train_count = int(x.shape[0] * TRAIN_FRACTION)
    train_indices = permutation[:train_count]
    test_indices = permutation[train_count:]
    if train_indices.numel() == 0 or test_indices.numel() == 0:
        raise ValueError("paper split must contain nonempty train and test sets")

    return PreparedPaperSplit(
        train=_split_tensor(
            name=f"paper-nbody-{n_particles}-train",
            indices=train_indices,
            x=x,
            value=value,
            gradient=gradient,
        ),
        test=_split_tensor(
            name=f"paper-nbody-{n_particles}-test",
            indices=test_indices,
            x=x,
            value=value,
            gradient=gradient,
        ),
        normalization={
            "ordering": "normalize_complete_filtered_dataset_then_seeded_random_split",
            "input_minimum": [float(item) for item in minimum],
            "input_range_with_1e-8_floor": [float(item) for item in ranges],
            "value_mean": float(value_mean),
            "shared_value_gradient_scale": float(shared_scale),
        },
        train_indices=tuple(int(item) for item in train_indices),
        test_indices=tuple(int(item) for item in test_indices),
    )


def load_paper_task_data(
    repo_root: Path, task: PaperBenchmarkTask
) -> tuple[PreparedPaperSplit, Path]:
    path = repo_root / "data" / "nbody" / f"nbody_n{task.n_particles}_d3.npz"
    if not path.is_file():
        raise FileNotFoundError(f"paper N-body dataset is missing: {path}")
    with np.load(path, allow_pickle=False) as record:
        required = {"X", "E", "F", "n_particles", "n_dims"}
        if not required.issubset(record.files):
            raise ValueError(f"dataset is missing keys: {sorted(required - set(record.files))}")
        if int(record["n_particles"]) != task.n_particles or int(record["n_dims"]) != 3:
            raise ValueError("dataset particle/dimension metadata does not match the task")
        x = np.array(record["X"], copy=True)
        value = np.array(record["E"], copy=True)
        gradient = np.array(record["F"], copy=True)
    if x.shape[0] != PAPER_ROWS_AFTER_FILTER:
        raise ValueError(
            f"paper dataset must contain {PAPER_ROWS_AFTER_FILTER} filtered rows; got {x.shape[0]}"
        )
    return (
        prepare_paper_arrays(
            x,
            value,
            gradient,
            n_particles=task.n_particles,
            seed=task.seed,
        ),
        path,
    )


def _cast_split(split: TensorConfirmatorySplit, dtype: torch.dtype) -> TensorConfirmatorySplit:
    return TensorConfirmatorySplit(
        name=split.name,
        source_indices=split.source_indices.clone(),
        X=split.X.to(dtype=dtype),
        value=split.value.to(dtype=dtype),
        gradient=split.gradient.to(dtype=dtype),
        trajectory_id=split.trajectory_id.clone(),
        time_index=split.time_index.clone(),
        time_value=split.time_value.to(dtype=dtype),
    )


def _prediction_metrics(prediction: ScalarPrediction, target: torch.Tensor) -> dict[str, Any]:
    if not bool(torch.isfinite(prediction.mean).all()):
        raise RuntimeError("prediction mean contains nonfinite values")
    if not bool(torch.isfinite(prediction.latent_variance).all()) or bool(
        (prediction.latent_variance <= 0.0).any()
    ):
        raise RuntimeError("prediction latent variance must be finite and positive")
    if not bool(torch.isfinite(prediction.observation_variance).all()) or bool(
        (prediction.observation_variance <= 0.0).any()
    ):
        raise RuntimeError("prediction observation variance must be finite and positive")
    return {
        "value_rmse": rmse(prediction.mean, target),
        "value_nll": gaussian_nll(target, prediction.mean, prediction.observation_variance),
        "gradient_rmse": None,
        "gradient_status": "unsupported_by_orbit_scalar_conditional_v1",
        "minimum_latent_variance": float(prediction.latent_variance.min()),
        "maximum_latent_variance": float(prediction.latent_variance.max()),
    }


def _tera_resource_summary(m: int) -> dict[str, Any]:
    return {
        "schema": "tera_dense_local_v1",
        "m": m,
        "explicit_reduced_covariance_elements_per_target": m**4,
        "reduced_cholesky_leading_flops_per_target": (m**6) / 3.0,
    }


def _orbit_matmul_flops(m: int, rank: int) -> int:
    return 10 * m * m * rank + 2 * m * rank * rank + 2 * m * m


def _preconditioner_flops(m: int, rank: int) -> int:
    return 4 * m * m * rank + 4 * m * rank * rank


def _orbit_resource_summary(
    prediction: ScalarPrediction,
    m: int,
    dimension: int,
) -> dict[str, Any]:
    details = prediction.details
    if details is None:
        raise RuntimeError("ORBIT prediction is missing solver diagnostics")
    ranks = [int(item) for item in details.ranks.detach().cpu()]
    matvecs = [int(item) for item in details.operator_matvecs.detach().cpu()]
    preconditioners = [int(item) for item in details.preconditioner_applications.detach().cpu()]
    residuals = [float(item) for item in details.relative_residuals.detach().cpu()]
    converged = [bool(item) for item in details.converged.detach().cpu()]
    if not all(converged):
        raise RuntimeError("ORBIT did not converge for every paper test target")
    if not all(math.isfinite(item) and item <= ORBIT_CG_TOLERANCE for item in residuals):
        raise RuntimeError("ORBIT fresh residual exceeds the registered tolerance")

    solve_flops = [
        matvec_count * _orbit_matmul_flops(m, rank)
        + preconditioner_count * _preconditioner_flops(m, rank)
        for rank, matvec_count, preconditioner_count in zip(
            ranks,
            matvecs,
            preconditioners,
            strict=True,
        )
    ]
    build_flops = [8 * dimension * m * m + 12 * m**3 + 4 * rank**3 for rank in ranks]
    counted_flops = [build + solve for build, solve in zip(build_flops, solve_flops, strict=True)]
    state_elements = [
        7 * m * m + 2 * m * rank + rank * rank + rank + 2 * m * dimension for rank in ranks
    ]
    return {
        "schema": "orbit_structured_proxy_v1",
        "m": m,
        "rank_minimum": min(ranks),
        "rank_maximum": max(ranks),
        "iterations_mean": float(details.iterations.double().mean()),
        "iterations_maximum": int(details.iterations.max()),
        "maximum_fresh_relative_residual": max(residuals),
        "all_converged": True,
        "counted_state_elements_maximum": max(state_elements),
        "build_flops_proxy_maximum_per_target": max(build_flops),
        "solve_flops_mean_per_target": float(np.mean(solve_flops)),
        "solve_flops_maximum_per_target": max(solve_flops),
        "counted_flops_mean_per_target": float(np.mean(counted_flops)),
        "counted_flops_maximum_per_target": max(counted_flops),
    }


def _parameters_record(parameters: FrozenTERAParameters) -> dict[str, Any]:
    return {
        "lengthscale": [float(item) for item in parameters.lengthscale.detach().cpu()],
        "outputscale": parameters.outputscale,
        "value_noise_variance": parameters.sigma_f,
        "gradient_noise_variance": parameters.sigma_g,
        "kernel": parameters.kernel,
        "gradient_noise_model": parameters.gradient_noise_model,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_task(repo_root: Path, task: PaperBenchmarkTask) -> dict[str, Any]:
    if _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("paper benchmark requires a clean repository")
    prepared, data_path = load_paper_task_data(repo_root, task)
    torch.manual_seed(task.seed)

    fit_start = time.perf_counter()
    model = fit_released_tera(
        prepared.train,
        training_m=TERA_TRAIN_M,
        train_epochs=TERA_TRAIN_EPOCHS,
        kernel="rbf",
        outputscale=1.0,
        sigma_f=1e-3,
        sigma_g=1e-3,
        lengthscale=1.0,
        lengthscale_init="median",
        use_ard=False,
        seed=task.seed,
        batch_size=TERA_BATCH_SIZE,
        lr=TERA_LEARNING_RATE,
        weight_decay=0.0,
    )
    fit_seconds = time.perf_counter() - fit_start
    parameters = freeze_tera_parameters(model)

    train = _cast_split(prepared.train, PREDICTION_DTYPE)
    test = _cast_split(prepared.test, PREDICTION_DTYPE)

    prediction_records: dict[str, tuple[ScalarPrediction, float]] = {}
    for label, predictor in (
        (
            "TERA-20",
            lambda: predict_released_tera(
                train,
                test.X,
                parameters,
                m=TERA_PREDICT_M,
            ),
        ),
        (
            "ORBIT-20",
            lambda: predict_orbit(
                train,
                test.X,
                parameters,
                m=TERA_PREDICT_M,
                cg_tolerance=ORBIT_CG_TOLERANCE,
                cg_max_iterations=ORBIT_CG_MAX_ITERATIONS,
                use_preconditioner=True,
            ),
        ),
        (
            "ORBIT-30",
            lambda: predict_orbit(
                train,
                test.X,
                parameters,
                m=ORBIT_CANDIDATE_M,
                cg_tolerance=ORBIT_CG_TOLERANCE,
                cg_max_iterations=ORBIT_CG_MAX_ITERATIONS,
                use_preconditioner=True,
            ),
        ),
    ):
        started = time.perf_counter()
        prediction = predictor()
        prediction_records[label] = (prediction, time.perf_counter() - started)

    tera_prediction = prediction_records["TERA-20"][0]
    orbit_same = prediction_records["ORBIT-20"][0]
    tera_resources = _tera_resource_summary(TERA_PREDICT_M)

    arms: dict[str, Any] = {}
    for label, (prediction, seconds) in prediction_records.items():
        arm = _prediction_metrics(prediction, test.value)
        arm["prediction_seconds_descriptive_only"] = seconds
        if label == "TERA-20":
            arm["analytic_resources"] = tera_resources
        else:
            arm["analytic_resources"] = _orbit_resource_summary(
                prediction,
                TERA_PREDICT_M if label == "ORBIT-20" else ORBIT_CANDIDATE_M,
                task.dimension,
            )
        arms[label] = arm

    candidate_resources = arms["ORBIT-30"]["analytic_resources"]
    resource_match = {
        "state_proxy_within_TERA_20": (
            candidate_resources["counted_state_elements_maximum"]
            <= tera_resources["explicit_reduced_covariance_elements_per_target"]
        ),
        "maximum_flop_proxy_within_TERA_20": (
            candidate_resources["counted_flops_maximum_per_target"]
            <= tera_resources["reduced_cholesky_leading_flops_per_target"]
        ),
    }

    return {
        "schema": SCHEMA,
        "status": "complete",
        "paper_reference": PAPER_REFERENCE,
        "task": {
            "task_index": task.task_index,
            "n_particles": task.n_particles,
            "dimension": task.dimension,
            "seed": task.seed,
        },
        "paper_protocol": {
            "rows_after_gradient_filter": PAPER_ROWS_AFTER_FILTER,
            "train_fraction": TRAIN_FRACTION,
            "train_rows": train.X.shape[0],
            "test_rows": test.X.shape[0],
            "normalization": prepared.normalization,
            "train_indices": list(prepared.train_indices),
            "test_indices": list(prepared.test_indices),
        },
        "model_protocol": {
            "fit_owner": "released_TERA",
            "training_m": TERA_TRAIN_M,
            "training_epochs": TERA_TRAIN_EPOCHS,
            "batch_size": TERA_BATCH_SIZE,
            "learning_rate": TERA_LEARNING_RATE,
            "prediction_dtype": str(PREDICTION_DTYPE).removeprefix("torch."),
            "orbit_cg_tolerance": ORBIT_CG_TOLERANCE,
            "orbit_cg_max_iterations": ORBIT_CG_MAX_ITERATIONS,
            "candidate_hypothesis": "ORBIT-30 improves RMSE and NLL over TERA-20 under both analytic resource proxies",
        },
        "learned_parameters": _parameters_record(parameters),
        "fit_seconds_descriptive_only": fit_seconds,
        "training_history": [
            {key: float(value) for key, value in row.items()} for row in model.training_history
        ],
        "arms": arms,
        "same_m_control": {
            "maximum_absolute_mean_difference": float(
                (tera_prediction.mean - orbit_same.mean).abs().max()
            ),
            "maximum_absolute_latent_variance_difference": float(
                (tera_prediction.latent_variance - orbit_same.latent_variance).abs().max()
            ),
        },
        "candidate_resource_match": {
            **resource_match,
            "passes_both": all(resource_match.values()),
        },
        "runtime": {
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "wall_clock_is_performance_evidence": False,
        },
        "provenance": {
            "git_commit": _git(repo_root, "rev-parse", "HEAD"),
            "git_tree": _git(repo_root, "rev-parse", "HEAD^{tree}"),
            "data_path": str(data_path.relative_to(repo_root)),
            "data_sha256": _sha256(data_path),
        },
    }


def write_result(output_root: Path, task: PaperBenchmarkTask, result: dict[str, Any]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"task-{task.task_index:03d}.json"
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise RuntimeError(f"refusing to overwrite benchmark result: {path}") from error
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/paper_nbody_v1"))
    args = parser.parse_args()

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    task = task_for_index(args.task_index)
    repo_root = Path(__file__).resolve().parents[1]
    result = run_task(repo_root, task)
    path = write_result(args.output_root, task, result)
    print(json.dumps({"status": "complete", "task_index": task.task_index, "path": str(path)}))


if __name__ == "__main__":
    main()
