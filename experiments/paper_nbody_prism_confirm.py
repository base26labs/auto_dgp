"""Independent paper-style N-body confirmation for PRISM-GP-30/16.

The method and decision rule are frozen before generating the seed-43 corpus.
Each task refits the released TERA model, evaluates TERA-20 and PRISM from the
same learned parameters, and reports value RMSE, observation-variance value
NLL, and gradient RMSE on the paper's complete 90/10 test split.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.f02_internal_models import (
    ScalarPrediction,
    fit_released_tera,
    freeze_tera_parameters,
    predict_released_tera_with_mean_gradient,
)
from experiments.paper_nbody_benchmark import (
    ORBIT_CG_MAX_ITERATIONS,
    ORBIT_CG_TOLERANCE,
    PAPER_GENERATOR_PROTOCOL,
    PAPER_GENERATOR_UPSTREAM_BLOB,
    PAPER_GENERATOR_UPSTREAM_COMMIT,
    PAPER_GENERATOR_UPSTREAM_REPOSITORY,
    PAPER_ROWS_AFTER_FILTER,
    PREDICTION_DTYPE,
    TASKS,
    TERA_BATCH_SIZE,
    TERA_LEARNING_RATE,
    TERA_PREDICT_M,
    TERA_TRAIN_EPOCHS,
    TERA_TRAIN_M,
    _cast_split,
    _git,
    _parameters_record,
    _prediction_metrics,
    _sha256,
    _tera_resource_summary,
    prepare_paper_arrays,
)
from experiments.paper_nbody_precision_rank_dev import (
    MAXIMUM_DIRECTION_RANK,
    SOURCE_RANK_EPSILON,
    TRUST_RADIUS_SIGMA,
    _resource_summary,
)
from gp.orbit.budgeted import predict_budgeted_guarded_marginals

SCHEMA = "paper_nbody_prism_confirmation_task_v1"
DATASET_GENERATION_SEED = 43
DATASET_DIRECTORY = "nbody_confirmatory_seed43"
CANDIDATE_NAME = "PRISM-GP-30/16"
VALUE_RMSE_NONINFERIORITY_MARGIN = 1e-4


def load_confirmation_data(repo_root: Path, task_index: int) -> tuple[Any, Path]:
    task = TASKS[task_index]
    path = repo_root / "data" / DATASET_DIRECTORY / f"nbody_n{task.n_particles}_d3.npz"
    if not path.is_file():
        raise FileNotFoundError(f"independent N-body dataset is missing: {path}")
    with np.load(path, allow_pickle=False) as record:
        required = {
            "X",
            "E",
            "F",
            "n_particles",
            "n_dims",
            "generator_protocol",
            "generator_upstream_repository",
            "generator_upstream_commit",
            "generator_upstream_get_nbody_blob",
            "generator_seed",
        }
        if not required.issubset(record.files):
            raise ValueError(f"dataset is missing keys: {sorted(required - set(record.files))}")
        expected = {
            "generator_protocol": PAPER_GENERATOR_PROTOCOL,
            "generator_upstream_repository": PAPER_GENERATOR_UPSTREAM_REPOSITORY,
            "generator_upstream_commit": PAPER_GENERATOR_UPSTREAM_COMMIT,
            "generator_upstream_get_nbody_blob": PAPER_GENERATOR_UPSTREAM_BLOB,
        }
        if int(record["n_particles"]) != task.n_particles or int(record["n_dims"]) != 3:
            raise ValueError("dataset particle/dimension metadata does not match the task")
        if int(record["generator_seed"]) != DATASET_GENERATION_SEED:
            raise ValueError("dataset is not the frozen independent seed-43 corpus")
        for key, value in expected.items():
            if str(record[key].item()) != value:
                raise ValueError(f"dataset {key} does not match the pinned paper generator")
        x = np.array(record["X"], copy=True)
        value = np.array(record["E"], copy=True)
        gradient = np.array(record["F"], copy=True)
    if x.shape[0] != PAPER_ROWS_AFTER_FILTER:
        raise ValueError(f"confirmation dataset must contain {PAPER_ROWS_AFTER_FILTER} rows")
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


def run_task(
    repo_root: Path,
    task_index: int,
    *,
    output_root: Path,
) -> tuple[dict[str, Any], Path]:
    if task_index < 0 or task_index >= len(TASKS):
        raise IndexError("confirmation task index must be in [0, 11]")
    if _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("independent confirmation requires a clean repository")

    task = TASKS[task_index]
    prepared, data_path = load_confirmation_data(repo_root, task_index)
    torch.manual_seed(task.seed)
    fit_started = time.perf_counter()
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
    fit_seconds = time.perf_counter() - fit_started
    parameters = freeze_tera_parameters(model)
    train = _cast_split(prepared.train, PREDICTION_DTYPE)
    test = _cast_split(prepared.test, PREDICTION_DTYPE)

    started = time.perf_counter()
    baseline, baseline_gradient = predict_released_tera_with_mean_gradient(
        train,
        test.X,
        parameters,
        m=TERA_PREDICT_M,
    )
    baseline_seconds = time.perf_counter() - started
    baseline_metrics = _prediction_metrics(
        baseline,
        baseline_gradient,
        test.value,
        test.gradient,
    )

    started = time.perf_counter()
    details = predict_budgeted_guarded_marginals(
        train.X,
        train.value,
        train.gradient,
        test.X,
        base_m=20,
        expanded_m=30,
        maximum_expanded_rank=MAXIMUM_DIRECTION_RANK,
        trust_radius_sigma=TRUST_RADIUS_SIGMA,
        rank_epsilon=SOURCE_RANK_EPSILON,
        lengthscale=parameters.lengthscale,
        outputscale=parameters.outputscale,
        value_noise_variance=parameters.sigma_f,
        gradient_noise_variance=parameters.sigma_g,
        kernel=parameters.kernel,
        gradient_noise_model=parameters.gradient_noise_model,
        cg_tolerance=ORBIT_CG_TOLERANCE,
        cg_max_iterations=ORBIT_CG_MAX_ITERATIONS,
        use_preconditioner=True,
    )
    candidate_seconds = time.perf_counter() - started
    candidate = ScalarPrediction(
        mean=details.mean,
        latent_variance=details.variance,
        observation_variance=details.variance + details.variance.new_tensor(parameters.sigma_f),
        details=details,
    )
    candidate_metrics = _prediction_metrics(
        candidate,
        details.mean_gradient,
        test.value,
        test.gradient,
    )
    candidate_resources = _resource_summary(details, task.dimension)
    baseline_resources = _tera_resource_summary(TERA_PREDICT_M)
    resource_match = {
        "state": candidate_resources["counted_state_elements_maximum"]
        <= baseline_resources["counted_value_gradient_state_elements_per_target"],
        "flops": candidate_resources["counted_flops_maximum_per_target"]
        <= baseline_resources["counted_value_gradient_flops_per_target"],
    }

    output_root.mkdir(parents=True, exist_ok=True)
    arrays_path = output_root / f"task-{task.task_index:03d}.npz"
    if arrays_path.exists():
        raise RuntimeError(f"refusing to overwrite confirmation arrays: {arrays_path}")
    np.savez_compressed(
        arrays_path,
        target_value=test.value.detach().cpu().numpy(),
        target_gradient=test.gradient.detach().cpu().numpy(),
        tera_mean=baseline.mean.detach().cpu().numpy(),
        tera_observation_variance=baseline.observation_variance.detach().cpu().numpy(),
        tera_mean_gradient=baseline_gradient.detach().cpu().numpy(),
        prism_mean=details.mean.detach().cpu().numpy(),
        prism_observation_variance=candidate.observation_variance.detach().cpu().numpy(),
        prism_mean_gradient=details.mean_gradient.detach().cpu().numpy(),
        prism_use_expanded=details.use_expanded.detach().cpu().numpy(),
        prism_expanded_eligible=details.expanded_eligible.detach().cpu().numpy(),
        prism_expanded_rank=details.expanded_ranks.detach().cpu().numpy(),
    )

    result = {
        "schema": SCHEMA,
        "status": "complete",
        "scope": "independent_confirmation_seed43",
        "task": {
            "task_index": task.task_index,
            "n_particles": task.n_particles,
            "dimension": task.dimension,
            "seed": task.seed,
        },
        "protocol": {
            "dataset_generation_seed": DATASET_GENERATION_SEED,
            "rows_after_gradient_filter": PAPER_ROWS_AFTER_FILTER,
            "train_fraction": 0.9,
            "normalization": prepared.normalization,
            "train_indices": list(prepared.train_indices),
            "test_indices": list(prepared.test_indices),
            "candidate": CANDIDATE_NAME,
            "base_m": 20,
            "expanded_m": 30,
            "maximum_direction_rank": MAXIMUM_DIRECTION_RANK,
            "trust_radius_sigma": TRUST_RADIUS_SIGMA,
            "rank_epsilon": SOURCE_RANK_EPSILON,
            "rank_epsilon_source": "torch.float32_input_arrays",
            "prediction_dtype": "float64",
            "cg_tolerance": ORBIT_CG_TOLERANCE,
            "value_nll_variance": "observation_variance",
            "value_rmse_noninferiority_margin": VALUE_RMSE_NONINFERIORITY_MARGIN,
            "strict_improvement_metrics": ["value_nll", "gradient_rmse"],
        },
        "learned_parameters": _parameters_record(parameters),
        "arms": {
            "TERA-20": {
                **baseline_metrics,
                "analytic_resources": baseline_resources,
                "prediction_seconds_descriptive_only": baseline_seconds,
            },
            CANDIDATE_NAME: {
                **candidate_metrics,
                "analytic_resources": candidate_resources,
                "prediction_seconds_descriptive_only": candidate_seconds,
            },
        },
        "candidate_minus_TERA_20": {
            metric: float(candidate_metrics[metric] - baseline_metrics[metric])
            for metric in ("value_rmse", "value_nll", "gradient_rmse")
        },
        "candidate_resource_match": {
            **resource_match,
            "passes_both": all(resource_match.values()),
            "state_ratio": candidate_resources["counted_state_elements_maximum"]
            / baseline_resources["counted_value_gradient_state_elements_per_target"],
            "maximum_flop_ratio": candidate_resources["counted_flops_maximum_per_target"]
            / baseline_resources["counted_value_gradient_flops_per_target"],
        },
        "fit_seconds_descriptive_only": fit_seconds,
        "training_history": [
            {key: float(value) for key, value in row.items()} for row in model.training_history
        ],
        "artifacts": {
            "arrays_path": str(arrays_path),
            "arrays_sha256": _sha256(arrays_path),
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
    return result, arrays_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/paper_nbody_prism_confirm_seed43_v1"),
    )
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    repo_root = Path(__file__).resolve().parents[1]
    result, arrays_path = run_task(
        repo_root,
        args.task_index,
        output_root=args.output_root,
    )
    result_path = args.output_root / f"task-{args.task_index:03d}.json"
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with result_path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise RuntimeError(f"refusing to overwrite confirmation result: {result_path}") from error
    print(
        json.dumps(
            {
                "status": "complete",
                "task_index": args.task_index,
                "result_path": str(result_path),
                "arrays_path": str(arrays_path),
            }
        )
    )


if __name__ == "__main__":
    main()
