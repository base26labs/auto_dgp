"""Development-only PRISM-GP probe on the already-read v4 N-body corpus.

The paper arrays and released TERA fit are float32.  The formal v4 benchmark
casts prediction arithmetic to float64 and consequently used float64 epsilon
for ORBIT's default numerical-rank rule.  This probe changes exactly one
method choice: PRISM-GP uses source-float32 numerical rank, admits the m=30
conditional only below a fixed 16-direction budget, applies a label-free
posterior trust/variance guard, and differentiates only the selected branch.

The already-read v4 corpus is development data.  Results from this module are
not confirmatory and cannot revive or modify the v4 assessment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.f02_internal_models import FrozenTERAParameters, ScalarPrediction
from experiments.paper_nbody_benchmark import (
    ORBIT_CG_MAX_ITERATIONS,
    ORBIT_CG_TOLERANCE,
    PREDICTION_DTYPE,
    TASKS,
    _cast_split,
    _orbit_matmul_flops,
    _preconditioner_flops,
    _prediction_metrics,
    _sha256,
    _tera_resource_summary,
    load_paper_task_data,
)
from gp.orbit.budgeted import predict_budgeted_guarded_marginals

SCHEMA = "paper_nbody_prism_development_task_v1"
SOURCE_SCHEMA = "paper_nbody_benchmark_task_v4"
SOURCE_COMMIT = "076315efdeef4492897651515eaeeed95e8dd863"
DEVELOPMENT_TASKS = TASKS
SOURCE_RANK_EPSILON = torch.finfo(torch.float32).eps
MAXIMUM_DIRECTION_RANK = 16
TRUST_RADIUS_SIGMA = 0.025


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _parameters_from_source(source: dict[str, Any]) -> FrozenTERAParameters:
    record = source.get("learned_parameters")
    if not isinstance(record, dict):
        raise ValueError("source learned_parameters are missing")
    return FrozenTERAParameters(
        lengthscale=torch.tensor(record["lengthscale"], dtype=PREDICTION_DTYPE),
        outputscale=record["outputscale"],
        sigma_f=record["value_noise_variance"],
        sigma_g=record["gradient_noise_variance"],
        kernel=record["kernel"],
        gradient_noise_model=record["gradient_noise_model"],
    )


def _load_source(path: Path, task_index: int, data_sha256: str) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    task = DEVELOPMENT_TASKS[task_index]
    expected_task = {
        "task_index": task.task_index,
        "n_particles": task.n_particles,
        "dimension": task.dimension,
        "seed": task.seed,
    }
    if source.get("schema") != SOURCE_SCHEMA or source.get("status") != "complete":
        raise ValueError("source must be a complete paper N-body v4 task")
    if source.get("task") != expected_task:
        raise ValueError("source task does not match the requested development task")
    provenance = source.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("source provenance is missing")
    if provenance.get("git_commit") != SOURCE_COMMIT:
        raise ValueError("source commit is not the frozen v4 benchmark commit")
    if provenance.get("data_sha256") != data_sha256:
        raise ValueError("source and current dataset hashes differ")
    baseline = source.get("arms", {}).get("TERA-20")
    required_metrics = ("value_rmse", "value_nll", "gradient_rmse")
    if not isinstance(baseline, dict) or any(
        not math.isfinite(float(baseline.get(metric, math.nan))) for metric in required_metrics
    ):
        raise ValueError("source TERA-20 metrics are incomplete")
    _parameters_from_source(source)
    return source


def _build_flops(m: int, rank: int, dimension: int) -> int:
    return 8 * dimension * m * m + 12 * m**3 + 4 * rank**3


def _state_elements(m: int, rank: int, dimension: int) -> int:
    return 7 * m * m + 2 * m * rank + rank * rank + rank + 2 * m * dimension


def _primal_flops(m: int, rank: int, matvecs: int, preconditioners: int) -> int:
    return matvecs * _orbit_matmul_flops(m, rank) + preconditioners * _preconditioner_flops(m, rank)


def _resource_summary(details: Any, dimension: int) -> dict[str, Any]:
    records = []
    for index in range(details.mean.shape[0]):
        base_rank = int(details.base_ranks[index])
        expanded_rank = int(details.expanded_ranks[index])
        selected_m = int(details.selected_m[index])
        selected_rank = expanded_rank if selected_m == 30 else base_rank
        base_build = _build_flops(20, base_rank, dimension)
        expanded_build = _build_flops(30, expanded_rank, dimension)
        base_primal = _primal_flops(
            20,
            base_rank,
            int(details.base_operator_matvecs[index]),
            int(details.base_preconditioner_applications[index]),
        )
        expanded_primal = _primal_flops(
            30,
            expanded_rank,
            int(details.expanded_operator_matvecs[index]),
            int(details.expanded_preconditioner_applications[index]),
        )
        selected_adjoint = _primal_flops(
            selected_m,
            selected_rank,
            int(details.selected_adjoint_operator_matvecs[index]),
            int(details.selected_adjoint_preconditioner_applications[index]),
        )
        pullback = 4 * (
            _build_flops(selected_m, selected_rank, dimension)
            + _orbit_matmul_flops(selected_m, selected_rank)
        )
        records.append(
            {
                "state": 4
                * (
                    _state_elements(20, base_rank, dimension)
                    + _state_elements(30, expanded_rank, dimension)
                ),
                "flops": (
                    base_build
                    + expanded_build
                    + base_primal
                    + expanded_primal
                    + selected_adjoint
                    + pullback
                ),
            }
        )

    eligible = details.expanded_eligible
    expanded_residuals = details.expanded_relative_residuals[eligible]
    residuals = torch.cat(
        [
            details.base_relative_residuals,
            expanded_residuals,
            details.selected_adjoint_relative_residuals,
        ]
    )
    all_converged = bool(details.base_converged.all())
    all_converged = all_converged and bool(details.expanded_converged[eligible].all())
    all_converged = all_converged and bool(details.selected_adjoint_converged.all())
    if not all_converged or not bool(torch.isfinite(residuals).all()):
        raise RuntimeError("PRISM-GP contains a failed or invalid solve")
    if bool((residuals > ORBIT_CG_TOLERANCE).any()):
        raise RuntimeError("PRISM-GP fresh residual exceeds the registered tolerance")
    return {
        "schema": "prism_budgeted_guarded_value_gradient_proxy_v1",
        "state_accounting": "simultaneous_base_plus_expanded_safety_states",
        "flop_accounting": "both_builds_and_primals_plus_selected_adjoint_and_pullback",
        "counted_state_elements_maximum": max(item["state"] for item in records),
        "counted_flops_mean_per_target": float(np.mean([item["flops"] for item in records])),
        "counted_flops_maximum_per_target": max(item["flops"] for item in records),
        "base_rank_minimum": int(details.base_ranks.min()),
        "base_rank_maximum": int(details.base_ranks.max()),
        "expanded_rank_minimum": int(details.expanded_ranks.min()),
        "expanded_rank_maximum": int(details.expanded_ranks.max()),
        "expanded_eligible_count": int(details.expanded_eligible.sum()),
        "expanded_selected_count": int(details.use_expanded.sum()),
        "maximum_fresh_relative_residual": float(residuals.max()),
        "all_solves_converged": True,
        "autograd_tape_excludes_cg_iterations": True,
        "selected_adjoint_only": True,
    }


def run_task(
    repo_root: Path,
    task_index: int,
    *,
    source_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], Path]:
    if task_index < 0 or task_index >= len(DEVELOPMENT_TASKS):
        raise IndexError("development task index must be in [0, 11]")
    if _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("precision-rank development probe requires a clean repository")

    task = DEVELOPMENT_TASKS[task_index]
    prepared, data_path = load_paper_task_data(repo_root, task)
    data_sha256 = _sha256(data_path)
    source_path = source_root / f"task-{task.task_index:03d}.json"
    source = _load_source(source_path, task_index, data_sha256)
    parameters = _parameters_from_source(source)
    train = _cast_split(prepared.train, PREDICTION_DTYPE)
    test = _cast_split(prepared.test, PREDICTION_DTYPE)

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
    tera_resources = _tera_resource_summary(20)

    output_root.mkdir(parents=True, exist_ok=True)
    arrays_path = output_root / f"task-{task.task_index:03d}.npz"
    if arrays_path.exists():
        raise RuntimeError(f"refusing to overwrite development arrays: {arrays_path}")
    arrays = {
        "target_value": test.value.detach().cpu().numpy(),
        "target_gradient": test.gradient.detach().cpu().numpy(),
        "prism_mean": details.mean.detach().cpu().numpy(),
        "prism_latent_variance": details.variance.detach().cpu().numpy(),
        "prism_mean_gradient": details.mean_gradient.detach().cpu().numpy(),
        "prism_use_expanded": details.use_expanded.detach().cpu().numpy(),
        "prism_expanded_eligible": details.expanded_eligible.detach().cpu().numpy(),
        "prism_expanded_rank": details.expanded_ranks.detach().cpu().numpy(),
        "prism_normalized_mean_shift": details.normalized_mean_shift.detach().cpu().numpy(),
    }
    np.savez_compressed(arrays_path, **arrays)

    deltas = {
        metric: float(candidate_metrics[metric] - source["arms"]["TERA-20"][metric])
        for metric in ("value_rmse", "value_nll", "gradient_rmse")
    }
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "scope": "development_only_v4_corpus_already_read",
        "task": {
            "task_index": task.task_index,
            "n_particles": task.n_particles,
            "dimension": task.dimension,
            "seed": task.seed,
        },
        "candidate": {
            "name": "PRISM-GP-30/16",
            "description": (
                "precision-ranked iterative structured marginals with guarded m=30 "
                "expansion, a rank-16 eligibility budget, and one selected adjoint"
            ),
            "base_m": 20,
            "expanded_m": 30,
            "maximum_direction_rank": MAXIMUM_DIRECTION_RANK,
            "trust_radius_sigma": TRUST_RADIUS_SIGMA,
            "variance_guard": "expanded_variance_not_above_base_beyond_float64_roundoff",
            "rank_epsilon": SOURCE_RANK_EPSILON,
            "rank_epsilon_source": "torch.float32_input_arrays",
            "prediction_dtype": "float64",
            "cg_tolerance": ORBIT_CG_TOLERANCE,
            "metrics": candidate_metrics,
            "analytic_resources": candidate_resources,
            "delta_from_frozen_TERA_20": deltas,
            "resource_match": {
                "state": candidate_resources["counted_state_elements_maximum"]
                <= tera_resources["counted_value_gradient_state_elements_per_target"],
                "flops": candidate_resources["counted_flops_maximum_per_target"]
                <= tera_resources["counted_value_gradient_flops_per_target"],
            },
        },
        "controls": {
            "frozen_TERA_20": {
                metric: source["arms"]["TERA-20"][metric]
                for metric in ("value_rmse", "value_nll", "gradient_rmse")
            },
        },
        "artifacts": {
            "arrays_path": str(arrays_path),
            "arrays_sha256": _sha256(arrays_path),
            "source_task_path": str(source_path),
            "source_task_sha256": _sha256(source_path),
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
            "data_sha256": data_sha256,
            "source_v4_git_commit": SOURCE_COMMIT,
        },
    }
    result["candidate"]["resource_match"]["passes_both"] = all(
        result["candidate"]["resource_match"].values()
    )
    return result, arrays_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("runs/paper_nbody_v4"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/paper_nbody_prism_dev_v1"),
    )
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    repo_root = Path(__file__).resolve().parents[1]
    result, arrays_path = run_task(
        repo_root,
        args.task_index,
        source_root=args.source_root,
        output_root=args.output_root,
    )
    result_path = args.output_root / f"task-{args.task_index:03d}.json"
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with result_path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise RuntimeError(f"refusing to overwrite development result: {result_path}") from error
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
