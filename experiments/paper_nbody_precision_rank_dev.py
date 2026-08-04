"""Development-only precision-aware ORBIT probe on the v4 N-body corpus.

The paper arrays and released TERA fit are float32.  The formal v4 benchmark
casts prediction arithmetic to float64 and consequently used float64 epsilon
for ORBIT's default numerical-rank rule.  This probe changes exactly one
method choice: ORBIT-PA30 uses the source float32 epsilon while retaining
float64 prediction arithmetic and the registered m=30, tolerance, and solver.

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

from experiments.f02_internal_models import FrozenTERAParameters, predict_orbit
from experiments.paper_nbody_benchmark import (
    ORBIT_CG_MAX_ITERATIONS,
    ORBIT_CG_TOLERANCE,
    PREDICTION_DTYPE,
    TASKS,
    _cast_split,
    _orbit_resource_summary,
    _prediction_metrics,
    _sha256,
    _tera_resource_summary,
    load_paper_task_data,
)

SCHEMA = "paper_nbody_precision_rank_development_task_v1"
SOURCE_SCHEMA = "paper_nbody_benchmark_task_v4"
SOURCE_COMMIT = "076315efdeef4492897651515eaeeed95e8dd863"
DEVELOPMENT_TASKS = TASKS[:4]
SOURCE_RANK_EPSILON = torch.finfo(torch.float32).eps


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


def _tensor_arrays(prefix: str, prediction: Any) -> dict[str, np.ndarray]:
    details = prediction.details
    if details is None or details.mean_gradient is None:
        raise RuntimeError("ORBIT prediction is missing mean-gradient diagnostics")
    return {
        f"{prefix}_mean": prediction.mean.detach().cpu().numpy(),
        f"{prefix}_latent_variance": prediction.latent_variance.detach().cpu().numpy(),
        f"{prefix}_mean_gradient": details.mean_gradient.detach().cpu().numpy(),
        f"{prefix}_rank": details.ranks.detach().cpu().numpy(),
        f"{prefix}_primal_iterations": details.iterations.detach().cpu().numpy(),
        f"{prefix}_adjoint_iterations": details.adjoint_iterations.detach().cpu().numpy(),
    }


def run_task(
    repo_root: Path,
    task_index: int,
    *,
    source_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], Path]:
    if task_index < 0 or task_index >= len(DEVELOPMENT_TASKS):
        raise IndexError("development task index must be in [0, 3]")
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

    base = predict_orbit(
        train,
        test.X,
        parameters,
        m=20,
        cg_tolerance=ORBIT_CG_TOLERANCE,
        cg_max_iterations=ORBIT_CG_MAX_ITERATIONS,
        use_preconditioner=True,
        include_mean_gradient=True,
    )
    candidate = predict_orbit(
        train,
        test.X,
        parameters,
        m=30,
        rank_epsilon=SOURCE_RANK_EPSILON,
        cg_tolerance=ORBIT_CG_TOLERANCE,
        cg_max_iterations=ORBIT_CG_MAX_ITERATIONS,
        use_preconditioner=True,
        include_mean_gradient=True,
    )
    if base.details is None or base.details.mean_gradient is None:
        raise RuntimeError("ORBIT-20 prediction is missing its mean gradient")
    if candidate.details is None or candidate.details.mean_gradient is None:
        raise RuntimeError("ORBIT-PA30 prediction is missing its mean gradient")

    base_metrics = _prediction_metrics(
        base,
        base.details.mean_gradient,
        test.value,
        test.gradient,
    )
    candidate_metrics = _prediction_metrics(
        candidate,
        candidate.details.mean_gradient,
        test.value,
        test.gradient,
    )
    base_resources = _orbit_resource_summary(base, 20, task.dimension)
    candidate_resources = _orbit_resource_summary(candidate, 30, task.dimension)
    tera_resources = _tera_resource_summary(20)

    output_root.mkdir(parents=True, exist_ok=True)
    arrays_path = output_root / f"task-{task.task_index:03d}.npz"
    if arrays_path.exists():
        raise RuntimeError(f"refusing to overwrite development arrays: {arrays_path}")
    arrays = {
        "target_value": test.value.detach().cpu().numpy(),
        "target_gradient": test.gradient.detach().cpu().numpy(),
        **_tensor_arrays("orbit20", base),
        **_tensor_arrays("orbit_pa30", candidate),
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
            "name": "ORBIT-PA30",
            "description": "m=30 with source-float32 numerical-rank epsilon and float64 solves",
            "m": 30,
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
            "rerun_ORBIT_20": {
                "metrics": base_metrics,
                "analytic_resources": base_resources,
            },
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
        default=Path("runs/paper_nbody_pa30_dev_v1"),
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
