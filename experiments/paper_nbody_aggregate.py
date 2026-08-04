"""Aggregate the exact 12-task paper-aligned N-body benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.paper_nbody_benchmark import SCHEMA, TASKS

AGGREGATE_SCHEMA = "paper_nbody_benchmark_aggregate_v1"
ARM_LABELS = ("TERA-20", "ORBIT-20", "ORBIT-30")


def load_complete_results(input_root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for expected in TASKS:
        path = input_root / f"task-{expected.task_index:03d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing paper benchmark result: {path}")
        result = json.loads(path.read_text())
        if result.get("schema") != SCHEMA or result.get("status") != "complete":
            raise ValueError(f"invalid or incomplete paper benchmark result: {path}")
        task = result.get("task")
        expected_task = {
            "task_index": expected.task_index,
            "n_particles": expected.n_particles,
            "dimension": expected.dimension,
            "seed": expected.seed,
        }
        if task != expected_task:
            raise ValueError(f"paper benchmark task identity mismatch: {path}")
        arms = result.get("arms")
        if not isinstance(arms, dict) or set(arms) != set(ARM_LABELS):
            raise ValueError(f"paper benchmark arms are missing or out of order: {path}")
        for label in ARM_LABELS:
            for metric in ("value_rmse", "value_nll"):
                value = arms[label].get(metric)
                if not isinstance(value, (int, float)) or not np.isfinite(value):
                    raise ValueError(f"invalid {label} {metric}: {path}")
        results.append(result)
    return results


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0))}


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(results) != len(TASKS):
        raise ValueError(f"expected exactly {len(TASKS)} paper benchmark results")
    by_index = {int(result["task"]["task_index"]): result for result in results}
    if set(by_index) != set(range(len(TASKS))):
        raise ValueError("paper benchmark result indices are incomplete or duplicated")

    datasets: dict[str, Any] = {}
    all_seed_joint_wins = True
    all_dataset_mean_joint_wins = True
    for n_particles in sorted({task.n_particles for task in TASKS}):
        selected = [by_index[task.task_index] for task in TASKS if task.n_particles == n_particles]
        arm_summary: dict[str, Any] = {}
        for label in ARM_LABELS:
            arm_summary[label] = {
                "value_rmse": _mean_std(
                    [float(result["arms"][label]["value_rmse"]) for result in selected]
                ),
                "value_nll": _mean_std(
                    [float(result["arms"][label]["value_nll"]) for result in selected]
                ),
            }

        paired_rmse = [
            float(result["arms"]["ORBIT-30"]["value_rmse"])
            - float(result["arms"]["TERA-20"]["value_rmse"])
            for result in selected
        ]
        paired_nll = [
            float(result["arms"]["ORBIT-30"]["value_nll"])
            - float(result["arms"]["TERA-20"]["value_nll"])
            for result in selected
        ]
        seed_joint_wins = [
            rmse_delta < 0.0 and nll_delta < 0.0
            for rmse_delta, nll_delta in zip(paired_rmse, paired_nll, strict=True)
        ]
        dataset_mean_joint_win = (
            arm_summary["ORBIT-30"]["value_rmse"]["mean"]
            < arm_summary["TERA-20"]["value_rmse"]["mean"]
            and arm_summary["ORBIT-30"]["value_nll"]["mean"]
            < arm_summary["TERA-20"]["value_nll"]["mean"]
        )
        all_seed_joint_wins = all_seed_joint_wins and all(seed_joint_wins)
        all_dataset_mean_joint_wins = all_dataset_mean_joint_wins and dataset_mean_joint_win
        datasets[str(n_particles)] = {
            "dimension": 6 * n_particles,
            "arms": arm_summary,
            "paired_ORBIT_30_minus_TERA_20": {
                "value_rmse": _mean_std(paired_rmse),
                "value_nll": _mean_std(paired_nll),
                "joint_win_by_seed": seed_joint_wins,
            },
            "candidate_mean_joint_win": dataset_mean_joint_win,
            "candidate_resource_match_all_seeds": all(
                bool(result["candidate_resource_match"]["passes_both"]) for result in selected
            ),
            "same_m_control_maximums": {
                "absolute_mean_difference": max(
                    float(result["same_m_control"]["maximum_absolute_mean_difference"])
                    for result in selected
                ),
                "absolute_latent_variance_difference": max(
                    float(result["same_m_control"]["maximum_absolute_latent_variance_difference"])
                    for result in selected
                ),
            },
        }

    resource_match = all(
        bool(result["candidate_resource_match"]["passes_both"]) for result in results
    )
    return {
        "schema": AGGREGATE_SCHEMA,
        "complete": True,
        "task_count": len(results),
        "datasets": datasets,
        "candidate_assessment": {
            "candidate": "ORBIT-30",
            "baseline": "TERA-20",
            "resource_match_all_tasks": resource_match,
            "lower_mean_rmse_and_nll_on_every_dataset": all_dataset_mean_joint_wins,
            "lower_rmse_and_nll_on_every_seed_task": all_seed_joint_wins,
            "beats_TERA_under_registered_rule": (
                resource_match and all_dataset_mean_joint_wins and all_seed_joint_wins
            ),
            "statistical_significance_claimed": False,
            "wall_clock_used_for_claim": False,
            "gradient_claimed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("runs/paper_nbody_v1"))
    parser.add_argument("--output", type=Path, default=Path("runs/paper_nbody_v1/aggregate.json"))
    args = parser.parse_args()
    aggregate = aggregate_results(load_complete_results(args.input_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(aggregate["candidate_assessment"], sort_keys=True))


if __name__ == "__main__":
    main()
