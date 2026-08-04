"""Aggregate the frozen 12-task independent PRISM-GP confirmation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.paper_nbody_benchmark import TASKS, _sha256
from experiments.paper_nbody_prism_confirm import (
    CANDIDATE_NAME,
    DATASET_GENERATION_SEED,
    SCHEMA,
    VALUE_RMSE_NONINFERIORITY_MARGIN,
)

AGGREGATE_SCHEMA = "paper_nbody_prism_confirmation_aggregate_v1"
METRICS = ("value_rmse", "value_nll", "gradient_rmse")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_results(input_root: Path) -> list[dict[str, Any]]:
    results = []
    commits: set[str] = set()
    trees: set[str] = set()
    data_hashes: set[str] = set()
    for task in TASKS:
        path = input_root / f"task-{task.task_index:03d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing confirmation result: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        expected_task = {
            "task_index": task.task_index,
            "n_particles": task.n_particles,
            "dimension": task.dimension,
            "seed": task.seed,
        }
        if (
            result.get("schema") != SCHEMA
            or result.get("status") != "complete"
            or result.get("scope") != "independent_confirmation_seed43"
            or result.get("task") != expected_task
        ):
            raise ValueError(f"invalid confirmation identity: {path}")
        protocol = result.get("protocol", {})
        if (
            protocol.get("dataset_generation_seed") != DATASET_GENERATION_SEED
            or protocol.get("candidate") != CANDIDATE_NAME
            or protocol.get("value_nll_variance") != "observation_variance"
            or protocol.get("value_rmse_noninferiority_margin") != VALUE_RMSE_NONINFERIORITY_MARGIN
            or protocol.get("strict_improvement_metrics") != ["value_nll", "gradient_rmse"]
        ):
            raise ValueError(f"confirmation protocol drift: {path}")
        arms = result.get("arms", {})
        if set(arms) != {"TERA-20", CANDIDATE_NAME}:
            raise ValueError(f"confirmation arms are incomplete: {path}")
        for arm in arms.values():
            if any(not _finite(arm.get(metric)) for metric in METRICS):
                raise ValueError(f"nonfinite confirmation metric: {path}")
            if arm.get("value_nll_variance") != "observation_variance":
                raise ValueError(f"NLL variance semantics drift: {path}")
        resources = arms[CANDIDATE_NAME].get("analytic_resources", {})
        if (
            resources.get("all_solves_converged") is not True
            or not _finite(resources.get("maximum_fresh_relative_residual"))
            or resources["maximum_fresh_relative_residual"] > 1e-10
            or result.get("candidate_resource_match", {}).get("passes_both") is not True
        ):
            raise ValueError(f"confirmation solve/resource gate failed: {path}")
        provenance = result.get("provenance", {})
        commits.add(str(provenance.get("git_commit")))
        trees.add(str(provenance.get("git_tree")))
        data_hashes.add(str(provenance.get("data_sha256")))
        arrays_path = input_root / f"task-{task.task_index:03d}.npz"
        if not arrays_path.is_file():
            raise FileNotFoundError(f"missing confirmation arrays: {arrays_path}")
        if _sha256(arrays_path) != result.get("artifacts", {}).get("arrays_sha256"):
            raise ValueError(f"confirmation arrays hash mismatch: {arrays_path}")
        results.append(result)
    if len(commits) != 1 or len(trees) != 1:
        raise ValueError("confirmation tasks must use one committed implementation")
    if len(data_hashes) != 4:
        raise ValueError("confirmation must bind four distinct particle-count corpora")
    return results


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0))}


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(results) != len(TASKS):
        raise ValueError(f"expected exactly {len(TASKS)} confirmation results")
    by_index = {result["task"]["task_index"]: result for result in results}
    if set(by_index) != set(range(len(TASKS))):
        raise ValueError("confirmation task indices are incomplete or duplicated")

    datasets: dict[str, Any] = {}
    all_dataset_gates = True
    for n_particles in (4, 6, 8, 10):
        selected = [by_index[task.task_index] for task in TASKS if task.n_particles == n_particles]
        arms = {
            label: {
                metric: _mean_std([float(result["arms"][label][metric]) for result in selected])
                for metric in METRICS
            }
            for label in ("TERA-20", CANDIDATE_NAME)
        }
        deltas = {
            metric: _mean_std(
                [
                    float(result["arms"][CANDIDATE_NAME][metric])
                    - float(result["arms"]["TERA-20"][metric])
                    for result in selected
                ]
            )
            for metric in METRICS
        }
        gates = {
            "value_rmse_noninferior": deltas["value_rmse"]["mean"]
            <= VALUE_RMSE_NONINFERIORITY_MARGIN,
            "value_nll_strictly_lower": deltas["value_nll"]["mean"] < 0.0,
            "gradient_rmse_strictly_lower": deltas["gradient_rmse"]["mean"] < 0.0,
            "resource_match_all_seeds": all(
                result["candidate_resource_match"]["passes_both"] for result in selected
            ),
            "all_solves_converged": all(
                result["arms"][CANDIDATE_NAME]["analytic_resources"]["all_solves_converged"]
                for result in selected
            ),
        }
        gates["passes"] = all(gates.values())
        all_dataset_gates = all_dataset_gates and gates["passes"]
        datasets[str(n_particles)] = {
            "dimension": 6 * n_particles,
            "arms": arms,
            "paired_PRISM_minus_TERA": deltas,
            "gates": gates,
        }

    first = results[0]
    return {
        "schema": AGGREGATE_SCHEMA,
        "complete": True,
        "task_count": len(results),
        "datasets": datasets,
        "decision_rule": {
            "aggregation": "population mean and standard deviation over three paper split seeds",
            "value_rmse": f"candidate mean delta <= {VALUE_RMSE_NONINFERIORITY_MARGIN}",
            "value_nll": "candidate mean delta < 0",
            "gradient_rmse": "candidate mean delta < 0",
            "value_nll_variance": "observation_variance",
            "resources": "state and maximum counted-operation proxies within TERA-20 on every task",
            "wall_clock_used": False,
        },
        "candidate_assessment": {
            "candidate": CANDIDATE_NAME,
            "baseline": "TERA-20",
            "passes_frozen_paper_style_rule": all_dataset_gates,
            "statistical_significance_claimed": False,
        },
        "provenance": {
            "git_commit": first["provenance"]["git_commit"],
            "git_tree": first["provenance"]["git_tree"],
            "dataset_generation_seed": DATASET_GENERATION_SEED,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("runs/paper_nbody_prism_confirm_seed43_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/paper_nbody_prism_confirm_seed43_v1/aggregate.json"),
    )
    args = parser.parse_args()
    aggregate = aggregate_results(load_results(args.input_root))
    payload = json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise RuntimeError(
            f"refusing to overwrite confirmation aggregate: {args.output}"
        ) from error
    print(json.dumps(aggregate["candidate_assessment"], sort_keys=True))


if __name__ == "__main__":
    main()
