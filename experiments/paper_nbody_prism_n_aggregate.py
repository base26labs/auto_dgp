"""Aggregate the 12-task secondary PRISM N-scaling benchmark."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.paper_nbody_benchmark import _sha256
from experiments.paper_nbody_prism_confirm import (
    CANDIDATE_NAME,
    DATASET_GENERATION_SEED,
    VALUE_RMSE_NONINFERIORITY_MARGIN,
)
from experiments.paper_nbody_prism_n_benchmark import (
    FIXED_DIMENSION,
    FIXED_N_PARTICLES,
    N_SCALING_TASKS,
    PAPER_SPLIT_SEEDS,
    SCHEMA,
    TRAINING_SIZES,
)

AGGREGATE_SCHEMA = "paper_nbody_prism_n_scaling_aggregate_v1"
METRICS = ("value_rmse", "value_nll", "gradient_rmse")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_results(input_root: Path) -> list[dict[str, Any]]:
    results = []
    commits: set[str] = set()
    trees: set[str] = set()
    data_hashes: set[str] = set()
    for expected in N_SCALING_TASKS:
        path = input_root / f"task-{expected.task_index:03d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing N-scaling result: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        task = result.get("task", {})
        if (
            result.get("schema") != SCHEMA
            or result.get("status") != "complete"
            or result.get("scope") != "secondary_n_scaling_on_already_evaluated_seed43_corpus"
            or task.get("task_index") != expected.task_index
            or task.get("n_train") != expected.n_train
            or task.get("n_test") != 950
            or task.get("n_particles") != FIXED_N_PARTICLES
            or task.get("dimension") != FIXED_DIMENSION
            or task.get("seed") != expected.seed
        ):
            raise ValueError(f"invalid N-scaling identity: {path}")
        protocol = result.get("protocol", {})
        if (
            protocol.get("dataset_generation_seed") != DATASET_GENERATION_SEED
            or protocol.get("nested_training_prefix") is not True
            or protocol.get("complete_fixed_test_split") is not True
            or protocol.get("candidate") != CANDIDATE_NAME
            or protocol.get("value_nll_variance") != "observation_variance"
            or protocol.get("value_rmse_noninferiority_margin") != VALUE_RMSE_NONINFERIORITY_MARGIN
            or protocol.get("strict_improvement_metrics") != ["value_nll", "gradient_rmse"]
            or protocol.get("confirmatory_claim") is not False
        ):
            raise ValueError(f"N-scaling protocol drift: {path}")
        arms = result.get("arms", {})
        if set(arms) != {"TERA-20", CANDIDATE_NAME}:
            raise ValueError(f"N-scaling arms are incomplete: {path}")
        for arm in arms.values():
            if any(not _finite(arm.get(metric)) for metric in METRICS):
                raise ValueError(f"nonfinite N-scaling metric: {path}")
            if arm.get("value_nll_variance") != "observation_variance":
                raise ValueError(f"N-scaling NLL semantics drift: {path}")
        resources = arms[CANDIDATE_NAME].get("analytic_resources", {})
        if (
            resources.get("all_solves_converged") is not True
            or not _finite(resources.get("maximum_fresh_relative_residual"))
            or resources["maximum_fresh_relative_residual"] > 1e-10
            or result.get("candidate_resource_match", {}).get("passes_both") is not True
        ):
            raise ValueError(f"N-scaling solve/resource gate failed: {path}")
        split = result.get("split", {})
        if (
            len(split.get("train_source_indices", ())) != expected.n_train
            or len(split.get("test_source_indices", ())) != 950
        ):
            raise ValueError(f"N-scaling split sizes drift: {path}")
        arrays_path = input_root / f"task-{expected.task_index:03d}.npz"
        if not arrays_path.is_file():
            raise FileNotFoundError(f"missing N-scaling arrays: {arrays_path}")
        if _sha256(arrays_path) != result.get("artifacts", {}).get("arrays_sha256"):
            raise ValueError(f"N-scaling arrays hash mismatch: {arrays_path}")
        provenance = result.get("provenance", {})
        commits.add(str(provenance.get("git_commit")))
        trees.add(str(provenance.get("git_tree")))
        data_hashes.add(str(provenance.get("data_sha256")))
        results.append(result)

    if len(commits) != 1 or len(trees) != 1 or len(data_hashes) != 1:
        raise ValueError("N-scaling tasks must bind one implementation and one D=60 corpus")
    for seed in PAPER_SPLIT_SEEDS:
        selected = sorted(
            (result for result in results if result["task"]["seed"] == seed),
            key=lambda result: result["task"]["n_train"],
        )
        if [result["task"]["n_train"] for result in selected] != list(TRAINING_SIZES):
            raise ValueError("N-scaling training-size grid is incomplete")
        reference_test = selected[0]["split"]["test_source_indices"]
        for smaller, larger in zip(selected, selected[1:], strict=True):
            if (
                larger["split"]["train_source_indices"][: smaller["task"]["n_train"]]
                != smaller["split"]["train_source_indices"]
            ):
                raise ValueError("N-scaling training subsets are not nested prefixes")
            if larger["split"]["test_source_indices"] != reference_test:
                raise ValueError("N-scaling test split changed within a seed")
    return results


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0))}


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(results) != len(N_SCALING_TASKS):
        raise ValueError(f"expected exactly {len(N_SCALING_TASKS)} N-scaling results")
    by_index = {result["task"]["task_index"]: result for result in results}
    if set(by_index) != set(range(len(N_SCALING_TASKS))):
        raise ValueError("N-scaling task indices are incomplete or duplicated")

    sizes: dict[str, Any] = {}
    all_size_gates = True
    for n_train in TRAINING_SIZES:
        selected = [result for result in results if result["task"]["n_train"] == n_train]
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
        all_size_gates = all_size_gates and gates["passes"]
        sizes[str(n_train)] = {
            "n_train": n_train,
            "n_test": 950,
            "arms": arms,
            "paired_PRISM_minus_TERA": deltas,
            "gates": gates,
        }

    first = results[0]
    return {
        "schema": AGGREGATE_SCHEMA,
        "complete": True,
        "scope": "secondary_n_scaling_no_new_confirmatory_claim",
        "task_count": len(results),
        "fixed_dimension": FIXED_DIMENSION,
        "fixed_n_particles": FIXED_N_PARTICLES,
        "sizes": sizes,
        "secondary_assessment": {
            "candidate": CANDIDATE_NAME,
            "baseline": "TERA-20",
            "passes_pareto_rule_at_every_training_size": all_size_gates,
            "confirmatory_claim": False,
            "statistical_significance_claimed": False,
            "wall_clock_used": False,
        },
        "provenance": {
            "git_commit": first["provenance"]["git_commit"],
            "git_tree": first["provenance"]["git_tree"],
            "data_sha256": first["provenance"]["data_sha256"],
            "dataset_generation_seed": DATASET_GENERATION_SEED,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("runs/paper_nbody_prism_n_scaling_seed43_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/paper_nbody_prism_n_scaling_seed43_v1/aggregate.json"),
    )
    args = parser.parse_args()
    aggregate = aggregate_results(load_results(args.input_root))
    payload = json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise RuntimeError(f"refusing to overwrite N-scaling aggregate: {args.output}") from error
    print(json.dumps(aggregate["secondary_assessment"], sort_keys=True))


if __name__ == "__main__":
    main()
