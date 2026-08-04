"""Aggregate the exact 12-task paper-aligned N-body benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.paper_nbody_benchmark import (
    ORBIT_EXPANSION_M,
    ORBIT_GUARD_LATENT_SIGMA,
    PAPER_GENERATOR_PROTOCOL,
    PAPER_GENERATOR_UPSTREAM_BLOB,
    PAPER_GENERATOR_UPSTREAM_COMMIT,
    PAPER_GENERATOR_UPSTREAM_REPOSITORY,
    PAPER_ROWS_AFTER_FILTER,
    SCHEMA,
    TASKS,
    TERA_PREDICT_M,
)

AGGREGATE_SCHEMA = "paper_nbody_benchmark_aggregate_v3"
ARM_LABELS = ("TERA-20", "ORBIT-20", "ORBIT-G30")


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def _validate_frozen_protocol(result: dict[str, Any], path: Path) -> None:
    expected_generator = {
        "protocol": PAPER_GENERATOR_PROTOCOL,
        "upstream_repository": PAPER_GENERATOR_UPSTREAM_REPOSITORY,
        "upstream_commit": PAPER_GENERATOR_UPSTREAM_COMMIT,
        "upstream_get_nbody_blob": PAPER_GENERATOR_UPSTREAM_BLOB,
    }
    paper_protocol = result.get("paper_protocol")
    if (
        not isinstance(paper_protocol, dict)
        or paper_protocol.get("generator") != expected_generator
    ):
        raise ValueError(f"paper generator protocol drift: {path}")

    model_protocol = result.get("model_protocol")
    expected_model_fields = {
        "candidate": "ORBIT-G30",
        "candidate_base_m": TERA_PREDICT_M,
        "candidate_expanded_m": ORBIT_EXPANSION_M,
        "candidate_guard_latent_sigma": ORBIT_GUARD_LATENT_SIGMA,
        "value_nll_variance": "observation_variance",
    }
    if not isinstance(model_protocol, dict) or any(
        model_protocol.get(key) != value for key, value in expected_model_fields.items()
    ):
        raise ValueError(f"paper candidate or NLL protocol drift: {path}")

    arms = result["arms"]
    guard = arms["ORBIT-G30"].get("guard")
    if (
        not isinstance(guard, dict)
        or guard.get("latent_sigma_threshold") != ORBIT_GUARD_LATENT_SIGMA
    ):
        raise ValueError(f"ORBIT-G30 guard protocol drift: {path}")
    expanded_count = guard.get("expanded_target_count")
    fallback_count = guard.get("fallback_target_count")
    if (
        type(expanded_count) is not int
        or type(fallback_count) is not int
        or expanded_count < 0
        or fallback_count < 0
        or expanded_count + fallback_count != PAPER_ROWS_AFTER_FILTER // 10
    ):
        raise ValueError(f"invalid ORBIT-G30 guard counts: {path}")

    tera_resources = arms["TERA-20"].get("analytic_resources")
    candidate_resources = arms["ORBIT-G30"].get("analytic_resources")
    if (
        not isinstance(tera_resources, dict)
        or tera_resources.get("schema") != "tera_dense_value_gradient_proxy_v2"
        or tera_resources.get("m") != TERA_PREDICT_M
        or tera_resources.get("value_gradient_safety_multiplier") != 4
    ):
        raise ValueError(f"invalid TERA value-gradient resource proxy: {path}")
    if (
        not isinstance(candidate_resources, dict)
        or candidate_resources.get("schema") != "orbit_guarded_expansion_proxy_v1"
        or candidate_resources.get("base_m") != TERA_PREDICT_M
        or candidate_resources.get("expanded_m") != ORBIT_EXPANSION_M
        or candidate_resources.get("guard_latent_sigma_threshold") != ORBIT_GUARD_LATENT_SIGMA
        or candidate_resources.get("expanded_target_count") != expanded_count
        or candidate_resources.get("fallback_target_count") != fallback_count
        or candidate_resources.get("state_accounting") != "sequential_component_maximum"
        or candidate_resources.get("flop_accounting") != "sum_of_both_component_proxies"
        or candidate_resources.get("all_primal_and_adjoint_solves_converged") is not True
    ):
        raise ValueError(f"invalid ORBIT-G30 resource proxy: {path}")

    tera_state = tera_resources.get("counted_value_gradient_state_elements_per_target")
    tera_flops = tera_resources.get("counted_value_gradient_flops_per_target")
    candidate_state = candidate_resources.get("counted_state_elements_maximum")
    candidate_flops = candidate_resources.get("counted_flops_maximum_per_target")
    if not all(
        _is_finite_number(value)
        for value in (tera_state, tera_flops, candidate_state, candidate_flops)
    ):
        raise ValueError(f"nonfinite paper resource proxy: {path}")
    expected_resource_match = {
        "state_proxy_within_TERA_20": candidate_state <= tera_state,
        "maximum_flop_proxy_within_TERA_20": candidate_flops <= tera_flops,
    }
    expected_resource_match["passes_both"] = all(expected_resource_match.values())
    if result.get("candidate_resource_match") != expected_resource_match:
        raise ValueError(f"candidate resource decision mismatch: {path}")

    raw_expansion = result.get("raw_ORBIT_30_diagnostic_not_an_assessment_arm")
    if not isinstance(raw_expansion, dict):
        raise ValueError(f"missing raw ORBIT-30 diagnostic: {path}")
    for metric in ("value_rmse", "value_nll", "gradient_rmse"):
        if not _is_finite_number(raw_expansion.get(metric)):
            raise ValueError(f"invalid raw ORBIT-30 {metric}: {path}")
    if raw_expansion.get("value_nll_variance") != "observation_variance":
        raise ValueError(f"invalid raw ORBIT-30 NLL variance semantics: {path}")


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
            for metric in ("value_rmse", "value_nll", "gradient_rmse"):
                value = arms[label].get(metric)
                if not _is_finite_number(value):
                    raise ValueError(f"invalid {label} {metric}: {path}")
            if arms[label].get("value_nll_variance") != "observation_variance":
                raise ValueError(f"invalid {label} value NLL variance semantics: {path}")
        _validate_frozen_protocol(result, path)
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
                "gradient_rmse": _mean_std(
                    [float(result["arms"][label]["gradient_rmse"]) for result in selected]
                ),
            }

        paired_rmse = [
            float(result["arms"]["ORBIT-G30"]["value_rmse"])
            - float(result["arms"]["TERA-20"]["value_rmse"])
            for result in selected
        ]
        paired_nll = [
            float(result["arms"]["ORBIT-G30"]["value_nll"])
            - float(result["arms"]["TERA-20"]["value_nll"])
            for result in selected
        ]
        paired_gradient_rmse = [
            float(result["arms"]["ORBIT-G30"]["gradient_rmse"])
            - float(result["arms"]["TERA-20"]["gradient_rmse"])
            for result in selected
        ]
        seed_joint_wins = [
            value_rmse_delta < 0.0 and nll_delta < 0.0 and gradient_rmse_delta < 0.0
            for value_rmse_delta, nll_delta, gradient_rmse_delta in zip(
                paired_rmse,
                paired_nll,
                paired_gradient_rmse,
                strict=True,
            )
        ]
        dataset_mean_joint_win = (
            arm_summary["ORBIT-G30"]["value_rmse"]["mean"]
            < arm_summary["TERA-20"]["value_rmse"]["mean"]
            and arm_summary["ORBIT-G30"]["value_nll"]["mean"]
            < arm_summary["TERA-20"]["value_nll"]["mean"]
            and arm_summary["ORBIT-G30"]["gradient_rmse"]["mean"]
            < arm_summary["TERA-20"]["gradient_rmse"]["mean"]
        )
        all_seed_joint_wins = all_seed_joint_wins and all(seed_joint_wins)
        all_dataset_mean_joint_wins = all_dataset_mean_joint_wins and dataset_mean_joint_win
        datasets[str(n_particles)] = {
            "dimension": 6 * n_particles,
            "arms": arm_summary,
            "paired_ORBIT_G30_minus_TERA_20": {
                "value_rmse": _mean_std(paired_rmse),
                "value_nll": _mean_std(paired_nll),
                "gradient_rmse": _mean_std(paired_gradient_rmse),
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
                "absolute_mean_gradient_difference": max(
                    float(result["same_m_control"]["maximum_absolute_mean_gradient_difference"])
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
            "candidate": "ORBIT-G30",
            "baseline": "TERA-20",
            "resource_match_all_tasks": resource_match,
            "lower_mean_value_rmse_value_nll_and_gradient_rmse_on_every_dataset": (
                all_dataset_mean_joint_wins
            ),
            "lower_value_rmse_value_nll_and_gradient_rmse_on_every_seed_task": (
                all_seed_joint_wins
            ),
            "beats_TERA_under_registered_rule": (
                resource_match and all_dataset_mean_joint_wins and all_seed_joint_wins
            ),
            "statistical_significance_claimed": False,
            "wall_clock_used_for_claim": False,
            "gradient_claimed": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("runs/paper_nbody_v3"))
    parser.add_argument("--output", type=Path, default=Path("runs/paper_nbody_v3/aggregate.json"))
    args = parser.parse_args()
    aggregate = aggregate_results(load_complete_results(args.input_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(aggregate["candidate_assessment"], sort_keys=True))


if __name__ == "__main__":
    main()
