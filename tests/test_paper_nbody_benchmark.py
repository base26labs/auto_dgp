"""Small contract tests for the paper-aligned benchmark."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.f02_internal_models import ScalarPrediction
from experiments.paper_nbody_aggregate import aggregate_results, load_complete_results
from experiments.paper_nbody_benchmark import (
    PAPER_GENERATOR_PROTOCOL,
    PAPER_GENERATOR_UPSTREAM_BLOB,
    PAPER_GENERATOR_UPSTREAM_COMMIT,
    PAPER_GENERATOR_UPSTREAM_REPOSITORY,
    PAPER_PARTICLES,
    PAPER_ROWS_AFTER_FILTER,
    PAPER_SEEDS,
    SCHEMA,
    TASKS,
    _guarded_expansion,
    prepare_paper_arrays,
    task_for_index,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_paper_grid_and_split_are_fixed_and_deterministic() -> None:
    assert len(TASKS) == 12
    assert [(task.seed, task.n_particles) for task in TASKS] == [
        (seed, particles) for seed in PAPER_SEEDS for particles in PAPER_PARTICLES
    ]
    assert task_for_index(11).dimension == 60

    x = np.arange(20 * 24, dtype=np.float64).reshape(20, 24)
    value = np.linspace(-2.0, 3.0, 20)
    gradient = np.arange(1, 20 * 24 + 1, dtype=np.float64).reshape(20, 24) / 100.0
    first = prepare_paper_arrays(x, value, gradient, n_particles=4, seed=6535)
    second = prepare_paper_arrays(x, value, gradient, n_particles=4, seed=6535)

    assert first.train_indices == second.train_indices
    assert first.test_indices == second.test_indices
    assert len(first.train_indices) == 18
    assert len(first.test_indices) == 2
    assert set(first.train_indices).isdisjoint(first.test_indices)
    assert set(first.train_indices) | set(first.test_indices) == set(range(20))
    assert first.normalization["ordering"].startswith("normalize_complete_filtered_dataset")


def test_slurm_entry_is_shared_8cpu_without_gpu_or_exclusive() -> None:
    scripts = {
        "paper_nbody_data.sbatch": "#SBATCH --array=0-3%2",
        "paper_nbody_benchmark.sbatch": "#SBATCH --array=0-11%3",
    }
    for filename, array_directive in scripts.items():
        source = (REPO_ROOT / "cluster" / filename).read_text()
        assert "#SBATCH --cpus-per-task=8" in source
        assert array_directive in source
        assert "#SBATCH --oversubscribe" not in source
        assert "#SBATCH --exclusive" not in source
        assert "#SBATCH --gres" not in source
        assert 'export CUDA_VISIBLE_DEVICES=""' in source

    generator_source = (REPO_ROOT / "data" / "get_nbody.py").read_text()
    pinned_values = {
        "GENERATOR_PROTOCOL": PAPER_GENERATOR_PROTOCOL,
        "UPSTREAM_REPOSITORY": PAPER_GENERATOR_UPSTREAM_REPOSITORY,
        "UPSTREAM_COMMIT": PAPER_GENERATOR_UPSTREAM_COMMIT,
        "UPSTREAM_GET_NBODY_BLOB": PAPER_GENERATOR_UPSTREAM_BLOB,
    }
    for name, value in pinned_values.items():
        assert f'{name} = "{value}"' in generator_source
    data_source = (REPO_ROOT / "cluster" / "paper_nbody_data.sbatch").read_text()
    for argument in (
        "--n_samples 10000",
        "--n_trajectories 100",
        "--steps_per_trajectory 200",
        "--dt 0.01",
        "--G 1.0",
        "--softening 0.1",
        "--percentile_filter 95",
        "--seed 42",
    ):
        assert argument in data_source


def test_paper_aggregate_reports_the_registered_joint_win(tmp_path: Path) -> None:
    for task in TASKS:
        result = {
            "schema": SCHEMA,
            "status": "complete",
            "task": {
                "task_index": task.task_index,
                "n_particles": task.n_particles,
                "dimension": task.dimension,
                "seed": task.seed,
            },
            "arms": {
                "TERA-20": {
                    "value_rmse": 1.0,
                    "value_nll": 2.0,
                    "value_nll_variance": "observation_variance",
                    "gradient_rmse": 3.0,
                    "analytic_resources": {
                        "schema": "tera_dense_value_gradient_proxy_v2",
                        "m": 20,
                        "value_gradient_safety_multiplier": 4,
                        "counted_value_gradient_state_elements_per_target": 100.0,
                        "counted_value_gradient_flops_per_target": 100.0,
                    },
                },
                "ORBIT-20": {
                    "value_rmse": 1.0,
                    "value_nll": 2.0,
                    "value_nll_variance": "observation_variance",
                    "gradient_rmse": 3.0,
                },
                "ORBIT-G30": {
                    "value_rmse": 0.9,
                    "value_nll": 1.9,
                    "value_nll_variance": "observation_variance",
                    "gradient_rmse": 2.9,
                    "guard": {
                        "latent_sigma_threshold": 0.02,
                        "expanded_target_count": 900,
                        "fallback_target_count": PAPER_ROWS_AFTER_FILTER // 10 - 900,
                    },
                    "analytic_resources": {
                        "schema": "orbit_guarded_expansion_proxy_v1",
                        "base_m": 20,
                        "expanded_m": 30,
                        "guard_latent_sigma_threshold": 0.02,
                        "expanded_target_count": 900,
                        "fallback_target_count": PAPER_ROWS_AFTER_FILTER // 10 - 900,
                        "state_accounting": "sequential_component_maximum",
                        "flop_accounting": "sum_of_both_component_proxies",
                        "all_primal_and_adjoint_solves_converged": True,
                        "counted_state_elements_maximum": 50.0,
                        "counted_flops_maximum_per_target": 50.0,
                    },
                },
            },
            "paper_protocol": {
                "generator": {
                    "protocol": PAPER_GENERATOR_PROTOCOL,
                    "upstream_repository": PAPER_GENERATOR_UPSTREAM_REPOSITORY,
                    "upstream_commit": PAPER_GENERATOR_UPSTREAM_COMMIT,
                    "upstream_get_nbody_blob": PAPER_GENERATOR_UPSTREAM_BLOB,
                }
            },
            "model_protocol": {
                "candidate": "ORBIT-G30",
                "candidate_base_m": 20,
                "candidate_expanded_m": 30,
                "candidate_guard_latent_sigma": 0.02,
                "value_nll_variance": "observation_variance",
            },
            "raw_ORBIT_30_diagnostic_not_an_assessment_arm": {
                "value_rmse": 1.1,
                "value_nll": 2.1,
                "value_nll_variance": "observation_variance",
                "gradient_rmse": 2.8,
            },
            "candidate_resource_match": {
                "state_proxy_within_TERA_20": True,
                "maximum_flop_proxy_within_TERA_20": True,
                "passes_both": True,
            },
            "same_m_control": {
                "maximum_absolute_mean_difference": 1e-12,
                "maximum_absolute_latent_variance_difference": 1e-12,
                "maximum_absolute_mean_gradient_difference": 1e-10,
            },
        }
        (tmp_path / f"task-{task.task_index:03d}.json").write_text(
            json.dumps(result, sort_keys=True)
        )

    aggregate = aggregate_results(load_complete_results(tmp_path))
    assert aggregate["candidate_assessment"]["beats_TERA_under_registered_rule"] is True
    assert aggregate["task_count"] == 12

    first_path = tmp_path / "task-000.json"
    drifted = json.loads(first_path.read_text())
    drifted["arms"]["ORBIT-G30"]["guard"]["latent_sigma_threshold"] = 0.03
    first_path.write_text(json.dumps(drifted, sort_keys=True))
    with pytest.raises(ValueError, match="guard protocol drift"):
        load_complete_results(tmp_path)


def test_guarded_expansion_falls_back_on_large_posterior_disagreement() -> None:
    variance = torch.tensor([0.01, 0.01], dtype=torch.float64)
    base = ScalarPrediction(torch.zeros(2), variance, variance + 0.1)
    expanded = ScalarPrediction(torch.tensor([0.001, 1.0]), variance / 2, variance / 2 + 0.1)
    base_gradient = torch.zeros(2, 3)
    expanded_gradient = torch.ones(2, 3)

    candidate, gradient, use_expanded, disagreement = _guarded_expansion(
        base,
        base_gradient,
        expanded,
        expanded_gradient,
    )

    assert use_expanded.tolist() == [True, False]
    torch.testing.assert_close(candidate.mean, torch.tensor([0.001, 0.0]))
    torch.testing.assert_close(gradient, torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))
    torch.testing.assert_close(disagreement, torch.tensor([0.01, 10.0], dtype=torch.float64))
