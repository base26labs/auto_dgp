"""Small contract tests for the paper-aligned benchmark."""

import json
from pathlib import Path

import numpy as np

from experiments.paper_nbody_aggregate import aggregate_results, load_complete_results
from experiments.paper_nbody_benchmark import (
    PAPER_PARTICLES,
    PAPER_SEEDS,
    SCHEMA,
    TASKS,
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
                },
                "ORBIT-20": {
                    "value_rmse": 1.0,
                    "value_nll": 2.0,
                    "value_nll_variance": "observation_variance",
                    "gradient_rmse": 3.0,
                },
                "ORBIT-30": {
                    "value_rmse": 0.9,
                    "value_nll": 1.9,
                    "value_nll_variance": "observation_variance",
                    "gradient_rmse": 2.9,
                },
            },
            "candidate_resource_match": {"passes_both": True},
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
