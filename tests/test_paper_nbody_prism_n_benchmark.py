"""Small contracts for the secondary PRISM N-scaling benchmark."""

from pathlib import Path

import torch

from experiments.f02_internal_models import TensorConfirmatorySplit
from experiments.paper_nbody_prism_confirm import CANDIDATE_NAME
from experiments.paper_nbody_prism_n_aggregate import aggregate_results
from experiments.paper_nbody_prism_n_benchmark import (
    FIXED_DIMENSION,
    N_SCALING_TASKS,
    PAPER_SPLIT_SEEDS,
    TRAINING_SIZES,
    _prefix,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_n_scaling_grid_and_nested_prefix_are_fixed() -> None:
    assert [(task.seed, task.n_train) for task in N_SCALING_TASKS] == [
        (seed, n_train) for seed in PAPER_SPLIT_SEEDS for n_train in TRAINING_SIZES
    ]
    split = TensorConfirmatorySplit(
        name="train",
        source_indices=torch.arange(6),
        X=torch.arange(6 * 2, dtype=torch.float64).reshape(6, 2),
        value=torch.arange(6, dtype=torch.float64),
        gradient=torch.arange(6 * 2, dtype=torch.float64).reshape(6, 2),
        trajectory_id=torch.arange(6),
        time_index=torch.arange(6),
        time_value=torch.arange(6, dtype=torch.float64),
    )
    small = _prefix(split, 2)
    large = _prefix(split, 5)
    assert torch.equal(small.source_indices, large.source_indices[:2])
    assert torch.equal(small.X, large.X[:2])


def _result(task):
    return {
        "task": {
            "task_index": task.task_index,
            "n_train": task.n_train,
            "seed": task.seed,
        },
        "arms": {
            "TERA-20": {"value_rmse": 1.0, "value_nll": 2.0, "gradient_rmse": 3.0},
            CANDIDATE_NAME: {
                "value_rmse": 1.00001,
                "value_nll": 1.99,
                "gradient_rmse": 2.9,
                "analytic_resources": {"all_solves_converged": True},
            },
        },
        "candidate_resource_match": {"passes_both": True},
        "provenance": {
            "git_commit": "a" * 40,
            "git_tree": "b" * 40,
            "data_sha256": "c" * 64,
        },
    }


def test_n_scaling_aggregate_reports_every_training_size() -> None:
    aggregate = aggregate_results([_result(task) for task in N_SCALING_TASKS])
    assert aggregate["fixed_dimension"] == FIXED_DIMENSION
    assert set(aggregate["sizes"]) == {str(value) for value in TRAINING_SIZES}
    assert aggregate["secondary_assessment"]["passes_pareto_rule_at_every_training_size"]
    assert aggregate["secondary_assessment"]["confirmatory_claim"] is False


def test_n_scaling_slurm_is_shared_8cpu() -> None:
    source = (REPO_ROOT / "cluster" / "paper_nbody_prism_n_benchmark.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --cpus-per-task=8" in source
    assert "#SBATCH --array=0-11%3" in source
    assert "#SBATCH --exclusive" not in source
    assert "#SBATCH --oversubscribe" not in source
    assert "#SBATCH --gres" not in source
    assert 'export CUDA_VISIBLE_DEVICES=""' in source
