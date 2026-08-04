"""Small contract tests for the frozen PRISM paper confirmation."""

from pathlib import Path

from experiments.paper_nbody_benchmark import TASKS
from experiments.paper_nbody_prism_aggregate import aggregate_results
from experiments.paper_nbody_prism_confirm import (
    CANDIDATE_NAME,
    DATASET_GENERATION_SEED,
    VALUE_RMSE_NONINFERIORITY_MARGIN,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _result(task):
    return {
        "task": {
            "task_index": task.task_index,
            "n_particles": task.n_particles,
            "dimension": task.dimension,
            "seed": task.seed,
        },
        "arms": {
            "TERA-20": {"value_rmse": 1.0, "value_nll": 2.0, "gradient_rmse": 3.0},
            CANDIDATE_NAME: {
                "value_rmse": 1.0 + VALUE_RMSE_NONINFERIORITY_MARGIN / 2,
                "value_nll": 1.99,
                "gradient_rmse": 2.9,
                "analytic_resources": {"all_solves_converged": True},
            },
        },
        "candidate_resource_match": {"passes_both": True},
        "provenance": {"git_commit": "a" * 40, "git_tree": "b" * 40},
    }


def test_frozen_aggregate_requires_value_noninferiority_and_strict_nll_gradient_wins() -> None:
    aggregate = aggregate_results([_result(task) for task in TASKS])
    assert aggregate["candidate_assessment"]["passes_frozen_paper_style_rule"] is True
    assert all(row["gates"]["passes"] for row in aggregate["datasets"].values())


def test_confirmation_scripts_are_shared_8cpu_and_seed43() -> None:
    scripts = (
        REPO_ROOT / "cluster" / "paper_nbody_prism_confirm_data.sbatch",
        REPO_ROOT / "cluster" / "paper_nbody_prism_confirm.sbatch",
    )
    for path in scripts:
        source = path.read_text(encoding="utf-8")
        assert "#SBATCH --cpus-per-task=8" in source
        assert "#SBATCH --exclusive" not in source
        assert "#SBATCH --oversubscribe" not in source
        assert "#SBATCH --gres" not in source
        assert 'export CUDA_VISIBLE_DEVICES=""' in source
    data_source = scripts[0].read_text(encoding="utf-8")
    assert f"--seed {DATASET_GENERATION_SEED}" in data_source
    assert "--output-file" in data_source
