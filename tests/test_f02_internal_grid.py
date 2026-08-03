from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from cluster.f02_internal_grid import (
    DEVELOPMENT_REPLICAS,
    N_DIMS,
    OPTIMIZER_SELECTION_TASKS,
    PARTICLE_COUNTS,
    PILOT_TASK_INDEX,
    SEEDS,
    TASK_COUNT,
    UPDATE_BUDGETS,
    task_for_index,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GRID_SCRIPT = REPO_ROOT / "cluster" / "f02_internal_grid.py"
SBATCH_SCRIPT = REPO_ROOT / "cluster" / "f02_internal_optimizer.sbatch"
SUBMIT_SCRIPT = REPO_ROOT / "cluster" / "submit_f02_internal_optimizer.sh"
CATALOG_SHA256 = "2dee429bdaf50cc78cb40ba8b038f7e4731bb07127927c55607b528c1db66942"


def test_optimizer_selection_grid_is_exact_cartesian_product() -> None:
    expected = [
        (replica, particles, N_DIMS, updates, seed)
        for replica in DEVELOPMENT_REPLICAS
        for particles in PARTICLE_COUNTS
        for updates in UPDATE_BUDGETS
        for seed in SEEDS
    ]
    observed = [
        (task.replica, task.n_particles, task.n_dims, task.train_steps, task.seed)
        for task in OPTIMIZER_SELECTION_TASKS
    ]

    assert DEVELOPMENT_REPLICAS == (0, 1, 2)
    assert PARTICLE_COUNTS == (2, 4, 6, 8, 10)
    assert UPDATE_BUDGETS == (20, 50, 100)
    assert SEEDS == (11, 29, 47)
    assert N_DIMS == 3
    assert TASK_COUNT == 135 == len(expected)
    assert observed == expected
    assert len(set(observed)) == TASK_COUNT
    assert [task.task_index for task in OPTIMIZER_SELECTION_TASKS] == list(range(TASK_COUNT))


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (0, (0, 2, 20, 11)),
        (1, (0, 2, 20, 29)),
        (2, (0, 2, 20, 47)),
        (3, (0, 2, 50, 11)),
        (8, (0, 2, 100, 47)),
        (9, (0, 4, 20, 11)),
        (44, (0, 10, 100, 47)),
        (45, (1, 2, 20, 11)),
        (89, (1, 10, 100, 47)),
        (90, (2, 2, 20, 11)),
        (134, (2, 10, 100, 47)),
    ],
)
def test_array_index_boundaries_are_stable(
    index: int,
    expected: tuple[int, int, int, int],
) -> None:
    task = task_for_index(index)
    assert (task.replica, task.n_particles, task.train_steps, task.seed) == expected


def test_pilot_and_dataset_stems_are_exact() -> None:
    pilot = task_for_index(PILOT_TASK_INDEX)

    assert PILOT_TASK_INDEX == 0
    assert pilot.as_record() == {
        "task_index": 0,
        "replica": 0,
        "n_particles": 2,
        "n_dims": 3,
        "train_steps": 20,
        "seed": 11,
        "dataset_stem": "nbody_fixedmass_n2_d3_replica0",
    }
    assert {task.dataset_stem for task in OPTIMIZER_SELECTION_TASKS} == {
        f"nbody_fixedmass_n{particles}_d3_replica{replica}"
        for replica in DEVELOPMENT_REPLICAS
        for particles in PARTICLE_COUNTS
    }


def test_task_records_are_immutable_and_out_of_range_indices_fail() -> None:
    with pytest.raises(FrozenInstanceError):
        task_for_index(0).seed = 999  # type: ignore[misc]
    with pytest.raises(IndexError):
        task_for_index(-1)
    with pytest.raises(IndexError):
        task_for_index(TASK_COUNT)


def test_grid_cli_has_machine_readable_and_shell_safe_forms(tmp_path: Path) -> None:
    count = subprocess.run(
        [sys.executable, str(GRID_SCRIPT), "--count"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = subprocess.run(
        [sys.executable, str(GRID_SCRIPT), "--task-index", "134"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = subprocess.run(
        [
            sys.executable,
            str(GRID_SCRIPT),
            "--task-index",
            "0",
            "--format",
            "lines",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert count.stdout.strip() == "135"
    assert json.loads(payload.stdout) == task_for_index(134).as_record()
    assert lines.stdout.splitlines() == [
        "0",
        "0",
        "2",
        "3",
        "20",
        "11",
        "nbody_fixedmass_n2_d3_replica0",
    ]


def test_grid_cli_rejects_invalid_or_ambiguous_requests() -> None:
    for arguments in (
        ("--task-index", "-1"),
        ("--task-index", "135"),
        ("--count", "--format", "lines"),
        ("--count", "--task-index", "0"),
    ):
        process = subprocess.run(
            [sys.executable, str(GRID_SCRIPT), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        assert process.returncode != 0


def test_slurm_and_submit_scripts_are_syntactically_valid() -> None:
    for path in (SBATCH_SCRIPT, SUBMIT_SCRIPT):
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_slurm_harness_locks_scientific_and_audit_invariants() -> None:
    batch = SBATCH_SCRIPT.read_text()
    submit = SUBMIT_SCRIPT.read_text()

    for source in (batch, submit):
        assert CATALOG_SHA256 in source
        assert "/projects/lucasbao/tengcc/auto_dgp2" in source
        assert "/projects/lucasbao/tengcc/datasets/f02_nbody_v1" in source
        assert "job-2810370/catalog.json" in source
        assert "status --porcelain=v1 --untracked-files=all" in source

    assert "#SBATCH --array=0-134%1" in batch
    assert "#SBATCH --nodes=1" in batch
    assert "#SBATCH --gres=gpu:l40s:1" in batch
    assert "#SBATCH --exclusive" in batch
    assert "expected exactly one allocated node" in batch
    assert "OverSubscribe=EXCLUSIVE" in batch
    assert "slurm_no_oversubscribe_full_node_sole_job" in batch
    assert "F02_INTERNAL_SLURM_EXCLUSIVE_VERIFIED=1" in batch
    assert "cluster/check_python_environment.py" in batch
    assert "-m pip" not in batch
    assert "-m pip" not in submit
    assert "ACTUAL_CATALOG_SHA256" in batch
    assert batch.index("ACTUAL_CATALOG_SHA256") < batch.index("DATASET_PATH=")
    assert "--evaluation-split validation" in batch
    assert "--evaluation-design optimizer_selection" in batch
    assert '--train-steps "${TRAIN_STEPS}"' in batch
    assert "--train-epochs 0" in batch
    assert "--kernel rbf" in batch
    assert '--seed "${SEED}"' in batch
    assert "--batch-size 256" in batch
    assert "--candidate-m none" in batch
    assert "--dtype float32" in batch
    assert "--device cuda" in batch
    for artifact in (
        "git-submodules.txt",
        "source-files.sha256",
        "dependency-files.sha256",
        "dependency-audit.json",
        "dependency-packages.txt",
        "runtime.json",
        "slurm-job.txt",
        "slurm-node.txt",
        "gpu.csv",
        "command.txt",
        "exit-code.txt",
        "artifacts.sha256",
    ):
        assert artifact in batch


def test_submit_helper_is_safe_by_default_and_pilot_needs_explicit_submit() -> None:
    source = SUBMIT_SCRIPT.read_text()
    help_result = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "MODE=preflight" in source
    assert "MAX_PARALLEL=${F02_INTERNAL_MAX_PARALLEL:-1}" in source
    assert "ARRAY_SPEC=0%1" in source
    assert "ARRAY_SPEC=0-134%${MAX_PARALLEL}" in source
    assert "--test-only" in source
    assert "submit)" in source
    assert 'mkdir -p "${RUN_ROOT}"' in source
    assert "Only --submit queues work" in help_result.stdout
    assert "--submit --pilot" in help_result.stdout
