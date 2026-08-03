from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from cluster.f02b_calibration_grid import (
    CALIBRATION_ID,
    CALIBRATION_MATRIX_SHA256,
    DEVELOPMENT_REPLICAS,
    FIT_TASK_COUNT,
    FIT_TASK_MATRIX_SHA256,
    FIT_TASKS,
    KERNEL,
    N_DIMS,
    PARTICLE_COUNTS,
    PROBE_TASK_COUNT,
    PROBE_TASK_MATRIX_SHA256,
    PROBE_TASKS,
    REFERENCE_M,
    REFERENCE_PROBE_COUNT,
    REPRODUCIBILITY_PROBE_COUNT,
    RESOURCE_SWEEP_M,
    RESOURCE_SWEEP_PROBE_COUNT,
    SEEDS,
    TRAIN_STEPS,
    TRAINING_M,
    build_parser,
    canonical_fit_task_records,
    canonical_probe_task_records,
    fit_task_for_index,
    probe_task_for_index,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GRID_SCRIPT = REPO_ROOT / "cluster" / "f02b_calibration_grid.py"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_fit_matrix_is_the_exact_fixed_budget_cartesian_product() -> None:
    expected = [
        (replica, particles, seed)
        for replica in DEVELOPMENT_REPLICAS
        for particles in PARTICLE_COUNTS
        for seed in SEEDS
    ]
    observed = [(task.replica, task.n_particles, task.seed) for task in FIT_TASKS]

    assert DEVELOPMENT_REPLICAS == (0, 1, 2)
    assert PARTICLE_COUNTS == (2, 4, 6, 8, 10)
    assert SEEDS == (11, 29, 47)
    assert N_DIMS == 3
    assert TRAIN_STEPS == 20
    assert TRAINING_M == 20
    assert KERNEL == "rbf"
    assert CALIBRATION_ID == "F02B_NUMERICAL_CALIBRATION_v1"
    assert FIT_TASK_COUNT == 45 == len(expected)
    assert observed == expected
    assert (
        len(
            {
                (
                    task.replica,
                    task.n_particles,
                    task.n_dims,
                    task.seed,
                    task.train_steps,
                    task.training_m,
                    task.kernel,
                )
                for task in FIT_TASKS
            }
        )
        == FIT_TASK_COUNT
    )
    assert [task.task_index for task in FIT_TASKS] == list(range(FIT_TASK_COUNT))
    assert all(
        (task.n_dims, task.train_steps, task.training_m, task.kernel)
        == (N_DIMS, TRAIN_STEPS, TRAINING_M, KERNEL)
        for task in FIT_TASKS
    )


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (0, (0, 2, 11, 12, "nbody_fixedmass_n2_d3_replica0")),
        (1, (0, 2, 29, 12, "nbody_fixedmass_n2_d3_replica0")),
        (2, (0, 2, 47, 12, "nbody_fixedmass_n2_d3_replica0")),
        (3, (0, 4, 11, 24, "nbody_fixedmass_n4_d3_replica0")),
        (14, (0, 10, 47, 60, "nbody_fixedmass_n10_d3_replica0")),
        (15, (1, 2, 11, 12, "nbody_fixedmass_n2_d3_replica1")),
        (42, (2, 10, 11, 60, "nbody_fixedmass_n10_d3_replica2")),
        (44, (2, 10, 47, 60, "nbody_fixedmass_n10_d3_replica2")),
    ],
)
def test_fit_index_boundaries_and_derived_identity_are_stable(
    index: int,
    expected: tuple[int, int, int, int, str],
) -> None:
    task = fit_task_for_index(index)
    assert (
        task.replica,
        task.n_particles,
        task.seed,
        task.dimension,
        task.dataset_stem,
    ) == expected
    assert task.as_record()["D"] == task.dimension
    assert task.as_record()["calibration_id"] == CALIBRATION_ID


def test_probe_matrix_has_exact_reference_sweep_and_repeat_segments() -> None:
    seed11_fit_indices = [task.task_index for task in FIT_TASKS if task.seed == 11]
    expected_sweep = [(fit_index, m) for fit_index in seed11_fit_indices for m in RESOURCE_SWEEP_M]

    assert REFERENCE_M == 50
    assert REFERENCE_PROBE_COUNT == 45
    assert RESOURCE_SWEEP_PROBE_COUNT == 75
    assert REPRODUCIBILITY_PROBE_COUNT == 2
    assert PROBE_TASK_COUNT == 122
    assert [task.task_index for task in PROBE_TASKS] == list(range(PROBE_TASK_COUNT))
    assert [
        (task.fit_task_index, task.m, task.repeat_id, task.role) for task in PROBE_TASKS[:45]
    ] == [(fit_index, 50, 0, "reference") for fit_index in range(FIT_TASK_COUNT)]
    assert [(task.fit_task_index, task.m) for task in PROBE_TASKS[45:120]] == expected_sweep
    assert all((task.repeat_id, task.role) == (0, "resource_sweep") for task in PROBE_TASKS[45:120])
    assert [
        (task.fit_task_index, task.m, task.repeat_id, task.role) for task in PROBE_TASKS[120:]
    ] == [
        (0, 50, 1, "reproducibility"),
        (0, 50, 2, "reproducibility"),
    ]


@pytest.mark.parametrize(
    ("index", "fit_index", "m", "role"),
    [
        (0, 0, 50, "reference"),
        (44, 44, 50, "reference"),
        (45, 0, 20, "resource_sweep"),
        (49, 0, 200, "resource_sweep"),
        (50, 3, 20, "resource_sweep"),
        (119, 42, 200, "resource_sweep"),
        (120, 0, 50, "reproducibility"),
        (121, 0, 50, "reproducibility"),
    ],
)
def test_probe_index_boundaries_are_stable(
    index: int,
    fit_index: int,
    m: int,
    role: str,
) -> None:
    task = probe_task_for_index(index)
    assert (task.fit_task_index, task.m, task.role) == (fit_index, m, role)
    assert task.as_record()["dataset_stem"] == task.fit_task.dataset_stem
    assert task.as_record()["D"] == task.fit_task.dimension


def test_geometry_support_and_stress_scopes_are_deterministic() -> None:
    seed11_reference = probe_task_for_index(0)
    seed29_reference = probe_task_for_index(1)
    high_dimension_reference = probe_task_for_index(42)
    sweep = probe_task_for_index(45)
    repeat = probe_task_for_index(120)

    assert seed11_reference.geometry_m_values == (50, 5, 6, 7)
    assert seed11_reference.support_target_count == 3
    assert seed11_reference.stress_m == 7
    assert seed11_reference.stress_support_target_count == 1
    assert seed29_reference.geometry_m_values == (50, 5, 6, 7)
    assert seed29_reference.support_target_count == 3
    assert seed29_reference.stress_m is None
    assert seed29_reference.stress_support_target_count == 0
    assert high_dimension_reference.geometry_m_values == (50, 53, 54, 55)
    assert high_dimension_reference.stress_m == 55
    assert sweep.geometry_m_values == (20,)
    assert sweep.support_target_count == 2
    assert sweep.stress_m is None
    assert repeat.geometry_m_values == (50,)
    assert repeat.support_target_count == 3
    assert repeat.stress_m is None

    for task in PROBE_TASKS:
        expected_targets = 3 if task.m == 50 else 2
        assert task.support_target_count == expected_targets
        expected_stress = (
            task.fit_task.dimension - 5
            if task.m == 50 and task.repeat_id == 0 and task.fit_task.seed == 11
            else None
        )
        assert task.stress_m == expected_stress
        assert task.stress_support_target_count == (1 if expected_stress is not None else 0)


def test_records_are_canonical_fresh_values_bound_by_literal_hashes() -> None:
    fit_records = canonical_fit_task_records()
    probe_records = canonical_probe_task_records()
    combined = {"fit_tasks": fit_records, "probe_tasks": probe_records}

    assert FIT_TASK_MATRIX_SHA256 == (
        "e53cabcb788e9383431b4a6b50bc6631499d9acf3f338ea659d654d76e24513e"
    )
    assert PROBE_TASK_MATRIX_SHA256 == (
        "b729e755300fb997a18c07bf0cff185a1e60a7ed884355f95127cdd2f36aae7c"
    )
    assert CALIBRATION_MATRIX_SHA256 == (
        "d81aee9b479adf437abd7f44782e4688227d3361458d686302c976bda5150114"
    )
    assert _canonical_sha256(fit_records) == FIT_TASK_MATRIX_SHA256
    assert _canonical_sha256(probe_records) == PROBE_TASK_MATRIX_SHA256
    assert _canonical_sha256(combined) == CALIBRATION_MATRIX_SHA256
    assert all(record["calibration_id"] == CALIBRATION_ID for record in fit_records)
    assert all(record["calibration_id"] == CALIBRATION_ID for record in probe_records)

    probe_records[0]["geometry_m_values"].append(999)
    assert canonical_probe_task_records()[0]["geometry_m_values"] == [50, 5, 6, 7]


def test_task_records_are_immutable_and_negative_indices_fail() -> None:
    with pytest.raises(FrozenInstanceError):
        fit_task_for_index(0).seed = 999  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        probe_task_for_index(0).m = 999  # type: ignore[misc]
    for index in (-1, FIT_TASK_COUNT):
        with pytest.raises(IndexError):
            fit_task_for_index(index)
    for index in (-1, PROBE_TASK_COUNT):
        with pytest.raises(IndexError):
            probe_task_for_index(index)


def test_cli_counts_json_and_shell_safe_lines() -> None:
    fit_count = subprocess.run(
        [sys.executable, str(GRID_SCRIPT), "--matrix", "fit", "--count"],
        check=True,
        capture_output=True,
        text=True,
    )
    probe_count = subprocess.run(
        [sys.executable, str(GRID_SCRIPT), "--matrix", "probe", "--count"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = subprocess.run(
        [
            sys.executable,
            str(GRID_SCRIPT),
            "--matrix",
            "probe",
            "--task-index",
            "119",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fit_lines = subprocess.run(
        [
            sys.executable,
            str(GRID_SCRIPT),
            "--matrix",
            "fit",
            "--task-index",
            "0",
            "--format",
            "lines",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    probe_lines = subprocess.run(
        [
            sys.executable,
            str(GRID_SCRIPT),
            "--matrix",
            "probe",
            "--task-index",
            "0",
            "--format",
            "lines",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert fit_count.stdout.strip() == "45"
    assert probe_count.stdout.strip() == "122"
    assert json.loads(payload.stdout) == probe_task_for_index(119).as_record()
    assert fit_lines.stdout.splitlines() == [
        "0",
        "0",
        "2",
        "3",
        "12",
        "11",
        "20",
        "20",
        "rbf",
        "nbody_fixedmass_n2_d3_replica0",
        "F02B_NUMERICAL_CALIBRATION_v1",
    ]
    assert probe_lines.stdout.splitlines() == [
        "0",
        "0",
        "0",
        "2",
        "3",
        "12",
        "11",
        "50",
        "0",
        "reference",
        "nbody_fixedmass_n2_d3_replica0",
        "50,5,6,7",
        "3",
        "7",
        "1",
        "F02B_NUMERICAL_CALIBRATION_v1",
    ]


def test_cli_rejects_invalid_or_ambiguous_requests_and_exposes_no_science_knobs() -> None:
    for arguments in (
        ("--matrix", "fit", "--task-index", "-1"),
        ("--matrix", "fit", "--task-index", "45"),
        ("--matrix", "probe", "--task-index", "122"),
        ("--matrix", "probe", "--count", "--format", "lines"),
        ("--matrix", "fit", "--count", "--task-index", "0"),
    ):
        process = subprocess.run(
            [sys.executable, str(GRID_SCRIPT), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        assert process.returncode != 0

    option_strings = {
        option for action in build_parser()._actions for option in action.option_strings
    }
    assert option_strings == {
        "-h",
        "--help",
        "--matrix",
        "--task-index",
        "--count",
        "--format",
    }
