from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch

from cluster.f02b_calibration_grid import PROBE_TASKS, probe_task_for_index
from experiments.f02_design import EVALUATION_TIME_INDICES
from experiments.f02b_calibration_probe_core import (
    FP64_ONLY_TOLERANCE_SWEEP,
    PRIMARY_EVALUATION_ROW_COUNT,
    PROBE_WORK_PLAN_HASH_DOMAIN,
    PROBE_WORK_PLAN_SCHEMA_VERSION,
    PROBE_WORK_PLAN_SHA256,
    PRODUCTION_TOLERANCE,
    SHARED_TOLERANCE_SWEEP,
    FixedNeighbourRows,
    ProbeCoreInputError,
    ProbeEvaluationRows,
    ProbeWorkPlan,
    build_probe_work_plan,
    canonical_probe_work_plan_payload,
    canonical_probe_work_plan_records,
    fixed_fp32_neighbours,
    select_primary_probe_rows,
)


class _ValidationSplitSpy:
    def __init__(
        self,
        *,
        source_indices: np.ndarray,
        X: np.ndarray,
        trajectory_id: np.ndarray,
        time_index: np.ndarray,
        time_value: np.ndarray,
    ) -> None:
        self.name = "validation"
        self.source_indices = source_indices
        self.X = X
        self.trajectory_id = trajectory_id
        self.time_index = time_index
        self.time_value = time_value

    @property
    def E(self) -> np.ndarray:
        raise AssertionError("energy labels must not be accessed")

    @property
    def F(self) -> np.ndarray:
        raise AssertionError("force labels must not be accessed")


def _validation_split(*, trajectories: int = 20) -> _ValidationSplitSpy:
    rows_per_trajectory = 100
    row_count = trajectories * rows_per_trajectory
    trajectory_id = np.repeat(np.arange(100, 100 + trajectories, dtype=np.int64), 100)
    time_index = np.tile(np.arange(rows_per_trajectory, dtype=np.int64), trajectories)
    source_indices = np.arange(1_000, 1_000 + row_count, dtype=np.int64)
    X = np.column_stack(
        (
            source_indices.astype(np.float64),
            trajectory_id.astype(np.float64),
            time_index.astype(np.float64),
        )
    )
    return _ValidationSplitSpy(
        source_indices=source_indices,
        X=X,
        trajectory_id=trajectory_id,
        time_index=time_index,
        time_value=time_index.astype(np.float64) / 99.0,
    )


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def test_primary_rows_are_exact_label_free_read_only_copies() -> None:
    split = _validation_split()

    selected = select_primary_probe_rows(split)

    requested = np.asarray(EVALUATION_TIME_INDICES)
    expected_source = np.concatenate(
        [1_000 + trajectory * 100 + requested for trajectory in range(20)]
    )
    assert isinstance(selected, ProbeEvaluationRows)
    assert selected.source_indices.tolist() == expected_source.tolist()
    assert selected.X.shape == (PRIMARY_EVALUATION_ROW_COUNT, 3)
    assert selected.trajectory_id.tolist() == np.repeat(np.arange(100, 120), 5).tolist()
    assert selected.time_index.tolist() == np.tile(requested, 20).tolist()
    assert selected.time_value.tolist() == np.tile(requested / 99.0, 20).tolist()
    for value in (
        selected.source_indices,
        selected.X,
        selected.trajectory_id,
        selected.time_index,
        selected.time_value,
    ):
        assert not value.flags.writeable
        assert not np.shares_memory(value, split.source_indices)

    original = selected.X[0, 0]
    split.X[0, 0] = -999.0
    assert selected.X[0, 0] == original
    with pytest.raises(ValueError, match="read-only"):
        selected.X[0, 0] = 0.0


def test_primary_row_dataclass_is_frozen() -> None:
    selected = select_primary_probe_rows(_validation_split())
    with pytest.raises(FrozenInstanceError):
        selected.X = selected.X.copy()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda split: setattr(split, "trajectory_id", split.trajectory_id[:-100]), "row counts"),
        (lambda split: setattr(split, "source_indices", split.source_indices[::-1]), "increasing"),
        (
            lambda split: split.time_index.__setitem__(25, 24),
            "does not contain the evaluation time indices",
        ),
        (lambda split: split.X.__setitem__((0, 0), np.nan), "finite"),
        (lambda split: split.time_value.__setitem__(0, np.inf), "finite"),
    ],
)
def test_primary_rows_fail_closed_on_noncanonical_public_coordinates(mutation, match) -> None:
    split = _validation_split()
    mutation(split)
    with pytest.raises(ProbeCoreInputError, match=match):
        select_primary_probe_rows(split)


def test_primary_rows_require_exactly_twenty_trajectories() -> None:
    with pytest.raises(ProbeCoreInputError, match="exactly 20 trajectories"):
        select_primary_probe_rows(_validation_split(trajectories=19))


def test_primary_rows_reject_same_shape_test_split() -> None:
    split = _validation_split()
    split.name = "test"
    with pytest.raises(ProbeCoreInputError, match="must be 'validation'"):
        select_primary_probe_rows(split)


def test_public_tolerance_constants_match_the_frozen_protocol() -> None:
    assert PRODUCTION_TOLERANCE == 1e-5
    assert SHARED_TOLERANCE_SWEEP == (
        1e-3,
        1e-4,
        3e-5,
        1e-5,
        3e-6,
        1e-6,
        3e-7,
        1e-7,
        1e-8,
    )
    assert FP64_ONLY_TOLERANCE_SWEEP == (1e-9, 1e-10, 1e-11, 1e-12)


def test_work_plans_exactly_expand_reference_sweep_and_replay_roles() -> None:
    seed11_reference = build_probe_work_plan(probe_task_for_index(0))
    seed29_reference = build_probe_work_plan(probe_task_for_index(1))
    high_dimension_reference = build_probe_work_plan(probe_task_for_index(42))
    sweep20 = build_probe_work_plan(probe_task_for_index(45))
    sweep200 = build_probe_work_plan(probe_task_for_index(49))
    replay = build_probe_work_plan(probe_task_for_index(120))

    assert seed11_reference == ProbeWorkPlan(
        task_index=0,
        fit_task_index=0,
        role="reference",
        repeat_id=0,
        dimension=12,
        geometry_m_values=(50, 5, 6, 7),
        production_m=50,
        physical_rank=6,
        support_target_count=3,
        stress_m=7,
        stress_support_target_count=1,
        stress_max_iterations=168,
        full_q_m=50,
        production_tolerance=PRODUCTION_TOLERANCE,
        shared_tolerance_sweep=SHARED_TOLERANCE_SWEEP,
        fp64_only_tolerance_sweep=FP64_ONLY_TOLERANCE_SWEEP,
        max_iterations=1_200,
    )
    assert seed11_reference.run_full_q
    assert seed11_reference.full_q == 50
    assert seed11_reference.solver_max_iterations == 1_200
    assert seed29_reference.stress_m is None
    assert high_dimension_reference.geometry_m_values == (50, 53, 54, 55)
    assert high_dimension_reference.physical_rank == 50
    assert high_dimension_reference.max_iterations == 4_096
    assert sweep20.geometry_m_values == (20,)
    assert sweep20.support_target_count == 2
    assert sweep20.physical_rank == 6
    assert sweep20.max_iterations == 480
    assert sweep20.full_q_m is None
    assert sweep200.max_iterations == 4_096
    assert replay.role == "reproducibility"
    assert replay.repeat_id == 1
    assert replay.geometry_m_values == (50,)
    assert replay.support_target_count == 3
    assert replay.stress_m is None
    assert replay.stress_support_target_count == 0
    assert replay.full_q_m == 50


def test_all_122_work_plans_are_canonical_and_bound_by_literal_hash() -> None:
    records = canonical_probe_work_plan_records()
    payload = canonical_probe_work_plan_payload()

    assert len(records) == len(PROBE_TASKS) == 122
    assert [record["task_index"] for record in records] == list(range(122))
    assert PROBE_WORK_PLAN_SHA256 == (
        "11b3dd9863cbd010eb50e95f4f4a5941080eb10186731a34f0625dd9fd5b6586"
    )
    assert payload["hash_domain"] == PROBE_WORK_PLAN_HASH_DOMAIN
    assert payload["schema_version"] == PROBE_WORK_PLAN_SCHEMA_VERSION
    assert payload["development_scope"] == {
        "allowed_replicas": [0, 1, 2],
        "evaluation_split": "validation",
        "trajectory_count": 20,
        "time_indices": list(EVALUATION_TIME_INDICES),
        "target_count": 100,
        "row_order": "trajectory_id_then_registered_time_index",
        "labels_may_select_rows": False,
    }
    assert payload["neighbour_rule"]["m_exceeds_training_rows"] == "structural_failure"
    assert payload["neighbour_rule"]["float64_reselection"] is False
    assert payload["rank_rule"]["identity"] == "source-fp32-smax-maxshape-eps-v1"
    assert len(payload["full_q_registry"]["arms"]) == 4
    assert payload["work_plans"] == list(records)
    assert _canonical_sha256(payload) == PROBE_WORK_PLAN_SHA256
    assert all(record["production_tolerance"] == PRODUCTION_TOLERANCE for record in records)

    records[0]["geometry_m_values"].append(999)
    assert canonical_probe_work_plan_records()[0]["geometry_m_values"] == [50, 5, 6, 7]


def test_work_plan_rejects_forged_grid_tasks_and_is_frozen() -> None:
    task = probe_task_for_index(0)
    with pytest.raises(ProbeCoreInputError, match="frozen probe matrix"):
        build_probe_work_plan(replace(task, m=49))

    plan = build_probe_work_plan(task)
    with pytest.raises(FrozenInstanceError):
        plan.production_m = 49  # type: ignore[misc]


def test_fixed_neighbours_save_positions_and_source_identities() -> None:
    train = torch.tensor([[0.0], [3.0], [8.0]], dtype=torch.float32)
    evaluation = torch.tensor([[1.0], [6.0]], dtype=torch.float32)
    source = torch.tensor([10, 20, 30], dtype=torch.long)
    evaluation_source = torch.tensor([40, 50], dtype=torch.long)

    fixed = fixed_fp32_neighbours(
        train,
        evaluation,
        torch.tensor(1.0),
        source,
        evaluation_source,
        m=2,
    )

    assert isinstance(fixed, FixedNeighbourRows)
    assert fixed.m == 2
    assert fixed.positions.shape == (2, 2)
    assert fixed.positions.dtype == torch.long
    assert fixed.source_indices.shape == (2, 2)
    assert torch.equal(fixed.source_indices, source[fixed.positions])
    assert all(torch.unique(row).numel() == 2 for row in fixed.positions)


def test_fixed_neighbours_call_pinned_vendor_on_scaled_fp32(monkeypatch) -> None:
    importlib.import_module("gp.tera")
    ordering = importlib.import_module("gp_sim_kl.ordering")

    observed: dict[str, object] = {}

    def fake_knn(train_scaled: torch.Tensor, eval_scaled: torch.Tensor, m: int):
        observed.update(train=train_scaled.clone(), evaluation=eval_scaled.clone(), m=m)
        return [torch.tensor([1, 0], dtype=torch.long) for _ in range(eval_scaled.shape[0])]

    monkeypatch.setattr(ordering, "knn_to_eval", fake_knn)
    train = torch.tensor([[2.0, 8.0], [4.0, 12.0]], dtype=torch.float32)
    evaluation = torch.tensor([[6.0, 16.0]], dtype=torch.float32)
    lengthscale = torch.tensor([2.0], dtype=torch.float32)
    source = torch.tensor([101, 303], dtype=torch.long)
    evaluation_source = torch.tensor([505], dtype=torch.long)

    fixed = fixed_fp32_neighbours(
        train,
        evaluation,
        lengthscale,
        source,
        evaluation_source,
        m=2,
    )

    assert observed["m"] == 2
    assert torch.equal(observed["train"], torch.tensor([[1.0, 4.0], [2.0, 6.0]]))
    assert torch.equal(observed["evaluation"], torch.tensor([[3.0, 8.0]]))
    assert observed["train"].dtype == torch.float32
    assert observed["evaluation"].dtype == torch.float32
    assert fixed.positions.tolist() == [[1, 0]]
    assert fixed.source_indices.tolist() == [[303, 101]]
    assert fixed.positions.shape[1] == 2


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("train", torch.tensor([[0.0]], dtype=torch.float64), "torch.float32"),
        ("evaluation", torch.tensor([[float("nan")]], dtype=torch.float32), "finite"),
        ("lengthscale", torch.tensor([0.0], dtype=torch.float32), "strictly positive"),
        (
            "lengthscale",
            torch.tensor([1.0, 1.0], dtype=torch.float32),
            "exactly one non-ARD",
        ),
        ("source", torch.tensor([4, 4], dtype=torch.long), "unique"),
        ("source", torch.tensor([-1, 4], dtype=torch.long), "nonnegative"),
        ("m", 0, "positive integer"),
        ("m", True, "positive integer"),
    ],
)
def test_fixed_neighbours_reject_invalid_source_fp32_contract(field, replacement, match) -> None:
    arguments = {
        "train": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        "evaluation": torch.tensor([[0.5]], dtype=torch.float32),
        "lengthscale": torch.tensor([1.0], dtype=torch.float32),
        "source": torch.tensor([4, 9], dtype=torch.long),
        "evaluation_source": torch.tensor([12], dtype=torch.long),
        "m": 1,
    }
    arguments[field] = replacement

    with pytest.raises(ProbeCoreInputError, match=match):
        fixed_fp32_neighbours(
            arguments["train"],
            arguments["evaluation"],
            arguments["lengthscale"],
            arguments["source"],
            arguments["evaluation_source"],
            arguments["m"],
        )


def test_fixed_neighbours_reject_vendor_duplicates(monkeypatch) -> None:
    importlib.import_module("gp.tera")
    ordering = importlib.import_module("gp_sim_kl.ordering")

    def duplicate_knn(_train: torch.Tensor, _evaluation: torch.Tensor, _m: int):
        return [torch.tensor([0, 0], dtype=torch.long)]

    monkeypatch.setattr(ordering, "knn_to_eval", duplicate_knn)
    with pytest.raises(ProbeCoreInputError, match="duplicates"):
        fixed_fp32_neighbours(
            torch.tensor([[0.0], [1.0]], dtype=torch.float32),
            torch.tensor([[0.5]], dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([4, 9], dtype=torch.long),
            torch.tensor([12], dtype=torch.long),
            2,
        )


def test_fixed_neighbours_reject_m_larger_than_training_population(monkeypatch) -> None:
    importlib.import_module("gp.tera")
    ordering = importlib.import_module("gp_sim_kl.ordering")
    called = False

    def forbidden_knn(*_args):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(ordering, "knn_to_eval", forbidden_knn)
    with pytest.raises(ProbeCoreInputError, match="must not exceed"):
        fixed_fp32_neighbours(
            torch.tensor([[0.0], [1.0]], dtype=torch.float32),
            torch.tensor([[0.5]], dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([4, 9], dtype=torch.long),
            torch.tensor([12], dtype=torch.long),
            3,
        )
    assert not called


def test_fixed_neighbours_reject_overflowing_source_fp32_distances() -> None:
    with pytest.raises(ProbeCoreInputError, match="distances"):
        fixed_fp32_neighbours(
            torch.tensor([[0.0], [1.0]], dtype=torch.float32),
            torch.tensor([[0.5]], dtype=torch.float32),
            torch.tensor([torch.finfo(torch.float32).tiny], dtype=torch.float32),
            torch.tensor([4, 9], dtype=torch.long),
            torch.tensor([12], dtype=torch.long),
            1,
        )


def test_fixed_neighbours_reject_ties_and_incorrect_vendor_order(monkeypatch) -> None:
    importlib.import_module("gp.tera")
    ordering = importlib.import_module("gp_sim_kl.ordering")

    with pytest.raises(ProbeCoreInputError, match="strict deterministic distance order"):
        fixed_fp32_neighbours(
            torch.tensor([[0.0], [2.0], [5.0]], dtype=torch.float32),
            torch.tensor([[1.0]], dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([4, 9, 11], dtype=torch.long),
            torch.tensor([12], dtype=torch.long),
            2,
        )

    def wrong_order(_train: torch.Tensor, _evaluation: torch.Tensor, _m: int):
        return [torch.tensor([1, 0], dtype=torch.long)]

    monkeypatch.setattr(ordering, "knn_to_eval", wrong_order)
    with pytest.raises(ProbeCoreInputError, match="strict deterministic distance order"):
        fixed_fp32_neighbours(
            torch.tensor([[0.0], [3.0], [8.0]], dtype=torch.float32),
            torch.tensor([[1.0]], dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([4, 9, 11], dtype=torch.long),
            torch.tensor([12], dtype=torch.long),
            2,
        )


def test_fixed_neighbours_reject_source_overlap_and_noncanonical_identity_order() -> None:
    common = {
        "train_x32": torch.tensor([[0.0], [3.0]], dtype=torch.float32),
        "evaluation_x32": torch.tensor([[1.0]], dtype=torch.float32),
        "lengthscale32": torch.tensor([1.0], dtype=torch.float32),
        "m": 1,
    }
    with pytest.raises(ProbeCoreInputError, match="must be disjoint"):
        fixed_fp32_neighbours(
            **common,
            train_source_indices=torch.tensor([4, 9], dtype=torch.long),
            evaluation_source_indices=torch.tensor([9], dtype=torch.long),
        )
    with pytest.raises(ProbeCoreInputError, match="strictly increasing"):
        fixed_fp32_neighbours(
            **common,
            train_source_indices=torch.tensor([9, 4], dtype=torch.long),
            evaluation_source_indices=torch.tensor([12], dtype=torch.long),
        )
