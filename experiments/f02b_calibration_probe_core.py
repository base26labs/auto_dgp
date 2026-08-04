"""Label-free core primitives for the development-only F02b probes.

This module deliberately stops before model prediction and artifact emission.
It selects only public validation coordinates, freezes the released TERA
neighbour identities in source-fp32 geometry, and expands an immutable grid
task into its preregistered numerical work plan.  It never reads energy or
force labels and never reselects neighbours in float64.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from cluster.f02b_calibration_grid import (
    CALIBRATION_ID,
    CALIBRATION_MATRIX_SHA256,
    DEVELOPMENT_REPLICAS,
    FIT_TASK_MATRIX_SHA256,
    PROBE_TASK_COUNT,
    PROBE_TASK_MATRIX_SHA256,
    PROBE_TASKS,
    REFERENCE_M,
    F02bCalibrationProbeTask,
    probe_task_for_index,
)
from experiments.f02_design import EVALUATION_TIME_INDICES

PRIMARY_EVALUATION_TRAJECTORY_COUNT = 20
PRIMARY_EVALUATION_ROWS_PER_TRAJECTORY = len(EVALUATION_TIME_INDICES)
PRIMARY_EVALUATION_ROW_COUNT = (
    PRIMARY_EVALUATION_TRAJECTORY_COUNT * PRIMARY_EVALUATION_ROWS_PER_TRAJECTORY
)

PRODUCTION_TOLERANCE = 1e-5
SHARED_TOLERANCE_SWEEP = (
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
FP64_ONLY_TOLERANCE_SWEEP = (1e-9, 1e-10, 1e-11, 1e-12)
SOLVER_MAX_ITERATIONS_CAP = 4096

PROBE_WORK_PLAN_HASH_DOMAIN = "auto_dgp2.f02b.probe_work_plan"
PROBE_WORK_PLAN_SCHEMA_VERSION = "f02b_calibration_probe_work_plan_v1"


class ProbeCoreInputError(ValueError):
    """Raised when label-free probe inputs violate the frozen core contract."""


@dataclass(frozen=True, slots=True)
class ProbeEvaluationRows:
    """Copied public coordinates for the 100 primary validation targets."""

    source_indices: np.ndarray
    X: np.ndarray
    trajectory_id: np.ndarray
    time_index: np.ndarray
    time_value: np.ndarray


@dataclass(frozen=True, slots=True)
class FixedNeighbourRows:
    """Pinned released-TERA neighbours by train position and source identity."""

    positions: torch.Tensor
    source_indices: torch.Tensor
    m: int


@dataclass(frozen=True, slots=True)
class ProbeWorkPlan:
    """Complete label-independent numerical work registered for one probe task."""

    task_index: int
    fit_task_index: int
    role: str
    repeat_id: int
    dimension: int
    geometry_m_values: tuple[int, ...]
    production_m: int
    physical_rank: int
    support_target_count: int
    stress_m: int | None
    stress_support_target_count: int
    full_q_m: int | None
    production_tolerance: float
    shared_tolerance_sweep: tuple[float, ...]
    fp64_only_tolerance_sweep: tuple[float, ...]
    max_iterations: int

    @property
    def run_full_q(self) -> bool:
        """Whether the registered four-arm released full-q diagnostic runs."""

        return self.full_q_m is not None

    @property
    def full_q(self) -> int | None:
        """Return the sole registered full-q neighbourhood, when applicable."""

        return self.full_q_m

    @property
    def solver_max_iterations(self) -> int:
        """Return the registered ORBIT iteration cap for the production support."""

        return self.max_iterations

    def as_record(self) -> dict[str, Any]:
        """Return a fresh canonical-JSON-safe plan record."""

        record = asdict(self)
        record["geometry_m_values"] = list(self.geometry_m_values)
        record["shared_tolerance_sweep"] = list(self.shared_tolerance_sweep)
        record["fp64_only_tolerance_sweep"] = list(self.fp64_only_tolerance_sweep)
        return record


def _require_numpy_array(value: object, label: str, *, ndim: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ProbeCoreInputError(f"{label} must be a numpy.ndarray")
    if value.ndim != ndim:
        raise ProbeCoreInputError(f"{label} must be {ndim}-dimensional")
    return value


def _require_integer_numpy_array(value: object, label: str) -> np.ndarray:
    array = _require_numpy_array(value, label, ndim=1)
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(array.dtype, np.integer):
        raise ProbeCoreInputError(f"{label} must have an integer dtype")
    return array


def _frozen_copy(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.array(value[mask], copy=True)
    result.flags.writeable = False
    return result


def select_primary_probe_rows(prepared_validation_split: object) -> ProbeEvaluationRows:
    """Select the fixed 20-trajectory by five-time validation geometry.

    Apart from binding the split name to ``validation``, only
    ``source_indices``, ``X``, ``trajectory_id``, ``time_index``, and
    ``time_value`` are obtained from ``prepared_validation_split``.  In
    particular, this function must remain usable with objects whose label
    attributes raise on access.
    """

    if prepared_validation_split.name != "validation":
        raise ProbeCoreInputError("prepared split name must be 'validation'")
    source_indices = _require_integer_numpy_array(
        prepared_validation_split.source_indices,
        "source_indices",
    )
    X = _require_numpy_array(prepared_validation_split.X, "X", ndim=2)
    trajectory_id = _require_integer_numpy_array(
        prepared_validation_split.trajectory_id,
        "trajectory_id",
    )
    time_index = _require_integer_numpy_array(
        prepared_validation_split.time_index,
        "time_index",
    )
    time_value = _require_numpy_array(
        prepared_validation_split.time_value,
        "time_value",
        ndim=1,
    )

    row_count = source_indices.shape[0]
    if row_count == 0:
        raise ProbeCoreInputError("prepared validation split must be nonempty")
    if any(value.shape[0] != row_count for value in (X, trajectory_id, time_index, time_value)):
        raise ProbeCoreInputError(
            "prepared validation coordinate fields must have equal row counts"
        )
    if not np.issubdtype(X.dtype, np.floating) or not np.isfinite(X).all():
        raise ProbeCoreInputError("X must contain only finite floating-point values")
    if not np.issubdtype(time_value.dtype, np.floating) or not np.isfinite(time_value).all():
        raise ProbeCoreInputError("time_value must contain only finite floating-point values")
    if np.any(source_indices < 0) or np.any(np.diff(source_indices) <= 0):
        raise ProbeCoreInputError(
            "source_indices must be unique, increasing, and nonnegative canonical row identities"
        )
    if np.any(trajectory_id < 0) or np.any(time_index < 0):
        raise ProbeCoreInputError("trajectory_id and time_index must be nonnegative")

    requested = np.asarray(EVALUATION_TIME_INDICES, dtype=time_index.dtype)
    mask = np.isin(time_index, requested)
    trajectories = np.unique(trajectory_id)
    if trajectories.size != PRIMARY_EVALUATION_TRAJECTORY_COUNT:
        raise ProbeCoreInputError(
            "prepared validation split must contain exactly "
            f"{PRIMARY_EVALUATION_TRAJECTORY_COUNT} trajectories"
        )
    for trajectory in trajectories:
        observed = time_index[mask & (trajectory_id == trajectory)]
        if not np.array_equal(observed, requested):
            raise ProbeCoreInputError(
                f"trajectory {int(trajectory)} does not contain the evaluation time indices "
                "exactly once in registered order"
            )

    selected_trajectory = trajectory_id[mask]
    selected_time = time_index[mask]
    expected_trajectory = np.repeat(trajectories, PRIMARY_EVALUATION_ROWS_PER_TRAJECTORY)
    expected_time = np.tile(requested, PRIMARY_EVALUATION_TRAJECTORY_COUNT)
    if selected_time.size != PRIMARY_EVALUATION_ROW_COUNT:
        raise ProbeCoreInputError(
            f"primary evaluation design must contain exactly {PRIMARY_EVALUATION_ROW_COUNT} rows"
        )
    if not np.array_equal(selected_trajectory, expected_trajectory) or not np.array_equal(
        selected_time,
        expected_time,
    ):
        raise ProbeCoreInputError(
            "primary evaluation rows must retain canonical trajectory blocks and time order"
        )

    return ProbeEvaluationRows(
        source_indices=_frozen_copy(source_indices, mask),
        X=_frozen_copy(X, mask),
        trajectory_id=_frozen_copy(trajectory_id, mask),
        time_index=_frozen_copy(time_index, mask),
        time_value=_frozen_copy(time_value, mask),
    )


def _require_fp32_tensor(
    value: object,
    label: str,
    *,
    ndim: int | None,
    nonempty: bool = True,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ProbeCoreInputError(f"{label} must be a torch.Tensor")
    if value.dtype != torch.float32:
        raise ProbeCoreInputError(f"{label} must have dtype torch.float32")
    if ndim is not None and value.ndim != ndim:
        raise ProbeCoreInputError(f"{label} must be {ndim}-dimensional")
    if nonempty and value.numel() == 0:
        raise ProbeCoreInputError(f"{label} must be nonempty")
    if not bool(torch.isfinite(value).all().item()):
        raise ProbeCoreInputError(f"{label} must contain only finite values")
    return value


def fixed_fp32_neighbours(
    train_x32: torch.Tensor,
    evaluation_x32: torch.Tensor,
    lengthscale32: torch.Tensor,
    train_source_indices: torch.Tensor,
    evaluation_source_indices: torch.Tensor,
    m: int,
) -> FixedNeighbourRows:
    """Freeze an unambiguous pinned-TERA KNN result in source-fp32 geometry.

    The caller must already have authorized the development corpus against the
    frozen task and catalog.  Split names and public row identities alone are
    deliberately not treated as corpus authorization here.
    """

    train_x32 = _require_fp32_tensor(train_x32, "train_x32", ndim=2)
    evaluation_x32 = _require_fp32_tensor(evaluation_x32, "evaluation_x32", ndim=2)
    lengthscale32 = _require_fp32_tensor(lengthscale32, "lengthscale32", ndim=None)
    if type(m) is not int or m <= 0:
        raise ProbeCoreInputError("m must be a positive integer, not bool")
    if m > train_x32.shape[0]:
        raise ProbeCoreInputError("m must not exceed the number of training rows")
    if evaluation_x32.device != train_x32.device or lengthscale32.device != train_x32.device:
        raise ProbeCoreInputError("all floating-point inputs must be on the same device")
    if evaluation_x32.shape[1] != train_x32.shape[1]:
        raise ProbeCoreInputError("train and evaluation feature dimensions must match")
    if lengthscale32.ndim not in (0, 1) or lengthscale32.numel() != 1:
        raise ProbeCoreInputError("lengthscale32 must contain exactly one non-ARD fit value")
    if bool((lengthscale32 <= 0.0).any().item()):
        raise ProbeCoreInputError("lengthscale32 must be strictly positive")

    for source_indices, label, expected_rows in (
        (train_source_indices, "train_source_indices", train_x32.shape[0]),
        (
            evaluation_source_indices,
            "evaluation_source_indices",
            evaluation_x32.shape[0],
        ),
    ):
        if not isinstance(source_indices, torch.Tensor):
            raise ProbeCoreInputError(f"{label} must be a torch.Tensor")
        if source_indices.dtype != torch.long or source_indices.ndim != 1:
            raise ProbeCoreInputError(f"{label} must be a one-dimensional torch.long tensor")
        if source_indices.device != train_x32.device:
            raise ProbeCoreInputError(f"{label} must be on the source-data device")
        if source_indices.shape[0] != expected_rows:
            raise ProbeCoreInputError(f"{label} must match its coordinate row count")
        if bool((source_indices < 0).any().item()):
            raise ProbeCoreInputError(f"{label} must be nonnegative")
        if source_indices.numel() > 1 and bool((torch.diff(source_indices) <= 0).any().item()):
            raise ProbeCoreInputError(f"{label} must be unique and strictly increasing")
    if bool(torch.isin(evaluation_source_indices, train_source_indices).any().item()):
        raise ProbeCoreInputError("training and evaluation source rows must be disjoint")

    scale = lengthscale32.reshape(1, 1)
    train_scaled = train_x32 / scale
    evaluation_scaled = evaluation_x32 / scale
    if train_scaled.dtype != torch.float32 or evaluation_scaled.dtype != torch.float32:
        raise ProbeCoreInputError("neighbour scaling must remain in source float32")
    if not bool(torch.isfinite(train_scaled).all().item()) or not bool(
        torch.isfinite(evaluation_scaled).all().item()
    ):
        raise ProbeCoreInputError("source-fp32 neighbour scaling produced a nonfinite value")

    distance_matrix = torch.cdist(evaluation_scaled, train_scaled)
    if distance_matrix.dtype != torch.float32 or not bool(
        torch.isfinite(distance_matrix).all().item()
    ):
        raise ProbeCoreInputError("source-fp32 neighbour distances must be finite float32")

    # Importing gp.tera first installs the pinned vendor source on sys.path.
    importlib.import_module("gp.tera")
    from gp_sim_kl.ordering import knn_to_eval

    vendor_rows = knn_to_eval(train_scaled, evaluation_scaled, m)
    if not isinstance(vendor_rows, (list, tuple)):
        raise ProbeCoreInputError("pinned vendor KNN must return a row sequence")
    if len(vendor_rows) != evaluation_x32.shape[0]:
        raise ProbeCoreInputError("pinned vendor KNN returned the wrong evaluation row count")

    checked_rows: list[torch.Tensor] = []
    for row_index, row in enumerate(vendor_rows):
        if not isinstance(row, torch.Tensor):
            raise ProbeCoreInputError(f"vendor neighbour row {row_index} must be a torch.Tensor")
        if row.dtype != torch.long or row.device != train_x32.device or row.ndim != 1:
            raise ProbeCoreInputError(
                f"vendor neighbour row {row_index} must be one-dimensional torch.long on input device"
            )
        if row.numel() != m:
            raise ProbeCoreInputError(
                f"vendor neighbour row {row_index} must contain exactly {m} positions"
            )
        if bool(((row < 0) | (row >= train_x32.shape[0])).any().item()):
            raise ProbeCoreInputError(f"vendor neighbour row {row_index} is out of bounds")
        if torch.unique(row).numel() != row.numel():
            raise ProbeCoreInputError(f"vendor neighbour row {row_index} contains duplicates")
        selected_distances = distance_matrix[row_index, row]
        if selected_distances.numel() > 1 and not bool(
            (selected_distances[1:] > selected_distances[:-1]).all().item()
        ):
            raise ProbeCoreInputError(
                f"vendor neighbour row {row_index} lacks a strict deterministic distance order"
            )
        if m < train_x32.shape[0]:
            excluded = torch.ones(
                train_x32.shape[0],
                dtype=torch.bool,
                device=train_x32.device,
            )
            excluded[row] = False
            if not bool(
                (selected_distances[-1] < distance_matrix[row_index, excluded].min()).item()
            ):
                raise ProbeCoreInputError(
                    f"vendor neighbour row {row_index} has an ambiguous or incorrect m-boundary"
                )
        checked_rows.append(row.detach().clone().contiguous())

    positions = torch.stack(checked_rows, dim=0)
    source_indices = train_source_indices[positions].detach().clone().contiguous()
    return FixedNeighbourRows(positions=positions, source_indices=source_indices, m=m)


def build_probe_work_plan(task: F02bCalibrationProbeTask) -> ProbeWorkPlan:
    """Expand one authoritative grid task into its complete numerical plan."""

    if type(task) is not F02bCalibrationProbeTask:
        raise ProbeCoreInputError("task must be an F02bCalibrationProbeTask")
    integer_fields = (task.task_index, task.fit_task_index, task.m, task.repeat_id)
    if any(type(value) is not int for value in integer_fields) or type(task.role) is not str:
        raise ProbeCoreInputError("task fields must retain their exact frozen scalar types")
    try:
        authoritative = probe_task_for_index(int(task.task_index))
    except IndexError as error:
        raise ProbeCoreInputError("task is outside the frozen probe matrix") from error
    if task != authoritative:
        raise ProbeCoreInputError("task does not match the frozen probe matrix record")

    dimension = task.fit_task.dimension
    physical_rank = min(task.m, dimension - 6)
    max_iterations = min(4 * task.m * physical_rank, SOLVER_MAX_ITERATIONS_CAP)
    full_q_m = REFERENCE_M if task.m == REFERENCE_M else None

    if task.role == "reproducibility":
        if task.geometry_m_values != (REFERENCE_M,) or task.stress_m is not None:
            raise ProbeCoreInputError(
                "reproducibility tasks must use only m=50 geometry without stress"
            )
    elif task.role not in {"reference", "resource_sweep"}:
        raise ProbeCoreInputError(f"unregistered probe role {task.role!r}")

    return ProbeWorkPlan(
        task_index=task.task_index,
        fit_task_index=task.fit_task_index,
        role=task.role,
        repeat_id=task.repeat_id,
        dimension=dimension,
        geometry_m_values=tuple(task.geometry_m_values),
        production_m=task.m,
        physical_rank=physical_rank,
        support_target_count=task.support_target_count,
        stress_m=task.stress_m,
        stress_support_target_count=task.stress_support_target_count,
        full_q_m=full_q_m,
        production_tolerance=PRODUCTION_TOLERANCE,
        shared_tolerance_sweep=SHARED_TOLERANCE_SWEEP,
        fp64_only_tolerance_sweep=FP64_ONLY_TOLERANCE_SWEEP,
        max_iterations=max_iterations,
    )


def canonical_probe_work_plan_records() -> tuple[dict[str, Any], ...]:
    """Return fresh records for all 122 work plans in frozen task order."""

    return tuple(build_probe_work_plan(task).as_record() for task in PROBE_TASKS)


def canonical_probe_work_plan_payload() -> dict[str, Any]:
    """Return the domain-separated, label-independent probe-core plan."""

    return {
        "hash_domain": PROBE_WORK_PLAN_HASH_DOMAIN,
        "schema_version": PROBE_WORK_PLAN_SCHEMA_VERSION,
        "calibration_id": CALIBRATION_ID,
        "matrix_hashes": {
            "fit_task_matrix_sha256": FIT_TASK_MATRIX_SHA256,
            "probe_task_matrix_sha256": PROBE_TASK_MATRIX_SHA256,
            "calibration_matrix_sha256": CALIBRATION_MATRIX_SHA256,
        },
        "development_scope": {
            "allowed_replicas": list(DEVELOPMENT_REPLICAS),
            "evaluation_split": "validation",
            "trajectory_count": PRIMARY_EVALUATION_TRAJECTORY_COUNT,
            "time_indices": list(EVALUATION_TIME_INDICES),
            "target_count": PRIMARY_EVALUATION_ROW_COUNT,
            "row_order": "trajectory_id_then_registered_time_index",
            "labels_may_select_rows": False,
        },
        "neighbour_rule": {
            "implementation": "pinned_vendor_gp_sim_kl.ordering.knn_to_eval",
            "source_dtype": "float32",
            "scaling": "learned_isotropic_lengthscale",
            "m_exceeds_training_rows": "structural_failure",
            "source_identity_policy": ("strictly_increasing_train_and_evaluation_disjoint"),
            "tie_policy": "reject_non_strict_selected_or_boundary_distance_order",
            "float64_reselection": False,
        },
        "rank_rule": {
            "identity": "source-fp32-smax-maxshape-eps-v1",
            "cutoff": "s1*max(D,m)*eps(float32)",
            "selection": "strict_s_greater_than_cutoff",
            "physical_rank": "min(m,D-6)",
        },
        "strata_rule": {
            "sort_key": ["guard", "target_source_index"],
            "median": "upper_median_index_target_count_floor_div_2",
            "m50": ["worst", "median", "best"],
            "other_prediction_m": ["worst", "best"],
        },
        "stress_registry": {
            "eligible": "repeat_id_0_seed_11_at_m_D_minus_5",
            "target": "worst",
            "tests": [
                "support_complement",
                "permutation",
                "support_rotation",
                "exact_zero_augmentation",
                "discarded_mode_leakage",
            ],
        },
        "full_q_registry": {
            "eligible": "m_50_support_strata",
            "arms": [
                "native_fp32_assembly_fp32_solve",
                "fp32_assembly_promoted_fp64_solve",
                "native_quantized_input_fp64_assembly_fp64_solve",
                "fp64_assembly_cast_fp32_solve",
            ],
        },
        "work_plans": list(canonical_probe_work_plan_records()),
    }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


# Frozen after building the complete domain-separated probe-core plan.
PROBE_WORK_PLAN_SHA256 = "c815d848c8866ed085522d56f9db7aedef304a6d3c6e4ef3c24ee0be7f25498e"


def _validate_work_plan_matrix() -> None:
    records = canonical_probe_work_plan_records()
    if len(records) != PROBE_TASK_COUNT or PROBE_TASK_COUNT != 122:
        raise RuntimeError("F02b probe work-plan matrix must contain exactly 122 records")
    if _sha256(canonical_probe_work_plan_payload()) != PROBE_WORK_PLAN_SHA256:
        raise RuntimeError("F02b probe work-plan SHA-256 does not match its frozen literal")


_validate_work_plan_matrix()
