"""Pure numerical execution core for registered F02b ORBIT probe arms.

This module is intentionally downstream of authorization and upstream of
artifact emission.  It accepts an already tensorized training split, copied
public evaluation coordinates, and one source-fp32-pinned neighbour matrix.
It does not read paths, environment variables, Slurm state, catalogs, corpus
bytes, or evaluation labels.

For one target and compute dtype, :func:`execute_registered_orbit_target`
builds the represented ORBIT system exactly once and runs every registered
tolerance as an independent zero-start solve.  The production ``1e-5`` result
is the object already present in that sweep.  An additional operator
application immediately recomputes ``A(x)`` and ``b-A(x)`` for evidence; the
source-dtype result is diagnostic rather than a directed-rounding certificate.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from cluster.f02b_calibration_grid import probe_task_for_index
from experiments.f02_design import EVALUATION_TIME_INDICES
from experiments.f02_internal_models import FrozenTERAParameters, TensorConfirmatorySplit
from experiments.f02b_calibration_metrics import (
    cholesky_backward_error_metrics,
    matrix_free_solve_error_metrics,
    rank_boundary_metrics,
    select_geometry_strata,
)
from experiments.f02b_calibration_probe_core import (
    FP64_ONLY_TOLERANCE_SWEEP,
    PRIMARY_EVALUATION_ROW_COUNT,
    PRODUCTION_TOLERANCE,
    SHARED_TOLERANCE_SWEEP,
    FixedNeighbourRows,
    ProbeEvaluationRows,
    ProbeWorkPlan,
    build_probe_work_plan,
    fixed_fp32_neighbours,
)
from gp.orbit import (
    LocalGeometry,
    LocalPrediction,
    LocalValueSystem,
    build_local_geometry_from_differences,
    build_local_value_system,
    solve_local_value_system,
)
from gp.orbit.predictor import _build_local_value_system_from_registered_geometry

SOURCE_DTYPE = torch.float32
SOURCE_RANK_EPSILON = float(torch.finfo(SOURCE_DTYPE).eps)
SOURCE_FUNCTION_JITTER = float(torch.tensor(1e-8, dtype=SOURCE_DTYPE))
SOURCE_REDUCED_JITTER = SOURCE_FUNCTION_JITTER
MATRIX_FREE_ROUNDOFF_QUALIFICATION = (
    "source-dtype recomputation without directed rounding; a later correctness "
    "bound must add an explicit arithmetic-roundoff margin"
)
OPERATOR_ACTION_PROVENANCE = "solver_final_fresh_operator_application"
RESIDUAL_PROVENANCE = "solver_final_fresh_rhs_minus_operator_action"
REPLAY_ACTION_PROVENANCE = (
    "immediate_independent_replay_of_the_same_reusable_orbit_operator"
)
_ARM_CONSTRUCTION_TOKEN = object()
_SOURCE_GEOMETRY_CONSTRUCTION_TOKEN = object()
_STRATA_CONSTRUCTION_TOKEN = object()


class ProbeExecutionInputError(ValueError):
    """Raised when an input is outside the frozen numerical probe contract."""


class ProbeExecutionEvidenceError(RuntimeError):
    """Raised when a solver result contradicts its bound numerical evidence."""


@dataclass(frozen=True, slots=True)
class LabelFreeEvaluationTensors:
    """The 100 copied public validation rows, with no energy or force fields."""

    source_indices: torch.Tensor
    X: torch.Tensor
    trajectory_id: torch.Tensor
    time_index: torch.Tensor
    time_value: torch.Tensor

    def __post_init__(self) -> None:
        if self.X.ndim != 2 or self.X.shape[0] != PRIMARY_EVALUATION_ROW_COUNT:
            raise ProbeExecutionInputError(
                f"X must have shape ({PRIMARY_EVALUATION_ROW_COUNT}, d)"
            )
        if self.X.dtype not in {torch.float32, torch.float64}:
            raise ProbeExecutionInputError("X must have dtype float32 or float64")
        if not bool(torch.isfinite(self.X).all().item()):
            raise ProbeExecutionInputError("X must contain only finite values")
        expected = {
            "source_indices": (PRIMARY_EVALUATION_ROW_COUNT,),
            "trajectory_id": (PRIMARY_EVALUATION_ROW_COUNT,),
            "time_index": (PRIMARY_EVALUATION_ROW_COUNT,),
            "time_value": (PRIMARY_EVALUATION_ROW_COUNT,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or value.shape != shape:
                raise ProbeExecutionInputError(f"{name} must have shape {shape}")
            if value.device != self.X.device:
                raise ProbeExecutionInputError("all evaluation tensors must share one device")
        for name in ("source_indices", "trajectory_id", "time_index"):
            if getattr(self, name).dtype != torch.long:
                raise ProbeExecutionInputError(f"{name} must have dtype torch.long")
        if self.time_value.dtype != self.X.dtype:
            raise ProbeExecutionInputError("time_value must match the X dtype")
        if not bool(torch.isfinite(self.time_value).all().item()):
            raise ProbeExecutionInputError("time_value must contain only finite values")
        if bool((self.source_indices < 0).any().item()) or bool(
            (self.trajectory_id < 0).any().item()
        ) or bool((self.time_index < 0).any().item()):
            raise ProbeExecutionInputError("evaluation identities must be nonnegative")
        if bool((torch.diff(self.source_indices) <= 0).any().item()):
            raise ProbeExecutionInputError(
                "evaluation source_indices must be unique and strictly increasing"
            )
        trajectories = torch.unique(self.trajectory_id)
        expected_trajectory_count = PRIMARY_EVALUATION_ROW_COUNT // len(
            EVALUATION_TIME_INDICES
        )
        if trajectories.numel() != expected_trajectory_count:
            raise ProbeExecutionInputError(
                "evaluation rows must contain exactly 20 trajectory identities"
            )
        expected_trajectories = trajectories.repeat_interleave(
            len(EVALUATION_TIME_INDICES)
        )
        expected_times = torch.tensor(
            EVALUATION_TIME_INDICES,
            dtype=torch.long,
            device=self.X.device,
        ).repeat(expected_trajectory_count)
        if not torch.equal(self.trajectory_id, expected_trajectories) or not torch.equal(
            self.time_index,
            expected_times,
        ):
            raise ProbeExecutionInputError(
                "evaluation rows must retain canonical trajectory blocks and time order"
            )


def _copy_float_array(
    value: np.ndarray,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> torch.Tensor:
    return torch.as_tensor(
        np.array(value, copy=True),
        dtype=dtype,
        device=device,
    ).contiguous()


def _copy_index_array(
    value: np.ndarray,
    *,
    device: torch.device | str | None,
) -> torch.Tensor:
    return torch.as_tensor(
        np.array(value, copy=True),
        dtype=torch.long,
        device=device,
    ).contiguous()


def evaluation_rows_to_tensors(
    rows: ProbeEvaluationRows,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> LabelFreeEvaluationTensors:
    """Copy the label-free numpy row object without consulting another field."""

    if type(rows) is not ProbeEvaluationRows:
        raise ProbeExecutionInputError("rows must be an exact ProbeEvaluationRows object")
    if dtype != SOURCE_DTYPE:
        raise ProbeExecutionInputError(
            "public evaluation rows must first be tensorized in source float32"
        )
    return LabelFreeEvaluationTensors(
        source_indices=_copy_index_array(rows.source_indices, device=device),
        X=_copy_float_array(rows.X, dtype=dtype, device=device),
        trajectory_id=_copy_index_array(rows.trajectory_id, device=device),
        time_index=_copy_index_array(rows.time_index, device=device),
        time_value=_copy_float_array(rows.time_value, dtype=dtype, device=device),
    )


def promote_training_split_to_float64(
    train32: TensorConfirmatorySplit,
) -> TensorConfirmatorySplit:
    """Exactly promote the authorized source-fp32 training representation."""

    _validate_training_split(train32)
    if train32.X.dtype != SOURCE_DTYPE:
        raise ProbeExecutionInputError("training promotion requires a float32 source split")
    return TensorConfirmatorySplit(
        name=train32.name,
        source_indices=train32.source_indices.detach().clone().contiguous(),
        X=train32.X.to(dtype=torch.float64),
        value=train32.value.to(dtype=torch.float64),
        gradient=train32.gradient.to(dtype=torch.float64),
        trajectory_id=train32.trajectory_id.detach().clone().contiguous(),
        time_index=train32.time_index.detach().clone().contiguous(),
        time_value=train32.time_value.to(dtype=torch.float64),
    )


def promote_evaluation_to_float64(
    evaluation32: LabelFreeEvaluationTensors,
) -> LabelFreeEvaluationTensors:
    """Exactly promote copied public coordinates without adding label fields."""

    if type(evaluation32) is not LabelFreeEvaluationTensors:
        raise ProbeExecutionInputError(
            "evaluation promotion requires exact label-free tensors"
        )
    if evaluation32.X.dtype != SOURCE_DTYPE:
        raise ProbeExecutionInputError("evaluation promotion requires float32 source tensors")
    return LabelFreeEvaluationTensors(
        source_indices=evaluation32.source_indices.detach().clone().contiguous(),
        X=evaluation32.X.to(dtype=torch.float64),
        trajectory_id=evaluation32.trajectory_id.detach().clone().contiguous(),
        time_index=evaluation32.time_index.detach().clone().contiguous(),
        time_value=evaluation32.time_value.to(dtype=torch.float64),
    )


def promote_parameters_to_float64(
    parameters32: FrozenTERAParameters,
) -> FrozenTERAParameters:
    """Exactly promote the tensor part of a binary32-bound fit artifact."""

    if type(parameters32) is not FrozenTERAParameters:
        raise ProbeExecutionInputError("parameter promotion requires frozen TERA parameters")
    if parameters32.lengthscale.dtype != SOURCE_DTYPE:
        raise ProbeExecutionInputError("parameter promotion requires float32 lengthscale")
    for value, name in (
        (parameters32.outputscale, "outputscale"),
        (parameters32.sigma_f, "sigma_f"),
        (parameters32.sigma_g, "sigma_g"),
    ):
        if not _is_binary32_scalar(float(value)):
            raise ProbeExecutionInputError(f"{name} must retain an exact binary32 fit value")
    return FrozenTERAParameters(
        lengthscale=parameters32.lengthscale.to(dtype=torch.float64),
        outputscale=parameters32.outputscale,
        sigma_f=parameters32.sigma_f,
        sigma_g=parameters32.sigma_g,
        kernel=parameters32.kernel,
        gradient_noise_model=parameters32.gradient_noise_model,
    )


def _is_exact_source_promotion(value: torch.Tensor) -> bool:
    if value.dtype != torch.float64:
        return True
    return torch.equal(value, value.to(dtype=SOURCE_DTYPE).to(dtype=torch.float64))


def _is_binary32_scalar(value: float) -> bool:
    if not math.isfinite(value):
        return False
    represented = float(torch.tensor(value, dtype=SOURCE_DTYPE))
    return represented == value


def _validate_training_split(train: TensorConfirmatorySplit) -> None:
    if type(train) is not TensorConfirmatorySplit or train.name != "train":
        raise ProbeExecutionInputError("train must be an exact tensorized train split")
    if train.X.ndim != 2 or min(train.X.shape) == 0:
        raise ProbeExecutionInputError("training X must have nonzero shape (n, d)")
    count, dimension = train.X.shape
    expected_shapes = {
        "source_indices": (count,),
        "value": (count,),
        "gradient": (count, dimension),
        "trajectory_id": (count,),
        "time_index": (count,),
        "time_value": (count,),
    }
    for name, shape in expected_shapes.items():
        value = getattr(train, name)
        if type(value) is not torch.Tensor or value.shape != shape:
            raise ProbeExecutionInputError(f"training {name} must have shape {shape}")
        if value.device != train.X.device:
            raise ProbeExecutionInputError("all training tensors must share one device")
    if train.X.dtype not in {torch.float32, torch.float64}:
        raise ProbeExecutionInputError("training tensors must use float32 or float64")
    for value, name in (
        (train.source_indices, "source_indices"),
        (train.trajectory_id, "trajectory_id"),
        (train.time_index, "time_index"),
    ):
        if value.dtype != torch.long:
            raise ProbeExecutionInputError(f"training {name} must have dtype torch.long")
    for value, name in (
        (train.X, "X"),
        (train.value, "value"),
        (train.gradient, "gradient"),
        (train.time_value, "time_value"),
    ):
        if value.dtype != train.X.dtype or value.device != train.X.device:
            raise ProbeExecutionInputError(
                f"training {name} must match the training X dtype and device"
            )
        if not bool(torch.isfinite(value).all().item()):
            raise ProbeExecutionInputError(f"training {name} must contain only finite values")
    if bool((train.source_indices < 0).any().item()) or bool(
        (torch.diff(train.source_indices) <= 0).any().item()
    ):
        raise ProbeExecutionInputError(
            "training source_indices must be unique, increasing, and nonnegative"
        )
    if bool((train.trajectory_id < 0).any().item()) or bool(
        (train.time_index < 0).any().item()
    ):
        raise ProbeExecutionInputError("training trajectory/time identities must be nonnegative")


def _validate_parameters(
    parameters: FrozenTERAParameters,
    *,
    train: TensorConfirmatorySplit,
) -> None:
    if type(parameters) is not FrozenTERAParameters:
        raise ProbeExecutionInputError("parameters must be exact frozen TERA parameters")
    if parameters.kernel != "rbf" or parameters.gradient_noise_model != "iid":
        raise ProbeExecutionInputError("F02b probe arms require the frozen RBF/iid fit")
    if parameters.lengthscale.numel() != 1:
        raise ProbeExecutionInputError("F02b probe arms require one isotropic lengthscale")
    if parameters.lengthscale.dtype != train.X.dtype:
        raise ProbeExecutionInputError("lengthscale must match the arm compute dtype")
    if parameters.lengthscale.device != train.X.device:
        raise ProbeExecutionInputError("lengthscale must be on the arm compute device")
    for value, name in (
        (parameters.outputscale, "outputscale"),
        (parameters.sigma_f, "sigma_f"),
        (parameters.sigma_g, "sigma_g"),
    ):
        if not _is_binary32_scalar(float(value)):
            raise ProbeExecutionInputError(f"{name} must retain an exact binary32 fit value")
    if train.X.dtype == torch.float64 and not _is_exact_source_promotion(
        parameters.lengthscale
    ):
        raise ProbeExecutionInputError(
            "float64 lengthscale must be an exact promotion of its binary32 fit value"
        )


def _snapshot_training_split(train: TensorConfirmatorySplit) -> TensorConfirmatorySplit:
    return TensorConfirmatorySplit(
        name=train.name,
        source_indices=train.source_indices.detach().clone().contiguous(),
        X=train.X.detach().clone().contiguous(),
        value=train.value.detach().clone().contiguous(),
        gradient=train.gradient.detach().clone().contiguous(),
        trajectory_id=train.trajectory_id.detach().clone().contiguous(),
        time_index=train.time_index.detach().clone().contiguous(),
        time_value=train.time_value.detach().clone().contiguous(),
    )


def _snapshot_evaluation(
    evaluation: LabelFreeEvaluationTensors,
) -> LabelFreeEvaluationTensors:
    return LabelFreeEvaluationTensors(
        source_indices=evaluation.source_indices.detach().clone().contiguous(),
        X=evaluation.X.detach().clone().contiguous(),
        trajectory_id=evaluation.trajectory_id.detach().clone().contiguous(),
        time_index=evaluation.time_index.detach().clone().contiguous(),
        time_value=evaluation.time_value.detach().clone().contiguous(),
    )


def _snapshot_parameters(parameters: FrozenTERAParameters) -> FrozenTERAParameters:
    return FrozenTERAParameters(
        lengthscale=parameters.lengthscale.detach().clone().contiguous(),
        outputscale=parameters.outputscale,
        sigma_f=parameters.sigma_f,
        sigma_g=parameters.sigma_g,
        kernel=parameters.kernel,
        gradient_noise_model=parameters.gradient_noise_model,
    )


def _snapshot_neighbours(neighbours: FixedNeighbourRows) -> FixedNeighbourRows:
    return FixedNeighbourRows(
        positions=neighbours.positions.detach().clone().contiguous(),
        source_indices=neighbours.source_indices.detach().clone().contiguous(),
        m=neighbours.m,
    )


def _update_hash_bytes(hasher: Any, label: str, payload: bytes) -> None:
    label_bytes = label.encode("utf-8")
    hasher.update(len(label_bytes).to_bytes(8, "big"))
    hasher.update(label_bytes)
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def _update_hash_tensor(hasher: Any, label: str, value: torch.Tensor) -> None:
    detached = value.detach().cpu().contiguous()
    metadata = json.dumps(
        {"dtype": str(detached.dtype), "shape": list(detached.shape)},
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    _update_hash_bytes(hasher, f"{label}.metadata", metadata)
    _update_hash_bytes(hasher, f"{label}.values", detached.numpy().tobytes(order="C"))


def _source_arm_binding_sha256(
    train: TensorConfirmatorySplit,
    evaluation: LabelFreeEvaluationTensors,
    parameters: FrozenTERAParameters,
    neighbours: FixedNeighbourRows,
    work_plan: ProbeWorkPlan,
) -> str:
    if train.X.dtype != SOURCE_DTYPE or evaluation.X.dtype != SOURCE_DTYPE:
        raise ProbeExecutionInputError("source arm binding requires float32 tensors")
    hasher = hashlib.sha256()
    _update_hash_bytes(hasher, "domain", b"auto_dgp2.f02b.registered_orbit_source_arm_v1")
    for value, label in (
        (train.source_indices, "train.source_indices"),
        (train.X, "train.X"),
        (train.value, "train.value"),
        (train.gradient, "train.gradient"),
        (train.trajectory_id, "train.trajectory_id"),
        (train.time_index, "train.time_index"),
        (train.time_value, "train.time_value"),
        (evaluation.source_indices, "evaluation.source_indices"),
        (evaluation.X, "evaluation.X"),
        (evaluation.trajectory_id, "evaluation.trajectory_id"),
        (evaluation.time_index, "evaluation.time_index"),
        (evaluation.time_value, "evaluation.time_value"),
        (parameters.lengthscale, "parameters.lengthscale"),
        (neighbours.positions, "neighbours.positions"),
        (neighbours.source_indices, "neighbours.source_indices"),
    ):
        _update_hash_tensor(hasher, label, value)
    scalar_record = {
        "gradient_noise_model": parameters.gradient_noise_model,
        "kernel": parameters.kernel,
        "m": neighbours.m,
        "outputscale_hex": float(parameters.outputscale).hex(),
        "sigma_f_hex": float(parameters.sigma_f).hex(),
        "sigma_g_hex": float(parameters.sigma_g).hex(),
        "work_plan": work_plan.as_record(),
    }
    _update_hash_bytes(
        hasher,
        "scalar_record",
        json.dumps(
            scalar_record,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8"),
    )
    return hasher.hexdigest()


def _float64_arm_source_binding_sha256(
    train64: TensorConfirmatorySplit,
    evaluation64: LabelFreeEvaluationTensors,
    parameters64: FrozenTERAParameters,
    neighbours: FixedNeighbourRows,
    work_plan: ProbeWorkPlan,
) -> str:
    """Reconstruct and hash the unique binary32 source of an exact promotion."""

    source_train = TensorConfirmatorySplit(
        name=train64.name,
        source_indices=train64.source_indices,
        X=train64.X.to(dtype=SOURCE_DTYPE),
        value=train64.value.to(dtype=SOURCE_DTYPE),
        gradient=train64.gradient.to(dtype=SOURCE_DTYPE),
        trajectory_id=train64.trajectory_id,
        time_index=train64.time_index,
        time_value=train64.time_value.to(dtype=SOURCE_DTYPE),
    )
    source_evaluation = LabelFreeEvaluationTensors(
        source_indices=evaluation64.source_indices,
        X=evaluation64.X.to(dtype=SOURCE_DTYPE),
        trajectory_id=evaluation64.trajectory_id,
        time_index=evaluation64.time_index,
        time_value=evaluation64.time_value.to(dtype=SOURCE_DTYPE),
    )
    source_parameters = FrozenTERAParameters(
        lengthscale=parameters64.lengthscale.to(dtype=SOURCE_DTYPE),
        outputscale=parameters64.outputscale,
        sigma_f=parameters64.sigma_f,
        sigma_g=parameters64.sigma_g,
        kernel=parameters64.kernel,
        gradient_noise_model=parameters64.gradient_noise_model,
    )
    return _source_arm_binding_sha256(
        source_train,
        source_evaluation,
        source_parameters,
        neighbours,
        work_plan,
    )


def _arm_tensors(arm: RegisteredOrbitArmInputs) -> tuple[torch.Tensor, ...]:
    return (
        arm.train.source_indices,
        arm.train.X,
        arm.train.value,
        arm.train.gradient,
        arm.train.trajectory_id,
        arm.train.time_index,
        arm.train.time_value,
        arm.evaluation.source_indices,
        arm.evaluation.X,
        arm.evaluation.trajectory_id,
        arm.evaluation.time_index,
        arm.evaluation.time_value,
        arm.parameters.lengthscale,
        arm.fixed_neighbours.positions,
        arm.fixed_neighbours.source_indices,
    )


@dataclass(frozen=True, slots=True)
class RegisteredOrbitArmInputs:
    """Privately snapshotted numerical inputs bound to one source-fp32 arm.

    Construction is restricted to :func:`build_source_orbit_arm_inputs` and
    :func:`promote_registered_orbit_arm_to_float64`.  This binding is still not
    corpus/catalog authorization; the future runner must establish that first.
    """

    train: TensorConfirmatorySplit
    evaluation: LabelFreeEvaluationTensors
    parameters: FrozenTERAParameters
    fixed_neighbours: FixedNeighbourRows
    work_plan: ProbeWorkPlan
    source_arm_binding_sha256: str
    binding_kind: str
    _construction_token: object = field(repr=False, compare=False)
    _tensor_version_snapshot: tuple[int, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._construction_token is not _ARM_CONSTRUCTION_TOKEN:
            raise ProbeExecutionInputError(
                "RegisteredOrbitArmInputs must be created by its audited factory"
            )
        _validate_training_split(self.train)
        if type(self.evaluation) is not LabelFreeEvaluationTensors:
            raise ProbeExecutionInputError(
                "evaluation must be an exact label-free tensor object"
            )
        if self.evaluation.X.dtype != self.train.X.dtype or (
            self.evaluation.X.device != self.train.X.device
        ):
            raise ProbeExecutionInputError(
                "training and evaluation coordinates must share dtype and device"
            )
        if self.evaluation.X.shape[1] != self.train.X.shape[1]:
            raise ProbeExecutionInputError("training and evaluation dimensions must match")
        if bool(
            torch.isin(self.evaluation.source_indices, self.train.source_indices).any().item()
        ):
            raise ProbeExecutionInputError("training and evaluation source rows must be disjoint")

        if type(self.work_plan) is not ProbeWorkPlan:
            raise ProbeExecutionInputError("work_plan must be an exact ProbeWorkPlan")
        try:
            authoritative_plan = build_probe_work_plan(
                probe_task_for_index(self.work_plan.task_index)
            )
        except (IndexError, ValueError) as error:
            raise ProbeExecutionInputError("work_plan task is outside the frozen grid") from error
        if self.work_plan != authoritative_plan:
            raise ProbeExecutionInputError("work_plan does not match the frozen grid")
        if self.train.X.shape[1] != self.work_plan.dimension:
            raise ProbeExecutionInputError("tensor dimension does not match the work plan")

        neighbours = self.fixed_neighbours
        if type(neighbours) is not FixedNeighbourRows:
            raise ProbeExecutionInputError("fixed_neighbours must be an exact core object")
        expected_shape = (PRIMARY_EVALUATION_ROW_COUNT, self.work_plan.production_m)
        if neighbours.m != self.work_plan.production_m:
            raise ProbeExecutionInputError("fixed-neighbour m does not match production m")
        if neighbours.positions.shape != expected_shape or (
            neighbours.source_indices.shape != expected_shape
        ):
            raise ProbeExecutionInputError(
                f"fixed-neighbour tensors must have shape {expected_shape}"
            )
        for value, name in (
            (neighbours.positions, "positions"),
            (neighbours.source_indices, "source_indices"),
        ):
            if value.dtype != torch.long or value.device != self.train.X.device:
                raise ProbeExecutionInputError(
                    f"fixed-neighbour {name} must be torch.long on the arm device"
                )
        if bool(((neighbours.positions < 0) | (neighbours.positions >= self.train.X.shape[0])).any()):
            raise ProbeExecutionInputError("fixed-neighbour positions are out of range")
        ordered = torch.sort(neighbours.positions, dim=1).values
        if bool((ordered[:, 1:] == ordered[:, :-1]).any().item()):
            raise ProbeExecutionInputError("each fixed-neighbour row must be unique")
        expected_sources = self.train.source_indices[neighbours.positions]
        if not torch.equal(expected_sources, neighbours.source_indices):
            raise ProbeExecutionInputError(
                "fixed-neighbour source identities do not match training positions"
            )

        _validate_parameters(self.parameters, train=self.train)
        if self.train.X.dtype == torch.float64:
            promoted_tensors = (
                (self.train.X, "train.X"),
                (self.train.value, "train.value"),
                (self.train.gradient, "train.gradient"),
                (self.train.time_value, "train.time_value"),
                (self.evaluation.X, "evaluation.X"),
                (self.evaluation.time_value, "evaluation.time_value"),
            )
            for value, name in promoted_tensors:
                if not _is_exact_source_promotion(value):
                    raise ProbeExecutionInputError(
                        f"{name} must be an exact binary32-to-binary64 promotion"
                    )
            if self.binding_kind != "exact_promotion_of_bound_source_fp32_arm":
                raise ProbeExecutionInputError("float64 arm lacks source-fp32 derivation")
            observed_binding = _float64_arm_source_binding_sha256(
                self.train,
                self.evaluation,
                self.parameters,
                self.fixed_neighbours,
                self.work_plan,
            )
            if observed_binding != self.source_arm_binding_sha256:
                raise ProbeExecutionInputError(
                    "float64 arm does not match its bound source-fp32 SHA-256"
                )
        elif self.binding_kind != "bound_source_fp32_arm":
            raise ProbeExecutionInputError("float32 arm has an invalid binding kind")
        if (
            type(self.source_arm_binding_sha256) is not str
            or len(self.source_arm_binding_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_arm_binding_sha256)
        ):
            raise ProbeExecutionInputError("source arm binding must be a lowercase SHA-256")
        if self.train.X.dtype == SOURCE_DTYPE:
            observed_binding = _source_arm_binding_sha256(
                self.train,
                self.evaluation,
                self.parameters,
                self.fixed_neighbours,
                self.work_plan,
            )
            if observed_binding != self.source_arm_binding_sha256:
                raise ProbeExecutionInputError("source arm binding SHA-256 is mismatched")
        object.__setattr__(
            self,
            "_tensor_version_snapshot",
            tuple(value._version for value in _arm_tensors(self)),
        )

    def assert_unchanged(self) -> None:
        """Revalidate versions and the full source-content binding."""

        observed_versions = tuple(value._version for value in _arm_tensors(self))
        if observed_versions != self._tensor_version_snapshot:
            raise ProbeExecutionInputError(
                "registered arm tensors changed after their private snapshot"
            )
        if self.train.X.dtype == SOURCE_DTYPE:
            observed_binding = _source_arm_binding_sha256(
                self.train,
                self.evaluation,
                self.parameters,
                self.fixed_neighbours,
                self.work_plan,
            )
        else:
            observed_binding = _float64_arm_source_binding_sha256(
                self.train,
                self.evaluation,
                self.parameters,
                self.fixed_neighbours,
                self.work_plan,
            )
        if observed_binding != self.source_arm_binding_sha256:
            raise ProbeExecutionInputError(
                "registered arm content changed after its source binding"
            )


def build_source_orbit_arm_inputs(
    train32: TensorConfirmatorySplit,
    evaluation32: LabelFreeEvaluationTensors,
    parameters32: FrozenTERAParameters,
    work_plan: ProbeWorkPlan,
) -> RegisteredOrbitArmInputs:
    """Snapshot a source-fp32 arm and recompute its pinned vendor neighbours."""

    if type(train32) is not TensorConfirmatorySplit:
        raise ProbeExecutionInputError("train32 must be an exact tensorized train split")
    if type(evaluation32) is not LabelFreeEvaluationTensors:
        raise ProbeExecutionInputError(
            "evaluation32 must be exact label-free tensors"
        )
    if type(parameters32) is not FrozenTERAParameters:
        raise ProbeExecutionInputError("parameters32 must be exact frozen TERA parameters")
    train = _snapshot_training_split(train32)
    evaluation = _snapshot_evaluation(evaluation32)
    parameters = _snapshot_parameters(parameters32)
    _validate_training_split(train)
    if train.X.dtype != SOURCE_DTYPE or evaluation.X.dtype != SOURCE_DTYPE:
        raise ProbeExecutionInputError("source ORBIT arm requires float32 tensors")
    _validate_parameters(parameters, train=train)
    if type(work_plan) is not ProbeWorkPlan:
        raise ProbeExecutionInputError("work_plan must be an exact ProbeWorkPlan")
    try:
        authoritative = build_probe_work_plan(probe_task_for_index(work_plan.task_index))
    except (IndexError, ValueError) as error:
        raise ProbeExecutionInputError("work_plan task is outside the frozen grid") from error
    if work_plan != authoritative:
        raise ProbeExecutionInputError("work_plan does not match the frozen grid")
    neighbours = fixed_fp32_neighbours(
        train.X,
        evaluation.X,
        parameters.lengthscale,
        train.source_indices,
        evaluation.source_indices,
        work_plan.production_m,
    )
    binding = _source_arm_binding_sha256(
        train,
        evaluation,
        parameters,
        neighbours,
        work_plan,
    )
    return RegisteredOrbitArmInputs(
        train=train,
        evaluation=evaluation,
        parameters=parameters,
        fixed_neighbours=neighbours,
        work_plan=work_plan,
        source_arm_binding_sha256=binding,
        binding_kind="bound_source_fp32_arm",
        _construction_token=_ARM_CONSTRUCTION_TOKEN,
    )


def promote_registered_orbit_arm_to_float64(
    source_arm: RegisteredOrbitArmInputs,
) -> RegisteredOrbitArmInputs:
    """Derive the only permitted fp64 arm from one bound source-fp32 arm."""

    if type(source_arm) is not RegisteredOrbitArmInputs:
        raise ProbeExecutionInputError("source_arm must be registered ORBIT inputs")
    source_arm.assert_unchanged()
    if source_arm.train.X.dtype != SOURCE_DTYPE:
        raise ProbeExecutionInputError("only a source-fp32 arm may be promoted")
    return RegisteredOrbitArmInputs(
        train=promote_training_split_to_float64(source_arm.train),
        evaluation=promote_evaluation_to_float64(source_arm.evaluation),
        parameters=promote_parameters_to_float64(source_arm.parameters),
        fixed_neighbours=_snapshot_neighbours(source_arm.fixed_neighbours),
        work_plan=source_arm.work_plan,
        source_arm_binding_sha256=source_arm.source_arm_binding_sha256,
        binding_kind="exact_promotion_of_bound_source_fp32_arm",
        _construction_token=_ARM_CONSTRUCTION_TOKEN,
    )


def registered_orbit_tolerances(
    dtype: torch.dtype,
    *,
    include_stratum_sweep: bool,
) -> tuple[float, ...]:
    """Return the sole registered tolerance sequence for one arm/target role."""

    if dtype not in {torch.float32, torch.float64}:
        raise ProbeExecutionInputError("ORBIT arm dtype must be float32 or float64")
    if type(include_stratum_sweep) is not bool:
        raise ProbeExecutionInputError("include_stratum_sweep must be bool")
    if not include_stratum_sweep:
        return (PRODUCTION_TOLERANCE,)
    if dtype == torch.float32:
        return SHARED_TOLERANCE_SWEEP
    return SHARED_TOLERANCE_SWEEP + FP64_ONLY_TOLERANCE_SWEEP


@dataclass(frozen=True, slots=True)
class RegisteredSourceGeometry:
    """One label-free source-fp32 SVD frozen before strata selection."""

    task_index: int
    source_arm_binding_sha256: str
    source_rank_reference_sha256: str
    target_position: int
    target_source_index: int
    neighbour_positions: torch.Tensor
    neighbour_source_indices: torch.Tensor
    standardized_differences: torch.Tensor
    geometry: LocalGeometry
    rank_boundary: dict[str, Any]
    _construction_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._construction_token is not _SOURCE_GEOMETRY_CONSTRUCTION_TOKEN:
            raise ProbeExecutionInputError(
                "RegisteredSourceGeometry must be created by its audited scan"
            )


@dataclass(frozen=True, slots=True)
class RegisteredOrbitStrata:
    """The deterministic strata selected from all 100 source geometry rows."""

    task_index: int
    source_arm_binding_sha256: str
    source_rank_reference_sha256: tuple[str, ...]
    source_rank_grid_sha256: str
    selected_target_positions: tuple[int, ...]
    selection_record: dict[str, Any]
    selection_sha256: str
    _construction_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._construction_token is not _STRATA_CONSTRUCTION_TOKEN:
            raise ProbeExecutionInputError(
                "RegisteredOrbitStrata must be created by its audited selector"
            )

    def includes(self, target_position: int) -> bool:
        """Return whether a target receives the registered full tolerance sweep."""

        return target_position in self.selected_target_positions


@dataclass(frozen=True, slots=True)
class OrbitSolveEvidence:
    """One solve plus an immediate source-dtype replay of its final action."""

    requested_tolerance: float
    prediction: LocalPrediction
    verified_operator_action: torch.Tensor
    verified_residual: torch.Tensor
    replayed_operator_action: torch.Tensor
    replayed_residual: torch.Tensor
    replay_consistency: dict[str, float]
    matrix_free_error: dict[str, Any] | None
    matrix_free_error_unavailable_reason: str | None
    operator_action_provenance: str
    residual_provenance: str
    replay_action_provenance: str
    operator_norm_upper_bound_provenance: str
    roundoff_qualification: str
    diagnostic_operator_matvecs: int


@dataclass(frozen=True, slots=True)
class OrbitTargetExecution:
    """One target's single built system and registered independent solves."""

    task_index: int
    source_arm_binding_sha256: str
    source_rank_reference_sha256: str
    source_rank_grid_sha256: str
    strata_selection_sha256: str
    target_position: int
    target_source_index: int
    neighbour_positions: torch.Tensor
    neighbour_source_indices: torch.Tensor
    compute_dtype: torch.dtype
    system: LocalValueSystem
    rank_boundary: dict[str, Any]
    function_cholesky_error: dict[str, Any]
    solves: tuple[OrbitSolveEvidence, ...]
    production_tolerance: float
    include_stratum_sweep: bool

    @property
    def production_solve(self) -> OrbitSolveEvidence:
        """Return the existing sweep object; never issue a duplicate solve."""

        matches = [
            evidence
            for evidence in self.solves
            if evidence.requested_tolerance == self.production_tolerance
        ]
        if len(matches) != 1:  # pragma: no cover - construction invariant
            raise ProbeExecutionEvidenceError(
                "execution does not contain exactly one production-tolerance solve"
            )
        return matches[0]


def _validate_target_position(value: int, count: int) -> int:
    if type(value) is not int or not 0 <= value < count:
        raise ProbeExecutionInputError(f"target_position must lie in [0, {count})")
    return value


def _direct_svd_rank_record(
    geometry: LocalGeometry,
    *,
    expected_rank: int,
    source_fp32_cutoff: float,
    source_fp32_selected_rank: int,
) -> dict[str, Any]:
    if (
        geometry.singular_values is None
        or geometry.operational_singular_value_cutoff is None
        or geometry.native_singular_value_cutoff is None
        or geometry.operational_cutoff_source is None
    ):
        raise ProbeExecutionEvidenceError("ORBIT system lacks direct-SVD rank evidence")
    represented_source_cutoff = geometry.operational_singular_value_cutoff.new_tensor(
        source_fp32_cutoff
    )
    if not torch.equal(
        geometry.operational_singular_value_cutoff.reshape(()),
        represented_source_cutoff,
    ):
        raise ProbeExecutionEvidenceError("ORBIT system did not use the source-fp32 cutoff")
    result = rank_boundary_metrics(
        geometry.singular_values,
        cutoff=geometry.operational_singular_value_cutoff,
        expected_rank=expected_rank,
    )
    if result["strict_selected_rank"] != geometry.rank:
        raise ProbeExecutionEvidenceError(
            "rank record does not match the geometry used by the solver"
        )
    result.update(
        {
            "rank_evidence_source": "same_direct_svd_used_by_orbit_system",
            "rank_epsilon_source_dtype": "float32",
            "rank_epsilon": SOURCE_RANK_EPSILON,
            "source_fp32_operational_cutoff": source_fp32_cutoff,
            "source_fp32_strict_selected_rank": source_fp32_selected_rank,
            "operational_cutoff_is_source_fp32_bound": True,
            "operational_cutoff_source": geometry.operational_cutoff_source,
            "compute_singular_value_dtype": str(
                geometry.singular_values.dtype
            ).removeprefix("torch."),
            "native_compute_cutoff": float(geometry.native_singular_value_cutoff),
            "native_compute_strict_selected_rank": int(
                (
                    geometry.singular_values
                    > geometry.native_singular_value_cutoff
                ).sum().item()
            ),
            "geometry_basis_is_exact_at_native_cutoff": geometry.is_exact,
        }
    )
    result["native_compute_rank_matches_expected"] = (
        result["native_compute_strict_selected_rank"] == expected_rank
    )
    result["operational_minus_native_selected_rank"] = (
        result["strict_selected_rank"]
        - result["native_compute_strict_selected_rank"]
    )
    return result


def _source_rank_reference_sha256(
    *,
    source_arm_binding_sha256: str,
    task_index: int,
    target_position: int,
    target_source_index: int,
    neighbour_positions: torch.Tensor,
    neighbour_source_indices: torch.Tensor,
    standardized_differences: torch.Tensor,
    geometry: LocalGeometry,
    rank_boundary: dict[str, Any],
) -> str:
    if (
        geometry.singular_values is None
        or geometry.singular_values.dtype != SOURCE_DTYPE
        or geometry.operational_singular_value_cutoff is None
        or geometry.rank_epsilon_used is None
    ):
        raise ProbeExecutionEvidenceError(
            "source rank binding requires complete source-fp32 SVD evidence"
        )
    hasher = hashlib.sha256()
    scalar_record = {
        "domain": "auto_dgp2.f02b.orbit_source_rank_reference_v2",
        "geometry_is_exact": geometry.is_exact,
        "operational_cutoff_source": geometry.operational_cutoff_source,
        "rank": geometry.rank,
        "rank_epsilon_hex": float(geometry.rank_epsilon_used).hex(),
        "source_arm_binding_sha256": source_arm_binding_sha256,
        "source_cutoff_hex": float(
            geometry.operational_singular_value_cutoff
        ).hex(),
        "target_position": target_position,
        "target_source_index": target_source_index,
        "task_index": task_index,
    }
    _update_hash_bytes(
        hasher,
        "scalar_record",
        json.dumps(
            scalar_record,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8"),
    )
    _update_hash_tensor(hasher, "neighbour_positions", neighbour_positions)
    _update_hash_tensor(hasher, "neighbour_source_indices", neighbour_source_indices)
    _update_hash_tensor(
        hasher,
        "standardized_differences",
        standardized_differences,
    )
    _update_hash_tensor(hasher, "geometry.coordinates", geometry.coordinates)
    _update_hash_tensor(hasher, "geometry.q_to_z", geometry.q_to_z)
    _update_hash_tensor(hasher, "geometry.eigenvalues", geometry.eigenvalues)
    _update_hash_tensor(
        hasher,
        "geometry.discarded_eigenvalue_sum",
        geometry.discarded_eigenvalue_sum,
    )
    _update_hash_tensor(hasher, "source_singular_values", geometry.singular_values)
    _update_hash_tensor(
        hasher,
        "geometry.operational_singular_value_cutoff",
        geometry.operational_singular_value_cutoff,
    )
    if geometry.native_singular_value_cutoff is None:
        raise ProbeExecutionEvidenceError("source geometry lacks its native cutoff")
    _update_hash_tensor(
        hasher,
        "geometry.native_singular_value_cutoff",
        geometry.native_singular_value_cutoff,
    )
    _update_hash_tensor(
        hasher,
        "geometry.rank_epsilon_used",
        geometry.rank_epsilon_used,
    )
    _update_hash_bytes(
        hasher,
        "rank_boundary",
        json.dumps(
            rank_boundary,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8"),
    )
    return hasher.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def scan_registered_source_geometry(
    source_arm: RegisteredOrbitArmInputs,
    target_position: int,
) -> RegisteredSourceGeometry:
    """Run the sole source-fp32 SVD for one target without reading labels."""

    if type(source_arm) is not RegisteredOrbitArmInputs:
        raise ProbeExecutionInputError("source_arm must be registered ORBIT inputs")
    source_arm.assert_unchanged()
    if source_arm.train.X.dtype != SOURCE_DTYPE:
        raise ProbeExecutionInputError("geometry scan requires the bound source-fp32 arm")
    position = _validate_target_position(
        target_position,
        source_arm.evaluation.X.shape[0],
    )
    neighbour_positions = source_arm.fixed_neighbours.positions[position]
    neighbour_sources = source_arm.fixed_neighbours.source_indices[position]
    target = source_arm.evaluation.X[position].unsqueeze(0)
    with torch.inference_mode():
        standardized_differences = (
            source_arm.train.X[neighbour_positions] - target
        ).T.contiguous()
        lengthscale = source_arm.parameters.lengthscale.reshape(1, 1)
        scaled_differences = standardized_differences / lengthscale
        geometry = build_local_geometry_from_differences(
            scaled_differences,
            rank_epsilon=SOURCE_RANK_EPSILON,
        )
        if geometry.operational_singular_value_cutoff is None:
            raise ProbeExecutionEvidenceError("source geometry lacks its rank cutoff")
        cutoff = float(geometry.operational_singular_value_cutoff)
        rank_record = _direct_svd_rank_record(
            geometry,
            expected_rank=source_arm.work_plan.physical_rank,
            source_fp32_cutoff=cutoff,
            source_fp32_selected_rank=geometry.rank,
        )
        target_source_index = int(source_arm.evaluation.source_indices[position].item())
        rank_record["target_source_index"] = target_source_index
        reference_sha256 = _source_rank_reference_sha256(
            source_arm_binding_sha256=source_arm.source_arm_binding_sha256,
            task_index=source_arm.work_plan.task_index,
            target_position=position,
            target_source_index=target_source_index,
            neighbour_positions=neighbour_positions,
            neighbour_source_indices=neighbour_sources,
            standardized_differences=standardized_differences,
            geometry=geometry,
            rank_boundary=rank_record,
        )
    source_arm.assert_unchanged()
    return RegisteredSourceGeometry(
        task_index=source_arm.work_plan.task_index,
        source_arm_binding_sha256=source_arm.source_arm_binding_sha256,
        source_rank_reference_sha256=reference_sha256,
        target_position=position,
        target_source_index=target_source_index,
        neighbour_positions=neighbour_positions.detach().clone().contiguous(),
        neighbour_source_indices=neighbour_sources.detach().clone().contiguous(),
        standardized_differences=(
            standardized_differences.detach().clone().contiguous()
        ),
        geometry=geometry,
        rank_boundary=rank_record,
        _construction_token=_SOURCE_GEOMETRY_CONSTRUCTION_TOKEN,
    )


def _validate_source_geometry_reference(
    arm: RegisteredOrbitArmInputs,
    source_geometry: RegisteredSourceGeometry,
    *,
    target_position: int,
) -> tuple[float, int, str]:
    if type(source_geometry) is not RegisteredSourceGeometry:
        raise ProbeExecutionInputError(
            "execution requires a registered source geometry reference"
        )
    expected_source_index = int(arm.evaluation.source_indices[target_position].item())
    expected_positions = arm.fixed_neighbours.positions[target_position]
    expected_sources = arm.fixed_neighbours.source_indices[target_position]
    if (
        source_geometry.task_index != arm.work_plan.task_index
        or source_geometry.source_arm_binding_sha256 != arm.source_arm_binding_sha256
        or source_geometry.target_position != target_position
        or source_geometry.target_source_index != expected_source_index
        or not torch.equal(source_geometry.neighbour_positions, expected_positions)
        or not torch.equal(source_geometry.neighbour_source_indices, expected_sources)
    ):
        raise ProbeExecutionInputError(
            "source geometry target or fixed-neighbour identity is mismatched"
        )
    expected_differences = (
        arm.train.X[expected_positions].to(dtype=SOURCE_DTYPE)
        - arm.evaluation.X[target_position].to(dtype=SOURCE_DTYPE).unsqueeze(0)
    ).T.contiguous()
    if not torch.equal(source_geometry.standardized_differences, expected_differences):
        raise ProbeExecutionEvidenceError(
            "source geometry standardized differences are mismatched"
        )
    geometry = source_geometry.geometry
    if (
        geometry.operational_singular_value_cutoff is None
        or geometry.singular_values is None
        or geometry.rank_epsilon_used is None
        or geometry.singular_values.dtype != SOURCE_DTYPE
    ):
        raise ProbeExecutionEvidenceError("source geometry lacks fp32 SVD evidence")
    expected_epsilon = geometry.rank_epsilon_used.new_tensor(SOURCE_RANK_EPSILON)
    if not torch.equal(geometry.rank_epsilon_used.reshape(()), expected_epsilon):
        raise ProbeExecutionEvidenceError("source geometry used the wrong rank epsilon")
    cutoff = float(geometry.operational_singular_value_cutoff)
    if not math.isfinite(cutoff) or cutoff < 0.0:
        raise ProbeExecutionEvidenceError("source geometry cutoff is invalid")
    observed_rank_record = _direct_svd_rank_record(
        geometry,
        expected_rank=arm.work_plan.physical_rank,
        source_fp32_cutoff=cutoff,
        source_fp32_selected_rank=geometry.rank,
    )
    observed_rank_record["target_source_index"] = expected_source_index
    if _canonical_json_sha256(observed_rank_record) != _canonical_json_sha256(
        source_geometry.rank_boundary
    ):
        raise ProbeExecutionEvidenceError("source geometry rank record is mismatched")
    observed_reference_sha256 = _source_rank_reference_sha256(
        source_arm_binding_sha256=source_geometry.source_arm_binding_sha256,
        task_index=source_geometry.task_index,
        target_position=source_geometry.target_position,
        target_source_index=source_geometry.target_source_index,
        neighbour_positions=source_geometry.neighbour_positions,
        neighbour_source_indices=source_geometry.neighbour_source_indices,
        standardized_differences=source_geometry.standardized_differences,
        geometry=geometry,
        rank_boundary=observed_rank_record,
    )
    if observed_reference_sha256 != source_geometry.source_rank_reference_sha256:
        raise ProbeExecutionEvidenceError("source geometry reference SHA-256 is mismatched")
    return cutoff, geometry.rank, observed_reference_sha256


def select_registered_orbit_strata(
    source_arm: RegisteredOrbitArmInputs,
    source_geometries: tuple[RegisteredSourceGeometry, ...],
) -> RegisteredOrbitStrata:
    """Bind worst/median/best selection to the complete ordered N0 population."""

    if type(source_arm) is not RegisteredOrbitArmInputs:
        raise ProbeExecutionInputError("source_arm must be registered ORBIT inputs")
    source_arm.assert_unchanged()
    if source_arm.train.X.dtype != SOURCE_DTYPE:
        raise ProbeExecutionInputError("strata selection requires the source-fp32 arm")
    if type(source_geometries) is not tuple or len(source_geometries) != (
        PRIMARY_EVALUATION_ROW_COUNT
    ):
        raise ProbeExecutionInputError(
            "strata selection requires the ordered complete 100-target geometry tuple"
        )
    rank_hashes: list[str] = []
    rank_records: list[dict[str, Any]] = []
    source_to_position: dict[int, int] = {}
    for position, geometry in enumerate(source_geometries):
        if geometry.target_position != position:
            raise ProbeExecutionInputError("source geometry tuple is out of target order")
        _, _, reference_sha256 = _validate_source_geometry_reference(
            source_arm,
            geometry,
            target_position=position,
        )
        rank_hashes.append(reference_sha256)
        rank_records.append(geometry.rank_boundary)
        source_to_position[geometry.target_source_index] = position
    selection_record = select_geometry_strata(
        rank_records,
        count=source_arm.work_plan.support_target_count,
    )
    selected_positions = tuple(
        source_to_position[int(record["target_source_index"])]
        for record in selection_record["selected"]
    )
    selection_payload = {
        "domain": "auto_dgp2.f02b.registered_orbit_strata_v2",
        "selection_record": selection_record,
        "selected_target_positions": list(selected_positions),
        "source_arm_binding_sha256": source_arm.source_arm_binding_sha256,
        "source_rank_grid_sha256": _canonical_json_sha256(rank_hashes),
        "source_rank_reference_sha256": rank_hashes,
        "task_index": source_arm.work_plan.task_index,
    }
    source_arm.assert_unchanged()
    return RegisteredOrbitStrata(
        task_index=source_arm.work_plan.task_index,
        source_arm_binding_sha256=source_arm.source_arm_binding_sha256,
        source_rank_reference_sha256=tuple(rank_hashes),
        source_rank_grid_sha256=selection_payload["source_rank_grid_sha256"],
        selected_target_positions=selected_positions,
        selection_record=json.loads(
            json.dumps(selection_record, allow_nan=False, sort_keys=True)
        ),
        selection_sha256=_canonical_json_sha256(selection_payload),
        _construction_token=_STRATA_CONSTRUCTION_TOKEN,
    )


def _validate_registered_strata(
    arm: RegisteredOrbitArmInputs,
    source_geometry: RegisteredSourceGeometry,
    strata: RegisteredOrbitStrata,
) -> bool:
    if type(strata) is not RegisteredOrbitStrata:
        raise ProbeExecutionInputError("execution requires registered geometry strata")
    if (
        strata.task_index != arm.work_plan.task_index
        or strata.source_arm_binding_sha256 != arm.source_arm_binding_sha256
        or len(strata.source_rank_reference_sha256) != PRIMARY_EVALUATION_ROW_COUNT
        or strata.source_rank_reference_sha256[source_geometry.target_position]
        != source_geometry.source_rank_reference_sha256
    ):
        raise ProbeExecutionInputError("registered strata belong to a different geometry grid")
    observed_grid_sha256 = _canonical_json_sha256(
        list(strata.source_rank_reference_sha256)
    )
    if observed_grid_sha256 != strata.source_rank_grid_sha256:
        raise ProbeExecutionEvidenceError("registered rank-grid SHA-256 is mismatched")
    payload = {
        "domain": "auto_dgp2.f02b.registered_orbit_strata_v2",
        "selection_record": strata.selection_record,
        "selected_target_positions": list(strata.selected_target_positions),
        "source_arm_binding_sha256": strata.source_arm_binding_sha256,
        "source_rank_grid_sha256": strata.source_rank_grid_sha256,
        "source_rank_reference_sha256": list(strata.source_rank_reference_sha256),
        "task_index": strata.task_index,
    }
    if _canonical_json_sha256(payload) != strata.selection_sha256:
        raise ProbeExecutionEvidenceError("registered strata SHA-256 is mismatched")
    return strata.includes(source_geometry.target_position)


def _verified_solve_evidence(
    system: LocalValueSystem,
    prediction: LocalPrediction,
    *,
    tolerance: float,
    max_iterations: int,
) -> OrbitSolveEvidence:
    solve = prediction.solve
    if solve.requested_tolerance != tolerance or solve.max_iterations != max_iterations:
        raise ProbeExecutionEvidenceError("solver result is not bound to its registered request")
    if not solve.residual_is_fresh:
        raise ProbeExecutionEvidenceError("ORBIT solve did not return a fresh residual")
    if not bool(torch.isfinite(solve.solution).all().item()):
        raise ProbeExecutionEvidenceError("ORBIT solution contains a nonfinite value")

    if system.geometry.rank == 0:
        action = system.conditional_cross.clone()
        residual = system.conditional_cross.clone()
        if not torch.equal(action, solve.operator_action) or not torch.equal(
            residual,
            solve.residual,
        ):
            raise ProbeExecutionEvidenceError("rank-zero solve evidence is inconsistent")
        return OrbitSolveEvidence(
            requested_tolerance=tolerance,
            prediction=prediction,
            verified_operator_action=action,
            verified_residual=residual,
            replayed_operator_action=action.clone(),
            replayed_residual=residual.clone(),
            replay_consistency={
                "operator_action_maxabs_difference": 0.0,
                "operator_action_norm2_difference": 0.0,
                "residual_maxabs_difference": 0.0,
                "residual_norm2_difference": 0.0,
            },
            matrix_free_error=None,
            matrix_free_error_unavailable_reason="rank_zero_no_reduced_system",
            operator_action_provenance="rank_zero_no_operator_application",
            residual_provenance="rank_zero_empty_rhs_minus_empty_action",
            replay_action_provenance="rank_zero_no_operator_replay",
            operator_norm_upper_bound_provenance=(
                system.operator_norm_upper_bound_provenance
            ),
            roundoff_qualification=MATRIX_FREE_ROUNDOFF_QUALIFICATION,
            diagnostic_operator_matvecs=0,
        )

    if system.operator is None:
        raise ProbeExecutionEvidenceError("positive-rank system has no operator")
    canonical_action = solve.operator_action
    canonical_residual = solve.residual
    if not torch.equal(
        system.conditional_cross - canonical_action,
        canonical_residual,
    ):
        raise ProbeExecutionEvidenceError(
            "solver fresh residual is not exactly rhs minus its saved final action"
        )
    replayed_action = system.operator.matmul(solve.solution)
    replayed_residual = system.conditional_cross - replayed_action
    if not bool(torch.isfinite(replayed_action).all().item()) or not bool(
        torch.isfinite(replayed_residual).all().item()
    ):
        raise ProbeExecutionEvidenceError("independent action replay is nonfinite")
    action_difference = replayed_action - canonical_action
    residual_difference = replayed_residual - canonical_residual
    replay_consistency = {
        "operator_action_maxabs_difference": float(torch.max(torch.abs(action_difference))),
        "operator_action_norm2_difference": float(torch.linalg.vector_norm(action_difference)),
        "residual_maxabs_difference": float(torch.max(torch.abs(residual_difference))),
        "residual_norm2_difference": float(torch.linalg.vector_norm(residual_difference)),
    }
    if not all(math.isfinite(value) for value in replay_consistency.values()):
        raise ProbeExecutionEvidenceError("action replay comparison is nonfinite")
    upper_bound = system.operator_norm_upper_bound
    if upper_bound is None or not math.isfinite(upper_bound) or upper_bound <= 0.0:
        raise ProbeExecutionEvidenceError(
            "positive-rank matrix-free evidence requires a finite positive norm bound"
        )
    if solve.operator_norm_upper_bound != upper_bound:
        raise ProbeExecutionEvidenceError(
            "solver and reusable system disagree on the operator norm upper bound"
        )
    error = matrix_free_solve_error_metrics(
        canonical_residual,
        system.conditional_cross,
        solve.solution,
        operator_norm_upper_bound=upper_bound,
        residual_compute_dtype=solve.solution.dtype,
    )
    return OrbitSolveEvidence(
        requested_tolerance=tolerance,
        prediction=prediction,
        verified_operator_action=canonical_action.detach().clone().contiguous(),
        verified_residual=canonical_residual.detach().clone().contiguous(),
        replayed_operator_action=replayed_action.detach().clone().contiguous(),
        replayed_residual=replayed_residual.detach().clone().contiguous(),
        replay_consistency=replay_consistency,
        matrix_free_error=error,
        matrix_free_error_unavailable_reason=None,
        operator_action_provenance=OPERATOR_ACTION_PROVENANCE,
        residual_provenance=RESIDUAL_PROVENANCE,
        replay_action_provenance=REPLAY_ACTION_PROVENANCE,
        operator_norm_upper_bound_provenance=(
            system.operator_norm_upper_bound_provenance
        ),
        roundoff_qualification=MATRIX_FREE_ROUNDOFF_QUALIFICATION,
        diagnostic_operator_matvecs=1,
    )


def execute_registered_orbit_target(
    arm: RegisteredOrbitArmInputs,
    source_geometry: RegisteredSourceGeometry,
    strata: RegisteredOrbitStrata,
) -> OrbitTargetExecution:
    """Build once and execute the strata-bound target/dtype ORBIT arm.

    Nonconvergence, a physical-rank mismatch, or a nonpositive raw moment is
    scientific output handled by later gates.  Contradictory identities or
    numerical evidence fail structurally here.
    """

    if type(arm) is not RegisteredOrbitArmInputs:
        raise ProbeExecutionInputError("arm must be exact RegisteredOrbitArmInputs")
    arm.assert_unchanged()
    if type(source_geometry) is not RegisteredSourceGeometry:
        raise ProbeExecutionInputError("source_geometry must be registered N0 evidence")
    position = _validate_target_position(
        source_geometry.target_position,
        arm.evaluation.X.shape[0],
    )
    source_cutoff, source_selected_rank, source_rank_reference_sha256 = (
        _validate_source_geometry_reference(
            arm,
            source_geometry,
            target_position=position,
        )
    )
    include_stratum_sweep = _validate_registered_strata(
        arm,
        source_geometry,
        strata,
    )
    tolerances = registered_orbit_tolerances(
        arm.train.X.dtype,
        include_stratum_sweep=include_stratum_sweep,
    )
    neighbour_positions = arm.fixed_neighbours.positions[position]
    neighbour_sources = arm.fixed_neighbours.source_indices[position]
    target = arm.evaluation.X[position].unsqueeze(0)

    with torch.inference_mode():
        if arm.train.X.dtype == SOURCE_DTYPE:
            system = _build_local_value_system_from_registered_geometry(
                arm.train.X[neighbour_positions],
                arm.train.value[neighbour_positions],
                arm.train.gradient[neighbour_positions],
                target,
                lengthscale=arm.parameters.lengthscale,
                outputscale=arm.parameters.outputscale,
                value_noise_variance=arm.parameters.sigma_f,
                gradient_noise_variance=arm.parameters.sigma_g,
                kernel=arm.parameters.kernel,
                gradient_noise_model=arm.parameters.gradient_noise_model,
                precomputed_geometry=source_geometry.geometry,
                function_jitter=SOURCE_FUNCTION_JITTER,
                reduced_jitter=SOURCE_REDUCED_JITTER,
                build_preconditioner=True,
            )
            if system.geometry is not source_geometry.geometry:
                raise ProbeExecutionEvidenceError(
                    "source-fp32 execution did not reuse its sole N0 geometry object"
                )
        else:
            system = build_local_value_system(
                arm.train.X[neighbour_positions],
                arm.train.value[neighbour_positions],
                arm.train.gradient[neighbour_positions],
                target,
                lengthscale=arm.parameters.lengthscale,
                outputscale=arm.parameters.outputscale,
                value_noise_variance=arm.parameters.sigma_f,
                gradient_noise_variance=arm.parameters.sigma_g,
                kernel=arm.parameters.kernel,
                gradient_noise_model=arm.parameters.gradient_noise_model,
                absolute_rank_cutoff=source_cutoff,
                function_jitter=SOURCE_FUNCTION_JITTER,
                reduced_jitter=SOURCE_REDUCED_JITTER,
                build_preconditioner=True,
            )
        rank_record = _direct_svd_rank_record(
            system.geometry,
            expected_rank=arm.work_plan.physical_rank,
            source_fp32_cutoff=source_cutoff,
            source_fp32_selected_rank=source_selected_rank,
        )
        factor_error = cholesky_backward_error_metrics(
            system.function_system_matrix,
            system.function_cholesky,
            compute_dtype=arm.train.X.dtype,
        )
        solves = tuple(
            _verified_solve_evidence(
                system,
                solve_local_value_system(
                    system,
                    tolerance=tolerance,
                    max_iterations=arm.work_plan.max_iterations,
                    use_preconditioner=True,
                ),
                tolerance=tolerance,
                max_iterations=arm.work_plan.max_iterations,
            )
            for tolerance in tolerances
        )

    arm.assert_unchanged()
    post_cutoff, post_selected_rank, post_reference_sha256 = (
        _validate_source_geometry_reference(
            arm,
            source_geometry,
            target_position=position,
        )
    )
    if (
        post_cutoff != source_cutoff
        or post_selected_rank != source_selected_rank
        or post_reference_sha256 != source_rank_reference_sha256
    ):
        raise ProbeExecutionEvidenceError(
            "source geometry changed during registered target execution"
        )
    if _validate_registered_strata(arm, source_geometry, strata) != include_stratum_sweep:
        raise ProbeExecutionEvidenceError(
            "registered stratum role changed during target execution"
        )

    result = OrbitTargetExecution(
        task_index=arm.work_plan.task_index,
        source_arm_binding_sha256=arm.source_arm_binding_sha256,
        source_rank_reference_sha256=source_rank_reference_sha256,
        source_rank_grid_sha256=strata.source_rank_grid_sha256,
        strata_selection_sha256=strata.selection_sha256,
        target_position=position,
        target_source_index=int(arm.evaluation.source_indices[position].item()),
        neighbour_positions=neighbour_positions.detach().clone().contiguous(),
        neighbour_source_indices=neighbour_sources.detach().clone().contiguous(),
        compute_dtype=arm.train.X.dtype,
        system=system,
        rank_boundary=rank_record,
        function_cholesky_error=factor_error,
        solves=solves,
        production_tolerance=arm.work_plan.production_tolerance,
        include_stratum_sweep=include_stratum_sweep,
    )
    # Exercise the identity invariant before returning an artifact-bearing object.
    _ = result.production_solve
    return result


__all__ = [
    "LabelFreeEvaluationTensors",
    "MATRIX_FREE_ROUNDOFF_QUALIFICATION",
    "OPERATOR_ACTION_PROVENANCE",
    "OrbitSolveEvidence",
    "OrbitTargetExecution",
    "ProbeExecutionEvidenceError",
    "ProbeExecutionInputError",
    "REPLAY_ACTION_PROVENANCE",
    "RegisteredOrbitArmInputs",
    "RegisteredOrbitStrata",
    "RegisteredSourceGeometry",
    "RESIDUAL_PROVENANCE",
    "SOURCE_FUNCTION_JITTER",
    "SOURCE_RANK_EPSILON",
    "SOURCE_REDUCED_JITTER",
    "build_source_orbit_arm_inputs",
    "evaluation_rows_to_tensors",
    "execute_registered_orbit_target",
    "promote_evaluation_to_float64",
    "promote_parameters_to_float64",
    "promote_registered_orbit_arm_to_float64",
    "promote_training_split_to_float64",
    "registered_orbit_tolerances",
    "scan_registered_source_geometry",
    "select_registered_orbit_strata",
]
