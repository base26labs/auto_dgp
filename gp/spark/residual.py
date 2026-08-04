"""Immutable kinetic residualization shared by SPARK and structure-aware controls."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import torch

from gp.spark.structure import (
    DiagonalQuadraticMean,
    HamiltonianSplit,
    fit_diagonal_quadratic_mean,
    infer_hamiltonian_split,
)


@dataclass(frozen=True)
class PotentialResidualTransform:
    """Center and scalar scale for the value-minus-kinetic residual."""

    offset: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def fit(cls, residual_value: torch.Tensor) -> PotentialResidualTransform:
        if residual_value.ndim != 1 or residual_value.numel() < 2:
            raise ValueError("residual_value must contain at least two scalar observations")
        if not bool(torch.isfinite(residual_value).all()):
            raise ValueError("residual_value must be finite")
        offset = residual_value.mean()
        scale = residual_value.std(unbiased=False)
        residual_rms = residual_value.square().mean().clamp_min(1).sqrt()
        minimum = torch.finfo(residual_value.dtype).eps ** 0.5 * residual_rms
        if float(scale) <= float(minimum):
            raise RuntimeError("potential residual has no numerically useful value variation")
        return cls(offset=offset, scale=scale)

    def normalize_value(self, residual_value: torch.Tensor) -> torch.Tensor:
        return (residual_value - self.offset) / self.scale

    def normalize_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        return gradient / self.scale


@dataclass(frozen=True)
class HamiltonianResidualization:
    """A serializable capability learned from one declared set of labeled rows."""

    split: HamiltonianSplit
    kinetic_mean: DiagonalQuadraticMean
    residual_transform: PotentialResidualTransform
    position_q: torch.Tensor
    potential_values: torch.Tensor
    potential_gradients: torch.Tensor
    position_offset: torch.Tensor
    stage: Literal["selection", "final"]
    source_row_ids_sha256: str | None


class SupportsPositionResidualization(Protocol):
    split: HamiltonianSplit
    kinetic_mean: DiagonalQuadraticMean
    residual_transform: PotentialResidualTransform
    position_q: torch.Tensor
    potential_values: torch.Tensor
    potential_gradients: torch.Tensor
    position_offset: torch.Tensor


@dataclass(frozen=True)
class PositionResidualBatch:
    """Position inputs and normalized potential observations for a generic GP."""

    X: torch.Tensor
    value: torch.Tensor
    gradient: torch.Tensor


@dataclass(frozen=True)
class ReconstructedPrediction:
    """Full-state standardized prediction after restoring the kinetic mean."""

    value: torch.Tensor
    gradient: torch.Tensor
    variance: torch.Tensor


@dataclass(frozen=True)
class ReconstructedPredictionSuffix:
    """Values for all queries and gradients for one explicit trailing partition."""

    value: torch.Tensor
    variance: torch.Tensor
    gradient: torch.Tensor
    gradient_start: int


RESIDUALIZATION_ARRAY_FIELDS = (
    "candidate_scores",
    "selected_block",
    "n_particles",
    "spatial_dims",
    "kinetic_indices",
    "position_indices",
    "kinetic_slopes",
    "kinetic_intercepts",
    "input_dimension",
    "residual_offset",
    "residual_scale",
    "position_q",
    "potential_values",
    "potential_gradients",
    "position_offset",
    "stage",
    "source_row_ids_sha256",
)


def _validate_frozen_split(
    split: HamiltonianSplit,
    X: torch.Tensor,
    *,
    n_particles: int,
    spatial_dims: int,
) -> HamiltonianSplit:
    expected_dimension = 2 * n_particles * spatial_dims
    if split.state_dimension != expected_dimension or X.shape[1] != expected_dimension:
        raise ValueError("frozen split does not match the requested state schema")
    kinetic = split.kinetic_indices.to(device=X.device, dtype=torch.long)
    position = split.position_indices.to(device=X.device, dtype=torch.long)
    combined = torch.cat((kinetic, position)).sort().values
    if not torch.equal(combined, torch.arange(expected_dimension, device=X.device)):
        raise ValueError("frozen split indices must partition the state coordinates")
    return HamiltonianSplit(
        kinetic_indices=kinetic,
        position_indices=position,
        candidate_scores=split.candidate_scores,
        selected_block=split.selected_block,
        n_particles=n_particles,
        spatial_dims=spatial_dims,
    )


def _row_ids_sha256(source_row_ids: torch.Tensor | None, expected_rows: int) -> str | None:
    if source_row_ids is None:
        return None
    row_ids = torch.as_tensor(source_row_ids, device="cpu", dtype=torch.int64)
    if row_ids.ndim != 1 or row_ids.numel() != expected_rows:
        raise ValueError("source_row_ids must have one entry per labeled row")
    if torch.unique(row_ids).numel() != row_ids.numel():
        raise ValueError("source_row_ids must be unique")
    return hashlib.sha256(row_ids.contiguous().numpy().tobytes()).hexdigest()


def prepare_hamiltonian_residualization(
    X: torch.Tensor,
    value: torch.Tensor,
    gradient: torch.Tensor,
    trajectory_id: torch.Tensor | None,
    *,
    n_particles: int,
    spatial_dims: int,
    coordinate_offset: torch.Tensor | None = None,
    frozen_split: HamiltonianSplit | None = None,
    stage: Literal["selection", "final"] = "selection",
    source_row_ids: torch.Tensor | None = None,
) -> HamiltonianResidualization:
    """Discover or refit one shared kinetic residualization capability."""

    if X.ndim != 2 or not X.is_floating_point():
        raise ValueError("X must be a two-dimensional floating-point tensor")
    if value.shape != (X.shape[0],) or gradient.shape != X.shape:
        raise ValueError("value and gradient shapes must match X")
    if value.device != X.device or gradient.device != X.device:
        raise ValueError("all observations must use the same device")
    if value.dtype != X.dtype or gradient.dtype != X.dtype:
        raise ValueError("all observations must use the same dtype")
    if not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(gradient).all()):
        raise ValueError("value and gradient must be finite")
    if stage not in {"selection", "final"}:
        raise ValueError(f"unknown residualization stage: {stage}")
    source_row_ids_sha256 = _row_ids_sha256(source_row_ids, X.shape[0])

    if frozen_split is None:
        if trajectory_id is None:
            raise ValueError("trajectory_id is required for fit-only structure discovery")
        split = infer_hamiltonian_split(
            X,
            gradient,
            trajectory_id,
            n_particles=n_particles,
            spatial_dims=spatial_dims,
        )
    else:
        split = _validate_frozen_split(
            frozen_split,
            X,
            n_particles=n_particles,
            spatial_dims=spatial_dims,
        )

    kinetic_mean = fit_diagonal_quadratic_mean(X, gradient, split.kinetic_indices)
    if coordinate_offset is None:
        coordinate_offset = X.new_zeros(X.shape[1])
    coordinate_offset = torch.as_tensor(coordinate_offset, device=X.device, dtype=X.dtype)
    if coordinate_offset.shape != (X.shape[1],):
        raise ValueError(f"coordinate_offset must have shape ({X.shape[1]},)")
    if not bool(torch.isfinite(coordinate_offset).all()):
        raise ValueError("coordinate_offset must be finite")
    position_offset = coordinate_offset[split.position_indices]
    position_q = X[:, split.position_indices] + position_offset

    kinetic_value = kinetic_mean.value(X)
    transform = PotentialResidualTransform.fit(value - kinetic_value)
    return HamiltonianResidualization(
        split=split,
        kinetic_mean=kinetic_mean,
        residual_transform=transform,
        position_q=position_q,
        potential_values=transform.normalize_value(value - kinetic_value),
        potential_gradients=transform.normalize_gradient(gradient[:, split.position_indices]),
        position_offset=position_offset,
        stage=stage,
        source_row_ids_sha256=source_row_ids_sha256,
    )


def residualization_digest(state: SupportsPositionResidualization) -> str:
    """Hash every learned parameter needed to share a residualization exactly."""

    metadata = {
        "candidate_scores": list(state.split.candidate_scores),
        "selected_block": state.split.selected_block,
        "n_particles": state.split.n_particles,
        "spatial_dims": state.split.spatial_dims,
        "input_dimension": state.kinetic_mean.input_dimension,
        "stage": getattr(state, "stage", None),
        "source_row_ids_sha256": getattr(state, "source_row_ids_sha256", None),
    }
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for tensor in (
        state.split.kinetic_indices,
        state.split.position_indices,
        state.kinetic_mean.indices,
        state.kinetic_mean.slopes,
        state.kinetic_mean.intercepts,
        state.residual_transform.offset,
        state.residual_transform.scale,
        state.position_offset,
        state.position_q,
        state.potential_values,
        state.potential_gradients,
    ):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def residualization_to_numpy(
    state: HamiltonianResidualization,
    *,
    prefix: str = "",
) -> dict[str, np.ndarray]:
    """Serialize a shared state without adding pair-specific information."""

    if state.source_row_ids_sha256 is None:
        raise ValueError("serialized residualization requires a source-row digest")
    arrays = {
        "candidate_scores": np.asarray(state.split.candidate_scores, dtype=np.float64),
        "selected_block": np.asarray(state.split.selected_block, dtype=np.int64),
        "n_particles": np.asarray(state.split.n_particles, dtype=np.int64),
        "spatial_dims": np.asarray(state.split.spatial_dims, dtype=np.int64),
        "kinetic_indices": state.split.kinetic_indices.detach().cpu().numpy(),
        "position_indices": state.split.position_indices.detach().cpu().numpy(),
        "kinetic_slopes": state.kinetic_mean.slopes.detach().cpu().numpy(),
        "kinetic_intercepts": state.kinetic_mean.intercepts.detach().cpu().numpy(),
        "input_dimension": np.asarray(state.kinetic_mean.input_dimension, dtype=np.int64),
        "residual_offset": state.residual_transform.offset.detach().cpu().numpy(),
        "residual_scale": state.residual_transform.scale.detach().cpu().numpy(),
        "position_q": state.position_q.detach().cpu().numpy(),
        "potential_values": state.potential_values.detach().cpu().numpy(),
        "potential_gradients": state.potential_gradients.detach().cpu().numpy(),
        "position_offset": state.position_offset.detach().cpu().numpy(),
        "stage": np.asarray(state.stage),
        "source_row_ids_sha256": np.asarray(state.source_row_ids_sha256),
    }
    return {f"{prefix}{key}": np.asarray(value) for key, value in arrays.items()}


def residualization_from_numpy(
    arrays: Mapping[str, np.ndarray],
    *,
    prefix: str = "",
) -> HamiltonianResidualization:
    """Restore an immutable shared state from its strict array schema."""

    keys = {f"{prefix}{field}" for field in RESIDUALIZATION_ARRAY_FIELDS}
    missing = keys.difference(arrays)
    if missing:
        raise ValueError(f"serialized residualization is missing fields: {sorted(missing)}")
    allowed = keys | ({f"{prefix}digest"} if prefix else set())
    unexpected = {key for key in arrays if key.startswith(prefix)}.difference(allowed)
    if unexpected:
        raise ValueError(f"serialized residualization has unexpected fields: {sorted(unexpected)}")

    def get(name: str) -> np.ndarray:
        return np.asarray(arrays[f"{prefix}{name}"])

    def tensor(name: str, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        result = torch.from_numpy(get(name).copy())
        return result if dtype is None else result.to(dtype=dtype)

    stage = str(get("stage").item())
    if stage not in {"selection", "final"}:
        raise ValueError(f"unknown serialized residualization stage: {stage}")
    candidate_scores = get("candidate_scores")
    selected_block = int(get("selected_block").item())
    n_particles = int(get("n_particles").item())
    spatial_dims = int(get("spatial_dims").item())
    input_dimension = int(get("input_dimension").item())
    kinetic_indices = get("kinetic_indices")
    position_indices = get("position_indices")
    position_width = n_particles * spatial_dims
    integer_scalars = ("selected_block", "n_particles", "spatial_dims", "input_dimension")
    if any(get(name).shape or get(name).dtype != np.int64 for name in integer_scalars):
        raise ValueError("serialized split scalars must be canonical int64 values")
    if candidate_scores.shape != (2,) or candidate_scores.dtype != np.float64:
        raise ValueError("serialized candidate scores must be a float64 pair")
    if selected_block not in {0, 1} or n_particles <= 0 or spatial_dims <= 0:
        raise ValueError("serialized split decision or dimensions are invalid")
    if input_dimension != 2 * position_width:
        raise ValueError("serialized input dimension does not match the particle schema")
    for name, indices in (("kinetic", kinetic_indices), ("position", position_indices)):
        if indices.dtype != np.int64 or indices.shape != (position_width,):
            raise ValueError(f"serialized {name} indices are not canonical")
    if not np.array_equal(
        np.sort(np.concatenate((kinetic_indices, position_indices))),
        np.arange(input_dimension),
    ):
        raise ValueError("serialized split indices do not partition the input dimension")
    split = HamiltonianSplit(
        kinetic_indices=tensor("kinetic_indices", dtype=torch.long),
        position_indices=tensor("position_indices", dtype=torch.long),
        candidate_scores=tuple(float(value) for value in candidate_scores),
        selected_block=selected_block,
        n_particles=n_particles,
        spatial_dims=spatial_dims,
    )
    floating_fields = (
        "kinetic_slopes",
        "kinetic_intercepts",
        "residual_offset",
        "residual_scale",
        "position_q",
        "potential_values",
        "potential_gradients",
        "position_offset",
    )
    float_dtype = get("position_q").dtype
    if not np.issubdtype(float_dtype, np.floating):
        raise ValueError("serialized residualization tensors must be floating point")
    if any(get(name).dtype != float_dtype for name in floating_fields):
        raise ValueError("serialized residualization tensors must share one floating dtype")
    if not all(np.isfinite(get(name)).all() for name in floating_fields):
        raise ValueError("serialized residualization tensors must be finite")
    if get("kinetic_slopes").shape != (position_width,) or bool((get("kinetic_slopes") <= 0).any()):
        raise ValueError("serialized kinetic slopes must be a positive block vector")
    if get("kinetic_intercepts").shape != (position_width,):
        raise ValueError("serialized kinetic intercepts have an invalid shape")
    if get("residual_offset").shape or get("residual_scale").shape:
        raise ValueError("serialized residual transform must contain scalars")
    if float(get("residual_scale")) <= 0:
        raise ValueError("serialized residual scale must be positive")
    if get("position_offset").shape != (position_width,):
        raise ValueError("serialized position offset has an invalid shape")
    source_digest = str(get("source_row_ids_sha256").item())
    if len(source_digest) != 64 or any(
        character not in "0123456789abcdef" for character in source_digest
    ):
        raise ValueError("serialized source-row digest is not canonical SHA-256")
    kinetic = DiagonalQuadraticMean(
        indices=split.kinetic_indices,
        slopes=tensor("kinetic_slopes"),
        intercepts=tensor("kinetic_intercepts"),
        input_dimension=input_dimension,
    )
    transform = PotentialResidualTransform(
        offset=tensor("residual_offset"),
        scale=tensor("residual_scale"),
    )
    state = HamiltonianResidualization(
        split=split,
        kinetic_mean=kinetic,
        residual_transform=transform,
        position_q=tensor("position_q"),
        potential_values=tensor("potential_values"),
        potential_gradients=tensor("potential_gradients"),
        position_offset=tensor("position_offset"),
        stage=stage,
        source_row_ids_sha256=source_digest,
    )
    if state.position_q.shape[1] != split.n_particles * split.spatial_dims:
        raise ValueError("serialized position inputs are inconsistent with the split")
    if state.potential_values.shape != (state.position_q.shape[0],):
        raise ValueError("serialized potential values have an invalid shape")
    if state.potential_gradients.shape != state.position_q.shape:
        raise ValueError("serialized potential gradients have an invalid shape")
    return state


def position_residual_training(
    state: SupportsPositionResidualization,
) -> PositionResidualBatch:
    """Expose exactly the residual problem used by SPARK to a generic baseline."""

    return PositionResidualBatch(
        X=state.position_q,
        value=state.potential_values,
        gradient=state.potential_gradients,
    )


def position_residual_inputs(
    state: SupportsPositionResidualization,
    X: torch.Tensor,
) -> torch.Tensor:
    """Project standardized full states onto the restored position coordinates."""

    if X.ndim != 2 or X.shape[1] != state.split.state_dimension:
        raise ValueError(f"X must have shape (N, {state.split.state_dimension})")
    if X.device != state.position_offset.device or X.dtype != state.position_offset.dtype:
        raise ValueError("X and the prepared residualization must share device and dtype")
    return X[:, state.split.position_indices] + state.position_offset


def reconstruct_position_prediction(
    state: SupportsPositionResidualization,
    X: torch.Tensor,
    potential_value: torch.Tensor,
    potential_gradient: torch.Tensor,
    potential_variance: torch.Tensor,
) -> ReconstructedPrediction:
    """Restore full value, gradient, and variance from a position-only GP prediction."""

    reconstructed = reconstruct_position_prediction_suffix(
        state,
        X,
        potential_value,
        potential_gradient,
        potential_variance,
        gradient_start=0,
    )
    return ReconstructedPrediction(
        value=reconstructed.value,
        gradient=reconstructed.gradient,
        variance=reconstructed.variance,
    )


def reconstruct_position_prediction_suffix(
    state: SupportsPositionResidualization,
    X: torch.Tensor,
    potential_value: torch.Tensor,
    potential_gradient: torch.Tensor,
    potential_variance: torch.Tensor,
    *,
    gradient_start: int,
) -> ReconstructedPredictionSuffix:
    """Restore values for all rows and gradients aligned to ``X[gradient_start:]``."""

    position_width = state.split.n_particles * state.split.spatial_dims
    if potential_value.shape != (X.shape[0],):
        raise ValueError("potential_value must have one entry per query")
    if not 0 <= gradient_start < X.shape[0]:
        raise ValueError("gradient_start must index a non-empty query suffix")
    if potential_gradient.shape != (X.shape[0] - gradient_start, position_width):
        raise ValueError("potential_gradient has an invalid shape")
    if potential_variance.shape != (X.shape[0],):
        raise ValueError("potential_variance must have one entry per query")
    tensors = (X, potential_value, potential_gradient, potential_variance)
    if any(tensor.device != X.device or tensor.dtype != X.dtype for tensor in tensors[1:]):
        raise ValueError("queries and potential predictions must share device and dtype")
    if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise ValueError("queries and potential predictions must be finite")
    if bool((potential_variance <= 0).any()):
        raise ValueError("potential_variance must be positive")

    value = (
        state.kinetic_mean.value(X)
        + state.residual_transform.offset
        + state.residual_transform.scale * potential_value
    )
    gradient = state.kinetic_mean.gradient(X[gradient_start:])
    gradient[:, state.split.position_indices] = state.residual_transform.scale * potential_gradient
    variance = state.residual_transform.scale.square() * potential_variance
    return ReconstructedPredictionSuffix(
        value=value,
        variance=variance,
        gradient=gradient,
        gradient_start=gradient_start,
    )
