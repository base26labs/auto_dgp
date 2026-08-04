"""Registered F02b rank-boundary stress execution.

Only repeat-zero, seed-11 reference tasks own this suite.  Their extra
``m=D-5`` neighbourhood has one more q coordinate than the expected physical
rank ``D-6`` and is therefore the registered sentinel for support leakage.
Neighbour selection and numerical rank are performed once in source fp32;
every stress solve uses the exact-promoted CPU-float64 tensors and the frozen
absolute source cutoff.

The suite is intentionally compact: one worst geometry target receives five
predeclared checks (support/complement probes, neighbour permutation, retained
support rotation, exact-zero ambient augmentation, and native-fp64 discarded
mode leakage).  No file, label, catalog, environment, or scheduler I/O occurs
in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from experiments.f02b_calibration_metrics import (
    projector_metrics,
    select_geometry_strata,
)
from experiments.f02b_calibration_probe_core import (
    PRIMARY_EVALUATION_ROW_COUNT,
    STRESS_FIXED_PROBE_COUNT,
    STRESS_PERMUTATION_RULE,
    STRESS_PROBE_HASH_DOMAIN,
    STRESS_SUPPORT_ROTATION_RULE,
    STRESS_TOLERANCE,
    FixedNeighbourRows,
    fixed_fp32_neighbours,
)
from experiments.f02b_calibration_probe_execution import (
    SOURCE_DTYPE,
    SOURCE_FUNCTION_JITTER,
    SOURCE_RANK_EPSILON,
    SOURCE_REDUCED_JITTER,
    ProbeExecutionEvidenceError,
    ProbeExecutionInputError,
    RegisteredOrbitArmInputs,
    _direct_svd_rank_record,
    _source_rank_reference_sha256,
    _validate_target_position,
)
from gp.orbit import (
    LocalGeometry,
    LocalPrediction,
    LocalValueSystem,
    OrthonormalReducedOperator,
    ReducedKroneckerPreconditioner,
    build_local_geometry_from_differences,
    build_local_value_system,
    solve_local_value_system,
)

_STRESS_INPUT_TOKEN = object()
_STRESS_GEOMETRY_TOKEN = object()
_STRESS_STRATA_TOKEN = object()
_STRESS_EXECUTION_TOKEN = object()
_STRESS_BINDING_DOMAIN = "auto_dgp2.f02b.registered_stress_inputs_v1"
_STRESS_STRATA_DOMAIN = "auto_dgp2.f02b.registered_stress_strata_v1"
_SQRT_HALF = math.sqrt(0.5)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _update_tensor(hasher: Any, label: str, value: torch.Tensor) -> None:
    tensor = value.detach().contiguous().cpu()
    hasher.update(label.encode("utf-8"))
    hasher.update(str(tensor.dtype).encode("ascii"))
    hasher.update(repr(tuple(tensor.shape)).encode("ascii"))
    hasher.update(tensor.numpy().tobytes(order="C"))


def _tensor_sha256(value: torch.Tensor) -> str:
    hasher = hashlib.sha256()
    _update_tensor(hasher, "tensor", value)
    return hasher.hexdigest()


def _finite_float(value: torch.Tensor | float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ProbeExecutionEvidenceError(f"stress {label} is nonfinite")
    return result


def _stress_binding(
    source_arm: RegisteredOrbitArmInputs,
    neighbours: FixedNeighbourRows,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(_STRESS_BINDING_DOMAIN.encode("ascii"))
    hasher.update(source_arm.source_arm_binding_sha256.encode("ascii"))
    hasher.update(str(source_arm.work_plan.task_index).encode("ascii"))
    hasher.update(str(neighbours.m).encode("ascii"))
    _update_tensor(hasher, "positions", neighbours.positions)
    _update_tensor(hasher, "source_indices", neighbours.source_indices)
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class RegisteredStressInputs:
    task_index: int
    source_arm_binding_sha256: str
    stress_binding_sha256: str
    m: int
    fixed_neighbours: FixedNeighbourRows
    _construction_token: object = field(repr=False, compare=False)
    _tensor_versions: tuple[int, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._construction_token is not _STRESS_INPUT_TOKEN:
            raise ProbeExecutionInputError(
                "RegisteredStressInputs must be created by its audited factory"
            )
        object.__setattr__(
            self,
            "_tensor_versions",
            (
                self.fixed_neighbours.positions._version,
                self.fixed_neighbours.source_indices._version,
            ),
        )

    def assert_unchanged(self, source_arm: RegisteredOrbitArmInputs) -> None:
        source_arm.assert_unchanged()
        if (
            self.task_index != source_arm.work_plan.task_index
            or self.source_arm_binding_sha256 != source_arm.source_arm_binding_sha256
            or self.m != source_arm.work_plan.stress_m
        ):
            raise ProbeExecutionInputError("stress inputs belong to a different task")
        versions = (
            self.fixed_neighbours.positions._version,
            self.fixed_neighbours.source_indices._version,
        )
        if versions != self._tensor_versions:
            raise ProbeExecutionInputError("stress neighbour tensors changed after binding")
        if _stress_binding(source_arm, self.fixed_neighbours) != self.stress_binding_sha256:
            raise ProbeExecutionInputError("stress input binding SHA-256 is mismatched")


@dataclass(frozen=True, slots=True)
class RegisteredStressGeometry:
    task_index: int
    stress_binding_sha256: str
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
        if self._construction_token is not _STRESS_GEOMETRY_TOKEN:
            raise ProbeExecutionInputError(
                "RegisteredStressGeometry must be created by its audited scan"
            )


@dataclass(frozen=True, slots=True)
class RegisteredStressStrata:
    task_index: int
    stress_binding_sha256: str
    source_rank_reference_sha256: tuple[str, ...]
    source_rank_grid_sha256: str
    selected_target_position: int
    selection_record: dict[str, Any]
    selection_sha256: str
    _construction_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._construction_token is not _STRESS_STRATA_TOKEN:
            raise ProbeExecutionInputError(
                "RegisteredStressStrata must be created by its audited selector"
            )


@dataclass(frozen=True, slots=True)
class StressTargetExecution:
    task_index: int
    source_arm_binding_sha256: str
    stress_binding_sha256: str
    source_rank_reference_sha256: str
    source_rank_grid_sha256: str
    strata_selection_sha256: str
    target_position: int
    target_source_index: int
    neighbour_positions: torch.Tensor
    neighbour_source_indices: torch.Tensor
    m: int
    selected_rank: int
    requested_tolerance: float
    max_iterations: int
    base_solve: dict[str, Any]
    tests: dict[str, Any]
    source_q_projector: torch.Tensor
    native_fp64_q_projector: torch.Tensor
    _construction_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._construction_token is not _STRESS_EXECUTION_TOKEN:
            raise ProbeExecutionInputError(
                "StressTargetExecution must be created by the registered executor"
            )
        expected = {
            "support_complement",
            "permutation",
            "support_rotation",
            "exact_zero_augmentation",
            "discarded_mode_leakage",
        }
        if set(self.tests) != expected:
            raise ProbeExecutionEvidenceError(
                "stress execution does not contain the five registered tests"
            )


def build_registered_stress_inputs(
    source_arm: RegisteredOrbitArmInputs,
) -> RegisteredStressInputs:
    """Freeze the sole source-fp32 neighbour grid for the registered stress m."""

    if type(source_arm) is not RegisteredOrbitArmInputs:
        raise ProbeExecutionInputError("stress inputs require a registered source arm")
    source_arm.assert_unchanged()
    plan = source_arm.work_plan
    if source_arm.train.X.dtype != SOURCE_DTYPE or source_arm.train.X.device.type != "cpu":
        raise ProbeExecutionInputError("stress neighbour selection requires CPU float32")
    if (
        plan.stress_m is None
        or plan.stress_support_target_count != 1
        or plan.stress_max_iterations is None
        or plan.repeat_id != 0
        or plan.role != "reference"
        or plan.stress_m != plan.dimension - 5
    ):
        raise ProbeExecutionInputError("task is not eligible for the registered stress suite")
    neighbours = fixed_fp32_neighbours(
        source_arm.train.X,
        source_arm.evaluation.X,
        source_arm.parameters.lengthscale,
        source_arm.train.source_indices,
        source_arm.evaluation.source_indices,
        plan.stress_m,
    )
    neighbours = FixedNeighbourRows(
        positions=neighbours.positions.detach().clone().contiguous(),
        source_indices=neighbours.source_indices.detach().clone().contiguous(),
        m=neighbours.m,
    )
    result = RegisteredStressInputs(
        task_index=plan.task_index,
        source_arm_binding_sha256=source_arm.source_arm_binding_sha256,
        stress_binding_sha256=_stress_binding(source_arm, neighbours),
        m=plan.stress_m,
        fixed_neighbours=neighbours,
        _construction_token=_STRESS_INPUT_TOKEN,
    )
    result.assert_unchanged(source_arm)
    return result


def _stress_expected_rank(arm: RegisteredOrbitArmInputs, stress: RegisteredStressInputs) -> int:
    return min(stress.m, arm.work_plan.dimension - 6)


def scan_registered_stress_geometry(
    source_arm: RegisteredOrbitArmInputs,
    stress: RegisteredStressInputs,
    target_position: int,
) -> RegisteredStressGeometry:
    """Run the one source-fp32 stress SVD for a target."""

    stress.assert_unchanged(source_arm)
    position = _validate_target_position(
        target_position,
        source_arm.evaluation.X.shape[0],
    )
    neighbour_positions = stress.fixed_neighbours.positions[position]
    neighbour_sources = stress.fixed_neighbours.source_indices[position]
    target = source_arm.evaluation.X[position].unsqueeze(0)
    with torch.inference_mode():
        raw_differences = (source_arm.train.X[neighbour_positions] - target).T.contiguous()
        scaled_differences = raw_differences / source_arm.parameters.lengthscale.reshape(1, 1)
        geometry = build_local_geometry_from_differences(
            scaled_differences,
            rank_epsilon=SOURCE_RANK_EPSILON,
        )
        if geometry.operational_singular_value_cutoff is None:
            raise ProbeExecutionEvidenceError("stress geometry lacks a source cutoff")
        cutoff = float(geometry.operational_singular_value_cutoff)
        expected_rank = _stress_expected_rank(source_arm, stress)
        rank_record = _direct_svd_rank_record(
            geometry,
            expected_rank=expected_rank,
            source_fp32_cutoff=cutoff,
            source_fp32_selected_rank=geometry.rank,
        )
        target_source = int(source_arm.evaluation.source_indices[position])
        rank_record["target_source_index"] = target_source
        reference = _source_rank_reference_sha256(
            source_arm_binding_sha256=stress.stress_binding_sha256,
            task_index=source_arm.work_plan.task_index,
            target_position=position,
            target_source_index=target_source,
            neighbour_positions=neighbour_positions,
            neighbour_source_indices=neighbour_sources,
            standardized_differences=raw_differences,
            geometry=geometry,
            rank_boundary=rank_record,
        )
    stress.assert_unchanged(source_arm)
    return RegisteredStressGeometry(
        task_index=source_arm.work_plan.task_index,
        stress_binding_sha256=stress.stress_binding_sha256,
        source_rank_reference_sha256=reference,
        target_position=position,
        target_source_index=target_source,
        neighbour_positions=neighbour_positions.detach().clone().contiguous(),
        neighbour_source_indices=neighbour_sources.detach().clone().contiguous(),
        standardized_differences=raw_differences.detach().clone().contiguous(),
        geometry=geometry,
        rank_boundary=rank_record,
        _construction_token=_STRESS_GEOMETRY_TOKEN,
    )


def _validate_stress_geometry(
    source_arm: RegisteredOrbitArmInputs,
    stress: RegisteredStressInputs,
    geometry: RegisteredStressGeometry,
    *,
    target_position: int,
) -> tuple[float, int, str]:
    stress.assert_unchanged(source_arm)
    if type(geometry) is not RegisteredStressGeometry:
        raise ProbeExecutionInputError("stress execution requires registered geometry")
    expected_positions = stress.fixed_neighbours.positions[target_position]
    expected_sources = stress.fixed_neighbours.source_indices[target_position]
    target_source = int(source_arm.evaluation.source_indices[target_position])
    if (
        geometry.task_index != source_arm.work_plan.task_index
        or geometry.stress_binding_sha256 != stress.stress_binding_sha256
        or geometry.target_position != target_position
        or geometry.target_source_index != target_source
        or not torch.equal(geometry.neighbour_positions, expected_positions)
        or not torch.equal(geometry.neighbour_source_indices, expected_sources)
    ):
        raise ProbeExecutionInputError("stress geometry identity is mismatched")
    expected_raw = (
        source_arm.train.X[expected_positions]
        - source_arm.evaluation.X[target_position].unsqueeze(0)
    ).T.contiguous()
    if not torch.equal(expected_raw, geometry.standardized_differences):
        raise ProbeExecutionEvidenceError("stress geometry differences are mismatched")
    evidence = geometry.geometry
    if (
        evidence.singular_values is None
        or evidence.operational_singular_value_cutoff is None
        or evidence.rank_epsilon_used is None
        or evidence.singular_values.dtype != torch.float32
    ):
        raise ProbeExecutionEvidenceError("stress geometry lacks source-fp32 SVD evidence")
    cutoff = float(evidence.operational_singular_value_cutoff)
    observed = _direct_svd_rank_record(
        evidence,
        expected_rank=_stress_expected_rank(source_arm, stress),
        source_fp32_cutoff=cutoff,
        source_fp32_selected_rank=evidence.rank,
    )
    observed["target_source_index"] = target_source
    if _canonical_sha256(observed) != _canonical_sha256(geometry.rank_boundary):
        raise ProbeExecutionEvidenceError("stress rank record is mismatched")
    reference = _source_rank_reference_sha256(
        source_arm_binding_sha256=stress.stress_binding_sha256,
        task_index=source_arm.work_plan.task_index,
        target_position=target_position,
        target_source_index=target_source,
        neighbour_positions=expected_positions,
        neighbour_source_indices=expected_sources,
        standardized_differences=expected_raw,
        geometry=evidence,
        rank_boundary=observed,
    )
    if reference != geometry.source_rank_reference_sha256:
        raise ProbeExecutionEvidenceError("stress geometry SHA-256 is mismatched")
    return cutoff, evidence.rank, reference


def select_registered_stress_stratum(
    source_arm: RegisteredOrbitArmInputs,
    stress: RegisteredStressInputs,
    geometries: tuple[RegisteredStressGeometry, ...],
) -> RegisteredStressStrata:
    """Select the sole worst stress target from the complete ordered N0 grid."""

    stress.assert_unchanged(source_arm)
    if type(geometries) is not tuple or len(geometries) != PRIMARY_EVALUATION_ROW_COUNT:
        raise ProbeExecutionInputError("stress strata require all 100 ordered geometries")
    hashes: list[str] = []
    records: list[dict[str, Any]] = []
    source_to_position: dict[int, int] = {}
    for position, geometry in enumerate(geometries):
        if geometry.target_position != position:
            raise ProbeExecutionInputError("stress geometry grid is out of order")
        _, _, reference = _validate_stress_geometry(
            source_arm,
            stress,
            geometry,
            target_position=position,
        )
        hashes.append(reference)
        records.append(geometry.rank_boundary)
        source_to_position[geometry.target_source_index] = position
    boundary_selection = select_geometry_strata(records, count=2)
    worst = boundary_selection["selected"][0]
    selection = {
        "available_count": boundary_selection["available_count"],
        "selected_count": 1,
        "selection_rule": (
            "ascending minimum log2 guard; target_source_index tie-break; worst only"
        ),
        "selected": [worst],
    }
    selected_source = int(worst["target_source_index"])
    selected_position = source_to_position[selected_source]
    grid_hash = _canonical_sha256(hashes)
    payload = {
        "domain": _STRESS_STRATA_DOMAIN,
        "selection_record": selection,
        "selected_target_position": selected_position,
        "source_rank_grid_sha256": grid_hash,
        "source_rank_reference_sha256": hashes,
        "stress_binding_sha256": stress.stress_binding_sha256,
        "task_index": source_arm.work_plan.task_index,
    }
    stress.assert_unchanged(source_arm)
    return RegisteredStressStrata(
        task_index=source_arm.work_plan.task_index,
        stress_binding_sha256=stress.stress_binding_sha256,
        source_rank_reference_sha256=tuple(hashes),
        source_rank_grid_sha256=grid_hash,
        selected_target_position=selected_position,
        selection_record=json.loads(json.dumps(selection, allow_nan=False, sort_keys=True)),
        selection_sha256=_canonical_sha256(payload),
        _construction_token=_STRESS_STRATA_TOKEN,
    )


def _validate_stress_strata(
    source_arm: RegisteredOrbitArmInputs,
    stress: RegisteredStressInputs,
    geometry: RegisteredStressGeometry,
    strata: RegisteredStressStrata,
) -> None:
    if type(strata) is not RegisteredStressStrata:
        raise ProbeExecutionInputError("stress execution requires registered strata")
    if (
        strata.task_index != source_arm.work_plan.task_index
        or strata.stress_binding_sha256 != stress.stress_binding_sha256
        or strata.selected_target_position != geometry.target_position
        or len(strata.source_rank_reference_sha256) != PRIMARY_EVALUATION_ROW_COUNT
        or strata.source_rank_reference_sha256[geometry.target_position]
        != geometry.source_rank_reference_sha256
    ):
        raise ProbeExecutionInputError("stress target is not the registered worst stratum")
    grid_hash = _canonical_sha256(list(strata.source_rank_reference_sha256))
    payload = {
        "domain": _STRESS_STRATA_DOMAIN,
        "selection_record": strata.selection_record,
        "selected_target_position": strata.selected_target_position,
        "source_rank_grid_sha256": grid_hash,
        "source_rank_reference_sha256": list(strata.source_rank_reference_sha256),
        "stress_binding_sha256": strata.stress_binding_sha256,
        "task_index": strata.task_index,
    }
    if grid_hash != strata.source_rank_grid_sha256 or (
        _canonical_sha256(payload) != strata.selection_sha256
    ):
        raise ProbeExecutionEvidenceError("stress strata SHA-256 is mismatched")


def _solve_record(prediction: LocalPrediction) -> dict[str, Any]:
    solve = prediction.solve
    return {
        "rank": prediction.rank,
        "mean": _finite_float(prediction.mean, "mean"),
        "raw_latent_variance": _finite_float(prediction.variance, "variance"),
        "requested_tolerance": solve.requested_tolerance,
        "max_iterations": solve.max_iterations,
        "converged": solve.converged,
        "termination_reason": solve.termination_reason,
        "iterations": solve.iterations,
        "operator_matvecs": solve.operator_matvecs,
        "relative_residual": _finite_float(solve.relative_residual, "relative residual"),
        "residual_is_fresh": solve.residual_is_fresh,
        "variance_error_upper_bound": _finite_float(
            prediction.certificate.variance_error_upper_bound,
            "variance error bound",
        ),
        "mean_error_upper_bound": _finite_float(
            prediction.certificate.mean_error_upper_bound,
            "mean error bound",
        ),
    }


def _moment_difference(reference: LocalPrediction, candidate: LocalPrediction) -> dict[str, float]:
    return {
        "mean_absolute_difference": _finite_float(
            torch.abs(candidate.mean - reference.mean),
            "mean difference",
        ),
        "variance_absolute_difference": _finite_float(
            torch.abs(candidate.variance - reference.variance),
            "variance difference",
        ),
    }


def _q_projector(geometry: LocalGeometry) -> torch.Tensor:
    return (geometry.coordinates @ geometry.q_to_z.T).to(
        dtype=torch.float64,
        device="cpu",
    )


def _q_space_solution(system: LocalValueSystem, prediction: LocalPrediction) -> torch.Tensor:
    if system.geometry.rank == 0:
        return torch.zeros(
            (system.value_condition.numel(), system.value_condition.numel()),
            dtype=torch.float64,
        )
    return (
        prediction.solve.solution.reshape(-1, system.geometry.rank)
        @ system.geometry.q_to_z.T
    ).to(dtype=torch.float64, device="cpu")


class _FullQStressOperator:
    """Matrix-free released q-coordinate operator used only for fixed probes."""

    def __init__(
        self,
        *,
        H: torch.Tensor,
        q: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        function_cholesky: torch.Tensor,
        value_cross: torch.Tensor,
        noise_gram: torch.Tensor,
        gradient_noise: torch.Tensor,
        jitter: float,
    ) -> None:
        self.H = H
        self.q = q
        self.alpha = alpha
        self.beta = beta
        self.function_cholesky = function_cholesky
        self.value_cross = value_cross
        self.noise_gram = noise_gram
        self.gradient_noise = gradient_noise
        self.jitter = jitter
        self.m = H.shape[0]

    def matmul(self, value: torch.Tensor) -> torch.Tensor:
        matrix = value.reshape(self.m, self.m)
        result = self.alpha @ matrix @ self.H.T
        pair_dot = torch.einsum("ijb,jb->ij", self.q, matrix)
        result = result + torch.einsum(
            "ij,ija->ia",
            self.beta * pair_dot,
            self.q,
        )
        result = result + self.gradient_noise * (matrix @ self.noise_gram.T)
        q_t_value = self.value_cross.T @ value
        conditioned = torch.cholesky_solve(
            q_t_value.unsqueeze(1),
            self.function_cholesky,
        ).squeeze(1)
        result = result.reshape(-1) - self.value_cross @ conditioned
        return result + self.jitter * value


def _full_q_stress_state(
    system: LocalValueSystem,
    x_condition: torch.Tensor,
    values: torch.Tensor,
    gradients: torch.Tensor,
    target: torch.Tensor,
    *,
    lengthscale: torch.Tensor,
    outputscale: float,
    kernel: str,
    gradient_noise: float,
) -> tuple[
    _FullQStressOperator,
    torch.Tensor,
    torch.Tensor,
    dict[str, float],
]:
    if system.operator is None:
        raise ProbeExecutionEvidenceError("stress support/complement requires positive rank")
    raw = (x_condition - target).T.contiguous()
    scaled = raw / lengthscale.reshape(1, 1)
    H = scaled.T @ scaled
    columns = H.T.contiguous()
    q = columns[:, None, :] - columns[None, :, :]
    alpha = system.operator.alpha
    beta = system.operator.beta
    value_cross = (
        (-alpha[:, :, None] * q)
        .permute(0, 2, 1)
        .reshape(x_condition.shape[0] ** 2, x_condition.shape[0])
        .contiguous()
    )
    noise_gram = 0.5 * (raw.T @ raw + (raw.T @ raw).T)
    operator = _FullQStressOperator(
        H=H,
        q=q,
        alpha=alpha,
        beta=beta,
        function_cholesky=system.function_cholesky,
        value_cross=value_cross,
        noise_gram=noise_gram,
        gradient_noise=H.new_tensor(gradient_noise),
        jitter=SOURCE_REDUCED_JITTER,
    )
    basis_map = torch.kron(
        torch.eye(x_condition.shape[0], dtype=H.dtype, device=H.device),
        system.geometry.coordinates.contiguous(),
    )
    target_radii = torch.diagonal(H)
    scale = torch.as_tensor(outputscale, dtype=H.dtype, device=H.device)
    if kernel == "rbf":
        target_alpha = scale * torch.exp(-0.5 * target_radii)
    elif kernel == "matern52":
        scaled_radius = math.sqrt(5.0) * torch.sqrt(
            torch.clamp(target_radii, min=1e-12)
        )
        target_alpha = (
            (5.0 / 3.0)
            * scale
            * (1.0 + scaled_radius)
            * torch.exp(-scaled_radius)
        )
    else:
        raise ProbeExecutionInputError(f"unsupported stress kernel: {kernel}")
    target_q_cross = (-target_alpha[:, None] * columns).reshape(-1)
    rhs = target_q_cross - value_cross @ system.function_weights
    mapped_rhs = basis_map @ system.conditional_cross
    alpha_values = torch.cholesky_solve(
        values.unsqueeze(1),
        system.function_cholesky,
    ).squeeze(1)
    raw_observations = (gradients @ raw).reshape(-1)
    observations = raw_observations - value_cross @ alpha_values
    mapped_observations = basis_map @ system.conditional_observation_functional
    rhs_difference = torch.max(torch.abs(rhs - mapped_rhs))
    observation_difference = torch.max(torch.abs(observations - mapped_observations))
    return operator, rhs, observations, {
        "rhs_support_map_maxabs_difference": _finite_float(
            rhs_difference,
            "full-q RHS support-map difference",
        ),
        "conditional_observation_support_map_maxabs_difference": _finite_float(
            observation_difference,
            "full-q observation support-map difference",
        ),
    }


def _rademacher_probes(
    *,
    rows: int,
    columns: int,
    binding: str,
    geometry_reference: str,
) -> torch.Tensor:
    total = STRESS_FIXED_PROBE_COUNT * rows * columns
    bits: list[float] = []
    counter = 0
    prefix = (
        f"{STRESS_PROBE_HASH_DOMAIN}|{binding}|{geometry_reference}|"
    ).encode("ascii")
    while len(bits) < total:
        digest = hashlib.sha256(prefix + str(counter).encode("ascii")).digest()
        for byte in digest:
            bits.extend(1.0 if (byte >> shift) & 1 else -1.0 for shift in range(8))
        counter += 1
    probes = torch.tensor(bits[:total], dtype=torch.float64).reshape(
        STRESS_FIXED_PROBE_COUNT,
        rows,
        columns,
    )
    return probes / math.sqrt(rows * columns)


def _norm(value: torch.Tensor) -> float:
    return _finite_float(torch.linalg.vector_norm(value), "norm")


def _support_complement_test(
    system: LocalValueSystem,
    x_condition: torch.Tensor,
    values: torch.Tensor,
    gradients: torch.Tensor,
    target: torch.Tensor,
    projector: torch.Tensor,
    *,
    lengthscale: torch.Tensor,
    outputscale: float,
    kernel: str,
    gradient_noise: float,
    stress_binding: str,
    geometry_reference: str,
) -> dict[str, Any]:
    operator, rhs, observations, mapping_errors = _full_q_stress_state(
        system,
        x_condition,
        values,
        gradients,
        target,
        lengthscale=lengthscale,
        outputscale=outputscale,
        kernel=kernel,
        gradient_noise=gradient_noise,
    )
    m = x_condition.shape[0]
    projector = projector.to(dtype=torch.float64)

    def split(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        matrix = value.reshape(m, m)
        support = matrix @ projector
        return support.reshape(-1), (matrix - support).reshape(-1)

    rhs_support, rhs_complement = split(rhs)
    obs_support, obs_complement = split(observations)
    probes = _rademacher_probes(
        rows=m,
        columns=m,
        binding=stress_binding,
        geometry_reference=geometry_reference,
    )
    records: list[dict[str, float]] = []
    for probe in probes:
        probe_support, probe_complement = split(probe.reshape(-1))
        action_support = operator.matmul(probe_support)
        action_complement = operator.matmul(probe_complement)
        ss, cs = split(action_support)
        sc, cc = split(action_complement)
        records.append(
            {
                "input_support_norm": _norm(probe_support),
                "input_complement_norm": _norm(probe_complement),
                "support_to_support_action_norm": _norm(ss),
                "support_to_complement_action_norm": _norm(cs),
                "complement_to_support_action_norm": _norm(sc),
                "complement_to_complement_action_norm": _norm(cc),
            }
        )
    return {
        "probe_count": STRESS_FIXED_PROBE_COUNT,
        "probe_hash_domain": STRESS_PROBE_HASH_DOMAIN,
        "probe_sha256": _tensor_sha256(probes),
        "q_jitter": SOURCE_REDUCED_JITTER,
        "rhs_support_norm": _norm(rhs_support),
        "rhs_complement_norm": _norm(rhs_complement),
        "conditional_observation_support_norm": _norm(obs_support),
        "conditional_observation_complement_norm": _norm(obs_complement),
        "support_map_differences": mapping_errors,
        "probes": records,
    }


def _build_and_solve(
    x: torch.Tensor,
    values: torch.Tensor,
    gradients: torch.Tensor,
    target: torch.Tensor,
    *,
    lengthscale: torch.Tensor,
    outputscale: float,
    sigma_f: float,
    sigma_g: float,
    kernel: str,
    gradient_noise_model: str,
    cutoff: float,
    max_iterations: int,
) -> tuple[LocalValueSystem, LocalPrediction]:
    system = build_local_value_system(
        x,
        values,
        gradients,
        target,
        lengthscale=lengthscale,
        outputscale=outputscale,
        value_noise_variance=sigma_f,
        gradient_noise_variance=sigma_g,
        kernel=kernel,
        gradient_noise_model=gradient_noise_model,
        absolute_rank_cutoff=cutoff,
        function_jitter=SOURCE_FUNCTION_JITTER,
        reduced_jitter=SOURCE_REDUCED_JITTER,
        build_preconditioner=True,
    )
    prediction = solve_local_value_system(
        system,
        tolerance=STRESS_TOLERANCE,
        max_iterations=max_iterations,
        use_preconditioner=True,
    )
    return system, prediction


def _permutation_test(
    base_system: LocalValueSystem,
    base_prediction: LocalPrediction,
    x: torch.Tensor,
    values: torch.Tensor,
    gradients: torch.Tensor,
    target: torch.Tensor,
    *,
    cutoff: float,
    max_iterations: int,
    parameters: Any,
) -> dict[str, Any]:
    permutation = torch.arange(x.shape[0] - 1, -1, -1, device=x.device)
    system, prediction = _build_and_solve(
        x[permutation],
        values[permutation],
        gradients[permutation],
        target,
        lengthscale=parameters.lengthscale,
        outputscale=parameters.outputscale,
        sigma_f=parameters.sigma_f,
        sigma_g=parameters.sigma_g,
        kernel=parameters.kernel,
        gradient_noise_model=parameters.gradient_noise_model,
        cutoff=cutoff,
        max_iterations=max_iterations,
    )
    expected_projector = _q_projector(base_system.geometry)[permutation][:, permutation]
    observed_projector = _q_projector(system.geometry)
    base_q_solution = _q_space_solution(base_system, base_prediction)
    observed_q_solution = _q_space_solution(system, prediction)
    expected_q_solution = base_q_solution[permutation][:, permutation]
    return {
        "rule": STRESS_PERMUTATION_RULE,
        "permutation": [int(value) for value in permutation.tolist()],
        "rank": system.geometry.rank,
        "projector": projector_metrics(
            expected_projector,
            observed_projector,
            reference_rank=base_system.geometry.rank,
            candidate_rank=system.geometry.rank,
        ),
        "q_solution_maxabs_difference": _finite_float(
            torch.max(torch.abs(observed_q_solution - expected_q_solution)),
            "permuted q solution difference",
        ),
        "moments": _moment_difference(base_prediction, prediction),
        "solve": _solve_record(prediction),
    }


def _givens_rotation(rank: int, *, like: torch.Tensor) -> torch.Tensor:
    rotation = torch.eye(rank, dtype=like.dtype, device=like.device)
    for start in range(0, rank - 1, 2):
        rotation[start, start] = _SQRT_HALF
        rotation[start, start + 1] = -_SQRT_HALF
        rotation[start + 1, start] = _SQRT_HALF
        rotation[start + 1, start + 1] = _SQRT_HALF
    return rotation


def _support_rotation_test(
    base_system: LocalValueSystem,
    base_prediction: LocalPrediction,
    *,
    max_iterations: int,
) -> dict[str, Any]:
    operator = base_system.operator
    if operator is None or base_system.preconditioner is None:
        raise ProbeExecutionEvidenceError("support rotation requires a positive-rank system")
    rotation = _givens_rotation(base_system.geometry.rank, like=operator.coordinates)
    rotated_geometry = replace(
        base_system.geometry,
        coordinates=base_system.geometry.coordinates @ rotation,
        q_to_z=base_system.geometry.q_to_z @ rotation,
    )
    rotated_operator = OrthonormalReducedOperator(
        rotated_geometry.coordinates,
        operator.alpha,
        operator.beta,
        operator.function_cholesky,
        rotation.T @ operator.gradient_noise @ rotation,
        jitter=operator.jitter,
    )

    def rotate_blocks(value: torch.Tensor) -> torch.Tensor:
        return (value.reshape(-1, rotation.shape[0]) @ rotation).reshape(-1)

    rotated_system = replace(
        base_system,
        geometry=rotated_geometry,
        operator=rotated_operator,
        preconditioner=ReducedKroneckerPreconditioner(rotated_operator),
        conditional_cross=rotate_blocks(base_system.conditional_cross),
        orthonormal_observations=rotate_blocks(base_system.orthonormal_observations),
        conditional_observation_functional=rotate_blocks(
            base_system.conditional_observation_functional
        ),
    )
    prediction = solve_local_value_system(
        rotated_system,
        tolerance=STRESS_TOLERANCE,
        max_iterations=max_iterations,
        use_preconditioner=True,
    )
    expected_solution = rotate_blocks(base_prediction.solve.solution)
    return {
        "rule": STRESS_SUPPORT_ROTATION_RULE,
        "rotation_orthogonality_maxabs": _finite_float(
            torch.max(torch.abs(rotation.T @ rotation - torch.eye(
                rotation.shape[0], dtype=rotation.dtype, device=rotation.device
            ))),
            "rotation orthogonality",
        ),
        "projector_maxabs_difference": _finite_float(
            torch.max(
                torch.abs(
                    _q_projector(rotated_geometry) - _q_projector(base_system.geometry)
                )
            ),
            "rotated projector difference",
        ),
        "solution_equivariance_maxabs_difference": _finite_float(
            torch.max(torch.abs(prediction.solve.solution - expected_solution)),
            "rotated solution difference",
        ),
        "moments": _moment_difference(base_prediction, prediction),
        "solve": _solve_record(prediction),
    }


def _zero_augmentation_test(
    base_system: LocalValueSystem,
    base_prediction: LocalPrediction,
    x: torch.Tensor,
    values: torch.Tensor,
    gradients: torch.Tensor,
    target: torch.Tensor,
    *,
    cutoff: float,
    max_iterations: int,
    parameters: Any,
) -> dict[str, Any]:
    augmented_x = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
    augmented_target = torch.cat([target, torch.zeros_like(target[:, :1])], dim=1)
    augmented_gradients = torch.cat(
        [gradients, torch.zeros_like(gradients[:, :1])],
        dim=1,
    )
    system, prediction = _build_and_solve(
        augmented_x,
        values,
        augmented_gradients,
        augmented_target,
        lengthscale=parameters.lengthscale,
        outputscale=parameters.outputscale,
        sigma_f=parameters.sigma_f,
        sigma_g=parameters.sigma_g,
        kernel=parameters.kernel,
        gradient_noise_model=parameters.gradient_noise_model,
        cutoff=cutoff,
        max_iterations=max_iterations,
    )
    return {
        "rule": "append_one_exact_zero_ambient_coordinate_and_gradient",
        "cutoff_held_exactly": cutoff,
        "original_ambient_dimension": x.shape[1],
        "augmented_ambient_dimension": augmented_x.shape[1],
        "original_rank": base_system.geometry.rank,
        "augmented_rank": system.geometry.rank,
        "projector": projector_metrics(
            _q_projector(base_system.geometry),
            _q_projector(system.geometry),
            reference_rank=base_system.geometry.rank,
            candidate_rank=system.geometry.rank,
        ),
        "moments": _moment_difference(base_prediction, prediction),
        "solve": _solve_record(prediction),
    }


def _discarded_mode_test(
    base_system: LocalValueSystem,
    base_prediction: LocalPrediction,
    x: torch.Tensor,
    values: torch.Tensor,
    gradients: torch.Tensor,
    target: torch.Tensor,
    *,
    source_cutoff: float,
    max_iterations: int,
    parameters: Any,
) -> tuple[dict[str, Any], torch.Tensor]:
    native_cutoff_tensor = base_system.geometry.native_singular_value_cutoff
    if native_cutoff_tensor is None:
        raise ProbeExecutionEvidenceError("base stress geometry lacks native fp64 cutoff")
    native_cutoff = float(native_cutoff_tensor)
    system, prediction = _build_and_solve(
        x,
        values,
        gradients,
        target,
        lengthscale=parameters.lengthscale,
        outputscale=parameters.outputscale,
        sigma_f=parameters.sigma_f,
        sigma_g=parameters.sigma_g,
        kernel=parameters.kernel,
        gradient_noise_model=parameters.gradient_noise_model,
        cutoff=native_cutoff,
        max_iterations=max_iterations,
    )
    selected_projector = _q_projector(base_system.geometry)
    native_projector = _q_projector(system.geometry)
    projector_difference = native_projector - selected_projector
    record = {
        "source_fp32_cutoff": source_cutoff,
        "native_fp64_cutoff": native_cutoff,
        "source_selected_rank": base_system.geometry.rank,
        "native_fp64_rank": system.geometry.rank,
        "additional_native_modes": system.geometry.rank - base_system.geometry.rank,
        "projector_difference_frobenius_norm": _finite_float(
            torch.linalg.matrix_norm(projector_difference, ord="fro"),
            "discarded projector difference",
        ),
        "projector_difference_spectral_norm": _finite_float(
            torch.linalg.matrix_norm(projector_difference, ord=2),
            "discarded projector spectral difference",
        ),
        "moments": _moment_difference(base_prediction, prediction),
        "solve": _solve_record(prediction),
    }
    return record, native_projector


def execute_registered_stress_target(
    source_arm: RegisteredOrbitArmInputs,
    promoted_arm: RegisteredOrbitArmInputs,
    stress: RegisteredStressInputs,
    source_geometry: RegisteredStressGeometry,
    strata: RegisteredStressStrata,
) -> StressTargetExecution:
    """Execute the five registered tests on the sole worst stress target."""

    if type(source_arm) is not RegisteredOrbitArmInputs or type(
        promoted_arm
    ) is not RegisteredOrbitArmInputs:
        raise ProbeExecutionInputError("stress execution requires registered fp32/fp64 arms")
    stress.assert_unchanged(source_arm)
    source_arm.assert_unchanged()
    promoted_arm.assert_unchanged()
    if (
        promoted_arm.train.X.dtype != torch.float64
        or promoted_arm.train.X.device.type != "cpu"
        or promoted_arm.binding_kind != "exact_promotion_of_bound_source_fp32_arm"
        or promoted_arm.source_arm_binding_sha256 != source_arm.source_arm_binding_sha256
        or promoted_arm.work_plan != source_arm.work_plan
    ):
        raise ProbeExecutionInputError("stress fp64 arm is not the exact CPU promotion")
    position = _validate_target_position(
        source_geometry.target_position,
        source_arm.evaluation.X.shape[0],
    )
    cutoff, selected_rank, reference = _validate_stress_geometry(
        source_arm,
        stress,
        source_geometry,
        target_position=position,
    )
    _validate_stress_strata(source_arm, stress, source_geometry, strata)
    max_iterations = source_arm.work_plan.stress_max_iterations
    if max_iterations is None:
        raise ProbeExecutionEvidenceError("registered stress iteration cap is unavailable")

    neighbours = stress.fixed_neighbours.positions[position]
    neighbour_sources = stress.fixed_neighbours.source_indices[position]
    x = promoted_arm.train.X[neighbours]
    values = promoted_arm.train.value[neighbours]
    gradients = promoted_arm.train.gradient[neighbours]
    target = promoted_arm.evaluation.X[position].unsqueeze(0)
    parameters = promoted_arm.parameters
    with torch.inference_mode():
        base_system, base_prediction = _build_and_solve(
            x,
            values,
            gradients,
            target,
            lengthscale=parameters.lengthscale,
            outputscale=parameters.outputscale,
            sigma_f=parameters.sigma_f,
            sigma_g=parameters.sigma_g,
            kernel=parameters.kernel,
            gradient_noise_model=parameters.gradient_noise_model,
            cutoff=cutoff,
            max_iterations=max_iterations,
        )
        if base_system.geometry.rank != selected_rank:
            raise ProbeExecutionEvidenceError(
                "promoted stress system did not preserve the source-selected rank"
            )
        source_projector = _q_projector(source_geometry.geometry)
        promoted_projector = _q_projector(base_system.geometry)
        source_projector_metrics = projector_metrics(
            source_projector,
            promoted_projector,
            reference_rank=selected_rank,
            candidate_rank=base_system.geometry.rank,
        )
        support_complement = _support_complement_test(
            base_system,
            x,
            values,
            gradients,
            target,
            source_projector,
            lengthscale=parameters.lengthscale,
            outputscale=parameters.outputscale,
            kernel=parameters.kernel,
            gradient_noise=parameters.sigma_g,
            stress_binding=stress.stress_binding_sha256,
            geometry_reference=reference,
        )
        permutation = _permutation_test(
            base_system,
            base_prediction,
            x,
            values,
            gradients,
            target,
            cutoff=cutoff,
            max_iterations=max_iterations,
            parameters=parameters,
        )
        support_rotation = _support_rotation_test(
            base_system,
            base_prediction,
            max_iterations=max_iterations,
        )
        zero_augmentation = _zero_augmentation_test(
            base_system,
            base_prediction,
            x,
            values,
            gradients,
            target,
            cutoff=cutoff,
            max_iterations=max_iterations,
            parameters=parameters,
        )
        discarded, native_projector = _discarded_mode_test(
            base_system,
            base_prediction,
            x,
            values,
            gradients,
            target,
            source_cutoff=cutoff,
            max_iterations=max_iterations,
            parameters=parameters,
        )

    stress.assert_unchanged(source_arm)
    source_arm.assert_unchanged()
    promoted_arm.assert_unchanged()
    post_cutoff, post_rank, post_reference = _validate_stress_geometry(
        source_arm,
        stress,
        source_geometry,
        target_position=position,
    )
    _validate_stress_strata(source_arm, stress, source_geometry, strata)
    if (post_cutoff, post_rank, post_reference) != (cutoff, selected_rank, reference):
        raise ProbeExecutionEvidenceError("stress identities changed during execution")

    tests = {
        "support_complement": support_complement,
        "permutation": permutation,
        "support_rotation": support_rotation,
        "exact_zero_augmentation": zero_augmentation,
        "discarded_mode_leakage": discarded,
    }
    base_record = _solve_record(base_prediction)
    base_record["source_to_promoted_projector"] = source_projector_metrics
    return StressTargetExecution(
        task_index=source_arm.work_plan.task_index,
        source_arm_binding_sha256=source_arm.source_arm_binding_sha256,
        stress_binding_sha256=stress.stress_binding_sha256,
        source_rank_reference_sha256=reference,
        source_rank_grid_sha256=strata.source_rank_grid_sha256,
        strata_selection_sha256=strata.selection_sha256,
        target_position=position,
        target_source_index=int(source_arm.evaluation.source_indices[position]),
        neighbour_positions=neighbours.detach().clone().contiguous(),
        neighbour_source_indices=neighbour_sources.detach().clone().contiguous(),
        m=stress.m,
        selected_rank=selected_rank,
        requested_tolerance=STRESS_TOLERANCE,
        max_iterations=max_iterations,
        base_solve=base_record,
        tests=tests,
        source_q_projector=source_projector.detach().clone().contiguous(),
        native_fp64_q_projector=native_projector.detach().clone().contiguous(),
        _construction_token=_STRESS_EXECUTION_TOKEN,
    )


__all__ = [
    "RegisteredStressGeometry",
    "RegisteredStressInputs",
    "RegisteredStressStrata",
    "StressTargetExecution",
    "build_registered_stress_inputs",
    "execute_registered_stress_target",
    "scan_registered_stress_geometry",
    "select_registered_stress_stratum",
]
