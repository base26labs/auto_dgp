"""Pure, fail-closed selection rules for F02 ORBIT neighbourhood sizes.

This module consumes already-scored *development-validation* records.  It does
not load a corpus and has no test-split entry point.  In particular, optimizer
seeds are averaged inside each fixed-mass corpus; they are not treated as
independent experimental units in the resource bootstrap.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

REFERENCE_M = 50
DEFAULT_REPLICAS = (0, 1, 2)
DEFAULT_SEEDS = (11, 29, 47)
DEFAULT_DIMENSIONS = (12, 24, 36, 48, 60)
DEFAULT_BOOTSTRAP_DRAWS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20_260_803
DEFAULT_RESIDUAL_TOLERANCE = 1e-5
DEFAULT_RMSE_SLACK = 0.01
TERA_STATE_ENVELOPE = REFERENCE_M**4
TERA_FLOP_ENVELOPE = REFERENCE_M**6 / 3.0


class ResourceSelectionError(ValueError):
    """Raised when a complete preregistered selection cannot be made."""


@dataclass(frozen=True, slots=True)
class ValidationTrajectoryRecord:
    """One candidate's scalar metrics for one held-out trajectory."""

    dimension: int
    replica: int
    seed: int
    candidate_m: int
    trajectory_id: int
    n_points: int
    standardized_mse: float
    latent_gaussian_nll: float


@dataclass(frozen=True, slots=True)
class ValidationTargetResource:
    """One candidate's solver/resource diagnostics for one prediction target."""

    dimension: int
    replica: int
    seed: int
    candidate_m: int
    trajectory_id: int
    source_index: int
    counted_flops: int
    operator_core_elements: int
    fresh_relative_residual: float
    converged: bool
    basis_exact: bool
    raw_latent_variance: float


@dataclass(frozen=True, slots=True)
class CandidateResourceSummary:
    """All preregistered metrics and gates for one ``(dimension, m)``."""

    dimension: int
    candidate_m: int
    macro_rmse: float
    macro_latent_nll: float
    mean_counted_flops_per_target: float
    counted_flops_bootstrap_upper_95: float
    bootstrap_upper_quantile: float
    operator_core_elements_max: int
    maximum_fresh_relative_residual: float
    all_converged: bool
    all_basis_exact: bool
    all_raw_variances_valid: bool
    eligible: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DimensionResourceSelection:
    """The frozen development-only neighbour choice for one state dimension."""

    dimension: int
    selected_m: int
    best_eligible_rmse: float
    rmse_admissibility_threshold: float
    summaries: tuple[CandidateResourceSummary, ...]


def _as_tuple(values: Iterable[int], name: str) -> tuple[int, ...]:
    result = tuple(values)
    if not result or any(isinstance(value, bool) or not isinstance(value, int) for value in result):
        raise ResourceSelectionError(f"{name} must be a nonempty integer sequence")
    if len(set(result)) != len(result):
        raise ResourceSelectionError(f"{name} must not contain duplicates")
    return result


def _validate_scalar(value: float, name: str, *, nonnegative: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite and non-negative" if nonnegative else "finite"
        raise ResourceSelectionError(f"{name} must be {qualifier}")
    return result


def _validated_records(
    dimension: int,
    candidate_m: tuple[int, ...],
    trajectories: tuple[ValidationTrajectoryRecord, ...],
    resources: tuple[ValidationTargetResource, ...],
    *,
    replicas: tuple[int, ...],
    seeds: tuple[int, ...],
    trajectories_per_corpus: int,
    points_per_trajectory: int,
) -> None:
    if dimension <= 0:
        raise ResourceSelectionError("dimension must be positive")
    if trajectories_per_corpus <= 0 or points_per_trajectory <= 0:
        raise ResourceSelectionError("expected trajectory and point counts must be positive")

    expected_tasks = {
        (replica, seed, m) for replica in replicas for seed in seeds for m in candidate_m
    }
    metric_tasks: dict[tuple[int, int, int], list[ValidationTrajectoryRecord]] = {}
    metric_keys: set[tuple[int, int, int, int]] = set()
    for record in trajectories:
        if record.dimension != dimension:
            raise ResourceSelectionError("trajectory record dimension mismatch")
        task = (record.replica, record.seed, record.candidate_m)
        if task not in expected_tasks:
            raise ResourceSelectionError("unexpected trajectory task identity")
        key = (*task, record.trajectory_id)
        if key in metric_keys:
            raise ResourceSelectionError("duplicate trajectory metric identity")
        metric_keys.add(key)
        if record.n_points != points_per_trajectory:
            raise ResourceSelectionError("trajectory metric has the wrong point count")
        _validate_scalar(record.standardized_mse, "standardized_mse", nonnegative=True)
        _validate_scalar(record.latent_gaussian_nll, "latent_gaussian_nll")
        metric_tasks.setdefault(task, []).append(record)

    resource_tasks: dict[tuple[int, int, int], list[ValidationTargetResource]] = {}
    resource_keys: set[tuple[int, int, int, int]] = set()
    for record in resources:
        if record.dimension != dimension:
            raise ResourceSelectionError("resource record dimension mismatch")
        task = (record.replica, record.seed, record.candidate_m)
        if task not in expected_tasks:
            raise ResourceSelectionError("unexpected resource task identity")
        key = (*task, record.source_index)
        if key in resource_keys:
            raise ResourceSelectionError("duplicate prediction-target resource identity")
        resource_keys.add(key)
        if (
            isinstance(record.counted_flops, bool)
            or not isinstance(record.counted_flops, (int, np.integer))
            or record.counted_flops < 0
        ):
            raise ResourceSelectionError("counted_flops must be a non-negative integer")
        if (
            isinstance(record.operator_core_elements, bool)
            or not isinstance(record.operator_core_elements, (int, np.integer))
            or record.operator_core_elements < 0
        ):
            raise ResourceSelectionError("operator_core_elements must be a non-negative integer")
        _validate_scalar(
            record.fresh_relative_residual,
            "fresh_relative_residual",
            nonnegative=True,
        )
        resource_tasks.setdefault(task, []).append(record)

    if set(metric_tasks) != expected_tasks or set(resource_tasks) != expected_tasks:
        raise ResourceSelectionError("missing metric or resource task identities")

    for task in sorted(expected_tasks):
        metric_rows = metric_tasks[task]
        resource_rows = resource_tasks[task]
        if len(metric_rows) != trajectories_per_corpus:
            raise ResourceSelectionError("task has the wrong number of trajectory metrics")
        if len(resource_rows) != trajectories_per_corpus * points_per_trajectory:
            raise ResourceSelectionError("task has the wrong number of prediction resources")
        metric_trajectory_ids = {record.trajectory_id for record in metric_rows}
        resource_counts: dict[int, int] = {}
        for record in resource_rows:
            resource_counts[record.trajectory_id] = resource_counts.get(record.trajectory_id, 0) + 1
        if set(resource_counts) != metric_trajectory_ids or any(
            count != points_per_trajectory for count in resource_counts.values()
        ):
            raise ResourceSelectionError(
                "resource trajectory identities/counts do not match trajectory metrics"
            )


def _macro_metrics(
    records: tuple[ValidationTrajectoryRecord, ...],
    *,
    replicas: tuple[int, ...],
    seeds: tuple[int, ...],
    candidate_m: int,
) -> tuple[float, float]:
    """Apply the frozen ordering: trajectory -> seed -> corpus/replica."""

    replica_rmse: list[float] = []
    replica_nll: list[float] = []
    for replica in replicas:
        seed_rmse: list[float] = []
        seed_nll: list[float] = []
        for seed in seeds:
            selected = [
                record
                for record in records
                if record.replica == replica
                and record.seed == seed
                and record.candidate_m == candidate_m
            ]
            seed_rmse.append(
                math.sqrt(float(np.mean([record.standardized_mse for record in selected])))
            )
            seed_nll.append(float(np.mean([record.latent_gaussian_nll for record in selected])))
        replica_rmse.append(float(np.mean(seed_rmse)))
        replica_nll.append(float(np.mean(seed_nll)))
    return float(np.mean(replica_rmse)), float(np.mean(replica_nll))


def _seed_averaged_trajectory_flops(
    records: tuple[ValidationTargetResource, ...],
    *,
    replicas: tuple[int, ...],
    seeds: tuple[int, ...],
    candidate_m: int,
) -> dict[int, tuple[float, ...]]:
    """Return one mean-flop number per trajectory after paired seed averaging."""

    result: dict[int, tuple[float, ...]] = {}
    for replica in replicas:
        per_seed: dict[int, dict[int, float]] = {}
        for seed in seeds:
            chosen = [
                record
                for record in records
                if record.replica == replica
                and record.seed == seed
                and record.candidate_m == candidate_m
            ]
            trajectory_values: dict[int, list[int]] = {}
            for record in chosen:
                trajectory_values.setdefault(record.trajectory_id, []).append(record.counted_flops)
            per_seed[seed] = {
                trajectory_id: float(np.mean(values))
                for trajectory_id, values in trajectory_values.items()
            }
        identities = set(per_seed[seeds[0]])
        if any(set(per_seed[seed]) != identities for seed in seeds[1:]):
            raise ResourceSelectionError("trajectory identities differ between optimizer seeds")
        result[replica] = tuple(
            float(np.mean([per_seed[seed][trajectory_id] for seed in seeds]))
            for trajectory_id in sorted(identities)
        )
    return result


def _hierarchical_flop_bootstrap(
    by_replica: dict[int, tuple[float, ...]],
    *,
    replicas: tuple[int, ...],
    draws: int,
    seed: int,
) -> tuple[float, float]:
    if draws <= 0:
        raise ResourceSelectionError("bootstrap draws must be positive")
    generator = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled_replicas = generator.choice(replicas, size=len(replicas), replace=True)
        sampled_values: list[float] = []
        for replica in sampled_replicas:
            values = np.asarray(by_replica[int(replica)], dtype=np.float64)
            indices = generator.integers(0, values.size, size=values.size)
            sampled_values.extend(values[indices].tolist())
        estimates[draw] = float(np.mean(sampled_values))
    # "upper endpoint of a 95% bootstrap interval" is conservatively treated
    # as the 97.5th percentile of the two-sided percentile interval.
    upper_quantile = 0.975
    return float(np.mean([value for values in by_replica.values() for value in values])), float(
        np.quantile(estimates, upper_quantile)
    )


def evaluate_dimension_candidates(
    dimension: int,
    candidate_m: Iterable[int],
    trajectory_records: Iterable[ValidationTrajectoryRecord],
    resource_records: Iterable[ValidationTargetResource],
    *,
    replicas: Iterable[int] = DEFAULT_REPLICAS,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    trajectories_per_corpus: int = 20,
    points_per_trajectory: int = 5,
    residual_tolerance: float = DEFAULT_RESIDUAL_TOLERANCE,
    state_envelope: int = TERA_STATE_ENVELOPE,
    flop_envelope: float = TERA_FLOP_ENVELOPE,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[CandidateResourceSummary, ...]:
    """Evaluate all four analytic eligibility gates without selecting a winner."""

    candidates = _as_tuple(candidate_m, "candidate_m")
    expected_replicas = _as_tuple(replicas, "replicas")
    expected_seeds = _as_tuple(seeds, "seeds")
    if tuple(sorted(candidates)) != candidates or any(value <= 0 for value in candidates):
        raise ResourceSelectionError("candidate_m must be strictly increasing and positive")
    if not math.isfinite(residual_tolerance) or residual_tolerance <= 0.0:
        raise ResourceSelectionError("residual_tolerance must be finite and positive")
    if state_envelope < 0 or not math.isfinite(flop_envelope) or flop_envelope < 0.0:
        raise ResourceSelectionError("resource envelopes must be non-negative and finite")

    trajectories = tuple(trajectory_records)
    resources = tuple(resource_records)
    _validated_records(
        dimension,
        candidates,
        trajectories,
        resources,
        replicas=expected_replicas,
        seeds=expected_seeds,
        trajectories_per_corpus=trajectories_per_corpus,
        points_per_trajectory=points_per_trajectory,
    )

    summaries: list[CandidateResourceSummary] = []
    for m in candidates:
        selected_resources = tuple(record for record in resources if record.candidate_m == m)
        macro_rmse, macro_nll = _macro_metrics(
            trajectories,
            replicas=expected_replicas,
            seeds=expected_seeds,
            candidate_m=m,
        )
        by_replica = _seed_averaged_trajectory_flops(
            selected_resources,
            replicas=expected_replicas,
            seeds=expected_seeds,
            candidate_m=m,
        )
        mean_flops, upper_flops = _hierarchical_flop_bootstrap(
            by_replica,
            replicas=expected_replicas,
            draws=bootstrap_draws,
            seed=bootstrap_seed,
        )
        all_converged = all(record.converged for record in selected_resources)
        all_basis_exact = all(record.basis_exact for record in selected_resources)
        all_variances_valid = all(
            math.isfinite(float(record.raw_latent_variance))
            and float(record.raw_latent_variance) > 0.0
            for record in selected_resources
        )
        max_residual = max(record.fresh_relative_residual for record in selected_resources)
        max_core = max(record.operator_core_elements for record in selected_resources)
        reasons: list[str] = []
        if not all_converged or max_residual > residual_tolerance:
            reasons.append("solver")
        if not all_basis_exact:
            reasons.append("basis")
        if not all_variances_valid:
            reasons.append("variance")
        if max_core > state_envelope:
            reasons.append("state_envelope")
        if upper_flops > flop_envelope:
            reasons.append("flop_envelope")
        summaries.append(
            CandidateResourceSummary(
                dimension=dimension,
                candidate_m=m,
                macro_rmse=macro_rmse,
                macro_latent_nll=macro_nll,
                mean_counted_flops_per_target=mean_flops,
                counted_flops_bootstrap_upper_95=upper_flops,
                bootstrap_upper_quantile=0.975,
                operator_core_elements_max=max_core,
                maximum_fresh_relative_residual=max_residual,
                all_converged=all_converged,
                all_basis_exact=all_basis_exact,
                all_raw_variances_valid=all_variances_valid,
                eligible=not reasons,
                failure_reasons=tuple(reasons),
            )
        )
    return tuple(summaries)


def select_dimension_neighbor_count(
    dimension: int,
    candidate_m: Iterable[int],
    trajectory_records: Iterable[ValidationTrajectoryRecord],
    resource_records: Iterable[ValidationTargetResource],
    *,
    rmse_slack: float = DEFAULT_RMSE_SLACK,
    **evaluation_kwargs: object,
) -> DimensionResourceSelection:
    """Apply the frozen RMSE-admissibility, NLL-minimization, and tie rule."""

    if not math.isfinite(rmse_slack) or rmse_slack < 0.0:
        raise ResourceSelectionError("rmse_slack must be finite and non-negative")
    summaries = evaluate_dimension_candidates(
        dimension,
        candidate_m,
        trajectory_records,
        resource_records,
        **evaluation_kwargs,
    )
    eligible = [summary for summary in summaries if summary.eligible]
    if not eligible:
        raise ResourceSelectionError(f"dimension {dimension} has no resource-eligible candidate_m")
    best_rmse = min(summary.macro_rmse for summary in eligible)
    threshold = best_rmse * (1.0 + rmse_slack)
    admissible = [summary for summary in eligible if summary.macro_rmse <= threshold]
    selected = min(admissible, key=lambda summary: (summary.macro_latent_nll, summary.candidate_m))
    return DimensionResourceSelection(
        dimension=dimension,
        selected_m=selected.candidate_m,
        best_eligible_rmse=best_rmse,
        rmse_admissibility_threshold=threshold,
        summaries=summaries,
    )


def select_neighbor_schedule(
    candidate_m: Iterable[int],
    trajectory_records: Iterable[ValidationTrajectoryRecord],
    resource_records: Iterable[ValidationTargetResource],
    *,
    dimensions: Iterable[int] = DEFAULT_DIMENSIONS,
    rmse_slack: float = DEFAULT_RMSE_SLACK,
    **evaluation_kwargs: object,
) -> tuple[DimensionResourceSelection, ...]:
    """Require and select one complete, fixed F02 development dimension grid."""

    expected_dimensions = _as_tuple(dimensions, "dimensions")
    if tuple(sorted(expected_dimensions)) != expected_dimensions or any(
        dimension <= 0 for dimension in expected_dimensions
    ):
        raise ResourceSelectionError("dimensions must be strictly increasing and positive")
    candidates = tuple(candidate_m)
    trajectories = tuple(trajectory_records)
    resources = tuple(resource_records)
    if {record.dimension for record in trajectories} != set(expected_dimensions):
        raise ResourceSelectionError("trajectory records do not cover the exact dimension grid")
    if {record.dimension for record in resources} != set(expected_dimensions):
        raise ResourceSelectionError("resource records do not cover the exact dimension grid")
    return tuple(
        select_dimension_neighbor_count(
            dimension,
            candidates,
            (record for record in trajectories if record.dimension == dimension),
            (record for record in resources if record.dimension == dimension),
            rmse_slack=rmse_slack,
            **evaluation_kwargs,
        )
        for dimension in expected_dimensions
    )


__all__ = [
    "CandidateResourceSummary",
    "DEFAULT_BOOTSTRAP_DRAWS",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_DIMENSIONS",
    "DimensionResourceSelection",
    "ResourceSelectionError",
    "TERA_FLOP_ENVELOPE",
    "TERA_STATE_ENVELOPE",
    "ValidationTargetResource",
    "ValidationTrajectoryRecord",
    "evaluate_dimension_candidates",
    "select_dimension_neighbor_count",
    "select_neighbor_schedule",
]
