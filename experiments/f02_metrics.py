"""Frozen scoring primitives for the F02 confirmatory N-body experiment.

All scalar targets passed here are already in canonical train-standardized
units.  Variances must likewise be in those units, and callers must record
whether they are latent-function or observation variances.  Invalid variances
are rejected; they are never clipped or silently omitted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

DEFAULT_BOOTSTRAP_DRAWS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260803
DEFAULT_INTERVAL_LEVELS = (0.50, 0.90, 0.95)


class MetricInputError(ValueError):
    """Raised when F02 metrics cannot be computed without hiding a failure."""


@dataclass(frozen=True, slots=True, order=True)
class TrajectoryKey:
    """One fixed-mass replica, state dimension, and held-out trajectory."""

    replica: int
    dimension: int
    trajectory_id: int


@dataclass(frozen=True, slots=True)
class TrajectoryMetrics:
    """Scalar metrics computed before pooling correlated trajectory rows."""

    key: TrajectoryKey
    n_points: int
    standardized_mse: float
    gaussian_nll: float
    interval_coverage: tuple[tuple[float, float], ...]

    @property
    def standardized_rmse(self) -> float:
        return math.sqrt(self.standardized_mse)

    def coverage(self, level: float) -> float:
        for stored_level, value in self.interval_coverage:
            if stored_level == level:
                return value
        raise KeyError(level)


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """Point estimate and equal-tailed bootstrap confidence interval."""

    estimate: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    """Candidate-minus-reference changes; negative values are beneficial."""

    draws: int
    seed: int
    confidence_level: float
    n_replicas: int
    n_dimensions: int
    n_trajectories: int
    rmse_absolute_change: BootstrapInterval
    rmse_relative_change: BootstrapInterval
    nll_absolute_change: BootstrapInterval


def _finite_vectors(*values: np.ndarray | Sequence[float]) -> tuple[np.ndarray, ...]:
    try:
        arrays = tuple(np.asarray(value, dtype=np.float64) for value in values)
    except (TypeError, ValueError) as error:
        raise MetricInputError("metric inputs must be numeric arrays") from error
    if not arrays or arrays[0].ndim != 1 or arrays[0].size == 0:
        raise MetricInputError("metric inputs must be nonempty one-dimensional arrays")
    shape = arrays[0].shape
    for value in arrays:
        if value.ndim != 1 or value.shape != shape:
            raise MetricInputError("metric inputs must have identical one-dimensional shapes")
        if not np.isfinite(value).all():
            raise MetricInputError("metric inputs must be finite")
    return arrays


def _validated_observations(
    target: np.ndarray | Sequence[float],
    mean: np.ndarray | Sequence[float],
    variance: np.ndarray | Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_array, mean_array, variance_array = _finite_vectors(target, mean, variance)
    if np.any(variance_array <= 0.0):
        raise MetricInputError("Gaussian variances must be strictly positive")
    return target_array, mean_array, variance_array


def standardized_rmse(
    target: np.ndarray | Sequence[float],
    prediction: np.ndarray | Sequence[float],
) -> float:
    """Root mean square scalar error in canonical standardized units."""

    target_array, prediction_array = _finite_vectors(target, prediction)
    result = float(np.sqrt(np.mean((prediction_array - target_array) ** 2)))
    if not math.isfinite(result):
        raise MetricInputError("standardized RMSE is nonfinite")
    return result


def standardized_mse(
    target: np.ndarray | Sequence[float],
    prediction: np.ndarray | Sequence[float],
) -> float:
    """Mean square scalar error in canonical standardized units."""

    target_array, prediction_array = _finite_vectors(target, prediction)
    result = float(np.mean((prediction_array - target_array) ** 2))
    if not math.isfinite(result):
        raise MetricInputError("standardized MSE is nonfinite")
    return result


def gaussian_nll(
    target: np.ndarray | Sequence[float],
    mean: np.ndarray | Sequence[float],
    variance: np.ndarray | Sequence[float],
) -> float:
    """Mean Gaussian NLL in nat/point, without variance clipping.

    The caller, not this formula, determines whether ``variance`` is latent or
    observational and must preserve that label in the surrounding result row.
    """

    target_array, mean_array, variance_array = _validated_observations(
        target,
        mean,
        variance,
    )
    squared_error = (target_array - mean_array) ** 2
    values = 0.5 * (np.log(2.0 * math.pi * variance_array) + squared_error / variance_array)
    if not np.isfinite(values).all():
        raise MetricInputError("Gaussian NLL is nonfinite")
    return float(np.mean(values))


def interval_coverage(
    target: np.ndarray | Sequence[float],
    mean: np.ndarray | Sequence[float],
    variance: np.ndarray | Sequence[float],
    *,
    level: float,
) -> float:
    """Empirical coverage of a central Gaussian interval."""

    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise MetricInputError("interval level must lie strictly between zero and one")
    target_array, mean_array, variance_array = _validated_observations(
        target,
        mean,
        variance,
    )
    quantile = NormalDist().inv_cdf(0.5 * (1.0 + level))
    covered = np.abs(target_array - mean_array) <= quantile * np.sqrt(variance_array)
    return float(np.mean(covered))


def per_trajectory_metrics(
    target: np.ndarray | Sequence[float],
    mean: np.ndarray | Sequence[float],
    variance: np.ndarray | Sequence[float],
    trajectory_id: np.ndarray | Sequence[int],
    *,
    replica: int,
    dimension: int,
    interval_levels: Sequence[float] = DEFAULT_INTERVAL_LEVELS,
) -> tuple[TrajectoryMetrics, ...]:
    """Compute metrics independently for every complete held-out trajectory."""

    target_array, mean_array, variance_array = _validated_observations(
        target,
        mean,
        variance,
    )
    trajectory_array = np.asarray(trajectory_id)
    if trajectory_array.ndim != 1 or trajectory_array.shape != target_array.shape:
        raise MetricInputError("trajectory_id must match the scalar target shape")
    if not np.issubdtype(trajectory_array.dtype, np.integer):
        raise MetricInputError("trajectory_id must have an integer dtype")
    if not isinstance(replica, int) or not isinstance(dimension, int) or dimension <= 0:
        raise MetricInputError("replica and positive dimension must be integers")

    levels = tuple(float(level) for level in interval_levels)
    if not levels or len(set(levels)) != len(levels):
        raise MetricInputError("interval_levels must be nonempty and unique")
    for level in levels:
        if not math.isfinite(level) or not 0.0 < level < 1.0:
            raise MetricInputError("every interval level must lie in (0, 1)")

    result: list[TrajectoryMetrics] = []
    for trajectory in np.unique(trajectory_array):
        mask = trajectory_array == trajectory
        group_target = target_array[mask]
        group_mean = mean_array[mask]
        group_variance = variance_array[mask]
        coverage = tuple(
            (
                level,
                interval_coverage(
                    group_target,
                    group_mean,
                    group_variance,
                    level=level,
                ),
            )
            for level in levels
        )
        result.append(
            TrajectoryMetrics(
                key=TrajectoryKey(
                    replica=replica,
                    dimension=dimension,
                    trajectory_id=int(trajectory),
                ),
                n_points=int(mask.sum()),
                standardized_mse=standardized_mse(group_target, group_mean),
                gaussian_nll=gaussian_nll(
                    group_target,
                    group_mean,
                    group_variance,
                ),
                interval_coverage=coverage,
            )
        )
    return tuple(result)


def _validated_metric_map(
    records: Sequence[TrajectoryMetrics],
    label: str,
) -> dict[TrajectoryKey, TrajectoryMetrics]:
    if not records:
        raise MetricInputError(f"{label} trajectory metrics are empty")
    result: dict[TrajectoryKey, TrajectoryMetrics] = {}
    for record in records:
        key_parts = (record.key.replica, record.key.dimension, record.key.trajectory_id)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in key_parts
        ):
            raise MetricInputError(f"{label} trajectory keys must contain integers")
        if record.key.replica < 0 or record.key.dimension <= 0 or record.key.trajectory_id < 0:
            raise MetricInputError(f"{label} trajectory key is outside the valid range")
        if record.key in result:
            raise MetricInputError(f"duplicate {label} trajectory key: {record.key}")
        if (
            isinstance(record.n_points, bool)
            or not isinstance(record.n_points, (int, np.integer))
            or record.n_points <= 0
        ):
            raise MetricInputError(f"{label} trajectory group sizes must be positive integers")
        if not math.isfinite(record.standardized_mse) or record.standardized_mse < 0.0:
            raise MetricInputError(f"{label} MSE values must be finite and nonnegative")
        if not math.isfinite(record.gaussian_nll):
            raise MetricInputError(f"{label} NLL values must be finite")
        result[record.key] = record
    return result


def _paired_hierarchy(
    reference: Sequence[TrajectoryMetrics],
    candidate: Sequence[TrajectoryMetrics],
) -> dict[int, dict[int, np.ndarray]]:
    reference_map = _validated_metric_map(reference, "reference")
    candidate_map = _validated_metric_map(candidate, "candidate")
    reference_keys = set(reference_map)
    candidate_keys = set(candidate_map)
    if reference_keys != candidate_keys:
        missing_candidate = sorted(reference_keys - candidate_keys)
        missing_reference = sorted(candidate_keys - reference_keys)
        raise MetricInputError(
            "paired trajectory keys differ; "
            f"missing candidate={missing_candidate}, missing reference={missing_reference}"
        )
    for key in reference_keys:
        if reference_map[key].n_points != candidate_map[key].n_points:
            raise MetricInputError(f"paired trajectory sizes differ for {key}")
        if reference_map[key].standardized_mse <= 0.0:
            raise MetricInputError(
                "relative RMSE bootstrap requires positive reference MSE in every trajectory"
            )
    hierarchy_lists: dict[int, dict[int, list[tuple[float, float, float, float]]]] = {}
    for key in sorted(reference_keys):
        dimensions = hierarchy_lists.setdefault(key.replica, {})
        dimensions.setdefault(key.dimension, []).append(
            (
                reference_map[key].standardized_mse,
                candidate_map[key].standardized_mse,
                reference_map[key].gaussian_nll,
                candidate_map[key].gaussian_nll,
            )
        )
    dimension_sets = {tuple(sorted(dimensions)) for dimensions in hierarchy_lists.values()}
    if len(dimension_sets) != 1:
        raise MetricInputError("every replica must contain the same state dimensions")

    hierarchy = {
        replica: {
            dimension: np.asarray(values, dtype=np.float64)
            for dimension, values in dimensions.items()
        }
        for replica, dimensions in hierarchy_lists.items()
    }
    return hierarchy


def _finite_column_mean(values: np.ndarray, label: str) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.mean(values, axis=0)
    if not np.isfinite(result).all():
        raise MetricInputError(f"{label} aggregation is nonfinite")
    return result


def _dimension_metrics(trajectory_values: np.ndarray) -> np.ndarray:
    means = _finite_column_mean(trajectory_values, "trajectory")
    result = np.asarray(
        (math.sqrt(means[0]), math.sqrt(means[1]), means[2], means[3]),
        dtype=np.float64,
    )
    if not np.isfinite(result).all():
        raise MetricInputError("dimension metric is nonfinite")
    return result


def _hierarchical_point_metrics(
    hierarchy: dict[int, dict[int, np.ndarray]],
) -> np.ndarray:
    replica_values: list[np.ndarray] = []
    for replica in sorted(hierarchy):
        dimension_values: list[np.ndarray] = []
        for dimension in sorted(hierarchy[replica]):
            dimension_values.append(_dimension_metrics(hierarchy[replica][dimension]))
        replica_values.append(
            _finite_column_mean(np.asarray(dimension_values), "dimension")
        )
    return _finite_column_mean(np.asarray(replica_values), "replica")


def _one_hierarchical_draw(
    rng: np.random.Generator,
    hierarchy: dict[int, dict[int, np.ndarray]],
) -> np.ndarray:
    replicas = np.asarray(sorted(hierarchy), dtype=np.int64)
    sampled_replicas = rng.choice(replicas, size=replicas.size, replace=True)
    replica_values: list[np.ndarray] = []
    for replica_value in sampled_replicas:
        replica = int(replica_value)
        dimensions = np.asarray(sorted(hierarchy[replica]), dtype=np.int64)
        sampled_dimensions = rng.choice(dimensions, size=dimensions.size, replace=True)
        dimension_values: list[np.ndarray] = []
        for dimension_value in sampled_dimensions:
            dimension = int(dimension_value)
            trajectory_values = hierarchy[replica][dimension]
            sampled_indices = rng.integers(
                trajectory_values.shape[0],
                size=trajectory_values.shape[0],
            )
            dimension_values.append(_dimension_metrics(trajectory_values[sampled_indices]))
        replica_values.append(
            _finite_column_mean(np.asarray(dimension_values), "dimension")
        )
    return _finite_column_mean(np.asarray(replica_values), "replica")


def _changes(values: np.ndarray) -> np.ndarray:
    if not np.isfinite(values).all():
        raise MetricInputError("paired aggregate metrics are nonfinite")
    reference_rmse, candidate_rmse, reference_nll, candidate_nll = values
    if reference_rmse <= 0.0:
        raise MetricInputError("relative RMSE change requires positive reference RMSE")
    result = np.asarray(
        (
            candidate_rmse - reference_rmse,
            candidate_rmse / reference_rmse - 1.0,
            candidate_nll - reference_nll,
        ),
        dtype=np.float64,
    )
    if not np.isfinite(result).all():
        raise MetricInputError("paired metric changes are nonfinite")
    return result


def _interval(
    estimate: float,
    draws: np.ndarray,
    confidence_level: float,
) -> BootstrapInterval:
    if not math.isfinite(estimate) or draws.ndim != 1 or not np.isfinite(draws).all():
        raise MetricInputError("bootstrap interval inputs must be finite")
    tail = 0.5 * (1.0 - confidence_level)
    lower, upper = np.quantile(draws, (tail, 1.0 - tail))
    return BootstrapInterval(float(estimate), float(lower), float(upper))


def paired_hierarchical_bootstrap(
    reference: Sequence[TrajectoryMetrics],
    candidate: Sequence[TrajectoryMetrics],
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = 0.95,
) -> PairedBootstrapResult:
    """Paired replica -> dimension -> trajectory bootstrap for F02.

    Each sampled hierarchy key is applied to both arms.  Trajectory MSE is
    averaged within a sampled dimension before taking its square root; the
    resulting dimension RMSEs are then averaged across dimensions and replicas.
    NLL is averaged at every level.  Relative RMSE is the ratio of the two
    draw-level aggregated RMSEs minus one.
    """

    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise MetricInputError("draws must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise MetricInputError("seed must be an integer")
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise MetricInputError("confidence_level must lie in (0, 1)")

    hierarchy = _paired_hierarchy(reference, candidate)
    point_changes = _changes(_hierarchical_point_metrics(hierarchy))
    rng = np.random.default_rng(seed)
    bootstrap_changes = np.empty((draws, 3), dtype=np.float64)
    for draw in range(draws):
        bootstrap_changes[draw] = _changes(_one_hierarchical_draw(rng, hierarchy))

    dimensions = next(iter(hierarchy.values()))
    return PairedBootstrapResult(
        draws=draws,
        seed=seed,
        confidence_level=confidence_level,
        n_replicas=len(hierarchy),
        n_dimensions=len(dimensions),
        n_trajectories=sum(
            trajectory_values.shape[0]
            for replica_dimensions in hierarchy.values()
            for trajectory_values in replica_dimensions.values()
        ),
        rmse_absolute_change=_interval(
            point_changes[0],
            bootstrap_changes[:, 0],
            confidence_level,
        ),
        rmse_relative_change=_interval(
            point_changes[1],
            bootstrap_changes[:, 1],
            confidence_level,
        ),
        nll_absolute_change=_interval(
            point_changes[2],
            bootstrap_changes[:, 2],
            confidence_level,
        ),
    )
