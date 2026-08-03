"""Correctness tests for the frozen F02 scoring primitives."""

import math
from dataclasses import replace
from statistics import NormalDist

import numpy as np
import pytest

from experiments.f02_metrics import (
    DEFAULT_BOOTSTRAP_DRAWS,
    DEFAULT_BOOTSTRAP_SEED,
    MetricInputError,
    TrajectoryKey,
    TrajectoryMetrics,
    gaussian_nll,
    interval_coverage,
    paired_hierarchical_bootstrap,
    per_trajectory_metrics,
    standardized_rmse,
)


def _record(
    replica: int,
    dimension: int,
    trajectory: int,
    *,
    rmse: float,
    nll: float,
    n_points: int = 100,
) -> TrajectoryMetrics:
    return TrajectoryMetrics(
        key=TrajectoryKey(replica, dimension, trajectory),
        n_points=n_points,
        standardized_mse=rmse**2,
        gaussian_nll=nll,
        interval_coverage=((0.5, 0.5), (0.9, 0.9), (0.95, 0.95)),
    )


def test_scalar_metrics_match_known_analytic_values():
    target = np.asarray([0.0, 1.0])
    mean = np.asarray([0.0, 0.0])
    variance = np.asarray([1.0, 4.0])

    assert standardized_rmse(target, mean) == pytest.approx(math.sqrt(0.5))
    expected_nll = np.mean(
        [
            0.5 * math.log(2.0 * math.pi),
            0.5 * (math.log(8.0 * math.pi) + 0.25),
        ]
    )
    assert gaussian_nll(target, mean, variance) == pytest.approx(expected_nll)

    one_sigma_level = 2.0 * NormalDist().cdf(1.0) - 1.0
    coverage = interval_coverage(
        np.asarray([0.0, 0.5, 1.5]),
        np.zeros(3),
        np.ones(3),
        level=one_sigma_level,
    )
    assert coverage == pytest.approx(2.0 / 3.0)


@pytest.mark.parametrize("bad_variance", [0.0, -1.0, float("nan"), float("inf")])
def test_gaussian_nll_fails_closed_on_invalid_raw_variance(bad_variance):
    with pytest.raises(MetricInputError):
        gaussian_nll([0.0, 1.0], [0.0, 1.0], [1.0, bad_variance])
    with pytest.raises(MetricInputError):
        interval_coverage(
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, bad_variance],
            level=0.9,
        )


def test_rmse_fails_closed_if_finite_inputs_overflow():
    with np.errstate(over="ignore"):
        with pytest.raises(MetricInputError, match="RMSE is nonfinite"):
            standardized_rmse([1.0e308], [-1.0e308])


def test_metrics_are_computed_per_complete_trajectory_group():
    one_sigma_level = 2.0 * NormalDist().cdf(1.0) - 1.0
    grouped = per_trajectory_metrics(
        target=np.asarray([0.0, 2.0, 0.0, 0.0, 0.0]),
        mean=np.asarray([0.0, 0.0, 1.0, -1.0, 0.0]),
        variance=np.ones(5),
        trajectory_id=np.asarray([7, 7, 2, 2, 2], dtype=np.int64),
        replica=101,
        dimension=12,
        interval_levels=(one_sigma_level,),
    )

    assert tuple(record.key.trajectory_id for record in grouped) == (2, 7)
    assert tuple(record.n_points for record in grouped) == (3, 2)
    assert grouped[0].standardized_rmse == pytest.approx(math.sqrt(2.0 / 3.0))
    assert grouped[1].standardized_rmse == pytest.approx(math.sqrt(2.0))
    gaussian_constant = 0.5 * math.log(2.0 * math.pi)
    assert grouped[0].gaussian_nll == pytest.approx(gaussian_constant + 1.0 / 3.0)
    assert grouped[1].gaussian_nll == pytest.approx(gaussian_constant + 1.0)
    assert grouped[0].coverage(one_sigma_level) == 1.0
    assert grouped[1].coverage(one_sigma_level) == 0.5


def test_constant_paired_effect_has_exact_hierarchical_interval():
    reference = []
    candidate = []
    for replica in (101, 102):
        for dimension in (12, 24):
            for trajectory in (3, 9):
                reference.append(_record(replica, dimension, trajectory, rmse=2.0, nll=0.5))
                candidate.append(_record(replica, dimension, trajectory, rmse=1.0, nll=0.25))

    result = paired_hierarchical_bootstrap(reference, candidate, draws=128)
    assert DEFAULT_BOOTSTRAP_DRAWS == 10_000
    assert result.seed == DEFAULT_BOOTSTRAP_SEED
    assert result.n_replicas == 2
    assert result.n_dimensions == 2
    assert result.n_trajectories == 8
    for value in (
        result.rmse_absolute_change.estimate,
        result.rmse_absolute_change.lower,
        result.rmse_absolute_change.upper,
    ):
        assert value == pytest.approx(-1.0)
    for value in (
        result.rmse_relative_change.estimate,
        result.rmse_relative_change.lower,
        result.rmse_relative_change.upper,
    ):
        assert value == pytest.approx(-0.5)
    for value in (
        result.nll_absolute_change.estimate,
        result.nll_absolute_change.lower,
        result.nll_absolute_change.upper,
    ):
        assert value == pytest.approx(-0.25)


def test_pairing_is_by_key_and_missing_or_mismatched_pairs_fail():
    reference = [
        _record(101, 12, 1, rmse=1.0, nll=0.2),
        _record(101, 12, 2, rmse=2.0, nll=0.4),
    ]
    candidate = [
        _record(101, 12, 1, rmse=0.8, nll=0.1),
        _record(101, 12, 2, rmse=1.5, nll=0.3),
    ]
    ordered = paired_hierarchical_bootstrap(reference, candidate, draws=64, seed=17)
    reversed_order = paired_hierarchical_bootstrap(
        list(reversed(reference)),
        list(reversed(candidate)),
        draws=64,
        seed=17,
    )
    assert ordered == reversed_order

    with pytest.raises(MetricInputError, match="paired trajectory keys differ"):
        paired_hierarchical_bootstrap(reference, candidate[:-1], draws=4)
    with pytest.raises(MetricInputError, match="duplicate reference trajectory key"):
        paired_hierarchical_bootstrap(reference + reference[:1], candidate, draws=4)
    mismatched_size = [candidate[0], replace(candidate[1], n_points=99)]
    with pytest.raises(MetricInputError, match="paired trajectory sizes differ"):
        paired_hierarchical_bootstrap(reference, mismatched_size, draws=4)
    zero_reference = [replace(reference[0], standardized_mse=0.0), reference[1]]
    with pytest.raises(MetricInputError, match="positive reference MSE in every trajectory"):
        paired_hierarchical_bootstrap(zero_reference, candidate, draws=4)


def test_hierarchical_bootstrap_is_reproducible_and_draw_count_is_configurable():
    reference = []
    candidate = []
    for replica in (101, 102, 103):
        for dimension in (12, 24):
            for trajectory in (2, 5, 8):
                offset = 0.01 * replica + 0.001 * dimension + 0.0001 * trajectory
                reference.append(
                    _record(replica, dimension, trajectory, rmse=offset, nll=2.0 * offset)
                )
                candidate.append(
                    _record(
                        replica,
                        dimension,
                        trajectory,
                        rmse=0.9 * offset + 0.01 * trajectory,
                        nll=1.7 * offset - 0.01 * dimension,
                    )
                )

    first = paired_hierarchical_bootstrap(reference, candidate, draws=257, seed=73)
    repeated = paired_hierarchical_bootstrap(reference, candidate, draws=257, seed=73)
    different_seed = paired_hierarchical_bootstrap(reference, candidate, draws=257, seed=74)
    assert first == repeated
    assert first.draws == 257
    assert first.seed == 73
    assert first != different_seed


def test_hierarchy_gives_equal_weight_to_dimensions_then_replicas():
    reference = [
        _record(101, 12, 1, rmse=1.0, nll=1.0, n_points=1_000),
        _record(101, 24, 1, rmse=10.0, nll=5.0, n_points=10),
        _record(101, 24, 2, rmse=10.0, nll=5.0, n_points=10),
        _record(101, 24, 3, rmse=10.0, nll=5.0, n_points=10),
    ]
    candidate = [
        _record(101, 12, 1, rmse=2.0, nll=0.0, n_points=1_000),
        _record(101, 24, 1, rmse=10.0, nll=5.0, n_points=10),
        _record(101, 24, 2, rmse=10.0, nll=5.0, n_points=10),
        _record(101, 24, 3, rmse=10.0, nll=5.0, n_points=10),
    ]
    result = paired_hierarchical_bootstrap(reference, candidate, draws=64, seed=9)

    # Dimension means are averaged equally: reference=(1+10)/2 and
    # candidate=(2+10)/2.  Neither row count nor trajectory count reweights it.
    assert result.rmse_absolute_change.estimate == pytest.approx(0.5)
    assert result.rmse_relative_change.estimate == pytest.approx(6.0 / 5.5 - 1.0)
    assert result.nll_absolute_change.estimate == pytest.approx(-0.5)


def test_rmse_transformation_occurs_after_trajectory_mse_within_each_dimension():
    reference = [
        _record(101, 12, 1, rmse=1.0, nll=1.0),
        _record(101, 12, 2, rmse=3.0, nll=3.0),
        _record(101, 24, 1, rmse=4.0, nll=4.0),
        _record(101, 24, 2, rmse=4.0, nll=4.0),
    ]
    candidate = [
        _record(101, 12, 1, rmse=2.0, nll=0.5),
        _record(101, 12, 2, rmse=2.0, nll=0.5),
        _record(101, 24, 1, rmse=5.0, nll=3.0),
        _record(101, 24, 2, rmse=5.0, nll=3.0),
    ]
    result = paired_hierarchical_bootstrap(reference, candidate, draws=32, seed=31)

    reference_rmse = (math.sqrt((1.0**2 + 3.0**2) / 2.0) + 4.0) / 2.0
    candidate_rmse = (2.0 + 5.0) / 2.0
    assert result.rmse_absolute_change.estimate == pytest.approx(candidate_rmse - reference_rmse)
    assert result.rmse_relative_change.estimate == pytest.approx(
        candidate_rmse / reference_rmse - 1.0
    )
    assert result.nll_absolute_change.estimate == pytest.approx(-1.25)


def test_bootstrap_rejects_invalid_keys_sizes_and_nonfinite_aggregation():
    valid_reference = [_record(101, 12, 1, rmse=1.0, nll=0.0)]
    valid_candidate = [_record(101, 12, 1, rmse=0.9, nll=-0.1)]

    invalid_key = TrajectoryKey(101, 0, 1)
    with pytest.raises(MetricInputError, match="outside the valid range"):
        paired_hierarchical_bootstrap(
            [replace(valid_reference[0], key=invalid_key)],
            [replace(valid_candidate[0], key=invalid_key)],
            draws=2,
        )
    with pytest.raises(MetricInputError, match="positive integers"):
        paired_hierarchical_bootstrap(
            [replace(valid_reference[0], n_points=1.5)],
            [replace(valid_candidate[0], n_points=1.5)],
            draws=2,
        )

    enormous_reference = [
        replace(valid_reference[0], standardized_mse=1.0e308),
        replace(_record(101, 12, 2, rmse=1.0, nll=0.0), standardized_mse=1.0e308),
    ]
    enormous_candidate = [
        replace(valid_candidate[0], standardized_mse=1.0e308),
        replace(_record(101, 12, 2, rmse=0.9, nll=-0.1), standardized_mse=1.0e308),
    ]
    with pytest.raises(MetricInputError, match="trajectory aggregation is nonfinite"):
        paired_hierarchical_bootstrap(enormous_reference, enormous_candidate, draws=2)


@pytest.mark.parametrize(("draws", "seed"), [(True, 7), (2, False)])
def test_bootstrap_rejects_boolean_draw_and_seed_values(draws, seed):
    reference = [_record(101, 12, 1, rmse=1.0, nll=0.0)]
    candidate = [_record(101, 12, 1, rmse=0.9, nll=-0.1)]
    with pytest.raises(MetricInputError):
        paired_hierarchical_bootstrap(reference, candidate, draws=draws, seed=seed)
