from __future__ import annotations

from dataclasses import replace

import pytest

from experiments.f02_resource_selection import (
    ResourceSelectionError,
    ValidationTargetResource,
    ValidationTrajectoryRecord,
    evaluate_dimension_candidates,
    select_dimension_neighbor_count,
    select_neighbor_schedule,
)


def _records(
    *,
    candidates: tuple[int, ...] = (75, 100),
    replicas: tuple[int, ...] = (0, 1),
    seeds: tuple[int, ...] = (11, 29),
    trajectories: tuple[int, ...] = (3, 7),
    points: int = 2,
):
    metrics: list[ValidationTrajectoryRecord] = []
    resources: list[ValidationTargetResource] = []
    for replica in replicas:
        for seed in seeds:
            for m in candidates:
                for trajectory in trajectories:
                    metrics.append(
                        ValidationTrajectoryRecord(
                            dimension=12,
                            replica=replica,
                            seed=seed,
                            candidate_m=m,
                            trajectory_id=trajectory,
                            n_points=points,
                            standardized_mse=float((m / 100) ** 2),
                            latent_gaussian_nll=float(m / 1000),
                        )
                    )
                    for time in range(points):
                        resources.append(
                            ValidationTargetResource(
                                dimension=12,
                                replica=replica,
                                seed=seed,
                                candidate_m=m,
                                trajectory_id=trajectory,
                                source_index=10_000 * replica + 100 * trajectory + time,
                                counted_flops=m * 10,
                                operator_core_elements=m * 5,
                                fresh_relative_residual=1e-7,
                                converged=True,
                                basis_exact=True,
                                raw_latent_variance=0.2,
                            )
                        )
    return metrics, resources


def _kwargs():
    return {
        "replicas": (0, 1),
        "seeds": (11, 29),
        "trajectories_per_corpus": 2,
        "points_per_trajectory": 2,
        "bootstrap_draws": 128,
        "bootstrap_seed": 5,
        "state_envelope": 10_000,
        "flop_envelope": 10_000.0,
    }


def test_selects_lowest_nll_inside_one_percent_rmse_set() -> None:
    metrics, resources = _records()
    # m=75 has best RMSE.  Put m=100 exactly inside the 1% admissibility
    # boundary and give it the better NLL.
    metrics = [
        replace(
            record,
            standardized_mse=(0.75 * 1.01) ** 2,
            latent_gaussian_nll=0.01,
        )
        if record.candidate_m == 100
        else record
        for record in metrics
    ]
    selected = select_dimension_neighbor_count(
        12,
        (75, 100),
        metrics,
        resources,
        **_kwargs(),
    )
    assert selected.best_eligible_rmse == pytest.approx(0.75)
    assert selected.rmse_admissibility_threshold == pytest.approx(0.7575)
    assert selected.selected_m == 100


def test_metric_order_is_trajectory_then_seed_then_replica() -> None:
    metrics, resources = _records(candidates=(75,))
    changed: list[ValidationTrajectoryRecord] = []
    for record in metrics:
        # Seed 11 has RMSE 1, seed 29 has RMSE 9.  The frozen average is 5;
        # pooling MSE before sqrt would incorrectly return sqrt(41).
        mse = 1.0 if record.seed == 11 else 81.0
        changed.append(replace(record, standardized_mse=mse))
    summary = evaluate_dimension_candidates(
        12,
        (75,),
        changed,
        resources,
        **_kwargs(),
    )[0]
    assert summary.macro_rmse == pytest.approx(5.0)
    assert summary.macro_rmse != pytest.approx(41.0**0.5)


def test_optimizer_seeds_are_averaged_before_resource_bootstrap() -> None:
    metrics, resources = _records(candidates=(75,))
    changed = [
        replace(record, counted_flops=0 if record.seed == 11 else 100) for record in resources
    ]
    summary = evaluate_dimension_candidates(
        12,
        (75,),
        metrics,
        changed,
        **_kwargs(),
    )[0]
    assert summary.mean_counted_flops_per_target == pytest.approx(50.0)
    assert summary.counted_flops_bootstrap_upper_95 == pytest.approx(50.0)
    assert summary.bootstrap_upper_quantile == 0.975


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    (
        ("fresh_relative_residual", 2e-5, "solver"),
        ("converged", False, "solver"),
        ("basis_exact", False, "basis"),
        ("raw_latent_variance", 0.0, "variance"),
        ("operator_core_elements", 10_001, "state_envelope"),
        ("counted_flops", 10_001, "flop_envelope"),
    ),
)
def test_each_resource_gate_fails_closed(field: str, value: object, reason: str) -> None:
    metrics, resources = _records(candidates=(75,))
    if field == "counted_flops":
        resources = [replace(record, counted_flops=value) for record in resources]
    else:
        resources[0] = replace(resources[0], **{field: value})
    summary = evaluate_dimension_candidates(
        12,
        (75,),
        metrics,
        resources,
        **_kwargs(),
    )[0]
    assert summary.eligible is False
    assert reason in summary.failure_reasons


def test_exact_nll_tie_prefers_smaller_m() -> None:
    metrics, resources = _records()
    metrics = [
        replace(record, standardized_mse=1.0, latent_gaussian_nll=0.25) for record in metrics
    ]
    selected = select_dimension_neighbor_count(
        12,
        (75, 100),
        metrics,
        resources,
        **_kwargs(),
    )
    assert selected.selected_m == 75


def test_incomplete_or_duplicate_grid_is_rejected() -> None:
    metrics, resources = _records(candidates=(75,))
    with pytest.raises(ResourceSelectionError, match="wrong number"):
        evaluate_dimension_candidates(
            12,
            (75,),
            metrics[:-1],
            resources,
            **_kwargs(),
        )
    with pytest.raises(ResourceSelectionError, match="duplicate"):
        evaluate_dimension_candidates(
            12,
            (75,),
            [*metrics, metrics[0]],
            resources,
            **_kwargs(),
        )


def test_no_eligible_candidate_raises_instead_of_relaxing_envelope() -> None:
    metrics, resources = _records(candidates=(75,))
    resources = [replace(record, converged=False) for record in resources]
    with pytest.raises(ResourceSelectionError, match="no resource-eligible"):
        select_dimension_neighbor_count(
            12,
            (75,),
            metrics,
            resources,
            **_kwargs(),
        )


def test_schedule_requires_the_exact_declared_dimension_grid() -> None:
    metrics, resources = _records(candidates=(75,))
    with pytest.raises(ResourceSelectionError, match="exact dimension grid"):
        select_neighbor_schedule(
            (75,),
            metrics,
            resources,
            dimensions=(12, 24),
            **_kwargs(),
        )

    duplicated_metrics = [
        *metrics,
        *(replace(record, dimension=24) for record in metrics),
    ]
    duplicated_resources = [
        *resources,
        *(replace(record, dimension=24) for record in resources),
    ]
    schedule = select_neighbor_schedule(
        (75,),
        duplicated_metrics,
        duplicated_resources,
        dimensions=(12, 24),
        **_kwargs(),
    )
    assert [(selection.dimension, selection.selected_m) for selection in schedule] == [
        (12, 75),
        (24, 75),
    ]
