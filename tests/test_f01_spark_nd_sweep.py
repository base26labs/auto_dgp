"""Protocol tests for the compact SPARK-versus-TERA (n, d) matrix."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.f01_spark_nd_sweep as benchmark


def test_task_matrix_is_three_by_three_by_three_by_two() -> None:
    configurations = {(cell["n_particles"], cell["spatial_dims"]) for cell in benchmark.CELL_SPECS}
    assignments = [benchmark.task_assignment(task_id) for task_id in benchmark.TASK_IDS]

    assert configurations == {
        (n_particles, n_dims)
        for n_particles in benchmark.PARTICLE_COUNTS
        for n_dims in benchmark.SPATIAL_DIMS
    }
    assert len(benchmark.CELL_SPECS) == 27
    assert len(assignments) == 54
    assert benchmark.ARMS == ("spark", "tera")
    assert len({(cell["cell_id"], arm) for cell, arm in assignments}) == 54
    assert benchmark.TRAIN_ROWS == 1500
    assert benchmark.TEST_ROWS == 500
    assert benchmark.TERA_CONFIG == {
        "neighbors": 20,
        "kernel": "rbf",
        "epochs": 20,
        "precision": "float32",
    }
    for invalid in (-1, 54, True):
        with pytest.raises(ValueError, match="benchmark task ID"):
            benchmark.task_assignment(invalid)


def _fake_result(task_id: int, spark_nll: float = -1.0) -> dict:
    cell, arm = benchmark.task_assignment(task_id)
    spark = arm == "spark"
    return {
        "protocol": benchmark.PROTOCOL,
        "complete": True,
        "task_id": task_id,
        **cell,
        "arm": arm,
        "dataset_sha256": f"{task_id // 2:064x}",
        "training_rows": benchmark.TRAIN_ROWS,
        "test_rows": benchmark.TEST_ROWS,
        "fixed_before_data": True,
        "metrics": {
            "value_rmse": 0.5 if spark else 1.0,
            "gradient_rmse": 0.5 if spark else 1.0,
            "raw_nll": spark_nll if spark else 0.0,
        },
        "details": {},
        "resource_record": {},
        "runtime": {},
        "source_snapshot": {"git_commit": "a" * 40},
    }


def _write_fake_results(root: Path, spark_nll: float = -1.0) -> None:
    for task_id in benchmark.TASK_IDS:
        cell, arm = benchmark.task_assignment(task_id)
        path = root / benchmark.result_relative_path(cell, arm)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(benchmark.canonical_json_bytes(_fake_result(task_id, spark_nll)))


def test_summary_requires_all_metrics_in_every_configuration(tmp_path) -> None:
    _write_fake_results(tmp_path)
    summary = benchmark.summarize(tmp_path)

    assert summary["complete"] is True
    assert summary["task_count"] == 54
    assert summary["all_configurations_pass"] is True
    assert len(summary["configurations"]) == 9
    for result in summary["configurations"].values():
        assert result["seed_count"] == 3
        assert result["spark_lower_all_three_metrics"] is True
        assert result["spark_vs_tera"] == {
            "value_rmse_ratio": 0.5,
            "gradient_rmse_ratio": 0.5,
            "raw_nll_difference": -1.0,
        }

    failing = tmp_path / "failing"
    _write_fake_results(failing, spark_nll=0.1)
    assert benchmark.summarize(failing)["all_configurations_pass"] is False


def test_summary_rejects_unpaired_dataset_bytes(tmp_path) -> None:
    _write_fake_results(tmp_path)
    cell, arm = benchmark.task_assignment(1)
    path = tmp_path / benchmark.result_relative_path(cell, arm)
    result = json.loads(path.read_bytes())
    result["dataset_sha256"] = "f" * 64
    path.write_bytes(benchmark.canonical_json_bytes(result))

    with pytest.raises(ValueError, match="different dataset bytes"):
        benchmark.summarize(tmp_path)
