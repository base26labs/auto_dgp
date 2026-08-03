from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from cluster.aggregate_f02_nbody_data import EXPECTED_SOURCE_PATHS, aggregate
from cluster.generate_f02_nbody import GenerationTask, generation_tasks, run_generation_task
from data.generate_nbody_confirmatory import ConfirmatoryConfig, write_bundle
from data.load_nbody_confirmatory import load_confirmatory_bundle

_GENERATION = {
    "n_trajectories": 5,
    "steps_per_trajectory": 2,
    "dt": 0.01,
    "mass_seed": 1729,
    "trajectory_seed": 2718,
    "split_seed": 31415,
    "validation_seed": 1618,
}
_COMMIT = "1" * 40
_TREE = "2" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_manifest() -> str:
    return "".join(
        f"{hashlib.sha256(source.encode()).hexdigest()}  {source}\n"
        for source in sorted(EXPECTED_SOURCE_PATHS)
    )


def _write_task_artifacts(
    run_root: Path,
    task: GenerationTask,
    result: dict[str, Any],
    *,
    replicas: list[int],
    particle_counts: list[int],
    commit: str = _COMMIT,
) -> Path:
    for source in EXPECTED_SOURCE_PATHS:
        source_path = run_root.parent / source
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(source)
    task_dir = run_root / f"task-{task.task_index}"
    task_dir.mkdir(parents=True)
    (task_dir / "exit-code.txt").write_text("0\n")
    (task_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (task_dir / "provenance.env").write_text(
        "\n".join(
            (
                f"repo_root={run_root.parent}",
                f"git_commit={commit}",
                f"git_tree={_TREE}",
                "git_describe=f02-aggregate-test",
                "slurm_job_id=8000",
                "slurm_array_job_id=7000",
                f"slurm_array_task_id={task.task_index}",
                "python=/usr/bin/python",
                f"replicas={','.join(map(str, replicas))}",
                f"particle_counts={','.join(map(str, particle_counts))}",
            )
        )
        + "\n"
    )
    (task_dir / "git-submodules.txt").write_text(
        f" {'3' * 40} third_party/tera (heads/main)\n"
    )
    (task_dir / "source-files.sha256").write_text(_source_manifest())
    (task_dir / "artifacts.sha256").write_text(
        f"{_sha256(task_dir / 'result.json')}  {task_dir / 'result.json'}\n"
        f"{_sha256(task_dir / 'source-files.sha256')}  "
        f"{task_dir / 'source-files.sha256'}\n"
    )
    return task_dir


def _generate_result(task: GenerationTask, data_dir: Path) -> dict[str, Any]:
    return run_generation_task(
        task,
        output_dir=data_dir,
        verify_existing=False,
        **_GENERATION,
    )


def _aggregate(
    run_root: Path,
    replicas: list[int],
    particle_counts: list[int],
    *,
    development_replicas: list[int],
) -> dict[str, Any]:
    return aggregate(
        run_root,
        replicas,
        particle_counts,
        development_replicas=development_replicas,
        n_dims=1,
        **_GENERATION,
    )


def test_aggregate_emits_ready_replica_major_catalog(tmp_path: Path) -> None:
    replicas = [0, 101]
    particle_counts = [2, 3]
    run_root = tmp_path / "job-7000"
    data_dir = tmp_path / "data"
    tasks = generation_tasks(replicas, particle_counts, n_dims=1)
    for task in tasks:
        _write_task_artifacts(
            run_root,
            task,
            _generate_result(task, data_dir),
            replicas=replicas,
            particle_counts=particle_counts,
        )

    report = _aggregate(
        run_root,
        replicas,
        particle_counts,
        development_replicas=[0],
    )

    assert report["overall_ready"] is True
    assert report["task_accounting"]["status_counts"] == {
        "valid": 4,
        "missing": 0,
        "failed": 0,
        "invalid": 0,
        "unexpected": 0,
    }
    bundles = report["catalog"]["bundles"]
    assert [(item["replica"], item["n_particles"]) for item in bundles] == [
        (0, 2),
        (0, 3),
        (101, 2),
        (101, 3),
    ]
    assert [item["phase"] for item in bundles] == [
        "development",
        "development",
        "confirmatory",
        "confirmatory",
    ]
    assert [item["D"] for item in bundles] == [4, 6, 4, 6]
    assert all(item["eligible_for_catalog"] for item in bundles)
    assert all(item["unique_content"] for item in bundles)
    assert report["provenance"]["verified"] is True
    assert report["independence"]["verified"] is True


def test_aggregate_retains_valid_failed_and_missing_tasks(tmp_path: Path) -> None:
    replicas = [0, 1, 101]
    particle_counts = [2]
    run_root = tmp_path / "job-7000"
    data_dir = tmp_path / "data"
    tasks = generation_tasks(replicas, particle_counts, n_dims=1)
    _write_task_artifacts(
        run_root,
        tasks[0],
        _generate_result(tasks[0], data_dir),
        replicas=replicas,
        particle_counts=particle_counts,
    )
    failed_dir = run_root / "task-1"
    failed_dir.mkdir(parents=True)
    (failed_dir / "exit-code.txt").write_text("17\n")

    report = _aggregate(
        run_root,
        replicas,
        particle_counts,
        development_replicas=[0, 1],
    )

    assert report["overall_ready"] is False
    assert [task["status"] for task in report["task_accounting"]["tasks"]] == [
        "valid",
        "failed",
        "missing",
    ]
    assert report["catalog"]["bundle_count"] == 1
    assert report["catalog"]["bundles"][0]["replica"] == 0
    assert report["task_accounting"]["tasks"][1]["exit_code"] == 17


def test_aggregate_rejects_mismatched_recorded_bundle_hashes(tmp_path: Path) -> None:
    replicas = [101]
    particle_counts = [2]
    run_root = tmp_path / "job-7000"
    data_dir = tmp_path / "data"
    task = generation_tasks(replicas, particle_counts, n_dims=1)[0]
    result = _generate_result(task, data_dir)
    result["artifacts"]["file_sha256"][next(iter(result["artifacts"]["file_sha256"]))] = (
        "0" * 64
    )
    _write_task_artifacts(
        run_root,
        task,
        result,
        replicas=replicas,
        particle_counts=particle_counts,
    )

    report = _aggregate(
        run_root,
        replicas,
        particle_counts,
        development_replicas=[],
    )

    record = report["task_accounting"]["tasks"][0]
    assert report["overall_ready"] is False
    assert record["status"] == "invalid"
    assert any("recorded bundle file hashes" in error for error in record["errors"])
    assert report["catalog"]["bundle_count"] == 0


def test_aggregate_rejects_source_manifest_that_no_longer_matches_checkout(
    tmp_path: Path,
) -> None:
    replicas = [0]
    particle_counts = [2]
    run_root = tmp_path / "job-7000"
    data_dir = tmp_path / "data"
    task = generation_tasks(replicas, particle_counts, n_dims=1)[0]
    _write_task_artifacts(
        run_root,
        task,
        _generate_result(task, data_dir),
        replicas=replicas,
        particle_counts=particle_counts,
    )
    changed = run_root.parent / "docs/F02_NBODY_PROTOCOL.md"
    changed.write_text("changed after generation")

    report = _aggregate(
        run_root,
        replicas,
        particle_counts,
        development_replicas=[0],
    )

    record = report["task_accounting"]["tasks"][0]
    assert record["status"] == "invalid"
    assert any("does not verify current file" in error for error in record["errors"])


def test_aggregate_requires_common_commit_provenance(tmp_path: Path) -> None:
    replicas = [0, 101]
    particle_counts = [2]
    run_root = tmp_path / "job-7000"
    data_dir = tmp_path / "data"
    tasks = generation_tasks(replicas, particle_counts, n_dims=1)
    for task, commit in zip(tasks, (_COMMIT, "4" * 40), strict=True):
        _write_task_artifacts(
            run_root,
            task,
            _generate_result(task, data_dir),
            replicas=replicas,
            particle_counts=particle_counts,
            commit=commit,
        )

    report = _aggregate(
        run_root,
        replicas,
        particle_counts,
        development_replicas=[0],
    )

    assert report["overall_ready"] is False
    assert report["provenance"]["same_commit"] is False
    assert report["provenance"]["verified"] is False
    assert all(
        task["status"] == "valid" and not task["eligible_for_catalog"]
        for task in report["task_accounting"]["tasks"]
    )
    assert all(
        "cross-task" in task["errors"][0]
        for task in report["task_accounting"]["tasks"]
    )


def test_aggregate_rejects_duplicate_semantic_dataset_content(tmp_path: Path) -> None:
    replicas = [0, 101]
    particle_counts = [2]
    run_root = tmp_path / "job-7000"
    data_dir = tmp_path / "data"
    tasks = generation_tasks(replicas, particle_counts, n_dims=1)

    first_result = _generate_result(tasks[0], data_dir)
    first_loaded = load_confirmatory_bundle(first_result["artifacts"]["dataset"])
    copied_config = ConfirmatoryConfig(
        n_particles=2,
        n_dims=1,
        replica=101,
        **_GENERATION,
    )
    copied_dataset = replace(first_loaded.dataset, config=copied_config)
    write_bundle(
        copied_dataset,
        data_dir,
        stem="nbody_fixedmass_n2_d1_replica101",
    )
    second_result = run_generation_task(
        tasks[1],
        output_dir=data_dir,
        verify_existing=True,
        **_GENERATION,
    )
    for task, result in zip(tasks, (first_result, second_result), strict=True):
        _write_task_artifacts(
            run_root,
            task,
            result,
            replicas=replicas,
            particle_counts=particle_counts,
        )

    report = _aggregate(
        run_root,
        replicas,
        particle_counts,
        development_replicas=[0],
    )

    assert report["overall_ready"] is False
    assert report["task_accounting"]["all_expected_tasks_valid"] is True
    assert report["task_accounting"]["all_expected_tasks_unique"] is False
    assert len(report["independence"]["duplicate_dataset_content_sha256"]) == 1
    assert report["independence"]["unique_bundle_count"] == 0
    assert all(
        bundle["unique_content"] is False
        and bundle["eligible_for_catalog"] is False
        for bundle in report["catalog"]["bundles"]
    )
