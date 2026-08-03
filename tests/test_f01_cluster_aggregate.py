from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cluster.aggregate_f01_orbit import SCOPE, aggregate


def _canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _write_artifact_manifest(task_dir: Path) -> None:
    artifact_names = (
        "datasets.json",
        "runtime.json",
        "result.json",
        "source-files.sha256",
    )
    (task_dir / "artifacts.sha256").write_text(
        "".join(
            f"{hashlib.sha256((task_dir / name).read_bytes()).hexdigest()}  {task_dir / name}\n"
            for name in artifact_names
        )
    )


def _write_task(
    run_root: Path,
    task_index: int,
    seed: int,
    *,
    repeats: int = 1,
    commit: str = "a" * 40,
    tree: str = "b" * 40,
    submodules: str = " cccccccc gp/tera/vendor (heads/main)",
    config_override: dict | None = None,
    exclusive: bool = True,
    cluster_wall_inferential: bool = False,
    prereg_wall_inferential: bool = False,
    dirty: bool = False,
    dataset_key: str | None = None,
) -> Path:
    task_dir = run_root / f"seed-{seed}-task-{task_index}"
    task_dir.mkdir(parents=True)

    config = {
        "n_train": 8,
        "n_eval": 3,
        "d": 2,
        "m_values": [2, 4],
        "tera_max_m": 2,
        "repeats": repeats,
        "seed": seed,
        "kernel": "matern52",
        "sampling": "dense",
        "device": "cuda",
        "dtype": "float64",
        "cg_tolerance": 1e-8,
        "out": str(task_dir / "result.json"),
    }
    if config_override:
        config.update(config_override)

    dataset_records = []
    rows = []
    for repeat in range(repeats):
        token = dataset_key if dataset_key is not None else f"{seed}:{repeat}"
        dataset_payload = {
            "d": 2,
            "repeat": repeat,
            "kernel_name": "matern52",
            "outputscale": 1.0,
            "sigma_f": 0.001,
            "sigma_g": 0.0,
            "sampling_backend": "dense",
            "tensors": [
                {
                    "name": "X_train",
                    "shape": [8, 2],
                    "dtype": "torch.float64",
                    "nbytes": 128,
                    "sha256": hashlib.sha256(token.encode()).hexdigest(),
                }
            ],
        }
        dataset_records.append(
            {**dataset_payload, "combined_sha256": _canonical_hash(dataset_payload)}
        )

        baseline_kl = 0.60 + 0.01 * task_index + 0.02 * repeat
        candidate_kl = 0.25 + 0.01 * task_index + 0.01 * repeat
        rows.extend(
            [
                {"repeat": repeat, "method": "TERA", "m": 2},
                {
                    "repeat": repeat,
                    "method": "ORBIT-exact",
                    "m": 2,
                    "avg_marginal_kl": baseline_kl,
                    "maxabs_mean_to_same_m_tera": 2e-8,
                    "maxabs_variance_to_same_m_tera": 3e-8,
                    "cg_converged_fraction": 1.0,
                    "cg_relative_residual_max": 2e-10,
                    "variance_valid_fraction": 1.0,
                    "variance_min_raw": 0.01,
                    "seconds_descriptive": 1000.0 + task_index,
                },
                {
                    "repeat": repeat,
                    "method": "ORBIT-exact",
                    "m": 4,
                    "avg_marginal_kl": candidate_kl,
                    "cg_converged_fraction": 1.0,
                    "cg_relative_residual_max": 3e-10,
                    "variance_valid_fraction": 1.0,
                    "variance_min_raw": 0.005,
                    "seconds_descriptive": 2000.0 + task_index,
                },
            ]
        )

    datasets = {
        "schema_version": 1,
        "algorithm": "sha256",
        "base_seed": seed,
        "datasets": dataset_records,
    }
    packages = [{"name": "torch", "version": "2.4.1"}]
    runtime = {
        "packages": packages,
        "packages_sha256": _canonical_hash(packages),
        "selected_environment": {"F01_SLURM_EXCLUSIVE_VERIFIED": "1"},
    }
    _write_json(task_dir / "datasets.json", datasets)
    _write_json(task_dir / "runtime.json", runtime)

    result = {
        "preregistration": {
            "same_m_equivalence_tolerance": 1e-6,
            "cg_tolerance": 1e-8,
            "wall_time_is_inferential": prereg_wall_inferential,
        },
        "config": config,
        "rows": rows,
        "cluster_provenance": {
            "dataset_manifest_sha256": hashlib.sha256(
                (task_dir / "datasets.json").read_bytes()
            ).hexdigest(),
            "runtime_manifest_sha256": hashlib.sha256(
                (task_dir / "runtime.json").read_bytes()
            ).hexdigest(),
            "packages_sha256": runtime["packages_sha256"],
            "slurm_job_id": "12345",
            "slurm_array_job_id": "12345",
            "slurm_array_task_id": str(task_index),
            "exclusive_node_verified": exclusive,
            "wall_time_is_inferential": cluster_wall_inferential,
        },
    }
    _write_json(task_dir / "result.json", result)
    (task_dir / "source-files.sha256").write_text(f"{'f' * 64}  cluster/run_f01_orbit.py\n")
    _write_artifact_manifest(task_dir)
    (task_dir / "exit-code.txt").write_text("0\n")
    (task_dir / "git-status.txt").write_text(" M bad.py\n" if dirty else "\n")
    (task_dir / "git-submodules.txt").write_text(submodules + "\n")
    (task_dir / "provenance.env").write_text(
        "\n".join(
            [
                f"git_commit={commit}",
                f"git_tree={tree}",
                "git_describe=f01-test",
                f"seed={seed}",
                "array_job_id=12345",
                f"array_task_id={task_index}",
            ]
        )
        + "\n"
    )
    return task_dir


def test_aggregates_independent_seed_repeat_pairs_with_accuracy_uncertainty(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "job-12345"
    seeds = [101, 102, 103]
    for task_index, seed in enumerate(seeds):
        _write_task(run_root, task_index, seed, repeats=2)

    report = aggregate(run_root, seeds, bootstrap_samples=500, bootstrap_seed=7)

    assert report["scope"] == SCOPE
    assert report["task_accounting"]["all_expected_tasks_valid"] is True
    assert report["task_accounting"]["status_counts"]["valid"] == 3
    assert report["provenance"]["verified"] is True
    assert report["provenance"]["same_config_except_seed_and_output"] is True
    assert report["pool"]["analysis_ready"] is True
    assert report["pool"]["independent_dataset_instance_count"] == 6
    assert {
        (instance["base_seed"], instance["repeat"]) for instance in report["pool"]["instances"]
    } == {(seed, repeat) for seed in seeds for repeat in range(2)}

    assert report["gates"]["h1_same_m_equivalence"]["status"] == "pass"
    assert report["gates"]["solver"]["status"] == "pass"
    assert report["gates"]["variance"]["status"] == "pass"
    h2 = report["gates"]["h2_larger_m_headroom"]
    assert h2["status"] == "pass"
    assert h2["candidate_is_nontrivial"] is True
    assert h2["paired_instance_count"] == 6
    assert h2["mean_paired_improvement"] > 0.0
    assert h2["uncertainty"]["method"] == "paired nonparametric bootstrap percentile"
    assert h2["uncertainty"]["mean_paired_improvement_ci"][0] > 0.0
    assert report["gates"]["overall_mechanism_pass"] is True
    assert report["timing"] == {
        "wall_time_is_inferential": False,
        "included_in_hypothesis_tests": False,
        "uncertainty_reported_for_wall_time": False,
        "note": (
            "seconds_descriptive and other wall-time fields are intentionally excluded; "
            "the bootstrap applies only to paired marginal-KL accuracy differences"
        ),
    }


def test_retains_failed_and_never_started_tasks(tmp_path: Path) -> None:
    run_root = tmp_path / "job-12345"
    seeds = [201, 202, 203]
    _write_task(run_root, 0, seeds[0], repeats=2)
    failed = run_root / f"seed-{seeds[1]}-task-1"
    failed.mkdir(parents=True)
    (failed / "exit-code.txt").write_text("17\n")

    report = aggregate(run_root, seeds, bootstrap_samples=100)

    statuses = {
        task["task_index"]: task["status"]
        for task in report["task_accounting"]["tasks"]
        if task["task_index"] is not None
    }
    assert statuses == {0: "valid", 1: "failed", 2: "missing"}
    assert report["task_accounting"]["status_counts"]["failed"] == 1
    assert report["task_accounting"]["status_counts"]["missing"] == 1
    assert report["provenance"]["verified"] is False
    assert report["pool"]["independent_dataset_instance_count"] == 2
    assert report["gates"]["h1_same_m_equivalence"]["observed_pass"] is True
    assert report["gates"]["h1_same_m_equivalence"]["confirmatory_pass"] is False
    assert report["gates"]["h2_larger_m_headroom"]["status"] == "insufficient"
    assert report["gates"]["overall_mechanism_pass"] is False


def test_recomputes_equivalence_solver_and_variance_failures(tmp_path: Path) -> None:
    run_root = tmp_path / "job-12345"
    seeds = [251, 252, 253]
    for task_index, seed in enumerate(seeds):
        task_dir = _write_task(run_root, task_index, seed)
        if task_index == 1:
            result_path = task_dir / "result.json"
            result = json.loads(result_path.read_text())
            bad_row = next(
                row for row in result["rows"] if row["method"] == "ORBIT-exact" and row["m"] == 2
            )
            bad_row["maxabs_mean_to_same_m_tera"] = 2e-6
            bad_row["cg_converged_fraction"] = 0.5
            bad_row["cg_relative_residual_max"] = 2e-6
            bad_row["variance_valid_fraction"] = 0.5
            bad_row["variance_min_raw"] = -0.01
            _write_json(result_path, result)
            _write_artifact_manifest(task_dir)

    report = aggregate(run_root, seeds, bootstrap_samples=100)

    assert report["provenance"]["verified"] is True
    assert report["pool"]["analysis_ready"] is True
    assert report["gates"]["h1_same_m_equivalence"]["status"] == "fail"
    assert report["gates"]["solver"]["status"] == "fail"
    assert report["gates"]["variance"]["status"] == "fail"
    assert report["gates"]["h2_larger_m_headroom"]["status"] == "pass"
    assert report["gates"]["overall_mechanism_pass"] is False


def test_h2_rejects_full_training_set_candidate_as_trivial(tmp_path: Path) -> None:
    run_root = tmp_path / "job-12345"
    seeds = [271, 272, 273]
    for task_index, seed in enumerate(seeds):
        _write_task(
            run_root,
            task_index,
            seed,
            config_override={"n_train": 4},
        )

    report = aggregate(run_root, seeds, bootstrap_samples=100)

    h2 = report["gates"]["h2_larger_m_headroom"]
    assert report["provenance"]["verified"] is True
    assert h2["reference_m"] == 2
    assert h2["candidate_m"] == 4
    assert h2["candidate_is_nontrivial"] is False
    assert h2["status"] == "insufficient"
    assert h2["confirmatory_pass"] is False


@pytest.mark.parametrize(
    ("defect", "overrides"),
    [
        ("commit", {"commit": "d" * 40}),
        ("tree", {"tree": "e" * 40}),
        ("submodule", {"submodules": " ffffffff gp/tera/vendor (heads/other)"}),
        ("config", {"config_override": {"kernel": "rbf"}}),
        ("exclusive", {"exclusive": False}),
        ("cluster timing", {"cluster_wall_inferential": True}),
        ("experiment timing", {"prereg_wall_inferential": True}),
        ("dirty", {"dirty": True}),
    ],
)
def test_provenance_defects_prevent_confirmatory_pooling(
    tmp_path: Path,
    defect: str,
    overrides: dict,
) -> None:
    run_root = tmp_path / "job-12345"
    seeds = [301, 302, 303]
    for task_index, seed in enumerate(seeds):
        _write_task(
            run_root,
            task_index,
            seed,
            **(overrides if task_index == 2 else {}),
        )

    report = aggregate(run_root, seeds, bootstrap_samples=100)

    assert report["provenance"]["verified"] is False, defect
    assert report["pool"]["analysis_ready"] is False, defect
    assert report["gates"]["overall_mechanism_pass"] is False, defect
    assert (
        report["task_accounting"]["status_counts"]["invalid"] > 0
        or report["provenance"]["consistent_across_valid_tasks"] is False
    ), defect


def test_duplicate_dataset_hashes_are_not_counted_as_independent(tmp_path: Path) -> None:
    run_root = tmp_path / "job-12345"
    seeds = [401, 402, 403]
    for task_index, seed in enumerate(seeds):
        _write_task(
            run_root,
            task_index,
            seed,
            repeats=2,
            dataset_key="same-simulated-data",
        )

    report = aggregate(run_root, seeds, bootstrap_samples=100)

    assert report["provenance"]["verified"] is True
    assert report["pool"]["candidate_dataset_instance_count"] == 6
    assert report["pool"]["independent_dataset_instance_count"] == 0
    assert len(report["pool"]["duplicate_dataset_content_sha256"]) == 1
    assert report["pool"]["independence_pass"] is False
    assert report["gates"]["overall_mechanism_pass"] is False
