from __future__ import annotations

import hashlib
import json
import math
import shlex
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from cluster import aggregate_f02_internal_optimizer as aggregator
from cluster.f02_internal_grid import OPTIMIZER_SELECTION_TASKS, TASK_COUNT, task_for_index
from experiments.f02_design import TRAIN_TIME_INDICES

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "cluster" / "aggregate_f02_internal_optimizer.py"
_COMMIT = "1" * 40
_TREE = "2" * 40
_TERA = "3" * 40
_CATALOG_COMMIT = "4" * 40
_CATALOG_TREE = "5" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def _synthetic_catalog_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic fixtures cannot invert SHA-256 to reproduce the frozen catalog."""

    real = aggregator._sha256_file

    def controlled(path: Path) -> str:
        if Path(path).name == "catalog.json":
            return aggregator.EXPECTED_CATALOG_SHA256
        return real(Path(path))

    monkeypatch.setattr(aggregator, "_sha256_file", controlled)


def _metric_record(replica: int, dimension: int, trajectory: int, mse: float, nll: float) -> dict:
    return {
        "replica": replica,
        "dimension": dimension,
        "trajectory_id": trajectory,
        "n_points": 1,
        "standardized_mse": mse,
        "standardized_rmse": math.sqrt(mse),
        "gaussian_nll": nll,
        "interval_coverage": [
            {"level": 0.5, "coverage": 1.0},
            {"level": 0.9, "coverage": 1.0},
            {"level": 0.95, "coverage": 1.0},
        ],
    }


def _checks(*, tera: bool) -> dict:
    return {
        "row_count": 20,
        "mean_all_finite": True,
        "latent": {
            "all_finite": True,
            "all_positive": True,
            "minimum_raw": 1.0,
            "maximum_raw": 1.0,
        },
        "observation": {
            "all_finite": True,
            "all_positive": True,
            "minimum_raw": 1.1,
            "maximum_raw": 1.1,
        },
        "value_noise_variance": 0.1,
        "released_variance_epsilon_floor": (
            {
                "value": 1.1920928955078125e-07,
                "inactive": True,
                "failure_policy": "equality-to-floor-fails-before-scoring",
            }
            if tera
            else None
        ),
        "observation_is_latent_plus_noise": True,
        "all_valid": True,
    }


def _moments() -> dict:
    return {
        "mean": [0.0] * 20,
        "latent_variance": [1.0] * 20,
        "observation_variance": [1.1] * 20,
    }


def _metrics(task, mses: list[float], nll: float) -> dict:
    dimension = 6 * task.n_particles
    latent = [
        _metric_record(task.replica, dimension, trajectory, mse, nll)
        for trajectory, mse in zip(range(60, 80), mses, strict=True)
    ]
    observation = [
        _metric_record(task.replica, dimension, trajectory, mse, nll + 0.1)
        for trajectory, mse in zip(range(60, 80), mses, strict=True)
    ]
    return {"latent": latent, "observation": observation}


def _orbit_resources(rank: int) -> tuple[dict, dict]:
    iterations = [3] * 20
    matvecs = [4] * 20
    preconditioners = [3] * 20
    core = 3 * 50**2 + 2 * 50 * rank + rank**2 + rank
    operator = 4 * aggregator._orbit_matmul_flops(50, rank)
    preconditioner = 3 * aggregator._preconditioner_flops(50, rank)
    per_target = [
        {
            "rank": rank,
            "iterations": 3,
            "operator_matvecs": 4,
            "preconditioner_applications": 3,
            "reduced_system_dimension": 50 * rank,
            "operator_core_elements": core,
            "structured_operator_flops": operator,
            "preconditioner_flops": preconditioner,
            "counted_flops": operator + preconditioner,
        }
        for _ in range(20)
    ]
    solver = {
        "fresh_relative_residuals": [1e-6] * 20,
        "maximum_fresh_relative_residual": 1e-6,
        "converged": [True] * 20,
        "all_converged": True,
        "iterations": iterations,
        "operator_matvecs": matvecs,
        "preconditioner_applications": preconditioners,
        "ranks": [rank] * 20,
        "basis_exact": [True] * 20,
        "exact_arithmetic_certified": [True] * 20,
        "floating_point_rigorous": [False] * 20,
        "variance_error_upper_bounds": [1e-5] * 20,
        "expected_kl_upper_bounds": [1e-5] * 20,
        "finite_precision_variance_corrections": [0.0] * 20,
    }
    resources = {
        "counting_schema": "orbit_structured_proxy_v1",
        "requested_m": 50,
        "effective_m": 50,
        "preconditioner_counted": True,
        "per_target": per_target,
        "operator_core_elements_max": core,
        "counted_flops_total": 20 * (operator + preconditioner),
        "counted_flops_mean_per_target": operator + preconditioner,
    }
    return solver, resources


def _arms(task, mses: list[float], nll: float) -> dict:
    metrics = _metrics(task, mses, nll)
    common = {
        "requested_m": 50,
        "effective_m": 50,
        "hyperparameters_source": "TERA-gradient-fit",
        "metrics": metrics,
        "prediction_moments": _moments(),
        "prediction_seconds_descriptive": 0.25,
        "prediction_peak_gpu_allocated_bytes": 1024,
    }
    tera = {
        **common,
        "label": "TERA-50",
        "family": "TERA",
        "raw_prediction_checks": _checks(tera=True),
        "analytic_resources": {
            "counting_schema": "tera_dense_local_v1",
            "requested_m": 50,
            "effective_m": 50,
            "explicit_reduced_covariance_elements_per_target": 50**4,
            "reduced_cholesky_leading_flops_per_target": (50**6) / 3.0,
        },
    }
    rank = min(6 * task.n_particles, 50)
    solver, resources = _orbit_resources(rank)
    orbit = {
        **common,
        "label": "ORBIT-50",
        "family": "ORBIT",
        "raw_prediction_checks": _checks(tera=False),
        "solver": solver,
        "analytic_resources": resources,
        "same_m_agreement_to_TERA_50": {
            "maxabs_mean": 0.0,
            "maxabs_latent_variance": 0.0,
            "absolute_tolerance": 1e-4,
            "passes": True,
        },
    }
    value = {
        **common,
        "label": "value-only-conditional-50",
        "family": "value-only-conditioning-ablation",
        "raw_prediction_checks": _checks(tera=False),
        "control_semantics": (
            "prediction-time conditioning ablation; not a standalone value-only hyperparameter fit"
        ),
    }
    return {
        "TERA-50": tera,
        "ORBIT-50": orbit,
        "value-only-conditional-50": value,
    }


def _write_source_tree(repo: Path) -> None:
    paths = sorted(aggregator._REQUIRED_SOURCE_PATHS)
    for relative in paths:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(f"synthetic {relative}\n")


def _result(
    task,
    *,
    repo: Path,
    data_dir: Path,
    catalog: Path,
    mses: list[float],
    nll: float,
    commit: str,
) -> dict:
    dataset = data_dir / f"{task.dataset_stem}.npz"
    metadata = data_dir / f"{task.dataset_stem}.metadata.json"
    manifest = data_dir / f"{task.dataset_stem}.sha256.json"
    dimension = 6 * task.n_particles
    train_sources = [
        trajectory * 100 + time for trajectory in range(60) for time in TRAIN_TIME_INDICES
    ]
    eval_sources = [trajectory * 100 + 50 for trajectory in range(60, 80)]
    task_config = aggregator._expected_task_config(task)
    runtime = {
        "python": "3.12 synthetic",
        "platform": "linux-synthetic",
        "torch": "2.4.1",
        "numpy": "2.0.0",
        "device": "cuda",
        "dtype": "float32",
    }
    return {
        "schema_version": "f02_internal_task_v1",
        "status": "complete",
        "task_config": task_config,
        "training": {
            "split": "train",
            "time_indices": list(TRAIN_TIME_INDICES),
            "rows": 1500,
            "training_m": 20,
            "train_steps": task.train_steps,
            "train_epochs": 0,
            "batch_size": 256,
            "effective_batch_size": 256,
            "optimizer_updates": task.train_steps,
            "vecchia_target_factors_processed": task.train_steps * 256,
            "fit_seconds_descriptive": 1.0,
            "fit_peak_gpu_allocated_bytes": 2048,
        },
        "evaluation": {
            "split": "validation",
            "design": "optimizer_selection",
            "time_indices": [50],
            "test_gate": {
                "required": False,
                "validated": False,
                "committed_at_head": False,
                "path": None,
                "payload_sha256": None,
                "schema_version": None,
            },
        },
        "corpus": {
            "replica": task.replica,
            "dimension": dimension,
            "train_rows": 1500,
            "evaluation_rows": 20,
            "train_source_indices": train_sources,
            "evaluation_source_indices": eval_sources,
            "evaluation_trajectory_ids": list(range(60, 80)),
        },
        "frozen_parameters": {
            "kernel": "rbf",
            "lengthscale": [1.0],
            "outputscale": 1.0,
            "sigma_f_variance": 0.1,
            "sigma_g_variance": 0.1,
            "gradient_noise_model": "iid",
        },
        "arms": _arms(task, mses, nll),
        "catalog": {
            "path": str(catalog.resolve()),
            "sha256": aggregator.EXPECTED_CATALOG_SHA256,
            "generation_git_commit": _CATALOG_COMMIT,
            "generation_git_tree": _CATALOG_TREE,
            "task_index": task.replica * 5 + (2, 4, 6, 8, 10).index(task.n_particles),
        },
        "provenance": {
            "git": {
                "commit": commit,
                "tree": _TREE,
                "describe": "synthetic-clean",
                "status_porcelain": [],
            },
            "data": {
                "dataset_path": str(dataset.resolve()),
                "metadata_path": str(metadata.resolve()),
                "manifest_path": str(manifest.resolve()),
                "file_sha256": {
                    dataset.name: _sha256(dataset),
                    metadata.name: _sha256(metadata),
                },
                "manifest_sha256": _sha256(manifest),
                "generator_config": aggregator._expected_generator_config(task),
            },
            "task_config": task_config,
            "dependencies": {
                "pyproject.toml": {"sha256": _sha256(repo / "pyproject.toml")},
                "uv.lock": {"sha256": _sha256(repo / "uv.lock")},
            },
            "submodules": {
                "status": [f" {_TERA} gp/tera/vendor (heads/main)"],
                "tera_gitlink": _TERA,
            },
            "runtime": runtime,
        },
    }


def _write_task(
    repo: Path,
    run_root: Path,
    index: int,
    *,
    mses: list[float] | None = None,
    nll: float = 0.5,
    commit: str = _COMMIT,
) -> Path:
    task = task_for_index(index)
    task_dir = run_root / aggregator._expected_directory_name(task)
    task_dir.mkdir(parents=True)
    data_dir = repo / "data-artifacts"
    data_dir.mkdir(exist_ok=True)
    catalog = data_dir / "catalog.json"
    catalog.write_text("synthetic catalog\n")
    for suffix in (".npz", ".metadata.json", ".sha256.json"):
        (data_dir / f"{task.dataset_stem}{suffix}").write_text(f"{task.dataset_stem}{suffix}\n")
    result = _result(
        task,
        repo=repo,
        data_dir=data_dir,
        catalog=catalog,
        mses=mses or [1.0] * 20,
        nll=nll,
        commit=commit,
    )
    (task_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    provenance = {
        "captured_at_utc": "2026-08-03T12:00:00Z",
        "repo_root": str(repo.resolve()),
        "git_commit": commit,
        "git_tree": _TREE,
        "git_describe": "synthetic-clean",
        "tera_gitlink": _TERA,
        "hostname": "l40s.synthetic",
        "python": str((repo / ".venv/bin/python").resolve()),
        "array_job_id": run_root.name.removeprefix("job-"),
        "array_task_id": str(index),
        "replica": str(task.replica),
        "n_particles": str(task.n_particles),
        "n_dims": str(task.n_dims),
        "train_steps": str(task.train_steps),
        "seed": str(task.seed),
        "dataset_stem": task.dataset_stem,
        "catalog_path": str(catalog.resolve()),
        "catalog_sha256": aggregator.EXPECTED_CATALOG_SHA256,
        "exclusive_verification_mode": "slurm_explicit",
        "slurm_job_id": f"{run_root.name.removeprefix('job-')}_{index}",
        "slurm_job_nodelist": "l40s-node",
        "slurm_cpus_per_task": "16",
        "cuda_visible_devices": "0",
    }
    (task_dir / "provenance.env").write_text(
        "".join(f"{name}={value}\n" for name, value in provenance.items())
    )
    (task_dir / "git-status.txt").write_text("")
    (task_dir / "git-submodules.txt").write_text(f" {_TERA} gp/tera/vendor (heads/main)\n")
    source_paths = sorted(aggregator._REQUIRED_SOURCE_PATHS)
    (task_dir / "source-files.sha256").write_text(
        "".join(f"{_sha256(repo / name)}  {name}\n" for name in source_paths)
    )
    (task_dir / "dependency-files.sha256").write_text(
        f"{_sha256(repo / 'pyproject.toml')}  pyproject.toml\n"
        f"{_sha256(repo / 'uv.lock')}  uv.lock\n"
    )
    packages = [
        {"name": "numpy", "version": "2.0.0"},
        {"name": "torch", "version": "2.4.1"},
    ]
    dependency_audit = {
        "schema_version": 1,
        "status": "pass",
        "python_executable": provenance["python"],
        "package_count": 2,
        "packages": packages,
        "issues": [],
    }
    (task_dir / "dependency-audit.json").write_text(
        json.dumps(dependency_audit, indent=2, sort_keys=True) + "\n"
    )
    (task_dir / "dependency-packages.txt").write_text("numpy==2.0.0\ntorch==2.4.1\n")
    runtime = {
        "cuda_visible_devices": "0",
        "numpy": "2.0.0",
        "platform": "linux-synthetic",
        "python_executable": provenance["python"],
        "python_version": "3.12 synthetic",
        "scipy": "1.14.0",
        "torch": "2.4.1",
        "torch_cuda_available": True,
        "torch_cuda_devices": [
            {
                "capability": [8, 9],
                "index": 0,
                "name": "NVIDIA L40S",
                "total_memory_bytes": 48_000_000_000,
            }
        ],
        "torch_cuda_runtime": "12.4",
    }
    (task_dir / "runtime.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    dataset_paths = [
        catalog,
        data_dir / f"{task.dataset_stem}.npz",
        data_dir / f"{task.dataset_stem}.metadata.json",
        data_dir / f"{task.dataset_stem}.sha256.json",
    ]

    def dataset_digest(path: Path) -> str:
        return aggregator.EXPECTED_CATALOG_SHA256 if path.name == "catalog.json" else _sha256(path)

    (task_dir / "dataset-files.sha256").write_text(
        "".join(f"{dataset_digest(path)}  {path}\n" for path in dataset_paths)
    )
    options: list[str] = [
        str(repo / ".venv/bin/python"),
        "experiments/f02_internal_task.py",
        str(data_dir / f"{task.dataset_stem}.npz"),
        "--catalog",
        str(catalog.resolve()),
        "--out",
        str(task_dir / "result.json"),
        "--evaluation-split",
        "validation",
        "--evaluation-design",
        "optimizer_selection",
        "--training-m",
        "20",
        "--train-steps",
        str(task.train_steps),
        "--train-epochs",
        "0",
        "--kernel",
        "rbf",
        "--outputscale",
        "1.0",
        "--sigma-f",
        "0.001",
        "--sigma-g",
        "0.001",
        "--lengthscale",
        "1.0",
        "--seed",
        str(task.seed),
        "--batch-size",
        "256",
        "--lr",
        "0.01",
        "--weight-decay",
        "0.0",
        "--candidate-m",
        "none",
        "--cg-tolerance",
        "1e-5",
        "--use-preconditioner",
        "--function-jitter",
        "1e-8",
        "--reduced-jitter",
        "1e-8",
        "--dtype",
        "float32",
        "--device",
        "cuda",
    ]
    (task_dir / "command.txt").write_text(f"command={shlex.join(options)}\n")
    (task_dir / "exit-code.txt").write_text("0\n")
    (task_dir / "finished-at.txt").write_text("2026-08-03T12:01:00Z\n")
    (task_dir / "slurm-job.txt").write_text(
        "JobId=7000_0 OverSubscribe=EXCLUSIVE AllocTRES=cpu=16,gres/gpu=1\n"
    )
    (task_dir / "slurm-node.txt").write_text("NodeName=l40s CfgTRES=cpu=16,gres/gpu=1\n")
    (task_dir / "slurm-jobs-on-node.txt").write_text(f"{run_root.name.removeprefix('job-')}\n")
    for name in (
        "gpu.csv",
        "gpu-topology.txt",
        "gpu-processes-before.csv",
        "gpu-processes-after.csv",
    ):
        (task_dir / name).write_text(f"synthetic {name}\n")
    (task_dir / "artifacts.sha256").write_text(
        "".join(
            f"{_sha256(task_dir / name)}  {task_dir / name}\n"
            for name in sorted(aggregator._HASHED_ARTIFACTS)
        )
    )
    return task_dir


@pytest.fixture
def synthetic_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    run_root = repo / "runs" / "job-7000"
    run_root.mkdir(parents=True)
    _write_source_tree(repo)
    return repo, run_root


def _refresh_artifact_hash(task_dir: Path, name: str) -> None:
    lines = (task_dir / "artifacts.sha256").read_text().splitlines()
    rewritten = []
    for line in lines:
        _, raw_path = line.split(maxsplit=1)
        path = Path(raw_path)
        digest = _sha256(path) if path.name == name else line.split(maxsplit=1)[0]
        rewritten.append(f"{digest}  {raw_path}")
    (task_dir / "artifacts.sha256").write_text("\n".join(rewritten) + "\n")


def test_cli_is_importable_outside_repository(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "caller-declared set of array indices" in result.stdout


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [("0", (0,)), ("0,2-4,134", (0, 2, 3, 4, 134)), ("0-134", tuple(range(135)))],
)
def test_task_index_parser_requires_explicit_ordered_population(
    declaration: str, expected: tuple[int, ...]
) -> None:
    assert aggregator.parse_task_indices(declaration) == expected


@pytest.mark.parametrize("declaration", ("", "0,0", "1,0", "3-1", "-1", "135", "0,,1"))
def test_task_index_parser_rejects_ambiguous_or_out_of_grid_input(declaration: str) -> None:
    with pytest.raises(ValueError):
        aggregator.parse_task_indices(declaration)


def test_valid_pilot_is_audited_but_cannot_select_an_update(synthetic_repo) -> None:
    repo, run_root = synthetic_repo
    mses = [0.0, 4.0] * 10
    _write_task(repo, run_root, 0, mses=mses, nll=-999.0)

    report = aggregator.aggregate(run_root, (0,))

    assert report["declared_subset_ready"] is True
    assert report["analysis_ready"] is False
    assert report["selected_update"] is None
    assert report["metrics"]["selection_status"].startswith("not_run")
    task_metric = report["metrics"]["per_task"][0]
    assert task_metric["standardized_scalar_rmse"] == math.sqrt(2.0)
    assert task_metric["trajectory_mse_aggregation_order"] == (
        "sqrt(mean(trajectory standardized_mse))"
    )
    assert report["metrics"]["uses_nll"] is False


def test_missing_failed_and_unexpected_duplicate_directories_stay_visible(synthetic_repo) -> None:
    repo, run_root = synthetic_repo
    _write_task(repo, run_root, 0)
    failed = run_root / aggregator._expected_directory_name(task_for_index(1))
    failed.mkdir()
    (failed / "exit-code.txt").write_text("17\n")
    duplicate = run_root / "task-0-replica-99-n-2-steps-20-seed-11"
    duplicate.mkdir()

    report = aggregator.aggregate(run_root, (0, 1, 2))

    assert report["analysis_ready"] is False
    assert report["selected_update"] is None
    assert report["task_accounting"]["status_counts"] == {
        "valid": 1,
        "missing": 1,
        "failed": 1,
        "invalid": 0,
        "unexpected": 1,
    }
    assert report["task_accounting"]["duplicate_task_indices"] == [0]
    assert "duplicate" in report["task_accounting"]["unexpected_task_directories"][0]["reason"]


@pytest.mark.parametrize(
    ("mutation", "error_fragment"),
    [
        (lambda result: result["evaluation"].update({"time_indices": [25]}), "time-50"),
        (lambda result: result["task_config"].update({"candidate_m": [75]}), "task_config"),
        (
            lambda result: result["arms"]["ORBIT-50"]["solver"].update(
                {"maximum_fresh_relative_residual": 2e-5}
            ),
            "maximum residual",
        ),
        (
            lambda result: result["arms"]["ORBIT-50"]["same_m_agreement_to_TERA_50"].update(
                {"passes": False}
            ),
            "same-m agreement",
        ),
        (
            lambda result: result["arms"]["TERA-50"]["raw_prediction_checks"]["latent"].update(
                {"all_positive": False}
            ),
            "variance gate",
        ),
    ],
)
def test_result_schema_and_scientific_gates_fail_closed(
    synthetic_repo,
    mutation: Callable[[dict], None],
    error_fragment: str,
) -> None:
    repo, run_root = synthetic_repo
    task_dir = _write_task(repo, run_root, 0)
    path = task_dir / "result.json"
    result = json.loads(path.read_text())
    mutation(result)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _refresh_artifact_hash(task_dir, "result.json")

    report = aggregator.aggregate(run_root, (0,))

    task_record = report["task_accounting"]["tasks"][0]
    assert task_record["status"] == "invalid"
    assert any(error_fragment in error for error in task_record["errors"])
    assert report["selected_update"] is None


def test_tampered_hashed_artifact_is_rejected(synthetic_repo) -> None:
    repo, run_root = synthetic_repo
    task_dir = _write_task(repo, run_root, 0)
    (task_dir / "runtime.json").write_text("{}\n")

    report = aggregator.aggregate(run_root, (0,))

    errors = report["task_accounting"]["tasks"][0]["errors"]
    assert any("artifacts.sha256 does not verify runtime.json" in error for error in errors)
    assert report["declared_subset_ready"] is False


def test_cross_task_commit_inconsistency_blocks_declared_subset(synthetic_repo) -> None:
    repo, run_root = synthetic_repo
    _write_task(repo, run_root, 0, commit=_COMMIT)
    _write_task(repo, run_root, 1, commit="a" * 40)

    report = aggregator.aggregate(run_root, (0, 1))

    assert report["task_accounting"]["status_counts"]["valid"] == 2
    assert report["provenance"]["same_commit"] is False
    assert report["provenance"]["verified"] is False
    assert report["declared_subset_ready"] is False
    assert report["selected_update"] is None


def test_full_grid_selection_preserves_all_three_aggregation_layers_and_exact_tie(
    synthetic_repo,
) -> None:
    repo, run_root = synthetic_repo
    seed_factor = {11: 0.5, 29: 1.0, 47: 1.5}
    for task in OPTIMIZER_SELECTION_TASKS:
        corpus_factor = 1.0 + 0.1 * task.replica + 0.01 * task.n_particles
        if task.train_steps == 20:
            task_rmse = 2.0 * corpus_factor * seed_factor[task.seed]
            nll = -10_000.0
        else:
            # 50 and 100 have an exact RMSE tie.  Fewer updates must win.
            task_rmse = corpus_factor * seed_factor[task.seed]
            nll = 10_000.0 if task.train_steps == 50 else -20_000.0
        mses = [task_rmse**2] * 20
        _write_task(repo, run_root, task.task_index, mses=mses, nll=nll)

    report = aggregator.aggregate(run_root, tuple(range(TASK_COUNT)))

    assert report["task_accounting"]["status_counts"] == {
        "valid": 135,
        "missing": 0,
        "failed": 0,
        "invalid": 0,
        "unexpected": 0,
    }
    assert report["analysis_ready"] is True
    assert report["selected_update"] == 50
    summaries = {
        row["train_steps"]: row["macro_mean_standardized_scalar_rmse"]
        for row in report["metrics"]["update_summaries"]
    }
    corpus_factors = [
        1.0 + 0.1 * replica + 0.01 * particles
        for replica in (0, 1, 2)
        for particles in (2, 4, 6, 8, 10)
    ]
    expected_macro = sum(corpus_factors) / 15
    assert summaries[20] == pytest.approx(2.0 * expected_macro)
    assert summaries[50] == pytest.approx(expected_macro)
    assert summaries[100] == summaries[50]
    first_50_corpus = next(
        row
        for row in report["metrics"]["per_corpus"]
        if row["train_steps"] == 50 and row["replica"] == 0 and row["n_particles"] == 2
    )
    assert first_50_corpus["seed_standardized_scalar_rmse"] == pytest.approx([0.51, 1.02, 1.53])
    assert first_50_corpus["mean_seed_standardized_scalar_rmse"] == pytest.approx(1.02)
    assert first_50_corpus["mean_seed_standardized_scalar_rmse"] != math.sqrt(
        sum(value**2 for value in first_50_corpus["seed_standardized_scalar_rmse"]) / 3
    )
    assert len(report["metrics"]["per_task"]) == 135
    assert len(report["metrics"]["per_corpus"]) == 45
    assert all(row["corpus_count"] == 15 for row in report["metrics"]["update_summaries"])
    assert report["metrics"]["uses_nll"] is False


def test_atomic_save_round_trips_strict_json(synthetic_repo) -> None:
    repo, run_root = synthetic_repo
    _write_task(repo, run_root, 0)
    report = aggregator.aggregate(run_root, (0,))
    output = run_root / "aggregate.json"

    returned = aggregator.save_report(report, output)

    assert returned == output
    assert json.loads(output.read_text()) == report
    assert output.read_bytes().endswith(b"\n")
    assert not list(output.parent.glob(".aggregate.json.tmp-*"))
