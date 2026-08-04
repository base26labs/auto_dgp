from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SBATCH_SCRIPT = REPO_ROOT / "cluster" / "f02b_fit_array.sbatch"
CATALOG_SHA256 = "2dee429bdaf50cc78cb40ba8b038f7e4731bb07127927c55607b528c1db66942"


@dataclass(frozen=True)
class FakeCluster:
    environment: dict[str, str]
    capture_root: Path
    dataset: Path
    catalog: Path
    output_root: Path

    def run(self, environment: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SBATCH_SCRIPT)],
            check=False,
            capture_output=True,
            cwd=self.capture_root,
            env=self.environment if environment is None else environment,
            text=True,
        )


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


@pytest.fixture
def fake_cluster(tmp_path: Path) -> FakeCluster:
    deployed_repo = tmp_path / "deployed-repo"
    fake_bin = tmp_path / "fake-bin"
    capture_root = tmp_path / "capture"
    data_root = tmp_path / "development-data"
    catalog = tmp_path / "catalog.json"
    output_root = tmp_path / "fit-output"
    python_bin = deployed_repo / ".venv" / "bin" / "python"
    grid_helper = deployed_repo / "cluster" / "f02b_calibration_grid.py"
    fit_runner = deployed_repo / "experiments" / "f02b_calibration_fit.py"
    tera_root = deployed_repo / "gp" / "tera" / "vendor"
    dataset = data_root / "nbody_fixedmass_n2_d3_replica1.npz"

    for directory in (
        fake_bin,
        capture_root,
        data_root,
        python_bin.parent,
        grid_helper.parent,
        fit_runner.parent,
        tera_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    grid_helper.write_text("# fake grid entrypoint\n")
    fit_runner.write_text("# fake fit entrypoint\n")
    dataset.write_bytes(b"frozen development bundle")
    catalog.write_bytes(b"frozen catalog")

    _write_executable(
        python_bin,
        r"""#!/usr/bin/env bash
set -Eeuo pipefail
if [[ ${1:-} == */cluster/f02b_calibration_grid.py ]]; then
    printf '%s\n' "${5}"
    printf '%s\n' 1 2 3 12 47 20 20 rbf nbody_fixedmass_n2_d3_replica1
    printf '%s\n' F02B_NUMERICAL_CALIBRATION_v2
elif [[ ${1:-} == -c ]]; then
    printf '%s\n' 32
elif [[ ${1:-} == -m ]]; then
    printf '%s\n' "$@" > "${FAKE_CAPTURE_ROOT}/runner-args.txt"
    env | sort > "${FAKE_CAPTURE_ROOT}/runner-env.txt"
else
    echo "unexpected fake Python invocation: $*" >&2
    exit 90
fi
""",
    )
    _write_executable(
        fake_bin / "git",
        r"""#!/usr/bin/env bash
set -Eeuo pipefail
printf '%s\n' "$*" >> "${FAKE_CAPTURE_ROOT}/git-calls.txt"
git_root=
if [[ ${1:-} == -C ]]; then
    git_root=$2
    shift 2
fi
if [[ ${1:-} == status && ${2:-} == --porcelain=v1 ]]; then
    exit 0
fi
if [[ ${1:-} != rev-parse || ${2:-} != --verify ]]; then
    echo "unexpected fake git invocation: $*" >&2
    exit 91
fi
case ${3:-} in
    HEAD)
        if [[ ${git_root} == */gp/tera/vendor ]]; then
            printf '%s\n' "${FAKE_CHECKED_OUT_GITLINK}"
        else
            printf '%s\n' "${FAKE_COMMIT}"
        fi
        ;;
    'HEAD^{tree}') printf '%s\n' "${FAKE_TREE}" ;;
    HEAD:gp/tera/vendor) printf '%s\n' "${FAKE_GITLINK}" ;;
    *)
        echo "unexpected fake git revision: ${3:-}" >&2
        exit 92
        ;;
esac
""",
    )
    _write_executable(
        fake_bin / "sha256sum",
        r"""#!/usr/bin/env bash
set -Eeuo pipefail
printf '%s  %s\n' "${FAKE_CATALOG_SHA256}" "${*: -1}"
""",
    )
    _write_executable(
        fake_bin / "scontrol",
        r"""#!/usr/bin/env bash
set -Eeuo pipefail
printf '%s\n' "${FAKE_SCONTROL_RECORD}"
""",
    )

    environment = {key: value for key, value in os.environ.items() if not key.startswith("F02B_")}
    environment.update(
        {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "FAKE_CAPTURE_ROOT": str(capture_root),
            "FAKE_CATALOG_SHA256": CATALOG_SHA256,
            "FAKE_COMMIT": "a" * 40,
            "FAKE_TREE": "b" * 40,
            "FAKE_GITLINK": "c" * 40,
            "FAKE_CHECKED_OUT_GITLINK": "c" * 40,
            "FAKE_SCONTROL_RECORD": (
                "JobId=9001 ArrayJobId=9000 ArrayTaskId=17 ArrayTaskThrottle=1 "
                "Partition=short NumNodes=1 NumCPUs=8 CPUs/Task=8 "
                "MinMemoryNode=64G TimeLimit=08:00:00 OverSubscribe=OK"
            ),
            "F02B_REPO_ROOT": str(deployed_repo),
            "F02B_DATA_ROOT": str(data_root),
            "F02B_CATALOG": str(catalog),
            "F02B_OUTPUT_ROOT": str(output_root),
            "F02B_EXPECTED_GIT_COMMIT": "a" * 40,
            "F02B_EXPECTED_GIT_TREE": "b" * 40,
            "SLURM_JOB_ID": "9001",
            "SLURM_ARRAY_JOB_ID": "9000",
            "SLURM_ARRAY_TASK_ID": "17",
            "SLURM_ARRAY_TASK_COUNT": "45",
            "SLURM_ARRAY_TASK_MIN": "0",
            "SLURM_ARRAY_TASK_MAX": "44",
            "SLURM_ARRAY_TASK_STEP": "1",
            "SLURM_JOB_NODELIST": "cpu001",
            "SLURM_JOB_NUM_NODES": "1",
            "SLURM_NTASKS": "1",
            "SLURM_CPUS_PER_TASK": "8",
            "SLURM_MEM_PER_NODE": "65536",
            "SLURM_JOB_PARTITION": "short",
        }
    )
    return FakeCluster(environment, capture_root, dataset, catalog, output_root)


def test_sbatch_directives_bind_the_exact_fit_array_resources() -> None:
    directives = [
        line for line in SBATCH_SCRIPT.read_text().splitlines() if line.startswith("#SBATCH ")
    ]

    assert directives == [
        "#SBATCH --job-name=f02b-fit",
        "#SBATCH --account=lucasbao",
        "#SBATCH --partition=short",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=64G",
        "#SBATCH --time=08:00:00",
        "#SBATCH --array=0-44%1",
        "#SBATCH --output=runs/f02b-fit-%A_%a.out",
        "#SBATCH --error=runs/f02b-fit-%A_%a.err",
    ]


def test_launcher_has_no_task_fallback_python_override_or_science_argument_surface() -> None:
    source = SBATCH_SCRIPT.read_text()

    assert "TASK_INDEX=${SLURM_ARRAY_TASK_ID}" in source
    assert "SLURM_ARRAY_TASK_ID:-" not in source
    assert "PYTHON_BIN=${REPO_ROOT}/.venv/bin/python" in source
    assert "F02B_PYTHON" not in source
    assert "eval " not in source
    assert "#SBATCH --exclusive" not in source
    assert "#SBATCH --gres" not in source
    for obsolete_evidence in (
        "F02B_EXCLUSIVE_VERIFIED",
        "F02B_EXCLUSIVE_MODE",
        "F02B_REQUESTED_GPU_COUNT",
        "F02B_VISIBLE_GPU_COUNT",
        "F02B_VISIBLE_GPU_MODELS_JSON",
        "F02B_VISIBLE_GPU_MEMORY_BYTES_JSON",
        "F02B_AVAILABLE_CPU_COUNT",
        "F02B_AVAILABLE_HOST_MEMORY_BYTES",
        "F02B_REQUESTED_WALLTIME_SECONDS",
        "F02B_WALLTIME_LIMIT_SECONDS",
        "F02B_ARRAY_CONCURRENCY",
    ):
        assert f"export {obsolete_evidence}=" not in source
    for forbidden_argument in (
        "--batch-size",
        "--kernel",
        "--lr",
        "--seed",
        "--train-steps",
        "--training-m",
    ):
        assert forbidden_argument not in source


def test_fake_allocation_preflights_but_does_not_export_attested_evidence(
    fake_cluster: FakeCluster,
) -> None:
    result = fake_cluster.run()

    assert result.returncode == 0, result.stderr
    assert (fake_cluster.capture_root / "runner-args.txt").read_text().splitlines() == [
        "-m",
        "experiments.f02b_calibration_fit",
        "--task-index",
        "17",
        "--dataset",
        str(fake_cluster.dataset),
        "--catalog",
        str(fake_cluster.catalog),
        "--output-root",
        str(fake_cluster.output_root),
    ]
    runner_environment = dict(
        line.split("=", 1)
        for line in (fake_cluster.capture_root / "runner-env.txt").read_text().splitlines()
    )
    assert runner_environment["SLURM_JOB_ID"] == "9001"
    assert runner_environment["SLURM_ARRAY_JOB_ID"] == "9000"
    assert runner_environment["SLURM_ARRAY_TASK_ID"] == "17"
    assert runner_environment["SLURM_CPUS_PER_TASK"] == "8"
    assert runner_environment["SLURM_JOB_PARTITION"] == "short"
    assert runner_environment["OMP_NUM_THREADS"] == "8"
    assert runner_environment["MKL_NUM_THREADS"] == "8"
    assert runner_environment["OPENBLAS_NUM_THREADS"] == "8"
    assert runner_environment["NUMEXPR_NUM_THREADS"] == "8"
    assert runner_environment["CUDA_VISIBLE_DEVICES"] == ""
    for forbidden in (
        "F02B_EXCLUSIVE_VERIFIED",
        "F02B_EXCLUSIVE_MODE",
        "F02B_REQUESTED_GPU_COUNT",
        "F02B_VISIBLE_GPU_COUNT",
        "F02B_VISIBLE_GPU_MODELS_JSON",
        "F02B_VISIBLE_GPU_MEMORY_BYTES_JSON",
        "F02B_AVAILABLE_CPU_COUNT",
        "F02B_AVAILABLE_HOST_MEMORY_BYTES",
        "F02B_REQUESTED_WALLTIME_SECONDS",
        "F02B_WALLTIME_LIMIT_SECONDS",
        "F02B_ARRAY_CONCURRENCY",
    ):
        assert forbidden not in runner_environment


def test_interactivegpu_partition_is_rejected_before_git(fake_cluster: FakeCluster) -> None:
    environment = dict(fake_cluster.environment)
    environment["SLURM_JOB_PARTITION"] = "interactivegpu"
    environment["FAKE_SCONTROL_RECORD"] = environment["FAKE_SCONTROL_RECORD"].replace(
        "Partition=short", "Partition=interactivegpu"
    )

    result = fake_cluster.run(environment)

    assert result.returncode == 2
    assert "only the short partition" in result.stderr
    assert not (fake_cluster.capture_root / "git-calls.txt").exists()
    assert not (fake_cluster.capture_root / "runner-args.txt").exists()


def test_unregistered_partition_is_rejected_before_git(fake_cluster: FakeCluster) -> None:
    environment = dict(fake_cluster.environment)
    environment["SLURM_JOB_PARTITION"] = "gpu"

    result = fake_cluster.run(environment)

    assert result.returncode == 2
    assert "only the short partition" in result.stderr
    assert not (fake_cluster.capture_root / "git-calls.txt").exists()
    assert not (fake_cluster.capture_root / "runner-args.txt").exists()


@pytest.mark.parametrize(
    ("old", "new", "error"),
    [
        (
            "OverSubscribe=OK",
            "OverSubscribe=EXCLUSIVE",
            "scontrol-confirmed shared allocation",
        ),
        ("ArrayTaskThrottle=1", "ArrayTaskThrottle=2", "array concurrency of one"),
        ("TimeLimit=08:00:00", "TimeLimit=04:00:00", "eight-hour job time limit"),
        ("CPUs/Task=8", "CPUs/Task=16", "8-CPU task request"),
        ("MinMemoryNode=64G", "MinMemoryNode=32G", "64-GiB node-memory request"),
        (
            "MinMemoryNode=64G",
            "TresPerNode=gres/gpu:l40s:1 MinMemoryNode=64G",
            "requests any GPU",
        ),
    ],
)
def test_fake_allocation_fails_closed_on_scontrol_contract_drift(
    fake_cluster: FakeCluster,
    old: str,
    new: str,
    error: str,
) -> None:
    environment = dict(fake_cluster.environment)
    environment["FAKE_SCONTROL_RECORD"] = environment["FAKE_SCONTROL_RECORD"].replace(old, new)

    result = fake_cluster.run(environment)

    assert result.returncode == 2
    assert error in result.stderr
    assert not (fake_cluster.capture_root / "runner-args.txt").exists()


def test_malformed_expected_identity_is_rejected_before_git(fake_cluster: FakeCluster) -> None:
    environment = dict(fake_cluster.environment)
    environment["F02B_EXPECTED_GIT_COMMIT"] = "A" * 40

    result = fake_cluster.run(environment)

    assert result.returncode == 2
    assert "full lowercase 40-hex" in result.stderr
    assert not (fake_cluster.capture_root / "git-calls.txt").exists()
    assert not (fake_cluster.capture_root / "runner-args.txt").exists()


@pytest.mark.parametrize(
    ("environment_name", "replacement", "error"),
    [
        ("FAKE_COMMIT", "d" * 40, "deployed git commit"),
        ("FAKE_TREE", "d" * 40, "deployed git tree"),
        ("FAKE_CHECKED_OUT_GITLINK", "d" * 40, "deployed gitlink"),
        ("FAKE_CATALOG_SHA256", "d" * 64, "catalog SHA-256 mismatch"),
    ],
)
def test_fake_deployment_fails_closed_on_source_or_catalog_identity_drift(
    fake_cluster: FakeCluster,
    environment_name: str,
    replacement: str,
    error: str,
) -> None:
    environment = dict(fake_cluster.environment)
    environment[environment_name] = replacement

    result = fake_cluster.run(environment)

    assert result.returncode == 2
    assert error in result.stderr
    assert not (fake_cluster.capture_root / "runner-args.txt").exists()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("F02B_SEED", "999"),
        ("F02B_EXCLUSIVE_VERIFIED", "1"),
        ("F02B_VISIBLE_GPU_MODELS_JSON", '["NVIDIA L40S"]'),
        ("F02B_AVAILABLE_CPU_COUNT", "16"),
    ],
)
def test_unregistered_f02b_environment_input_is_rejected(
    fake_cluster: FakeCluster,
    name: str,
    value: str,
) -> None:
    environment = dict(fake_cluster.environment)
    environment[name] = value

    result = fake_cluster.run(environment)

    assert result.returncode == 2
    assert f"unregistered F02b environment input: {name}" in result.stderr
    assert not (fake_cluster.capture_root / "git-calls.txt").exists()
    assert not (fake_cluster.capture_root / "runner-args.txt").exists()
