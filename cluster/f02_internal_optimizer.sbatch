#!/usr/bin/env bash
#SBATCH --job-name=f02-int-opt
#SBATCH --account=lucasbao
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --array=0-134%1
#SBATCH --output=runs/f02-internal-opt-%A_%a.out
#SBATCH --error=runs/f02-internal-opt-%A_%a.err

set -Eeuo pipefail
umask 027

readonly EXPECTED_CATALOG_SHA256=2dee429bdaf50cc78cb40ba8b038f7e4731bb07127927c55607b528c1db66942

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${F02_INTERNAL_REPO_ROOT:-/projects/lucasbao/tengcc/auto_dgp2}
PYTHON_BIN=${F02_INTERNAL_PYTHON:-${REPO_ROOT}/.venv/bin/python}
DATA_DIR=${F02_INTERNAL_DATA_DIR:-/projects/lucasbao/tengcc/datasets/f02_nbody_v1}
CATALOG_PATH=${F02_INTERNAL_CATALOG:-${REPO_ROOT}/runs/f02_nbody_data/job-2810370/catalog.json}
RUN_ROOT=${F02_INTERNAL_RUN_ROOT:-${REPO_ROOT}/runs/f02_internal_optimizer}

if [[ -z ${SLURM_JOB_ID:-} || -z ${SLURM_ARRAY_TASK_ID:-} ]]; then
    echo "refusing to run: F02 internal tasks require a Slurm array allocation" >&2
    exit 2
fi
if [[ ${SLURM_JOB_NUM_NODES:-0} != 1 ]]; then
    echo "refusing to run: expected exactly one allocated node" >&2
    exit 2
fi
if [[ ! -x ${PYTHON_BIN} ]]; then
    echo "Python executable is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi

TASK_INDEX=${SLURM_ARRAY_TASK_ID}
mapfile -t TASK_FIELDS < <(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/f02_internal_grid.py" \
        --task-index "${TASK_INDEX}" --format lines
)
if (( ${#TASK_FIELDS[@]} != 7 )); then
    echo "task map returned ${#TASK_FIELDS[@]} fields instead of seven" >&2
    exit 2
fi
MAPPED_INDEX=${TASK_FIELDS[0]}
REPLICA=${TASK_FIELDS[1]}
N_PARTICLES=${TASK_FIELDS[2]}
N_DIMS=${TASK_FIELDS[3]}
TRAIN_STEPS=${TASK_FIELDS[4]}
SEED=${TASK_FIELDS[5]}
DATASET_STEM=${TASK_FIELDS[6]}
if [[ ${MAPPED_INDEX} != "${TASK_INDEX}" ]]; then
    echo "task-map index mismatch: requested ${TASK_INDEX}, got ${MAPPED_INDEX}" >&2
    exit 2
fi

ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}
TASK_LABEL=replica-${REPLICA}-n-${N_PARTICLES}-steps-${TRAIN_STEPS}-seed-${SEED}
TASK_DIR=${RUN_ROOT}/job-${ARRAY_JOB_ID}/task-${TASK_INDEX}-${TASK_LABEL}
mkdir -p "${TASK_DIR}"

exec > >(tee -a "${TASK_DIR}/runner.log") 2>&1

finish() {
    status=$?
    trap - EXIT
    set +e
    printf '%s\n' "${status}" > "${TASK_DIR}/exit-code.txt"
    date -u +'%Y-%m-%dT%H:%M:%SZ' > "${TASK_DIR}/finished-at.txt"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
            --format=csv,noheader > "${TASK_DIR}/gpu-processes-after.csv" 2>&1
    fi
    artifact_paths=(
        "${TASK_DIR}/command.txt"
        "${TASK_DIR}/dataset-files.sha256"
        "${TASK_DIR}/dependency-files.sha256"
        "${TASK_DIR}/dependency-packages.txt"
        "${TASK_DIR}/exit-code.txt"
        "${TASK_DIR}/finished-at.txt"
        "${TASK_DIR}/git-status.txt"
        "${TASK_DIR}/git-submodules.txt"
        "${TASK_DIR}/pip-check.txt"
        "${TASK_DIR}/provenance.env"
        "${TASK_DIR}/runtime.json"
        "${TASK_DIR}/slurm-job.txt"
        "${TASK_DIR}/slurm-jobs-on-node.txt"
        "${TASK_DIR}/slurm-node.txt"
        "${TASK_DIR}/source-files.sha256"
    )
    if [[ -f ${TASK_DIR}/result.json ]]; then
        artifact_paths+=("${TASK_DIR}/result.json")
    fi
    if [[ -f ${TASK_DIR}/gpu.csv ]]; then
        artifact_paths+=("${TASK_DIR}/gpu.csv")
    fi
    if [[ -f ${TASK_DIR}/gpu-topology.txt ]]; then
        artifact_paths+=("${TASK_DIR}/gpu-topology.txt")
    fi
    if [[ -f ${TASK_DIR}/gpu-processes-before.csv ]]; then
        artifact_paths+=("${TASK_DIR}/gpu-processes-before.csv")
    fi
    if [[ -f ${TASK_DIR}/gpu-processes-after.csv ]]; then
        artifact_paths+=("${TASK_DIR}/gpu-processes-after.csv")
    fi
    existing_artifacts=()
    for path in "${artifact_paths[@]}"; do
        [[ -f ${path} ]] && existing_artifacts+=("${path}")
    done
    if (( ${#existing_artifacts[@]} > 0 )); then
        sha256sum "${existing_artifacts[@]}" > "${TASK_DIR}/artifacts.sha256"
    fi
    exit "${status}"
}
trap finish EXIT

echo "F02 internal task directory: ${TASK_DIR}"
echo "F02 internal task: index=${TASK_INDEX} ${TASK_LABEL}"

for required in \
    "${REPO_ROOT}/experiments/f02_internal_task.py" \
    "${REPO_ROOT}/cluster/f02_internal_grid.py" \
    "${REPO_ROOT}/gp/tera/vendor/src/gp_sim_kl/simulation.py" \
    "${REPO_ROOT}/pyproject.toml" \
    "${REPO_ROOT}/uv.lock"; do
    if [[ ! -f ${required} ]]; then
        echo "required source is unavailable: ${required}" >&2
        exit 2
    fi
done

cd "${REPO_ROOT}"
GIT_STATUS=$(git status --porcelain=v1 --untracked-files=all)
printf '%s\n' "${GIT_STATUS}" > "${TASK_DIR}/git-status.txt"
if [[ -n ${GIT_STATUS} ]]; then
    echo "refusing to run an uncommitted source tree" >&2
    printf '%s\n' "${GIT_STATUS}" >&2
    exit 2
fi
git submodule status --recursive > "${TASK_DIR}/git-submodules.txt"
if grep -Eq '^[-+U]' "${TASK_DIR}/git-submodules.txt"; then
    echo "refusing to run with an uninitialized or non-HEAD submodule" >&2
    exit 2
fi

# Authenticate the strict aggregate before touching any corpus payload.  The
# model runner independently verifies that this exact catalog authorizes the
# selected bundle and validates the bundle's own SHA-256 sidecars.
if [[ ! -f ${CATALOG_PATH} ]]; then
    echo "F02 strict catalog is unavailable: ${CATALOG_PATH}" >&2
    exit 2
fi
ACTUAL_CATALOG_SHA256=$(sha256sum "${CATALOG_PATH}" | awk '{print $1}')
if [[ ${ACTUAL_CATALOG_SHA256} != "${EXPECTED_CATALOG_SHA256}" ]]; then
    echo "F02 catalog SHA-256 mismatch" >&2
    echo "expected: ${EXPECTED_CATALOG_SHA256}" >&2
    echo "actual:   ${ACTUAL_CATALOG_SHA256}" >&2
    exit 2
fi

DATASET_PATH=${DATA_DIR}/${DATASET_STEM}.npz
METADATA_PATH=${DATA_DIR}/${DATASET_STEM}.metadata.json
MANIFEST_PATH=${DATA_DIR}/${DATASET_STEM}.sha256.json
for required in "${DATASET_PATH}" "${METADATA_PATH}" "${MANIFEST_PATH}"; do
    if [[ ! -f ${required} ]]; then
        echo "required frozen-corpus artifact is unavailable: ${required}" >&2
        exit 2
    fi
done
sha256sum \
    "${CATALOG_PATH}" \
    "${DATASET_PATH}" \
    "${METADATA_PATH}" \
    "${MANIFEST_PATH}" \
    > "${TASK_DIR}/dataset-files.sha256"

SLURM_JOB_RECORD=$(scontrol show job "${SLURM_JOB_ID}" -o)
printf '%s\n' "${SLURM_JOB_RECORD}" > "${TASK_DIR}/slurm-job.txt"

NODE_NAME=${SLURMD_NODENAME:-}
if [[ -z ${NODE_NAME} ]]; then
    mapfile -t ALLOCATED_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
    if (( ${#ALLOCATED_NODES[@]} != 1 )); then
        echo "could not resolve exactly one allocated node" >&2
        exit 2
    fi
    NODE_NAME=${ALLOCATED_NODES[0]}
fi
SLURM_NODE_RECORD=$(scontrol show node "${NODE_NAME}" -o)
SLURM_NODE_JOBS=$(
    squeue --noheader --nodes "${NODE_NAME}" --states=RUNNING --format='%A' | sort -u
)
printf '%s\n' "${SLURM_NODE_RECORD}" > "${TASK_DIR}/slurm-node.txt"
printf '%s\n' "${SLURM_NODE_JOBS}" > "${TASK_DIR}/slurm-jobs-on-node.txt"

tres_value() {
    local tres_list=$1
    local key=$2
    local entry
    local entries
    IFS=',' read -r -a entries <<< "${tres_list}"
    for entry in "${entries[@]}"; do
        if [[ ${entry} == "${key}="* ]]; then
            printf '%s' "${entry#*=}"
            return 0
        fi
    done
    return 1
}

EXCLUSIVE_MODE=
if [[ ${SLURM_JOB_RECORD} == *"OverSubscribe=EXCLUSIVE"* ]]; then
    EXCLUSIVE_MODE=slurm_explicit
elif [[ ${SLURM_JOB_RECORD} == *"OverSubscribe=NO"* ]]; then
    JOB_ALLOC_TRES=
    NODE_CONFIG_TRES=
    if [[ ${SLURM_JOB_RECORD} =~ AllocTRES=([^[:space:]]+) ]]; then
        JOB_ALLOC_TRES=${BASH_REMATCH[1]}
    fi
    if [[ ${SLURM_NODE_RECORD} =~ CfgTRES=([^[:space:]]+) ]]; then
        NODE_CONFIG_TRES=${BASH_REMATCH[1]}
    fi
    JOB_CPU=$(tres_value "${JOB_ALLOC_TRES}" cpu || true)
    NODE_CPU=$(tres_value "${NODE_CONFIG_TRES}" cpu || true)
    JOB_GPU=$(tres_value "${JOB_ALLOC_TRES}" gres/gpu || true)
    NODE_GPU=$(tres_value "${NODE_CONFIG_TRES}" gres/gpu || true)
    CURRENT_JOB_FAMILY=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}
    if [[ -n ${JOB_CPU} && ${JOB_CPU} == "${NODE_CPU}" \
        && -n ${JOB_GPU} && ${JOB_GPU} == "${NODE_GPU}" \
        && ${JOB_GPU} != 0 && ${SLURM_NODE_JOBS} == "${CURRENT_JOB_FAMILY}" ]]; then
        EXCLUSIVE_MODE=slurm_no_oversubscribe_full_node_sole_job
    fi
fi
if [[ -z ${EXCLUSIVE_MODE} ]]; then
    echo "refusing to run without a verified exclusive full-node allocation" >&2
    exit 2
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONHASHSEED=${SEED}
export PYTHONUNBUFFERED=1
export F02_INTERNAL_SLURM_EXCLUSIVE_VERIFIED=1
export F02_INTERNAL_SLURM_EXCLUSIVE_MODE=${EXCLUSIVE_MODE}

{
    printf 'captured_at_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    printf 'repo_root=%s\n' "${REPO_ROOT}"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'git_tree=%s\n' "$(git rev-parse 'HEAD^{tree}')"
    printf 'git_describe=%s\n' "$(git describe --always --dirty --tags)"
    printf 'tera_gitlink=%s\n' "$(git rev-parse HEAD:gp/tera/vendor)"
    printf 'hostname=%s\n' "$(hostname -f)"
    printf 'python=%s\n' "${PYTHON_BIN}"
    printf 'array_job_id=%s\n' "${ARRAY_JOB_ID}"
    printf 'array_task_id=%s\n' "${TASK_INDEX}"
    printf 'replica=%s\n' "${REPLICA}"
    printf 'n_particles=%s\n' "${N_PARTICLES}"
    printf 'n_dims=%s\n' "${N_DIMS}"
    printf 'train_steps=%s\n' "${TRAIN_STEPS}"
    printf 'seed=%s\n' "${SEED}"
    printf 'dataset_stem=%s\n' "${DATASET_STEM}"
    printf 'catalog_path=%s\n' "${CATALOG_PATH}"
    printf 'catalog_sha256=%s\n' "${ACTUAL_CATALOG_SHA256}"
    printf 'exclusive_verification_mode=%s\n' "${EXCLUSIVE_MODE}"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID}"
    printf 'slurm_job_nodelist=%s\n' "${SLURM_JOB_NODELIST}"
    printf 'slurm_cpus_per_task=%s\n' "${SLURM_CPUS_PER_TASK}"
    printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-}"
} > "${TASK_DIR}/provenance.env"

git ls-files -z -- \
    cluster/f02_internal_grid.py \
    cluster/f02_internal_optimizer.sbatch \
    cluster/submit_f02_internal_optimizer.sh \
    data/generate_nbody_confirmatory.py \
    data/load_nbody_confirmatory.py \
    experiments/f02_design.py \
    experiments/f02_internal_models.py \
    experiments/f02_internal_task.py \
    experiments/f02_metrics.py \
    gp/orbit \
    pyproject.toml \
    uv.lock \
    | sort -z \
    | xargs -0 sha256sum \
    > "${TASK_DIR}/source-files.sha256"
sha256sum pyproject.toml uv.lock > "${TASK_DIR}/dependency-files.sha256"
"${PYTHON_BIN}" -m pip freeze --all | LC_ALL=C sort > "${TASK_DIR}/dependency-packages.txt"
"${PYTHON_BIN}" -m pip check > "${TASK_DIR}/pip-check.txt"

"${PYTHON_BIN}" -c '
import importlib.metadata
import json
import os
import platform
import sys

import numpy
import scipy
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable inside the allocated L40S task")
devices = []
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    devices.append({
        "capability": list(torch.cuda.get_device_capability(index)),
        "index": index,
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
    })
if len(devices) != 1 or "L40S" not in devices[0]["name"].upper():
    raise RuntimeError(f"expected one visible L40S GPU, got {devices!r}")
record = {
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "numpy": numpy.__version__,
    "platform": platform.platform(),
    "python_executable": sys.executable,
    "python_version": sys.version,
    "scipy": scipy.__version__,
    "torch": torch.__version__,
    "torch_cuda_available": torch.cuda.is_available(),
    "torch_cuda_devices": devices,
    "torch_cuda_runtime": torch.version.cuda,
}
print(json.dumps(record, indent=2, sort_keys=True))
' > "${TASK_DIR}/runtime.json"

nvidia-smi \
    --query-gpu=index,uuid,name,driver_version,memory.total \
    --format=csv,noheader > "${TASK_DIR}/gpu.csv"
nvidia-smi topo -m > "${TASK_DIR}/gpu-topology.txt"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader > "${TASK_DIR}/gpu-processes-before.csv" || true

RUN_COMMAND=(
    "${PYTHON_BIN}"
    experiments/f02_internal_task.py
    "${DATASET_PATH}"
    --catalog "${CATALOG_PATH}"
    --out "${TASK_DIR}/result.json"
    --evaluation-split validation
    --evaluation-design optimizer_selection
    --training-m 20
    --train-steps "${TRAIN_STEPS}"
    --train-epochs 0
    --kernel rbf
    --outputscale 1.0
    --sigma-f 0.001
    --sigma-g 0.001
    --lengthscale 1.0
    --seed "${SEED}"
    --batch-size 256
    --lr 0.01
    --weight-decay 0.0
    --candidate-m none
    --cg-tolerance 1e-5
    --use-preconditioner
    --function-jitter 1e-8
    --reduced-jitter 1e-8
    --dtype float32
    --device cuda
)
{
    printf 'command='
    printf '%q ' "${RUN_COMMAND[@]}"
    printf '\n'
} > "${TASK_DIR}/command.txt"

"${RUN_COMMAND[@]}"
