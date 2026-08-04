#!/usr/bin/env bash
#SBATCH --job-name=f02b-fit
#SBATCH --account=lucasbao
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --array=0-44%1
#SBATCH --output=runs/f02b-fit-%A_%a.out
#SBATCH --error=runs/f02b-fit-%A_%a.err

set -Eeuo pipefail
umask 027

readonly EXPECTED_CATALOG_SHA256=2dee429bdaf50cc78cb40ba8b038f7e4731bb07127927c55607b528c1db66942
readonly EXPECTED_TASK_COUNT=45
readonly EXPECTED_MEMORY_MIB=65536

# Fail closed on inherited F02b controls.  The only caller-owned inputs are
# deployment paths and the exact source identity approved for this deployment.
while IFS= read -r environment_name; do
    case ${environment_name} in
        F02B_REPO_ROOT|F02B_DATA_ROOT|F02B_CATALOG|F02B_OUTPUT_ROOT|\
            F02B_EXPECTED_GIT_COMMIT|F02B_EXPECTED_GIT_TREE)
            ;;
        F02B_*)
            echo "refusing unregistered F02b environment input: ${environment_name}" >&2
            exit 2
            ;;
    esac
done < <(compgen -e)

REPO_ROOT=${F02B_REPO_ROOT:-/projects/lucasbao/tengcc/auto_dgp2}
DATA_ROOT=${F02B_DATA_ROOT:-/projects/lucasbao/tengcc/datasets/f02_nbody_v1}
CATALOG_PATH=${F02B_CATALOG:-${REPO_ROOT}/runs/f02_nbody_data/job-2810370/catalog.json}
OUTPUT_ROOT=${F02B_OUTPUT_ROOT:-${REPO_ROOT}/runs/f02b_calibration_fits}
EXPECTED_GIT_COMMIT=${F02B_EXPECTED_GIT_COMMIT:-}
EXPECTED_GIT_TREE=${F02B_EXPECTED_GIT_TREE:-}

# Validate caller-supplied identities before the first git invocation.
if [[ ! ${EXPECTED_GIT_COMMIT} =~ ^[0-9a-f]{40}$ ]]; then
    echo "F02B_EXPECTED_GIT_COMMIT must be a full lowercase 40-hex object ID" >&2
    exit 2
fi
if [[ ! ${EXPECTED_GIT_TREE} =~ ^[0-9a-f]{40}$ ]]; then
    echo "F02B_EXPECTED_GIT_TREE must be a full lowercase 40-hex object ID" >&2
    exit 2
fi
for infrastructure_path in "${REPO_ROOT}" "${DATA_ROOT}" "${CATALOG_PATH}" "${OUTPUT_ROOT}"; do
    if [[ ${infrastructure_path} != /* ]]; then
        echo "F02b infrastructure paths must be absolute: ${infrastructure_path}" >&2
        exit 2
    fi
done

PYTHON_BIN=${REPO_ROOT}/.venv/bin/python
GRID_HELPER=${REPO_ROOT}/cluster/f02b_calibration_grid.py
FIT_RUNNER=${REPO_ROOT}/experiments/f02b_calibration_fit.py
TERA_ROOT=${REPO_ROOT}/gp/tera/vendor

for required in "${PYTHON_BIN}" "${GRID_HELPER}" "${FIT_RUNNER}"; do
    if [[ ! -f ${required} ]]; then
        echo "required deployed file is unavailable: ${required}" >&2
        exit 2
    fi
done
if [[ ! -x ${PYTHON_BIN} ]]; then
    echo "repository .venv Python is not executable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -d ${TERA_ROOT} ]]; then
    echo "TERA vendor checkout is unavailable: ${TERA_ROOT}" >&2
    exit 2
fi
if [[ ! -f ${CATALOG_PATH} ]]; then
    echo "frozen F02 catalog is unavailable: ${CATALOG_PATH}" >&2
    exit 2
fi

for required_environment in \
    SLURM_JOB_ID \
    SLURM_ARRAY_JOB_ID \
    SLURM_ARRAY_TASK_ID \
    SLURM_ARRAY_TASK_COUNT \
    SLURM_ARRAY_TASK_MIN \
    SLURM_ARRAY_TASK_MAX \
    SLURM_ARRAY_TASK_STEP \
    SLURM_JOB_NODELIST \
    SLURM_JOB_NUM_NODES \
    SLURM_NTASKS \
    SLURM_CPUS_PER_TASK \
    SLURM_JOB_PARTITION; do
    if [[ -z ${!required_environment:-} ]]; then
        echo "required Slurm environment is missing: ${required_environment}" >&2
        exit 2
    fi
done
if [[ ! ${SLURM_JOB_ID} =~ ^[0-9]+$ || ! ${SLURM_ARRAY_JOB_ID} =~ ^[0-9]+$ ]]; then
    echo "Slurm job identifiers must be decimal integers" >&2
    exit 2
fi
if [[ ! ${SLURM_ARRAY_TASK_ID} =~ ^[0-9]+$ ]]; then
    echo "SLURM_ARRAY_TASK_ID must be a non-negative decimal integer" >&2
    exit 2
fi
if [[ ${SLURM_ARRAY_TASK_COUNT} != "${EXPECTED_TASK_COUNT}" \
    || ${SLURM_ARRAY_TASK_MIN} != 0 \
    || ${SLURM_ARRAY_TASK_MAX} != 44 \
    || ${SLURM_ARRAY_TASK_STEP} != 1 ]]; then
    echo "refusing a Slurm array other than the exact 0-44 F02b fit grid" >&2
    exit 2
fi
if [[ ${SLURM_JOB_NUM_NODES} != 1 || ${SLURM_NTASKS} != 1 ]]; then
    echo "F02b fits require exactly one node and one task" >&2
    exit 2
fi
if [[ ${SLURM_CPUS_PER_TASK} != 16 ]]; then
    echo "F02b fits require exactly 16 requested CPUs per task" >&2
    exit 2
fi
if [[ -n ${SLURM_MEM_PER_NODE:-} && ${SLURM_MEM_PER_NODE} != "${EXPECTED_MEMORY_MIB}" ]]; then
    echo "F02b fits require exactly 64 GiB of requested host memory" >&2
    exit 2
fi
case ${SLURM_JOB_PARTITION} in
    short|interactivegpu) ;;
    *)
        echo "F02b fits permit only the short or interactivegpu partition" >&2
        exit 2
        ;;
esac

# SLURM_ARRAY_TASK_ID is the sole source of runner task identity.
TASK_INDEX=${SLURM_ARRAY_TASK_ID}
if (( TASK_INDEX < 0 || TASK_INDEX >= EXPECTED_TASK_COUNT )); then
    echo "array task index is outside the exact F02b fit grid: ${TASK_INDEX}" >&2
    exit 2
fi

cd "${REPO_ROOT}"
GIT_STATUS=$(git status --porcelain=v1 --untracked-files=all)
if [[ -n ${GIT_STATUS} ]]; then
    echo "refusing to run from an uncommitted repository" >&2
    printf '%s\n' "${GIT_STATUS}" >&2
    exit 2
fi
TERA_STATUS=$(git -C "${TERA_ROOT}" status --porcelain=v1 --untracked-files=all)
if [[ -n ${TERA_STATUS} ]]; then
    echo "refusing to run from an uncommitted TERA vendor checkout" >&2
    printf '%s\n' "${TERA_STATUS}" >&2
    exit 2
fi

ACTUAL_GIT_COMMIT=$(git rev-parse --verify HEAD)
ACTUAL_GIT_TREE=$(git rev-parse --verify 'HEAD^{tree}')
if [[ ${ACTUAL_GIT_COMMIT} != "${EXPECTED_GIT_COMMIT}" ]]; then
    echo "deployed git commit does not match F02B_EXPECTED_GIT_COMMIT" >&2
    exit 2
fi
if [[ ${ACTUAL_GIT_TREE} != "${EXPECTED_GIT_TREE}" ]]; then
    echo "deployed git tree does not match F02B_EXPECTED_GIT_TREE" >&2
    exit 2
fi
EXPECTED_TERA_GITLINK=$(git rev-parse --verify HEAD:gp/tera/vendor)
ACTUAL_TERA_COMMIT=$(git -C "${TERA_ROOT}" rev-parse --verify HEAD)
if [[ ! ${EXPECTED_TERA_GITLINK} =~ ^[0-9a-f]{40}$ \
    || ${ACTUAL_TERA_COMMIT} != "${EXPECTED_TERA_GITLINK}" ]]; then
    echo "TERA vendor checkout does not match the deployed gitlink" >&2
    exit 2
fi

read -r ACTUAL_CATALOG_SHA256 _ < <(sha256sum -- "${CATALOG_PATH}")
if [[ ${ACTUAL_CATALOG_SHA256} != "${EXPECTED_CATALOG_SHA256}" ]]; then
    echo "F02 catalog SHA-256 mismatch" >&2
    exit 2
fi

mapfile -t TASK_FIELDS < <(
    "${PYTHON_BIN}" "${GRID_HELPER}" \
        --matrix fit \
        --task-index "${TASK_INDEX}" \
        --format lines
)
if (( ${#TASK_FIELDS[@]} != 11 )); then
    echo "F02b fit grid returned ${#TASK_FIELDS[@]} fields instead of eleven" >&2
    exit 2
fi
if [[ ${TASK_FIELDS[0]} != "${TASK_INDEX}" ]]; then
    echo "F02b fit-grid task index does not match SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi
DATASET_STEM=${TASK_FIELDS[9]}
if [[ ! ${DATASET_STEM} =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "F02b fit grid returned an unsafe dataset stem" >&2
    exit 2
fi
DATASET_PATH=${DATA_ROOT}/${DATASET_STEM}.npz
if [[ ! -f ${DATASET_PATH} ]]; then
    echo "frozen development dataset is unavailable: ${DATASET_PATH}" >&2
    exit 2
fi

SLURM_JOB_RECORD=$(scontrol show job "${SLURM_JOB_ID}" -o)
if [[ " ${SLURM_JOB_RECORD} " != *" OverSubscribe=EXCLUSIVE "* ]]; then
    echo "refusing to run without scontrol-confirmed exclusive allocation" >&2
    exit 2
fi
if [[ " ${SLURM_JOB_RECORD} " != *" ArrayTaskThrottle=1 "* ]]; then
    echo "refusing to run without an effective array concurrency of one" >&2
    exit 2
fi
if [[ " ${SLURM_JOB_RECORD} " != *" TimeLimit=08:00:00 "* ]]; then
    echo "refusing to run without an exact eight-hour job time limit" >&2
    exit 2
fi
if [[ " ${SLURM_JOB_RECORD} " != *" CPUs/Task=16 "* ]]; then
    echo "refusing to run without an exact 16-CPU task request" >&2
    exit 2
fi
if [[ " ${SLURM_JOB_RECORD} " != *" MinMemoryNode=64G "* ]]; then
    echo "refusing to run without an exact 64-GiB node-memory request" >&2
    exit 2
fi
if [[ " ${SLURM_JOB_RECORD} " != *" TresPerNode=gres/gpu:l40s:1 "* ]]; then
    echo "refusing to run without an exact one-L40S request" >&2
    exit 2
fi

# Query the Python process's actual CPU affinity and CUDA-visible devices with
# the same repository interpreter that will execute the fit runner.
mapfile -t HARDWARE_FIELDS < <(
    "${PYTHON_BIN}" -c '
import json
import os

import torch

affinity_count = len(os.sched_getaffinity(0))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
devices = [torch.cuda.get_device_properties(index) for index in range(torch.cuda.device_count())]
if affinity_count < 16:
    raise SystemExit("fewer than 16 CPUs are available in process affinity")
if not devices:
    raise SystemExit("no CUDA devices are visible")
if any(device.name != "NVIDIA L40S" for device in devices):
    raise SystemExit("a visible CUDA device is not NVIDIA L40S")
if any(device.total_memory < 48_000_000_000 for device in devices):
    raise SystemExit("a visible CUDA device has less than 48e9 bytes")
print(affinity_count)
print(len(devices))
print(json.dumps([device.name for device in devices], separators=(",", ":")))
print(json.dumps([device.total_memory for device in devices], separators=(",", ":")))
'
)
if (( ${#HARDWARE_FIELDS[@]} != 4 )); then
    echo "could not obtain the exact F02b runtime allocation fields" >&2
    exit 2
fi
AVAILABLE_CPU_COUNT=${HARDWARE_FIELDS[0]}
VISIBLE_GPU_COUNT=${HARDWARE_FIELDS[1]}
VISIBLE_GPU_MODELS_JSON=${HARDWARE_FIELDS[2]}
VISIBLE_GPU_MEMORY_BYTES_JSON=${HARDWARE_FIELDS[3]}
if [[ ! ${AVAILABLE_CPU_COUNT} =~ ^[0-9]+$ || ! ${VISIBLE_GPU_COUNT} =~ ^[0-9]+$ ]]; then
    echo "runtime CPU/GPU counts are not non-negative decimal integers" >&2
    exit 2
fi
if (( AVAILABLE_CPU_COUNT < 16 || VISIBLE_GPU_COUNT < 1 )); then
    echo "runtime allocation exposes insufficient CPUs or GPUs" >&2
    exit 2
fi

RUN_COMMAND=(
    "${PYTHON_BIN}"
    -m
    experiments.f02b_calibration_fit
    --task-index
    "${TASK_INDEX}"
    --dataset
    "${DATASET_PATH}"
    --catalog
    "${CATALOG_PATH}"
    --output-root
    "${OUTPUT_ROOT}"
)
"${RUN_COMMAND[@]}"
