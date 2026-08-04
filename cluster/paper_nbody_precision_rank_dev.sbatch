#!/usr/bin/env bash
#SBATCH --job-name=prism-dev
#SBATCH --account=lucasbao
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%2
#SBATCH --output=runs/prism-dev-%A_%a.out
#SBATCH --error=runs/prism-dev-%A_%a.err

set -Eeuo pipefail
umask 027

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""

REPO_ROOT=${PAPER_NBODY_REPO_ROOT:-/projects/lucasbao/tengcc/auto_dgp2}
PYTHON_BIN=${REPO_ROOT}/.venv/bin/python
SOURCE_ROOT=${PAPER_NBODY_SOURCE_ROOT:-${REPO_ROOT}/runs/paper_nbody_v4}
OUTPUT_ROOT=${PAPER_NBODY_DEV_OUTPUT_ROOT:-${REPO_ROOT}/runs/paper_nbody_prism_dev_v1}

if [[ ${SLURM_CPUS_PER_TASK:-} != 8 ]]; then
    echo "precision-rank development requires exactly 8 CPUs per task" >&2
    exit 2
fi
if [[ ${SLURM_JOB_NUM_NODES:-} != 1 || ${SLURM_NTASKS:-} != 1 ]]; then
    echo "precision-rank development requires one shared node and one task" >&2
    exit 2
fi
if [[ ${SLURM_JOB_PARTITION:-} != short ]]; then
    echo "precision-rank development requires the short partition" >&2
    exit 2
fi
if [[ ${SLURM_ARRAY_TASK_COUNT:-} != 4 \
    || ${SLURM_ARRAY_TASK_MIN:-} != 0 \
    || ${SLURM_ARRAY_TASK_MAX:-} != 3 \
    || ${SLURM_ARRAY_TASK_STEP:-} != 1 ]]; then
    echo "precision-rank development requires the exact 0-3 array" >&2
    exit 2
fi
if [[ ! ${SLURM_ARRAY_TASK_ID:-} =~ ^[0-3]$ ]]; then
    echo "invalid precision-rank development task index" >&2
    exit 2
fi
if [[ ! -x ${PYTHON_BIN} ]]; then
    echo "repository Python is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi

cd "${REPO_ROOT}"
if [[ -n $(git status --porcelain=v1 --untracked-files=all) ]]; then
    echo "refusing to run precision-rank development in an uncommitted repository" >&2
    exit 2
fi

"${PYTHON_BIN}" -m experiments.paper_nbody_precision_rank_dev \
    --task-index "${SLURM_ARRAY_TASK_ID}" \
    --source-root "${SOURCE_ROOT}" \
    --output-root "${OUTPUT_ROOT}"
