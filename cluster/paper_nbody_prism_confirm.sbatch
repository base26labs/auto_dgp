#!/usr/bin/env bash
#SBATCH --job-name=prism-confirm
#SBATCH --account=lucasbao
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-11%3
#SBATCH --output=runs/prism-confirm-%A_%a.out
#SBATCH --error=runs/prism-confirm-%A_%a.err

set -Eeuo pipefail
umask 027

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""

REPO_ROOT=${PAPER_NBODY_REPO_ROOT:-/projects/lucasbao/tengcc/auto_dgp2}
OUTPUT_ROOT=${PAPER_NBODY_OUTPUT_ROOT:-${REPO_ROOT}/runs/paper_nbody_prism_confirm_seed43_v1}
PYTHON_BIN=${REPO_ROOT}/.venv/bin/python

if [[ ${SLURM_CPUS_PER_TASK:-} != 8 ]]; then
    echo "PRISM confirmation requires exactly 8 CPUs per task" >&2
    exit 2
fi
if [[ ${SLURM_JOB_NUM_NODES:-} != 1 || ${SLURM_NTASKS:-} != 1 ]]; then
    echo "PRISM confirmation requires one shared node and one task" >&2
    exit 2
fi
if [[ ${SLURM_JOB_PARTITION:-} != short ]]; then
    echo "PRISM confirmation requires the short partition" >&2
    exit 2
fi
if [[ ${SLURM_ARRAY_TASK_COUNT:-} != 12 \
    || ${SLURM_ARRAY_TASK_MIN:-} != 0 \
    || ${SLURM_ARRAY_TASK_MAX:-} != 11 \
    || ${SLURM_ARRAY_TASK_STEP:-} != 1 ]]; then
    echo "PRISM confirmation requires the exact 0-11 array" >&2
    exit 2
fi
if [[ ! ${SLURM_ARRAY_TASK_ID:-} =~ ^([0-9]|1[01])$ ]]; then
    echo "invalid PRISM confirmation task index" >&2
    exit 2
fi
if [[ ! -x ${PYTHON_BIN} ]]; then
    echo "repository Python is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi

cd "${REPO_ROOT}"
if [[ -n $(git status --porcelain=v1 --untracked-files=all) ]]; then
    echo "refusing to benchmark an uncommitted repository" >&2
    exit 2
fi

"${PYTHON_BIN}" -m experiments.paper_nbody_prism_confirm \
    --task-index "${SLURM_ARRAY_TASK_ID}" \
    --output-root "${OUTPUT_ROOT}"
