#!/usr/bin/env bash
#SBATCH --job-name=prism-data43
#SBATCH --account=lucasbao
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%2
#SBATCH --output=runs/prism-data43-%A_%a.out
#SBATCH --error=runs/prism-data43-%A_%a.err

set -Eeuo pipefail
umask 027

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""

REPO_ROOT=${PAPER_NBODY_REPO_ROOT:-/projects/lucasbao/tengcc/auto_dgp2}
PYTHON_BIN=${REPO_ROOT}/.venv/bin/python
PARTICLES=(4 6 8 10)

if [[ ${SLURM_CPUS_PER_TASK:-} != 8 ]]; then
    echo "confirmation data generation requires exactly 8 CPUs per task" >&2
    exit 2
fi
if [[ ${SLURM_JOB_NUM_NODES:-} != 1 || ${SLURM_NTASKS:-} != 1 ]]; then
    echo "confirmation data generation requires one shared node and one task" >&2
    exit 2
fi
if [[ ${SLURM_JOB_PARTITION:-} != short ]]; then
    echo "confirmation data generation requires the short partition" >&2
    exit 2
fi
if [[ ${SLURM_ARRAY_TASK_COUNT:-} != 4 \
    || ${SLURM_ARRAY_TASK_MIN:-} != 0 \
    || ${SLURM_ARRAY_TASK_MAX:-} != 3 \
    || ${SLURM_ARRAY_TASK_STEP:-} != 1 ]]; then
    echo "confirmation data generation requires the exact 0-3 array" >&2
    exit 2
fi
if [[ ! ${SLURM_ARRAY_TASK_ID:-} =~ ^[0-3]$ ]]; then
    echo "invalid confirmation data task index" >&2
    exit 2
fi
if [[ ! -x ${PYTHON_BIN} ]]; then
    echo "repository Python is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ -n $(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all) ]]; then
    echo "refusing to generate confirmation data from an uncommitted repository" >&2
    exit 2
fi

N_PARTICLES=${PARTICLES[SLURM_ARRAY_TASK_ID]}
TARGET=${REPO_ROOT}/data/nbody_confirmatory_seed43/nbody_n${N_PARTICLES}_d3.npz
if [[ -e ${TARGET} ]]; then
    echo "refusing to overwrite independent dataset: ${TARGET}" >&2
    exit 2
fi

cd "${REPO_ROOT}/data"
"${PYTHON_BIN}" get_nbody.py \
    --n_particles "${N_PARTICLES}" \
    --n_dims 3 \
    --n_samples 10000 \
    --n_trajectories 100 \
    --steps_per_trajectory 200 \
    --dt 0.01 \
    --G 1.0 \
    --softening 0.1 \
    --percentile_filter 95 \
    --seed 43 \
    --output-file "${TARGET}"
