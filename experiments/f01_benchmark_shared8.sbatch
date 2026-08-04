#!/bin/bash
#SBATCH --job-name=spark_nd_benchmark
#SBATCH --partition=medium
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --oversubscribe
#SBATCH --constraint=64c
#SBATCH --array=0-53%18

set -euo pipefail
: "${RUN_ROOT:?RUN_ROOT must name a unique cluster directory}"
: "${BENCH_PYTHON:?BENCH_PYTHON must point to the locked Python interpreter}"
: "${SOURCE_COMMIT:?SOURCE_COMMIT must bind the source checkout}"

REPO=${RUN_ROOT}/repo
TASK_ID=${SLURM_ARRAY_TASK_ID}
TASK_TMP=${RUN_ROOT}/tmp/benchmark_${SLURM_ARRAY_JOB_ID}_${TASK_ID}
mkdir -p "${TASK_TMP}" "${RUN_ROOT}/results"
export TMPDIR=${TASK_TMP}
export PYTHONPATH=${REPO}:${REPO}/gp/tera/vendor/src
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONHASHSEED=0

test "$(git -C "${REPO}" rev-parse HEAD)" = "${SOURCE_COMMIT}"
test -z "$(git -C "${REPO}" status --porcelain --untracked-files=no)"
/usr/bin/time -v -o "${RUN_ROOT}/resource_${TASK_ID}.txt" \
    "${BENCH_PYTHON}" "${REPO}/experiments/f01_spark_nd_sweep.py" run-task \
    --task-id "${TASK_ID}" \
    --data-dir "${RUN_ROOT}/data" \
    --result-root "${RUN_ROOT}"
