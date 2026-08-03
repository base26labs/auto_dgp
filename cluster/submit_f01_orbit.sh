#!/usr/bin/env bash
# Preflight or explicitly submit the exclusive-node F01 Slurm array.

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: bash cluster/submit_f01_orbit.sh [--preflight|--scheduler-test|--submit]

The default mode is --preflight and never contacts the scheduler.  --scheduler-test
uses `sbatch --test-only`; only --submit creates run directories and queues jobs.

Configuration is supplied through F01_* environment variables.  Important ones:
  F01_SEEDS             comma-separated non-negative seeds (default: 3 seeds)
  F01_PYTHON            cluster Python executable
  F01_RUN_ROOT          root for task-specific result directories
  F01_MAX_PARALLEL      maximum simultaneous exclusive nodes (default: 1)
  F01_TIME_LIMIT        Slurm time limit (default: 02:00:00)
  F01_N_TRAIN, F01_N_EVAL, F01_D, F01_M_VALUES, F01_TERA_MAX_M
  F01_REPEATS, F01_KERNEL, F01_SAMPLING, F01_DTYPE
EOF
}

MODE=preflight
if (( $# > 1 )); then
    usage >&2
    exit 2
fi
if (( $# == 1 )); then
    case $1 in
        --preflight) MODE=preflight ;;
        --scheduler-test) MODE=scheduler-test ;;
        --submit) MODE=submit ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${F01_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
PYTHON_BIN=${F01_PYTHON:-/home/tengcc/.conda/envs/madgp/bin/python}
RUN_ROOT=${F01_RUN_ROOT:-${REPO_ROOT}/runs/f01_orbit_cluster}
SEED_SPEC=${F01_SEEDS:-20260803,20260804,20260805}
MAX_PARALLEL=${F01_MAX_PARALLEL:-1}
TIME_LIMIT=${F01_TIME_LIMIT:-02:00:00}
SBATCH_SCRIPT=${SCRIPT_DIR}/f01_orbit.sbatch

if [[ ! ${MAX_PARALLEL} =~ ^[1-9][0-9]*$ ]]; then
    echo "F01_MAX_PARALLEL must be a positive integer" >&2
    exit 2
fi
if [[ ! ${TIME_LIMIT} =~ ^([0-9]+-)?[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]]; then
    echo "F01_TIME_LIMIT must use Slurm's [days-]HH:MM:SS form" >&2
    exit 2
fi
IFS=',' read -r -a SEEDS <<< "${SEED_SPEC}"
if (( ${#SEEDS[@]} == 0 )); then
    echo "F01_SEEDS must contain at least one seed" >&2
    exit 2
fi
declare -A SEEN_SEEDS=()
for seed in "${SEEDS[@]}"; do
    if [[ ! ${seed} =~ ^[0-9]+$ ]]; then
        echo "invalid non-negative integer seed: ${seed}" >&2
        exit 2
    fi
    if [[ -n ${SEEN_SEEDS[${seed}]:-} ]]; then
        echo "duplicate seed: ${seed}" >&2
        exit 2
    fi
    SEEN_SEEDS[${seed}]=1
done

for required in \
    "${SBATCH_SCRIPT}" \
    "${REPO_ROOT}/cluster/run_f01_orbit.py" \
    "${REPO_ROOT}/experiments/f01_orbit_gp_sim.py" \
    "${REPO_ROOT}/gp/tera/vendor/src/gp_sim_kl/simulation.py"; do
    if [[ ! -f ${required} ]]; then
        echo "required file is unavailable: ${required}" >&2
        exit 2
    fi
done
if [[ ! -x ${PYTHON_BIN} ]]; then
    echo "Python executable is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ${MODE} != preflight ]] && ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is unavailable; run this helper on a Slurm login node" >&2
    exit 2
fi

cd "${REPO_ROOT}"
GIT_STATUS=$(git status --porcelain=v1 --untracked-files=all)
if [[ -n ${GIT_STATUS} && ${F01_ALLOW_DIRTY:-0} != 1 ]]; then
    echo "preflight failed: commit the worktree before cluster execution" >&2
    printf '%s\n' "${GIT_STATUS}" >&2
    exit 2
fi

PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" -c \
    'import torch; import gp.tera; import experiments.f01_orbit_gp_sim; print(torch.__version__)'

export F01_REPO_ROOT=${REPO_ROOT}
export F01_PYTHON=${PYTHON_BIN}
export F01_RUN_ROOT=${RUN_ROOT}
export F01_SEEDS=${SEED_SPEC}

LAST_INDEX=$((${#SEEDS[@]} - 1))
SBATCH_COMMAND=(
    sbatch
    --account=lucasbao
    --partition=short
    --nodes=1
    --ntasks=1
    --cpus-per-task=16
    --mem=64G
    --gres=gpu:l40s:1
    --exclusive
    --time="${TIME_LIMIT}"
    --array="0-${LAST_INDEX}%${MAX_PARALLEL}"
    --chdir="${REPO_ROOT}"
    --export=ALL
    "${SBATCH_SCRIPT}"
)

printf 'mode=%s\n' "${MODE}"
printf 'repo=%s\n' "${REPO_ROOT}"
printf 'commit=%s\n' "$(git rev-parse HEAD)"
printf 'python=%s\n' "${PYTHON_BIN}"
printf 'run_root=%s\n' "${RUN_ROOT}"
printf 'seeds=%s\n' "${SEED_SPEC}"
printf 'time_limit=%s\n' "${TIME_LIMIT}"
printf 'command='
printf '%q ' "${SBATCH_COMMAND[@]}"
printf '\n'

case ${MODE} in
    preflight)
        echo "preflight passed; no scheduler command was run"
        ;;
    scheduler-test)
        "${SBATCH_COMMAND[@]:0:1}" --test-only "${SBATCH_COMMAND[@]:1}"
        ;;
    submit)
        mkdir -p "${RUN_ROOT}" "${REPO_ROOT}/runs"
        "${SBATCH_COMMAND[@]}"
        ;;
esac
