#!/usr/bin/env bash
# Preflight or explicitly submit the exclusive-node F02 internal selection grid.

set -Eeuo pipefail

readonly EXPECTED_CATALOG_SHA256=2dee429bdaf50cc78cb40ba8b038f7e4731bb07127927c55607b528c1db66942
readonly EXPECTED_TASK_COUNT=135

usage() {
    cat <<'EOF'
Usage: bash cluster/submit_f02_internal_optimizer.sh \
         [--preflight|--scheduler-test|--submit] [--pilot]

The default action is a local-only preflight and never contacts Slurm.
--scheduler-test calls `sbatch --test-only`.  Only --submit queues work.
--pilot selects only array index 0 (replica 0, n=2, 20 updates, seed 11);
it does not itself submit, so a real pilot requires `--submit --pilot`.

Fixed scientific grid:
  replicas       0,1,2
  particles      2,4,6,8,10 (three spatial dimensions)
  update budgets 20,50,100
  seeds          11,29,47

Optional infrastructure environment variables:
  F02_INTERNAL_REPO_ROOT     default /projects/lucasbao/tengcc/auto_dgp2
  F02_INTERNAL_PYTHON        default <repo>/.venv/bin/python
  F02_INTERNAL_DATA_DIR      default /projects/lucasbao/tengcc/datasets/f02_nbody_v1
  F02_INTERNAL_CATALOG       default <repo>/runs/f02_nbody_data/job-2810370/catalog.json
  F02_INTERNAL_RUN_ROOT      default <repo>/runs/f02_internal_optimizer
  F02_INTERNAL_MAX_PARALLEL  full-grid concurrency, default 1
  F02_INTERNAL_TIME_LIMIT    default 08:00:00
EOF
}

MODE=preflight
MODE_SEEN=0
PILOT=0
for argument in "$@"; do
    case ${argument} in
        --preflight|--scheduler-test|--submit)
            if (( MODE_SEEN )); then
                echo "choose exactly one action mode" >&2
                usage >&2
                exit 2
            fi
            MODE=${argument#--}
            MODE_SEEN=1
            ;;
        --pilot)
            if (( PILOT )); then
                echo "--pilot may be supplied only once" >&2
                exit 2
            fi
            PILOT=1
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
done

REPO_ROOT=${F02_INTERNAL_REPO_ROOT:-/projects/lucasbao/tengcc/auto_dgp2}
PYTHON_BIN=${F02_INTERNAL_PYTHON:-${REPO_ROOT}/.venv/bin/python}
DATA_DIR=${F02_INTERNAL_DATA_DIR:-/projects/lucasbao/tengcc/datasets/f02_nbody_v1}
CATALOG_PATH=${F02_INTERNAL_CATALOG:-${REPO_ROOT}/runs/f02_nbody_data/job-2810370/catalog.json}
RUN_ROOT=${F02_INTERNAL_RUN_ROOT:-${REPO_ROOT}/runs/f02_internal_optimizer}
MAX_PARALLEL=${F02_INTERNAL_MAX_PARALLEL:-1}
TIME_LIMIT=${F02_INTERNAL_TIME_LIMIT:-08:00:00}
SBATCH_SCRIPT=${REPO_ROOT}/cluster/f02_internal_optimizer.sbatch
GRID_HELPER=${REPO_ROOT}/cluster/f02_internal_grid.py

if [[ ! ${MAX_PARALLEL} =~ ^[1-9][0-9]*$ ]]; then
    echo "F02_INTERNAL_MAX_PARALLEL must be a positive integer" >&2
    exit 2
fi
if [[ ! ${TIME_LIMIT} =~ ^([0-9]+-)?[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]]; then
    echo "F02_INTERNAL_TIME_LIMIT must use Slurm's [days-]HH:MM:SS form" >&2
    exit 2
fi
if [[ ! -d ${REPO_ROOT} ]]; then
    echo "repository is unavailable: ${REPO_ROOT}" >&2
    exit 2
fi
for required in \
    "${SBATCH_SCRIPT}" \
    "${GRID_HELPER}" \
    "${REPO_ROOT}/cluster/check_python_environment.py" \
    "${REPO_ROOT}/experiments/f02_internal_task.py" \
    "${REPO_ROOT}/gp/tera/vendor/src/gp_sim_kl/simulation.py" \
    "${REPO_ROOT}/pyproject.toml" \
    "${REPO_ROOT}/uv.lock"; do
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
    echo "sbatch is unavailable; use scheduler modes on a Slurm login node" >&2
    exit 2
fi

cd "${REPO_ROOT}"
GIT_STATUS=$(git status --porcelain=v1 --untracked-files=all)
if [[ -n ${GIT_STATUS} ]]; then
    echo "preflight failed: commit the worktree before cluster execution" >&2
    printf '%s\n' "${GIT_STATUS}" >&2
    exit 2
fi
SUBMODULE_STATUS=$(git submodule status --recursive)
if [[ -z ${SUBMODULE_STATUS} || ${SUBMODULE_STATUS} =~ (^|$'\n')[-+U] ]]; then
    echo "preflight failed: submodules must be initialized at committed gitlinks" >&2
    printf '%s\n' "${SUBMODULE_STATUS}" >&2
    exit 2
fi

# Verify the frozen aggregate before importing the runner or inspecting any
# corpus payload.  The batch task repeats this check inside the allocation.
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

for replica in 0 1 2; do
    for n_particles in 2 4 6 8 10; do
        stem=nbody_fixedmass_n${n_particles}_d3_replica${replica}
        for suffix in .npz .metadata.json .sha256.json; do
            if [[ ! -f ${DATA_DIR}/${stem}${suffix} ]]; then
                echo "frozen-corpus artifact is unavailable: ${DATA_DIR}/${stem}${suffix}" >&2
                exit 2
            fi
        done
    done
done

TASK_COUNT=$(PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" "${GRID_HELPER}" --count)
if [[ ${TASK_COUNT} != "${EXPECTED_TASK_COUNT}" ]]; then
    echo "task-map count mismatch: expected ${EXPECTED_TASK_COUNT}, got ${TASK_COUNT}" >&2
    exit 2
fi
mapfile -t PILOT_FIELDS < <(
    PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" "${GRID_HELPER}" \
        --task-index 0 --format lines
)
EXPECTED_PILOT=(0 0 2 3 20 11 nbody_fixedmass_n2_d3_replica0)
if (( ${#PILOT_FIELDS[@]} != ${#EXPECTED_PILOT[@]} )); then
    echo "pilot task map has the wrong number of fields" >&2
    exit 2
fi
for index in "${!EXPECTED_PILOT[@]}"; do
    if [[ ${PILOT_FIELDS[${index}]} != "${EXPECTED_PILOT[${index}]}" ]]; then
        echo "pilot task-map invariant failed at field ${index}" >&2
        exit 2
    fi
done

# Import-only dependency validation; no model is fit and no corpus is loaded.
PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" -c \
    'import torch; import experiments.f02_internal_task; print(torch.__version__)'
PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" cluster/check_python_environment.py >/dev/null

export F02_INTERNAL_REPO_ROOT=${REPO_ROOT}
export F02_INTERNAL_PYTHON=${PYTHON_BIN}
export F02_INTERNAL_DATA_DIR=${DATA_DIR}
export F02_INTERNAL_CATALOG=${CATALOG_PATH}
export F02_INTERNAL_RUN_ROOT=${RUN_ROOT}

if (( PILOT )); then
    ARRAY_SPEC=0%1
else
    ARRAY_SPEC=0-134%${MAX_PARALLEL}
fi

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
    --array="${ARRAY_SPEC}"
    --chdir="${REPO_ROOT}"
    --export=ALL
    "${SBATCH_SCRIPT}"
)

printf 'mode=%s\n' "${MODE}"
printf 'pilot=%s\n' "${PILOT}"
printf 'repo=%s\n' "${REPO_ROOT}"
printf 'commit=%s\n' "$(git rev-parse HEAD)"
printf 'tree=%s\n' "$(git rev-parse 'HEAD^{tree}')"
printf 'python=%s\n' "${PYTHON_BIN}"
printf 'data_dir=%s\n' "${DATA_DIR}"
printf 'catalog=%s\n' "${CATALOG_PATH}"
printf 'catalog_sha256=%s\n' "${ACTUAL_CATALOG_SHA256}"
printf 'run_root=%s\n' "${RUN_ROOT}"
printf 'task_count=%s\n' "${TASK_COUNT}"
printf 'array=%s\n' "${ARRAY_SPEC}"
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
