#!/usr/bin/env bash
#SBATCH --job-name=f02-nbody-data
#SBATCH --account=lucasbao
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-64%10
#SBATCH --output=runs/f02-nbody-data-%A_%a.out
#SBATCH --error=runs/f02-nbody-data-%A_%a.err

set -Eeuo pipefail
umask 027

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${F02_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
PYTHON_BIN=${F02_PYTHON:-${REPO_ROOT}/.venv/bin/python}
DATA_DIR=${F02_DATA_DIR:-${REPO_ROOT}/data/nbody_confirmatory}
RUN_ROOT=${F02_DATA_RUN_ROOT:-${REPO_ROOT}/runs/f02_nbody_data}
REPLICAS=${F02_REPLICAS:-0,1,2,101,102,103,104,105,106,107,108,109,110}
PARTICLE_COUNTS=${F02_PARTICLE_COUNTS:-2,4,6,8,10}
TASK_INDEX=${SLURM_ARRAY_TASK_ID:?F02 data generation requires a Slurm array task}
ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}
TASK_DIR=${RUN_ROOT}/job-${ARRAY_JOB_ID}/task-${TASK_INDEX}
mkdir -p "${TASK_DIR}" "${DATA_DIR}"

exec > >(tee -a "${TASK_DIR}/runner.log") 2>&1

finish() {
    status=$?
    trap - EXIT
    printf '%s\n' "${status}" > "${TASK_DIR}/exit-code.txt"
    date -u +'%Y-%m-%dT%H:%M:%SZ' > "${TASK_DIR}/finished-at.txt"
    exit "${status}"
}
trap finish EXIT

if [[ ! -x ${PYTHON_BIN} ]]; then
    echo "Python executable is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -f ${REPO_ROOT}/cluster/generate_f02_nbody.py ]]; then
    echo "F02 data generator is unavailable under ${REPO_ROOT}" >&2
    exit 2
fi

cd "${REPO_ROOT}"
GIT_STATUS=$(git status --porcelain=v1 --untracked-files=all)
if [[ -n ${GIT_STATUS} ]]; then
    echo "refusing to generate frozen data from an uncommitted tree" >&2
    printf '%s\n' "${GIT_STATUS}" >&2
    exit 2
fi

{
    printf 'captured_at_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    printf 'repo_root=%s\n' "${REPO_ROOT}"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'git_tree=%s\n' "$(git rev-parse 'HEAD^{tree}')"
    printf 'git_describe=%s\n' "$(git describe --always --dirty --tags)"
    printf 'hostname=%s\n' "$(hostname -f)"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID}"
    printf 'slurm_array_job_id=%s\n' "${ARRAY_JOB_ID}"
    printf 'slurm_array_task_id=%s\n' "${TASK_INDEX}"
    printf 'python=%s\n' "${PYTHON_BIN}"
    printf 'replicas=%s\n' "${REPLICAS}"
    printf 'particle_counts=%s\n' "${PARTICLE_COUNTS}"
} > "${TASK_DIR}/provenance.env"
git submodule status --recursive > "${TASK_DIR}/git-submodules.txt"
git ls-files -z -- \
    data/generate_nbody_confirmatory.py \
    data/load_nbody_confirmatory.py \
    cluster/generate_f02_nbody.py \
    docs/F02_NBODY_PROTOCOL.md \
    pyproject.toml \
    uv.lock \
    | sort -z \
    | xargs -0 sha256sum \
    > "${TASK_DIR}/source-files.sha256"

COMMAND=(
    "${PYTHON_BIN}"
    cluster/generate_f02_nbody.py
    --output-dir "${DATA_DIR}"
    --result "${TASK_DIR}/result.json"
    --task-index "${TASK_INDEX}"
    --replicas "${REPLICAS}"
    --particle-counts "${PARTICLE_COUNTS}"
    --n-dims 3
    --n-trajectories 100
    --steps-per-trajectory 100
    --dt 0.01
    --mass-seed 1729
    --trajectory-seed 2718
    --split-seed 31415
    --validation-seed 1618
)
if [[ ${F02_VERIFY_EXISTING:-0} == 1 ]]; then
    COMMAND+=(--verify-existing)
fi
{
    printf 'command='
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
} > "${TASK_DIR}/command.txt"

"${COMMAND[@]}"
sha256sum \
    "${TASK_DIR}/result.json" \
    "${TASK_DIR}/source-files.sha256" \
    > "${TASK_DIR}/artifacts.sha256"
