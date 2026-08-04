#!/usr/bin/env bash
#SBATCH --job-name=f01-orbit
#SBATCH --account=lucasbao
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --array=0-2%1
#SBATCH --output=runs/f01-orbit-%A_%a.out
#SBATCH --error=runs/f01-orbit-%A_%a.err

set -Eeuo pipefail
umask 027

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${F01_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
PYTHON_BIN=${F01_PYTHON:-/home/tengcc/.conda/envs/madgp/bin/python}
RUN_ROOT=${F01_RUN_ROOT:-${REPO_ROOT}/runs/f01_orbit_cluster}
SEED_SPEC=${F01_SEEDS:-20260803,20260804,20260805}

IFS=',' read -r -a SEED_VALUES <<< "${SEED_SPEC}"
TASK_INDEX=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_INDEX < 0 || TASK_INDEX >= ${#SEED_VALUES[@]} )); then
    echo "array index ${TASK_INDEX} is outside the ${#SEED_VALUES[@]} configured seeds" >&2
    exit 2
fi
SEED=${SEED_VALUES[${TASK_INDEX}]}
if [[ ! ${SEED} =~ ^[0-9]+$ ]]; then
    echo "invalid non-negative integer seed: ${SEED}" >&2
    exit 2
fi

ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
TASK_DIR=${RUN_ROOT}/job-${ARRAY_JOB_ID}/seed-${SEED}-task-${TASK_INDEX}
mkdir -p "${TASK_DIR}"

exec > >(tee -a "${TASK_DIR}/runner.log") 2>&1

finish() {
    status=$?
    trap - EXIT
    printf '%s\n' "${status}" > "${TASK_DIR}/exit-code.txt"
    date -u +'%Y-%m-%dT%H:%M:%SZ' > "${TASK_DIR}/finished-at.txt"
    exit "${status}"
}
trap finish EXIT

echo "F01 task directory: ${TASK_DIR}"
echo "F01 base seed: ${SEED}"

if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo "refusing to run: this timing harness requires a Slurm allocation" >&2
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
if [[ ! -f ${REPO_ROOT}/experiments/f01_orbit_gp_sim.py ]]; then
    echo "F01 experiment is unavailable under ${REPO_ROOT}" >&2
    exit 2
fi
if [[ ! -f ${REPO_ROOT}/gp/tera/vendor/src/gp_sim_kl/simulation.py ]]; then
    echo "TERA submodule is not populated under ${REPO_ROOT}" >&2
    exit 2
fi

cd "${REPO_ROOT}"

GIT_STATUS=$(git status --porcelain=v1 --untracked-files=all)
if [[ -n ${GIT_STATUS} && ${F01_ALLOW_DIRTY:-0} != 1 ]]; then
    echo "refusing to run an uncommitted tree; set F01_ALLOW_DIRTY=1 only for diagnostics" >&2
    printf '%s\n' "${GIT_STATUS}" >&2
    exit 2
fi

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
SLURM_NODE_JOBS=$(squeue --noheader --nodes "${NODE_NAME}" --states=RUNNING --format='%A' | sort -u)
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
    echo "refusing to record timing without a verified exclusive full-node allocation" >&2
    exit 2
fi
export F01_SLURM_EXCLUSIVE_VERIFIED=1
export F01_SLURM_EXCLUSIVE_MODE=${EXCLUSIVE_MODE}
export F01_REQUIRE_SLURM=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONHASHSEED=${SEED}
export PYTHONUNBUFFERED=1

{
    printf 'captured_at_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    printf 'repo_root=%s\n' "${REPO_ROOT}"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'git_tree=%s\n' "$(git rev-parse 'HEAD^{tree}')"
    printf 'git_describe=%s\n' "$(git describe --always --dirty --tags)"
    printf 'hostname=%s\n' "$(hostname -f)"
    printf 'seed=%s\n' "${SEED}"
    printf 'array_job_id=%s\n' "${ARRAY_JOB_ID}"
    printf 'array_task_id=%s\n' "${TASK_INDEX}"
    printf 'python=%s\n' "${PYTHON_BIN}"
    printf 'exclusive_verification_mode=%s\n' "${EXCLUSIVE_MODE}"
} > "${TASK_DIR}/provenance.env"
printf '%s\n' "${GIT_STATUS}" > "${TASK_DIR}/git-status.txt"
git submodule status --recursive > "${TASK_DIR}/git-submodules.txt"

nvidia-smi \
    --query-gpu=index,uuid,name,driver_version,memory.total \
    --format=csv,noheader > "${TASK_DIR}/gpu.csv"
nvidia-smi topo -m > "${TASK_DIR}/gpu-topology.txt"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader > "${TASK_DIR}/gpu-processes-before.csv" || true

git ls-files -z -- \
    experiments/f01_orbit_gp_sim.py \
    gp/orbit \
    cluster \
    pyproject.toml \
    uv.lock \
    | sort -z \
    | xargs -0 sha256sum \
    > "${TASK_DIR}/source-files.sha256"

: "${F01_N_TRAIN:=40}"
: "${F01_N_EVAL:=20}"
: "${F01_D:=8}"
: "${F01_M_VALUES:=5,10,20}"
: "${F01_TERA_MAX_M:=20}"
: "${F01_REPEATS:=1}"
: "${F01_KERNEL:=matern52}"
: "${F01_SIGMA_F:=0.001}"
: "${F01_TARGET_MEDIAN_CORRELATION:=0.3}"
: "${F01_SAMPLING:=dense}"
: "${F01_DTYPE:=float64}"
: "${F01_CG_TOLERANCE:=1e-8}"
: "${F01_FUNCTION_JITTER:=1e-8}"
: "${F01_REDUCED_JITTER:=1e-8}"
: "${F01_DENSE_SAMPLING_MAX_OBS_DIM:=20000}"
: "${F01_DEROOS_SAMPLING_MAX_N2:=6400}"

EXPERIMENT_ARGS=(
    --n-train "${F01_N_TRAIN}"
    --n-eval "${F01_N_EVAL}"
    --d "${F01_D}"
    --m-values "${F01_M_VALUES}"
    --tera-max-m "${F01_TERA_MAX_M}"
    --repeats "${F01_REPEATS}"
    --seed "${SEED}"
    --kernel "${F01_KERNEL}"
    --sigma-f "${F01_SIGMA_F}"
    --target-median-correlation "${F01_TARGET_MEDIAN_CORRELATION}"
    --sampling "${F01_SAMPLING}"
    --device cuda
    --dtype "${F01_DTYPE}"
    --cg-tolerance "${F01_CG_TOLERANCE}"
    --function-jitter "${F01_FUNCTION_JITTER}"
    --reduced-jitter "${F01_REDUCED_JITTER}"
    --dense-sampling-max-obs-dim "${F01_DENSE_SAMPLING_MAX_OBS_DIM}"
    --deroos-sampling-max-n2 "${F01_DEROOS_SAMPLING_MAX_N2}"
    --out "${TASK_DIR}/result.json"
)
if [[ -n ${F01_CG_MAX_ITERATIONS:-} ]]; then
    EXPERIMENT_ARGS+=(--cg-max-iterations "${F01_CG_MAX_ITERATIONS}")
fi
if [[ ${F01_NO_PRECONDITIONER:-0} == 1 ]]; then
    EXPERIMENT_ARGS+=(--no-preconditioner)
fi

RUN_COMMAND=(
    "${PYTHON_BIN}"
    cluster/run_f01_orbit.py
    --dataset-manifest "${TASK_DIR}/datasets.json"
    --runtime-manifest "${TASK_DIR}/runtime.json"
    "${EXPERIMENT_ARGS[@]}"
)

{
    printf 'command='
    printf '%q ' "${RUN_COMMAND[@]}"
    printf '\n'
} > "${TASK_DIR}/command.txt"

"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
"${RUN_COMMAND[@]}"

sha256sum \
    "${TASK_DIR}/datasets.json" \
    "${TASK_DIR}/runtime.json" \
    "${TASK_DIR}/result.json" \
    "${TASK_DIR}/source-files.sha256" \
    > "${TASK_DIR}/artifacts.sha256"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader > "${TASK_DIR}/gpu-processes-after.csv" || true
