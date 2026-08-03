# F01 ORBIT cluster harness

This directory runs `experiments/f01_orbit_gp_sim.py` without changing its
scientific logic.  The Slurm job is deliberately exclusive because F01 marks
wall time as descriptive only; timing from a login node, workstation, shared
GPU node, or different array tasks must not be used for a performance claim.
Accuracy and solver diagnostics may be aggregated across independent seeds.

## Cluster preparation

The Boston College cluster can clone the repository over SSH.  Put the working
copy and environment on project storage rather than the smaller home volume:

```bash
cd /projects/lucasbao/tengcc
git clone --recurse-submodules git@github.com:base26labs/auto_dgp.git auto_dgp2
cd auto_dgp2
```

The existing `/home/tengcc/.conda/envs/madgp/bin/python` has Torch 2.4.1 with
working CUDA and all runtime/test dependencies, but it is Python 3.10.  For a
canonical run, install `uv` and run `uv sync --frozen` from a compute node using
the `miniconda/3` module's Python 3.12, then set `F01_PYTHON` to `.venv/bin/python`.
The frozen Linux environment uses Torch 2.12.1+cu126: the cluster's R575 driver
supports CUDA 12.x, whereas current PyPI Torch defaults to CUDA 13 and requires
an R580-or-newer driver.

F01 simulates its dataset in memory, so there is no `.npz` to checksum.  The
cluster launcher intercepts the exact `SimulatedDataset` objects consumed by
F01 and hashes every tensor's dtype, shape, and raw contiguous bytes.  It does
not call or modify the N-body generator.

## Preflight and submission

The helper defaults to a local preflight and will not contact Slurm:

```bash
bash cluster/submit_f01_orbit.sh
bash cluster/submit_f01_orbit.sh --scheduler-test  # sbatch --test-only
bash cluster/submit_f01_orbit.sh --submit          # explicit submission only
```

Defaults mirror F01's small smoke configuration and launch three independent
base seeds, one exclusive node at a time.  Override the preregistered research
configuration explicitly before submission, for example:

```bash
export F01_PYTHON=/projects/lucasbao/tengcc/auto_dgp2/.venv/bin/python
export F01_SEEDS=20260803,20260804,20260805
export F01_N_TRAIN=40
export F01_N_EVAL=20
export F01_D=8
export F01_M_VALUES=5,10,20
export F01_TERA_MAX_M=20
export F01_REPEATS=1
export F01_SAMPLING=dense
bash cluster/submit_f01_orbit.sh --submit
```

For the preregistered official-scale mechanism run, use the upstream Matérn-5/2
`m` sweep dimensions and exact de Roos sampling.  The five array seeds below
reproduce upstream base seed 42's five repeat streams as separate failure
domains (`42 + 1000003 * repeat`):

```bash
export F01_SEEDS=42,1000045,2000048,3000051,4000054
export F01_MAX_PARALLEL=1
export F01_TIME_LIMIT=04:00:00
export F01_N_TRAIN=150
export F01_N_EVAL=100
export F01_D=500
export F01_M_VALUES=3,5,10,20,30,50,100
export F01_TERA_MAX_M=50
export F01_REPEATS=1
export F01_KERNEL=matern52
export F01_SAMPLING=deroos
export F01_DTYPE=float32
export F01_DEROOS_SAMPLING_MAX_N2=50000
bash cluster/submit_f01_orbit.sh --scheduler-test
bash cluster/submit_f01_orbit.sh --submit
```

`TERA_MAX_M=50` is deliberate: `m=50` already requires a 2,500-dimensional
dense reduced system and about `m^6/3 = 5.21e9` leading Cholesky FLOPs per
target, while ORBIT tests whether the same resource can support `m=100` without
claiming wall time as the matching rule.  Same-`m` equivalence is separately
tested in float64 and by the dense-q regression suite.

Each array task writes to
`runs/f01_orbit_cluster/job-<array-job>/seed-<seed>-task-<index>/` and owns its
result, manifests, hashes, and runner log.  The artifacts include:

- `result.json`: F01 rows plus cluster provenance;
- `datasets.json`: hashes of the exact simulated tensors consumed by F01;
- `runtime.json`: Python, Torch/CUDA, selected Slurm environment, and packages;
- `provenance.env`, `slurm-job.txt`, `git-status.txt`, and submodule state;
- `gpu.csv`, topology and before/after compute-process snapshots;
- source, package/runtime, dataset, and final artifact SHA-256 records.

The batch job refuses a dirty tree by default and verifies that Slurm reports
`OverSubscribe=EXCLUSIVE`.  `F01_ALLOW_DIRTY=1` exists only for diagnostics;
results from that mode should not enter the research log.

After every expected task has terminated, aggregate by declaring the exact seed
array rather than globbing only successful directories:

```bash
uv run python cluster/aggregate_f01_orbit.py \
  runs/f01_orbit_cluster/job-<array-job-id> \
  --expected-seeds 42,1000045,2000048,3000051,4000054 \
  --out runs/f01_orbit_cluster/job-<array-job-id>/aggregate.json
```
