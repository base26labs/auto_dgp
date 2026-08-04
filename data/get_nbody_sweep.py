"""Generate fixed-system, trajectory-disjoint N-body sweep datasets."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from data.get_nbody import hamiltonian, hamiltonian_dynamics, hamiltonian_gradients

N_TRAJECTORIES = 40
ROWS_PER_TRAJECTORY = 50
STEPS_PER_TRAJECTORY = 100
DT = 0.01
GRAVITATIONAL_CONSTANT = 1.0
SOFTENING = 0.1
MASS_RANGE = (0.5, 2.0)
POSITION_SCALE = 2.0
MOMENTUM_SCALE = 0.5

ARCHIVE_KEYS = frozenset(
    {
        "X",
        "E",
        "F",
        "trajectory_id",
        "masses",
        "n_particles",
        "n_dims",
        "seed",
        "n_trajectories",
        "rows_per_trajectory",
        "steps_per_trajectory",
        "dt",
        "G",
        "softening",
    }
)


def dataset_filename(n_particles: int, n_dims: int, seed: int) -> str:
    """Return the canonical filename for one independent simulator system."""

    return f"nbody_sweep_n{n_particles}_d{n_dims}_s{seed}.npz"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_nbody_sweep_dataset(
    *,
    n_particles: int,
    n_dims: int,
    seed: int,
    n_trajectories: int = N_TRAJECTORIES,
    rows_per_trajectory: int = ROWS_PER_TRAJECTORY,
    steps_per_trajectory: int = STEPS_PER_TRAJECTORY,
    dt: float = DT,
    gravitational_constant: float = GRAVITATIONAL_CONSTANT,
    softening: float = SOFTENING,
) -> dict[str, np.ndarray]:
    """Generate one fixed-mass system without target-based filtering.

    A dataset uses one mass vector for every trajectory. Rows are sampled at fixed time indices;
    the benchmark assigns complete trajectories to train or test, so conserved-energy neighbors
    cannot cross the evaluation boundary.
    """

    if n_particles < 2 or n_dims < 1 or seed < 0:
        raise ValueError("n_particles >= 2, n_dims >= 1, and a nonnegative seed are required")
    if n_trajectories < 2 or not 1 <= rows_per_trajectory <= steps_per_trajectory:
        raise ValueError("trajectory and row counts are inconsistent")
    if dt <= 0 or gravitational_constant <= 0 or softening <= 0:
        raise ValueError("physical scale parameters must be positive")

    rng = np.random.default_rng(seed)
    masses = rng.uniform(*MASS_RANGE, size=n_particles)
    sample_indices = np.linspace(
        0,
        steps_per_trajectory - 1,
        rows_per_trajectory,
        dtype=np.int64,
    )
    if len(np.unique(sample_indices)) != rows_per_trajectory:
        raise RuntimeError("fixed time-index selection contains duplicates")

    states: list[np.ndarray] = []
    values: list[float] = []
    gradients: list[np.ndarray] = []
    trajectory_ids: list[int] = []
    for trajectory_id in range(n_trajectories):
        q0 = rng.normal(size=(n_particles, n_dims)) * POSITION_SCALE
        p0 = rng.normal(size=(n_particles, n_dims)) * MOMENTUM_SCALE * masses[:, None]
        q0 -= np.average(q0, weights=masses, axis=0)
        p0 -= np.average(p0, weights=masses, axis=0)
        state0 = np.concatenate((q0.ravel(), p0.ravel()))
        duration = (steps_per_trajectory - 1) * dt
        solution = solve_ivp(
            hamiltonian_dynamics,
            (0.0, duration),
            state0,
            args=(masses, gravitational_constant, softening),
            t_eval=np.linspace(0.0, duration, steps_per_trajectory),
            method="DOP853",
            rtol=1e-10,
            atol=1e-12,
        )
        if not solution.success or solution.y.shape[1] != steps_per_trajectory:
            raise RuntimeError(f"trajectory {trajectory_id} integration failed")

        for index in sample_indices:
            state = solution.y[:, index]
            width = n_particles * n_dims
            q = state[:width].reshape(n_particles, n_dims)
            p = state[width:].reshape(n_particles, n_dims)
            dH_dq, dH_dp = hamiltonian_gradients(
                q,
                p,
                masses,
                gravitational_constant,
                softening,
            )
            states.append(state)
            values.append(hamiltonian(q, p, masses, gravitational_constant, softening))
            gradients.append(np.concatenate((dH_dq.ravel(), dH_dp.ravel())))
            trajectory_ids.append(trajectory_id)

    arrays = {
        "X": np.asarray(states, dtype=np.float64),
        "E": np.asarray(values, dtype=np.float64),
        "F": np.asarray(gradients, dtype=np.float64),
        "trajectory_id": np.asarray(trajectory_ids, dtype=np.int64),
        "masses": np.asarray(masses, dtype=np.float64),
        "n_particles": np.asarray(n_particles, dtype=np.int64),
        "n_dims": np.asarray(n_dims, dtype=np.int64),
        "seed": np.asarray(seed, dtype=np.int64),
        "n_trajectories": np.asarray(n_trajectories, dtype=np.int64),
        "rows_per_trajectory": np.asarray(rows_per_trajectory, dtype=np.int64),
        "steps_per_trajectory": np.asarray(steps_per_trajectory, dtype=np.int64),
        "dt": np.asarray(dt, dtype=np.float64),
        "G": np.asarray(gravitational_constant, dtype=np.float64),
        "softening": np.asarray(softening, dtype=np.float64),
    }
    _validate_arrays(arrays)
    return arrays


def _validate_arrays(arrays: dict[str, np.ndarray]) -> None:
    if set(arrays) != set(ARCHIVE_KEYS):
        raise ValueError("dataset archive schema drifted")
    n_particles = int(arrays["n_particles"])
    n_dims = int(arrays["n_dims"])
    n_trajectories = int(arrays["n_trajectories"])
    rows_per_trajectory = int(arrays["rows_per_trajectory"])
    row_count = n_trajectories * rows_per_trajectory
    dimension = 2 * n_particles * n_dims
    if arrays["X"].shape != (row_count, dimension):
        raise ValueError("state array has the wrong shape")
    if arrays["E"].shape != (row_count,) or arrays["F"].shape != (row_count, dimension):
        raise ValueError("label arrays have the wrong shape")
    expected_ids = np.repeat(np.arange(n_trajectories), rows_per_trajectory)
    if not np.array_equal(arrays["trajectory_id"], expected_ids):
        raise ValueError("trajectory IDs are not canonical")
    if not all(np.isfinite(arrays[key]).all() for key in ("X", "E", "F", "masses")):
        raise ValueError("dataset contains non-finite values")


def save_dataset(output_dir: str | Path, arrays: dict[str, np.ndarray]) -> Path:
    """Write one canonical archive without overwriting an existing dataset."""

    _validate_arrays(arrays)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / dataset_filename(
        int(arrays["n_particles"]),
        int(arrays["n_dims"]),
        int(arrays["seed"]),
    )
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("xb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-particles", type=int, required=True)
    parser.add_argument("--n-dims", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    arrays = generate_nbody_sweep_dataset(
        n_particles=args.n_particles,
        n_dims=args.n_dims,
        seed=args.seed,
    )
    path = save_dataset(args.output_dir, arrays)
    print(f"{path}\t{file_sha256(path)}")


if __name__ == "__main__":
    main()
