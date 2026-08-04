"""Leakage-free companion generator for confirmatory N-body GP experiments.

The legacy generator in :mod:`data.get_nbody` intentionally remains untouched for
reproduction.  Its orchestration draws a new, unobserved mass vector for every
trajectory and saves no trajectory identifiers.  This module reuses only its
Hamiltonian physics and creates a separately versioned corpus in which one task
replica is one well-defined Hamiltonian:

* masses are drawn once per task replica and persisted;
* complete trajectories are assigned to deterministic, disjoint splits;
* every state is retained (there is no target- or gradient-based filtering);
* raw data plus train-only normalization statistics are persisted; and
* metadata and data files are covered by a SHA-256 manifest.

Run from the repository root, for example::

    uv run python -m data.generate_nbody_confirmatory --output-dir data/nbody_confirmatory
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

SCHEMA_VERSION = "nbody_confirmatory_v1"
_SPLIT_NAMES = ("train", "validation", "test")


class DataIntegrityError(ValueError):
    """Raised when a generated corpus or checksum manifest fails validation."""


@dataclass(frozen=True, slots=True)
class ConfirmatoryConfig:
    """Fully specified generation, split, and validation configuration."""

    n_particles: int = 2
    n_dims: int = 3
    n_trajectories: int = 100
    steps_per_trajectory: int = 100
    dt: float = 0.01
    gravitational_constant: float = 1.0
    softening: float = 0.1
    mass_low: float = 0.5
    mass_high: float = 2.0
    position_scale: float = 2.0
    momentum_scale: float = 0.5
    replica: int = 0
    mass_seed: int = 1729
    trajectory_seed: int = 2718
    split_seed: int = 31415
    validation_seed: int = 1618
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    integrator_rtol: float = 1e-10
    integrator_atol: float = 1e-12
    finite_difference_checks: int = 8
    finite_difference_epsilon: float = 1e-6
    finite_difference_rtol: float = 1e-5
    max_relative_energy_drift: float = 1e-5

    @property
    def state_dim(self) -> int:
        return 2 * self.n_particles * self.n_dims

    @property
    def test_fraction(self) -> float:
        return 1.0 - self.train_fraction - self.validation_fraction

    def validate(self) -> None:
        if self.n_particles < 2:
            raise ValueError("n_particles must be at least 2")
        if self.n_dims < 1:
            raise ValueError("n_dims must be positive")
        if self.n_trajectories < 3:
            raise ValueError("n_trajectories must be at least 3 for three nonempty splits")
        if self.steps_per_trajectory < 2:
            raise ValueError("steps_per_trajectory must be at least 2")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if not 0.0 < self.softening:
            raise ValueError("softening must be positive")
        if not 0.0 < self.mass_low < self.mass_high:
            raise ValueError("mass bounds must satisfy 0 < mass_low < mass_high")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must lie in (0, 1)")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie in (0, 1)")
        if self.test_fraction <= 0.0:
            raise ValueError("train_fraction + validation_fraction must be less than 1")
        if int(np.floor(self.n_trajectories * self.train_fraction)) < 1:
            raise ValueError("train split would contain no trajectories")
        if int(np.floor(self.n_trajectories * self.validation_fraction)) < 1:
            raise ValueError("validation split would contain no trajectories")
        if self.finite_difference_checks < 1:
            raise ValueError("finite_difference_checks must be positive")
        if self.finite_difference_epsilon <= 0.0 or self.finite_difference_rtol <= 0.0:
            raise ValueError("finite-difference epsilon and tolerance must be positive")
        if self.max_relative_energy_drift <= 0.0:
            raise ValueError("max_relative_energy_drift must be positive")


@dataclass(frozen=True, slots=True)
class SplitManifest:
    train_trajectory_ids: np.ndarray
    validation_trajectory_ids: np.ndarray
    test_trajectory_ids: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray

    def trajectory_ids(self, split: str) -> np.ndarray:
        if split not in _SPLIT_NAMES:
            raise KeyError(split)
        return getattr(self, f"{split}_trajectory_ids")

    def indices(self, split: str) -> np.ndarray:
        if split not in _SPLIT_NAMES:
            raise KeyError(split)
        return getattr(self, f"{split}_indices")


@dataclass(frozen=True, slots=True)
class TrainNormalization:
    x_min: np.ndarray
    x_span: np.ndarray
    energy_mean: float
    energy_std: float
    gradient_scale: np.ndarray


@dataclass(frozen=True, slots=True)
class ConfirmatoryDataset:
    config: ConfirmatoryConfig
    X: np.ndarray
    E: np.ndarray
    F: np.ndarray
    masses: np.ndarray
    trajectory_id: np.ndarray
    time_index: np.ndarray
    time_value: np.ndarray
    splits: SplitManifest
    normalization: TrainNormalization


@dataclass(frozen=True, slots=True)
class WrittenBundle:
    dataset_path: Path
    metadata_path: Path
    sha256_manifest_path: Path
    validation: dict[str, float | int]


def _rng(*seed_parts: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(part) for part in seed_parts]))


def _hamiltonian(
    q: np.ndarray,
    p: np.ndarray,
    masses: np.ndarray,
    gravitational_constant: float,
    softening: float,
) -> float:
    """Self-contained Plummer-softened Hamiltonian used by this companion corpus."""
    kinetic = 0.5 * np.sum(p**2 / masses[:, None])
    displacement = q[:, None, :] - q[None, :, :]
    squared_distance = np.sum(displacement**2, axis=-1)
    inverse_distance = (masses[:, None] * masses[None, :]) / np.sqrt(
        squared_distance + softening**2
    )
    np.fill_diagonal(inverse_distance, 0.0)
    potential = -0.5 * gravitational_constant * inverse_distance.sum()
    return float(kinetic + potential)


def _hamiltonian_gradients(
    q: np.ndarray,
    p: np.ndarray,
    masses: np.ndarray,
    gravitational_constant: float,
    softening: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic gradient of :func:`_hamiltonian` with respect to ``(q, p)``."""
    dH_dp = p / masses[:, None]
    displacement = q[:, None, :] - q[None, :, :]
    squared_distance = np.sum(displacement**2, axis=-1)
    coefficient = (masses[:, None] * masses[None, :]) / (squared_distance + softening**2) ** 1.5
    np.fill_diagonal(coefficient, 0.0)
    dH_dq = gravitational_constant * np.sum(
        coefficient[:, :, None] * displacement,
        axis=1,
    )
    return dH_dq, dH_dp


def _hamiltonian_dynamics(
    _time: float,
    state: np.ndarray,
    masses: np.ndarray,
    gravitational_constant: float,
    softening: float,
) -> np.ndarray:
    """Hamilton's equations for the self-contained companion physics."""
    n_particles = masses.size
    n_dims = state.size // (2 * n_particles)
    q = state[: n_particles * n_dims].reshape(n_particles, n_dims)
    p = state[n_particles * n_dims :].reshape(n_particles, n_dims)
    dH_dq, dH_dp = _hamiltonian_gradients(
        q,
        p,
        masses,
        gravitational_constant,
        softening,
    )
    return np.concatenate([dH_dp.reshape(-1), (-dH_dq).reshape(-1)])


def draw_fixed_masses(config: ConfirmatoryConfig) -> np.ndarray:
    """Draw the single mass vector defining one task replica."""
    return _rng(config.mass_seed, config.replica).uniform(
        config.mass_low,
        config.mass_high,
        size=config.n_particles,
    )


def make_group_split(
    trajectory_id: np.ndarray,
    *,
    n_trajectories: int,
    train_fraction: float,
    validation_fraction: float,
    split_seed: int,
    replica: int,
) -> SplitManifest:
    """Create deterministic, group-disjoint sample and trajectory manifests."""
    n_train = int(np.floor(n_trajectories * train_fraction))
    n_validation = int(np.floor(n_trajectories * validation_fraction))
    n_test = n_trajectories - n_train - n_validation
    if min(n_train, n_validation, n_test) < 1:
        raise ValueError("each split must contain at least one trajectory")

    order = _rng(split_seed, replica).permutation(n_trajectories)
    train_ids = np.sort(order[:n_train]).astype(np.int64)
    validation_ids = np.sort(order[n_train : n_train + n_validation]).astype(np.int64)
    test_ids = np.sort(order[n_train + n_validation :]).astype(np.int64)

    def sample_indices(ids: np.ndarray) -> np.ndarray:
        return np.flatnonzero(np.isin(trajectory_id, ids)).astype(np.int64)

    return SplitManifest(
        train_trajectory_ids=train_ids,
        validation_trajectory_ids=validation_ids,
        test_trajectory_ids=test_ids,
        train_indices=sample_indices(train_ids),
        validation_indices=sample_indices(validation_ids),
        test_indices=sample_indices(test_ids),
    )


def compute_train_normalization(
    X: np.ndarray,
    E: np.ndarray,
    F: np.ndarray,
    train_indices: np.ndarray,
) -> TrainNormalization:
    """Compute the legacy-compatible transform using training samples only.

    If ``X_std=(X-x_min)/x_span`` and ``E_std=(E-energy_mean)/energy_std``, then
    ``F_std=F*gradient_scale`` follows from the chain rule.
    """
    if train_indices.size == 0:
        raise ValueError("cannot normalize without training samples")
    X_train = X[train_indices]
    E_train = E[train_indices]
    x_min = X_train.min(axis=0)
    x_span = np.maximum(X_train.max(axis=0) - x_min, 1e-12)
    energy_mean = float(E_train.mean())
    energy_std = max(float(E_train.std(ddof=0)), 1e-12)
    gradient_scale = x_span / energy_std
    if not all(
        np.isfinite(part).all() for part in (x_min, x_span, gradient_scale, F[train_indices])
    ):
        raise DataIntegrityError("nonfinite training normalization input or output")
    return TrainNormalization(
        x_min=x_min,
        x_span=x_span,
        energy_mean=energy_mean,
        energy_std=energy_std,
        gradient_scale=gradient_scale,
    )


def _initial_state(config: ConfirmatoryConfig, masses: np.ndarray, trajectory: int) -> np.ndarray:
    rng = _rng(config.trajectory_seed, config.replica, trajectory)
    q0 = rng.normal(size=(config.n_particles, config.n_dims)) * config.position_scale
    p0 = (
        rng.normal(size=(config.n_particles, config.n_dims))
        * config.momentum_scale
        * masses[:, None]
    )

    # Center position by mass and set the physical total momentum to zero.
    q0 -= np.average(q0, weights=masses, axis=0)
    center_velocity = p0.sum(axis=0) / masses.sum()
    p0 -= masses[:, None] * center_velocity
    return np.concatenate([q0.reshape(-1), p0.reshape(-1)])


def _evaluate_state(
    state: np.ndarray,
    config: ConfirmatoryConfig,
    masses: np.ndarray,
) -> tuple[float, np.ndarray]:
    q = state[: config.n_particles * config.n_dims].reshape(config.n_particles, config.n_dims)
    p = state[config.n_particles * config.n_dims :].reshape(config.n_particles, config.n_dims)
    energy = _hamiltonian(
        q,
        p,
        masses,
        config.gravitational_constant,
        config.softening,
    )
    dH_dq, dH_dp = _hamiltonian_gradients(
        q,
        p,
        masses,
        config.gravitational_constant,
        config.softening,
    )
    gradient = np.concatenate([dH_dq.reshape(-1), dH_dp.reshape(-1)])
    return float(energy), gradient


def generate_dataset(config: ConfirmatoryConfig) -> ConfirmatoryDataset:
    """Generate all states without target-dependent filtering."""
    config.validate()
    masses = draw_fixed_masses(config)
    t_eval = np.arange(config.steps_per_trajectory, dtype=np.float64) * config.dt

    states: list[np.ndarray] = []
    energies: list[np.ndarray] = []
    gradients: list[np.ndarray] = []
    trajectory_ids: list[np.ndarray] = []
    time_indices: list[np.ndarray] = []
    time_values: list[np.ndarray] = []

    for trajectory in range(config.n_trajectories):
        state0 = _initial_state(config, masses, trajectory)
        solution = solve_ivp(
            _hamiltonian_dynamics,
            (float(t_eval[0]), float(t_eval[-1])),
            state0,
            args=(masses, config.gravitational_constant, config.softening),
            t_eval=t_eval,
            method="DOP853",
            rtol=config.integrator_rtol,
            atol=config.integrator_atol,
        )
        if not solution.success:
            raise DataIntegrityError(
                f"trajectory {trajectory} integration failed: {solution.message}"
            )
        if solution.y.shape[1] != config.steps_per_trajectory:
            raise DataIntegrityError(
                f"trajectory {trajectory} returned {solution.y.shape[1]} time points, "
                f"expected {config.steps_per_trajectory}"
            )

        trajectory_states = solution.y.T.astype(np.float64, copy=False)
        evaluated = [_evaluate_state(state, config, masses) for state in trajectory_states]
        states.append(trajectory_states)
        energies.append(np.asarray([item[0] for item in evaluated], dtype=np.float64))
        gradients.append(np.stack([item[1] for item in evaluated]).astype(np.float64))
        trajectory_ids.append(np.full(config.steps_per_trajectory, trajectory, dtype=np.int64))
        time_indices.append(np.arange(config.steps_per_trajectory, dtype=np.int64))
        time_values.append(t_eval.copy())

    X = np.concatenate(states, axis=0)
    E = np.concatenate(energies, axis=0)
    F = np.concatenate(gradients, axis=0)
    trajectory_id = np.concatenate(trajectory_ids, axis=0)
    time_index = np.concatenate(time_indices, axis=0)
    time_value = np.concatenate(time_values, axis=0)
    splits = make_group_split(
        trajectory_id,
        n_trajectories=config.n_trajectories,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        split_seed=config.split_seed,
        replica=config.replica,
    )
    normalization = compute_train_normalization(X, E, F, splits.train_indices)
    dataset = ConfirmatoryDataset(
        config=config,
        X=X,
        E=E,
        F=F,
        masses=masses,
        trajectory_id=trajectory_id,
        time_index=time_index,
        time_value=time_value,
        splits=splits,
        normalization=normalization,
    )
    validate_dataset(dataset)
    return dataset


def finite_difference_gradient(
    state: np.ndarray,
    config: ConfirmatoryConfig,
    masses: np.ndarray,
) -> np.ndarray:
    """Central finite-difference gradient of the Hamiltonian at one state."""
    epsilon = config.finite_difference_epsilon
    result = np.empty_like(state, dtype=np.float64)
    for coordinate in range(state.size):
        plus = state.copy()
        minus = state.copy()
        plus[coordinate] += epsilon
        minus[coordinate] -= epsilon
        plus_energy, _ = _evaluate_state(plus, config, masses)
        minus_energy, _ = _evaluate_state(minus, config, masses)
        result[coordinate] = (plus_energy - minus_energy) / (2.0 * epsilon)
    return result


def _assert_split_integrity(dataset: ConfirmatoryDataset) -> None:
    split_group_sets = {
        split: set(dataset.splits.trajectory_ids(split).tolist()) for split in _SPLIT_NAMES
    }
    for left_index, left in enumerate(_SPLIT_NAMES):
        for right in _SPLIT_NAMES[left_index + 1 :]:
            overlap = split_group_sets[left] & split_group_sets[right]
            if overlap:
                raise DataIntegrityError(
                    f"trajectory leakage between {left} and {right}: {sorted(overlap)}"
                )

    all_indices = np.concatenate([dataset.splits.indices(split) for split in _SPLIT_NAMES])
    expected = np.arange(dataset.X.shape[0], dtype=np.int64)
    if not np.array_equal(np.sort(all_indices), expected):
        raise DataIntegrityError("split sample indices do not form an exact partition")
    if np.unique(all_indices).size != all_indices.size:
        raise DataIntegrityError("a sample appears in more than one split")

    for split in _SPLIT_NAMES:
        indices = dataset.splits.indices(split)
        observed_groups = set(dataset.trajectory_id[indices].tolist())
        if observed_groups != split_group_sets[split]:
            raise DataIntegrityError(f"{split} sample and trajectory manifests disagree")


def validate_dataset(dataset: ConfirmatoryDataset) -> dict[str, float | int]:
    """Run structural, physical, and finite-difference integrity checks."""
    config = dataset.config
    config.validate()
    expected_rows = config.n_trajectories * config.steps_per_trajectory
    expected_shapes = {
        "X": (expected_rows, config.state_dim),
        "E": (expected_rows,),
        "F": (expected_rows, config.state_dim),
        "trajectory_id": (expected_rows,),
        "time_index": (expected_rows,),
        "time_value": (expected_rows,),
        "masses": (config.n_particles,),
    }
    for name, shape in expected_shapes.items():
        value = getattr(dataset, name)
        if value.shape != shape:
            raise DataIntegrityError(f"{name} has shape {value.shape}, expected {shape}")
        if not np.isfinite(value).all():
            raise DataIntegrityError(f"{name} contains nonfinite values")

    if np.any(dataset.masses <= 0.0):
        raise DataIntegrityError("masses must be positive")
    if not np.array_equal(np.unique(dataset.trajectory_id), np.arange(config.n_trajectories)):
        raise DataIntegrityError("trajectory IDs are incomplete or noncontiguous")
    for trajectory in range(config.n_trajectories):
        mask = dataset.trajectory_id == trajectory
        if int(mask.sum()) != config.steps_per_trajectory:
            raise DataIntegrityError(f"trajectory {trajectory} has the wrong sample count")
        if not np.array_equal(
            dataset.time_index[mask], np.arange(config.steps_per_trajectory, dtype=np.int64)
        ):
            raise DataIntegrityError(f"trajectory {trajectory} has invalid time indices")

    _assert_split_integrity(dataset)
    expected_normalization = compute_train_normalization(
        dataset.X,
        dataset.E,
        dataset.F,
        dataset.splits.train_indices,
    )
    for name in ("x_min", "x_span", "gradient_scale"):
        if not np.array_equal(
            getattr(dataset.normalization, name), getattr(expected_normalization, name)
        ):
            raise DataIntegrityError(f"{name} was not computed solely from the train split")
    if dataset.normalization.energy_mean != expected_normalization.energy_mean:
        raise DataIntegrityError("energy_mean was not computed solely from the train split")
    if dataset.normalization.energy_std != expected_normalization.energy_std:
        raise DataIntegrityError("energy_std was not computed solely from the train split")

    energy_drifts: list[float] = []
    for trajectory in range(config.n_trajectories):
        energy = dataset.E[dataset.trajectory_id == trajectory]
        denominator = max(float(np.max(np.abs(energy))), 1.0)
        energy_drifts.append(float(np.ptp(energy)) / denominator)
    max_energy_drift = max(energy_drifts)
    if max_energy_drift > config.max_relative_energy_drift:
        raise DataIntegrityError(
            f"relative energy drift {max_energy_drift:.3e} exceeds "
            f"{config.max_relative_energy_drift:.3e}"
        )

    n_checks = min(config.finite_difference_checks, expected_rows)
    check_indices = _rng(config.validation_seed, config.replica).choice(
        expected_rows,
        size=n_checks,
        replace=False,
    )
    gradient_errors: list[float] = []
    for sample_index in check_indices:
        reference = finite_difference_gradient(dataset.X[int(sample_index)], config, dataset.masses)
        observed = dataset.F[int(sample_index)]
        denominator = max(
            float(np.linalg.norm(reference)),
            float(np.linalg.norm(observed)),
            1e-12,
        )
        gradient_errors.append(float(np.linalg.norm(reference - observed)) / denominator)
    max_gradient_error = max(gradient_errors)
    if max_gradient_error > config.finite_difference_rtol:
        raise DataIntegrityError(
            f"finite-difference relative error {max_gradient_error:.3e} exceeds "
            f"{config.finite_difference_rtol:.3e}"
        )

    return {
        "n_samples": expected_rows,
        "n_trajectories": config.n_trajectories,
        "n_train_samples": int(dataset.splits.train_indices.size),
        "n_validation_samples": int(dataset.splits.validation_indices.size),
        "n_test_samples": int(dataset.splits.test_indices.size),
        "finite_difference_checks": n_checks,
        "max_finite_difference_relative_error": max_gradient_error,
        "max_relative_energy_drift": max_energy_drift,
    }


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256_manifest(paths: list[Path], manifest_path: Path) -> dict[str, str]:
    """Write standard, path-local SHA-256 entries for generated artifacts."""
    entries = {path.name: sha256_file(path) for path in paths}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": "sha256",
        "files": entries,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return entries


def verify_sha256_manifest(manifest_path: str | Path) -> dict[str, str]:
    """Verify every file in a local SHA-256 manifest, returning actual hashes."""
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text())
    if payload.get("algorithm") != "sha256" or not isinstance(payload.get("files"), dict):
        raise DataIntegrityError("invalid SHA-256 manifest schema")
    actual: dict[str, str] = {}
    for name, expected in payload["files"].items():
        if Path(name).name != name:
            raise DataIntegrityError(f"manifest entry must be a local basename: {name!r}")
        path = manifest_path.parent / name
        if not path.is_file():
            raise DataIntegrityError(f"manifest file is missing: {path}")
        actual[name] = sha256_file(path)
        if actual[name] != expected:
            raise DataIntegrityError(f"SHA-256 mismatch for {name}")
    return actual


def _metadata_payload(
    dataset: ConfirmatoryDataset,
    validation: dict[str, float | int],
    dataset_filename: str,
) -> dict:
    normalization = dataset.normalization
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_file": dataset_filename,
        "config": asdict(dataset.config),
        "masses": dataset.masses.tolist(),
        "arrays": {
            "X": {"shape": list(dataset.X.shape), "dtype": str(dataset.X.dtype)},
            "E": {"shape": list(dataset.E.shape), "dtype": str(dataset.E.dtype)},
            "F": {"shape": list(dataset.F.shape), "dtype": str(dataset.F.dtype)},
        },
        "splits": {
            split: {
                "trajectory_ids": dataset.splits.trajectory_ids(split).tolist(),
                "n_samples": int(dataset.splits.indices(split).size),
            }
            for split in _SPLIT_NAMES
        },
        "normalization": {
            "source_split": "train",
            "x_min": normalization.x_min.tolist(),
            "x_span": normalization.x_span.tolist(),
            "energy_mean": normalization.energy_mean,
            "energy_std": normalization.energy_std,
            "gradient_scale": normalization.gradient_scale.tolist(),
        },
        "validation": validation,
    }


def write_bundle(
    dataset: ConfirmatoryDataset,
    output_dir: str | Path,
    *,
    stem: str | None = None,
) -> WrittenBundle:
    """Persist NPZ, readable metadata, and a SHA-256 checksum manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if stem is None:
        stem = (
            f"nbody_fixedmass_n{dataset.config.n_particles}_d{dataset.config.n_dims}"
            f"_replica{dataset.config.replica}"
        )
    if Path(stem).name != stem:
        raise ValueError("stem must be a local filename stem")

    validation = validate_dataset(dataset)
    dataset_path = output_dir / f"{stem}.npz"
    metadata_path = output_dir / f"{stem}.metadata.json"
    manifest_path = output_dir / f"{stem}.sha256.json"
    config_json = json.dumps(asdict(dataset.config), sort_keys=True)
    validation_json = json.dumps(validation, sort_keys=True)
    np.savez_compressed(
        dataset_path,
        schema_version=np.asarray(SCHEMA_VERSION),
        config_json=np.asarray(config_json),
        validation_json=np.asarray(validation_json),
        X=dataset.X,
        E=dataset.E,
        F=dataset.F,
        masses=dataset.masses,
        trajectory_id=dataset.trajectory_id,
        time_index=dataset.time_index,
        time_value=dataset.time_value,
        train_indices=dataset.splits.train_indices,
        validation_indices=dataset.splits.validation_indices,
        test_indices=dataset.splits.test_indices,
        train_trajectory_ids=dataset.splits.train_trajectory_ids,
        validation_trajectory_ids=dataset.splits.validation_trajectory_ids,
        test_trajectory_ids=dataset.splits.test_trajectory_ids,
        x_train_min=dataset.normalization.x_min,
        x_train_span=dataset.normalization.x_span,
        energy_train_mean=np.asarray(dataset.normalization.energy_mean),
        energy_train_std=np.asarray(dataset.normalization.energy_std),
        gradient_scale=dataset.normalization.gradient_scale,
    )
    metadata_path.write_text(
        json.dumps(
            _metadata_payload(dataset, validation, dataset_path.name),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_sha256_manifest([dataset_path, metadata_path], manifest_path)
    verify_sha256_manifest(manifest_path)
    return WrittenBundle(
        dataset_path=dataset_path,
        metadata_path=metadata_path,
        sha256_manifest_path=manifest_path,
        validation=validation,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a fixed-mass, trajectory-disjoint confirmatory N-body corpus."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default=None)
    parser.add_argument("--n-particles", type=int, default=2)
    parser.add_argument("--n-dims", type=int, default=3)
    parser.add_argument("--n-trajectories", type=int, default=100)
    parser.add_argument("--steps-per-trajectory", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--replica", type=int, default=0)
    parser.add_argument("--mass-seed", type=int, default=1729)
    parser.add_argument("--trajectory-seed", type=int, default=2718)
    parser.add_argument("--split-seed", type=int, default=31415)
    parser.add_argument("--validation-seed", type=int, default=1618)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--finite-difference-checks", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = ConfirmatoryConfig(
        n_particles=args.n_particles,
        n_dims=args.n_dims,
        n_trajectories=args.n_trajectories,
        steps_per_trajectory=args.steps_per_trajectory,
        dt=args.dt,
        replica=args.replica,
        mass_seed=args.mass_seed,
        trajectory_seed=args.trajectory_seed,
        split_seed=args.split_seed,
        validation_seed=args.validation_seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        finite_difference_checks=args.finite_difference_checks,
    )
    dataset = generate_dataset(config)
    bundle = write_bundle(dataset, args.output_dir, stem=args.stem)
    print(f"dataset: {bundle.dataset_path}")
    print(f"metadata: {bundle.metadata_path}")
    print(f"sha256 manifest: {bundle.sha256_manifest_path}")
    print(json.dumps(bundle.validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
