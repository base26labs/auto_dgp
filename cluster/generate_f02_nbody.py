"""Generate one provenance-checked F02 N-body corpus per array task.

The Cartesian product is ordered by replica and then particle count.  Existing
artifacts are never overwritten: a task either creates a complete three-file
bundle or verifies that the existing bundle is already valid.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.generate_nbody_confirmatory import ConfirmatoryConfig, generate_dataset, write_bundle
from data.load_nbody_confirmatory import load_confirmatory_bundle


@dataclass(frozen=True, slots=True)
class GenerationTask:
    task_index: int
    replica: int
    n_particles: int
    n_dims: int


def generation_tasks(
    replicas: list[int],
    particle_counts: list[int],
    *,
    n_dims: int,
) -> list[GenerationTask]:
    if not replicas or not particle_counts:
        raise ValueError("replicas and particle_counts must be nonempty")
    if len(set(replicas)) != len(replicas) or len(set(particle_counts)) != len(particle_counts):
        raise ValueError("replicas and particle_counts must not contain duplicates")
    if any(replica < 0 for replica in replicas):
        raise ValueError("replicas must be non-negative")
    if any(count < 2 for count in particle_counts):
        raise ValueError("particle counts must be at least two")
    if n_dims <= 0:
        raise ValueError("n_dims must be positive")
    return [
        GenerationTask(index, replica, particles, n_dims)
        for index, (replica, particles) in enumerate(
            (replica, particles)
            for replica in replicas
            for particles in particle_counts
        )
    ]


def _parse_csv_ints(value: str, label: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{label} must be comma-separated integers") from error
    if not parsed:
        raise argparse.ArgumentTypeError(f"{label} must not be empty")
    return parsed


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_generation_task(
    task: GenerationTask,
    *,
    output_dir: Path,
    n_trajectories: int,
    steps_per_trajectory: int,
    dt: float,
    mass_seed: int,
    trajectory_seed: int,
    split_seed: int,
    validation_seed: int,
    verify_existing: bool,
) -> dict[str, object]:
    config = ConfirmatoryConfig(
        n_particles=task.n_particles,
        n_dims=task.n_dims,
        n_trajectories=n_trajectories,
        steps_per_trajectory=steps_per_trajectory,
        dt=dt,
        replica=task.replica,
        mass_seed=mass_seed,
        trajectory_seed=trajectory_seed,
        split_seed=split_seed,
        validation_seed=validation_seed,
    )
    config.validate()
    stem = f"nbody_fixedmass_n{task.n_particles}_d{task.n_dims}_replica{task.replica}"
    dataset_path = output_dir / f"{stem}.npz"
    metadata_path = output_dir / f"{stem}.metadata.json"
    manifest_path = output_dir / f"{stem}.sha256.json"
    expected = (dataset_path, metadata_path, manifest_path)
    existing = [path for path in expected if path.exists()]
    if existing and len(existing) != len(expected):
        raise FileExistsError(f"incomplete existing bundle for {stem}: {existing}")
    if existing and not verify_existing:
        raise FileExistsError(
            f"refusing to overwrite existing bundle {stem}; pass --verify-existing"
        )

    if not existing:
        dataset = generate_dataset(config)
        write_bundle(dataset, output_dir, stem=stem)

    loaded = load_confirmatory_bundle(dataset_path)
    if loaded.dataset.config != config:
        raise ValueError("loaded bundle config does not match the requested generation task")
    return {
        "schema_version": 1,
        "task": asdict(task),
        "config": asdict(config),
        "artifacts": {
            "dataset": str(dataset_path),
            "metadata": str(metadata_path),
            "sha256_manifest": str(manifest_path),
            "file_sha256": loaded.provenance.file_sha256,
        },
        "validation": loaded.validation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument(
        "--replicas",
        default="0,1,2,101,102,103,104,105,106,107,108,109,110",
    )
    parser.add_argument("--particle-counts", default="2,4,6,8,10")
    parser.add_argument("--n-dims", type=int, default=3)
    parser.add_argument("--n-trajectories", type=int, default=100)
    parser.add_argument("--steps-per-trajectory", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--mass-seed", type=int, default=1729)
    parser.add_argument("--trajectory-seed", type=int, default=2718)
    parser.add_argument("--split-seed", type=int, default=31415)
    parser.add_argument("--validation-seed", type=int, default=1618)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replicas = _parse_csv_ints(args.replicas, "replicas")
    particle_counts = _parse_csv_ints(args.particle_counts, "particle-counts")
    tasks = generation_tasks(replicas, particle_counts, n_dims=args.n_dims)
    task_index = args.task_index
    if task_index is None:
        raw_task_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw_task_index is None:
            raise ValueError("set --task-index or run inside a Slurm array task")
        task_index = int(raw_task_index)
    if task_index < 0 or task_index >= len(tasks):
        raise IndexError(f"task index {task_index} is outside [0, {len(tasks)})")
    result = run_generation_task(
        tasks[task_index],
        output_dir=args.output_dir,
        n_trajectories=args.n_trajectories,
        steps_per_trajectory=args.steps_per_trajectory,
        dt=args.dt,
        mass_seed=args.mass_seed,
        trajectory_seed=args.trajectory_seed,
        split_seed=args.split_seed,
        validation_seed=args.validation_seed,
        verify_existing=args.verify_existing,
    )
    _write_json_atomic(args.result, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
