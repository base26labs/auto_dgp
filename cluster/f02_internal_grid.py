"""Deterministic task map for the F02 internal optimizer-selection array.

The grid is deliberately not configurable from the command line.  Changing it
is a protocol change, not a scheduler convenience.  Array order is replica,
particle count, optimizer-update budget, then seed, with the seed varying
fastest.  Consequently task zero is the preregistered pilot.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

DEVELOPMENT_REPLICAS = (0, 1, 2)
PARTICLE_COUNTS = (2, 4, 6, 8, 10)
UPDATE_BUDGETS = (20, 50, 100)
SEEDS = (11, 29, 47)
N_DIMS = 3


@dataclass(frozen=True, slots=True)
class InternalOptimizerTask:
    """One immutable F02 internal optimizer-selection task."""

    task_index: int
    replica: int
    n_particles: int
    n_dims: int
    train_steps: int
    seed: int

    @property
    def dataset_stem(self) -> str:
        """Return the exact frozen-corpus stem generated for this task."""

        return f"nbody_fixedmass_n{self.n_particles}_d{self.n_dims}_replica{self.replica}"

    def as_record(self) -> dict[str, int | str]:
        """Return a JSON-safe record including the derived dataset stem."""

        return {**asdict(self), "dataset_stem": self.dataset_stem}


def optimizer_selection_tasks() -> tuple[InternalOptimizerTask, ...]:
    """Return the exact 135-task development grid in array-index order."""

    combinations = (
        (replica, n_particles, train_steps, seed)
        for replica in DEVELOPMENT_REPLICAS
        for n_particles in PARTICLE_COUNTS
        for train_steps in UPDATE_BUDGETS
        for seed in SEEDS
    )
    return tuple(
        InternalOptimizerTask(
            task_index=index,
            replica=replica,
            n_particles=n_particles,
            n_dims=N_DIMS,
            train_steps=train_steps,
            seed=seed,
        )
        for index, (replica, n_particles, train_steps, seed) in enumerate(combinations)
    )


OPTIMIZER_SELECTION_TASKS = optimizer_selection_tasks()
TASK_COUNT = len(OPTIMIZER_SELECTION_TASKS)
PILOT_TASK_INDEX = 0

if TASK_COUNT != 135:  # pragma: no cover - import-time protocol invariant
    raise RuntimeError(f"F02 optimizer-selection grid must contain 135 tasks, got {TASK_COUNT}")


def task_for_index(task_index: int) -> InternalOptimizerTask:
    """Resolve one array index, rejecting Python's negative-index semantics."""

    if task_index < 0 or task_index >= TASK_COUNT:
        raise IndexError(f"task index {task_index} is outside [0, {TASK_COUNT})")
    return OPTIMIZER_SELECTION_TASKS[task_index]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task-index", type=int)
    selection.add_argument("--count", action="store_true")
    parser.add_argument(
        "--format",
        choices=("json", "lines"),
        default="json",
        help="lines emits fixed shell-safe scalar fields and requires --task-index",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.count:
        if args.format != "json":
            build_parser().error("--format lines requires --task-index")
        print(TASK_COUNT)
        return 0

    task = task_for_index(args.task_index)
    if args.format == "json":
        print(json.dumps(task.as_record(), sort_keys=True, separators=(",", ":")))
    else:
        for value in (
            task.task_index,
            task.replica,
            task.n_particles,
            task.n_dims,
            task.train_steps,
            task.seed,
            task.dataset_stem,
        ):
            print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
