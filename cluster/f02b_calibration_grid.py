"""Immutable task maps for the development-only F02b numerical calibration.

The scientific matrix is deliberately not configurable from the command line.
Fit tasks vary replica, particle count, and optimizer seed at one fixed
calibration budget.  Probe tasks reference those fits and vary only the
predeclared prediction neighbourhood or repetition identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

CALIBRATION_ID = "F02B_NUMERICAL_CALIBRATION_v2"
DEVELOPMENT_REPLICAS = (0, 1, 2)
PARTICLE_COUNTS = (2, 4, 6, 8, 10)
SEEDS = (11, 29, 47)
RESOURCE_SWEEP_M = (20, 75, 100, 150, 200)
N_DIMS = 3
TRAIN_STEPS = 20
TRAINING_M = 20
KERNEL = "rbf"
REFERENCE_M = 50


@dataclass(frozen=True, slots=True)
class F02bCalibrationFitTask:
    """One fixed-budget TERA fit used only for numerical calibration."""

    task_index: int
    replica: int
    n_particles: int
    n_dims: int
    seed: int
    train_steps: int
    training_m: int
    kernel: str

    @property
    def dimension(self) -> int:
        """Return the phase-space state dimension."""

        return 2 * self.n_particles * self.n_dims

    @property
    def dataset_stem(self) -> str:
        """Return the exact frozen development-corpus stem."""

        return f"nbody_fixedmass_n{self.n_particles}_d{self.n_dims}_replica{self.replica}"

    def as_record(self) -> dict[str, int | str]:
        """Return the canonical JSON-safe fit-task record."""

        return {
            "calibration_id": CALIBRATION_ID,
            **asdict(self),
            "D": self.dimension,
            "dataset_stem": self.dataset_stem,
        }


def _fit_tasks() -> tuple[F02bCalibrationFitTask, ...]:
    combinations = (
        (replica, n_particles, seed)
        for replica in DEVELOPMENT_REPLICAS
        for n_particles in PARTICLE_COUNTS
        for seed in SEEDS
    )
    return tuple(
        F02bCalibrationFitTask(
            task_index=index,
            replica=replica,
            n_particles=n_particles,
            n_dims=N_DIMS,
            seed=seed,
            train_steps=TRAIN_STEPS,
            training_m=TRAINING_M,
            kernel=KERNEL,
        )
        for index, (replica, n_particles, seed) in enumerate(combinations)
    )


FIT_TASKS = _fit_tasks()
FIT_TASK_COUNT = len(FIT_TASKS)


def fit_task_for_index(task_index: int) -> F02bCalibrationFitTask:
    """Resolve a fit-task index without permitting negative indexing."""

    if task_index < 0 or task_index >= FIT_TASK_COUNT:
        raise IndexError(f"fit task index {task_index} is outside [0, {FIT_TASK_COUNT})")
    return FIT_TASKS[task_index]


@dataclass(frozen=True, slots=True)
class F02bCalibrationProbeTask:
    """One immutable numerical probe against a previously frozen fit."""

    task_index: int
    fit_task_index: int
    m: int
    repeat_id: int
    role: str

    @property
    def fit_task(self) -> F02bCalibrationFitTask:
        """Return the fit identity referenced by this probe."""

        return fit_task_for_index(self.fit_task_index)

    @property
    def geometry_m_values(self) -> tuple[int, ...]:
        """Return the primary and predeclared geometry-only neighbourhoods."""

        if self.m == REFERENCE_M and self.repeat_id == 0:
            dimension = self.fit_task.dimension
            return (self.m, dimension - 7, dimension - 6, dimension - 5)
        return (self.m,)

    @property
    def support_target_count(self) -> int:
        """Return the fixed number of geometry-stratified support solves."""

        return 3 if self.m == REFERENCE_M else 2

    @property
    def stress_m(self) -> int | None:
        """Return the first redundant-direction stress neighbourhood, if registered."""

        if self.m == REFERENCE_M and self.repeat_id == 0 and self.fit_task.seed == SEEDS[0]:
            return self.fit_task.dimension - 5
        return None

    @property
    def stress_support_target_count(self) -> int:
        """Return the registered target count for the optional stress probe."""

        return 1 if self.stress_m is not None else 0

    def as_record(self) -> dict[str, Any]:
        """Return the canonical JSON-safe probe-task record."""

        fit = self.fit_task
        return {
            "calibration_id": CALIBRATION_ID,
            **asdict(self),
            "replica": fit.replica,
            "n_particles": fit.n_particles,
            "n_dims": fit.n_dims,
            "D": fit.dimension,
            "seed": fit.seed,
            "train_steps": fit.train_steps,
            "training_m": fit.training_m,
            "kernel": fit.kernel,
            "dataset_stem": fit.dataset_stem,
            "geometry_m_values": list(self.geometry_m_values),
            "support_target_count": self.support_target_count,
            "stress_m": self.stress_m,
            "stress_support_target_count": self.stress_support_target_count,
        }


def _probe_tasks() -> tuple[F02bCalibrationProbeTask, ...]:
    tasks: list[F02bCalibrationProbeTask] = []
    for fit in FIT_TASKS:
        tasks.append(
            F02bCalibrationProbeTask(
                task_index=len(tasks),
                fit_task_index=fit.task_index,
                m=REFERENCE_M,
                repeat_id=0,
                role="reference",
            )
        )
    for fit in FIT_TASKS:
        if fit.seed != SEEDS[0]:
            continue
        for m in RESOURCE_SWEEP_M:
            tasks.append(
                F02bCalibrationProbeTask(
                    task_index=len(tasks),
                    fit_task_index=fit.task_index,
                    m=m,
                    repeat_id=0,
                    role="resource_sweep",
                )
            )
    for repeat_id in (1, 2):
        tasks.append(
            F02bCalibrationProbeTask(
                task_index=len(tasks),
                fit_task_index=0,
                m=REFERENCE_M,
                repeat_id=repeat_id,
                role="reproducibility",
            )
        )
    return tuple(tasks)


PROBE_TASKS = _probe_tasks()
PROBE_TASK_COUNT = len(PROBE_TASKS)
REFERENCE_PROBE_COUNT = FIT_TASK_COUNT
RESOURCE_SWEEP_PROBE_COUNT = (
    len(DEVELOPMENT_REPLICAS) * len(PARTICLE_COUNTS) * len(RESOURCE_SWEEP_M)
)
REPRODUCIBILITY_PROBE_COUNT = 2


def probe_task_for_index(task_index: int) -> F02bCalibrationProbeTask:
    """Resolve a probe-task index without permitting negative indexing."""

    if task_index < 0 or task_index >= PROBE_TASK_COUNT:
        raise IndexError(f"probe task index {task_index} is outside [0, {PROBE_TASK_COUNT})")
    return PROBE_TASKS[task_index]


def canonical_fit_task_records() -> tuple[dict[str, int | str], ...]:
    """Return the complete canonical fit-task matrix."""

    return tuple(task.as_record() for task in FIT_TASKS)


def canonical_probe_task_records() -> tuple[dict[str, Any], ...]:
    """Return the complete canonical probe-task matrix."""

    return tuple(task.as_record() for task in PROBE_TASKS)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


# These literals make accidental changes to task order or record semantics fail
# at import, before a scheduler can resolve an array index.
FIT_TASK_MATRIX_SHA256 = "7272f823a2bfc0f52cbfc2e27ae3a56b2f668e3ca2abff054de9209cd2fa5a39"
PROBE_TASK_MATRIX_SHA256 = "98a44d167f6a34d3e94dcffd026d030e56bcf30f77a0bd43810f16b311e54eca"
CALIBRATION_MATRIX_SHA256 = "0ead06b0e2f6de24c49f4bf6f999f90690ff1fb82be3585cc212bdd11fd411f4"


def _validate_matrix() -> None:
    if FIT_TASK_COUNT != 45:
        raise RuntimeError(f"F02b fit matrix must contain 45 tasks, got {FIT_TASK_COUNT}")
    if PROBE_TASK_COUNT != 122:
        raise RuntimeError(f"F02b probe matrix must contain 122 tasks, got {PROBE_TASK_COUNT}")
    expected_probe_count = (
        REFERENCE_PROBE_COUNT + RESOURCE_SWEEP_PROBE_COUNT + REPRODUCIBILITY_PROBE_COUNT
    )
    if PROBE_TASK_COUNT != expected_probe_count:
        raise RuntimeError("F02b probe role counts do not sum to the full matrix")
    if [task.task_index for task in FIT_TASKS] != list(range(FIT_TASK_COUNT)):
        raise RuntimeError("F02b fit task indices are not contiguous")
    if [task.task_index for task in PROBE_TASKS] != list(range(PROBE_TASK_COUNT)):
        raise RuntimeError("F02b probe task indices are not contiguous")
    fit_identities = {
        (
            task.replica,
            task.n_particles,
            task.n_dims,
            task.seed,
            task.train_steps,
            task.training_m,
            task.kernel,
        )
        for task in FIT_TASKS
    }
    if len(fit_identities) != FIT_TASK_COUNT:
        raise RuntimeError("F02b fit scientific identities are not unique")
    probe_identities = {
        (task.fit_task_index, task.m, task.repeat_id, task.role) for task in PROBE_TASKS
    }
    if len(probe_identities) != PROBE_TASK_COUNT:
        raise RuntimeError("F02b probe task identities are not unique")
    actual_fit_hash = _sha256(canonical_fit_task_records())
    actual_probe_hash = _sha256(canonical_probe_task_records())
    actual_combined_hash = _sha256(
        {
            "fit_tasks": canonical_fit_task_records(),
            "probe_tasks": canonical_probe_task_records(),
        }
    )
    if actual_fit_hash != FIT_TASK_MATRIX_SHA256:
        raise RuntimeError("F02b fit task matrix SHA-256 does not match its frozen literal")
    if actual_probe_hash != PROBE_TASK_MATRIX_SHA256:
        raise RuntimeError("F02b probe task matrix SHA-256 does not match its frozen literal")
    if actual_combined_hash != CALIBRATION_MATRIX_SHA256:
        raise RuntimeError("F02b combined matrix SHA-256 does not match its frozen literal")


_validate_matrix()


def _fit_lines(task: F02bCalibrationFitTask) -> tuple[int | str, ...]:
    return (
        task.task_index,
        task.replica,
        task.n_particles,
        task.n_dims,
        task.dimension,
        task.seed,
        task.train_steps,
        task.training_m,
        task.kernel,
        task.dataset_stem,
        CALIBRATION_ID,
    )


def _probe_lines(task: F02bCalibrationProbeTask) -> tuple[int | str, ...]:
    fit = task.fit_task
    return (
        task.task_index,
        task.fit_task_index,
        fit.replica,
        fit.n_particles,
        fit.n_dims,
        fit.dimension,
        fit.seed,
        task.m,
        task.repeat_id,
        task.role,
        fit.dataset_stem,
        ",".join(str(value) for value in task.geometry_m_values),
        task.support_target_count,
        "none" if task.stress_m is None else task.stress_m,
        task.stress_support_target_count,
        CALIBRATION_ID,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", choices=("fit", "probe"), required=True)
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
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.count:
        if args.format != "json":
            parser.error("--format lines requires --task-index")
        print(FIT_TASK_COUNT if args.matrix == "fit" else PROBE_TASK_COUNT)
        return 0

    if args.matrix == "fit":
        task: F02bCalibrationFitTask | F02bCalibrationProbeTask = fit_task_for_index(
            args.task_index
        )
        lines = _fit_lines(task)
    else:
        task = probe_task_for_index(args.task_index)
        lines = _probe_lines(task)
    if args.format == "json":
        print(json.dumps(task.as_record(), sort_keys=True, separators=(",", ":")))
    else:
        for value in lines:
            print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
