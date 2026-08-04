"""Frozen, label-independent temporal design for F02 N-body prediction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from data.generate_nbody_confirmatory import DataIntegrityError
from data.load_nbody_confirmatory import (
    PreparedConfirmatoryDataset,
    PreparedConfirmatorySplit,
)

TRAIN_TIME_INDICES = (
    0,
    4,
    8,
    12,
    16,
    21,
    25,
    29,
    33,
    37,
    41,
    45,
    50,
    54,
    58,
    62,
    66,
    70,
    74,
    78,
    82,
    87,
    91,
    95,
    99,
)
EVALUATION_TIME_INDICES = (0, 25, 50, 74, 99)
OPTIMIZER_SELECTION_TIME_INDICES = (50,)


def _frozen_selection(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.array(value[mask], copy=True)
    result.flags.writeable = False
    return result


def select_time_indices(
    split: PreparedConfirmatorySplit,
    time_indices: Sequence[int],
) -> PreparedConfirmatorySplit:
    """Select the same declared time indices from every trajectory.

    The selection mask depends only on ``time_index``.  Targets and gradients
    are copied after the mask is fixed and never participate in row selection.
    """

    requested = np.asarray(tuple(time_indices), dtype=np.int64)
    if requested.ndim != 1 or requested.size == 0:
        raise ValueError("time_indices must be a nonempty one-dimensional sequence")
    if np.any(requested < 0) or np.any(np.diff(requested) <= 0):
        raise ValueError("time_indices must be unique, increasing, and non-negative")
    mask = np.isin(split.time_index, requested)
    trajectories = np.unique(split.trajectory_id)
    if trajectories.size == 0:
        raise DataIntegrityError(f"{split.name} split contains no trajectories")
    for trajectory in trajectories:
        observed = split.time_index[mask & (split.trajectory_id == trajectory)]
        if not np.array_equal(observed, requested):
            raise DataIntegrityError(
                f"{split.name} trajectory {int(trajectory)} does not contain the declared design"
            )
    return PreparedConfirmatorySplit(
        name=split.name,
        source_indices=_frozen_selection(split.source_indices, mask),
        X=_frozen_selection(split.X, mask),
        E=_frozen_selection(split.E, mask),
        F=_frozen_selection(split.F, mask),
        trajectory_id=_frozen_selection(split.trajectory_id, mask),
        time_index=_frozen_selection(split.time_index, mask),
        time_value=_frozen_selection(split.time_value, mask),
    )


def apply_f02_temporal_design(
    prepared: PreparedConfirmatoryDataset,
) -> PreparedConfirmatoryDataset:
    """Apply the preregistered 25/5/5 time-index design to one corpus."""

    return PreparedConfirmatoryDataset(
        train=select_time_indices(prepared.train, TRAIN_TIME_INDICES),
        validation=select_time_indices(prepared.validation, EVALUATION_TIME_INDICES),
        test=select_time_indices(prepared.test, EVALUATION_TIME_INDICES),
        normalization=prepared.normalization,
        masses=prepared.masses,
    )
