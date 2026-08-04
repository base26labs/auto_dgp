from __future__ import annotations

import numpy as np
import pytest

from data.generate_nbody_confirmatory import DataIntegrityError, TrainNormalization
from data.load_nbody_confirmatory import (
    PreparedConfirmatoryDataset,
    PreparedConfirmatorySplit,
)
from experiments.f02_design import (
    EVALUATION_TIME_INDICES,
    OPTIMIZER_SELECTION_TIME_INDICES,
    TRAIN_TIME_INDICES,
    apply_f02_temporal_design,
    select_time_indices,
)


def _split(name: str, trajectory_ids: tuple[int, ...]) -> PreparedConfirmatorySplit:
    rows = [(trajectory, time) for trajectory in trajectory_ids for time in range(100)]
    trajectory = np.asarray([row[0] for row in rows], dtype=np.int64)
    time = np.asarray([row[1] for row in rows], dtype=np.int64)
    source = np.arange(len(rows), dtype=np.int64) + 1000 * min(trajectory_ids)
    X = np.stack([source, -source], axis=1).astype(np.float64)
    E = (10_000 + source).astype(np.float64)
    F = np.stack([E, -E], axis=1)
    return PreparedConfirmatorySplit(
        name=name,
        source_indices=source,
        X=X,
        E=E,
        F=F,
        trajectory_id=trajectory,
        time_index=time,
        time_value=0.01 * time,
    )


def test_f02_temporal_design_is_exact_and_represents_every_trajectory() -> None:
    normalization = TrainNormalization(
        x_min=np.zeros(2),
        x_span=np.ones(2),
        energy_mean=0.0,
        energy_std=1.0,
        gradient_scale=np.ones(2),
    )
    prepared = PreparedConfirmatoryDataset(
        train=_split("train", (0, 1, 2)),
        validation=_split("validation", (3, 4)),
        test=_split("test", (5, 6)),
        normalization=normalization,
        masses=np.ones(2),
    )

    designed = apply_f02_temporal_design(prepared)

    assert designed.train.X.shape[0] == 3 * len(TRAIN_TIME_INDICES)
    assert designed.validation.X.shape[0] == 2 * len(EVALUATION_TIME_INDICES)
    assert designed.test.X.shape[0] == 2 * len(EVALUATION_TIME_INDICES)
    assert OPTIMIZER_SELECTION_TIME_INDICES == (50,)
    for split, expected_times in (
        (designed.train, TRAIN_TIME_INDICES),
        (designed.validation, EVALUATION_TIME_INDICES),
        (designed.test, EVALUATION_TIME_INDICES),
    ):
        for trajectory in np.unique(split.trajectory_id):
            assert tuple(split.time_index[split.trajectory_id == trajectory]) == expected_times
        assert np.array_equal(split.E, 10_000 + split.source_indices)
        assert split.X.flags.writeable is False
        assert split.E.flags.writeable is False
        assert split.F.flags.writeable is False


def test_time_selection_rejects_malformed_or_missing_design() -> None:
    split = _split("test", (5, 6))
    with pytest.raises(ValueError):
        select_time_indices(split, (0, 0, 99))
    with pytest.raises(ValueError):
        select_time_indices(split, (-1, 5))
    with pytest.raises(ValueError):
        select_time_indices(split, ())
    with pytest.raises(DataIntegrityError, match="does not contain the declared design"):
        select_time_indices(split, (0, 100))
