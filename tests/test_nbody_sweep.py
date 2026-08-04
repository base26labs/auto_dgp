"""Tests for the fixed-system varying-(n, d) dataset generator."""

from __future__ import annotations

import numpy as np
import pytest

from data.get_nbody_sweep import (
    ARCHIVE_KEYS,
    dataset_filename,
    generate_nbody_sweep_dataset,
    save_dataset,
)


def test_small_sweep_dataset_is_balanced_fixed_system(tmp_path) -> None:
    arrays = generate_nbody_sweep_dataset(
        n_particles=2,
        n_dims=1,
        seed=17,
        n_trajectories=4,
        rows_per_trajectory=3,
        steps_per_trajectory=6,
    )

    assert set(arrays) == set(ARCHIVE_KEYS)
    assert arrays["X"].shape == (12, 4)
    assert arrays["E"].shape == (12,)
    assert arrays["F"].shape == (12, 4)
    assert arrays["masses"].shape == (2,)
    assert np.array_equal(arrays["trajectory_id"], np.repeat(np.arange(4), 3))
    assert np.isfinite(arrays["X"]).all()
    assert np.isfinite(arrays["E"]).all()
    assert np.isfinite(arrays["F"]).all()
    for trajectory_id in range(4):
        energy = arrays["E"][arrays["trajectory_id"] == trajectory_id]
        assert np.ptp(energy) < 1e-7

    path = save_dataset(tmp_path, arrays)
    assert path.name == dataset_filename(2, 1, 17)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == set(ARCHIVE_KEYS)
        assert np.array_equal(archive["masses"], arrays["masses"])
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_dataset(tmp_path, arrays)


@pytest.mark.parametrize(
    ("n_particles", "n_dims", "seed"),
    [(1, 1, 0), (2, 0, 0), (2, 1, -1)],
)
def test_sweep_generator_rejects_invalid_schema(n_particles: int, n_dims: int, seed: int) -> None:
    with pytest.raises(ValueError):
        generate_nbody_sweep_dataset(
            n_particles=n_particles,
            n_dims=n_dims,
            seed=seed,
            n_trajectories=2,
            rows_per_trajectory=2,
            steps_per_trajectory=2,
        )
