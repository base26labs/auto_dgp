"""Focused integrity tests for the leakage-free confirmatory N-body corpus."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from data.generate_nbody_confirmatory import (
    ConfirmatoryConfig,
    DataIntegrityError,
    compute_train_normalization,
    generate_dataset,
    validate_dataset,
    verify_sha256_manifest,
    write_bundle,
)


@pytest.fixture(scope="module")
def small_config() -> ConfirmatoryConfig:
    return ConfirmatoryConfig(
        n_particles=2,
        n_dims=2,
        n_trajectories=6,
        steps_per_trajectory=5,
        replica=3,
        mass_seed=101,
        trajectory_seed=202,
        split_seed=303,
        validation_seed=404,
        train_fraction=0.5,
        validation_fraction=1.0 / 6.0,
        finite_difference_checks=3,
    )


@pytest.fixture(scope="module")
def small_dataset(small_config):
    return generate_dataset(small_config)


def test_fixed_mass_task_has_complete_trajectory_metadata(small_config, small_dataset):
    expected_rows = small_config.n_trajectories * small_config.steps_per_trajectory
    assert small_dataset.X.shape == (expected_rows, small_config.state_dim)
    assert small_dataset.E.shape == (expected_rows,)
    assert small_dataset.F.shape == (expected_rows, small_config.state_dim)
    assert small_dataset.masses.shape == (small_config.n_particles,)

    # Every trajectory is complete: generation never filters by energy or gradient labels.
    for trajectory in range(small_config.n_trajectories):
        mask = small_dataset.trajectory_id == trajectory
        assert int(mask.sum()) == small_config.steps_per_trajectory
        np.testing.assert_array_equal(
            small_dataset.time_index[mask], np.arange(small_config.steps_per_trajectory)
        )

        # The initial state uses the same persisted masses and satisfies the physical centering rules.
        initial = small_dataset.X[np.flatnonzero(mask)[0]]
        q = initial[: small_config.n_particles * small_config.n_dims].reshape(
            small_config.n_particles, small_config.n_dims
        )
        p = initial[small_config.n_particles * small_config.n_dims :].reshape(
            small_config.n_particles, small_config.n_dims
        )
        np.testing.assert_allclose((small_dataset.masses[:, None] * q).sum(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(p.sum(axis=0), 0.0, atol=1e-12)


def test_generation_and_group_splits_are_deterministic_and_disjoint(small_config, small_dataset):
    repeated = generate_dataset(small_config)
    for name in ("X", "E", "F", "masses", "trajectory_id", "time_index", "time_value"):
        np.testing.assert_array_equal(getattr(small_dataset, name), getattr(repeated, name))

    group_sets = {
        split: set(small_dataset.splits.trajectory_ids(split).tolist())
        for split in ("train", "validation", "test")
    }
    assert group_sets["train"].isdisjoint(group_sets["validation"])
    assert group_sets["train"].isdisjoint(group_sets["test"])
    assert group_sets["validation"].isdisjoint(group_sets["test"])
    assert set.union(*group_sets.values()) == set(range(small_config.n_trajectories))

    sample_indices = np.concatenate(
        [small_dataset.splits.indices(split) for split in ("train", "validation", "test")]
    )
    np.testing.assert_array_equal(np.sort(sample_indices), np.arange(small_dataset.X.shape[0]))


def test_normalization_is_unchanged_when_only_held_out_labels_change(small_dataset):
    original = small_dataset.normalization
    X_changed = small_dataset.X.copy()
    E_changed = small_dataset.E.copy()
    held_out = np.concatenate(
        [small_dataset.splits.validation_indices, small_dataset.splits.test_indices]
    )
    X_changed[held_out] += 1e6
    E_changed[held_out] -= 1e6
    recomputed = compute_train_normalization(
        X_changed,
        E_changed,
        small_dataset.F,
        small_dataset.splits.train_indices,
    )
    np.testing.assert_array_equal(recomputed.x_min, original.x_min)
    np.testing.assert_array_equal(recomputed.x_span, original.x_span)
    np.testing.assert_array_equal(recomputed.gradient_scale, original.gradient_scale)
    assert recomputed.energy_mean == original.energy_mean
    assert recomputed.energy_std == original.energy_std


def test_finite_difference_validation_rejects_corrupted_gradients(small_dataset):
    report = validate_dataset(small_dataset)
    assert report["max_finite_difference_relative_error"] < 1e-5
    corrupted_gradient = small_dataset.F.copy()
    corrupted_gradient += 1.0
    corrupted = replace(small_dataset, F=corrupted_gradient)
    with pytest.raises(DataIntegrityError, match="finite-difference relative error"):
        validate_dataset(corrupted)


def test_bundle_persists_full_metadata_and_detects_tampering(tmp_path, small_config, small_dataset):
    bundle = write_bundle(small_dataset, tmp_path, stem="confirmatory_fixture")
    actual_hashes = verify_sha256_manifest(bundle.sha256_manifest_path)
    assert set(actual_hashes) == {
        bundle.dataset_path.name,
        bundle.metadata_path.name,
    }

    metadata = json.loads(bundle.metadata_path.read_text())
    assert metadata["config"] == asdict(small_config)
    assert metadata["config"]["mass_seed"] == small_config.mass_seed
    assert metadata["config"]["trajectory_seed"] == small_config.trajectory_seed
    assert metadata["config"]["split_seed"] == small_config.split_seed
    np.testing.assert_array_equal(metadata["masses"], small_dataset.masses)
    assert metadata["normalization"]["source_split"] == "train"

    with np.load(bundle.dataset_path, allow_pickle=False) as record:
        required = {
            "X",
            "E",
            "F",
            "masses",
            "trajectory_id",
            "time_index",
            "train_indices",
            "validation_indices",
            "test_indices",
            "config_json",
        }
        assert required.issubset(record.files)
        assert json.loads(str(record["config_json"]))["replica"] == small_config.replica

    bundle.metadata_path.write_text(bundle.metadata_path.read_text() + " ")
    with pytest.raises(DataIntegrityError, match="SHA-256 mismatch"):
        verify_sha256_manifest(bundle.sha256_manifest_path)
