"""Tests for fail-closed loading and preparation of confirmatory N-body data."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from data.generate_nbody_confirmatory import (
    ConfirmatoryConfig,
    DataIntegrityError,
    generate_dataset,
    sha256_file,
    write_bundle,
    write_sha256_manifest,
)
from data.load_nbody_confirmatory import (
    load_confirmatory_bundle,
    load_prepared_confirmatory_bundle,
    prepare_confirmatory_dataset,
)


@pytest.fixture(scope="module")
def loader_config() -> ConfirmatoryConfig:
    return ConfirmatoryConfig(
        n_particles=2,
        n_dims=1,
        n_trajectories=6,
        steps_per_trajectory=4,
        replica=7,
        mass_seed=1101,
        trajectory_seed=2202,
        split_seed=3303,
        validation_seed=4404,
        train_fraction=0.5,
        validation_fraction=1.0 / 6.0,
        finite_difference_checks=2,
    )


@pytest.fixture(scope="module")
def loader_dataset(loader_config):
    return generate_dataset(loader_config)


def _write_fixture(loader_dataset, tmp_path):
    return write_bundle(loader_dataset, tmp_path, stem="loader_fixture")


def _rewrite_npz(path, update) -> None:
    with np.load(path, allow_pickle=False) as record:
        arrays = {name: np.array(record[name], copy=True) for name in record.files}
    update(arrays)
    np.savez_compressed(path, **arrays)


def _refresh_manifest(bundle) -> None:
    write_sha256_manifest(
        [bundle.dataset_path, bundle.metadata_path],
        bundle.sha256_manifest_path,
    )


def test_loader_preserves_config_metadata_hashes_and_is_deterministic(
    tmp_path,
    loader_config,
    loader_dataset,
):
    bundle = _write_fixture(loader_dataset, tmp_path)
    first = load_prepared_confirmatory_bundle(bundle.dataset_path)
    second = load_prepared_confirmatory_bundle(bundle.dataset_path)

    assert first.loaded.dataset.config == loader_config
    assert first.loaded.provenance.config_payload == asdict(loader_config)
    assert first.loaded.provenance.metadata["config"] == asdict(loader_config)
    assert first.loaded.provenance.validation_payload == first.loaded.validation
    assert first.loaded.provenance.file_sha256 == {
        bundle.dataset_path.name: sha256_file(bundle.dataset_path),
        bundle.metadata_path.name: sha256_file(bundle.metadata_path),
    }
    np.testing.assert_array_equal(first.loaded.dataset.masses, loader_dataset.masses)

    for name in ("train", "validation", "test"):
        left = first.prepared.split(name)
        right = second.prepared.split(name)
        for field in ("source_indices", "X", "E", "F", "trajectory_id", "time_index"):
            np.testing.assert_array_equal(getattr(left, field), getattr(right, field))


def test_preparation_uses_group_disjoint_manifests_and_train_only_normalization(
    tmp_path,
    loader_dataset,
):
    bundle = _write_fixture(loader_dataset, tmp_path)
    result = load_prepared_confirmatory_bundle(bundle.dataset_path)
    raw = result.loaded.dataset
    prepared = result.prepared

    groups = {
        name: set(prepared.split(name).trajectory_id.tolist())
        for name in ("train", "validation", "test")
    }
    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])
    assert set.union(*groups.values()) == set(range(raw.config.n_trajectories))

    all_indices = np.concatenate(
        [prepared.split(name).source_indices for name in ("train", "validation", "test")]
    )
    np.testing.assert_array_equal(np.sort(all_indices), np.arange(raw.X.shape[0]))
    for name in ("train", "validation", "test"):
        split = prepared.split(name)
        indices = raw.splits.indices(name)
        np.testing.assert_array_equal(split.source_indices, indices)
        np.testing.assert_allclose(
            split.X,
            (raw.X[indices] - raw.normalization.x_min) / raw.normalization.x_span,
        )
        np.testing.assert_allclose(
            split.E,
            (raw.E[indices] - raw.normalization.energy_mean) / raw.normalization.energy_std,
        )
        np.testing.assert_allclose(
            split.F,
            raw.F[indices] * raw.normalization.gradient_scale,
        )


def test_preparation_never_filters_rows_by_held_out_labels(loader_dataset):
    original = prepare_confirmatory_dataset(loader_dataset)
    held_out = np.concatenate(
        [
            loader_dataset.splits.validation_indices,
            loader_dataset.splits.test_indices,
        ]
    )
    X = loader_dataset.X.copy()
    E = loader_dataset.E.copy()
    F = loader_dataset.F.copy()
    X[held_out] += 1.0e4
    E[held_out] = 1.0e12
    F[held_out] = -1.0e12
    changed = prepare_confirmatory_dataset(replace(loader_dataset, X=X, E=E, F=F))

    np.testing.assert_array_equal(changed.train.X, original.train.X)
    np.testing.assert_array_equal(changed.train.E, original.train.E)
    np.testing.assert_array_equal(changed.train.F, original.train.F)
    for name in ("train", "validation", "test"):
        np.testing.assert_array_equal(
            changed.split(name).source_indices,
            original.split(name).source_indices,
        )
    prepared_rows = sum(changed.split(name).E.size for name in ("train", "validation", "test"))
    assert prepared_rows == E.size
    assert np.all(changed.validation.E > 1.0e6)
    assert np.all(changed.test.F < -1.0e6)


def test_loader_rejects_semantic_split_leakage_even_with_fresh_hashes(
    tmp_path,
    loader_dataset,
):
    bundle = _write_fixture(loader_dataset, tmp_path)

    def add_group_overlap(arrays):
        validation_groups = arrays["validation_trajectory_ids"].copy()
        validation_groups[0] = arrays["train_trajectory_ids"][0]
        arrays["validation_trajectory_ids"] = validation_groups

    _rewrite_npz(bundle.dataset_path, add_group_overlap)
    _refresh_manifest(bundle)
    with pytest.raises(DataIntegrityError, match="trajectory leakage"):
        load_confirmatory_bundle(bundle.dataset_path)


def test_loader_rejects_non_train_normalization_even_with_fresh_hashes(
    tmp_path,
    loader_dataset,
):
    bundle = _write_fixture(loader_dataset, tmp_path)

    def corrupt_normalization(arrays):
        x_min = arrays["x_train_min"].copy()
        x_min[0] += 0.5
        arrays["x_train_min"] = x_min

    _rewrite_npz(bundle.dataset_path, corrupt_normalization)
    _refresh_manifest(bundle)
    with pytest.raises(DataIntegrityError, match="x_min was not computed solely from the train"):
        load_confirmatory_bundle(bundle.dataset_path)


def test_loader_rejects_hash_tampering_and_cross_artifact_config_mismatch(
    tmp_path,
    loader_dataset,
):
    hash_bundle = _write_fixture(loader_dataset, tmp_path / "hash")
    hash_bundle.metadata_path.write_text(hash_bundle.metadata_path.read_text() + " ")
    with pytest.raises(DataIntegrityError, match="SHA-256 mismatch"):
        load_confirmatory_bundle(hash_bundle.dataset_path)

    config_bundle = _write_fixture(loader_dataset, tmp_path / "config")
    metadata = json.loads(config_bundle.metadata_path.read_text())
    metadata["config"]["replica"] += 1
    config_bundle.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    _refresh_manifest(config_bundle)
    with pytest.raises(DataIntegrityError, match="metadata config does not match"):
        load_confirmatory_bundle(config_bundle.dataset_path)
