from __future__ import annotations

import copy
import random
from dataclasses import fields, replace

import numpy as np
import pytest
import torch

from data.load_nbody_confirmatory import PreparedConfirmatorySplit
from experiments.f02_design import EVALUATION_TIME_INDICES, TRAIN_TIME_INDICES
from experiments.f02_external_adapter import (
    ARTIFACT_SCHEMA_VERSION,
    CONFIG_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    CanonicalEvaluationFeatures,
    CorpusIdentity,
    ExternalAdapterError,
    ExternalBaselineConfig,
    ExternalPredictions,
    TrainingAccounting,
    join_central_labels,
    load_central_evaluation_labels,
    load_evaluation_features,
    load_training_artifact,
    run_external_worker,
    seed_external_runtime,
    sha256_file,
    validate_external_config_payload,
    validate_external_result,
    validate_training_accounting,
    write_external_artifact_bundle,
)


def _split(
    name: str,
    *,
    rows_per_trajectory: int,
    trajectory_count: int,
    time_indices: tuple[int, ...],
    source_start: int,
    dimension: int = 12,
) -> PreparedConfirmatorySplit:
    rows = rows_per_trajectory * trajectory_count
    assert rows_per_trajectory == len(time_indices)
    source_indices = np.arange(source_start, source_start + rows, dtype=np.int64)
    X = np.linspace(-0.9, 0.9, rows * dimension, dtype=np.float64).reshape(rows, dimension)
    # Standardized positive-H convention means H is not negated; standardized
    # values themselves may of course lie on either side of zero.
    H = np.linspace(-1.0, 1.0, rows, dtype=np.float64)
    dH_dx = np.linspace(-0.4, 0.6, rows * dimension, dtype=np.float64).reshape(rows, dimension)
    trajectory_id = np.repeat(np.arange(trajectory_count, dtype=np.int64), rows_per_trajectory)
    time_index = np.tile(np.asarray(time_indices, dtype=np.int64), trajectory_count)
    return PreparedConfirmatorySplit(
        name=name,
        source_indices=source_indices,
        X=X,
        E=H,
        F=dH_dx,
        trajectory_id=trajectory_id,
        time_index=time_index,
        time_value=time_index.astype(np.float64) * 0.01,
    )


@pytest.fixture
def identity() -> CorpusIdentity:
    return CorpusIdentity(
        bundle_id="replica-0-n-2-d-12",
        catalog_task_index=0,
        replica=0,
        n_particles=2,
        dimension=12,
        phase="development",
        dataset_content_sha256="a" * 64,
        catalog_sha256="b" * 64,
    )


def test_corpus_identity_binds_canonical_bundle_id_and_catalog_index(identity):
    with pytest.raises(ValueError, match="bundle_id"):
        replace(identity, bundle_id="replica-0-n-2")
    with pytest.raises(ValueError, match="catalog_task_index"):
        replace(identity, catalog_task_index=1)


@pytest.fixture
def artifact_paths(tmp_path, identity):
    training = _split(
        "train",
        rows_per_trajectory=25,
        trajectory_count=60,
        time_indices=TRAIN_TIME_INDICES,
        source_start=0,
    )
    evaluation = _split(
        "validation",
        rows_per_trajectory=5,
        trajectory_count=20,
        time_indices=EVALUATION_TIME_INDICES,
        source_start=2_000,
    )
    return write_external_artifact_bundle(
        tmp_path / "artifacts",
        identity=identity,
        normalization_sha256="c" * 64,
        training=training,
        evaluation=evaluation,
    )


def _accounting(method_id: str, updates: int = 20) -> TrainingAccounting:
    epochs = updates // 2
    return TrainingAccounting(
        epochs_completed=epochs,
        logical_update_attempts=updates,
        logical_updates_applied=updates,
        skipped_nonfinite_updates=0,
        optimizer_step_calls=updates if method_id == "dsoftki-512" else 2 * updates,
        states_processed=epochs * 1_500,
        posterior_fit_solve_passes=epochs if method_id == "dsoftki-512" else 0,
        posterior_fit_pseudoinverse_passes=0,
        directional_coordinate_draw_count=0 if method_id == "dsoftki-512" else 2 * updates,
        directional_coordinate_sequence_sha256=(None if method_id == "dsoftki-512" else "d" * 64),
        fit_seconds_descriptive=1.25,
        fit_peak_gpu_allocated_bytes=12_345,
    )


def test_artifacts_are_deterministic_and_worker_view_has_no_label_path(
    tmp_path,
    identity,
):
    training = _split(
        "train",
        rows_per_trajectory=25,
        trajectory_count=60,
        time_indices=TRAIN_TIME_INDICES,
        source_start=0,
    )
    evaluation = _split(
        "validation",
        rows_per_trajectory=5,
        trajectory_count=20,
        time_indices=EVALUATION_TIME_INDICES,
        source_start=2_000,
    )
    first = write_external_artifact_bundle(
        tmp_path / "first",
        identity=identity,
        normalization_sha256="c" * 64,
        training=training,
        evaluation=evaluation,
    )
    second = write_external_artifact_bundle(
        tmp_path / "second",
        identity=identity,
        normalization_sha256="c" * 64,
        training=training,
        evaluation=evaluation,
    )
    assert sha256_file(first.training) == sha256_file(second.training)
    assert sha256_file(first.evaluation_features) == sha256_file(second.evaluation_features)
    assert sha256_file(first.evaluation_labels_central_only) == sha256_file(
        second.evaluation_labels_central_only
    )

    with np.load(first.evaluation_features, allow_pickle=False) as record:
        assert set(record.files) == {
            "metadata_json",
            "X_standardized",
            "source_ids",
            "source_indices",
            "trajectory_ids",
            "time_indices",
        }
        assert "H_standardized" not in record.files
        assert "dH_dx_standardized" not in record.files

    worker_fields = {field.name for field in fields(first.worker_inputs())}
    assert worker_fields == {"training", "evaluation_features"}
    assert "labels" not in " ".join(worker_fields)


def test_strict_feature_loader_rejects_injected_evaluation_label(artifact_paths, tmp_path):
    with np.load(artifact_paths.evaluation_features, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    arrays["H_standardized"] = np.zeros(len(arrays["source_ids"]))
    malicious = tmp_path / "malicious.features.npz"
    np.savez(malicious, **arrays)
    with pytest.raises(ExternalAdapterError, match="unexpected=.*H_standardized"):
        load_evaluation_features(malicious)


@pytest.mark.parametrize("method_id", ["dsoftki-512", "ddsvgp-512"])
def test_external_configs_lock_released_semantics_and_f02_overrides(method_id):
    config = ExternalBaselineConfig(
        method_id=method_id,
        dimension=60,
        seed=29,
        selected_updates=50,
    )
    payload = config.to_payload()
    assert payload["schema_version"] == CONFIG_SCHEMA_VERSION
    assert payload["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert payload["result_schema_version"] == RESULT_SCHEMA_VERSION
    assert payload["training"]["batch_size"] == 1_024
    assert payload["training"]["epochs"] == 25
    assert payload["training"]["selected_logical_optimizer_updates"] == 50
    assert payload["training"]["dataloader_workers"] == 0
    assert payload["training"]["test_dataset_passed_to_train"] is False
    assert payload["preprocessing"]["energy_sign_flip"] is False
    assert payload["preprocessing"]["joint_value_gradient_scaling"] is False
    assert payload["variance_contract"]["common_latent_verified"] is False
    assert payload["variance_contract"]["eligible_for_common_latent_nll"] is False
    if method_id == "dsoftki-512":
        assert payload["model"]["num_interp"] == 512
        assert payload["model"]["derivative_noise_initial_variance"] == 6.0
        assert payload["model"]["use_scale"] is False
        assert payload["model"]["use_ard"] is False
        assert payload["released_nbody_reference"]["use_ard"] is True
        assert payload["training"]["learning_rate"] == 0.02
    else:
        assert payload["model"]["num_inducing"] == 512
        assert payload["model"]["num_directions"] == 2
        assert payload["model"]["mll_type"] == "PLL"
        assert payload["training"]["optimizer_step_calls_per_logical_update"] == 2
        assert payload["training"]["learning_rate"] == 0.03
    assert validate_external_config_payload(payload) == config

    changed = copy.deepcopy(payload)
    changed["training"]["dataloader_workers"] = 1
    with pytest.raises(ExternalAdapterError, match="differs from the frozen"):
        validate_external_config_payload(changed)


def test_seed_external_runtime_seeds_python_numpy_torch_and_cuda_all(monkeypatch):
    cuda_calls: list[int] = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", cuda_calls.append)
    first_report = seed_external_runtime(11)
    first = (random.random(), float(np.random.rand()), float(torch.rand(())))
    second_report = seed_external_runtime(11)
    second = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert first == second
    assert first_report == second_report
    assert first_report["dataloader_workers"] == 0
    assert cuda_calls and all(seed == 11 for seed in cuda_calls)


def test_training_accounting_fails_on_silent_dsoftki_skip():
    config = ExternalBaselineConfig(
        method_id="dsoftki-512",
        dimension=12,
        seed=11,
        selected_updates=20,
    )
    accounting = _accounting("dsoftki-512")
    changed = replace(
        accounting,
        logical_updates_applied=19,
        skipped_nonfinite_updates=1,
        optimizer_step_calls=19,
    )
    with pytest.raises(ExternalAdapterError, match="cannot hide skipped"):
        validate_training_accounting(config, changed)


class _FakeDSoftKIBackend:
    method_id = "dsoftki-512"
    is_test_double = True

    def __init__(self) -> None:
        self.evaluation_type: type | None = None

    def train(self, training, config):
        assert training.H_standardized.shape == (1_500,)
        assert training.dH_dx_standardized.shape == (1_500, 12)
        return object(), _accounting(config.method_id, config.selected_updates)

    def predict(self, model, evaluation, config):
        del model
        self.evaluation_type = type(evaluation)
        assert isinstance(evaluation, CanonicalEvaluationFeatures)
        assert not hasattr(evaluation, "H_standardized")
        assert not hasattr(evaluation, "dH_dx_standardized")
        rows = len(evaluation.source_ids)
        return ExternalPredictions(
            source_ids=evaluation.source_ids,
            mean_standardized_H=np.zeros(rows, dtype=np.float64),
            native_variance_standardized_H=np.ones(rows, dtype=np.float64),
            gradient_standardized_dH_dx=None,
            observation_noise_variance=0.1,
            prediction_seconds_descriptive=0.25,
            prediction_peak_gpu_allocated_bytes=4_096,
        )


def test_worker_is_label_isolated_and_result_is_fail_closed(artifact_paths):
    backend = _FakeDSoftKIBackend()
    config = ExternalBaselineConfig(
        method_id="dsoftki-512",
        dimension=12,
        seed=11,
        selected_updates=20,
    )
    result = run_external_worker(
        artifacts=artifact_paths.worker_inputs(),
        config=config,
        backend=backend,
        repo_commit="1" * 40,
        repo_tree="2" * 40,
        dependency_lock_sha256="3" * 64,
        test_only=True,
    )
    validate_external_result(result)
    assert backend.evaluation_type is CanonicalEvaluationFeatures
    assert result["task_identity"] == {
        "method_id": "dsoftki-512",
        "bundle_id": "replica-0-n-2-d-12",
        "catalog_task_index": 0,
        "replica": 0,
        "n_particles": 2,
        "dimension": 12,
        "seed": 11,
        "optimizer_updates": 20,
        "evaluation_split": "validation",
        "evaluation_design": "primary",
    }
    assert result["variance_semantics"]["common_latent_verified"] is False
    assert result["variance_semantics"]["common_latent_variance"] is None
    assert "metrics" not in result

    labels = load_central_evaluation_labels(artifact_paths.evaluation_labels_central_only)
    H, dH_dx = join_central_labels(result, labels)
    assert H.shape == (100,)
    assert dH_dx.shape == (100, 12)

    false_claim = copy.deepcopy(result)
    false_claim["variance_semantics"]["common_latent_verified"] = True
    false_claim["variance_semantics"]["common_latent_variance"] = [1.0] * 100
    with pytest.raises(ExternalAdapterError, match="unverified common latent"):
        validate_external_result(false_claim)

    wrong_sources = copy.deepcopy(result)
    wrong_sources["predictions"]["source_ids"][0] = "wrong-source"
    with pytest.raises(ExternalAdapterError, match="bound hash"):
        validate_external_result(wrong_sources)


def test_production_worker_refuses_unattested_shared_host(artifact_paths):
    config = ExternalBaselineConfig(
        method_id="dsoftki-512",
        dimension=12,
        seed=11,
        selected_updates=20,
    )
    with pytest.raises(ExternalAdapterError, match="exclusive Slurm"):
        run_external_worker(
            artifacts=artifact_paths.worker_inputs(),
            config=config,
            backend=_FakeDSoftKIBackend(),
            repo_commit="1" * 40,
            repo_tree="2" * 40,
            dependency_lock_sha256="3" * 64,
            environment={},
        )


def test_ddsvgp_accounting_requires_two_steps_and_direction_sequence_hash():
    config = ExternalBaselineConfig(
        method_id="ddsvgp-512",
        dimension=12,
        seed=47,
        selected_updates=100,
    )
    valid = _accounting("ddsvgp-512", 100)
    validate_training_accounting(config, valid)
    invalid = replace(valid, optimizer_step_calls=100)
    with pytest.raises(ExternalAdapterError, match="two optimizer.step"):
        validate_training_accounting(config, invalid)


def test_loaders_keep_train_labels_and_eval_labels_in_separate_types(artifact_paths):
    training = load_training_artifact(artifact_paths.training)
    features = load_evaluation_features(artifact_paths.evaluation_features)
    labels = load_central_evaluation_labels(artifact_paths.evaluation_labels_central_only)
    assert training.H_standardized.shape == (1_500,)
    assert features.X_standardized.shape == (100, 12)
    assert labels.H_standardized.shape == (100,)
    assert features.source_ids == labels.source_ids
