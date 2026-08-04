from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from cluster.f02b_calibration_grid import fit_task_for_index
from data.generate_nbody_confirmatory import ConfirmatoryConfig
from data.load_nbody_confirmatory import PreparedConfirmatorySplit
from experiments import f02b_calibration_fit as fit_module
from experiments.f02_design import TRAIN_TIME_INDICES
from experiments.f02_internal_models import TensorConfirmatorySplit
from experiments.f02b_calibration_contract import (
    CALIBRATION_ID,
    CATALOG_IDENTITY,
    EXCLUDED_FIT_RECIPE_FIELDS,
    F02_CATALOG_SHA256,
    FIT_RECIPE,
    FIT_RECIPE_SHA256,
    FIT_TASK_MATRIX_SHA256,
    MATRIX_HASHES,
    MINIMUM_HOST_MEMORY_BYTES,
    NUMERICAL_POLICY,
    WALLTIME_SECONDS,
    canonical_json_bytes,
    canonical_sha256,
)
from experiments.f02b_calibration_fit import (
    FIT_PAYLOAD_SCHEMA_VERSION,
    SHARING_VERIFICATION_MODE,
    CalibrationFitError,
    FitInputs,
    build_fit_artifacts,
    build_fit_numeric_payload,
    build_parser,
    default_fit_callable,
    fit_artifact_paths,
    fit_config_for_task_index,
    main,
    run_fit_task,
    validate_fit_artifact_pair,
    validate_fit_numeric_payload,
    write_fit_artifacts_exclusive,
)


def _parameters() -> dict[str, object]:
    return {
        "lengthscale": [1.25],
        "outputscale": 2.0,
        "sigma_f": float(torch.tensor(0.001, dtype=torch.float32).item()),
        "sigma_g": float(torch.tensor(0.002, dtype=torch.float32).item()),
        "kernel": "rbf",
        "gradient_noise_model": "iid",
    }


def _runtime_allocation() -> dict[str, object]:
    return {
        "exclusive_node": False,
        "requested_gpu_count": 0,
        "visible_gpu_count": 0,
        "visible_gpu_models": [],
        "visible_gpu_memory_bytes": [],
        "requested_cpus_per_task": 8,
        "available_cpu_count": 32,
        "available_host_memory_bytes": MINIMUM_HOST_MEMORY_BYTES,
        "requested_walltime_seconds": WALLTIME_SECONDS,
        "walltime_limit_seconds": WALLTIME_SECONDS,
        "array_concurrency": 1,
        "partition": "short",
    }


def _identity(
    base: Path,
    task_index: int = 0,
    *,
    dataset_path: Path | None = None,
    catalog_path: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    task = fit_task_for_index(task_index)
    dataset = (dataset_path or base / "corpus" / f"{task.dataset_stem}.npz").resolve()
    catalog = (catalog_path or base / "catalog.json").resolve()
    repository = (repo_root or base / "repo").resolve()
    metadata = dataset.with_suffix(".metadata.json")
    manifest = dataset.with_suffix(".sha256.json")
    particle_index = (2, 4, 6, 8, 10).index(task.n_particles)
    return {
        "catalog": {
            "path": str(catalog),
            "sha256": F02_CATALOG_SHA256,
            "schema_version": 1,
            "generation_git_commit": "1" * 40,
            "generation_git_tree": "2" * 40,
        },
        "bundle": {
            "dataset_path": str(dataset),
            "metadata_path": str(metadata),
            "manifest_path": str(manifest),
            "file_sha256": {
                dataset.name: "a" * 64,
                metadata.name: "b" * 64,
            },
            "sha256_manifest_file_sha256": "c" * 64,
            "dataset_content_sha256": "d" * 64,
            "generator_config": asdict(
                ConfirmatoryConfig(
                    n_particles=task.n_particles,
                    n_dims=task.n_dims,
                    replica=task.replica,
                )
            ),
            "catalog_bundle_task_index": task.replica * 5 + particle_index,
            "phase": "development",
            "replica": task.replica,
            "n_particles": task.n_particles,
            "n_dims": task.n_dims,
            "D": task.dimension,
        },
        "source": {
            "repo_root": str(repository),
            "commit": "3" * 40,
            "tree": "4" * 40,
            "status_porcelain": [],
            "tera_gitlink": "5" * 40,
        },
        "dependencies": {
            "pyproject.toml": {"sha256": "6" * 64},
            "uv.lock": {"sha256": "7" * 64},
        },
        "runtime": {
            "python_executable": "/opt/f02b/python",
            "python_version": "3.12.4",
            "platform": "Linux-f02b",
            "packages": [
                {"name": "numpy", "version": "2.0.1"},
                {"name": "torch", "version": "2.4.1"},
            ],
        },
        "runtime_allocation": _runtime_allocation(),
        "scheduler": {
            "job_id": "41001",
            "array_job_id": "41000",
            "array_task_id": task.task_index,
            "node_list": "cpu-shared-01",
            "sharing_verification_mode": SHARING_VERIFICATION_MODE,
        },
    }


def _payload(tmp_path: Path, task_index: int = 0) -> dict[str, Any]:
    return build_fit_numeric_payload(task_index, _parameters(), _identity(tmp_path, task_index))


def _noop_postflight(*_args: object) -> None:
    return None


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _all_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _all_keys(item)}
    return set()


def test_private_internal_config_is_complete_but_payload_recipe_is_fit_only(
    tmp_path: Path,
) -> None:
    task = fit_task_for_index(0)
    config = fit_config_for_task_index(task.task_index)
    for name, expected in FIT_RECIPE.items():
        assert getattr(config, name) == expected
    assert config.seed == task.seed
    assert config.candidate_m == ()
    assert config.cg_tolerance == 1e-5
    assert config.cg_max_iterations is None
    assert config.use_preconditioner is True
    assert config.function_jitter == 1e-8
    assert config.reduced_jitter == 1e-8

    payload = _payload(tmp_path)
    assert payload["fit_recipe"] == dict(FIT_RECIPE)
    assert payload["fit_recipe_sha256"] == FIT_RECIPE_SHA256
    assert "internal_task_config" not in payload
    assert not (set(payload["fit_recipe"]) & EXCLUDED_FIT_RECIPE_FIELDS)
    assert payload["task_record"]["seed"] == task.seed


def test_payload_is_parameter_only_train_only_and_detached(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    assert payload["schema_version"] == FIT_PAYLOAD_SCHEMA_VERSION
    assert payload["task_role"] == "fit"
    assert payload["data_access"] == {
        "integrity_loading": "authoritative_complete_bundle",
        "tensorized_splits": ["train"],
        "fit_callable_splits": ["train"],
        "training_time_indices": list(TRAIN_TIME_INDICES),
        "prediction_or_scoring_performed": False,
    }
    assert set(payload["parameters"]) == {
        "gradient_noise_model",
        "kernel",
        "lengthscale",
        "outputscale",
        "sigma_f",
        "sigma_g",
    }
    forbidden = {"selected_update", "freeze_ready", "model_state", "optimizer_state"}
    assert not (_all_keys(payload) & forbidden)
    assert b"pickle" not in canonical_json_bytes(payload).lower()

    payload["data_access"]["tensorized_splits"].append("validation")
    fresh = _payload(tmp_path)
    assert fresh["data_access"]["tensorized_splits"] == ["train"]
    with pytest.raises(CalibrationFitError, match="data-access"):
        validate_fit_numeric_payload(payload)


def test_common_envelope_binds_canonical_payload_and_all_public_identities(
    tmp_path: Path,
) -> None:
    payload, payload_bytes, envelope = build_fit_artifacts(
        0,
        _parameters(),
        _identity(tmp_path),
        output_root=tmp_path / "output",
    )
    validated_payload, validated_envelope = validate_fit_artifact_pair(
        payload_bytes,
        canonical_json_bytes(envelope),
    )
    assert validated_payload == payload
    record = validated_envelope["canonical_record"]
    assert record["calibration_id"] == CALIBRATION_ID
    assert record["matrix_hashes"] == dict(MATRIX_HASHES)
    assert record["catalog_identity"] == dict(CATALOG_IDENTITY)
    assert record["fit_recipe"] == dict(FIT_RECIPE)
    assert record["fit_recipe_sha256"] == FIT_RECIPE_SHA256
    assert record["numerical_policy"] == dict(NUMERICAL_POLICY)
    assert record["runtime_allocation"]["requested_gpu_count"] == 0
    assert record["runtime_allocation"]["visible_gpu_count"] == 0
    binding = record["payload_binding"]
    expected = fit_artifact_paths(tmp_path / "output", 0)
    assert binding["numeric_payload_path"] == (
        f"{expected.directory.name}/{expected.numeric_payload.name}"
    )
    assert binding["numeric_payload_sha256"] == hashlib.sha256(payload_bytes).hexdigest()
    assert expected.directory.name.endswith(FIT_TASK_MATRIX_SHA256[:12])


def test_run_injection_activates_numerical_policy_before_fit_and_writes_canonical_pair(
    tmp_path: Path,
) -> None:
    task = fit_task_for_index(0)
    dataset = (tmp_path / f"{task.dataset_stem}.npz").resolve()
    catalog = (tmp_path / "catalog.json").resolve()
    repo = (tmp_path / "repo").resolve()
    identity = _identity(
        tmp_path,
        dataset_path=dataset,
        catalog_path=catalog,
        repo_root=repo,
    )
    sentinel_train = object()
    events: list[str] = []

    def identity_provider(*args: object) -> FitInputs:
        assert args[2:] == (dataset, catalog, repo)
        events.append("identity")
        return FitInputs(identity=identity, train=sentinel_train)

    def fit_callable(train: object, config: object) -> dict[str, object]:
        assert train is sentinel_train
        assert config.seed == task.seed
        assert torch.get_float32_matmul_precision() == "highest"
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        events.append("fit")
        return _parameters()

    def postflight(*args: object) -> None:
        assert args[-1] == identity
        events.append("postflight")

    original = (
        torch.get_float32_matmul_precision(),
        bool(torch.backends.cuda.matmul.allow_tf32),
        bool(torch.backends.cudnn.allow_tf32),
    )
    try:
        torch.set_float32_matmul_precision("medium")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        before = (
            torch.get_float32_matmul_precision(),
            bool(torch.backends.cuda.matmul.allow_tf32),
            bool(torch.backends.cudnn.allow_tf32),
        )
        payload, envelope, paths = run_fit_task(
            task.task_index,
            dataset_path=dataset,
            catalog_path=catalog,
            output_root=tmp_path / "results",
            repo_root=repo,
            fit_callable=fit_callable,
            identity_provider=identity_provider,
            postflight_validator=postflight,
        )
        after = (
            torch.get_float32_matmul_precision(),
            bool(torch.backends.cuda.matmul.allow_tf32),
            bool(torch.backends.cudnn.allow_tf32),
        )
        assert after == before
    finally:
        torch.set_float32_matmul_precision(original[0])
        torch.backends.cuda.matmul.allow_tf32 = original[1]
        torch.backends.cudnn.allow_tf32 = original[2]

    assert events == ["identity", "fit", "postflight"]
    assert paths.numeric_payload.read_bytes() == canonical_json_bytes(payload)
    assert paths.execution_envelope.read_bytes() == canonical_json_bytes(envelope)
    validate_fit_artifact_pair(paths.numeric_payload, paths.execution_envelope)


def test_default_fit_adapter_passes_only_the_exact_supported_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = TensorConfirmatorySplit(
        name="train",
        source_indices=torch.tensor([0]),
        X=torch.zeros((1, 2), dtype=torch.float32),
        value=torch.zeros(1, dtype=torch.float32),
        gradient=torch.zeros((1, 2), dtype=torch.float32),
        trajectory_id=torch.tensor([0]),
        time_index=torch.tensor([0]),
        time_value=torch.zeros(1, dtype=torch.float32),
    )
    seen: dict[str, object] = {}
    model = object()
    frozen = object()

    def fake_fit(received_train: object, **kwargs: object) -> object:
        assert received_train is train
        seen.update(kwargs)
        return model

    monkeypatch.setattr(fit_module, "fit_released_tera", fake_fit)
    monkeypatch.setattr(
        fit_module,
        "freeze_tera_parameters",
        lambda received: frozen if received is model else None,
    )
    config = fit_config_for_task_index(0)
    assert default_fit_callable(train, config) is frozen
    assert seen == {
        **{name: value for name, value in FIT_RECIPE.items() if name not in {"dtype", "device"}},
        "seed": config.seed,
    }


def test_default_fit_modules_preload_only_from_bound_source_tree(tmp_path: Path) -> None:
    fit_module._preload_default_fit_implementation(fit_module._REPO_ROOT)
    with pytest.raises(CalibrationFitError, match="outside the bound source tree"):
        fit_module._preload_default_fit_implementation(tmp_path)


def test_default_identity_provider_tensorizes_only_selected_training_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = fit_task_for_index(0)
    dataset = (tmp_path / f"{task.dataset_stem}.npz").resolve()
    catalog = (tmp_path / "catalog.json").resolve()
    repo = (tmp_path / "repo").resolve()
    identity = _identity(
        tmp_path,
        dataset_path=dataset,
        catalog_path=catalog,
        repo_root=repo,
    )
    dimension = 4
    time_index = np.arange(100, dtype=np.int64)
    train = PreparedConfirmatorySplit(
        name="train",
        source_indices=np.arange(100, dtype=np.int64),
        X=np.arange(100 * dimension, dtype=np.float64).reshape(100, dimension),
        E=np.arange(100, dtype=np.float64),
        F=np.arange(100 * dimension, dtype=np.float64).reshape(100, dimension),
        trajectory_id=np.zeros(100, dtype=np.int64),
        time_index=time_index,
        time_value=time_index.astype(np.float64) / 100.0,
    )

    class TrainOnlyPrepared:
        def __init__(self, selected_train: PreparedConfirmatorySplit) -> None:
            self.train = selected_train

        @property
        def validation(self) -> object:
            raise AssertionError("validation split must not be accessed")

        @property
        def test(self) -> object:
            raise AssertionError("test split must not be accessed")

    bundle = SimpleNamespace(prepared=TrainOnlyPrepared(train), loaded=object())
    preflight = object()
    calls: list[str] = []
    corpus = {name: identity[name] for name in ("catalog", "bundle")}
    snapshot = SimpleNamespace(dataset_path=dataset, catalog_path=catalog)

    class SnapshotContext:
        def __enter__(self) -> object:
            assert calls == ["preload", "allocation", "source", "dependencies", "runtime"]
            calls.append("snapshot")
            return snapshot

        def __exit__(self, *_args: object) -> None:
            return None

    def preflight_identity(
        *args: object,
        **kwargs: object,
    ) -> tuple[dict[str, Any], object]:
        assert args == (task, dataset, catalog)
        assert kwargs == {"snapshot": snapshot}
        calls.append("preflight")
        return copy.deepcopy(corpus), preflight

    def load_bundle(path: Path) -> object:
        assert path == dataset
        calls.append("load")
        return bundle

    def loaded_matches(
        received_bundle: object,
        received_preflight: object,
        received_dataset: object,
    ) -> None:
        assert received_bundle is bundle
        assert received_preflight is preflight
        assert received_dataset == dataset
        calls.append("integrity")

    def source_identity(*_args: object) -> dict[str, Any]:
        calls.append("source")
        return identity["source"]

    def dependency_identity(*_args: object) -> dict[str, Any]:
        calls.append("dependencies")
        return identity["dependencies"]

    def runtime_identity() -> dict[str, Any]:
        calls.append("runtime")
        return identity["runtime"]

    def runtime_evidence(*_args: object) -> tuple[dict[str, Any], dict[str, Any]]:
        calls.append("allocation")
        return identity["runtime_allocation"], identity["scheduler"]

    def preload(*_args: object) -> None:
        calls.append("preload")

    monkeypatch.setattr(
        fit_module,
        "_immutable_corpus_snapshot",
        lambda *args: SnapshotContext() if args == (dataset, catalog) else pytest.fail(),
    )
    monkeypatch.setattr(fit_module, "_preflight_corpus_identity", preflight_identity)
    monkeypatch.setattr(fit_module, "load_prepared_confirmatory_bundle", load_bundle)
    monkeypatch.setattr(fit_module, "_loaded_bundle_matches_preflight", loaded_matches)
    monkeypatch.setattr(fit_module, "_source_identity", source_identity)
    monkeypatch.setattr(fit_module, "_dependency_identity", dependency_identity)
    monkeypatch.setattr(fit_module, "_runtime_identity", runtime_identity)
    monkeypatch.setattr(fit_module, "_observe_runtime_evidence", runtime_evidence)
    monkeypatch.setattr(fit_module, "_preload_default_fit_implementation", preload)
    monkeypatch.setattr(
        fit_module,
        "fit_released_tera",
        lambda *_args, **_kwargs: pytest.fail("identity loading must not fit or predict"),
    )

    inputs = fit_module.default_identity_provider(
        task,
        replace(fit_config_for_task_index(0), device="cpu"),
        dataset,
        catalog,
        repo,
    )
    assert calls == [
        "preload",
        "allocation",
        "source",
        "dependencies",
        "runtime",
        "snapshot",
        "preflight",
        "load",
        "integrity",
    ]
    assert inputs.identity == identity
    assert isinstance(inputs.train, TensorConfirmatorySplit)
    assert inputs.train.name == "train"
    assert inputs.train.time_index.tolist() == list(TRAIN_TIME_INDICES)
    assert inputs.train.source_indices.tolist() == list(TRAIN_TIME_INDICES)
    assert inputs.train.X.shape == (len(TRAIN_TIME_INDICES), dimension)
    assert inputs.train.X.dtype == torch.float32
    assert inputs.train.value.dtype == torch.float32
    assert inputs.train.gradient.dtype == torch.float32
    assert inputs.train.time_value.dtype == torch.float32


@pytest.mark.parametrize("drift", ["catalog", "bundle", "source", "dependencies"])
def test_postflight_identity_change_prevents_any_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    task = fit_task_for_index(0)
    dataset = (tmp_path / f"{task.dataset_stem}.npz").resolve()
    catalog = (tmp_path / "catalog.json").resolve()
    repo = (tmp_path / "repo").resolve()
    identity = _identity(
        tmp_path,
        dataset_path=dataset,
        catalog_path=catalog,
        repo_root=repo,
    )
    corpus = copy.deepcopy({name: identity[name] for name in ("catalog", "bundle")})
    source = copy.deepcopy(identity["source"])
    dependencies = copy.deepcopy(identity["dependencies"])
    if drift == "catalog":
        corpus["catalog"]["generation_git_tree"] = "8" * 40
    elif drift == "bundle":
        corpus["bundle"]["dataset_content_sha256"] = "8" * 64
    elif drift == "source":
        source["commit"] = "8" * 40
    elif drift == "dependencies":
        dependencies["uv.lock"]["sha256"] = "8" * 64
    monkeypatch.setattr(
        fit_module,
        "_preflight_corpus_identity",
        lambda *_args: (copy.deepcopy(corpus), object()),
    )
    monkeypatch.setattr(fit_module, "_source_identity", lambda *_args: source)
    monkeypatch.setattr(
        fit_module,
        "_dependency_identity",
        lambda *_args: dependencies,
    )
    output_root = tmp_path / "output"

    with pytest.raises(CalibrationFitError, match="changed during fitting"):
        run_fit_task(
            0,
            dataset_path=dataset,
            catalog_path=catalog,
            output_root=output_root,
            repo_root=repo,
            fit_callable=lambda *_args: _parameters(),
            identity_provider=lambda *_args: FitInputs(identity=identity, train=object()),
        )
    assert not fit_artifact_paths(output_root, 0).directory.exists()


def test_overwrite_is_rejected_before_providers_are_called(tmp_path: Path) -> None:
    task = fit_task_for_index(0)
    dataset = (tmp_path / f"{task.dataset_stem}.npz").resolve()
    catalog = (tmp_path / "catalog.json").resolve()
    repo = (tmp_path / "repo").resolve()
    identity = _identity(
        tmp_path,
        dataset_path=dataset,
        catalog_path=catalog,
        repo_root=repo,
    )
    calls = 0

    def provider(*_args: object) -> FitInputs:
        nonlocal calls
        calls += 1
        return FitInputs(identity=identity, train=object())

    arguments = {
        "dataset_path": dataset,
        "catalog_path": catalog,
        "output_root": tmp_path / "output",
        "repo_root": repo,
        "fit_callable": lambda *_args: _parameters(),
        "identity_provider": provider,
        "postflight_validator": _noop_postflight,
    }
    run_fit_task(0, **arguments)
    assert calls == 1
    with pytest.raises(CalibrationFitError, match="overwrite"):
        run_fit_task(0, **arguments)
    assert calls == 1


def test_cli_exposes_only_identity_and_location_arguments(tmp_path: Path) -> None:
    parser = build_parser()
    assert {action.dest for action in parser._actions} == {
        "help",
        "task_index",
        "dataset",
        "catalog",
        "output_root",
    }
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--task-index",
                "0",
                "--dataset",
                "x.npz",
                "--catalog",
                "catalog.json",
                "--output-root",
                "out",
                "--seed",
                "29",
            ]
        )

    task = fit_task_for_index(0)
    dataset = (tmp_path / f"{task.dataset_stem}.npz").resolve()
    catalog = (tmp_path / "catalog.json").resolve()
    identity = _identity(
        tmp_path,
        dataset_path=dataset,
        catalog_path=catalog,
        repo_root=fit_module._REPO_ROOT,
    )
    assert (
        main(
            [
                "--task-index",
                "0",
                "--dataset",
                str(dataset),
                "--catalog",
                str(catalog),
                "--output-root",
                str(tmp_path / "cli-output"),
            ],
            fit_callable=lambda *_args: _parameters(),
            identity_provider=lambda *_args: FitInputs(identity=identity, train=object()),
            postflight_validator=_noop_postflight,
        )
        == 0
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "unexpected",
        "recipe",
        "task",
        "task_bool_alias",
        "catalog_hash",
        "catalog_schema_bool_alias",
        "bundle_replica_bool_alias",
        "data_access_bool_alias",
        "confirmatory_replica",
        "confirmatory_phase",
        "scheduler_mode",
        "scheduler_job_id",
        "dirty_source",
        "freeze_authority",
    ],
)
def test_payload_validator_fails_closed_on_schema_and_identity_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    payload = _payload(tmp_path)
    if mutation == "missing":
        payload.pop("parameters")
    elif mutation == "unexpected":
        payload["unexpected"] = True
    elif mutation == "recipe":
        payload["fit_recipe"]["train_steps"] = 21
    elif mutation == "task":
        payload["task_record"]["seed"] = 29
    elif mutation == "task_bool_alias":
        payload["task_record"]["replica"] = False
    elif mutation == "catalog_hash":
        payload["input_identity"]["catalog"]["sha256"] = "0" * 64
    elif mutation == "catalog_schema_bool_alias":
        payload["input_identity"]["catalog"]["schema_version"] = True
    elif mutation == "bundle_replica_bool_alias":
        payload["input_identity"]["bundle"]["replica"] = False
    elif mutation == "data_access_bool_alias":
        payload["data_access"]["prediction_or_scoring_performed"] = 0
    elif mutation == "confirmatory_replica":
        payload["input_identity"]["bundle"]["replica"] = 101
    elif mutation == "confirmatory_phase":
        payload["input_identity"]["bundle"]["phase"] = "confirmatory"
    elif mutation == "scheduler_mode":
        payload["provenance"]["scheduler"]["sharing_verification_mode"] = "asserted"
    elif mutation == "scheduler_job_id":
        payload["provenance"]["scheduler"]["job_id"] = "job-41001"
    elif mutation == "dirty_source":
        payload["provenance"]["source"]["status_porcelain"] = [" M pyproject.toml"]
    elif mutation == "freeze_authority":
        payload["freeze_ready"] = True
    with pytest.raises(CalibrationFitError):
        validate_fit_numeric_payload(payload)


def test_parameters_must_be_exact_binary32_values(tmp_path: Path) -> None:
    parameters = _parameters()
    parameters["outputscale"] = 1.1
    with pytest.raises(CalibrationFitError, match="binary32"):
        build_fit_numeric_payload(0, parameters, _identity(tmp_path))


def test_strict_json_rejects_duplicates_nonfinite_and_noncanonical_bytes(
    tmp_path: Path,
) -> None:
    payload, payload_bytes, envelope = build_fit_artifacts(
        0,
        _parameters(),
        _identity(tmp_path),
        output_root=tmp_path / "output",
    )
    duplicate = b'{"schema_version":"duplicate",' + payload_bytes[1:]
    with pytest.raises(CalibrationFitError):
        validate_fit_numeric_payload(duplicate)
    nonfinite = copy.deepcopy(payload)
    nonfinite["parameters"]["outputscale"] = float("nan")
    with pytest.raises(CalibrationFitError):
        validate_fit_numeric_payload(nonfinite)

    envelope_bytes = canonical_json_bytes(envelope)
    with pytest.raises(CalibrationFitError, match="not canonical"):
        validate_fit_artifact_pair(payload_bytes + b"\n", envelope_bytes)
    reversed_payload = {name: payload[name] for name in reversed(payload)}
    reordered_bytes = json.dumps(
        reversed_payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert reordered_bytes != payload_bytes
    with pytest.raises(CalibrationFitError, match="not canonical"):
        validate_fit_artifact_pair(reordered_bytes, envelope_bytes)
    with pytest.raises(CalibrationFitError, match="not canonical"):
        validate_fit_artifact_pair(payload_bytes, envelope_bytes + b"\n")


def test_pair_rejects_raw_hash_and_identity_derived_path_mismatch(tmp_path: Path) -> None:
    _, payload_bytes, envelope = build_fit_artifacts(
        0,
        _parameters(),
        _identity(tmp_path),
        output_root=tmp_path / "output",
    )
    wrong_hash = copy.deepcopy(envelope)
    wrong_hash["canonical_record"]["payload_binding"]["numeric_payload_sha256"] = "0" * 64
    wrong_hash["canonical_record_sha256"] = canonical_sha256(wrong_hash["canonical_record"])
    with pytest.raises(CalibrationFitError):
        validate_fit_artifact_pair(payload_bytes, canonical_json_bytes(wrong_hash))

    wrong_path = copy.deepcopy(envelope)
    wrong_path["canonical_record"]["payload_binding"]["numeric_payload_path"] = (
        "arbitrary/parameters.json"
    )
    wrong_path["canonical_record_sha256"] = canonical_sha256(wrong_path["canonical_record"])
    with pytest.raises(CalibrationFitError, match="identity-derived"):
        validate_fit_artifact_pair(payload_bytes, canonical_json_bytes(wrong_path))


def test_writer_cleans_reserved_directory_after_ordinary_io_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, payload_bytes, envelope = build_fit_artifacts(
        0,
        _parameters(),
        _identity(tmp_path),
        output_root=tmp_path / "output",
    )
    expected = fit_artifact_paths(tmp_path / "output", 0)
    original_open = Path.open

    def failing_open(path: Path, *args: object, **kwargs: object):
        if path == expected.execution_envelope:
            raise OSError("injected envelope write failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(CalibrationFitError, match="cannot write"):
        write_fit_artifacts_exclusive(
            tmp_path / "output",
            0,
            payload_bytes,
            envelope,
        )
    assert not expected.directory.exists()


def test_invalid_or_confirmatory_task_indices_are_not_resolvable(tmp_path: Path) -> None:
    for task_index in (False, True, -1, 45, 101):
        with pytest.raises(CalibrationFitError):
            fit_config_for_task_index(task_index)
        with pytest.raises(CalibrationFitError):
            fit_artifact_paths(tmp_path, task_index)


def _live_scontrol_record(task_index: int = 0) -> str:
    return (
        f"JobId=41001 ArrayJobId=41000 ArrayTaskId={task_index} "
        "ArrayTaskThrottle=1 Partition=short NumNodes=1 NumCPUs=8 CPUs/Task=8 "
        "MinMemoryNode=64G TimeLimit=08:00:00 "
        "OverSubscribe=OK NodeList=cpu-shared-01\n"
    )


def _mock_live_runtime_evidence(
    monkeypatch: pytest.MonkeyPatch,
    *,
    record: str | None = None,
    available_cpu_count: int = 32,
    cuda_available: bool = False,
    visible_gpu_count: int = 0,
) -> list[list[str]]:
    environment = {
        "SLURM_JOB_ID": "41001",
        "SLURM_ARRAY_JOB_ID": "41000",
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_ARRAY_TASK_COUNT": "45",
        "SLURM_ARRAY_TASK_MIN": "0",
        "SLURM_ARRAY_TASK_MAX": "44",
        "SLURM_ARRAY_TASK_STEP": "1",
        "SLURM_JOB_NODELIST": "cpu-shared-01",
        "SLURM_JOB_PARTITION": "short",
        # These former evidence variables are deliberate lies.  The production
        # observer must neither read nor turn them into provenance.
        "F02B_EXCLUSIVE_VERIFIED": "0",
        "F02B_EXCLUSIVE_MODE": "caller_asserted",
        "F02B_REQUESTED_GPU_COUNT": "99",
        "F02B_VISIBLE_GPU_COUNT": "1",
        "F02B_VISIBLE_GPU_MODELS_JSON": '["NVIDIA H100"]',
        "F02B_VISIBLE_GPU_MEMORY_BYTES_JSON": "[1]",
        "F02B_AVAILABLE_CPU_COUNT": "1",
        "F02B_AVAILABLE_HOST_MEMORY_BYTES": "1",
        "F02B_REQUESTED_WALLTIME_SECONDS": "1",
        "F02B_WALLTIME_LIMIT_SECONDS": "1",
        "F02B_ARRAY_CONCURRENCY": "99",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        calls.append(command)
        assert kwargs == {
            "check": True,
            "capture_output": True,
            "text": True,
            "timeout": 30,
        }
        return SimpleNamespace(stdout=record or _live_scontrol_record())

    monkeypatch.setattr(fit_module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        fit_module.os,
        "sched_getaffinity",
        lambda _pid: set(range(available_cpu_count)),
    )
    monkeypatch.setattr(fit_module.torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(fit_module.torch.cuda, "device_count", lambda: visible_gpu_count)
    return calls


def test_runtime_evidence_comes_from_live_scontrol_and_process_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _mock_live_runtime_evidence(monkeypatch)
    allocation, scheduler = fit_module._observe_runtime_evidence(fit_task_for_index(0))

    assert calls == [["scontrol", "show", "job", "41001", "-o"]]
    assert allocation == _runtime_allocation()
    assert scheduler == _identity(tmp_path)["scheduler"]

    identity = _identity(tmp_path)
    identity["runtime_allocation"] = allocation
    identity["scheduler"] = scheduler
    payload, _, envelope = build_fit_artifacts(
        0,
        _parameters(),
        identity,
        output_root=tmp_path / "output",
    )
    assert payload["provenance"]["scheduler"] == scheduler
    assert envelope["canonical_record"]["runtime_allocation"] == allocation
    assert scheduler["sharing_verification_mode"] == SHARING_VERIFICATION_MODE


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("JobId=41001", "JobId=99999"),
        ("ArrayJobId=41000", "ArrayJobId=99999"),
        ("ArrayTaskId=0", "ArrayTaskId=1"),
        ("ArrayTaskThrottle=1", "ArrayTaskThrottle=2"),
        ("OverSubscribe=OK", "OverSubscribe=EXCLUSIVE"),
        ("TimeLimit=08:00:00", "TimeLimit=04:00:00"),
        ("CPUs/Task=8", "CPUs/Task=16"),
        ("MinMemoryNode=64G", "MinMemoryNode=32G"),
        ("MinMemoryNode=64G", "TresPerNode=gres/gpu:l40s:1 MinMemoryNode=64G"),
        ("NumNodes=1", "NumNodes=2"),
        ("Partition=short", "Partition=gpu"),
        ("NodeList=cpu-shared-01", "NodeList=cpu-shared-02"),
    ],
)
def test_runtime_evidence_rejects_live_scontrol_drift(
    monkeypatch: pytest.MonkeyPatch,
    old: str,
    new: str,
) -> None:
    record = _live_scontrol_record().replace(old, new)
    _mock_live_runtime_evidence(monkeypatch, record=record)
    with pytest.raises(CalibrationFitError):
        fit_module._observe_runtime_evidence(fit_task_for_index(0))


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"available_cpu_count": 7}, "observed runtime allocation"),
        ({"cuda_available": True}, "zero CUDA devices"),
        ({"visible_gpu_count": 1}, "zero CUDA devices"),
        ({"cuda_available": 1}, "invalid values"),
    ],
)
def test_runtime_evidence_rejects_insufficient_process_hardware(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    error: str,
) -> None:
    _mock_live_runtime_evidence(monkeypatch, **kwargs)
    with pytest.raises(CalibrationFitError, match=error):
        fit_module._observe_runtime_evidence(fit_task_for_index(0))


def test_runtime_evidence_rejects_nonexact_array_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_live_runtime_evidence(monkeypatch)
    monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "44")
    with pytest.raises(CalibrationFitError, match="exact 0-44"):
        fit_module._observe_runtime_evidence(fit_task_for_index(0))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lengthscale", 1),
        ("outputscale", 2),
        ("sigma_f", 1),
        ("sigma_g", 0),
        ("lengthscale", -0.0),
        ("outputscale", -0.0),
        ("sigma_f", -0.0),
        ("sigma_g", -0.0),
    ],
)
def test_learned_parameters_require_canonical_json_floats(
    tmp_path: Path,
    field: str,
    value: int | float,
) -> None:
    parameters = _parameters()
    if field == "lengthscale":
        parameters[field] = [value]
    else:
        parameters[field] = value
    expected = "negative zero" if type(value) is float else "JSON float"
    with pytest.raises(CalibrationFitError, match=expected):
        build_fit_numeric_payload(0, parameters, _identity(tmp_path))


def test_authorization_and_loader_share_snapshot_across_transient_symlink_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = fit_task_for_index(0)
    dataset = (tmp_path / "corpus" / f"{task.dataset_stem}.npz").resolve()
    dataset.parent.mkdir()
    metadata = dataset.with_suffix(".metadata.json")
    manifest = dataset.with_suffix(".sha256.json")
    catalog = (tmp_path / "catalog.json").resolve()
    dataset_a = b"authenticated-dataset-A"
    metadata_a = canonical_json_bytes({"variant": "A"})
    manifest_a = canonical_json_bytes({"variant": "A"})
    catalog_a = canonical_json_bytes({"variant": "A"})
    dataset.write_bytes(dataset_a)
    metadata.write_bytes(metadata_a)
    manifest.write_bytes(manifest_a)
    catalog.write_bytes(catalog_a)

    alternate = tmp_path / "alternate" / dataset.name
    alternate.parent.mkdir()
    alternate.write_bytes(b"substituted-dataset-B")
    original_after_swap = tmp_path / "original-dataset-A"
    expected_config = asdict(
        ConfirmatoryConfig(
            n_particles=task.n_particles,
            n_dims=task.n_dims,
            replica=task.replica,
        )
    )
    file_hashes = {
        dataset.name: hashlib.sha256(dataset_a).hexdigest(),
        metadata.name: hashlib.sha256(metadata_a).hexdigest(),
    }
    identity = _identity(
        tmp_path,
        dataset_path=dataset,
        catalog_path=catalog,
        repo_root=(tmp_path / "repo").resolve(),
    )
    events: list[str] = []

    def preflight_snapshot(snapshot_dataset: Path) -> object:
        assert snapshot_dataset != dataset
        assert snapshot_dataset.name == dataset.name
        assert snapshot_dataset.read_bytes() == dataset_a
        assert snapshot_dataset.with_suffix(".metadata.json").read_bytes() == metadata_a
        assert snapshot_dataset.with_suffix(".sha256.json").read_bytes() == manifest_a
        events.append("preflight-A")
        return fit_module.BundleIdentity(
            dataset_path=snapshot_dataset,
            file_sha256=file_hashes,
            manifest_sha256=hashlib.sha256(manifest_a).hexdigest(),
            generator_config=expected_config,
        )

    def authorize_snapshot(snapshot_catalog: Path, preflight: object) -> object:
        assert snapshot_catalog != catalog
        assert snapshot_catalog.read_bytes() == catalog_a
        assert preflight.dataset_path == dataset
        events.append("authorize-A")
        return SimpleNamespace(
            catalog_sha256=F02_CATALOG_SHA256,
            generation_git_commit="1" * 40,
            generation_git_tree="2" * 40,
            bundle_entry={
                "task_index": 0,
                "phase": "development",
                "replica": task.replica,
                "n_particles": task.n_particles,
                "n_dims": task.n_dims,
                "D": task.dimension,
                "hashes": {"dataset_content_sha256": "d" * 64},
            },
        )

    time_index = np.arange(100, dtype=np.int64)
    prepared_train = PreparedConfirmatorySplit(
        name="train",
        source_indices=np.arange(100, dtype=np.int64),
        X=np.full((100, task.dimension), 17.0, dtype=np.float64),
        E=np.full(100, 19.0, dtype=np.float64),
        F=np.full((100, task.dimension), 23.0, dtype=np.float64),
        trajectory_id=np.zeros(100, dtype=np.int64),
        time_index=time_index,
        time_value=time_index.astype(np.float64) / 100.0,
    )

    def load_snapshot(snapshot_dataset: Path) -> object:
        assert snapshot_dataset.read_bytes() == dataset_a
        dataset.rename(original_after_swap)
        dataset.symlink_to(alternate)
        assert dataset.read_bytes() != dataset_a
        assert snapshot_dataset.read_bytes() == dataset_a
        events.append("load-A-after-swap")
        provenance = SimpleNamespace(
            dataset_path=snapshot_dataset,
            metadata_path=snapshot_dataset.with_suffix(".metadata.json"),
            sha256_manifest_path=snapshot_dataset.with_suffix(".sha256.json"),
            file_sha256=file_hashes,
            config_payload=expected_config,
        )
        return SimpleNamespace(
            loaded=SimpleNamespace(provenance=provenance),
            prepared=SimpleNamespace(train=prepared_train),
        )

    def preload(root: Path) -> None:
        assert root == Path(identity["source"]["repo_root"])
        events.append("preload")

    def source(root: Path) -> dict[str, Any]:
        assert events[-1] == "preload"
        events.append("source")
        return identity["source"]

    monkeypatch.setattr(fit_module, "_preflight_bundle_identity", preflight_snapshot)
    monkeypatch.setattr(fit_module, "validate_catalog_identity", authorize_snapshot)
    monkeypatch.setattr(fit_module, "load_prepared_confirmatory_bundle", load_snapshot)
    monkeypatch.setattr(fit_module, "_preload_default_fit_implementation", preload)
    monkeypatch.setattr(fit_module, "_source_identity", source)
    monkeypatch.setattr(
        fit_module,
        "_dependency_identity",
        lambda *_args: identity["dependencies"],
    )
    monkeypatch.setattr(fit_module, "_runtime_identity", lambda: identity["runtime"])
    monkeypatch.setattr(
        fit_module,
        "_observe_runtime_evidence",
        lambda *_args: (identity["runtime_allocation"], identity["scheduler"]),
    )

    inputs = fit_module.default_identity_provider(
        task,
        replace(fit_config_for_task_index(0), device="cpu"),
        dataset,
        catalog,
        Path(identity["source"]["repo_root"]),
    )
    assert events == ["preload", "source", "preflight-A", "authorize-A", "load-A-after-swap"]
    assert inputs.identity["bundle"]["file_sha256"] == file_hashes
    assert inputs.identity["bundle"]["dataset_path"] == str(dataset)
    assert inputs.train.X.unique().tolist() == [17.0]
    assert inputs.train.value.unique().tolist() == [19.0]
    assert inputs.train.gradient.unique().tolist() == [23.0]
    assert dataset.is_symlink()
    assert dataset.resolve() == alternate
