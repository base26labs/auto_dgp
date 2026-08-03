from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data.generate_nbody_confirmatory import ConfirmatoryConfig, TrainNormalization
from data.load_nbody_confirmatory import (
    PreparedConfirmatoryBundle,
    PreparedConfirmatoryDataset,
    PreparedConfirmatorySplit,
)
from experiments import f02_internal_task as task
from experiments.f02_design import (
    EVALUATION_TIME_INDICES,
    OPTIMIZER_SELECTION_TIME_INDICES,
    TRAIN_TIME_INDICES,
)
from experiments.f02_internal_models import FrozenTERAParameters, ScalarPrediction


def _split(name: str, trajectories: tuple[int, ...], dimension: int = 4):
    rows = [(trajectory, time) for trajectory in trajectories for time in range(100)]
    trajectory = np.asarray([row[0] for row in rows], dtype=np.int64)
    time = np.asarray([row[1] for row in rows], dtype=np.int64)
    source = trajectory * 100 + time
    base = source.astype(np.float64) / 1000.0
    X = np.stack([base + 0.01 * axis for axis in range(dimension)], axis=1)
    E = np.sin(base) + 0.01 * trajectory
    F = np.stack([np.cos(base) / (axis + 1) for axis in range(dimension)], axis=1)
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


@pytest.fixture
def task_config() -> task.InternalTaskConfig:
    return task.InternalTaskConfig(
        training_m=3,
        train_steps=20,
        train_epochs=0,
        lengthscale=0.9,
        outputscale=1.2,
        sigma_f=0.2,
        sigma_g=0.1,
        candidate_m=(75,),
        cg_tolerance=1e-8,
    )


@pytest.fixture
def prepared_bundle(tmp_path) -> PreparedConfirmatoryBundle:
    config = ConfirmatoryConfig(
        n_particles=2,
        n_dims=3,
        n_trajectories=100,
        steps_per_trajectory=100,
        replica=0,
    )
    prepared = PreparedConfirmatoryDataset(
        train=_split("train", (0, 1, 2), dimension=12),
        validation=_split("validation", (3, 4), dimension=12),
        test=_split("test", (5, 6), dimension=12),
        normalization=TrainNormalization(
            x_min=np.zeros(12),
            x_span=np.ones(12),
            energy_mean=0.0,
            energy_std=1.0,
            gradient_scale=np.ones(12),
        ),
        masses=np.ones(2),
    )
    stem = "nbody_fixedmass_n2_d3_replica0"
    provenance = SimpleNamespace(
        dataset_path=tmp_path / f"{stem}.npz",
        metadata_path=tmp_path / f"{stem}.metadata.json",
        sha256_manifest_path=tmp_path / f"{stem}.sha256.json",
        file_sha256={
            f"{stem}.npz": "a" * 64,
            f"{stem}.metadata.json": "b" * 64,
        },
        config_payload=asdict(config),
    )
    provenance.sha256_manifest_path.write_text("{}\n")
    loaded = SimpleNamespace(
        dataset=SimpleNamespace(config=config),
        provenance=provenance,
    )
    return PreparedConfirmatoryBundle(loaded=loaded, prepared=prepared)


@pytest.fixture
def catalog_authorization(tmp_path, prepared_bundle) -> task.CatalogAuthorization:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text("{}\n")
    return task.CatalogAuthorization(
        catalog_path=catalog_path,
        catalog_sha256="c" * 64,
        generation_git_commit="1" * 40,
        generation_git_tree="2" * 40,
        bundle_entry={
            "task_index": 17,
            "phase": "development",
            "hashes": {"dataset_content_sha256": "d" * 64},
        },
    )


def _orbit_details(rows: int, dtype: torch.dtype):
    return SimpleNamespace(
        ranks=torch.full((rows,), 2, dtype=torch.long),
        iterations=torch.full((rows,), 3, dtype=torch.long),
        operator_matvecs=torch.full((rows,), 4, dtype=torch.long),
        preconditioner_applications=torch.full((rows,), 3, dtype=torch.long),
        relative_residuals=torch.full((rows,), 1e-10, dtype=dtype),
        converged=torch.ones(rows, dtype=torch.bool),
        variance_error_upper_bounds=torch.full((rows,), 1e-12, dtype=dtype),
        expected_kl_upper_bounds=torch.full((rows,), 1e-12, dtype=dtype),
        exact_arithmetic_certified=torch.ones(rows, dtype=torch.bool),
        floating_point_rigorous=torch.zeros(rows, dtype=torch.bool),
        basis_exact=torch.ones(rows, dtype=torch.bool),
        finite_precision_variance_corrections=torch.zeros(rows, dtype=dtype),
    )


@pytest.fixture
def mocked_models(monkeypatch, prepared_bundle, catalog_authorization):
    calls = {
        "fit": [],
        "freeze": 0,
        "tera_m": [],
        "orbit_m": [],
        "value_m": [],
    }
    parameters = FrozenTERAParameters(
        lengthscale=torch.tensor([0.9], dtype=torch.float64),
        outputscale=1.2,
        sigma_f=0.2,
        sigma_g=0.1,
        kernel="rbf",
    )

    monkeypatch.setattr(task, "load_prepared_confirmatory_bundle", lambda path: prepared_bundle)
    monkeypatch.setattr(
        task,
        "_preflight_bundle_identity",
        lambda path: task._bundle_identity(prepared_bundle),
    )
    monkeypatch.setattr(
        task,
        "validate_catalog_identity",
        lambda path, identity: catalog_authorization,
    )
    monkeypatch.setattr(task, "_assert_repository_clean", lambda root: None)

    def fake_fit(train, **kwargs):
        calls["fit"].append((train, kwargs))
        return object()

    def fake_freeze(model):
        calls["freeze"] += 1
        return parameters

    def prediction(x_eval, *, orbit=False, released_tera=False):
        rows = x_eval.shape[0]
        latent = torch.full((rows,), 0.8, dtype=x_eval.dtype, device=x_eval.device)
        mean = 0.1 * x_eval[:, 0]
        return ScalarPrediction(
            mean=mean,
            latent_variance=latent,
            observation_variance=latent + parameters.sigma_f,
            details=_orbit_details(rows, x_eval.dtype) if orbit else None,
            released_variance_epsilon_floor=(
                torch.finfo(x_eval.dtype).eps if released_tera else None
            ),
            released_variance_epsilon_floor_inactive=(True if released_tera else None),
        )

    def fake_tera(train, x_eval, frozen, *, m):
        calls["tera_m"].append(m)
        return prediction(x_eval, released_tera=True)

    def fake_orbit(train, x_eval, frozen, *, m, **kwargs):
        calls["orbit_m"].append(m)
        return prediction(x_eval, orbit=True)

    def fake_value(train, x_eval, frozen, *, m):
        calls["value_m"].append(m)
        return prediction(x_eval)

    monkeypatch.setattr(task, "fit_released_tera", fake_fit)
    monkeypatch.setattr(task, "freeze_tera_parameters", fake_freeze)
    monkeypatch.setattr(task, "predict_released_tera", fake_tera)
    monkeypatch.setattr(task, "predict_orbit", fake_orbit)
    monkeypatch.setattr(task, "predict_value_only_local_gp", fake_value)
    monkeypatch.setattr(
        task,
        "_collect_provenance",
        lambda bundle, config, *, repo_root: {"mocked": True},
    )
    return calls


def test_default_primary_validation_is_train_only_and_json_serializable(
    tmp_path,
    task_config,
    mocked_models,
) -> None:
    output = tmp_path / "result.json"

    result = task.run_internal_task(
        tmp_path / "fixedmass.npz",
        catalog_path=tmp_path / "catalog.json",
        config=task_config,
        output_path=output,
    )

    assert result["schema_version"] == task.RESULT_SCHEMA_VERSION
    assert result["evaluation"] == {
        "split": "validation",
        "design": "primary",
        "time_indices": list(EVALUATION_TIME_INDICES),
        "test_gate": {
            "required": False,
            "validated": False,
            "committed_at_head": False,
            "path": None,
            "payload_sha256": None,
            "schema_version": None,
        },
    }
    assert len(mocked_models["fit"]) == 1
    fitted_split, fit_kwargs = mocked_models["fit"][0]
    assert fitted_split.name == "train"
    assert fitted_split.X.shape[0] == 3 * len(TRAIN_TIME_INDICES)
    assert set(fitted_split.trajectory_id.tolist()) == {0, 1, 2}
    assert result["training"]["time_indices"] == list(TRAIN_TIME_INDICES)
    assert result["task_config"]["candidate_m"] == [75]
    assert fit_kwargs["training_m"] == task_config.training_m
    assert fit_kwargs["train_steps"] == 20
    assert mocked_models["freeze"] == 1
    assert mocked_models["tera_m"] == [task.REFERENCE_M]
    assert mocked_models["orbit_m"] == [task.REFERENCE_M, 75]
    assert mocked_models["value_m"] == [task.REFERENCE_M]

    assert set(result["arms"]) == {
        "TERA-50",
        "ORBIT-50",
        "ORBIT-75",
        "value-only-conditional-50",
    }
    control = result["arms"]["value-only-conditional-50"]
    assert control["hyperparameters_source"] == "TERA-gradient-fit"
    assert "not a standalone value-only" in control["control_semantics"]
    assert result["arms"]["ORBIT-75"]["solver"]["all_converged"] is True
    assert result["arms"]["ORBIT-75"]["raw_prediction_checks"]["all_valid"] is True
    assert result["arms"]["ORBIT-50"]["same_m_agreement_to_TERA_50"] == {
        "maxabs_mean": 0.0,
        "maxabs_latent_variance": 0.0,
        "absolute_tolerance": 1e-4,
        "passes": True,
    }
    assert result["arms"]["TERA-50"]["raw_prediction_checks"][
        "released_variance_epsilon_floor"
    ] == {
        "value": torch.finfo(torch.float32).eps,
        "inactive": True,
        "failure_policy": "equality-to-floor-fails-before-scoring",
    }
    assert (
        result["arms"]["ORBIT-75"]["analytic_resources"]["per_target"][0]["operator_matvecs"] == 4
    )
    assert result["training"]["fit_seconds_descriptive"] >= 0.0
    assert result["training"]["optimizer_updates"] == 20
    assert result["training"]["effective_batch_size"] == 75
    assert result["training"]["vecchia_target_factors_processed"] == 1500
    assert result["arms"]["TERA-50"]["prediction_seconds_descriptive"] >= 0.0
    assert json.loads(output.read_text()) == result


def test_optimizer_selection_is_a_closed_registered_validation_design(
    tmp_path,
    task_config,
    mocked_models,
) -> None:
    result = task.run_internal_task(
        tmp_path / "fixedmass.npz",
        catalog_path=tmp_path / "catalog.json",
        config=replace(task_config, candidate_m=()),
        evaluation_design="optimizer_selection",
    )

    assert result["evaluation"]["time_indices"] == list(OPTIMIZER_SELECTION_TIME_INDICES)
    assert result["corpus"]["evaluation_rows"] == 2
    with pytest.raises(ValueError, match="primary.*optimizer_selection"):
        task.run_internal_task(
            tmp_path / "fixedmass.npz",
            catalog_path=tmp_path / "catalog.json",
            config=task_config,
            evaluation_design="arbitrary",
        )


@pytest.mark.parametrize("candidate_m", ((51,), (76,), (75.0,), (75, 250)))
def test_task_config_rejects_candidate_m_outside_preregistered_grid(candidate_m) -> None:
    with pytest.raises(ValueError, match="preregistered grid|must be integers"):
        task.InternalTaskConfig(candidate_m=candidate_m)


def test_task_config_rejects_looser_than_preregistered_cg_tolerance() -> None:
    with pytest.raises(ValueError, match=r"\(0, 1e-05\]"):
        task.InternalTaskConfig(cg_tolerance=1.0001e-5)


def test_test_access_is_locked_before_selection_or_fit(
    monkeypatch,
    tmp_path,
    task_config,
    prepared_bundle,
) -> None:
    selected_calls = []
    fit_calls = []
    monkeypatch.setattr(task, "load_prepared_confirmatory_bundle", lambda path: prepared_bundle)
    monkeypatch.setattr(
        task,
        "_selected_bundle",
        lambda *args, **kwargs: selected_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(task, "fit_released_tera", lambda *args, **kwargs: fit_calls.append(args))

    with pytest.raises(task.FrozenRecipeError, match="requires a committed"):
        task.run_internal_task(
            tmp_path / "fixedmass.npz",
            catalog_path=tmp_path / "catalog.json",
            config=task_config,
            evaluation_split="test",
        )
    assert selected_calls == []
    assert fit_calls == []

    with pytest.raises(task.FrozenRecipeError, match="only.*primary"):
        task.run_internal_task(
            tmp_path / "fixedmass.npz",
            catalog_path=tmp_path / "catalog.json",
            config=task_config,
            evaluation_split="test",
            evaluation_design="optimizer_selection",
            frozen_recipe_path=tmp_path / "recipe.json",
        )
    assert selected_calls == []
    assert fit_calls == []

    with pytest.raises(task.FrozenRecipeError, match="exactly one selected"):
        task.run_internal_task(
            tmp_path / "fixedmass.npz",
            catalog_path=tmp_path / "catalog.json",
            config=replace(task_config, candidate_m=(75, 100)),
            evaluation_split="test",
            frozen_recipe_path=tmp_path / "recipe.json",
        )
    assert selected_calls == []
    assert fit_calls == []


def _write_recipe(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def test_valid_per_bundle_recipe_still_cannot_unlock_confirmatory_test(
    monkeypatch,
    tmp_path,
    task_config,
    prepared_bundle,
    catalog_authorization,
    mocked_models,
) -> None:
    confirmatory_catalog = replace(
        catalog_authorization,
        bundle_entry={**catalog_authorization.bundle_entry, "phase": "confirmatory"},
    )
    monkeypatch.setattr(
        task,
        "validate_catalog_identity",
        lambda path, identity: confirmatory_catalog,
    )
    recipe_path = tmp_path / "recipe.json"
    document = task.build_frozen_recipe_document(
        prepared_bundle,
        task_config,
        confirmatory_catalog,
    )
    _write_recipe(recipe_path, document)
    monkeypatch.setattr(task, "_assert_recipe_committed", lambda path, root: None)

    with pytest.raises(task.FrozenRecipeError, match="global recipe.*one-release ledger"):
        task.run_internal_task(
            tmp_path / "fixedmass.npz",
            catalog_path=tmp_path / "catalog.json",
            config=task_config,
            evaluation_split="test",
            frozen_recipe_path=recipe_path,
        )
    assert mocked_models["fit"] == []


@pytest.mark.parametrize("tamper", ("schema", "hash", "config", "catalog"))
def test_recipe_schema_hash_and_configuration_mismatches_fail_closed(
    monkeypatch,
    tmp_path,
    task_config,
    prepared_bundle,
    catalog_authorization,
    tamper,
) -> None:
    confirmatory_catalog = replace(
        catalog_authorization,
        bundle_entry={**catalog_authorization.bundle_entry, "phase": "confirmatory"},
    )
    recipe_path = tmp_path / f"recipe-{tamper}.json"
    document = task.build_frozen_recipe_document(
        prepared_bundle,
        task_config,
        confirmatory_catalog,
    )
    if tamper == "schema":
        document["schema_version"] = "wrong"
    elif tamper == "hash":
        document["payload_sha256"] = "0" * 64
    else:
        if tamper == "config":
            document["payload"]["task_config"]["seed"] += 1
        else:
            document["payload"]["catalog"]["sha256"] = "9" * 64
        canonical = json.dumps(
            document["payload"],
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        document["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    _write_recipe(recipe_path, document)
    monkeypatch.setattr(task, "_assert_recipe_committed", lambda path, root: None)
    monkeypatch.setattr(task, "_assert_repository_clean", lambda root: None)
    monkeypatch.setattr(
        task,
        "_preflight_bundle_identity",
        lambda path: task._bundle_identity(prepared_bundle),
    )
    monkeypatch.setattr(
        task,
        "validate_catalog_identity",
        lambda path, identity: confirmatory_catalog,
    )
    fit_calls = []
    load_calls = []
    monkeypatch.setattr(
        task,
        "load_prepared_confirmatory_bundle",
        lambda path: load_calls.append(path),
    )
    monkeypatch.setattr(task, "fit_released_tera", lambda *args, **kwargs: fit_calls.append(args))

    with pytest.raises(task.FrozenRecipeError):
        task.run_internal_task(
            tmp_path / "fixedmass.npz",
            catalog_path=tmp_path / "catalog.json",
            config=task_config,
            evaluation_split="test",
            frozen_recipe_path=recipe_path,
        )
    assert fit_calls == []
    assert load_calls == []


def test_catalog_phase_separates_development_tuning_from_confirmatory_test(
    monkeypatch,
    tmp_path,
    task_config,
    catalog_authorization,
    mocked_models,
) -> None:
    confirmatory_catalog = replace(
        catalog_authorization,
        bundle_entry={**catalog_authorization.bundle_entry, "phase": "confirmatory"},
    )
    monkeypatch.setattr(
        task,
        "validate_catalog_identity",
        lambda path, identity: confirmatory_catalog,
    )
    with pytest.raises(task.InternalTaskError, match="validation.*development"):
        task.run_internal_task(
            tmp_path / "fixedmass.npz",
            catalog_path=tmp_path / "catalog.json",
            config=task_config,
        )
    assert mocked_models["fit"] == []

    monkeypatch.setattr(
        task,
        "validate_catalog_identity",
        lambda path, identity: catalog_authorization,
    )
    with pytest.raises(task.InternalTaskError, match="test.*confirmatory"):
        task.run_internal_task(
            tmp_path / "fixedmass.npz",
            catalog_path=tmp_path / "catalog.json",
            config=task_config,
            evaluation_split="test",
            frozen_recipe_path=tmp_path / "not-read.json",
        )
    assert mocked_models["fit"] == []


def test_released_paper_lengthscale_initialization_is_the_default(tmp_path) -> None:
    config = task.InternalTaskConfig()
    assert config.lengthscale == 1.0
    assert task._fit_kwargs(config)["lengthscale"] == 1.0

    args = task.build_parser().parse_args(
        [
            str(tmp_path / "bundle.npz"),
            "--catalog",
            str(tmp_path / "catalog.json"),
            "--out",
            str(tmp_path / "result.json"),
        ]
    )
    assert args.lengthscale == 1.0


def test_uncommitted_recipe_is_rejected(
    tmp_path,
    task_config,
    prepared_bundle,
    catalog_authorization,
) -> None:
    recipe_path = tmp_path / "uncommitted.json"
    _write_recipe(
        recipe_path,
        task.build_frozen_recipe_document(
            prepared_bundle,
            task_config,
            catalog_authorization,
        ),
    )

    with pytest.raises(task.FrozenRecipeError, match="inside the repository"):
        task.validate_frozen_recipe(
            recipe_path,
            prepared_bundle,
            task_config,
            catalog_authorization,
        )


def _ready_catalog(identity: task.BundleIdentity) -> dict:
    bundles = []
    for replica in task.F02_REPLICAS:
        for n_particles in task.F02_PARTICLE_COUNTS:
            task_index = len(bundles)
            stem = f"nbody_fixedmass_n{n_particles}_d3_replica{replica}"
            selected = replica == 0 and n_particles == 2
            dataset_path = (
                identity.dataset_path if selected else identity.dataset_path.parent / f"{stem}.npz"
            )
            metadata_path = dataset_path.with_suffix(".metadata.json")
            manifest_path = dataset_path.with_suffix(".sha256.json")
            bundles.append(
                {
                    "task_index": task_index,
                    "phase": "development"
                    if replica in task.F02_DEVELOPMENT_REPLICAS
                    else "confirmatory",
                    "replica": replica,
                    "n_particles": n_particles,
                    "n_dims": 3,
                    "D": 6 * n_particles,
                    "paths": {
                        "dataset": str(dataset_path),
                        "metadata": str(metadata_path),
                        "sha256_manifest": str(manifest_path),
                    },
                    "hashes": {
                        "dataset_file_sha256": (
                            identity.file_sha256[identity.dataset_path.name]
                            if selected
                            else f"{1000 + task_index:064x}"
                        ),
                        "metadata_file_sha256": (
                            identity.file_sha256[metadata_path.name]
                            if selected
                            else f"{2000 + task_index:064x}"
                        ),
                        "sha256_manifest_file_sha256": (
                            identity.manifest_sha256 if selected else f"{3000 + task_index:064x}"
                        ),
                        "dataset_content_sha256": f"{4000 + task_index:064x}",
                    },
                    "config": asdict(
                        ConfirmatoryConfig(
                            n_particles=n_particles,
                            n_dims=3,
                            replica=replica,
                        )
                    ),
                    "unique_content": True,
                    "eligible_for_catalog": True,
                }
            )
    return {
        "schema_version": 1,
        "catalog_type": "f02_nbody_confirmatory_data",
        "overall_ready": True,
        "input": {
            "replicas": list(task.F02_REPLICAS),
            "development_replicas": list(task.F02_DEVELOPMENT_REPLICAS),
            "particle_counts": list(task.F02_PARTICLE_COUNTS),
            "n_dims": 3,
            "generation": {
                "n_trajectories": 100,
                "steps_per_trajectory": 100,
                "dt": 0.01,
                "mass_seed": 1729,
                "trajectory_seed": 2718,
                "split_seed": 31415,
                "validation_seed": 1618,
            },
        },
        "task_accounting": {
            "expected_task_count": 65,
            "all_expected_tasks_valid": True,
            "all_expected_tasks_unique": True,
            "status_counts": {
                "valid": 65,
                "missing": 0,
                "failed": 0,
                "invalid": 0,
                "unexpected": 0,
            },
        },
        "provenance": {
            "verified": True,
            "same_commit": True,
            "same_tree": True,
            "same_submodules": True,
            "same_source_hashes": True,
            "same_source_manifest": True,
            "same_slurm_array_job": True,
            "same_repo_root": True,
            "git_commit": "1" * 40,
            "git_tree": "2" * 40,
        },
        "independence": {
            "verified": True,
            "candidate_bundle_count": 65,
            "unique_bundle_count": 65,
            "duplicate_dataset_content_sha256": [],
        },
        "catalog": {
            "bundle_count": 65,
            "phase_counts": {"development": 15, "confirmatory": 50},
            "bundles": bundles,
        },
    }


def test_catalog_must_list_one_eligible_hash_and_config_matching_bundle(
    tmp_path,
    prepared_bundle,
) -> None:
    identity = task._bundle_identity(prepared_bundle)
    catalog_path = tmp_path / "strict-catalog.json"
    document = _ready_catalog(identity)
    catalog_path.write_text(json.dumps(document))

    authorization = task.validate_catalog_identity(catalog_path, identity)
    assert authorization.bundle_entry["task_index"] == 0
    assert authorization.generation_git_commit == "1" * 40

    document["catalog"]["bundles"][0]["eligible_for_catalog"] = False
    catalog_path.write_text(json.dumps(document))
    with pytest.raises(task.InternalTaskError, match="frozen grid|not eligible"):
        task.validate_catalog_identity(catalog_path, identity)

    document = _ready_catalog(identity)
    document["catalog"]["bundles"][0]["paths"]["dataset"] = str(
        tmp_path / "unlisted" / identity.dataset_path.name
    )
    catalog_path.write_text(json.dumps(document))
    with pytest.raises(task.InternalTaskError, match="exactly once"):
        task.validate_catalog_identity(catalog_path, identity)


def test_catalog_rejects_reduced_grid_or_phase_relabelling(tmp_path, prepared_bundle) -> None:
    identity = task._bundle_identity(prepared_bundle)
    catalog_path = tmp_path / "tampered-catalog.json"

    document = _ready_catalog(identity)
    document["task_accounting"]["expected_task_count"] = 1
    document["task_accounting"]["status_counts"]["valid"] = 1
    document["catalog"]["bundle_count"] = 1
    document["catalog"]["bundles"] = document["catalog"]["bundles"][:1]
    catalog_path.write_text(json.dumps(document))
    with pytest.raises(task.InternalTaskError, match="exactly 65"):
        task.validate_catalog_identity(catalog_path, identity)

    document = _ready_catalog(identity)
    document["catalog"]["bundles"][0]["phase"] = "confirmatory"
    catalog_path.write_text(json.dumps(document))
    with pytest.raises(task.InternalTaskError, match="frozen grid"):
        task.validate_catalog_identity(catalog_path, identity)


@pytest.mark.parametrize("failure", ("converged", "residual", "basis", "variance"))
def test_orbit_validity_failures_never_produce_complete_results(
    monkeypatch,
    tmp_path,
    task_config,
    mocked_models,
    failure,
) -> None:
    original = task.predict_orbit

    def invalid_orbit(*args, **kwargs):
        prediction = original(*args, **kwargs)
        if failure == "converged":
            prediction.details.converged[0] = False
        elif failure == "residual":
            prediction.details.relative_residuals[0] = 1.0
        elif failure == "basis":
            prediction.details.basis_exact[0] = False
        else:
            prediction.latent_variance[0] = -1.0
            prediction.observation_variance[0] = -0.8
        return prediction

    monkeypatch.setattr(task, "predict_orbit", invalid_orbit)
    with pytest.raises(task.InternalTaskError):
        task.run_internal_task(
            tmp_path / "fixedmass.npz",
            catalog_path=tmp_path / "catalog.json",
            config=task_config,
        )


def test_same_m_control_uses_dtype_threshold_and_fails_closed(
    monkeypatch,
    tmp_path,
    task_config,
    mocked_models,
) -> None:
    float64_result = task.run_internal_task(
        tmp_path / "fixedmass.npz",
        catalog_path=tmp_path / "catalog.json",
        config=replace(task_config, dtype="float64"),
    )
    assert (
        float64_result["arms"]["ORBIT-50"]["same_m_agreement_to_TERA_50"]["absolute_tolerance"]
        == 1e-6
    )

    original = task.predict_orbit

    def disagreeing_orbit(*args, **kwargs):
        prediction = original(*args, **kwargs)
        if kwargs["m"] == task.REFERENCE_M:
            prediction.mean[0] += 2e-4
        return prediction

    monkeypatch.setattr(task, "predict_orbit", disagreeing_orbit)
    with pytest.raises(task.InternalTaskError, match="same-m agreement"):
        task.run_internal_task(
            tmp_path / "fixedmass.npz",
            catalog_path=tmp_path / "catalog.json",
            config=task_config,
        )
