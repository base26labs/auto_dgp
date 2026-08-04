"""Focused fail-closed tests for the F02 global recipe and one-release ledger."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from cluster.f02_confirmatory_ledger import (
    RELEASE_MUTATION_DISABLED_REASON,
    RELEASE_MUTATIONS_ENABLED,
    RESULT_ATTESTATION_SCHEMA_VERSION,
    ReleaseLedgerError,
    _authorization_payload,
    _new_ledger,
    _read_json_snapshot,
    _read_regular_file_snapshot,
    _recipe_bundle_for_task,
    _release_marker_document,
    _validate_result_for_release,
    _validate_success_attestation,
    audit_release,
    authorize_release,
    begin_attempt,
    catalog_release_marker_path,
    finish_attempt,
    seal_release,
)
from data.generate_nbody_confirmatory import ConfirmatoryConfig
from experiments.f02_external_adapter import ExternalBaselineConfig
from experiments.f02_global_release import (
    ALL_REPLICAS,
    DEVELOPMENT_EVIDENCE_SEMANTICALLY_VERIFIED,
    DIMENSIONS,
    EXPECTED_TASK_COUNT,
    METHOD_IDS,
    METHOD_SELECTION_SCHEMA_VERSION,
    PARTICLE_COUNTS,
    RECIPE_SCAFFOLD_STATUS,
    GlobalRecipeError,
    build_global_recipe_document,
    canonical_json_bytes,
    parse_method_selection,
    sha256_bytes,
    validate_global_recipe_document,
)
from experiments.f02_internal_task import InternalTaskConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TEST_PROTOCOL_PATH = "docs/F02B_TEST_PROTOCOL.md"
_RENAMED_DRAFT_PROTOCOL_PATH = "docs/F02B_RENAMED_DRAFT.md"
_COPIED_TERMINATED_PROTOCOL_PATH = "docs/F02_TERMINATED_COPY.md"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _ready_catalog(path: Path) -> Path:
    bundles = []
    for replica in ALL_REPLICAS:
        for n_particles in PARTICLE_COUNTS:
            task_index = len(bundles)
            dimension = 6 * n_particles
            stem = f"nbody_fixedmass_n{n_particles}_d3_replica{replica}"
            bundles.append(
                {
                    "task_index": task_index,
                    "phase": "development" if replica in (0, 1, 2) else "confirmatory",
                    "replica": replica,
                    "n_particles": n_particles,
                    "n_dims": 3,
                    "D": dimension,
                    "paths": {
                        "dataset": str(path.parent / "data" / f"{stem}.npz"),
                        "metadata": str(path.parent / "data" / f"{stem}.metadata.json"),
                        "sha256_manifest": str(path.parent / "data" / f"{stem}.sha256.json"),
                    },
                    "hashes": {
                        "dataset_file_sha256": f"{1000 + task_index:064x}",
                        "metadata_file_sha256": f"{2000 + task_index:064x}",
                        "sha256_manifest_file_sha256": f"{3000 + task_index:064x}",
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
    document = {
        "schema_version": 1,
        "catalog_type": "f02_nbody_confirmatory_data",
        "overall_ready": True,
        "input": {
            "run_root": str(path.parent.resolve()),
            "replicas": list(ALL_REPLICAS),
            "development_replicas": [0, 1, 2],
            "particle_counts": list(PARTICLE_COUNTS),
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
            "repo_root": str(path.parent.parent.resolve()),
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
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


def _configuration_record(
    method_id: str,
    dimension: int,
    seed: int,
    updates: int,
    schedule: dict[int, int],
) -> dict[str, Any]:
    if method_id == "internal-shared-fit":
        payload = _jsonable(
            asdict(
                InternalTaskConfig(
                    train_steps=updates,
                    seed=seed,
                    candidate_m=(schedule[dimension],),
                    dtype="float32",
                    device="cuda",
                )
            )
        )
    else:
        payload = ExternalBaselineConfig(
            method_id=method_id,
            dimension=dimension,
            seed=seed,
            selected_updates=updates,
        ).to_payload()
    return {
        "configuration_id": f"D{dimension}-seed{seed}",
        "dimension": dimension,
        "seed": seed,
        "payload": payload,
        "sha256": sha256_bytes(canonical_json_bytes(payload)),
    }


def _selection(method_id: str, updates: int, schedule: dict[int, int]):
    configuration = {
        "by_dimension_and_seed": [
            _configuration_record(method_id, dimension, seed, updates, schedule)
            for dimension in DIMENSIONS
            for seed in (11, 29, 47)
        ]
    }
    document = {
        "schema_version": METHOD_SELECTION_SCHEMA_VERSION,
        "method_id": method_id,
        "optimizer_updates": updates,
        "configuration": configuration,
        "selection_evidence": [
            {
                "role": role,
                "sha256": hashlib.sha256(f"{method_id}:{role}".encode()).hexdigest(),
            }
            for role in (
                ("optimizer_selection_report", "orbit_resource_selection_report")
                if method_id == "internal-shared-fit"
                else ("optimizer_selection_report", "runtime_dependency_lock")
            )
        ],
        "orbit_m_by_dimension": (
            {str(key): value for key, value in schedule.items()}
            if method_id == "internal-shared-fit"
            else None
        ),
    }
    return parse_method_selection(document, method_id)


def _selections() -> dict[str, Any]:
    schedule = {12: 75, 24: 100, 36: 100, 48: 150, 60: 200}
    return {
        "internal-shared-fit": _selection("internal-shared-fit", 20, schedule),
        "dsoftki-512": _selection("dsoftki-512", 50, schedule),
        "ddsvgp-512": _selection("ddsvgp-512", 100, schedule),
    }


def _clone_source(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "source-repo"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-local", str(_REPO_ROOT), str(repo)],
        check=True,
        capture_output=True,
    )
    _git(repo, "config", "user.email", "f02-test@example.invalid")
    _git(repo, "config", "user.name", "F02 Test")
    (repo / _TEST_PROTOCOL_PATH).write_text(
        "# F02b unit-test protocol\n\nProtocol ID: F02_NBODY_PROTOCOL_unit_v1\n"
    )
    (repo / _RENAMED_DRAFT_PROTOCOL_PATH).write_text(
        "# Renamed protocol\n\nStatus: DRAFT and not executable.\n"
    )
    (repo / _COPIED_TERMINATED_PROTOCOL_PATH).write_bytes(
        (repo / "docs/F02_NBODY_PROTOCOL.md").read_bytes()
    )
    _git(
        repo,
        "add",
        _TEST_PROTOCOL_PATH,
        _RENAMED_DRAFT_PROTOCOL_PATH,
        _COPIED_TERMINATED_PROTOCOL_PATH,
    )
    _git(repo, "commit", "--quiet", "-m", "Add unit-test protocol source")
    return repo, _git(repo, "rev-parse", "HEAD")


def _write_recipe_and_release(
    tmp_path: Path,
    *,
    extra_release_file: bool = False,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    catalog = _ready_catalog(tmp_path / "catalog.json")
    repo, source_commit = _clone_source(tmp_path)
    recipe_relative = "releases/f02_global_confirmatory_recipe.json"
    document = build_global_recipe_document(
        catalog,
        repo_root=repo,
        source_commit=source_commit,
        recipe_path=recipe_relative,
        experiment_id="F02b-unit",
        protocol_id="F02_NBODY_PROTOCOL_unit_v1",
        protocol_path=_TEST_PROTOCOL_PATH,
        selections=_selections(),
    )
    recipe = repo / recipe_relative
    recipe.parent.mkdir(parents=True)
    recipe.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    _git(repo, "add", recipe_relative)
    if extra_release_file:
        (repo / "unexpected.txt").write_text("not recipe-only\n")
        _git(repo, "add", "unexpected.txt")
    _git(repo, "commit", "--quiet", "-m", "Add F02 recipe-only release")
    return repo, catalog, recipe, document


def test_global_recipe_is_deterministic_and_enumerates_exact_complete_grids(
    tmp_path: Path,
) -> None:
    catalog = _ready_catalog(tmp_path / "catalog.json")
    repo, source_commit = _clone_source(tmp_path)
    first = build_global_recipe_document(
        catalog,
        repo_root=repo,
        source_commit=source_commit,
        recipe_path="releases/f02_global_confirmatory_recipe.json",
        experiment_id="F02b-unit",
        protocol_id="F02_NBODY_PROTOCOL_unit_v1",
        protocol_path=_TEST_PROTOCOL_PATH,
        selections=_selections(),
    )
    second = build_global_recipe_document(
        catalog,
        repo_root=repo,
        source_commit=source_commit,
        recipe_path="releases/f02_global_confirmatory_recipe.json",
        experiment_id="F02b-unit",
        protocol_id="F02_NBODY_PROTOCOL_unit_v1",
        protocol_path=_TEST_PROTOCOL_PATH,
        selections=_selections(),
    )
    assert first == second
    payload = first["payload"]
    assert payload["experiment_id"] == "F02b-unit"
    assert payload["protocol_id"] == "F02_NBODY_PROTOCOL_unit_v1"
    assert payload["release_scaffold_status"] == RECIPE_SCAFFOLD_STATUS
    assert payload["development_evidence_semantically_verified"] is False
    assert DEVELOPMENT_EVIDENCE_SEMANTICALLY_VERIFIED is False
    assert payload["artifacts"]["protocol"]["path"] == _TEST_PROTOCOL_PATH
    assert payload["artifacts"]["catalog"]["path"] == str(catalog.resolve())
    assert len(payload["confirmatory_bundles"]) == 50
    assert [method["method_id"] for method in payload["methods"]] == list(METHOD_IDS)
    assert [len(grid["tasks"]) for grid in payload["expected_task_grids"]] == [150] * 3
    assert payload["accounting"]["expected_task_count"] == EXPECTED_TASK_COUNT == 450
    assert (
        len({task["task_id"] for grid in payload["expected_task_grids"] for task in grid["tasks"]})
        == 450
    )
    external_task = payload["expected_task_grids"][1]["tasks"][4]
    record = payload["methods"][1]["configuration"]["by_dimension_and_seed"][4]
    assert external_task["method_configuration_sha256"] == record["sha256"]
    summary = validate_global_recipe_document(first, catalog_path=catalog, repo_root=repo)
    assert summary["expected_task_count"] == 450
    assert summary["protocol_path"] == _TEST_PROTOCOL_PATH
    assert summary["structurally_valid"] is True
    assert summary["releasable"] is False
    assert summary["development_evidence_semantically_verified"] is False
    assert "valid" not in summary

    with pytest.raises(GlobalRecipeError, match="explicitly blocked"):
        build_global_recipe_document(
            catalog,
            repo_root=repo,
            source_commit=source_commit,
            recipe_path="releases/forbidden-original-f02.json",
            experiment_id="F02",
            protocol_id="F02_NBODY_PROTOCOL_v1",
            protocol_path="docs/F02_NBODY_PROTOCOL.md",
            selections=_selections(),
        )
    with pytest.raises(GlobalRecipeError, match="must not contain a DRAFT"):
        build_global_recipe_document(
            catalog,
            repo_root=repo,
            source_commit=source_commit,
            recipe_path="releases/forbidden-renamed-draft.json",
            experiment_id="F02b-renamed-draft",
            protocol_id="F02B_RENAMED_DRAFT",
            protocol_path=_RENAMED_DRAFT_PROTOCOL_PATH,
            selections=_selections(),
        )
    with pytest.raises(GlobalRecipeError, match="Git blob SHA-256 is explicitly blocked"):
        build_global_recipe_document(
            catalog,
            repo_root=repo,
            source_commit=source_commit,
            recipe_path="releases/forbidden-terminated-copy.json",
            experiment_id="F02-terminated-copy",
            protocol_id="F02_TERMINATED_COPY",
            protocol_path=_COPIED_TERMINATED_PROTOCOL_PATH,
            selections=_selections(),
        )
    with pytest.raises(GlobalRecipeError, match="explicitly blocked"):
        build_global_recipe_document(
            catalog,
            repo_root=repo,
            source_commit=source_commit,
            recipe_path="releases/forbidden-draft-f02b.json",
            experiment_id="F02b",
            protocol_id="F02B_NBODY_PROTOCOL_draft",
            protocol_path="docs/F02B_NBODY_PROTOCOL.md",
            selections=_selections(),
        )


def test_recipe_rejects_per_bundle_partial_method_task_and_catalog_grids(tmp_path: Path) -> None:
    repo, catalog, _, document = _write_recipe_and_release(tmp_path)

    with pytest.raises(GlobalRecipeError, match="schema_version"):
        validate_global_recipe_document(
            {
                "schema_version": "f02_frozen_recipe_v1",
                "payload": {"bundle": {}},
                "payload_sha256": "0" * 64,
            },
            catalog_path=catalog,
            repo_root=repo,
        )

    missing_method = copy.deepcopy(document)
    missing_method["payload"]["methods"].pop()
    missing_method["payload_sha256"] = sha256_bytes(canonical_json_bytes(missing_method["payload"]))
    with pytest.raises(GlobalRecipeError, match="methods must be exactly"):
        validate_global_recipe_document(missing_method, catalog_path=catalog, repo_root=repo)

    partial_tasks = copy.deepcopy(document)
    partial_tasks["payload"]["expected_task_grids"][0]["tasks"].pop()
    partial_tasks["payload_sha256"] = sha256_bytes(canonical_json_bytes(partial_tasks["payload"]))
    with pytest.raises(GlobalRecipeError, match="task grids are partial"):
        validate_global_recipe_document(partial_tasks, catalog_path=catalog, repo_root=repo)

    detached_protocol = copy.deepcopy(document)
    detached_protocol["payload"]["protocol_id"] = "detached-protocol-id"
    detached_protocol["payload_sha256"] = sha256_bytes(
        canonical_json_bytes(detached_protocol["payload"])
    )
    with pytest.raises(GlobalRecipeError, match="detached"):
        validate_global_recipe_document(
            detached_protocol,
            catalog_path=catalog,
            repo_root=repo,
        )

    detached_path = copy.deepcopy(document)
    detached_path["payload"]["artifacts"]["protocol"]["path"] = "docs/CONFIRMATORY_DATA.md"
    detached_path["payload_sha256"] = sha256_bytes(canonical_json_bytes(detached_path["payload"]))
    with pytest.raises(GlobalRecipeError, match="protocol hash"):
        validate_global_recipe_document(detached_path, catalog_path=catalog, repo_root=repo)

    catalog_copy = tmp_path / "catalog-copy.json"
    catalog_copy.write_bytes(catalog.read_bytes())
    with pytest.raises(GlobalRecipeError, match="canonical input.run_root/catalog.json"):
        validate_global_recipe_document(
            document,
            catalog_path=catalog_copy,
            repo_root=repo,
        )

    catalog_document = json.loads(catalog.read_text())
    catalog_document["catalog"]["bundles"][0]["phase"] = "confirmatory"
    catalog.write_text(json.dumps(catalog_document))
    with pytest.raises(GlobalRecipeError, match="grid/config/phase"):
        validate_global_recipe_document(document, catalog_path=catalog, repo_root=repo)


def test_selection_rejects_missing_configuration_and_schedule_coordinates() -> None:
    schedule = {12: 75, 24: 100, 36: 100, 48: 150, 60: 200}
    selection = _selection("dsoftki-512", 50, schedule)
    document = {
        "schema_version": METHOD_SELECTION_SCHEMA_VERSION,
        "method_id": selection.method_id,
        "optimizer_updates": selection.optimizer_updates,
        "configuration": copy.deepcopy(selection.configuration),
        "selection_evidence": list(selection.selection_evidence),
        "orbit_m_by_dimension": None,
    }
    document["configuration"]["by_dimension_and_seed"].pop()
    with pytest.raises(GlobalRecipeError, match="exactly 15"):
        parse_method_selection(document, "dsoftki-512")

    internal = copy.deepcopy(document)
    internal["method_id"] = "internal-shared-fit"
    internal["orbit_m_by_dimension"] = {"12": 75}
    with pytest.raises(GlobalRecipeError, match="cover exactly"):
        parse_method_selection(internal, "internal-shared-fit")


def test_all_public_release_mutations_are_source_disabled(
    tmp_path: Path,
) -> None:
    recipe = tmp_path / "missing-recipe.json"
    catalog = tmp_path / "missing-catalog.json"
    ledger = tmp_path / "would-be-ledger.json"
    disabled = "actual runner, semantic evidence validators, immutable result store not integrated"
    assert RELEASE_MUTATIONS_ENABLED is False
    assert disabled in RELEASE_MUTATION_DISABLED_REASON

    with pytest.raises(ReleaseLedgerError, match=disabled):
        authorize_release(recipe, catalog, ledger, repo_root=tmp_path)
    with pytest.raises(ReleaseLedgerError, match=disabled):
        begin_attempt(
            recipe,
            catalog,
            ledger,
            task_id="not-a-task",
            attempt_id="not-an-attempt",
            slurm_job_id="0",
            slurm_array_job_id="0",
            slurm_array_task_id=0,
            repo_root=tmp_path,
        )
    with pytest.raises(ReleaseLedgerError, match=disabled):
        finish_attempt(
            recipe,
            catalog,
            ledger,
            task_id="not-a-task",
            attempt_id="not-an-attempt",
            outcome="failed",
            exit_code=1,
            failure_code="disabled",
            repo_root=tmp_path,
        )
    with pytest.raises(ReleaseLedgerError, match=disabled):
        seal_release(recipe, catalog, ledger, repo_root=tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("subcommand", "extra_arguments"),
    (
        ("authorize", ()),
        (
            "begin-attempt",
            (
                "--task-id",
                "not-a-task",
                "--attempt-id",
                "not-an-attempt",
                "--slurm-job-id",
                "0",
                "--slurm-array-job-id",
                "0",
                "--slurm-array-task-id",
                "0",
            ),
        ),
        (
            "finish-attempt",
            (
                "--task-id",
                "not-a-task",
                "--attempt-id",
                "not-an-attempt",
                "--outcome",
                "failed",
                "--exit-code",
                "1",
                "--failure-code",
                "disabled",
            ),
        ),
        ("seal", ()),
    ),
)
def test_each_mutation_cli_is_source_disabled_without_writes(
    tmp_path: Path,
    subcommand: str,
    extra_arguments: tuple[str, ...],
) -> None:
    recipe = tmp_path / "missing-recipe.json"
    catalog = tmp_path / "missing-catalog.json"
    ledger = tmp_path / "would-be-ledger.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "cluster/f02_confirmatory_ledger.py"),
            subcommand,
            "--recipe",
            str(recipe),
            "--catalog",
            str(catalog),
            "--ledger",
            str(ledger),
            "--repo-root",
            str(tmp_path),
            *extra_arguments,
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert RELEASE_MUTATION_DISABLED_REASON in completed.stderr
    assert list(tmp_path.iterdir()) == []


def test_result_snapshots_are_single_descriptor_no_follow_reads(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    encoded = b'{"status":"fixture"}\n'
    result.write_bytes(encoded)
    snapshot, digest = _read_regular_file_snapshot(result, label="fixture result")
    assert snapshot == encoded
    assert digest == hashlib.sha256(encoded).hexdigest()
    document, json_digest = _read_json_snapshot(result, label="fixture result")
    assert document == {"status": "fixture"}
    assert json_digest == digest

    symlink = tmp_path / "result-link.json"
    symlink.symlink_to(result)
    with pytest.raises(ReleaseLedgerError, match="snapshot|symlink"):
        _read_regular_file_snapshot(symlink, label="fixture result symlink")


def test_audit_is_read_only_and_does_not_create_lock_or_marker(tmp_path: Path) -> None:
    repo, catalog, recipe, _ = _write_recipe_and_release(tmp_path)
    ledger_path = tmp_path / "manual-scaffold-ledger.json"
    _, _, authorization = _authorization_payload(
        recipe,
        catalog,
        repo_root=repo,
    )
    ledger = _new_ledger(authorization)
    marker_path = catalog_release_marker_path(catalog)
    marker = _release_marker_document(catalog, ledger_path, authorization, ledger)
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ReleaseLedgerError, match="catalog release marker"):
        audit_release(
            recipe,
            catalog,
            ledger_path,
            repo_root=repo,
        )
    assert not marker_path.exists()
    assert not ledger_path.with_name(f".{ledger_path.name}.lock").exists()
    assert not marker_path.with_name(f".{marker_path.name}.lock").exists()

    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")

    before_names = {path.name for path in tmp_path.iterdir()}
    before_bytes = {
        ledger_path: ledger_path.read_bytes(),
        marker_path: marker_path.read_bytes(),
    }
    summary = audit_release(
        recipe,
        catalog,
        ledger_path,
        repo_root=repo,
    )

    assert summary["structurally_consistent"] is True
    assert summary["releasable"] is False
    assert summary["release_mutations_enabled"] is False
    assert summary["confirmatory_execution_enabled_by_this_module"] is False
    assert {path.name for path in tmp_path.iterdir()} == before_names
    assert ledger_path.read_bytes() == before_bytes[ledger_path]
    assert marker_path.read_bytes() == before_bytes[marker_path]
    assert not ledger_path.with_name(f".{ledger_path.name}.lock").exists()
    assert not marker_path.with_name(f".{marker_path.name}.lock").exists()


def _scaffold_authorization(document: dict[str, Any]) -> dict[str, Any]:
    payload = document["payload"]
    protocol = payload["artifacts"]["protocol"]
    return {
        "release_id": "a" * 64,
        "experiment_id": payload["experiment_id"],
        "protocol_id": payload["protocol_id"],
        "protocol_path": protocol["path"],
        "protocol_sha256": protocol["sha256"],
        "protocol_binding_sha256": protocol["binding_sha256"],
        "recipe_payload_sha256": document["payload_sha256"],
        "recipe_file_sha256": "b" * 64,
        "source_commit": payload["release"]["source_commit"],
        "execution_commit": "c" * 40,
        "execution_tree": "d" * 40,
        "catalog_sha256": payload["artifacts"]["catalog"]["sha256"],
    }


def test_result_helpers_reject_missing_semantic_hash_and_attestation_bundle_mismatch(
    tmp_path: Path,
) -> None:
    repo, _, _, document = _write_recipe_and_release(tmp_path)
    del repo  # The helper checks immutable recipe content, not the live checkout.
    task = document["payload"]["expected_task_grids"][0]["tasks"][0]
    bundle = _recipe_bundle_for_task(document, task)
    authorization = _scaffold_authorization(document)
    artifacts = document["payload"]["artifacts"]
    recipe_catalog = artifacts["catalog"]
    config_payload = document["payload"]["methods"][0]["configuration"]["by_dimension_and_seed"][0][
        "payload"
    ]
    result_document = {
        "schema_version": "f02_internal_task_v1",
        "status": "complete",
        "task_config": config_payload,
        "training": {"optimizer_updates": task["optimizer_updates"]},
        "evaluation": {"split": "test", "design": "primary"},
        "corpus": {"replica": task["replica"], "dimension": task["dimension"]},
        "arms": {
            "TERA-50": {"fixture": True},
            "ORBIT-50": {"fixture": True},
            f"ORBIT-{task['orbit_resource_m']}": {"fixture": True},
            "value-only-conditional-50": {"fixture": True},
        },
        "catalog": {
            "path": recipe_catalog["path"],
            "sha256": authorization["catalog_sha256"],
            "generation_git_commit": recipe_catalog["generation_git_commit"],
            "generation_git_tree": recipe_catalog["generation_git_tree"],
            "task_index": task["catalog_task_index"],
        },
        "provenance": {
            "git": {
                "commit": authorization["execution_commit"],
                "tree": authorization["execution_tree"],
                "status_porcelain": [],
            },
            "data": {
                "dataset_path": f"/protected/data/{bundle['dataset_filename']}",
                "metadata_path": f"/protected/data/{bundle['metadata_filename']}",
                "manifest_path": f"/protected/data/{bundle['manifest_filename']}",
                "file_sha256": {
                    bundle["dataset_filename"]: bundle["hashes"]["dataset_file_sha256"],
                    bundle["metadata_filename"]: bundle["hashes"]["metadata_file_sha256"],
                },
                "manifest_sha256": bundle["hashes"]["sha256_manifest_file_sha256"],
                "generator_config": bundle["generator_config"],
            },
            "dependencies": {
                item["path"]: {"sha256": item["sha256"]} for item in artifacts["dependencies"]
            },
            "submodules": {"tera_gitlink": artifacts["tera_submodule"]["commit"]},
        },
    }
    with pytest.raises(ReleaseLedgerError, match="dataset_content_sha256"):
        _validate_result_for_release(
            result_document,
            task=task,
            recipe_document=document,
            authorization=authorization,
        )

    result_hash = "e" * 64
    wrong_bundle_hashes = dict(bundle["hashes"])
    wrong_bundle_hashes["dataset_content_sha256"] = "0" * 64
    attestation = {
        "schema_version": RESULT_ATTESTATION_SCHEMA_VERSION,
        "task_identity": task,
        "bundle_hashes": wrong_bundle_hashes,
        "release": {
            "release_id": authorization["release_id"],
            "experiment_id": authorization["experiment_id"],
            "protocol_id": authorization["protocol_id"],
            "protocol_path": authorization["protocol_path"],
            "protocol_sha256": authorization["protocol_sha256"],
            "protocol_binding_sha256": authorization["protocol_binding_sha256"],
            "recipe_payload_sha256": authorization["recipe_payload_sha256"],
            "recipe_file_sha256": authorization["recipe_file_sha256"],
            "source_commit": authorization["source_commit"],
            "execution_commit": authorization["execution_commit"],
            "catalog_sha256": authorization["catalog_sha256"],
            "method_configuration_sha256": task["method_configuration_sha256"],
        },
        "result_file_sha256": result_hash,
    }
    with pytest.raises(ReleaseLedgerError, match="bundle hashes are mismatched"):
        _validate_success_attestation(
            attestation,
            attestation_file_sha256="f" * 64,
            result_file_sha256=result_hash,
            task=task,
            bundle=bundle,
            authorization=authorization,
        )


def test_schemas_are_valid_draft_2020_12_non_authoritative_scaffolds() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_names = (
        "f02_method_selection_v1.schema.json",
        "f02_global_confirmatory_recipe_v1.schema.json",
        "f02_result_attestation_v1.schema.json",
        "f02_one_release_ledger_v1.schema.json",
        "f02_catalog_release_marker_v1.schema.json",
    )
    for name in schema_names:
        schema = json.loads((_REPO_ROOT / "schemas" / name).read_text())
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        jsonschema.Draft202012Validator.check_schema(schema)
    recipe_schema = json.loads(
        (_REPO_ROOT / "schemas/f02_global_confirmatory_recipe_v1.schema.json").read_text()
    )
    recipe_properties = recipe_schema["properties"]["payload"]["properties"]
    assert recipe_properties["release_scaffold_status"]["const"] == RECIPE_SCAFFOLD_STATUS
    assert recipe_properties["development_evidence_semantically_verified"]["const"] is False
    ledger_schema = json.loads(
        (_REPO_ROOT / "schemas/f02_one_release_ledger_v1.schema.json").read_text()
    )
    authorization_properties = ledger_schema["properties"]["authorization"]["properties"]
    assert authorization_properties["release_mutations_enabled"]["const"] is False


def test_recipe_schema_is_explicitly_non_authoritative_for_nested_semantics(
    tmp_path: Path,
) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    repo, catalog, _, document = _write_recipe_and_release(tmp_path)
    malicious = copy.deepcopy(document)
    malicious["payload"]["confirmatory_bundles"] = [{} for _ in range(50)]
    for grid in malicious["payload"]["expected_task_grids"]:
        grid["tasks"] = [{} for _ in range(150)]
    malicious["payload_sha256"] = sha256_bytes(canonical_json_bytes(malicious["payload"]))
    schema = json.loads(
        (_REPO_ROOT / "schemas/f02_global_confirmatory_recipe_v1.schema.json").read_text()
    )

    # Deliberately record the v1 boundary: JSON Schema is descriptive only and accepts these
    # semantically empty nested records. The authoritative Python structural validator must reject.
    assert list(jsonschema.Draft202012Validator(schema).iter_errors(malicious)) == []
    with pytest.raises(GlobalRecipeError, match="50 confirmatory bundles"):
        validate_global_recipe_document(
            malicious,
            catalog_path=catalog,
            repo_root=repo,
        )


def test_cli_scripts_are_importable_outside_repository(tmp_path: Path) -> None:
    for script in (
        _REPO_ROOT / "experiments/f02_global_release.py",
        _REPO_ROOT / "cluster/f02_confirmatory_ledger.py",
    ):
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=tmp_path,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert "confirmatory" in completed.stdout.lower()
