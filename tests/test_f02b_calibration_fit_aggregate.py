from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
import shutil
import struct
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from cluster.f02b_calibration_grid import FIT_TASKS, fit_task_for_index
from data.generate_nbody_confirmatory import ConfirmatoryConfig
from experiments.f02b_calibration_contract import (
    F02_CATALOG_SHA256,
    MINIMUM_GPU_MEMORY_BYTES,
    MINIMUM_HOST_MEMORY_BYTES,
    WALLTIME_SECONDS,
    canonical_json_bytes,
    canonical_sha256,
    parse_strict_json_bytes,
)
from experiments.f02b_calibration_fit import (
    EXCLUSIVE_VERIFICATION_MODE,
    build_fit_artifacts,
    fit_artifact_paths,
    write_fit_artifacts_exclusive,
)
from experiments.f02b_calibration_fit_aggregate import (
    CalibrationFitAggregateError,
    aggregate_and_write_fit_stage,
    aggregate_fit_stage,
    expected_fit_stage_paths,
    validate_expected_deployment,
    validate_fit_catalog,
    write_fit_catalog_exclusive,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _binary32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


def _parameters() -> dict[str, object]:
    return {
        "lengthscale": [_binary32(1.25)],
        "outputscale": _binary32(2.0),
        "sigma_f": _binary32(0.001),
        "sigma_g": _binary32(0.002),
        "kernel": "rbf",
        "gradient_noise_model": "iid",
    }


def _runtime_allocation() -> dict[str, object]:
    return {
        "exclusive_node": True,
        "requested_gpu_count": 1,
        "visible_gpu_count": 2,
        "visible_gpu_models": ["NVIDIA L40S", "NVIDIA L40S"],
        "visible_gpu_memory_bytes": [
            MINIMUM_GPU_MEMORY_BYTES,
            MINIMUM_GPU_MEMORY_BYTES,
        ],
        "requested_cpus_per_task": 16,
        "available_cpu_count": 32,
        "available_host_memory_bytes": MINIMUM_HOST_MEMORY_BYTES,
        "requested_walltime_seconds": WALLTIME_SECONDS,
        "walltime_limit_seconds": WALLTIME_SECONDS,
        "array_concurrency": 1,
        "partition": "short",
    }


def _identity(base: Path, task_index: int) -> dict[str, Any]:
    task = fit_task_for_index(task_index)
    dataset = (base / "corpus" / f"{task.dataset_stem}.npz").resolve()
    metadata = dataset.with_suffix(".metadata.json")
    manifest = dataset.with_suffix(".sha256.json")
    catalog = (base / "catalog.json").resolve()
    repository = (base / "repo").resolve()
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
                dataset.name: _digest(f"{task.dataset_stem}:dataset"),
                metadata.name: _digest(f"{task.dataset_stem}:metadata"),
            },
            "sha256_manifest_file_sha256": _digest(f"{task.dataset_stem}:manifest"),
            "dataset_content_sha256": _digest(f"{task.dataset_stem}:content"),
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
            "job_id": f"41{task.task_index:03d}",
            "array_job_id": "41000",
            "array_task_id": task.task_index,
            "node_list": "gpu-l40s-01",
            "exclusive_verification_mode": EXCLUSIVE_VERIFICATION_MODE,
        },
    }


def _expected_deployment() -> dict[str, str]:
    return {
        "source_commit": "3" * 40,
        "source_tree": "4" * 40,
        "tera_gitlink": "5" * 40,
        "pyproject_sha256": "6" * 64,
        "uv_lock_sha256": "7" * 64,
        "catalog_generation_commit": "1" * 40,
        "catalog_generation_tree": "2" * 40,
        "catalog_sha256": F02_CATALOG_SHA256,
    }


def _write_task(
    root: Path,
    identity_base: Path,
    task_index: int,
    *,
    identity: dict[str, Any] | None = None,
) -> None:
    _, payload_bytes, envelope = build_fit_artifacts(
        task_index,
        _parameters(),
        identity or _identity(identity_base, task_index),
        output_root=root,
    )
    paths = fit_artifact_paths(root, task_index)
    if paths.directory.exists():
        paths.numeric_payload.write_bytes(payload_bytes)
        paths.execution_envelope.write_bytes(canonical_json_bytes(envelope))
    else:
        write_fit_artifacts_exclusive(root, task_index, payload_bytes, envelope)


def _populate(root: Path, identity_base: Path) -> None:
    for task in FIT_TASKS:
        _write_task(root, identity_base, task.task_index)


@pytest.fixture
def valid_stage(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    root = tmp_path / "fits"
    identity_base = tmp_path / "identity"
    _populate(root, identity_base)
    return root, identity_base, _expected_deployment()


def _task_codes(catalog: dict[str, Any], task_index: int) -> set[str]:
    return {failure["code"] for failure in catalog["tasks"][task_index]["errors"]}


def _rewrite_envelope_for_payload(root: Path, task_index: int, payload_bytes: bytes) -> None:
    paths = fit_artifact_paths(root, task_index)
    envelope = parse_strict_json_bytes(paths.execution_envelope.read_bytes())
    record = envelope["canonical_record"]
    record["payload_binding"]["numeric_payload_sha256"] = hashlib.sha256(payload_bytes).hexdigest()
    envelope["canonical_record_sha256"] = canonical_sha256(record)
    paths.execution_envelope.write_bytes(canonical_json_bytes(envelope))


def _rewrite_envelope_record(root: Path, task_index: int, mutate: Any) -> None:
    paths = fit_artifact_paths(root, task_index)
    envelope = parse_strict_json_bytes(paths.execution_envelope.read_bytes())
    mutate(envelope["canonical_record"])
    envelope["canonical_record_sha256"] = canonical_sha256(envelope["canonical_record"])
    paths.execution_envelope.write_bytes(canonical_json_bytes(envelope))


def _rehash_catalog(catalog: dict[str, Any]) -> None:
    fields = catalog["integrity"]["covered_fields"]
    catalog["integrity"]["payload_sha256"] = canonical_sha256(
        {field: catalog[field] for field in fields}
    )


def test_exact_45_identity_paths_and_valid_catalog_without_glob(
    valid_stage: tuple[Path, Path, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _, deployment = valid_stage
    expected = expected_fit_stage_paths(root)
    assert len(expected) == 45
    assert [record.task_index for record in expected] == list(range(45))
    assert len({record.directory.name for record in expected}) == 45
    for record in expected:
        runner = fit_artifact_paths(root, record.task_index)
        assert record.directory == runner.directory
        assert record.payload_path == runner.numeric_payload
        assert record.envelope_path == runner.execution_envelope

    def reject_glob(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("aggregation must not glob")

    monkeypatch.setattr(Path, "glob", reject_glob)
    monkeypatch.setattr(Path, "rglob", reject_glob)
    catalog = aggregate_fit_stage(root, expected_deployment=deployment)
    assert catalog["analysis_ready"] is True
    assert catalog["freeze_ready"] is False
    assert catalog["valid_task_count"] == 45
    assert catalog["invalid_task_count"] == 0
    assert catalog["structural_failure_count"] == 0
    assert catalog["cohort_identity_sha256"] is not None
    assert validate_fit_catalog(catalog) == catalog
    for result in catalog["tasks"]:
        assert result["status"] == "valid"
        assert result["payload_raw_sha256"] == result["payload_canonical_sha256"]
        assert all(
            result[field] is not None
            for field in (
                "payload_raw_sha256",
                "envelope_raw_sha256",
                "envelope_record_sha256",
            )
        )


def test_catalog_is_parameter_and_label_free(
    valid_stage: tuple[Path, Path, dict[str, str]],
) -> None:
    root, _, deployment = valid_stage
    catalog = aggregate_fit_stage(root, expected_deployment=deployment)
    assert catalog["label_policy"] == {
        "scope": "development_only",
        "validation_labels_exposed": False,
        "test_labels_exposed": False,
        "prediction_or_scoring_in_catalog": False,
    }
    assert catalog["freeze_ready"] is False
    assert all("parameters" not in result for result in catalog["tasks"])
    assert all("input_identity" not in result for result in catalog["tasks"])
    encoded = canonical_json_bytes(catalog)
    assert b'"selected_update"' not in encoded
    assert b'"model_state"' not in encoded
    assert b'"optimizer_state"' not in encoded


def test_missing_root_retains_all_45_failure_slots(tmp_path: Path) -> None:
    root = tmp_path / "does-not-exist"
    catalog = aggregate_fit_stage(root, expected_deployment=_expected_deployment())
    assert catalog["analysis_ready"] is False
    assert catalog["valid_task_count"] == 0
    assert catalog["invalid_task_count"] == 45
    assert len(catalog["tasks"]) == 45
    assert catalog["structural_failures"][0]["code"] == "missing_input_root"
    assert all(_task_codes(catalog, index) == {"missing_task_directory"} for index in range(45))
    validate_fit_catalog(catalog)


def test_missing_pair_member_and_unexpected_entries_are_reported(
    valid_stage: tuple[Path, Path, dict[str, str]],
) -> None:
    root, _, deployment = valid_stage
    fit_artifact_paths(root, 4).execution_envelope.unlink()
    (root / ".partial-upload").write_text("x")
    (fit_artifact_paths(root, 5).directory / ".parameters.json.tmp").write_text("x")
    catalog = aggregate_fit_stage(root, expected_deployment=deployment)
    assert catalog["analysis_ready"] is False
    assert "missing_envelope" in _task_codes(catalog, 4)
    assert "unexpected_path" in _task_codes(catalog, 5)
    assert catalog["unexpected_path_count"] == 2
    paths = {failure["path"] for failure in catalog["structural_failures"]}
    assert ".partial-upload" in paths
    assert any(path.endswith("/.parameters.json.tmp") for path in paths)
    validate_fit_catalog(catalog)


@pytest.mark.parametrize("target", ["task_directory", "payload", "envelope"])
def test_symlinked_expected_inputs_are_never_followed(
    valid_stage: tuple[Path, Path, dict[str, str]],
    tmp_path: Path,
    target: str,
) -> None:
    root, _, deployment = valid_stage
    paths = fit_artifact_paths(root, 7)
    outside = tmp_path / "outside"
    outside.mkdir()
    if target == "task_directory":
        shutil.rmtree(paths.directory)
        paths.directory.symlink_to(outside, target_is_directory=True)
        expected_code = "invalid_task_directory"
    else:
        selected = paths.numeric_payload if target == "payload" else paths.execution_envelope
        external = outside / selected.name
        external.write_bytes(selected.read_bytes())
        selected.unlink()
        selected.symlink_to(external)
        expected_code = "invalid_payload_file" if target == "payload" else "invalid_envelope_file"
    catalog = aggregate_fit_stage(root, expected_deployment=deployment)
    assert catalog["analysis_ready"] is False
    assert expected_code in _task_codes(catalog, 7)


def test_nonregular_payload_is_rejected(
    valid_stage: tuple[Path, Path, dict[str, str]],
) -> None:
    root, _, deployment = valid_stage
    path = fit_artifact_paths(root, 8).numeric_payload
    path.unlink()
    path.mkdir()
    catalog = aggregate_fit_stage(root, expected_deployment=deployment)
    assert "invalid_payload_file" in _task_codes(catalog, 8)
    assert catalog["analysis_ready"] is False


@pytest.mark.parametrize(
    "fault",
    ["noncanonical", "duplicate_key", "nonfinite", "unexpected_field"],
)
def test_malformed_or_noncanonical_payload_faults_fail_closed(
    valid_stage: tuple[Path, Path, dict[str, str]],
    fault: str,
) -> None:
    root, _, deployment = valid_stage
    paths = fit_artifact_paths(root, 9)
    original = parse_strict_json_bytes(paths.numeric_payload.read_bytes())
    if fault == "noncanonical":
        payload_bytes = (json.dumps(original, indent=2, sort_keys=True) + "\n").encode()
    elif fault == "duplicate_key":
        payload_bytes = b'{"x":1,"x":2}'
    elif fault == "nonfinite":
        payload_bytes = b'{"x":NaN}'
    else:
        original["unexpected"] = True
        payload_bytes = canonical_json_bytes(original)
    paths.numeric_payload.write_bytes(payload_bytes)
    _rewrite_envelope_for_payload(root, 9, payload_bytes)
    catalog = aggregate_fit_stage(root, expected_deployment=deployment)
    assert "malformed_payload" in _task_codes(catalog, 9)
    assert catalog["analysis_ready"] is False
    validate_fit_catalog(catalog)


@pytest.mark.parametrize(
    "fault",
    ["noncanonical", "wrong_path", "wrong_hash", "bad_allocation", "duplicate_key"],
)
def test_malformed_or_noncanonical_envelope_faults_fail_closed(
    valid_stage: tuple[Path, Path, dict[str, str]],
    fault: str,
) -> None:
    root, _, deployment = valid_stage
    paths = fit_artifact_paths(root, 10)
    if fault == "noncanonical":
        envelope = parse_strict_json_bytes(paths.execution_envelope.read_bytes())
        paths.execution_envelope.write_bytes(json.dumps(envelope, indent=2).encode())
    elif fault == "wrong_path":
        _rewrite_envelope_record(
            root,
            10,
            lambda record: record["payload_binding"].__setitem__(
                "numeric_payload_path", "elsewhere/parameters.json"
            ),
        )
    elif fault == "wrong_hash":
        _rewrite_envelope_record(
            root,
            10,
            lambda record: record["payload_binding"].__setitem__(
                "numeric_payload_sha256", "0" * 64
            ),
        )
    elif fault == "bad_allocation":
        _rewrite_envelope_record(
            root,
            10,
            lambda record: record["runtime_allocation"].__setitem__(
                "visible_gpu_models", ["not-L40S", "NVIDIA L40S"]
            ),
        )
    else:
        paths.execution_envelope.write_bytes(b'{"canonical_record":{},"canonical_record":{}}')
    catalog = aggregate_fit_stage(root, expected_deployment=deployment)
    assert "malformed_envelope" in _task_codes(catalog, 10)
    assert catalog["analysis_ready"] is False


def test_duplicate_hardlink_and_payload_bytes_invalidate_every_owner(
    valid_stage: tuple[Path, Path, dict[str, str]],
) -> None:
    root, _, deployment = valid_stage
    source = fit_artifact_paths(root, 0).numeric_payload
    destination = fit_artifact_paths(root, 1).numeric_payload
    destination.unlink()
    os.link(source, destination)
    catalog = aggregate_fit_stage(root, expected_deployment=deployment)
    for task_index in (0, 1):
        assert "invalid_payload_file" in _task_codes(catalog, task_index)
    assert catalog["analysis_ready"] is False


def test_external_hardlink_alias_is_rejected(
    valid_stage: tuple[Path, Path, dict[str, str]],
    tmp_path: Path,
) -> None:
    root, _, deployment = valid_stage
    payload = fit_artifact_paths(root, 2).numeric_payload
    external_alias = tmp_path / "external-payload-alias.json"
    os.link(payload, external_alias)
    catalog = aggregate_fit_stage(root, expected_deployment=deployment)
    assert "invalid_payload_file" in _task_codes(catalog, 2)
    assert catalog["analysis_ready"] is False
    assert external_alias.exists()


def test_payload_must_match_explicit_deployment_and_uniform_cohort(
    valid_stage: tuple[Path, Path, dict[str, str]],
) -> None:
    root, identity_base, deployment = valid_stage
    changed = _identity(identity_base, 11)
    changed["source"]["commit"] = "8" * 40
    _write_task(root, identity_base, 11, identity=changed)
    catalog = aggregate_fit_stage(root, expected_deployment=deployment)
    assert "deployment_identity_mismatch" in _task_codes(catalog, 11)
    assert "cohort_identity_mismatch" in _task_codes(catalog, 11)
    assert catalog["cohort_identity_sha256"] is None
    assert catalog["analysis_ready"] is False


def test_scheduler_cohort_rejects_mixed_array_jobs_deterministically(
    valid_stage: tuple[Path, Path, dict[str, str]],
) -> None:
    root, identity_base, deployment = valid_stage
    changed = _identity(identity_base, 11)
    changed["scheduler"]["array_job_id"] = "42000"
    _write_task(root, identity_base, 11, identity=changed)

    first = aggregate_fit_stage(root, expected_deployment=deployment)
    second = aggregate_fit_stage(root, expected_deployment=deployment)
    assert first == second
    assert first["analysis_ready"] is False
    assert first["valid_task_count"] == 0
    assert all(
        "scheduler_array_job_mismatch" in _task_codes(first, task_index) for task_index in range(45)
    )
    assert all(
        "cohort_identity_mismatch" not in _task_codes(first, task_index) for task_index in range(45)
    )


def test_scheduler_cohort_rejects_reused_per_task_job_id(
    valid_stage: tuple[Path, Path, dict[str, str]],
) -> None:
    root, identity_base, deployment = valid_stage
    changed = _identity(identity_base, 1)
    changed["scheduler"]["job_id"] = _identity(identity_base, 0)["scheduler"]["job_id"]
    _write_task(root, identity_base, 1, identity=changed)

    catalog = aggregate_fit_stage(root, expected_deployment=deployment)
    assert catalog["analysis_ready"] is False
    assert catalog["valid_task_count"] == 43
    for task_index in (0, 1):
        assert "duplicate_scheduler_job_id" in _task_codes(catalog, task_index)
        assert "cohort_identity_mismatch" not in _task_codes(catalog, task_index)
    assert all(
        "duplicate_scheduler_job_id" not in _task_codes(catalog, task_index)
        for task_index in range(2, 45)
    )


def test_seed_replicated_bundle_identity_must_be_exact(
    valid_stage: tuple[Path, Path, dict[str, str]],
) -> None:
    root, identity_base, deployment = valid_stage
    changed = _identity(identity_base, 1)
    changed["bundle"]["dataset_content_sha256"] = _digest("altered-bundle")
    _write_task(root, identity_base, 1, identity=changed)
    catalog = aggregate_fit_stage(root, expected_deployment=deployment)
    assert "bundle_identity_mismatch" in _task_codes(catalog, 1)
    assert catalog["analysis_ready"] is False


def test_dataset_content_cannot_be_reused_by_distinct_bundle_stems(
    valid_stage: tuple[Path, Path, dict[str, str]],
) -> None:
    root, identity_base, deployment = valid_stage
    changed = _identity(identity_base, 3)
    changed["bundle"]["dataset_content_sha256"] = _identity(identity_base, 0)["bundle"][
        "dataset_content_sha256"
    ]
    _write_task(root, identity_base, 3, identity=changed)
    catalog = aggregate_fit_stage(root, expected_deployment=deployment)
    assert "duplicate_dataset_content" in _task_codes(catalog, 3)
    assert catalog["analysis_ready"] is False


def test_expected_deployment_schema_and_frozen_catalog_are_strict() -> None:
    deployment = _expected_deployment()
    assert validate_expected_deployment(deployment) == deployment
    for field, replacement in (
        ("source_commit", "0" * 39),
        ("pyproject_sha256", "G" * 64),
        ("catalog_sha256", "0" * 64),
    ):
        changed = dict(deployment)
        changed[field] = replacement
        with pytest.raises(CalibrationFitAggregateError):
            validate_expected_deployment(changed)
    changed = dict(deployment)
    changed["unexpected"] = "x"
    with pytest.raises(CalibrationFitAggregateError, match="fields mismatch"):
        validate_expected_deployment(changed)


def test_catalog_semantics_reject_tampering_even_after_rehash(
    valid_stage: tuple[Path, Path, dict[str, str]],
) -> None:
    root, _, deployment = valid_stage
    catalog = aggregate_fit_stage(root, expected_deployment=deployment)
    tampered = copy.deepcopy(catalog)
    tampered["freeze_ready"] = True
    _rehash_catalog(tampered)
    with pytest.raises(CalibrationFitAggregateError, match="freeze_ready"):
        validate_fit_catalog(tampered)

    tampered = copy.deepcopy(catalog)
    tampered["tasks"][0]["payload_path"] = "parameters.json"
    _rehash_catalog(tampered)
    with pytest.raises(CalibrationFitAggregateError, match="noncanonical task path"):
        validate_fit_catalog(tampered)


def test_atomic_exclusive_catalog_output_and_no_temporary_residue(
    valid_stage: tuple[Path, Path, dict[str, str]],
    tmp_path: Path,
) -> None:
    root, _, deployment = valid_stage
    catalog = aggregate_fit_stage(root, expected_deployment=deployment)
    destination = tmp_path / "published" / "fit-catalog.json"
    assert write_fit_catalog_exclusive(destination, catalog) == destination
    original = destination.read_bytes()
    assert original == canonical_json_bytes(catalog)
    assert parse_strict_json_bytes(original) == catalog
    assert list(destination.parent.iterdir()) == [destination]

    with pytest.raises(CalibrationFitAggregateError, match="refusing to overwrite"):
        write_fit_catalog_exclusive(destination, catalog)
    assert destination.read_bytes() == original
    assert list(destination.parent.iterdir()) == [destination]


def test_injected_atomic_publication_failure_cleans_temporary_file(
    valid_stage: tuple[Path, Path, dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _, deployment = valid_stage
    catalog = aggregate_fit_stage(root, expected_deployment=deployment)
    destination = tmp_path / "failed-publication" / "fit-catalog.json"

    def fail_link(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.EIO, "injected link failure")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(CalibrationFitAggregateError, match="before publication"):
        write_fit_catalog_exclusive(destination, catalog)
    assert not destination.exists()
    assert list(destination.parent.iterdir()) == []


def test_failure_catalog_is_publishable_but_never_analysis_or_freeze_ready(
    tmp_path: Path,
) -> None:
    catalog = aggregate_fit_stage(
        tmp_path / "missing",
        expected_deployment=_expected_deployment(),
    )
    destination = tmp_path / "failure-catalog.json"
    write_fit_catalog_exclusive(destination, catalog)
    published = parse_strict_json_bytes(destination.read_bytes())
    assert published["analysis_ready"] is False
    assert published["freeze_ready"] is False
    assert len(published["tasks"]) == 45
    validate_fit_catalog(published)


@pytest.mark.parametrize("relative_output", ["catalog.json", "nested/catalog.json", "."])
def test_combined_aggregation_rejects_output_at_or_below_input_root(
    valid_stage: tuple[Path, Path, dict[str, str]],
    relative_output: str,
) -> None:
    root, _, deployment = valid_stage
    destination = root / relative_output
    before = {path.name for path in root.iterdir()}
    with pytest.raises(CalibrationFitAggregateError, match="outside the aggregated input root"):
        aggregate_and_write_fit_stage(
            root,
            destination,
            expected_deployment=deployment,
        )
    assert {path.name for path in root.iterdir()} == before


def test_symlink_output_directory_is_rejected_without_writing(
    valid_stage: tuple[Path, Path, dict[str, str]],
    tmp_path: Path,
) -> None:
    root, _, deployment = valid_stage
    catalog = aggregate_fit_stage(root, expected_deployment=deployment)
    real = tmp_path / "real-output"
    real.mkdir()
    alias = tmp_path / "output-alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(CalibrationFitAggregateError, match="no-follow output"):
        write_fit_catalog_exclusive(alias / "catalog.json", catalog)
    assert not (real / "catalog.json").exists()
