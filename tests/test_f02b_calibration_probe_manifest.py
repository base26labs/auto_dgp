from __future__ import annotations

import copy
import json
from typing import Any

import pytest

import experiments.f02b_calibration_probe_manifest as probe_manifest_module
from cluster.f02b_calibration_grid import CALIBRATION_ID
from experiments.f02b_calibration_contract import (
    F02_CATALOG_SHA256,
    FIT_RECIPE_SHA256,
    MATRIX_HASHES,
    NUMERICAL_POLICY,
    RESOURCE_CONTRACT,
    canonical_json_bytes,
    canonical_sha256,
)
from experiments.f02b_calibration_probe_core import PROBE_WORK_PLAN_SHA256
from experiments.f02b_calibration_probe_manifest import (
    PROBE_LAUNCH_MANIFEST_SCHEMA_VERSION,
    PROBE_LAUNCH_MANIFEST_TYPE,
    SCHEDULER_PLAN_SHA256,
    SUBMISSION_IDENTITY_POLICY,
    SUBMISSION_PLAN,
    ProbeLaunchManifestError,
    build_probe_launch_manifest,
    build_probe_launch_manifest_bytes,
    parse_probe_launch_manifest_bytes,
    validate_probe_launch_manifest,
)


def _expected_fit_deployment() -> dict[str, str]:
    return {
        "source_commit": "1" * 40,
        "source_tree": "2" * 40,
        "tera_gitlink": "3" * 40,
        "pyproject_sha256": "4" * 64,
        "uv_lock_sha256": "5" * 64,
        "catalog_generation_commit": "6" * 40,
        "catalog_generation_tree": "7" * 40,
        "catalog_sha256": F02_CATALOG_SHA256,
    }


def _probe_deployment() -> dict[str, str]:
    return {
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "tera_gitlink": "3" * 40,
        "pyproject_sha256": "c" * 64,
        "uv_lock_sha256": "d" * 64,
        "catalog_generation_commit": "6" * 40,
        "catalog_generation_tree": "7" * 40,
        "catalog_sha256": F02_CATALOG_SHA256,
    }


def _build_kwargs() -> dict[str, Any]:
    return {
        "probe_deployment": _probe_deployment(),
        "expected_fit_deployment": _expected_fit_deployment(),
        "data_root": "/srv/f02b/development-data",
        "fit_stage_root": "/srv/f02b/fit-stage",
        "fit_catalog_path": "/srv/f02b/published/fit-catalog.json",
        "probe_output_root": "/srv/f02b/probe-stage",
        "data_catalog_path": "/srv/f02b/catalogs/data-catalog.json",
        "fit_catalog_raw_sha256": "8" * 64,
        "fit_catalog_integrity_payload_sha256": "9" * 64,
        "fit_catalog_cohort_identity_sha256": "a" * 64,
    }


def _manifest() -> dict[str, Any]:
    return build_probe_launch_manifest(**_build_kwargs())


def _rehash(manifest: dict[str, Any]) -> None:
    fields = manifest["integrity"]["covered_fields"]
    manifest["integrity"]["payload_sha256"] = canonical_sha256(
        {field: manifest[field] for field in fields}
    )


def test_build_validate_and_parse_canonical_manifest_bytes() -> None:
    manifest = _manifest()
    encoded = canonical_json_bytes(manifest)

    assert manifest["schema_version"] == PROBE_LAUNCH_MANIFEST_SCHEMA_VERSION
    assert manifest["manifest_type"] == PROBE_LAUNCH_MANIFEST_TYPE
    assert manifest["calibration_id"] == CALIBRATION_ID
    assert manifest["matrix_hashes"] == dict(MATRIX_HASHES)
    assert canonical_json_bytes(manifest["resource_contract"]) == canonical_json_bytes(
        RESOURCE_CONTRACT
    )
    assert manifest["numerical_policy"] == dict(NUMERICAL_POLICY)
    assert manifest["fit_recipe_sha256"] == FIT_RECIPE_SHA256
    assert manifest["probe_work_plan_sha256"] == PROBE_WORK_PLAN_SHA256
    assert manifest["scheduler_plan_sha256"] == SCHEDULER_PLAN_SHA256
    assert manifest["probe_deployment"] == _probe_deployment()
    assert manifest["expected_fit_deployment"] == _expected_fit_deployment()
    assert manifest["probe_deployment"] != manifest["expected_fit_deployment"]
    assert validate_probe_launch_manifest(manifest) == manifest
    assert parse_probe_launch_manifest_bytes(encoded) == manifest
    assert build_probe_launch_manifest_bytes(**_build_kwargs()) == encoded
    assert manifest["integrity"] == {
        "algorithm": "sha256",
        "covered_fields": [
            "schema_version",
            "manifest_type",
            "calibration_id",
            "matrix_hashes",
            "resource_contract",
            "numerical_policy",
            "fit_recipe_sha256",
            "probe_work_plan_sha256",
            "scheduler_plan_sha256",
            "probe_deployment",
            "expected_fit_deployment",
            "paths",
            "fit_catalog_identity",
            "submission_identity_policy",
            "submission_plan",
        ],
        "payload_sha256": canonical_sha256(
            {field: manifest[field] for field in manifest["integrity"]["covered_fields"]}
        ),
    }


def test_submission_plan_is_exact_and_requires_three_scheduler_identities() -> None:
    manifest = _manifest()
    plan = manifest["submission_plan"]

    assert plan == [dict(entry) for entry in SUBMISSION_PLAN]
    assert [entry["array_spec"] for entry in plan] == [
        "0-119%1",
        "120-120%1",
        "121-121%1",
    ]
    assert [entry["array_task_count"] for entry in plan] == [120, 1, 1]
    assert [entry["array_concurrency"] for entry in plan] == [1, 1, 1]
    identities = [entry["submission_identity"] for entry in plan]
    assert len(identities) == len(set(identities)) == 3
    assert manifest["submission_identity_policy"] == {
        "scheduler_identity_field": "array_job_id",
        "require_pairwise_distinct": True,
    }
    assert manifest["submission_identity_policy"] == dict(SUBMISSION_IDENTITY_POLICY)


def test_builder_outputs_are_detached_and_public_scheduler_views_are_read_only() -> None:
    first = _manifest()
    first["submission_plan"][0]["array_task_max"] = 118
    first["paths"]["data_root"] = "/changed"
    second = _manifest()

    assert second["submission_plan"][0]["array_task_max"] == 119
    assert second["paths"]["data_root"] == "/srv/f02b/development-data"
    with pytest.raises(TypeError):
        SUBMISSION_PLAN[0]["array_task_max"] = 118
    with pytest.raises(TypeError):
        SUBMISSION_IDENTITY_POLICY["require_pairwise_distinct"] = False
    with pytest.raises(TypeError):
        dict.__setitem__(SUBMISSION_PLAN[0], "array_task_max", 118)
    with pytest.raises(TypeError):
        dict.__setitem__(SUBMISSION_IDENTITY_POLICY, "require_pairwise_distinct", False)


def test_polluted_public_scheduler_views_cannot_change_builder_or_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _manifest()
    monkeypatch.setattr(
        probe_manifest_module,
        "SUBMISSION_PLAN",
        ({"submission_identity": "attacker-controlled"},),
    )
    monkeypatch.setattr(
        probe_manifest_module,
        "SUBMISSION_IDENTITY_POLICY",
        {"scheduler_identity_field": "job_id", "require_pairwise_distinct": False},
    )
    monkeypatch.setattr(probe_manifest_module, "SCHEDULER_PLAN_SHA256", "0" * 64)

    assert _manifest() == baseline
    assert validate_probe_launch_manifest(baseline) == baseline


@pytest.mark.parametrize(
    ("constant", "constant_field", "replacement", "manifest_field"),
    [
        (MATRIX_HASHES, "fit_task_matrix_sha256", "0" * 64, "matrix_hashes"),
        (RESOURCE_CONTRACT, "requested_gpu_count", 2, "resource_contract"),
        (NUMERICAL_POLICY, "source_dtype", "float64", "numerical_policy"),
    ],
)
def test_dict_base_class_bypass_cannot_pollute_manifest_acceptance_semantics(
    constant: dict[str, Any],
    constant_field: str,
    replacement: Any,
    manifest_field: str,
) -> None:
    baseline = _manifest()
    original = constant[constant_field]
    dict.__setitem__(constant, constant_field, replacement)
    try:
        assert constant[constant_field] == replacement
        assert _manifest() == baseline
        assert validate_probe_launch_manifest(baseline) == baseline

        poisoned = copy.deepcopy(baseline)
        poisoned[manifest_field] = json.loads(canonical_json_bytes(constant))
        _rehash(poisoned)
        with pytest.raises(ProbeLaunchManifestError, match=manifest_field):
            validate_probe_launch_manifest(poisoned)
    finally:
        dict.__setitem__(constant, constant_field, original)


def test_build_and_validate_recompute_scheduler_domain_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _manifest()
    corrupted_plan = json.loads(probe_manifest_module._SUBMISSION_PLAN_CANONICAL_BYTES)
    corrupted_plan[0]["submission_role"] = "corrupted-role"
    monkeypatch.setattr(
        probe_manifest_module,
        "_SUBMISSION_PLAN_CANONICAL_BYTES",
        canonical_json_bytes(corrupted_plan),
    )

    with pytest.raises(RuntimeError, match="scheduler-plan domain SHA-256"):
        build_probe_launch_manifest(**_build_kwargs())
    with pytest.raises(RuntimeError, match="scheduler-plan domain SHA-256"):
        validate_probe_launch_manifest(baseline)


def test_build_and_validate_recompute_domain_separated_probe_work_plan_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _manifest()
    corrupted_payload = probe_manifest_module.canonical_probe_work_plan_payload()
    corrupted_payload["schema_version"] = "corrupted"
    monkeypatch.setattr(
        probe_manifest_module,
        "canonical_probe_work_plan_payload",
        lambda: corrupted_payload,
    )

    with pytest.raises(RuntimeError, match="probe work-plan payload SHA-256"):
        build_probe_launch_manifest(**_build_kwargs())
    with pytest.raises(RuntimeError, match="probe work-plan payload SHA-256"):
        validate_probe_launch_manifest(baseline)


@pytest.mark.parametrize(
    ("location", "operation"),
    [
        ("manifest", "extra"),
        ("manifest", "missing"),
        ("paths", "extra"),
        ("paths", "missing"),
        ("fit_catalog_identity", "extra"),
        ("fit_catalog_identity", "missing"),
        ("submission_identity_policy", "extra"),
        ("submission_identity_policy", "missing"),
        ("submission_plan", "extra"),
        ("submission_plan", "missing"),
        ("integrity", "extra"),
        ("integrity", "missing"),
    ],
)
def test_every_object_rejects_unknown_and_missing_fields(
    location: str,
    operation: str,
) -> None:
    manifest = _manifest()
    if location == "manifest":
        target = manifest
    elif location == "submission_plan":
        target = manifest["submission_plan"][0]
    else:
        target = manifest[location]
    if operation == "extra":
        target["unexpected"] = True
    else:
        target.pop(next(iter(target)))
    with pytest.raises(ProbeLaunchManifestError, match="fields mismatch"):
        validate_probe_launch_manifest(manifest)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("schema_version", "v2"),
        ("manifest_type", "other"),
        ("calibration_id", "other"),
        ("matrix_hashes", {**dict(MATRIX_HASHES), "probe_task_matrix_sha256": "0" * 64}),
        ("resource_contract", {**dict(RESOURCE_CONTRACT), "requested_gpu_count": 2}),
        ("numerical_policy", {**dict(NUMERICAL_POLICY), "source_dtype": "float64"}),
        ("fit_recipe_sha256", "0" * 64),
        ("probe_work_plan_sha256", "0" * 64),
        ("scheduler_plan_sha256", "0" * 64),
        (
            "submission_identity_policy",
            {"scheduler_identity_field": "job_id", "require_pairwise_distinct": True},
        ),
        ("submission_plan", [dict(entry) for entry in SUBMISSION_PLAN[:2]]),
    ],
)
def test_frozen_public_contract_fields_cannot_drift(
    field: str,
    replacement: object,
) -> None:
    manifest = _manifest()
    manifest[field] = replacement
    _rehash(manifest)
    with pytest.raises(ProbeLaunchManifestError):
        validate_probe_launch_manifest(manifest)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("data_root", "relative/data"),
        ("data_root", "/srv/f02b/../data"),
        ("data_root", "/srv//f02b/data"),
        ("data_root", "/srv/f02b/./data"),
        ("data_root", "/srv/f02b/data/"),
        ("data_root", "//srv/f02b/data"),
        ("data_root", "\\srv\\f02b\\data"),
        ("data_root", "/"),
        ("fit_catalog_path", "/srv/f02b/published/fit-catalog.txt"),
        ("data_catalog_path", "/srv/f02b/catalogs/data-catalog"),
    ],
)
def test_paths_must_be_canonical_absolute_posix_locations(
    field: str,
    value: str,
) -> None:
    manifest = _manifest()
    manifest["paths"][field] = value
    _rehash(manifest)
    with pytest.raises(ProbeLaunchManifestError):
        validate_probe_launch_manifest(manifest)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("probe_output_root", "/srv/f02b/development-data/output"),
        ("probe_output_root", "/srv/f02b"),
        ("fit_catalog_path", "/srv/f02b/fit-stage/fit-catalog.json"),
        ("fit_catalog_path", "/srv/f02b/probe-stage/fit-catalog.json"),
        ("data_catalog_path", "/srv/f02b/probe-stage/data-catalog.json"),
        ("data_catalog_path", "/srv/f02b/development-data"),
    ],
)
def test_paths_reject_aliases_and_input_output_overlap(field: str, value: str) -> None:
    manifest = _manifest()
    manifest["paths"][field] = value
    _rehash(manifest)
    with pytest.raises(ProbeLaunchManifestError):
        validate_probe_launch_manifest(manifest)


@pytest.mark.parametrize(
    "field",
    ["raw_sha256", "integrity_payload_sha256", "cohort_identity_sha256"],
)
def test_fit_catalog_identity_requires_three_sha256_digests(field: str) -> None:
    manifest = _manifest()
    manifest["fit_catalog_identity"][field] = "G" * 64
    _rehash(manifest)
    with pytest.raises(ProbeLaunchManifestError, match="SHA-256"):
        validate_probe_launch_manifest(manifest)


@pytest.mark.parametrize("field", ["probe_deployment", "expected_fit_deployment"])
def test_both_deployments_reuse_fit_catalog_contract(field: str) -> None:
    manifest = _manifest()
    manifest[field]["catalog_sha256"] = "0" * 64
    _rehash(manifest)
    with pytest.raises(ProbeLaunchManifestError, match=field):
        validate_probe_launch_manifest(manifest)

    manifest = _manifest()
    manifest[field]["source_commit"] = "not-a-git-object"
    _rehash(manifest)
    with pytest.raises(ProbeLaunchManifestError, match=field):
        validate_probe_launch_manifest(manifest)


@pytest.mark.parametrize(
    "field",
    ["tera_gitlink", "catalog_generation_commit", "catalog_generation_tree"],
)
def test_deployments_must_share_tera_and_catalog_generation_identity(field: str) -> None:
    kwargs = _build_kwargs()
    kwargs["probe_deployment"][field] = "f" * 40
    with pytest.raises(ProbeLaunchManifestError, match=field):
        build_probe_launch_manifest(**kwargs)

    manifest = _manifest()
    manifest["probe_deployment"][field] = "f" * 40
    _rehash(manifest)
    with pytest.raises(ProbeLaunchManifestError, match=field):
        validate_probe_launch_manifest(manifest)


def test_deployments_may_differ_in_probe_code_tree_and_dependency_locks() -> None:
    manifest = _manifest()

    for field in ("source_commit", "source_tree", "pyproject_sha256", "uv_lock_sha256"):
        assert manifest["probe_deployment"][field] != manifest["expected_fit_deployment"][field]
    assert validate_probe_launch_manifest(manifest) == manifest


@pytest.mark.parametrize(
    ("location", "field"),
    [
        ("manifest", "m"),
        ("manifest", "seed"),
        ("paths", "fit_task_index"),
        ("submission_plan", "repeat_id"),
        ("submission_plan", "kernel"),
    ],
)
def test_manifest_rejects_scientific_coordinate_fields(location: str, field: str) -> None:
    manifest = _manifest()
    if location == "manifest":
        target = manifest
    elif location == "submission_plan":
        target = manifest["submission_plan"][0]
    else:
        target = manifest[location]
    target[field] = 50
    _rehash(manifest)
    with pytest.raises(ProbeLaunchManifestError, match="forbidden scientific fields"):
        validate_probe_launch_manifest(manifest)


def test_integrity_hash_covers_every_nonintegrity_field() -> None:
    manifest = _manifest()
    for field in manifest["integrity"]["covered_fields"]:
        tampered = copy.deepcopy(manifest)
        value = tampered[field]
        if isinstance(value, str):
            tampered[field] = f"{value}-tampered"
        elif isinstance(value, list):
            value.reverse()
        else:
            value["tampered"] = True
        with pytest.raises(ProbeLaunchManifestError):
            validate_probe_launch_manifest(tampered)


def test_integrity_metadata_is_itself_strict() -> None:
    manifest = _manifest()
    manifest["integrity"]["algorithm"] = "sha512"
    with pytest.raises(ProbeLaunchManifestError, match="algorithm"):
        validate_probe_launch_manifest(manifest)

    manifest = _manifest()
    manifest["integrity"]["covered_fields"].reverse()
    with pytest.raises(ProbeLaunchManifestError, match="covered_fields"):
        validate_probe_launch_manifest(manifest)

    manifest = _manifest()
    manifest["integrity"]["payload_sha256"] = "0" * 64
    with pytest.raises(ProbeLaunchManifestError, match="integrity SHA-256"):
        validate_probe_launch_manifest(manifest)


def test_bytes_parser_rejects_duplicates_nonfinite_and_noncanonical_encoding() -> None:
    encoded = build_probe_launch_manifest_bytes(**_build_kwargs())
    duplicate = b'{"schema_version":"duplicate",' + encoded[1:]
    with pytest.raises(ProbeLaunchManifestError, match="strict JSON"):
        parse_probe_launch_manifest_bytes(duplicate)
    with pytest.raises(ProbeLaunchManifestError, match="strict JSON"):
        parse_probe_launch_manifest_bytes(b'{"value":NaN}')
    with pytest.raises(ProbeLaunchManifestError, match="canonical JSON"):
        parse_probe_launch_manifest_bytes(encoded + b"\n")

    parsed = json.loads(encoded)
    reversed_document = {name: parsed[name] for name in reversed(parsed)}
    reordered = json.dumps(
        reversed_document,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    assert reordered != encoded
    with pytest.raises(ProbeLaunchManifestError, match="canonical JSON"):
        parse_probe_launch_manifest_bytes(reordered)


def test_builder_rejects_invalid_paths_hashes_and_deployment() -> None:
    kwargs = _build_kwargs()
    kwargs["fit_catalog_raw_sha256"] = "short"
    with pytest.raises(ProbeLaunchManifestError):
        build_probe_launch_manifest(**kwargs)

    kwargs = _build_kwargs()
    kwargs["data_root"] = "relative"
    with pytest.raises(ProbeLaunchManifestError):
        build_probe_launch_manifest(**kwargs)

    kwargs = _build_kwargs()
    kwargs["probe_deployment"]["unexpected"] = True
    with pytest.raises(ProbeLaunchManifestError, match="probe_deployment"):
        build_probe_launch_manifest(**kwargs)
