from __future__ import annotations

import copy
import hashlib
import json

import pytest

from cluster.f02b_calibration_grid import FIT_TASKS, PROBE_TASKS
from experiments.f02b_calibration_contract import (
    CALIBRATION_ID,
    CALIBRATION_MATRIX_SHA256,
    CATALOG_IDENTITY,
    EXCLUDED_FIT_RECIPE_FIELDS,
    EXECUTION_ENVELOPE_SCHEMA_VERSION,
    F02_CATALOG_SHA256,
    FIT_RECIPE,
    FIT_RECIPE_SHA256,
    FIT_TASK_COUNT,
    FIT_TASK_MATRIX_SHA256,
    MATRIX_HASHES,
    MINIMUM_HOST_MEMORY_BYTES,
    NUMERICAL_POLICY,
    PROBE_TASK_COUNT,
    PROBE_TASK_MATRIX_SHA256,
    RESOURCE_CONTRACT,
    WALLTIME_SECONDS,
    CalibrationContractError,
    build_execution_envelope,
    build_fit_execution_envelope,
    build_payload_binding,
    build_probe_execution_envelope,
    canonical_json_bytes,
    canonical_sha256,
    parse_strict_json_bytes,
    validate_execution_envelope,
    validate_fit_execution_envelope,
    validate_fit_recipe,
    validate_payload_binding,
    validate_probe_execution_envelope,
    validate_resource_contract,
    validate_runtime_allocation,
    verify_numeric_payload_bytes,
)


def _runtime_allocation(partition: str = "short") -> dict[str, object]:
    return {
        "exclusive_node": False,
        "requested_gpu_count": 0,
        "visible_gpu_count": 0,
        "visible_gpu_models": [],
        "visible_gpu_memory_bytes": [],
        "requested_cpus_per_task": 8,
        "available_cpu_count": 8,
        "available_host_memory_bytes": MINIMUM_HOST_MEMORY_BYTES,
        "requested_walltime_seconds": WALLTIME_SECONDS,
        "walltime_limit_seconds": WALLTIME_SECONDS,
        "array_concurrency": 1,
        "partition": partition,
    }


def _payload_bytes(role: str, task_index: int) -> bytes:
    return b"\x00F02b-numeric\xff" + f"-{role}-{task_index}".encode()


def _payload_binding(role: str, task_index: int) -> dict[str, object]:
    payload = _payload_bytes(role, task_index)
    return build_payload_binding(
        role,
        task_index,
        numeric_payload_path=f"artifacts/f02b/{role}/{task_index:03d}.bin",
        numeric_payload_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _fit_envelope(task_index: int = 0) -> dict[str, object]:
    return build_fit_execution_envelope(
        task_index,
        runtime_allocation=_runtime_allocation(),
        payload_binding=_payload_binding("fit", task_index),
    )


def _probe_envelope(task_index: int = 0) -> dict[str, object]:
    return build_probe_execution_envelope(
        task_index,
        runtime_allocation=_runtime_allocation(),
        payload_binding=_payload_binding("probe", task_index),
    )


def _record(envelope: dict[str, object]) -> dict[str, object]:
    record = envelope["canonical_record"]
    assert isinstance(record, dict)
    return record


def _rehash(envelope: dict[str, object]) -> None:
    envelope["canonical_record_sha256"] = canonical_sha256(_record(envelope))


def test_frozen_resource_recipe_numeric_and_catalog_contracts_are_exact() -> None:
    expected_recipe = {
        "train_steps": 20,
        "train_epochs": 0,
        "training_m": 20,
        "kernel": "rbf",
        "outputscale": 1.0,
        "sigma_f": 1e-3,
        "sigma_g": 1e-3,
        "lengthscale": 1.0,
        "lengthscale_init": "median",
        "lengthscale_init_max_points": 2048,
        "use_ard": False,
        "batch_size": 256,
        "lr": 0.01,
        "weight_decay": 0.0,
        "graph_refresh_epochs": 0,
        "learn_lengthscale": True,
        "learn_outputscale": True,
        "learn_sigma_f": True,
        "learn_sigma_g": True,
        "min_sigma_f": 1e-6,
        "min_sigma_g": 0.0,
        "dtype": "float32",
        "device": "cpu",
    }
    assert FIT_RECIPE == expected_recipe
    assert canonical_sha256(FIT_RECIPE) == FIT_RECIPE_SHA256
    assert FIT_RECIPE_SHA256 == "cc4a891ab0f4ee3e0595291aadae961c445c23d1cd537ac8b85cb3888b0f44bb"
    assert not (set(FIT_RECIPE) & EXCLUDED_FIT_RECIPE_FIELDS)
    assert "seed" not in FIT_RECIPE
    assert "candidate_m" not in FIT_RECIPE
    assert not any("cg" in field or "prediction" in field for field in FIT_RECIPE)

    assert RESOURCE_CONTRACT == {
        "exclusive_node": False,
        "requested_gpu_count": 0,
        "required_gpu_model": None,
        "minimum_gpu_memory_bytes": 0,
        "requested_cpus_per_task": 8,
        "minimum_host_memory_bytes": 68_719_476_736,
        "requested_walltime_seconds": 28_800,
        "array_concurrency": 1,
        "allowed_partitions": ("short",),
    }
    assert NUMERICAL_POLICY["source_dtype"] == "float32"
    assert NUMERICAL_POLICY["source_device"] == "cpu"
    assert NUMERICAL_POLICY["canonical_comparison_dtype"] == "float64"
    assert NUMERICAL_POLICY["canonical_comparison_device"] == "cpu"
    assert NUMERICAL_POLICY["physical_compute_dtype"] == "float64"
    assert NUMERICAL_POLICY["physical_compute_device"] == "cpu"
    assert NUMERICAL_POLICY["cuda_matmul_allow_tf32"] is False
    assert NUMERICAL_POLICY["cudnn_allow_tf32"] is False
    assert NUMERICAL_POLICY["float32_matmul_precision"] == "highest"
    assert F02_CATALOG_SHA256 == (
        "2dee429bdaf50cc78cb40ba8b038f7e4731bb07127927c55607b528c1db66942"
    )
    assert CATALOG_IDENTITY == {
        "catalog_sha256": F02_CATALOG_SHA256,
        "binding_scope": "identity-only-no-payload-access",
    }

    with pytest.raises(TypeError):
        FIT_RECIPE["train_steps"] = 21
    with pytest.raises(TypeError):
        RESOURCE_CONTRACT["required_gpu_model"] = "H100"
    with pytest.raises(TypeError):
        NUMERICAL_POLICY["cuda_matmul_allow_tf32"] = True


def test_matrix_hashes_are_imported_as_one_immutable_three_hash_binding() -> None:
    assert FIT_TASK_COUNT == 45
    assert PROBE_TASK_COUNT == 122
    assert MATRIX_HASHES == {
        "fit_task_matrix_sha256": FIT_TASK_MATRIX_SHA256,
        "probe_task_matrix_sha256": PROBE_TASK_MATRIX_SHA256,
        "calibration_matrix_sha256": CALIBRATION_MATRIX_SHA256,
    }
    assert all(len(value) == 64 for value in MATRIX_HASHES.values())


def test_all_45_fit_envelopes_are_deterministic_complete_and_strict() -> None:
    for task in FIT_TASKS:
        first = _fit_envelope(task.task_index)
        second = _fit_envelope(task.task_index)
        assert first == second
        assert validate_execution_envelope(first) == first
        assert (
            validate_fit_execution_envelope(
                first,
                expected_task_index=task.task_index,
            )
            == first
        )
        record = _record(first)
        assert record["task_record"] == task.as_record()
        assert record["task_role"] == "fit"
        assert record["task_index"] == task.task_index
        assert record["fit_recipe"] == FIT_RECIPE
        assert record["fit_recipe_sha256"] == FIT_RECIPE_SHA256
        assert "seed" not in record
        assert "seed" not in record["fit_recipe"]
        assert first["canonical_record_sha256"] == canonical_sha256(record)


def test_all_122_probe_envelopes_bind_exact_probe_roles_and_fit_recipe() -> None:
    for task in PROBE_TASKS:
        first = _probe_envelope(task.task_index)
        second = build_execution_envelope(
            "probe",
            task.task_index,
            runtime_allocation=_runtime_allocation(),
            payload_binding=_payload_binding("probe", task.task_index),
        )
        assert first == second
        assert (
            validate_probe_execution_envelope(
                first,
                expected_task_index=task.task_index,
            )
            == first
        )
        record = _record(first)
        assert record["task_record"] == task.as_record()
        assert record["task_record"]["role"] == task.role
        assert record["task_record"]["replica"] in (0, 1, 2)
        assert record["fit_recipe"] == FIT_RECIPE


def test_envelope_hash_is_nonrecursive_and_numeric_payload_is_not_embedded() -> None:
    payload = _payload_bytes("fit", 0)
    envelope = _fit_envelope()
    record = _record(envelope)
    encoded = canonical_json_bytes(envelope)

    assert set(envelope) == {"canonical_record", "canonical_record_sha256"}
    assert "canonical_record_sha256" not in record
    assert "numeric_payload" not in record
    assert "numeric_payload" not in record["payload_binding"]
    assert payload not in encoded
    assert json.loads(encoded) == envelope
    assert envelope["canonical_record_sha256"] == canonical_sha256(record)
    assert canonical_json_bytes(_fit_envelope()) == encoded


def test_canonical_json_and_parser_are_strict_and_deterministic() -> None:
    first = {"z": [3, 2, 1], "a": {"b": 1.0}}
    second = {"a": {"b": 1.0}, "z": [3, 2, 1]}
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert canonical_sha256(first) == canonical_sha256(second)
    assert parse_strict_json_bytes(canonical_json_bytes(first)) == first

    for value in (
        {"bad": float("nan")},
        {"bad": float("inf")},
        {"bad": float("-inf")},
        {1: "non-string-key"},
        {"bad": object()},
    ):
        with pytest.raises(CalibrationContractError):
            canonical_json_bytes(value)
    recursive: list[object] = []
    recursive.append(recursive)
    with pytest.raises(CalibrationContractError, match="recursive"):
        canonical_json_bytes(recursive)
    with pytest.raises(CalibrationContractError, match="duplicate"):
        parse_strict_json_bytes(b'{"a":1,"a":2}')
    with pytest.raises(CalibrationContractError, match="nonfinite"):
        parse_strict_json_bytes(b'{"a":NaN}')
    with pytest.raises(CalibrationContractError):
        parse_strict_json_bytes("{}")  # type: ignore[arg-type]


def test_payload_binding_is_path_and_raw_bytes_identity_only() -> None:
    payload = b"not JSON and never parsed: \x00\xff\xfe"
    digest = hashlib.sha256(payload).hexdigest()
    binding = build_payload_binding(
        "probe",
        121,
        numeric_payload_path="artifacts/f02b/probe/121.raw",
        numeric_payload_sha256=digest,
    )
    assert (
        validate_payload_binding(
            binding,
            expected_task_role="probe",
            expected_task_index=121,
        )
        == binding
    )
    assert (
        verify_numeric_payload_bytes(
            binding,
            payload,
            expected_task_role="probe",
            expected_task_index=121,
        )
        == binding
    )
    with pytest.raises(CalibrationContractError, match="raw-bytes"):
        verify_numeric_payload_bytes(binding, payload + b"changed")
    with pytest.raises(CalibrationContractError, match="raw bytes"):
        verify_numeric_payload_bytes(binding, bytearray(payload))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("path", "digest"),
    [
        ("/absolute.bin", "0" * 64),
        ("../escape.bin", "0" * 64),
        ("a/../escape.bin", "0" * 64),
        ("a//noncanonical.bin", "0" * 64),
        ("a\\windows.bin", "0" * 64),
        ("a\nnewline.bin", "0" * 64),
        ("a\rcarriage-return.bin", "0" * 64),
        ("a\ttab.bin", "0" * 64),
        ("", "0" * 64),
        ("valid.bin", "A" * 64),
        ("valid.bin", "0" * 63),
    ],
)
def test_payload_binding_rejects_malformed_paths_and_hashes(path: str, digest: str) -> None:
    with pytest.raises(CalibrationContractError):
        build_payload_binding(
            "fit",
            0,
            numeric_payload_path=path,
            numeric_payload_sha256=digest,
        )


def test_payload_binding_rejects_unknown_missing_bool_index_and_role_mismatch() -> None:
    binding = _payload_binding("fit", 0)
    unknown = {**binding, "numeric_payload": [1.0]}
    missing = dict(binding)
    missing.pop("numeric_payload_path")
    boolean = {**binding, "task_index": False}
    for malformed in (unknown, missing, boolean):
        with pytest.raises(CalibrationContractError):
            validate_payload_binding(malformed)
    with pytest.raises(CalibrationContractError, match="role"):
        validate_payload_binding(binding, expected_task_role="probe")
    with pytest.raises(CalibrationContractError, match="index"):
        validate_payload_binding(binding, expected_task_index=1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("exclusive_node", True),
        ("requested_gpu_count", True),
        ("requested_gpu_count", 1),
        ("visible_gpu_count", 1),
        ("visible_gpu_count", True),
        ("visible_gpu_models", ["NVIDIA H100"]),
        ("visible_gpu_memory_bytes", [1]),
        ("visible_gpu_memory_bytes", [float("nan")]),
        ("requested_cpus_per_task", 7),
        ("requested_cpus_per_task", True),
        ("available_cpu_count", 7),
        ("available_cpu_count", True),
        ("available_host_memory_bytes", 64_000_000_000),
        ("requested_walltime_seconds", 28_799),
        ("walltime_limit_seconds", 28_799),
        ("array_concurrency", 2),
        ("array_concurrency", True),
        ("partition", "gpu"),
    ],
)
def test_runtime_allocation_fails_closed_on_resource_mismatch(
    field: str,
    value: object,
) -> None:
    allocation = _runtime_allocation()
    allocation[field] = value
    with pytest.raises(CalibrationContractError):
        validate_runtime_allocation(allocation)


def test_runtime_allocation_rejects_unknown_missing_gpu_and_unregistered_partition() -> None:
    assert validate_runtime_allocation(_runtime_allocation("short"))["partition"] == "short"
    with pytest.raises(CalibrationContractError):
        validate_runtime_allocation(_runtime_allocation("interactivegpu"))
    visible_gpu = _runtime_allocation()
    visible_gpu["visible_gpu_count"] = 1
    visible_gpu["visible_gpu_models"] = ["NVIDIA L40S"]
    visible_gpu["visible_gpu_memory_bytes"] = [48_000_000_000]
    with pytest.raises(CalibrationContractError):
        validate_runtime_allocation(visible_gpu)
    larger = _runtime_allocation()
    larger["available_cpu_count"] = 32
    larger["available_host_memory_bytes"] = MINIMUM_HOST_MEMORY_BYTES + 1
    assert validate_runtime_allocation(larger) == larger
    missing = _runtime_allocation()
    missing.pop("visible_gpu_models")
    unknown = {**_runtime_allocation(), "slurm_job_id": "123"}
    for malformed in (missing, unknown):
        with pytest.raises(CalibrationContractError, match="fields mismatch"):
            validate_runtime_allocation(malformed)


def test_fit_recipe_and_resource_validators_reject_unknown_missing_and_bool_as_int() -> None:
    assert validate_fit_recipe(dict(FIT_RECIPE)) == FIT_RECIPE
    assert validate_resource_contract(dict(RESOURCE_CONTRACT)) == json.loads(
        canonical_json_bytes(RESOURCE_CONTRACT)
    )
    unknown_recipe = {**FIT_RECIPE, "candidate_m": [50]}
    missing_recipe = dict(FIT_RECIPE)
    missing_recipe.pop("kernel")
    boolean_recipe = {**FIT_RECIPE, "train_steps": True}
    for malformed in (unknown_recipe, missing_recipe, boolean_recipe):
        with pytest.raises(CalibrationContractError):
            validate_fit_recipe(malformed)
    wrong_resource = dict(RESOURCE_CONTRACT)
    wrong_resource["requested_cpus_per_task"] = True
    with pytest.raises(CalibrationContractError):
        validate_resource_contract(wrong_resource)


def test_envelope_rejects_unknown_missing_and_outer_hash_tampering() -> None:
    envelope = _fit_envelope()
    unknown = copy.deepcopy(envelope)
    unknown["numeric_payload"] = [1.0]
    missing = copy.deepcopy(envelope)
    missing.pop("canonical_record_sha256")
    bad_hash = copy.deepcopy(envelope)
    bad_hash["canonical_record_sha256"] = "0" * 64
    record_unknown = copy.deepcopy(envelope)
    _record(record_unknown)["seed"] = 11
    _rehash(record_unknown)
    record_missing = copy.deepcopy(envelope)
    _record(record_missing).pop("numerical_policy")
    _rehash(record_missing)
    for malformed in (unknown, missing, bad_hash, record_unknown, record_missing):
        with pytest.raises(CalibrationContractError):
            validate_execution_envelope(malformed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "f02b_calibration_execution_envelope_v1"),
        ("calibration_id", "F02B_OTHER"),
        ("task_index", False),
        ("task_role", "probe"),
        ("fit_recipe_sha256", "0" * 64),
    ],
)
def test_envelope_rejects_rehashed_identity_role_index_and_recipe_mismatches(
    field: str,
    value: object,
) -> None:
    envelope = _fit_envelope()
    _record(envelope)[field] = value
    _rehash(envelope)
    with pytest.raises(CalibrationContractError):
        validate_execution_envelope(envelope)


def test_envelope_rejects_rehashed_record_hash_catalog_resource_and_policy_mismatches() -> None:
    mutations: list[tuple[str, str, object]] = [
        ("task_record", "replica", 101),
        ("task_record", "task_index", False),
        ("task_record", "seed", 29),
        ("matrix_hashes", "fit_task_matrix_sha256", "0" * 64),
        ("catalog_identity", "catalog_sha256", "0" * 64),
        ("fit_recipe", "train_steps", 21),
        ("resource_contract", "requested_cpus_per_task", 15),
        ("numerical_policy", "cuda_matmul_allow_tf32", True),
        ("numerical_policy", "canonical_comparison_dtype", "float32"),
        ("numerical_policy", "physical_compute_device", "cuda"),
        ("runtime_allocation", "visible_gpu_models", ["NVIDIA H100"]),
        ("payload_binding", "task_role", "probe"),
    ]
    for section, field, value in mutations:
        envelope = _fit_envelope()
        nested = _record(envelope)[section]
        assert isinstance(nested, dict)
        nested[field] = value
        _rehash(envelope)
        with pytest.raises(CalibrationContractError):
            validate_execution_envelope(envelope)


def test_probe_record_role_and_expected_wrapper_identity_fail_closed() -> None:
    envelope = _probe_envelope(0)
    task_record = _record(envelope)["task_record"]
    assert isinstance(task_record, dict)
    task_record["role"] = "reproducibility"
    _rehash(envelope)
    with pytest.raises(CalibrationContractError, match="task_record|role"):
        validate_probe_execution_envelope(envelope)

    with pytest.raises(CalibrationContractError, match="role"):
        validate_fit_execution_envelope(_probe_envelope(0))
    with pytest.raises(CalibrationContractError, match="index"):
        validate_probe_execution_envelope(_probe_envelope(0), expected_task_index=1)


@pytest.mark.parametrize(
    ("role", "index"),
    [
        ("fit", -1),
        ("fit", FIT_TASK_COUNT),
        ("fit", False),
        ("probe", -1),
        ("probe", PROBE_TASK_COUNT),
        ("probe", True),
        ("unknown", 0),
    ],
)
def test_build_rejects_invalid_roles_indices_and_bool_as_int(role: str, index: object) -> None:
    with pytest.raises(CalibrationContractError):
        build_execution_envelope(
            role,
            index,  # type: ignore[arg-type]
            runtime_allocation=_runtime_allocation(),
            payload_binding=_payload_binding("fit", 0),
        )


def test_public_contract_contains_no_label_corpus_file_or_scheduler_execution_api() -> None:
    envelope = _fit_envelope()
    encoded = canonical_json_bytes(envelope).decode()
    assert EXECUTION_ENVELOPE_SCHEMA_VERSION in encoded
    assert CALIBRATION_ID in encoded
    assert F02_CATALOG_SHA256 in encoded
    assert "label" not in encoded.lower()
    assert "corpus" not in encoded.lower()
    assert "slurm" not in encoded.lower()
    assert "numeric_payload_path" in encoded
    assert "numeric_payload_sha256" in encoded
