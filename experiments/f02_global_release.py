"""Fail-closed global release recipe for F02 confirmatory evaluation.

The recipe built here is deliberately broader than one model runner or one
corpus.  It binds all 50 confirmatory corpora and the three preregistered model
families, with three paired optimizer seeds per family.  Consequently a valid
recipe always describes exactly 450 task identities.

This module does not read any N-body arrays and does not enable test execution.
Version 1 is a non-executable structural scaffold: development evidence has not
yet received semantic validation and the cluster-side release mutations are
source-disabled until the missing runner and immutable result store exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

from data.generate_nbody_confirmatory import ConfirmatoryConfig

RECIPE_SCHEMA_VERSION = "f02_global_confirmatory_recipe_v1"
METHOD_SELECTION_SCHEMA_VERSION = "f02_method_selection_v1"
RECIPE_SCAFFOLD_STATUS = "non_executable_scaffold_v1"
DEVELOPMENT_EVIDENCE_SEMANTICALLY_VERIFIED = False
STUDY_ID = "F02"
EVALUATION_SPLIT = "test"
EVALUATION_DESIGN = "primary"
RECIPE_DEFAULT_PATH = "releases/f02_global_confirmatory_recipe.json"

ALL_REPLICAS = (0, 1, 2, *range(101, 111))
DEVELOPMENT_REPLICAS = (0, 1, 2)
CONFIRMATORY_REPLICAS = tuple(range(101, 111))
PARTICLE_COUNTS = (2, 4, 6, 8, 10)
N_DIMS = 3
DIMENSIONS = tuple(2 * n_particles * N_DIMS for n_particles in PARTICLE_COUNTS)
SEEDS = (11, 29, 47)
UPDATE_BUDGETS = (20, 50, 100)
ORBIT_M_CANDIDATES = (75, 100, 150, 200)
REFERENCE_M = 50

METHOD_IDS = ("internal-shared-fit", "dsoftki-512", "ddsvgp-512")
METHOD_ARMS = {
    "internal-shared-fit": (
        "TERA-50",
        "ORBIT-50",
        "ORBIT-resource",
        "value-only-conditional-50",
    ),
    "dsoftki-512": ("DSoftKI-512",),
    "ddsvgp-512": ("DDSVGP-512",),
}

EXPECTED_BUNDLE_COUNT = len(CONFIRMATORY_REPLICAS) * len(PARTICLE_COUNTS)
EXPECTED_TASKS_PER_METHOD = EXPECTED_BUNDLE_COUNT * len(SEEDS)
EXPECTED_METHOD_COUNT = len(METHOD_IDS)
EXPECTED_TASK_COUNT = EXPECTED_TASKS_PER_METHOD * EXPECTED_METHOD_COUNT

BLOCKED_PROTOCOL_PATHS = frozenset(
    {
        "docs/F02_NBODY_PROTOCOL.md",
        "docs/F02B_NBODY_PROTOCOL.md",
    }
)
BLOCKED_PROTOCOL_SHA256 = frozenset(
    {"6a103772e99e953d71a13f9655faea91f532613d750610b8358b6a1cc2bb2df8"}
)
DEPENDENCY_PATHS = ("pyproject.toml", "uv.lock")
TERA_GITLINK_PATH = "gp/tera/vendor"
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_RELEASE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


class GlobalRecipeError(RuntimeError):
    """Raised when a global release recipe cannot be trusted."""


@dataclass(frozen=True, slots=True)
class MethodSelection:
    """One method's globally selected development-only configuration."""

    method_id: str
    optimizer_updates: int
    configuration: dict[str, Any]
    selection_evidence: tuple[dict[str, str], ...]
    orbit_m_by_dimension: dict[int, int] | None = None


@dataclass(frozen=True, slots=True)
class SourceRelease:
    """Git-object identity of the source commit preceding the recipe-only commit."""

    source_commit: str
    source_tree: str
    tera_gitlink_commit: str
    protocol_path: str
    protocol_sha256: str
    dependency_sha256: dict[str, str]


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON deterministically, rejecting NaN and infinity."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise GlobalRecipeError("value is not finite canonical JSON") from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise GlobalRecipeError(f"cannot hash file: {path}") from error
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise GlobalRecipeError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def read_strict_json_bytes(value: bytes, *, label: str) -> dict[str, Any]:
    """Parse strict JSON bytes without duplicate keys or nonfinite constants."""

    try:
        parsed = json.loads(
            value.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                GlobalRecipeError(f"nonfinite JSON constant in {label}: {token}")
            ),
        )
    except GlobalRecipeError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise GlobalRecipeError(f"cannot parse {label}") from error
    if not isinstance(parsed, dict):
        raise GlobalRecipeError(f"{label} root must be an object")
    canonical_json_bytes(parsed)
    return parsed


def read_strict_json(path: str | Path, *, label: str) -> dict[str, Any]:
    """Read one strict JSON object without duplicate keys or nonfinite constants."""

    source = Path(path)
    try:
        value = source.read_bytes()
    except OSError as error:
        raise GlobalRecipeError(f"cannot read {label}: {source}") from error
    return read_strict_json_bytes(value, label=label)


def _run_git(repo_root: Path, *arguments: str, check: bool = True) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=check,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise GlobalRecipeError(f"git {' '.join(arguments)} failed") from error
    return completed.stdout


def _git_object_bytes(repo_root: Path, commit: str, path: str) -> bytes:
    return _run_git(repo_root, "show", f"{commit}:{path}")


def _validate_protocol_path(path: str) -> str:
    if not isinstance(path, str) or not path:
        raise GlobalRecipeError("protocol_path must be a nonempty repository-relative path")
    pure = PurePosixPath(path)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or path != pure.as_posix()
        or not pure.parts
        or pure.parts[0] != "docs"
        or pure.suffix.lower() != ".md"
    ):
        raise GlobalRecipeError("protocol_path must be a canonical docs/*.md source path")
    if path in BLOCKED_PROTOCOL_PATHS:
        raise GlobalRecipeError(f"protocol path is explicitly blocked and not releasable: {path}")
    return path


def _protocol_binding_sha256(protocol_id: str, protocol_path: str, protocol_sha256: str) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "protocol_id": protocol_id,
                "protocol_path": protocol_path,
                "protocol_sha256": protocol_sha256,
            }
        )
    )


def resolve_source_release(
    repo_root: str | Path,
    source_commit: str,
    *,
    protocol_path: str,
) -> SourceRelease:
    """Resolve source identity entirely from immutable Git objects."""

    root = Path(repo_root).resolve()
    protocol_path = _validate_protocol_path(protocol_path)
    commit = _run_git(root, "rev-parse", "--verify", f"{source_commit}^{{commit}}").decode().strip()
    tree = _run_git(root, "rev-parse", f"{commit}^{{tree}}").decode().strip()
    gitlink = _run_git(root, "rev-parse", f"{commit}:{TERA_GITLINK_PATH}").decode().strip()
    if not _HEX40.fullmatch(commit) or not _HEX40.fullmatch(tree):
        raise GlobalRecipeError("source commit/tree are not full SHA-1 object IDs")
    if not _HEX40.fullmatch(gitlink):
        raise GlobalRecipeError("TERA source entry is not a full gitlink commit")
    protocol_bytes = _git_object_bytes(root, commit, protocol_path)
    protocol_sha256 = sha256_bytes(protocol_bytes)
    if protocol_sha256 in BLOCKED_PROTOCOL_SHA256:
        raise GlobalRecipeError(
            "protocol Git blob SHA-256 is explicitly blocked and not releasable"
        )
    if re.search(rb"\bDRAFT\b", protocol_bytes, flags=re.IGNORECASE):
        raise GlobalRecipeError("release protocol Git blob must not contain a DRAFT marker")
    dependencies = {
        path: sha256_bytes(_git_object_bytes(root, commit, path)) for path in DEPENDENCY_PATHS
    }
    return SourceRelease(
        source_commit=commit,
        source_tree=tree,
        tera_gitlink_commit=gitlink,
        protocol_path=protocol_path,
        protocol_sha256=protocol_sha256,
        dependency_sha256=dependencies,
    )


def _require_mapping(parent: Mapping[str, Any], key: str, label: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise GlobalRecipeError(f"{label}.{key} must be an object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise GlobalRecipeError(
            f"{label} fields mismatch: missing={missing}, unexpected={unexpected}"
        )


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _validate_catalog_header(report: dict[str, Any]) -> list[dict[str, Any]]:
    if report.get("schema_version") != 1:
        raise GlobalRecipeError("unsupported F02 catalog schema_version")
    if report.get("catalog_type") != "f02_nbody_confirmatory_data":
        raise GlobalRecipeError("unexpected F02 catalog type")
    if report.get("overall_ready") is not True:
        raise GlobalRecipeError("F02 catalog is not overall_ready")

    inputs = _require_mapping(report, "input", "catalog")
    expected_inputs = {
        "replicas": list(ALL_REPLICAS),
        "development_replicas": list(DEVELOPMENT_REPLICAS),
        "particle_counts": list(PARTICLE_COUNTS),
        "n_dims": N_DIMS,
    }
    if any(inputs.get(key) != value for key, value in expected_inputs.items()):
        raise GlobalRecipeError("F02 catalog input grid is not the preregistered 65-bundle grid")
    expected_generation = {
        "n_trajectories": 100,
        "steps_per_trajectory": 100,
        "dt": 0.01,
        "mass_seed": 1729,
        "trajectory_seed": 2718,
        "split_seed": 31415,
        "validation_seed": 1618,
    }
    if inputs.get("generation") != expected_generation:
        raise GlobalRecipeError("F02 catalog generator settings do not match the protocol")

    accounting = _require_mapping(report, "task_accounting", "catalog")
    counts = _require_mapping(accounting, "status_counts", "catalog.task_accounting")
    expected_total = len(ALL_REPLICAS) * len(PARTICLE_COUNTS)
    if (
        accounting.get("expected_task_count") != expected_total
        or accounting.get("all_expected_tasks_valid") is not True
        or accounting.get("all_expected_tasks_unique") is not True
        or counts.get("valid") != expected_total
        or any(counts.get(name) != 0 for name in ("missing", "failed", "invalid", "unexpected"))
    ):
        raise GlobalRecipeError("F02 catalog task accounting is incomplete")

    provenance = _require_mapping(report, "provenance", "catalog")
    flags = (
        "verified",
        "same_commit",
        "same_tree",
        "same_submodules",
        "same_source_hashes",
        "same_source_manifest",
        "same_slurm_array_job",
        "same_repo_root",
    )
    if any(provenance.get(flag) is not True for flag in flags):
        raise GlobalRecipeError("F02 catalog provenance is not globally verified")
    for field in ("git_commit", "git_tree"):
        if not isinstance(provenance.get(field), str) or not _HEX40.fullmatch(provenance[field]):
            raise GlobalRecipeError(f"F02 catalog provenance {field} is invalid")

    independence = _require_mapping(report, "independence", "catalog")
    if (
        independence.get("verified") is not True
        or independence.get("candidate_bundle_count") != expected_total
        or independence.get("unique_bundle_count") != expected_total
        or independence.get("duplicate_dataset_content_sha256") != []
    ):
        raise GlobalRecipeError("F02 catalog corpus independence is incomplete")

    catalog = _require_mapping(report, "catalog", "catalog")
    bundles = catalog.get("bundles")
    if (
        not isinstance(bundles, list)
        or catalog.get("bundle_count") != expected_total
        or len(bundles) != expected_total
        or catalog.get("phase_counts") != {"development": 15, "confirmatory": 50}
    ):
        raise GlobalRecipeError("F02 catalog bundle/phase accounting is incomplete")
    return bundles


def _validate_catalog_canonical_path(
    catalog_path: str | Path,
    report: dict[str, Any],
) -> str:
    source = Path(catalog_path)
    if source.is_symlink() or not source.is_file():
        raise GlobalRecipeError("F02 catalog must be a regular non-symlink file")
    inputs = _require_mapping(report, "input", "catalog")
    provenance = _require_mapping(report, "provenance", "catalog")
    raw_run_root = inputs.get("run_root")
    raw_repo_root = provenance.get("repo_root")
    if not isinstance(raw_run_root, str) or not isinstance(raw_repo_root, str):
        raise GlobalRecipeError("F02 catalog canonical run/repository roots are missing")
    run_root = Path(raw_run_root)
    repo_root = Path(raw_repo_root)
    if not run_root.is_absolute() or not repo_root.is_absolute():
        raise GlobalRecipeError("F02 catalog run/repository roots must be absolute")
    resolved_run_root = run_root.resolve()
    resolved_repo_root = repo_root.resolve()
    if run_root != resolved_run_root or repo_root != resolved_repo_root:
        raise GlobalRecipeError(
            "F02 catalog run/repository roots must be canonical non-symlink paths"
        )
    run_root = resolved_run_root
    repo_root = resolved_repo_root
    try:
        run_root.relative_to(repo_root)
    except ValueError as error:
        raise GlobalRecipeError(
            "F02 catalog run_root must be inside provenance.repo_root"
        ) from error
    expected = (run_root / "catalog.json").resolve()
    if source.absolute() != expected or source.resolve() != expected:
        raise GlobalRecipeError("F02 catalog path must equal canonical input.run_root/catalog.json")
    return str(expected)


def _stable_bundle_identity(entry: dict[str, Any], expected_index: int) -> dict[str, Any]:
    replica_index, particle_index = divmod(expected_index, len(PARTICLE_COUNTS))
    replica = ALL_REPLICAS[replica_index]
    n_particles = PARTICLE_COUNTS[particle_index]
    dimension = 2 * n_particles * N_DIMS
    phase = "development" if replica in DEVELOPMENT_REPLICAS else "confirmatory"
    expected_config = asdict(
        ConfirmatoryConfig(n_particles=n_particles, n_dims=N_DIMS, replica=replica)
    )
    expected_fields = {
        "task_index": expected_index,
        "phase": phase,
        "replica": replica,
        "n_particles": n_particles,
        "n_dims": N_DIMS,
        "D": dimension,
        "config": expected_config,
        "eligible_for_catalog": True,
        "unique_content": True,
    }
    if any(entry.get(key) != value for key, value in expected_fields.items()):
        raise GlobalRecipeError("F02 catalog bundle grid/config/phase is mismatched")
    paths = _require_mapping(entry, "paths", "catalog bundle")
    hashes = _require_mapping(entry, "hashes", "catalog bundle")
    stem = f"nbody_fixedmass_n{n_particles}_d{N_DIMS}_replica{replica}"
    filenames = {
        "dataset": f"{stem}.npz",
        "metadata": f"{stem}.metadata.json",
        "sha256_manifest": f"{stem}.sha256.json",
    }
    for key, filename in filenames.items():
        if not isinstance(paths.get(key), str) or Path(paths[key]).name != filename:
            raise GlobalRecipeError("F02 catalog bundle filename is mismatched")
    hash_keys = (
        "dataset_file_sha256",
        "metadata_file_sha256",
        "sha256_manifest_file_sha256",
        "dataset_content_sha256",
    )
    if any(not _valid_sha256(hashes.get(key)) for key in hash_keys):
        raise GlobalRecipeError("F02 catalog bundle hash is invalid")
    return {
        "bundle_id": f"replica-{replica}-n-{n_particles}-d-{dimension}",
        "catalog_task_index": expected_index,
        "replica": replica,
        "n_particles": n_particles,
        "n_dims": N_DIMS,
        "dimension": dimension,
        "dataset_filename": filenames["dataset"],
        "metadata_filename": filenames["metadata"],
        "manifest_filename": filenames["sha256_manifest"],
        "hashes": {key: hashes[key] for key in hash_keys},
        "generator_config": expected_config,
    }


def confirmatory_bundle_identities(catalog: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Extract all 50 stable confirmatory identities from an exact 65-bundle catalog."""

    entries = _validate_catalog_header(catalog)
    all_identities = tuple(
        _stable_bundle_identity(entry, index) for index, entry in enumerate(entries)
    )
    all_content_hashes = [
        identity["hashes"]["dataset_content_sha256"] for identity in all_identities
    ]
    if len(set(all_content_hashes)) != len(all_identities):
        raise GlobalRecipeError("full catalog bundle content hashes are not unique")
    identities = tuple(
        identity for identity in all_identities if identity["replica"] in CONFIRMATORY_REPLICAS
    )
    if len(identities) != EXPECTED_BUNDLE_COUNT:
        raise GlobalRecipeError("global recipe must contain exactly 50 confirmatory bundles")
    content_hashes = [identity["hashes"]["dataset_content_sha256"] for identity in identities]
    if len(set(content_hashes)) != EXPECTED_BUNDLE_COUNT:
        raise GlobalRecipeError("confirmatory bundle content hashes are not unique")
    return identities


def _normalize_evidence(
    value: Any,
    *,
    method_id: str,
) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list) or not value:
        raise GlobalRecipeError("method selection_evidence must be a nonempty list")
    normalized: list[dict[str, str]] = []
    roles: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise GlobalRecipeError("method selection evidence entries must be objects")
        _require_exact_keys(item, {"role", "sha256"}, f"selection_evidence[{index}]")
        role = item["role"]
        digest = item["sha256"]
        if not isinstance(role, str) or not role or role in roles:
            raise GlobalRecipeError("method selection evidence roles must be unique and nonempty")
        if not _valid_sha256(digest):
            raise GlobalRecipeError("method selection evidence SHA-256 is invalid")
        roles.add(role)
        normalized.append({"role": role, "sha256": digest})
    expected_roles = (
        {"optimizer_selection_report", "orbit_resource_selection_report"}
        if method_id == "internal-shared-fit"
        else {"optimizer_selection_report", "runtime_dependency_lock"}
    )
    if roles != expected_roles:
        raise GlobalRecipeError(
            f"{method_id} selection evidence roles must be exactly {sorted(expected_roles)}"
        )
    return tuple(sorted(normalized, key=lambda item: item["role"]))


def _normalize_schedule(value: Any) -> dict[int, int]:
    if not isinstance(value, dict):
        raise GlobalRecipeError("internal selection requires orbit_m_by_dimension")
    schedule: dict[int, int] = {}
    for raw_dimension, raw_m in value.items():
        try:
            dimension = int(raw_dimension)
        except (TypeError, ValueError) as error:
            raise GlobalRecipeError("ORBIT schedule dimensions must be integers") from error
        if str(dimension) != str(raw_dimension):
            raise GlobalRecipeError("ORBIT schedule dimension keys must be canonical integers")
        if isinstance(raw_m, bool) or not isinstance(raw_m, int):
            raise GlobalRecipeError("ORBIT schedule m values must be integers")
        schedule[dimension] = raw_m
    if set(schedule) != set(DIMENSIONS):
        raise GlobalRecipeError(f"ORBIT schedule must cover exactly dimensions {DIMENSIONS}")
    if any(m not in ORBIT_M_CANDIDATES for m in schedule.values()):
        raise GlobalRecipeError("ORBIT schedule contains a non-preregistered m")
    return dict(sorted(schedule.items()))


def _configuration_key(dimension: int, seed: int) -> str:
    return f"D{dimension}-seed{seed}"


def _validate_external_configuration_payload(
    payload: dict[str, Any],
    *,
    method_id: str,
    dimension: int,
    seed: int,
    optimizer_updates: int,
) -> None:
    try:
        from experiments.f02_external_adapter import (
            ARTIFACT_SCHEMA_VERSION,
            CONFIG_SCHEMA_VERSION,
            RESULT_SCHEMA_VERSION,
            validate_external_config_payload,
        )

        config = validate_external_config_payload(payload)
    except (ImportError, RuntimeError, TypeError, ValueError) as error:
        raise GlobalRecipeError("external method configuration is not adapter-valid") from error
    if (
        CONFIG_SCHEMA_VERSION != "f02_external_config_v1"
        or ARTIFACT_SCHEMA_VERSION != "f02_external_artifact_v1"
        or RESULT_SCHEMA_VERSION != "f02_external_result_v1"
        or config.method_id != method_id
        or config.dimension != dimension
        or config.seed != seed
        or config.selected_updates != optimizer_updates
    ):
        raise GlobalRecipeError("external method configuration identity/budget is mismatched")


def _validate_internal_configuration_payload(
    payload: dict[str, Any],
    *,
    dimension: int,
    seed: int,
    optimizer_updates: int,
    orbit_m: int,
) -> None:
    try:
        from experiments.f02_internal_task import InternalTaskConfig

        expected = InternalTaskConfig(
            train_steps=optimizer_updates,
            seed=seed,
            candidate_m=(orbit_m,),
            dtype="float32",
            device="cuda",
        )
        expected_payload = json.loads(canonical_json_bytes(asdict(expected)))
    except (ImportError, RuntimeError, TypeError, ValueError) as error:
        raise GlobalRecipeError(
            "cannot construct the frozen internal task configuration"
        ) from error
    if payload != expected_payload:
        raise GlobalRecipeError(
            f"internal task configuration for D={dimension}, seed={seed} is not the frozen config"
        )


def _normalize_configuration_grid(
    value: Any,
    *,
    method_id: str,
    optimizer_updates: int,
    orbit_m_by_dimension: dict[int, int] | None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GlobalRecipeError("method configuration must be an object")
    _require_exact_keys(value, {"by_dimension_and_seed"}, "method configuration")
    records = value["by_dimension_and_seed"]
    if not isinstance(records, list):
        raise GlobalRecipeError("configuration.by_dimension_and_seed must be a list")
    expected_coordinates = [(dimension, seed) for dimension in DIMENSIONS for seed in SEEDS]
    if len(records) != len(expected_coordinates):
        raise GlobalRecipeError("method configuration must contain exactly 15 D/seed payloads")
    normalized: list[dict[str, Any]] = []
    for index, (raw, (dimension, seed)) in enumerate(
        zip(records, expected_coordinates, strict=True)
    ):
        if not isinstance(raw, dict):
            raise GlobalRecipeError("method configuration records must be objects")
        _require_exact_keys(
            raw,
            {"configuration_id", "dimension", "seed", "payload", "sha256"},
            f"method configuration record {index}",
        )
        configuration_id = _configuration_key(dimension, seed)
        if (
            raw["configuration_id"] != configuration_id
            or raw["dimension"] != dimension
            or raw["seed"] != seed
        ):
            raise GlobalRecipeError(
                "method configuration D/seed ordering or identity is mismatched"
            )
        payload = raw["payload"]
        if not isinstance(payload, dict) or not payload:
            raise GlobalRecipeError("method task configuration payload must be nonempty")
        digest = sha256_bytes(canonical_json_bytes(payload))
        if raw["sha256"] != digest:
            raise GlobalRecipeError("method task configuration SHA-256 is mismatched")
        if method_id == "internal-shared-fit":
            if orbit_m_by_dimension is None:
                raise GlobalRecipeError("internal configuration requires the ORBIT schedule")
            _validate_internal_configuration_payload(
                payload,
                dimension=dimension,
                seed=seed,
                optimizer_updates=optimizer_updates,
                orbit_m=orbit_m_by_dimension[dimension],
            )
        else:
            _validate_external_configuration_payload(
                payload,
                method_id=method_id,
                dimension=dimension,
                seed=seed,
                optimizer_updates=optimizer_updates,
            )
        normalized.append(
            {
                "configuration_id": configuration_id,
                "dimension": dimension,
                "seed": seed,
                "payload": json.loads(canonical_json_bytes(payload)),
                "sha256": digest,
            }
        )
    return {"by_dimension_and_seed": normalized}


def parse_method_selection(document: dict[str, Any], expected_method: str) -> MethodSelection:
    """Validate a complete method selection without interpreting adapter-owned fields."""

    _require_exact_keys(
        document,
        {
            "schema_version",
            "method_id",
            "optimizer_updates",
            "configuration",
            "selection_evidence",
            "orbit_m_by_dimension",
        },
        f"selection {expected_method}",
    )
    if document["schema_version"] != METHOD_SELECTION_SCHEMA_VERSION:
        raise GlobalRecipeError("unsupported method selection schema_version")
    if document["method_id"] != expected_method:
        raise GlobalRecipeError("method selection ID does not match its required slot")
    updates = document["optimizer_updates"]
    if isinstance(updates, bool) or updates not in UPDATE_BUDGETS:
        raise GlobalRecipeError("optimizer_updates must be one of 20, 50, or 100")
    schedule: dict[int, int] | None
    if expected_method == "internal-shared-fit":
        schedule = _normalize_schedule(document["orbit_m_by_dimension"])
    else:
        if document["orbit_m_by_dimension"] is not None:
            raise GlobalRecipeError("only internal-shared-fit may define an ORBIT schedule")
        schedule = None
    configuration = _normalize_configuration_grid(
        document["configuration"],
        method_id=expected_method,
        optimizer_updates=updates,
        orbit_m_by_dimension=schedule,
    )
    evidence = _normalize_evidence(
        document["selection_evidence"],
        method_id=expected_method,
    )
    return MethodSelection(
        method_id=expected_method,
        optimizer_updates=updates,
        configuration=json.loads(canonical_json_bytes(configuration)),
        selection_evidence=evidence,
        orbit_m_by_dimension=schedule,
    )


def load_method_selections(paths: Mapping[str, str | Path]) -> dict[str, MethodSelection]:
    """Load exactly the three required method selections."""

    if set(paths) != set(METHOD_IDS):
        raise GlobalRecipeError(f"method selection set must be exactly {METHOD_IDS}")
    return {
        method_id: parse_method_selection(
            read_strict_json(paths[method_id], label=f"{method_id} selection"),
            method_id,
        )
        for method_id in METHOD_IDS
    }


def _method_payload(selection: MethodSelection) -> dict[str, Any]:
    configuration = json.loads(canonical_json_bytes(selection.configuration))
    schedule = (
        None
        if selection.orbit_m_by_dimension is None
        else {str(key): value for key, value in selection.orbit_m_by_dimension.items()}
    )
    return {
        "method_id": selection.method_id,
        "arms": list(METHOD_ARMS[selection.method_id]),
        "optimizer_updates": selection.optimizer_updates,
        "optimizer_seeds": list(SEEDS),
        "configuration": configuration,
        "configuration_sha256": sha256_bytes(canonical_json_bytes(configuration)),
        "selection_evidence": list(selection.selection_evidence),
        "orbit_m_by_dimension": schedule,
    }


def _task_configuration_sha256(method: dict[str, Any], dimension: int, seed: int) -> str:
    configuration_id = _configuration_key(dimension, seed)
    matches = [
        record
        for record in method["configuration"]["by_dimension_and_seed"]
        if record["configuration_id"] == configuration_id
    ]
    if len(matches) != 1:
        raise GlobalRecipeError("method payload does not contain exactly one task configuration")
    return matches[0]["sha256"]


def _task_record(
    method: dict[str, Any],
    bundle: dict[str, Any],
    *,
    method_task_index: int,
    global_task_index: int,
    seed: int,
) -> dict[str, Any]:
    method_id = method["method_id"]
    task = {
        "task_id": f"f02-test__{method_id}__{bundle['bundle_id']}__seed-{seed}",
        "global_task_index": global_task_index,
        "method_task_index": method_task_index,
        "method_id": method_id,
        "bundle_id": bundle["bundle_id"],
        "catalog_task_index": bundle["catalog_task_index"],
        "replica": bundle["replica"],
        "n_particles": bundle["n_particles"],
        "dimension": bundle["dimension"],
        "seed": seed,
        "optimizer_updates": method["optimizer_updates"],
        "evaluation_split": EVALUATION_SPLIT,
        "evaluation_design": EVALUATION_DESIGN,
        "expected_arms": method["arms"],
        "method_configuration_sha256": _task_configuration_sha256(
            method,
            bundle["dimension"],
            seed,
        ),
    }
    if method_id == "internal-shared-fit":
        task["orbit_resource_m"] = method["orbit_m_by_dimension"][str(bundle["dimension"])]
    return task


def expected_task_grids(
    methods: Sequence[dict[str, Any]],
    bundles: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Enumerate the exact method-major, bundle-major, seed-major 450-task release grid."""

    grids: list[dict[str, Any]] = []
    global_index = 0
    for method in methods:
        tasks: list[dict[str, Any]] = []
        for bundle in bundles:
            for seed in SEEDS:
                tasks.append(
                    _task_record(
                        method,
                        bundle,
                        method_task_index=len(tasks),
                        global_task_index=global_index,
                        seed=seed,
                    )
                )
                global_index += 1
        grids.append(
            {
                "method_id": method["method_id"],
                "ordering": "confirmatory-replica-major,particle-major,seed-major",
                "expected_bundle_count": EXPECTED_BUNDLE_COUNT,
                "expected_seed_count": len(SEEDS),
                "expected_task_count": EXPECTED_TASKS_PER_METHOD,
                "tasks_sha256": sha256_bytes(canonical_json_bytes(tasks)),
                "tasks": tasks,
            }
        )
    return tuple(grids)


def _validate_recipe_path(path: str) -> str:
    if not isinstance(path, str) or not path:
        raise GlobalRecipeError("recipe_path must be a nonempty repository-relative path")
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or path != pure.as_posix():
        raise GlobalRecipeError("recipe_path must be canonical and repository-relative")
    if pure.suffix != ".json":
        raise GlobalRecipeError("recipe_path must name a JSON file")
    return path


def _validate_release_identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or _RELEASE_IDENTIFIER.fullmatch(value) is None:
        raise GlobalRecipeError(
            f"{label} must be a 1-64 character alphanumeric/dot/dash/underscore identifier"
        )
    return value


def build_global_recipe_document(
    catalog_path: str | Path,
    *,
    repo_root: str | Path,
    source_commit: str,
    recipe_path: str,
    experiment_id: str,
    protocol_id: str,
    protocol_path: str,
    selections: Mapping[str, MethodSelection],
) -> dict[str, Any]:
    """Build the deterministic complete F02 global recipe document."""

    recipe_path = _validate_recipe_path(recipe_path)
    experiment_id = _validate_release_identifier(experiment_id, "experiment_id")
    protocol_id = _validate_release_identifier(protocol_id, "protocol_id")
    protocol_path = _validate_protocol_path(protocol_path)
    if set(selections) != set(METHOD_IDS):
        raise GlobalRecipeError(f"recipe requires exactly method selections {METHOD_IDS}")
    source = resolve_source_release(
        repo_root,
        source_commit,
        protocol_path=protocol_path,
    )
    catalog = read_strict_json(catalog_path, label="F02 catalog")
    canonical_catalog_path = _validate_catalog_canonical_path(catalog_path, catalog)
    bundles = list(confirmatory_bundle_identities(catalog))
    catalog_sha256 = sha256_file(catalog_path)
    methods = [_method_payload(selections[method_id]) for method_id in METHOD_IDS]
    grids = list(expected_task_grids(methods, bundles))
    tasks = [task for grid in grids for task in grid["tasks"]]
    payload = {
        "study_id": STUDY_ID,
        "experiment_id": experiment_id,
        "protocol_id": protocol_id,
        "phase": "confirmatory",
        "release_scaffold_status": RECIPE_SCAFFOLD_STATUS,
        "development_evidence_semantically_verified": (DEVELOPMENT_EVIDENCE_SEMANTICALLY_VERIFIED),
        "release": {
            "source_commit": source.source_commit,
            "source_tree": source.source_tree,
            "recipe_path": recipe_path,
            "execution_commit_binding": "one-parent recipe-only add recorded by one-release ledger",
        },
        "artifacts": {
            "protocol": {
                "protocol_id": protocol_id,
                "path": source.protocol_path,
                "sha256": source.protocol_sha256,
                "binding_sha256": _protocol_binding_sha256(
                    protocol_id,
                    source.protocol_path,
                    source.protocol_sha256,
                ),
            },
            "catalog": {
                "schema_version": catalog["schema_version"],
                "catalog_type": catalog["catalog_type"],
                "path": canonical_catalog_path,
                "sha256": catalog_sha256,
                "generation_git_commit": catalog["provenance"]["git_commit"],
                "generation_git_tree": catalog["provenance"]["git_tree"],
            },
            "dependencies": [
                {"path": path, "sha256": source.dependency_sha256[path]}
                for path in DEPENDENCY_PATHS
            ],
            "tera_submodule": {
                "path": TERA_GITLINK_PATH,
                "commit": source.tera_gitlink_commit,
            },
        },
        "evaluation": {
            "split": EVALUATION_SPLIT,
            "design": EVALUATION_DESIGN,
            "optimizer_seeds": list(SEEDS),
        },
        "confirmatory_bundles": bundles,
        "confirmatory_bundles_sha256": sha256_bytes(canonical_json_bytes(bundles)),
        "methods": methods,
        "expected_task_grids": grids,
        "accounting": {
            "expected_bundle_count": EXPECTED_BUNDLE_COUNT,
            "expected_method_count": EXPECTED_METHOD_COUNT,
            "expected_tasks_per_method": EXPECTED_TASKS_PER_METHOD,
            "expected_task_count": EXPECTED_TASK_COUNT,
            "expected_task_ids_sha256": sha256_bytes(
                canonical_json_bytes([task["task_id"] for task in tasks])
            ),
        },
    }
    return {
        "schema_version": RECIPE_SCHEMA_VERSION,
        "payload": payload,
        "payload_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }


def _selection_from_method_payload(method: dict[str, Any]) -> MethodSelection:
    _require_exact_keys(
        method,
        {
            "method_id",
            "arms",
            "optimizer_updates",
            "optimizer_seeds",
            "configuration",
            "configuration_sha256",
            "selection_evidence",
            "orbit_m_by_dimension",
        },
        "recipe method",
    )
    method_id = method.get("method_id")
    if method_id not in METHOD_IDS:
        raise GlobalRecipeError("recipe contains an unknown method")
    selection_document = {
        "schema_version": METHOD_SELECTION_SCHEMA_VERSION,
        "method_id": method_id,
        "optimizer_updates": method["optimizer_updates"],
        "configuration": method["configuration"],
        "selection_evidence": method["selection_evidence"],
        "orbit_m_by_dimension": method["orbit_m_by_dimension"],
    }
    selection = parse_method_selection(selection_document, method_id)
    expected = _method_payload(selection)
    if method != expected:
        raise GlobalRecipeError("recipe method payload/configuration hash is mismatched")
    return selection


def validate_global_recipe_document(
    document: dict[str, Any],
    *,
    catalog_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Validate every recipe field against source Git objects and the full catalog."""

    _require_exact_keys(document, {"schema_version", "payload", "payload_sha256"}, "recipe")
    if document["schema_version"] != RECIPE_SCHEMA_VERSION:
        raise GlobalRecipeError("unsupported global recipe schema_version")
    payload = document["payload"]
    if not isinstance(payload, dict) or not _valid_sha256(document["payload_sha256"]):
        raise GlobalRecipeError("global recipe payload/hash types are invalid")
    if sha256_bytes(canonical_json_bytes(payload)) != document["payload_sha256"]:
        raise GlobalRecipeError("global recipe payload SHA-256 mismatch")
    _require_exact_keys(
        payload,
        {
            "study_id",
            "experiment_id",
            "protocol_id",
            "phase",
            "release_scaffold_status",
            "development_evidence_semantically_verified",
            "release",
            "artifacts",
            "evaluation",
            "confirmatory_bundles",
            "confirmatory_bundles_sha256",
            "methods",
            "expected_task_grids",
            "accounting",
        },
        "recipe payload",
    )
    if payload["study_id"] != STUDY_ID or payload["phase"] != "confirmatory":
        raise GlobalRecipeError("recipe is not the F02 global confirmatory release")
    if payload["release_scaffold_status"] != RECIPE_SCAFFOLD_STATUS:
        raise GlobalRecipeError("recipe release scaffold status is unsupported")
    if (
        payload["development_evidence_semantically_verified"]
        is not DEVELOPMENT_EVIDENCE_SEMANTICALLY_VERIFIED
    ):
        raise GlobalRecipeError(
            "recipe must record that development evidence is not semantically verified"
        )
    experiment_id = _validate_release_identifier(payload["experiment_id"], "experiment_id")
    protocol_id = _validate_release_identifier(payload["protocol_id"], "protocol_id")

    release = _require_mapping(payload, "release", "recipe payload")
    _require_exact_keys(
        release,
        {"source_commit", "source_tree", "recipe_path", "execution_commit_binding"},
        "recipe release",
    )
    recipe_path = _validate_recipe_path(release["recipe_path"])
    if release["execution_commit_binding"] != (
        "one-parent recipe-only add recorded by one-release ledger"
    ):
        raise GlobalRecipeError("recipe execution-commit policy is mismatched")
    artifacts = _require_mapping(payload, "artifacts", "recipe payload")
    _require_exact_keys(
        artifacts,
        {"protocol", "catalog", "dependencies", "tera_submodule"},
        "recipe artifacts",
    )
    protocol = _require_mapping(artifacts, "protocol", "recipe artifacts")
    _require_exact_keys(
        protocol,
        {"protocol_id", "path", "sha256", "binding_sha256"},
        "recipe protocol",
    )
    if protocol["protocol_id"] != protocol_id:
        raise GlobalRecipeError("recipe protocol_id is detached from its protocol artifact")
    protocol_path = _validate_protocol_path(protocol["path"])
    source = resolve_source_release(
        repo_root,
        release["source_commit"],
        protocol_path=protocol_path,
    )
    if release["source_tree"] != source.source_tree:
        raise GlobalRecipeError("recipe source tree does not match its source commit")
    expected_artifacts = {
        "protocol": {
            "protocol_id": protocol_id,
            "path": source.protocol_path,
            "sha256": source.protocol_sha256,
            "binding_sha256": _protocol_binding_sha256(
                protocol_id,
                source.protocol_path,
                source.protocol_sha256,
            ),
        },
        "catalog": None,
        "dependencies": [
            {"path": path, "sha256": source.dependency_sha256[path]} for path in DEPENDENCY_PATHS
        ],
        "tera_submodule": {"path": TERA_GITLINK_PATH, "commit": source.tera_gitlink_commit},
    }
    if artifacts["protocol"] != expected_artifacts["protocol"]:
        raise GlobalRecipeError("recipe protocol hash does not match source commit")
    if artifacts["dependencies"] != expected_artifacts["dependencies"]:
        raise GlobalRecipeError("recipe dependency hashes do not match source commit")
    if artifacts["tera_submodule"] != expected_artifacts["tera_submodule"]:
        raise GlobalRecipeError("recipe TERA gitlink does not match source commit")

    catalog = read_strict_json(catalog_path, label="F02 catalog")
    canonical_catalog_path = _validate_catalog_canonical_path(catalog_path, catalog)
    bundles = list(confirmatory_bundle_identities(catalog))
    expected_catalog = {
        "schema_version": catalog["schema_version"],
        "catalog_type": catalog["catalog_type"],
        "path": canonical_catalog_path,
        "sha256": sha256_file(catalog_path),
        "generation_git_commit": catalog["provenance"]["git_commit"],
        "generation_git_tree": catalog["provenance"]["git_tree"],
    }
    if artifacts["catalog"] != expected_catalog:
        raise GlobalRecipeError("recipe catalog hash/provenance does not match the full catalog")
    if payload["confirmatory_bundles"] != bundles:
        raise GlobalRecipeError("recipe does not enumerate the exact 50 confirmatory bundles")
    if payload["confirmatory_bundles_sha256"] != sha256_bytes(canonical_json_bytes(bundles)):
        raise GlobalRecipeError("recipe confirmatory bundle identity hash is mismatched")

    evaluation = _require_mapping(payload, "evaluation", "recipe payload")
    expected_evaluation = {
        "split": EVALUATION_SPLIT,
        "design": EVALUATION_DESIGN,
        "optimizer_seeds": list(SEEDS),
    }
    if evaluation != expected_evaluation:
        raise GlobalRecipeError("recipe evaluation split/design/seeds are mismatched")

    methods = payload["methods"]
    if not isinstance(methods, list) or [
        item.get("method_id") for item in methods if isinstance(item, dict)
    ] != list(METHOD_IDS):
        raise GlobalRecipeError(f"recipe methods must be exactly {METHOD_IDS} in order")
    selections: dict[str, MethodSelection] = {}
    for method in methods:
        if not isinstance(method, dict):
            raise GlobalRecipeError("recipe method entries must be objects")
        selection = _selection_from_method_payload(method)
        selections[selection.method_id] = selection
    if set(selections) != set(METHOD_IDS):
        raise GlobalRecipeError("recipe is missing one or more required methods")

    expected_grids = list(expected_task_grids(methods, bundles))
    if payload["expected_task_grids"] != expected_grids:
        raise GlobalRecipeError("recipe expected task grids are partial, reordered, or mismatched")
    tasks = [task for grid in expected_grids for task in grid["tasks"]]
    expected_accounting = {
        "expected_bundle_count": EXPECTED_BUNDLE_COUNT,
        "expected_method_count": EXPECTED_METHOD_COUNT,
        "expected_tasks_per_method": EXPECTED_TASKS_PER_METHOD,
        "expected_task_count": EXPECTED_TASK_COUNT,
        "expected_task_ids_sha256": sha256_bytes(
            canonical_json_bytes([task["task_id"] for task in tasks])
        ),
    }
    if payload["accounting"] != expected_accounting:
        raise GlobalRecipeError("recipe task accounting is not the exact 450-task grid")
    return {
        "structurally_valid": True,
        "releasable": False,
        "release_scaffold_status": RECIPE_SCAFFOLD_STATUS,
        "development_evidence_semantically_verified": (DEVELOPMENT_EVIDENCE_SEMANTICALLY_VERIFIED),
        "schema_version": RECIPE_SCHEMA_VERSION,
        "payload_sha256": document["payload_sha256"],
        "recipe_path": recipe_path,
        "source_commit": source.source_commit,
        "source_tree": source.source_tree,
        "experiment_id": experiment_id,
        "protocol_id": protocol_id,
        "protocol_path": source.protocol_path,
        "protocol_sha256": source.protocol_sha256,
        "protocol_binding_sha256": expected_artifacts["protocol"]["binding_sha256"],
        "catalog_sha256": expected_catalog["sha256"],
        "expected_bundle_count": EXPECTED_BUNDLE_COUNT,
        "expected_method_count": EXPECTED_METHOD_COUNT,
        "expected_task_count": EXPECTED_TASK_COUNT,
        "expected_task_ids_sha256": expected_accounting["expected_task_ids_sha256"],
    }


def validate_global_recipe(
    recipe_path: str | Path,
    *,
    catalog_path: str | Path,
    repo_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    document = read_strict_json(recipe_path, label="global F02 recipe")
    summary = validate_global_recipe_document(
        document,
        catalog_path=catalog_path,
        repo_root=repo_root,
    )
    summary["recipe_file_sha256"] = sha256_file(recipe_path)
    return document, summary


def write_json_exclusive(path: str | Path, document: dict[str, Any]) -> Path:
    """Create a deterministic JSON file without replacing any existing file."""

    output = Path(path)
    if not output.parent.is_dir():
        raise GlobalRecipeError(f"output parent directory does not exist: {output.parent}")
    encoded = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    except OSError as error:
        raise GlobalRecipeError(f"refusing to replace recipe output: {output}") from error
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build but do not authorize a global recipe")
    build.add_argument("--catalog", type=Path, required=True)
    build.add_argument("--repo-root", type=Path, default=_IMPORT_ROOT)
    build.add_argument("--source-commit", required=True)
    build.add_argument("--experiment-id", required=True)
    build.add_argument("--protocol-id", required=True)
    build.add_argument("--protocol-path", required=True)
    build.add_argument("--recipe-path", default=RECIPE_DEFAULT_PATH)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--internal-selection", type=Path, required=True)
    build.add_argument("--dsoftki-selection", type=Path, required=True)
    build.add_argument("--ddsvgp-selection", type=Path, required=True)

    validate = subparsers.add_parser("validate", help="validate without reading test arrays")
    validate.add_argument("--recipe", type=Path, required=True)
    validate.add_argument("--catalog", type=Path, required=True)
    validate.add_argument("--repo-root", type=Path, default=_IMPORT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "build":
        selections = load_method_selections(
            {
                "internal-shared-fit": args.internal_selection,
                "dsoftki-512": args.dsoftki_selection,
                "ddsvgp-512": args.ddsvgp_selection,
            }
        )
        document = build_global_recipe_document(
            args.catalog,
            repo_root=args.repo_root,
            source_commit=args.source_commit,
            recipe_path=args.recipe_path,
            experiment_id=args.experiment_id,
            protocol_id=args.protocol_id,
            protocol_path=args.protocol_path,
            selections=selections,
        )
        write_json_exclusive(args.out, document)
        print(
            json.dumps(
                {
                    "created": str(args.out),
                    "payload_sha256": document["payload_sha256"],
                    "release_scaffold_status": RECIPE_SCAFFOLD_STATUS,
                    "releasable": False,
                    "development_evidence_semantically_verified": (
                        DEVELOPMENT_EVIDENCE_SEMANTICALLY_VERIFIED
                    ),
                    "confirmatory_execution_enabled": False,
                },
                sort_keys=True,
            )
        )
        return 0
    _, summary = validate_global_recipe(
        args.recipe,
        catalog_path=args.catalog,
        repo_root=args.repo_root,
    )
    summary["confirmatory_execution_enabled"] = False
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIRMATORY_REPLICAS",
    "DEVELOPMENT_EVIDENCE_SEMANTICALLY_VERIFIED",
    "DIMENSIONS",
    "EXPECTED_BUNDLE_COUNT",
    "EXPECTED_METHOD_COUNT",
    "EXPECTED_TASK_COUNT",
    "EXPECTED_TASKS_PER_METHOD",
    "GlobalRecipeError",
    "METHOD_IDS",
    "METHOD_SELECTION_SCHEMA_VERSION",
    "MethodSelection",
    "ORBIT_M_CANDIDATES",
    "RECIPE_DEFAULT_PATH",
    "RECIPE_SCAFFOLD_STATUS",
    "RECIPE_SCHEMA_VERSION",
    "SEEDS",
    "SourceRelease",
    "UPDATE_BUDGETS",
    "build_global_recipe_document",
    "canonical_json_bytes",
    "confirmatory_bundle_identities",
    "expected_task_grids",
    "load_method_selections",
    "parse_method_selection",
    "read_strict_json",
    "read_strict_json_bytes",
    "resolve_source_release",
    "sha256_bytes",
    "sha256_file",
    "validate_global_recipe",
    "validate_global_recipe_document",
    "write_json_exclusive",
]
