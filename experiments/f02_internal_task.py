"""Single-corpus runner for the internal arms of the F02 experiment.

This module deliberately does not run the external DSoftKI or DDSVGP baselines
and does not select hyperparameters, optimizer budgets, or ORBIT neighbourhood
sizes.  It executes one already specified configuration on one verified corpus.

Validation is the default phase.  The current single-bundle recipe code is only
gate scaffolding: confirmatory execution remains deliberately disabled until a
global recipe binds the complete 50-corpus task matrix, all selected budgets,
the dimension-to-neighbour schedule, and one analysis source release.
Validation runs still use the repository's whole-bundle integrity loader, which
decompresses every stored array; they do not select, tensorize, or score the
test split.  In test mode, manifest/metadata preflight and the recipe gate occur
before that whole-bundle decompression.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

import numpy as np
import torch

from data.generate_nbody_confirmatory import (
    SCHEMA_VERSION,
    ConfirmatoryConfig,
    verify_sha256_manifest,
)
from data.load_nbody_confirmatory import (
    PreparedConfirmatoryBundle,
    PreparedConfirmatoryDataset,
    load_prepared_confirmatory_bundle,
)
from experiments.f02_design import (
    EVALUATION_TIME_INDICES,
    OPTIMIZER_SELECTION_TIME_INDICES,
    TRAIN_TIME_INDICES,
    select_time_indices,
)
from experiments.f02_internal_models import (
    FrozenTERAParameters,
    ScalarPrediction,
    fit_released_tera,
    freeze_tera_parameters,
    predict_orbit,
    predict_released_tera,
    predict_value_only_local_gp,
    prepared_split_to_tensors,
)
from experiments.f02_metrics import TrajectoryMetrics, per_trajectory_metrics

RESULT_SCHEMA_VERSION = "f02_internal_task_v1"
FROZEN_RECIPE_SCHEMA_VERSION = "f02_frozen_recipe_v1"
CONFIRMATORY_TEST_RELEASED = False
REFERENCE_M = 50
DEFAULT_CANDIDATE_M = (75, 100, 150, 200)
MAX_CG_TOLERANCE = 1e-5
SAME_M_ABSOLUTE_TOLERANCE = {
    "float32": 1e-4,
    "float64": 1e-6,
}
F02_DEVELOPMENT_REPLICAS = (0, 1, 2)
F02_CONFIRMATORY_REPLICAS = tuple(range(101, 111))
F02_REPLICAS = (*F02_DEVELOPMENT_REPLICAS, *F02_CONFIRMATORY_REPLICAS)
F02_PARTICLE_COUNTS = (2, 4, 6, 8, 10)
F02_N_DIMS = 3
F02_EXPECTED_BUNDLE_COUNT = len(F02_REPLICAS) * len(F02_PARTICLE_COUNTS)
_EVALUATION_DESIGNS = {
    "primary": EVALUATION_TIME_INDICES,
    "optimizer_selection": OPTIMIZER_SELECTION_TIME_INDICES,
}
_REPO_ROOT = Path(__file__).resolve().parents[1]


class InternalTaskError(RuntimeError):
    """Raised when an internal F02 task cannot produce an auditable result."""


class FrozenRecipeError(InternalTaskError):
    """Raised before any test split is selected or scored."""


@dataclass(frozen=True, slots=True)
class InternalTaskConfig:
    """Fully specified, non-tuning configuration for one internal task."""

    training_m: int = 20
    train_steps: int = 20
    train_epochs: int = 0
    kernel: str = "rbf"
    outputscale: float = 1.0
    sigma_f: float = 1e-3
    sigma_g: float = 1e-3
    # The paper-matching released MD22 recipe initializes the scalar value at
    # 1.0 and then learns it; ``None`` is an explicit median-init sensitivity.
    lengthscale: float | tuple[float, ...] | None = 1.0
    lengthscale_init: str = "median"
    lengthscale_init_max_points: int = 2048
    use_ard: bool = False
    seed: int = 11
    batch_size: int = 256
    lr: float = 0.01
    weight_decay: float = 0.0
    graph_refresh_epochs: int = 0
    learn_lengthscale: bool = True
    learn_outputscale: bool = True
    learn_sigma_f: bool = True
    learn_sigma_g: bool = True
    min_sigma_f: float = 1e-6
    min_sigma_g: float = 0.0
    candidate_m: tuple[int, ...] = DEFAULT_CANDIDATE_M
    cg_tolerance: float = 1e-5
    cg_max_iterations: int | None = None
    use_preconditioner: bool = True
    function_jitter: float = 1e-8
    reduced_jitter: float = 1e-8
    dtype: str = "float32"
    device: str = "cpu"

    def __post_init__(self) -> None:
        raw_candidates = tuple(self.candidate_m)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in raw_candidates
        ):
            raise ValueError("candidate_m values must be integers")
        candidates = tuple(int(value) for value in raw_candidates)
        if any(value not in DEFAULT_CANDIDATE_M for value in candidates):
            raise ValueError(
                f"candidate_m values must come from the preregistered grid "
                f"{DEFAULT_CANDIDATE_M}"
            )
        if tuple(sorted(set(candidates))) != candidates:
            raise ValueError("candidate_m must be unique and strictly increasing")
        object.__setattr__(self, "candidate_m", candidates)

        lengthscale = self.lengthscale
        if isinstance(lengthscale, list):
            lengthscale = tuple(float(value) for value in lengthscale)
            object.__setattr__(self, "lengthscale", lengthscale)
        if isinstance(lengthscale, tuple):
            if not lengthscale or any(
                not math.isfinite(value) or value <= 0.0 for value in lengthscale
            ):
                raise ValueError("lengthscale entries must be finite and strictly positive")
        elif lengthscale is not None and (
            not math.isfinite(float(lengthscale)) or float(lengthscale) <= 0.0
        ):
            raise ValueError("lengthscale must be finite and strictly positive")

        if (
            self.training_m <= 0
            or self.train_steps < 0
            or self.train_epochs < 0
            or self.batch_size <= 0
        ):
            raise ValueError(
                "training_m/batch_size must be positive and training budgets non-negative"
            )
        if self.train_steps > 0 and self.train_epochs > 0:
            raise ValueError("set at most one of train_steps and train_epochs")
        if self.kernel not in {"rbf", "matern52"}:
            raise ValueError("kernel must be 'rbf' or 'matern52'")
        if self.lengthscale_init not in {"median", "one"}:
            raise ValueError("lengthscale_init must be 'median' or 'one'")
        if self.lengthscale_init_max_points <= 0 or self.graph_refresh_epochs < 0:
            raise ValueError(
                "lengthscale point limit must be positive and refresh epochs non-negative"
            )
        if not math.isfinite(self.outputscale) or self.outputscale <= 0.0:
            raise ValueError("outputscale must be finite and strictly positive")
        for name in ("sigma_f", "sigma_g", "min_sigma_f", "min_sigma_g", "weight_decay"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.lr) or self.lr <= 0.0:
            raise ValueError("lr must be finite and strictly positive")
        if (
            not math.isfinite(self.cg_tolerance)
            or self.cg_tolerance <= 0.0
            or self.cg_tolerance > MAX_CG_TOLERANCE
        ):
            raise ValueError(
                f"cg_tolerance must be finite and lie in (0, {MAX_CG_TOLERANCE}]"
            )
        if self.cg_max_iterations is not None and self.cg_max_iterations <= 0:
            raise ValueError("cg_max_iterations must be positive when supplied")
        for name in ("function_jitter", "reduced_jitter"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
        if not self.device:
            raise ValueError("device must be nonempty")


@dataclass(frozen=True, slots=True)
class BundleIdentity:
    """Manifest/metadata identity available without decompressing the NPZ."""

    dataset_path: Path
    file_sha256: dict[str, str]
    manifest_sha256: str
    generator_config: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CatalogAuthorization:
    """One eligible catalog entry and the generation provenance it inherits."""

    catalog_path: Path
    catalog_sha256: str
    generation_git_commit: str
    generation_git_tree: str
    bundle_entry: dict[str, Any]


def _json_compatible(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_config_payload(config: InternalTaskConfig) -> dict[str, Any]:
    return _json_compatible(asdict(config))


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InternalTaskError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise InternalTaskError(f"{label} must contain a JSON object")
    return value


def _preflight_bundle_identity(dataset_path: str | Path) -> BundleIdentity:
    """Verify sidecar hashes/config without opening or decompressing the NPZ."""

    dataset = Path(dataset_path)
    if dataset.suffix != ".npz":
        raise InternalTaskError("dataset_path must name a .npz artifact")
    metadata_path = dataset.with_suffix(".metadata.json")
    manifest_path = dataset.with_suffix(".sha256.json")
    manifest = _read_json_mapping(manifest_path, "bundle SHA-256 manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise InternalTaskError("unsupported bundle manifest schema_version")
    try:
        file_sha256 = verify_sha256_manifest(manifest_path)
    except Exception as error:
        raise InternalTaskError("bundle SHA-256 preflight failed") from error
    if set(file_sha256) != {dataset.name, metadata_path.name}:
        raise InternalTaskError("bundle manifest must cover exactly the NPZ and metadata")
    metadata = _read_json_mapping(metadata_path, "bundle metadata")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise InternalTaskError("unsupported bundle metadata schema_version")
    if metadata.get("dataset_file") != dataset.name:
        raise InternalTaskError("bundle metadata dataset_file does not match the NPZ")
    generator_config = metadata.get("config")
    if not isinstance(generator_config, dict):
        raise InternalTaskError("bundle metadata config must be an object")
    return BundleIdentity(
        dataset_path=dataset,
        file_sha256=dict(sorted(file_sha256.items())),
        manifest_sha256=_sha256_file(manifest_path),
        generator_config=generator_config,
    )


def _bundle_identity(bundle: PreparedConfirmatoryBundle) -> BundleIdentity:
    provenance = bundle.loaded.provenance
    return BundleIdentity(
        dataset_path=provenance.dataset_path,
        file_sha256=dict(sorted(provenance.file_sha256.items())),
        manifest_sha256=_sha256_file(provenance.sha256_manifest_path),
        generator_config=provenance.config_payload,
    )


def _require_mapping(parent: dict[str, Any], name: str, label: str) -> dict[str, Any]:
    value = parent.get(name)
    if not isinstance(value, dict):
        raise InternalTaskError(f"catalog {label}.{name} must be an object")
    return value


def _validate_preregistered_catalog_grid(
    report: dict[str, Any],
    bundles: list[Any],
    expected_count: int,
) -> None:
    """Require the exact 65-corpus grid declared by the frozen F02 protocol."""

    if expected_count != F02_EXPECTED_BUNDLE_COUNT or len(bundles) != expected_count:
        raise InternalTaskError(
            f"F02 catalog must contain exactly {F02_EXPECTED_BUNDLE_COUNT} bundles"
        )
    inputs = _require_mapping(report, "input", "root")
    expected_inputs = {
        "replicas": list(F02_REPLICAS),
        "development_replicas": list(F02_DEVELOPMENT_REPLICAS),
        "particle_counts": list(F02_PARTICLE_COUNTS),
        "n_dims": F02_N_DIMS,
    }
    if any(inputs.get(name) != value for name, value in expected_inputs.items()):
        raise InternalTaskError("F02 catalog input grid does not match the preregistered grid")
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
        raise InternalTaskError("F02 catalog generator settings do not match the protocol")

    catalog = _require_mapping(report, "catalog", "root")
    if catalog.get("phase_counts") != {"development": 15, "confirmatory": 50}:
        raise InternalTaskError("F02 catalog phase counts must be development=15/confirmatory=50")

    expected_coordinates = [
        (task_index, replica, n_particles)
        for task_index, (replica, n_particles) in enumerate(
            (replica, n_particles)
            for replica in F02_REPLICAS
            for n_particles in F02_PARTICLE_COUNTS
        )
    ]
    content_hashes: list[str] = []
    for raw_entry, (task_index, replica, n_particles) in zip(
        bundles,
        expected_coordinates,
        strict=True,
    ):
        if not isinstance(raw_entry, dict):
            raise InternalTaskError("F02 catalog bundle entries must be objects")
        entry = raw_entry
        expected_phase = (
            "development" if replica in F02_DEVELOPMENT_REPLICAS else "confirmatory"
        )
        expected_config = asdict(
            ConfirmatoryConfig(
                n_particles=n_particles,
                n_dims=F02_N_DIMS,
                replica=replica,
            )
        )
        expected_fields = {
            "task_index": task_index,
            "phase": expected_phase,
            "replica": replica,
            "n_particles": n_particles,
            "n_dims": F02_N_DIMS,
            "D": 2 * n_particles * F02_N_DIMS,
            "config": expected_config,
            "eligible_for_catalog": True,
            "unique_content": True,
        }
        if any(entry.get(name) != value for name, value in expected_fields.items()):
            raise InternalTaskError(
                "F02 catalog bundle coordinates/config/phase do not match the frozen grid"
            )
        paths = entry.get("paths")
        hashes = entry.get("hashes")
        if not isinstance(paths, dict) or not isinstance(hashes, dict):
            raise InternalTaskError("F02 catalog bundle paths/hashes are invalid")
        stem = f"nbody_fixedmass_n{n_particles}_d{F02_N_DIMS}_replica{replica}"
        expected_names = {
            "dataset": f"{stem}.npz",
            "metadata": f"{stem}.metadata.json",
            "sha256_manifest": f"{stem}.sha256.json",
        }
        if any(
            not isinstance(paths.get(name), str)
            or Path(paths[name]).name != expected_name
            for name, expected_name in expected_names.items()
        ):
            raise InternalTaskError("F02 catalog bundle filenames do not match the frozen grid")
        content_hash = hashes.get("dataset_content_sha256")
        if not isinstance(content_hash, str) or len(content_hash) != 64:
            raise InternalTaskError("F02 catalog dataset content hash is invalid")
        content_hashes.append(content_hash)
    if len(set(content_hashes)) != F02_EXPECTED_BUNDLE_COUNT:
        raise InternalTaskError("F02 catalog dataset content hashes are not globally unique")


def validate_catalog_identity(
    catalog_path: str | Path,
    identity: BundleIdentity,
) -> CatalogAuthorization:
    """Authorize exactly one bundle from a globally ready strict catalog."""

    path = Path(catalog_path)
    report = _read_json_mapping(path, "F02 data catalog")
    if report.get("schema_version") != 1:
        raise InternalTaskError("unsupported F02 data catalog schema_version")
    if report.get("catalog_type") != "f02_nbody_confirmatory_data":
        raise InternalTaskError("unexpected F02 data catalog type")
    if report.get("overall_ready") is not True:
        raise InternalTaskError("F02 data catalog is not overall_ready")

    accounting = _require_mapping(report, "task_accounting", "root")
    counts = _require_mapping(accounting, "status_counts", "task_accounting")
    if accounting.get("all_expected_tasks_valid") is not True:
        raise InternalTaskError("F02 catalog task accounting is incomplete")
    if accounting.get("all_expected_tasks_unique") is not True:
        raise InternalTaskError("F02 catalog task accounting is not content-unique")
    expected_count = accounting.get("expected_task_count")
    if not isinstance(expected_count, int) or expected_count <= 0:
        raise InternalTaskError("F02 catalog expected_task_count is invalid")
    if counts.get("valid") != expected_count or any(
        counts.get(name) != 0 for name in ("missing", "failed", "invalid", "unexpected")
    ):
        raise InternalTaskError(
            "F02 catalog contains missing, failed, invalid, or unexpected tasks"
        )

    provenance = _require_mapping(report, "provenance", "root")
    required_provenance_flags = (
        "verified",
        "same_commit",
        "same_tree",
        "same_submodules",
        "same_source_hashes",
        "same_source_manifest",
        "same_slurm_array_job",
        "same_repo_root",
    )
    if any(provenance.get(name) is not True for name in required_provenance_flags):
        raise InternalTaskError("F02 catalog cross-task provenance is not fully verified")
    commit = provenance.get("git_commit")
    tree = provenance.get("git_tree")
    if not isinstance(commit, str) or not isinstance(tree, str) or not commit or not tree:
        raise InternalTaskError("F02 catalog generation commit/tree are missing")

    independence = _require_mapping(report, "independence", "root")
    if independence.get("verified") is not True:
        raise InternalTaskError("F02 catalog independence is not verified")
    if independence.get("duplicate_dataset_content_sha256") != []:
        raise InternalTaskError("F02 catalog reports duplicate dataset content")
    if independence.get("candidate_bundle_count") != independence.get("unique_bundle_count"):
        raise InternalTaskError("F02 catalog bundle independence counts disagree")

    catalog = _require_mapping(report, "catalog", "root")
    bundles = catalog.get("bundles")
    if not isinstance(bundles, list) or catalog.get("bundle_count") != len(bundles):
        raise InternalTaskError("F02 catalog bundle list/count is invalid")
    _validate_preregistered_catalog_grid(report, bundles, expected_count)
    resolved_dataset = identity.dataset_path.resolve()
    matches = []
    for entry in bundles:
        if not isinstance(entry, dict):
            raise InternalTaskError("F02 catalog bundle entries must be objects")
        paths = entry.get("paths")
        if not isinstance(paths, dict) or not isinstance(paths.get("dataset"), str):
            raise InternalTaskError("F02 catalog bundle paths are invalid")
        if Path(paths["dataset"]).resolve() == resolved_dataset:
            matches.append(entry)
    if len(matches) != 1:
        raise InternalTaskError("loaded bundle must appear exactly once in the F02 catalog")
    entry = matches[0]
    if entry.get("eligible_for_catalog") is not True or entry.get("unique_content") is not True:
        raise InternalTaskError("loaded bundle is not eligible and unique in the F02 catalog")
    hashes = entry.get("hashes")
    paths = entry.get("paths")
    if not isinstance(hashes, dict) or not isinstance(paths, dict):
        raise InternalTaskError("matched F02 catalog entry hashes/paths are invalid")
    expected_hashes = {
        "dataset_file_sha256": identity.file_sha256.get(identity.dataset_path.name),
        "metadata_file_sha256": identity.file_sha256.get(
            identity.dataset_path.with_suffix(".metadata.json").name
        ),
        "sha256_manifest_file_sha256": identity.manifest_sha256,
    }
    if any(hashes.get(name) != value for name, value in expected_hashes.items()):
        raise InternalTaskError("matched F02 catalog entry file hashes do not match the bundle")
    if entry.get("config") != identity.generator_config:
        raise InternalTaskError("matched F02 catalog entry config does not match the bundle")
    expected_sidecars = {
        "metadata": identity.dataset_path.with_suffix(".metadata.json").resolve(),
        "sha256_manifest": identity.dataset_path.with_suffix(".sha256.json").resolve(),
    }
    if any(
        not isinstance(paths.get(name), str) or Path(paths[name]).resolve() != expected
        for name, expected in expected_sidecars.items()
    ):
        raise InternalTaskError("matched F02 catalog sidecar paths do not match the bundle")
    return CatalogAuthorization(
        catalog_path=path,
        catalog_sha256=_sha256_file(path),
        generation_git_commit=commit,
        generation_git_tree=tree,
        bundle_entry=entry,
    )


def _authorize_evaluation_phase(
    catalog: CatalogAuthorization,
    evaluation_split: str,
) -> None:
    """Keep tuning corpora and confirmatory corpora on opposite sides of the gate."""

    expected_phase = "development" if evaluation_split == "validation" else "confirmatory"
    actual_phase = catalog.bundle_entry.get("phase")
    if actual_phase != expected_phase:
        raise InternalTaskError(
            f"{evaluation_split} evaluation requires a catalog bundle in phase "
            f"{expected_phase!r}, got {actual_phase!r}"
        )


def _frozen_recipe_payload(
    bundle: PreparedConfirmatoryBundle | BundleIdentity,
    config: InternalTaskConfig,
    catalog: CatalogAuthorization,
) -> dict[str, Any]:
    identity = (
        _bundle_identity(bundle) if isinstance(bundle, PreparedConfirmatoryBundle) else bundle
    )
    return {
        "bundle": {
            "dataset_file": identity.dataset_path.name,
            "file_sha256": identity.file_sha256,
            "generator_config": identity.generator_config,
        },
        "catalog": {
            "sha256": catalog.catalog_sha256,
            "generation_git_commit": catalog.generation_git_commit,
            "generation_git_tree": catalog.generation_git_tree,
            "task_index": catalog.bundle_entry.get("task_index"),
            "dataset_content_sha256": catalog.bundle_entry.get("hashes", {}).get(
                "dataset_content_sha256"
            ),
        },
        "evaluation": {
            "split": "test",
            "design": "primary",
            "time_indices": list(EVALUATION_TIME_INDICES),
        },
        "training_time_indices": list(TRAIN_TIME_INDICES),
        "reference_m": REFERENCE_M,
        "task_config": _task_config_payload(config),
    }


def build_frozen_recipe_document(
    bundle: PreparedConfirmatoryBundle | BundleIdentity,
    config: InternalTaskConfig,
    catalog: CatalogAuthorization,
) -> dict[str, Any]:
    """Build the document that must subsequently be reviewed and committed.

    This function uses only bundle provenance and registered configuration; it
    never reads validation or test target arrays.
    """

    if len(config.candidate_m) != 1:
        raise ValueError("a test frozen recipe requires exactly one selected candidate_m")
    payload = _frozen_recipe_payload(bundle, config, catalog)
    return {
        "schema_version": FROZEN_RECIPE_SCHEMA_VERSION,
        "payload": payload,
        "payload_sha256": _sha256_bytes(_canonical_json_bytes(payload)),
    }


def _run_git(
    repo_root: Path,
    *arguments: str,
    binary: bool = False,
) -> str | bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=not binary,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise InternalTaskError(f"git {' '.join(arguments)} failed") from error
    return completed.stdout


def _assert_recipe_committed(recipe_path: Path, repo_root: Path) -> None:
    root = repo_root.resolve()
    resolved = recipe_path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise FrozenRecipeError("frozen recipe must be inside the repository") from error
    relative_text = relative.as_posix()
    try:
        _run_git(root, "ls-files", "--error-unmatch", "--", relative_text)
        committed = _run_git(root, "show", f"HEAD:{relative_text}", binary=True)
        status = _run_git(
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            relative_text,
        )
    except InternalTaskError as error:
        raise FrozenRecipeError("frozen recipe is not committed at HEAD") from error
    if committed != resolved.read_bytes() or str(status).strip():
        raise FrozenRecipeError("frozen recipe must be byte-identical to the clean HEAD copy")


def _assert_repository_clean(repo_root: Path) -> None:
    status = str(
        _run_git(
            repo_root.resolve(),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
    ).strip()
    if status:
        raise FrozenRecipeError("test evaluation requires a globally clean HEAD worktree")


def _strict_json_object(path: Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FrozenRecipeError(f"duplicate JSON key in frozen recipe: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                FrozenRecipeError(f"nonfinite JSON constant in frozen recipe: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FrozenRecipeError(f"cannot read frozen recipe: {path}") from error
    if not isinstance(value, dict):
        raise FrozenRecipeError("frozen recipe root must be an object")
    return value


def validate_frozen_recipe(
    recipe_path: str | Path,
    bundle: PreparedConfirmatoryBundle | BundleIdentity,
    config: InternalTaskConfig,
    catalog: CatalogAuthorization,
    *,
    repo_root: str | Path = _REPO_ROOT,
) -> dict[str, Any]:
    """Validate the committed gate before the caller accesses ``split='test'``."""

    path = Path(recipe_path)
    root = Path(repo_root)
    _assert_recipe_committed(path, root)
    document = _strict_json_object(path)
    if set(document) != {"schema_version", "payload", "payload_sha256"}:
        raise FrozenRecipeError("frozen recipe has unexpected or missing root fields")
    if document["schema_version"] != FROZEN_RECIPE_SCHEMA_VERSION:
        raise FrozenRecipeError("unsupported frozen recipe schema_version")
    payload = document["payload"]
    payload_hash = document["payload_sha256"]
    if not isinstance(payload, dict) or not isinstance(payload_hash, str):
        raise FrozenRecipeError("frozen recipe payload and payload_sha256 have invalid types")
    actual_hash = _sha256_bytes(_canonical_json_bytes(payload))
    if payload_hash != actual_hash:
        raise FrozenRecipeError("frozen recipe payload SHA-256 mismatch")

    expected = _frozen_recipe_payload(bundle, config, catalog)
    if payload.get("bundle") != expected["bundle"]:
        raise FrozenRecipeError("frozen recipe bundle hashes/config do not match the loaded corpus")
    if payload.get("catalog") != expected["catalog"]:
        raise FrozenRecipeError("frozen recipe catalog identity/provenance do not match")
    if payload.get("task_config") != expected["task_config"]:
        raise FrozenRecipeError("frozen recipe task configuration does not match this run")
    if payload != expected:
        raise FrozenRecipeError(
            "frozen recipe registered design or reference settings do not match"
        )
    return {
        "required": True,
        "validated": True,
        "committed_at_head": True,
        "path": str(path.resolve()),
        "payload_sha256": payload_hash,
        "schema_version": FROZEN_RECIPE_SCHEMA_VERSION,
    }


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _measure(
    device_name: str,
    function: Any,
) -> tuple[Any, float, int | None]:
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    value = function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
    else:
        peak = None
    return value, time.perf_counter() - start, peak


def _selected_bundle(
    bundle: PreparedConfirmatoryBundle,
    evaluation_split: str,
    evaluation_design: str,
) -> PreparedConfirmatoryBundle:
    training = select_time_indices(bundle.prepared.train, TRAIN_TIME_INDICES)
    evaluation_indices = _EVALUATION_DESIGNS[evaluation_design]
    evaluation = select_time_indices(
        bundle.prepared.split(evaluation_split),
        evaluation_indices,
    )
    prepared = PreparedConfirmatoryDataset(
        train=training,
        validation=(evaluation if evaluation_split == "validation" else bundle.prepared.validation),
        test=(evaluation if evaluation_split == "test" else bundle.prepared.test),
        normalization=bundle.prepared.normalization,
        masses=bundle.prepared.masses,
    )
    return PreparedConfirmatoryBundle(loaded=bundle.loaded, prepared=prepared)


def _fit_kwargs(config: InternalTaskConfig) -> dict[str, Any]:
    lengthscale: float | list[float] | None
    if isinstance(config.lengthscale, tuple):
        lengthscale = list(config.lengthscale)
    else:
        lengthscale = config.lengthscale
    return {
        "training_m": config.training_m,
        "train_steps": config.train_steps,
        "train_epochs": config.train_epochs,
        "kernel": config.kernel,
        "outputscale": config.outputscale,
        "sigma_f": config.sigma_f,
        "sigma_g": config.sigma_g,
        "lengthscale": lengthscale,
        "lengthscale_init": config.lengthscale_init,
        "lengthscale_init_max_points": config.lengthscale_init_max_points,
        "use_ard": config.use_ard,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "graph_refresh_epochs": config.graph_refresh_epochs,
        "learn_lengthscale": config.learn_lengthscale,
        "learn_outputscale": config.learn_outputscale,
        "learn_sigma_f": config.learn_sigma_f,
        "learn_sigma_g": config.learn_sigma_g,
        "min_sigma_f": config.min_sigma_f,
        "min_sigma_g": config.min_sigma_g,
    }


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _trajectory_metrics_payload(records: tuple[TrajectoryMetrics, ...]) -> list[dict[str, Any]]:
    return [
        {
            "replica": record.key.replica,
            "dimension": record.key.dimension,
            "trajectory_id": record.key.trajectory_id,
            "n_points": record.n_points,
            "standardized_mse": record.standardized_mse,
            "standardized_rmse": record.standardized_rmse,
            "gaussian_nll": record.gaussian_nll,
            "interval_coverage": [
                {"level": level, "coverage": coverage}
                for level, coverage in record.interval_coverage
            ],
        }
        for record in records
    ]


def _prediction_checks(
    prediction: ScalarPrediction,
    *,
    expected_rows: int,
    value_noise_variance: float,
) -> dict[str, Any]:
    vectors = {
        "mean": prediction.mean,
        "latent_variance": prediction.latent_variance,
        "observation_variance": prediction.observation_variance,
    }
    for name, value in vectors.items():
        if value.shape != (expected_rows,):
            raise InternalTaskError(
                f"{name} has shape {tuple(value.shape)}, expected {(expected_rows,)}"
            )
    mean = _tensor_to_numpy(prediction.mean)
    latent = _tensor_to_numpy(prediction.latent_variance)
    observation = _tensor_to_numpy(prediction.observation_variance)
    checks = {
        "row_count": expected_rows,
        "mean_all_finite": bool(np.isfinite(mean).all()),
        "latent": {
            "all_finite": bool(np.isfinite(latent).all()),
            "all_positive": bool((latent > 0.0).all()),
            "minimum_raw": float(np.min(latent)),
            "maximum_raw": float(np.max(latent)),
        },
        "observation": {
            "all_finite": bool(np.isfinite(observation).all()),
            "all_positive": bool((observation > 0.0).all()),
            "minimum_raw": float(np.min(observation)),
            "maximum_raw": float(np.max(observation)),
        },
        "value_noise_variance": float(value_noise_variance),
        "released_variance_epsilon_floor": (
            None
            if prediction.released_variance_epsilon_floor is None
            else {
                "value": float(prediction.released_variance_epsilon_floor),
                "inactive": prediction.released_variance_epsilon_floor_inactive is True,
                "failure_policy": "equality-to-floor-fails-before-scoring",
            }
        ),
        "observation_is_latent_plus_noise": bool(
            torch.equal(
                prediction.observation_variance,
                prediction.latent_variance
                + prediction.latent_variance.new_tensor(value_noise_variance),
            )
        ),
    }
    valid = bool(
        checks["mean_all_finite"]
        and checks["latent"]["all_finite"]
        and checks["latent"]["all_positive"]
        and checks["observation"]["all_finite"]
        and checks["observation"]["all_positive"]
        and checks["observation_is_latent_plus_noise"]
    )
    checks["all_valid"] = valid
    if not valid:
        raise InternalTaskError("prediction failed raw mean/variance semantics checks")
    return checks


def _optional_float_list(value: torch.Tensor) -> list[float | None]:
    return [float(item) if math.isfinite(float(item)) else None for item in value.detach().cpu()]


def _orbit_matmul_flops(m: int, rank: int) -> int:
    return 10 * m * m * rank + 2 * m * rank * rank + 2 * m * m


def _preconditioner_flops(m: int, rank: int) -> int:
    return 4 * m * m * rank + 4 * m * rank * rank


def _orbit_diagnostics(
    prediction: ScalarPrediction,
    *,
    requested_m: int,
    training_rows: int,
    use_preconditioner: bool,
    tolerance: float,
) -> dict[str, Any]:
    details = prediction.details
    if details is None:
        raise InternalTaskError("ORBIT prediction is missing solver details")
    required = (
        "ranks",
        "iterations",
        "operator_matvecs",
        "preconditioner_applications",
        "relative_residuals",
        "converged",
        "variance_error_upper_bounds",
        "expected_kl_upper_bounds",
        "exact_arithmetic_certified",
        "floating_point_rigorous",
        "basis_exact",
        "finite_precision_variance_corrections",
    )
    if any(not hasattr(details, name) for name in required):
        raise InternalTaskError("ORBIT details do not expose the required solver diagnostics")
    count = prediction.mean.numel()
    if any(getattr(details, name).numel() != count for name in required):
        raise InternalTaskError("ORBIT solver diagnostics do not match prediction rows")

    ranks = [int(value) for value in details.ranks.detach().cpu().tolist()]
    iterations = [int(value) for value in details.iterations.detach().cpu().tolist()]
    operator_matvecs = [
        int(value) for value in details.operator_matvecs.detach().cpu().tolist()
    ]
    preconditioner_applications = [
        int(value) for value in details.preconditioner_applications.detach().cpu().tolist()
    ]
    if any(value < 0 for value in (*iterations, *operator_matvecs, *preconditioner_applications)):
        raise InternalTaskError("ORBIT returned negative iteration or linear-operator call counts")
    if not use_preconditioner and any(preconditioner_applications):
        raise InternalTaskError("ORBIT counted preconditioner applications while disabled")
    residuals = [float(value) for value in details.relative_residuals.detach().cpu().tolist()]
    converged = [bool(value) for value in details.converged.detach().cpu().tolist()]
    if not all(math.isfinite(value) and value >= 0.0 for value in residuals):
        raise InternalTaskError("ORBIT returned invalid freshly recomputed residuals")
    basis_exact = [bool(value) for value in details.basis_exact.detach().cpu().tolist()]
    if not all(converged):
        raise InternalTaskError("ORBIT did not converge for every evaluation target")
    if any(value > tolerance for value in residuals):
        raise InternalTaskError("ORBIT freshly recomputed residual exceeds cg_tolerance")
    if not all(basis_exact):
        raise InternalTaskError("ORBIT exact-rank basis check failed")

    effective_m = min(requested_m, training_rows)
    per_target_resources: list[dict[str, int]] = []
    for rank, iteration, matvec_count, preconditioner_count in zip(
        ranks,
        iterations,
        operator_matvecs,
        preconditioner_applications,
        strict=True,
    ):
        operator_flops = matvec_count * _orbit_matmul_flops(effective_m, rank)
        preconditioner_flops = (
            preconditioner_count * _preconditioner_flops(effective_m, rank)
            if use_preconditioner
            else 0
        )
        per_target_resources.append(
            {
                "rank": rank,
                "iterations": iteration,
                "operator_matvecs": matvec_count,
                "preconditioner_applications": preconditioner_count,
                "reduced_system_dimension": effective_m * rank,
                "operator_core_elements": (
                    3 * effective_m * effective_m + 2 * effective_m * rank + rank * rank + rank
                ),
                "structured_operator_flops": operator_flops,
                "preconditioner_flops": preconditioner_flops,
                "counted_flops": operator_flops + preconditioner_flops,
            }
        )
    counted = [record["counted_flops"] for record in per_target_resources]
    core = [record["operator_core_elements"] for record in per_target_resources]
    return {
        "solver": {
            "fresh_relative_residuals": residuals,
            "maximum_fresh_relative_residual": max(residuals),
            "converged": converged,
            "all_converged": all(converged),
            "iterations": iterations,
            "operator_matvecs": operator_matvecs,
            "preconditioner_applications": preconditioner_applications,
            "ranks": ranks,
            "basis_exact": basis_exact,
            "exact_arithmetic_certified": [
                bool(value) for value in details.exact_arithmetic_certified.detach().cpu().tolist()
            ],
            "floating_point_rigorous": [
                bool(value) for value in details.floating_point_rigorous.detach().cpu().tolist()
            ],
            "variance_error_upper_bounds": _optional_float_list(
                details.variance_error_upper_bounds
            ),
            "expected_kl_upper_bounds": _optional_float_list(details.expected_kl_upper_bounds),
            "finite_precision_variance_corrections": [
                float(value)
                for value in details.finite_precision_variance_corrections.detach().cpu().tolist()
            ],
        },
        "analytic_resources": {
            "counting_schema": "orbit_structured_proxy_v1",
            "requested_m": requested_m,
            "effective_m": effective_m,
            "preconditioner_counted": use_preconditioner,
            "per_target": per_target_resources,
            "operator_core_elements_max": max(core),
            "counted_flops_total": sum(counted),
            "counted_flops_mean_per_target": float(sum(counted) / len(counted)),
        },
    }


def _arm_result(
    *,
    label: str,
    family: str,
    prediction: ScalarPrediction,
    evaluation: Any,
    replica: int,
    dimension: int,
    parameters: FrozenTERAParameters,
    requested_m: int,
    training_rows: int,
    hyperparameters_source: str,
    orbit_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checks = _prediction_checks(
        prediction,
        expected_rows=evaluation.value.shape[0],
        value_noise_variance=parameters.sigma_f,
    )
    target = _tensor_to_numpy(evaluation.value)
    mean = _tensor_to_numpy(prediction.mean)
    trajectory = _tensor_to_numpy(evaluation.trajectory_id)
    latent_metrics = per_trajectory_metrics(
        target,
        mean,
        _tensor_to_numpy(prediction.latent_variance),
        trajectory,
        replica=replica,
        dimension=dimension,
    )
    observation_metrics = per_trajectory_metrics(
        target,
        mean,
        _tensor_to_numpy(prediction.observation_variance),
        trajectory,
        replica=replica,
        dimension=dimension,
    )
    result = {
        "label": label,
        "family": family,
        "requested_m": requested_m,
        "effective_m": min(requested_m, training_rows),
        "hyperparameters_source": hyperparameters_source,
        "raw_prediction_checks": checks,
        "metrics": {
            "latent": _trajectory_metrics_payload(latent_metrics),
            "observation": _trajectory_metrics_payload(observation_metrics),
        },
        "prediction_moments": {
            "mean": mean.tolist(),
            "latent_variance": _tensor_to_numpy(prediction.latent_variance).tolist(),
            "observation_variance": _tensor_to_numpy(prediction.observation_variance).tolist(),
        },
    }
    if orbit_diagnostics is not None:
        result.update(orbit_diagnostics)
    return result


def _tera_resources(training_rows: int) -> dict[str, Any]:
    effective_m = min(REFERENCE_M, training_rows)
    return {
        "counting_schema": "tera_dense_local_v1",
        "requested_m": REFERENCE_M,
        "effective_m": effective_m,
        "explicit_reduced_covariance_elements_per_target": effective_m**4,
        "reduced_cholesky_leading_flops_per_target": (effective_m**6) / 3.0,
    }


def _collect_provenance(
    bundle: PreparedConfirmatoryBundle,
    config: InternalTaskConfig,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    root = repo_root.resolve()
    dependency_files: dict[str, dict[str, str]] = {}
    for name in ("pyproject.toml", "uv.lock"):
        path = root / name
        if not path.is_file():
            raise InternalTaskError(f"required dependency file is missing: {path}")
        dependency_files[name] = {"sha256": _sha256_file(path)}
    data_provenance = bundle.loaded.provenance
    return {
        "git": {
            "commit": str(_run_git(root, "rev-parse", "HEAD")).strip(),
            "tree": str(_run_git(root, "rev-parse", "HEAD^{tree}")).strip(),
            "describe": str(_run_git(root, "describe", "--always", "--dirty", "--tags")).strip(),
            "status_porcelain": str(
                _run_git(root, "status", "--porcelain=v1", "--untracked-files=all")
            ).splitlines(),
        },
        "data": {
            "dataset_path": str(data_provenance.dataset_path.resolve()),
            "metadata_path": str(data_provenance.metadata_path.resolve()),
            "manifest_path": str(data_provenance.sha256_manifest_path.resolve()),
            "file_sha256": dict(sorted(data_provenance.file_sha256.items())),
            "manifest_sha256": _sha256_file(data_provenance.sha256_manifest_path),
            "generator_config": data_provenance.config_payload,
        },
        "task_config": _task_config_payload(config),
        "dependencies": dependency_files,
        "submodules": {
            "status": str(_run_git(root, "submodule", "status", "--recursive")).splitlines(),
            "tera_gitlink": str(_run_git(root, "rev-parse", "HEAD:gp/tera/vendor")).strip(),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": config.device,
            "dtype": config.dtype,
        },
    }


def save_internal_result(result: dict[str, Any], output_path: str | Path) -> Path:
    """Atomically persist a strict JSON result."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)
    return path


def run_internal_task(
    dataset_path: str | Path,
    *,
    catalog_path: str | Path,
    config: InternalTaskConfig | None = None,
    evaluation_split: str = "validation",
    evaluation_design: str = "primary",
    frozen_recipe_path: str | Path | None = None,
    output_path: str | Path | None = None,
    repo_root: str | Path = _REPO_ROOT,
) -> dict[str, Any]:
    """Run one fixed configuration on one verified F02 corpus."""

    config = InternalTaskConfig() if config is None else config
    if evaluation_split not in {"validation", "test"}:
        raise ValueError("evaluation_split must be 'validation' or 'test'")
    if evaluation_design not in _EVALUATION_DESIGNS:
        raise ValueError("evaluation_design must be 'primary' or 'optimizer_selection'")
    if evaluation_split == "test" and evaluation_design != "primary":
        raise FrozenRecipeError("test evaluation permits only the registered primary design")
    if evaluation_split == "validation" and frozen_recipe_path is not None:
        raise FrozenRecipeError("a frozen recipe is only accepted for test evaluation")
    if (
        evaluation_split == "validation"
        and evaluation_design == "primary"
        and not config.candidate_m
    ):
        raise ValueError("primary validation requires a nonempty candidate_m grid")
    if evaluation_split == "test" and len(config.candidate_m) != 1:
        raise FrozenRecipeError("test evaluation requires exactly one selected candidate_m")
    if evaluation_split == "test":
        if frozen_recipe_path is None:
            raise FrozenRecipeError("test evaluation requires a committed frozen-recipe JSON")
        _assert_repository_clean(Path(repo_root))

    identity = _preflight_bundle_identity(dataset_path)
    catalog_authorization = validate_catalog_identity(catalog_path, identity)
    _authorize_evaluation_phase(catalog_authorization, evaluation_split)
    if evaluation_split == "test":
        gate = validate_frozen_recipe(
            frozen_recipe_path,
            identity,
            config,
            catalog_authorization,
            repo_root=repo_root,
        )
        if not CONFIRMATORY_TEST_RELEASED:
            raise FrozenRecipeError(
                "confirmatory test execution remains disabled: the per-bundle recipe "
                "must be replaced by the preregistered global recipe and one-release ledger"
            )
    else:
        gate = {
            "required": False,
            "validated": False,
            "committed_at_head": False,
            "path": None,
            "payload_sha256": None,
            "schema_version": None,
        }

    # In test mode this whole-bundle decompression occurs only after the gate.
    # Validation also verifies the full archive but does not select test rows.
    bundle = load_prepared_confirmatory_bundle(dataset_path)
    loaded_identity = _bundle_identity(bundle)
    if loaded_identity != identity:
        raise InternalTaskError("fully loaded bundle identity changed after catalog preflight")

    # This is the first point at which split='test' can be selected.  It occurs
    # strictly after the recipe gate above.
    selected = _selected_bundle(bundle, evaluation_split, evaluation_design)
    dtype = _dtype(config.dtype)
    train = prepared_split_to_tensors(
        selected,
        "train",
        dtype=dtype,
        device=config.device,
    )
    evaluation = prepared_split_to_tensors(
        selected,
        evaluation_split,
        dtype=dtype,
        device=config.device,
    )

    # Fit receives only the selected training object.  Validation/test tensors
    # are never marshalled into the released MD22Split used for fitting.
    model, fit_seconds, fit_peak_bytes = _measure(
        config.device,
        lambda: fit_released_tera(train, **_fit_kwargs(config)),
    )
    parameters = freeze_tera_parameters(model)
    training_rows = train.X.shape[0]
    replica = int(bundle.loaded.dataset.config.replica)
    dimension = int(bundle.loaded.dataset.config.state_dim)

    arms: dict[str, dict[str, Any]] = {}
    tera_prediction, tera_seconds, tera_peak_bytes = _measure(
        config.device,
        lambda: predict_released_tera(
            train,
            evaluation.X,
            parameters,
            m=REFERENCE_M,
        ),
    )
    tera_arm = _arm_result(
        label="TERA-50",
        family="TERA",
        prediction=tera_prediction,
        evaluation=evaluation,
        replica=replica,
        dimension=dimension,
        parameters=parameters,
        requested_m=REFERENCE_M,
        training_rows=training_rows,
        hyperparameters_source="TERA-gradient-fit",
    )
    tera_arm["analytic_resources"] = _tera_resources(training_rows)
    tera_arm["prediction_seconds_descriptive"] = tera_seconds
    tera_arm["prediction_peak_gpu_allocated_bytes"] = tera_peak_bytes
    arms["TERA-50"] = tera_arm

    for requested_m in (REFERENCE_M, *config.candidate_m):
        prediction, prediction_seconds, prediction_peak_bytes = _measure(
            config.device,
            lambda requested_m=requested_m: predict_orbit(
                train,
                evaluation.X,
                parameters,
                m=requested_m,
                cg_tolerance=config.cg_tolerance,
                cg_max_iterations=config.cg_max_iterations,
                use_preconditioner=config.use_preconditioner,
                function_jitter=config.function_jitter,
                reduced_jitter=config.reduced_jitter,
            ),
        )
        diagnostics = _orbit_diagnostics(
            prediction,
            requested_m=requested_m,
            training_rows=training_rows,
            use_preconditioner=config.use_preconditioner,
            tolerance=config.cg_tolerance,
        )
        label = f"ORBIT-{requested_m}"
        orbit_arm = _arm_result(
            label=label,
            family="ORBIT",
            prediction=prediction,
            evaluation=evaluation,
            replica=replica,
            dimension=dimension,
            parameters=parameters,
            requested_m=requested_m,
            training_rows=training_rows,
            hyperparameters_source="TERA-gradient-fit",
            orbit_diagnostics=diagnostics,
        )
        orbit_arm["prediction_seconds_descriptive"] = prediction_seconds
        orbit_arm["prediction_peak_gpu_allocated_bytes"] = prediction_peak_bytes
        if requested_m == REFERENCE_M:
            maxabs_mean = float(
                torch.max(torch.abs(prediction.mean - tera_prediction.mean)).detach().cpu()
            )
            maxabs_latent_variance = float(
                torch.max(
                    torch.abs(prediction.latent_variance - tera_prediction.latent_variance)
                )
                .detach()
                .cpu()
            )
            same_m_tolerance = SAME_M_ABSOLUTE_TOLERANCE[config.dtype]
            same_m_pass = bool(
                maxabs_mean <= same_m_tolerance
                and maxabs_latent_variance <= same_m_tolerance
            )
            orbit_arm["same_m_agreement_to_TERA_50"] = {
                "maxabs_mean": maxabs_mean,
                "maxabs_latent_variance": maxabs_latent_variance,
                "absolute_tolerance": same_m_tolerance,
                "passes": same_m_pass,
            }
            if not same_m_pass:
                raise InternalTaskError(
                    "ORBIT-50 failed the preregistered same-m agreement control: "
                    f"mean={maxabs_mean}, latent_variance={maxabs_latent_variance}, "
                    f"tolerance={same_m_tolerance}"
                )
        arms[label] = orbit_arm

    value_prediction, value_seconds, value_peak_bytes = _measure(
        config.device,
        lambda: predict_value_only_local_gp(
            train,
            evaluation.X,
            parameters,
            m=REFERENCE_M,
        ),
    )
    arms["value-only-conditional-50"] = _arm_result(
        label="value-only-conditional-50",
        family="value-only-conditioning-ablation",
        prediction=value_prediction,
        evaluation=evaluation,
        replica=replica,
        dimension=dimension,
        parameters=parameters,
        requested_m=REFERENCE_M,
        training_rows=training_rows,
        hyperparameters_source="TERA-gradient-fit",
    )
    arms["value-only-conditional-50"]["control_semantics"] = (
        "prediction-time conditioning ablation; not a standalone value-only hyperparameter fit"
    )
    arms["value-only-conditional-50"]["prediction_seconds_descriptive"] = value_seconds
    arms["value-only-conditional-50"]["prediction_peak_gpu_allocated_bytes"] = value_peak_bytes

    effective_batch_size = min(config.batch_size, training_rows)
    if config.train_steps > 0:
        optimizer_updates = config.train_steps
        target_factors_processed = config.train_steps * effective_batch_size
    else:
        updates_per_epoch = math.ceil(training_rows / effective_batch_size)
        optimizer_updates = config.train_epochs * updates_per_epoch
        target_factors_processed = config.train_epochs * training_rows

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete",
        "task_config": _task_config_payload(config),
        "training": {
            "split": "train",
            "time_indices": list(TRAIN_TIME_INDICES),
            "rows": training_rows,
            "training_m": config.training_m,
            "train_steps": config.train_steps,
            "train_epochs": config.train_epochs,
            "batch_size": config.batch_size,
            "effective_batch_size": effective_batch_size,
            "optimizer_updates": optimizer_updates,
            "vecchia_target_factors_processed": target_factors_processed,
            "fit_seconds_descriptive": fit_seconds,
            "fit_peak_gpu_allocated_bytes": fit_peak_bytes,
        },
        "evaluation": {
            "split": evaluation_split,
            "design": evaluation_design,
            "time_indices": list(_EVALUATION_DESIGNS[evaluation_design]),
            "test_gate": gate,
        },
        "corpus": {
            "replica": replica,
            "dimension": dimension,
            "train_rows": training_rows,
            "evaluation_rows": evaluation.X.shape[0],
            "train_source_indices": train.source_indices.detach().cpu().tolist(),
            "evaluation_source_indices": evaluation.source_indices.detach().cpu().tolist(),
            "evaluation_trajectory_ids": sorted(
                set(int(value) for value in evaluation.trajectory_id.detach().cpu().tolist())
            ),
        },
        "frozen_parameters": {
            "kernel": parameters.kernel,
            "lengthscale": parameters.lengthscale.detach().cpu().tolist(),
            "outputscale": parameters.outputscale,
            "sigma_f_variance": parameters.sigma_f,
            "sigma_g_variance": parameters.sigma_g,
            "gradient_noise_model": parameters.gradient_noise_model,
        },
        "arms": arms,
        "catalog": {
            "path": str(catalog_authorization.catalog_path.resolve()),
            "sha256": catalog_authorization.catalog_sha256,
            "generation_git_commit": catalog_authorization.generation_git_commit,
            "generation_git_tree": catalog_authorization.generation_git_tree,
            "task_index": catalog_authorization.bundle_entry.get("task_index"),
        },
        "provenance": _collect_provenance(
            bundle,
            config,
            repo_root=Path(repo_root),
        ),
    }
    result = _json_compatible(result)
    if output_path is not None:
        save_internal_result(result, output_path)
    return result


def _parse_candidate_m(value: str) -> tuple[int, ...]:
    if value.strip().lower() in {"none", "empty"}:
        return ()
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "candidate m values must be comma-separated integers"
        ) from error
    if not parsed:
        raise argparse.ArgumentTypeError("use 'none' for an explicit empty candidate grid")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--evaluation-split",
        choices=("validation", "test"),
        default="validation",
    )
    parser.add_argument(
        "--evaluation-design",
        choices=("primary", "optimizer_selection"),
        default="primary",
    )
    parser.add_argument("--frozen-recipe", type=Path)
    parser.add_argument("--training-m", type=int, default=20)
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--train-epochs", type=int, default=0)
    parser.add_argument("--kernel", choices=("rbf", "matern52"), default="rbf")
    parser.add_argument("--outputscale", type=float, default=1.0)
    parser.add_argument("--sigma-f", type=float, default=1e-3)
    parser.add_argument("--sigma-g", type=float, default=1e-3)
    parser.add_argument(
        "--lengthscale",
        type=float,
        default=1.0,
        help="initial lengthscale (released paper recipe: 1.0)",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--candidate-m", type=_parse_candidate_m, default=DEFAULT_CANDIDATE_M)
    parser.add_argument("--cg-tolerance", type=float, default=1e-5)
    parser.add_argument("--cg-max-iterations", type=int)
    parser.add_argument(
        "--use-preconditioner",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--function-jitter", type=float, default=1e-8)
    parser.add_argument("--reduced-jitter", type=float, default=1e-8)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = InternalTaskConfig(
        training_m=args.training_m,
        train_steps=args.train_steps,
        train_epochs=args.train_epochs,
        kernel=args.kernel,
        outputscale=args.outputscale,
        sigma_f=args.sigma_f,
        sigma_g=args.sigma_g,
        lengthscale=args.lengthscale,
        seed=args.seed,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        candidate_m=args.candidate_m,
        cg_tolerance=args.cg_tolerance,
        cg_max_iterations=args.cg_max_iterations,
        use_preconditioner=args.use_preconditioner,
        function_jitter=args.function_jitter,
        reduced_jitter=args.reduced_jitter,
        dtype=args.dtype,
        device=args.device,
    )
    run_internal_task(
        args.dataset,
        catalog_path=args.catalog,
        config=config,
        evaluation_split=args.evaluation_split,
        evaluation_design=args.evaluation_design,
        frozen_recipe_path=args.frozen_recipe,
        output_path=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BundleIdentity",
    "CatalogAuthorization",
    "DEFAULT_CANDIDATE_M",
    "FROZEN_RECIPE_SCHEMA_VERSION",
    "FrozenRecipeError",
    "InternalTaskConfig",
    "InternalTaskError",
    "REFERENCE_M",
    "RESULT_SCHEMA_VERSION",
    "build_frozen_recipe_document",
    "build_parser",
    "run_internal_task",
    "save_internal_result",
    "validate_catalog_identity",
    "validate_frozen_recipe",
]
