"""Non-executable one-release ledger scaffold for F02 confirmatory recipes.

Version 1 contains structural validators only.  Every public state mutation is
source-disabled until the actual runner, semantic evidence validators, and an
immutable result store are integrated in a separately reviewed schema/code
revision.  The read-only audit machinery remains available for fixture and
format inspection, but this module cannot create a legal release ledger.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

from experiments.f02_global_release import (
    DEVELOPMENT_EVIDENCE_SEMANTICALLY_VERIFIED,
    EXPECTED_BUNDLE_COUNT,
    EXPECTED_METHOD_COUNT,
    EXPECTED_TASK_COUNT,
    METHOD_IDS,
    RECIPE_SCAFFOLD_STATUS,
    RECIPE_SCHEMA_VERSION,
    GlobalRecipeError,
    canonical_json_bytes,
    read_strict_json,
    read_strict_json_bytes,
    sha256_bytes,
    validate_global_recipe,
)

LEDGER_SCHEMA_VERSION = "f02_one_release_ledger_v1"
CATALOG_RELEASE_MARKER_SCHEMA_VERSION = "f02_catalog_release_marker_v1"
RESULT_ATTESTATION_SCHEMA_VERSION = "f02_result_attestation_v1"
RELEASE_MUTATIONS_ENABLED = False
RELEASE_MUTATION_DISABLED_REASON = (
    "F02 release mutations are disabled: actual runner, semantic evidence validators, "
    "immutable result store not integrated"
)
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_RECIPE_ROOT = Path(__file__).resolve().parents[1]
_BUNDLE_HASH_KEYS = (
    "dataset_file_sha256",
    "metadata_file_sha256",
    "sha256_manifest_file_sha256",
    "dataset_content_sha256",
)


class ReleaseLedgerError(RuntimeError):
    """Raised when release authorization or append-only accounting fails."""


def _reject_release_mutation() -> None:
    """Fail closed until a reviewed code and schema revision enables release."""

    raise ReleaseLedgerError(RELEASE_MUTATION_DISABLED_REASON)


def _run_git(repo_root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReleaseLedgerError(f"git {' '.join(arguments)} failed") from error
    return completed.stdout


def _require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ReleaseLedgerError(
            f"{label} fields mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _is_oid(value: Any) -> bool:
    return isinstance(value, str) and _HEX40.fullmatch(value) is not None


def _read_regular_file_snapshot(path: str | Path, *, label: str) -> tuple[bytes, str]:
    """Read and hash one regular file through the same no-follow descriptor."""

    source = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow
    elif source.is_symlink():
        raise ReleaseLedgerError(f"{label} must not be a symlink")
    descriptor: int | None = None
    try:
        descriptor = os.open(source, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ReleaseLedgerError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ReleaseLedgerError(
            f"cannot take a no-follow snapshot of {label}: {source}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    snapshot = b"".join(chunks)
    if before_identity != after_identity or len(snapshot) != before.st_size:
        raise ReleaseLedgerError(f"{label} changed while its snapshot was read")
    return snapshot, sha256_bytes(snapshot)


def _read_json_snapshot(path: str | Path, *, label: str) -> tuple[dict[str, Any], str]:
    snapshot, digest = _read_regular_file_snapshot(path, label=label)
    try:
        document = read_strict_json_bytes(snapshot, label=label)
    except GlobalRecipeError as error:
        raise ReleaseLedgerError(str(error)) from error
    return document, digest


def _recipe_tasks(document: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    try:
        grids = document["payload"]["expected_task_grids"]
        tasks = tuple(task for grid in grids for task in grid["tasks"])
    except (KeyError, TypeError) as error:
        raise ReleaseLedgerError("validated recipe task grids are unavailable") from error
    if len(tasks) != EXPECTED_TASK_COUNT:
        raise ReleaseLedgerError("validated recipe does not contain exactly 450 tasks")
    task_ids = [task.get("task_id") for task in tasks]
    if any(not isinstance(task_id, str) for task_id in task_ids) or len(set(task_ids)) != len(
        tasks
    ):
        raise ReleaseLedgerError("validated recipe task IDs are invalid or duplicated")
    return tasks


def _recipe_bundle_for_task(
    recipe_document: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, Any]:
    """Return the one recipe bundle whose stable coordinates match the task."""

    try:
        bundles = recipe_document["payload"]["confirmatory_bundles"]
    except (KeyError, TypeError) as error:
        raise ReleaseLedgerError("validated recipe bundle identities are unavailable") from error
    if not isinstance(bundles, list):
        raise ReleaseLedgerError("validated recipe bundle identities are not a list")
    coordinate_fields = (
        "bundle_id",
        "catalog_task_index",
        "replica",
        "n_particles",
        "dimension",
    )
    matches = [
        bundle
        for bundle in bundles
        if isinstance(bundle, dict)
        and all(bundle.get(field) == task.get(field) for field in coordinate_fields)
    ]
    if len(matches) != 1:
        raise ReleaseLedgerError("result task does not match exactly one recipe bundle")
    bundle = matches[0]
    hashes = bundle.get("hashes")
    if not isinstance(hashes, dict) or set(hashes) != set(_BUNDLE_HASH_KEYS):
        raise ReleaseLedgerError("recipe bundle does not contain the exact four registered hashes")
    if any(not _is_sha256(hashes[key]) for key in _BUNDLE_HASH_KEYS):
        raise ReleaseLedgerError("recipe bundle contains an invalid registered hash")
    return bundle


def verify_recipe_only_release(
    recipe_path: str | Path,
    document: dict[str, Any],
    *,
    repo_root: str | Path,
) -> dict[str, str]:
    """Verify current HEAD is R: one recipe-only child of source commit S."""

    root = Path(repo_root).resolve()
    recipe_input = Path(recipe_path)
    if recipe_input.is_symlink():
        raise ReleaseLedgerError("release recipe must be a regular non-symlink file")
    recipe = recipe_input.resolve()
    try:
        relative = recipe.relative_to(root).as_posix()
    except ValueError as error:
        raise ReleaseLedgerError("release recipe must be inside the source repository") from error
    expected_relative = document["payload"]["release"]["recipe_path"]
    if relative != expected_relative:
        raise ReleaseLedgerError("recipe location does not match the globally registered path")
    if not recipe.is_file():
        raise ReleaseLedgerError("release recipe must be a regular non-symlink file")

    status = _run_git(root, "status", "--porcelain=v1", "--untracked-files=all").decode().strip()
    if status:
        raise ReleaseLedgerError("release authorization requires a globally clean worktree")
    head = _run_git(root, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    tree = _run_git(root, "rev-parse", "HEAD^{tree}").decode().strip()
    parents = _run_git(root, "rev-list", "--parents", "-n", "1", head).decode().split()
    if len(parents) != 2:
        raise ReleaseLedgerError("execution release R must have exactly one parent")
    source_commit = document["payload"]["release"]["source_commit"]
    if parents[1] != source_commit:
        raise ReleaseLedgerError("execution release parent is not registered source commit S")

    changes = (
        _run_git(
            root,
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "--no-renames",
            "-r",
            source_commit,
            head,
        )
        .decode()
        .splitlines()
    )
    if changes != [f"A\t{relative}"]:
        raise ReleaseLedgerError("execution release R must only add the registered recipe file")
    committed = _run_git(root, "show", f"{head}:{relative}")
    if committed != recipe.read_bytes():
        raise ReleaseLedgerError("worktree recipe is not byte-identical to execution release R")
    if not _is_oid(head) or not _is_oid(tree):
        raise ReleaseLedgerError("execution release commit/tree are invalid")
    return {
        "source_commit": source_commit,
        "source_tree": document["payload"]["release"]["source_tree"],
        "execution_commit": head,
        "execution_tree": tree,
        "recipe_path": relative,
        "recipe_file_sha256": sha256_bytes(committed),
    }


def _authorization_payload(
    recipe_path: str | Path,
    catalog_path: str | Path,
    *,
    repo_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        document, summary = validate_global_recipe(
            recipe_path,
            catalog_path=catalog_path,
            repo_root=repo_root,
        )
    except GlobalRecipeError as error:
        raise ReleaseLedgerError(str(error)) from error
    release = verify_recipe_only_release(recipe_path, document, repo_root=repo_root)
    if release["recipe_file_sha256"] != summary["recipe_file_sha256"]:
        raise ReleaseLedgerError("release recipe file hash changed during authorization")
    authorization = {
        "recipe_schema_version": RECIPE_SCHEMA_VERSION,
        "release_scaffold_status": summary["release_scaffold_status"],
        "release_mutations_enabled": RELEASE_MUTATIONS_ENABLED,
        "development_evidence_semantically_verified": summary[
            "development_evidence_semantically_verified"
        ],
        "experiment_id": summary["experiment_id"],
        "protocol_id": summary["protocol_id"],
        "protocol_path": summary["protocol_path"],
        "protocol_sha256": summary["protocol_sha256"],
        "protocol_binding_sha256": summary["protocol_binding_sha256"],
        "recipe_path": release["recipe_path"],
        "recipe_file_sha256": release["recipe_file_sha256"],
        "recipe_payload_sha256": summary["payload_sha256"],
        "source_commit": release["source_commit"],
        "source_tree": release["source_tree"],
        "execution_commit": release["execution_commit"],
        "execution_tree": release["execution_tree"],
        "catalog_sha256": summary["catalog_sha256"],
        "required_method_ids": list(METHOD_IDS),
        "expected_bundle_count": summary["expected_bundle_count"],
        "expected_method_count": summary["expected_method_count"],
        "expected_task_count": summary["expected_task_count"],
        "expected_task_ids_sha256": summary["expected_task_ids_sha256"],
    }
    authorization["release_id"] = sha256_bytes(canonical_json_bytes(authorization))
    return document, summary, authorization


def _ledger_digest(authorization: dict[str, Any], events: list[dict[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes({"authorization": authorization, "events": events}))


def _new_ledger(authorization: dict[str, Any]) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "authorization": authorization,
        "events": events,
        "ledger_state_sha256": _ledger_digest(authorization, events),
    }


def _write_atomic(path: Path, document: dict[str, Any]) -> None:
    encoded = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            os.fchmod(handle.fileno(), 0o600)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as error:
        raise ReleaseLedgerError(f"cannot atomically write ledger: {path}") from error
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass


def _write_exclusive(path: Path, document: dict[str, Any], *, label: str) -> None:
    encoded = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as error:
        raise ReleaseLedgerError(f"refusing to replace {label}: {path}") from error


@contextmanager
def _ledger_lock(ledger_path: Path) -> Iterator[None]:
    if not ledger_path.parent.is_dir():
        raise ReleaseLedgerError(f"ledger parent directory does not exist: {ledger_path.parent}")
    lock_path = ledger_path.with_name(f".{ledger_path.name}.lock")
    try:
        with lock_path.open("a+b") as handle:
            os.fchmod(handle.fileno(), 0o600)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError as error:
        raise ReleaseLedgerError(f"cannot lock ledger: {ledger_path}") from error


def catalog_ledger_path(catalog_path: str | Path) -> Path:
    """Return the conventional sibling ledger path (mainly useful for isolated tests)."""

    catalog = Path(catalog_path)
    if catalog.is_symlink():
        raise ReleaseLedgerError("release catalog must not be a symlink")
    catalog = catalog.resolve()
    return catalog.with_name(f"{catalog.stem}.one-release-ledger.json")


def catalog_release_marker_path(catalog_path: str | Path) -> Path:
    """Return the immutable authorization marker owned by one canonical catalog."""

    catalog = Path(catalog_path)
    if catalog.is_symlink():
        raise ReleaseLedgerError("release catalog must not be a symlink")
    catalog = catalog.resolve()
    return catalog.with_name(f"{catalog.stem}.release-authorization.json")


def _require_external_ledger(ledger_path: Path, repo_root: Path) -> None:
    if ledger_path.is_symlink():
        raise ReleaseLedgerError("one-release ledger must not be a symlink")
    try:
        ledger_path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return
    raise ReleaseLedgerError("one-release ledger must be outside the Git checkout")


def _release_marker_document(
    catalog_path: str | Path,
    ledger_path: Path,
    authorization: dict[str, Any],
    initial_ledger: dict[str, Any],
) -> dict[str, Any]:
    base = {
        "catalog_path": str(Path(catalog_path).resolve()),
        "catalog_sha256": authorization["catalog_sha256"],
        "ledger_path": str(ledger_path.resolve()),
        "release_id": authorization["release_id"],
        "recipe_file_sha256": authorization["recipe_file_sha256"],
        "recipe_payload_sha256": authorization["recipe_payload_sha256"],
        "execution_commit": authorization["execution_commit"],
        "initial_ledger_state_sha256": initial_ledger["ledger_state_sha256"],
    }
    return {
        "schema_version": CATALOG_RELEASE_MARKER_SCHEMA_VERSION,
        **base,
        "marker_sha256": sha256_bytes(canonical_json_bytes(base)),
    }


def _validate_release_marker(
    catalog_path: str | Path,
    ledger_path: Path,
    authorization: dict[str, Any],
) -> None:
    marker_path = catalog_release_marker_path(catalog_path)
    try:
        marker = read_strict_json(marker_path, label="F02 catalog release marker")
    except GlobalRecipeError as error:
        raise ReleaseLedgerError(str(error)) from error
    expected_keys = {
        "schema_version",
        "catalog_path",
        "catalog_sha256",
        "ledger_path",
        "release_id",
        "recipe_file_sha256",
        "recipe_payload_sha256",
        "execution_commit",
        "initial_ledger_state_sha256",
        "marker_sha256",
    }
    _require_exact_keys(marker, expected_keys, "catalog release marker")
    if marker["schema_version"] != CATALOG_RELEASE_MARKER_SCHEMA_VERSION:
        raise ReleaseLedgerError("unsupported catalog release marker schema_version")
    base = {
        key: value
        for key, value in marker.items()
        if key not in {"schema_version", "marker_sha256"}
    }
    if marker["marker_sha256"] != sha256_bytes(canonical_json_bytes(base)):
        raise ReleaseLedgerError("catalog release marker SHA-256 is invalid")
    expected = {
        "catalog_path": str(Path(catalog_path).resolve()),
        "catalog_sha256": authorization["catalog_sha256"],
        "ledger_path": str(ledger_path.resolve()),
        "release_id": authorization["release_id"],
        "recipe_file_sha256": authorization["recipe_file_sha256"],
        "recipe_payload_sha256": authorization["recipe_payload_sha256"],
        "execution_commit": authorization["execution_commit"],
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise ReleaseLedgerError("catalog release marker does not match this ledger/release")
    expected_initial = _ledger_digest(authorization, [])
    if marker["initial_ledger_state_sha256"] != expected_initial:
        raise ReleaseLedgerError("catalog release marker initial ledger hash is invalid")


def authorize_release(
    recipe_path: str | Path,
    catalog_path: str | Path,
    ledger_path: str | Path,
    *,
    repo_root: str | Path = _RECIPE_ROOT,
) -> dict[str, Any]:
    """Reject v1 authorization before inspecting or mutating any supplied path."""

    _reject_release_mutation()
    ledger_file = Path(ledger_path)
    root = Path(repo_root)
    _require_external_ledger(ledger_file, root)
    _, _, authorization = _authorization_payload(
        recipe_path,
        catalog_path,
        repo_root=root,
    )
    document = _new_ledger(authorization)
    marker_path = catalog_release_marker_path(catalog_path)
    marker = _release_marker_document(catalog_path, ledger_file, authorization, document)
    with _ledger_lock(marker_path):
        if marker_path.exists():
            raise ReleaseLedgerError("ledger already exists; re-release is forbidden")
        if ledger_file.exists():
            raise ReleaseLedgerError("requested ledger path already exists")
        # The immutable catalog-owned reservation is written first.  If the
        # subsequent ledger write fails, the release stays fail-closed rather
        # than permitting a second authorization under another filename.
        _write_exclusive(marker_path, marker, label="catalog release marker")
        _write_exclusive(ledger_file, document, label="initial one-release ledger")
    return document


def _event_hash_base(
    sequence: int,
    kind: str,
    previous_event_sha256: str | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "kind": kind,
        "previous_event_sha256": previous_event_sha256,
        "payload": payload,
    }


def _make_event(
    events: list[dict[str, Any]],
    kind: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    previous = events[-1]["event_sha256"] if events else None
    base = _event_hash_base(len(events), kind, previous, payload)
    return {**base, "event_sha256": sha256_bytes(canonical_json_bytes(base))}


def _validate_authorization(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReleaseLedgerError("ledger authorization must be an object")
    expected = {
        "recipe_schema_version",
        "release_scaffold_status",
        "release_mutations_enabled",
        "development_evidence_semantically_verified",
        "experiment_id",
        "protocol_id",
        "protocol_path",
        "protocol_sha256",
        "protocol_binding_sha256",
        "recipe_path",
        "recipe_file_sha256",
        "recipe_payload_sha256",
        "source_commit",
        "source_tree",
        "execution_commit",
        "execution_tree",
        "catalog_sha256",
        "required_method_ids",
        "expected_bundle_count",
        "expected_method_count",
        "expected_task_count",
        "expected_task_ids_sha256",
        "release_id",
    }
    _require_exact_keys(value, expected, "ledger authorization")
    if value["recipe_schema_version"] != RECIPE_SCHEMA_VERSION:
        raise ReleaseLedgerError("ledger recipe schema is unsupported")
    if (
        value["release_scaffold_status"] != RECIPE_SCAFFOLD_STATUS
        or value["release_mutations_enabled"] is not False
        or value["development_evidence_semantically_verified"]
        is not DEVELOPMENT_EVIDENCE_SEMANTICALLY_VERIFIED
    ):
        raise ReleaseLedgerError("ledger authorization is not the non-executable v1 scaffold")
    if not isinstance(value["experiment_id"], str) or not value["experiment_id"]:
        raise ReleaseLedgerError("ledger experiment_id is invalid")
    if not isinstance(value["protocol_id"], str) or not value["protocol_id"]:
        raise ReleaseLedgerError("ledger protocol_id is invalid")
    if not isinstance(value["protocol_path"], str) or not value["protocol_path"]:
        raise ReleaseLedgerError("ledger protocol_path is invalid")
    if value["required_method_ids"] != list(METHOD_IDS):
        raise ReleaseLedgerError("ledger does not authorize all required methods")
    if (
        value["expected_bundle_count"] != EXPECTED_BUNDLE_COUNT
        or value["expected_method_count"] != EXPECTED_METHOD_COUNT
        or value["expected_task_count"] != EXPECTED_TASK_COUNT
    ):
        raise ReleaseLedgerError("ledger does not authorize the exact 50/3/450 release grid")
    for field in (
        "recipe_file_sha256",
        "recipe_payload_sha256",
        "protocol_sha256",
        "protocol_binding_sha256",
        "catalog_sha256",
        "expected_task_ids_sha256",
        "release_id",
    ):
        if not _is_sha256(value[field]):
            raise ReleaseLedgerError(f"ledger authorization {field} is invalid")
    for field in ("source_commit", "source_tree", "execution_commit", "execution_tree"):
        if not _is_oid(value[field]):
            raise ReleaseLedgerError(f"ledger authorization {field} is invalid")
    base = {key: item for key, item in value.items() if key != "release_id"}
    if value["release_id"] != sha256_bytes(canonical_json_bytes(base)):
        raise ReleaseLedgerError("ledger release_id is invalid")
    return value


def _validate_slurm(value: Any) -> None:
    if not isinstance(value, dict):
        raise ReleaseLedgerError("attempt Slurm identity must be an object")
    _require_exact_keys(value, {"job_id", "array_job_id", "array_task_id"}, "attempt slurm")
    if any(
        not isinstance(value[field], str) or not value[field]
        for field in ("job_id", "array_job_id")
    ):
        raise ReleaseLedgerError("attempt Slurm job IDs must be nonempty strings")
    if isinstance(value["array_task_id"], bool) or not isinstance(value["array_task_id"], int):
        raise ReleaseLedgerError("attempt Slurm array_task_id must be an integer")


def _validate_event_log(
    events: Any,
    expected_tasks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(events, list):
        raise ReleaseLedgerError("ledger events must be a list")
    attempts: dict[str, dict[str, Any]] = {}
    active_by_task: dict[str, str] = {}
    successful_tasks: set[str] = set()
    failed_attempt_count = 0
    sealed = False
    previous: str | None = None
    for sequence, event in enumerate(events):
        if not isinstance(event, dict):
            raise ReleaseLedgerError("ledger events must be objects")
        _require_exact_keys(
            event,
            {"sequence", "kind", "previous_event_sha256", "payload", "event_sha256"},
            f"ledger event {sequence}",
        )
        payload = event["payload"]
        if (
            event["sequence"] != sequence
            or event["previous_event_sha256"] != previous
            or not isinstance(payload, dict)
        ):
            raise ReleaseLedgerError("ledger event sequence/hash chain is invalid")
        base = _event_hash_base(sequence, event["kind"], previous, payload)
        if event["event_sha256"] != sha256_bytes(canonical_json_bytes(base)):
            raise ReleaseLedgerError("ledger event SHA-256 is invalid")
        if sealed:
            raise ReleaseLedgerError("ledger contains events after its release seal")

        kind = event["kind"]
        if kind == "attempt_started":
            _require_exact_keys(payload, {"attempt_id", "task_id", "slurm"}, "attempt_started")
            attempt_id = payload["attempt_id"]
            task_id = payload["task_id"]
            if not isinstance(attempt_id, str) or not _IDENTIFIER.fullmatch(attempt_id):
                raise ReleaseLedgerError("attempt_id is invalid")
            if task_id not in expected_tasks:
                raise ReleaseLedgerError("attempt references an unregistered task")
            if attempt_id in attempts:
                raise ReleaseLedgerError("attempt_id is duplicated")
            if task_id in active_by_task:
                raise ReleaseLedgerError("task has overlapping active attempts")
            if task_id in successful_tasks:
                raise ReleaseLedgerError("successful task was attempted again")
            _validate_slurm(payload["slurm"])
            attempts[attempt_id] = {"task_id": task_id, "finished": False, "outcome": None}
            active_by_task[task_id] = attempt_id
        elif kind == "attempt_finished":
            _require_exact_keys(
                payload,
                {"attempt_id", "task_id", "outcome", "exit_code", "failure_code", "result"},
                "attempt_finished",
            )
            attempt_id = payload["attempt_id"]
            task_id = payload["task_id"]
            attempt = attempts.get(attempt_id)
            if attempt is None or attempt["finished"] or attempt["task_id"] != task_id:
                raise ReleaseLedgerError("attempt_finished does not match one open attempt")
            if isinstance(payload["exit_code"], bool) or not isinstance(payload["exit_code"], int):
                raise ReleaseLedgerError("attempt exit_code must be an integer")
            outcome = payload["outcome"]
            result = payload["result"]
            if outcome == "succeeded":
                if payload["exit_code"] != 0 or payload["failure_code"] is not None:
                    raise ReleaseLedgerError(
                        "successful attempt must have exit_code=0 and no failure"
                    )
                if not isinstance(result, dict):
                    raise ReleaseLedgerError("successful attempt requires result hashes")
                _require_exact_keys(
                    result,
                    {"result_file_sha256", "result_attestation_sha256"},
                    "successful result",
                )
                if any(not _is_sha256(item) for item in result.values()):
                    raise ReleaseLedgerError("successful result SHA-256 is invalid")
                successful_tasks.add(task_id)
            elif outcome == "failed":
                if not isinstance(payload["failure_code"], str) or not payload["failure_code"]:
                    raise ReleaseLedgerError("failed attempt requires a failure_code")
                if result is not None:
                    if not isinstance(result, dict):
                        raise ReleaseLedgerError("failed result record must be null or an object")
                    _require_exact_keys(result, {"result_file_sha256"}, "failed result")
                    if not _is_sha256(result["result_file_sha256"]):
                        raise ReleaseLedgerError("failed result SHA-256 is invalid")
                failed_attempt_count += 1
            else:
                raise ReleaseLedgerError("attempt outcome must be succeeded or failed")
            attempt["finished"] = True
            attempt["outcome"] = outcome
            del active_by_task[task_id]
        elif kind == "release_sealed":
            _require_exact_keys(
                payload,
                {"completed_task_count", "completed_task_ids_sha256"},
                "release_sealed",
            )
            ordered_success = [task_id for task_id in expected_tasks if task_id in successful_tasks]
            if (
                active_by_task
                or len(successful_tasks) != len(expected_tasks)
                or payload["completed_task_count"] != len(expected_tasks)
                or payload["completed_task_ids_sha256"]
                != sha256_bytes(canonical_json_bytes(ordered_success))
            ):
                raise ReleaseLedgerError(
                    "release was sealed before the complete task grid succeeded"
                )
            sealed = True
        else:
            raise ReleaseLedgerError(f"unsupported ledger event kind: {kind}")
        previous = event["event_sha256"]
    return {
        "attempt_count": len(attempts),
        "failed_attempt_count": failed_attempt_count,
        "active_attempt_count": len(active_by_task),
        "successful_task_count": len(successful_tasks),
        "remaining_task_count": len(expected_tasks) - len(successful_tasks),
        "sealed": sealed,
        "active_by_task": active_by_task,
        "successful_tasks": successful_tasks,
        "attempts": attempts,
    }


def _load_ledger(
    path: Path, document_recipe: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        ledger = read_strict_json(path, label="F02 one-release ledger")
    except GlobalRecipeError as error:
        raise ReleaseLedgerError(str(error)) from error
    _require_exact_keys(
        ledger,
        {"schema_version", "authorization", "events", "ledger_state_sha256"},
        "ledger",
    )
    if ledger["schema_version"] != LEDGER_SCHEMA_VERSION:
        raise ReleaseLedgerError("unsupported ledger schema_version")
    authorization = _validate_authorization(ledger["authorization"])
    if ledger["ledger_state_sha256"] != _ledger_digest(authorization, ledger["events"]):
        raise ReleaseLedgerError("ledger state SHA-256 is invalid")
    tasks = _recipe_tasks(document_recipe)
    task_map = {task["task_id"]: task for task in tasks}
    state = _validate_event_log(ledger["events"], task_map)
    return ledger, state


def _verify_authorized_release(
    ledger: dict[str, Any],
    recipe_path: str | Path,
    catalog_path: str | Path,
    *,
    repo_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    document, _, expected_authorization = _authorization_payload(
        recipe_path,
        catalog_path,
        repo_root=repo_root,
    )
    if ledger["authorization"] != expected_authorization:
        raise ReleaseLedgerError(
            "ledger authorization does not match the immutable recipe/catalog/release"
        )
    return document, expected_authorization


def _load_for_mutation(
    ledger_path: Path,
    recipe_path: str | Path,
    catalog_path: str | Path,
    *,
    repo_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not ledger_path.is_file():
        raise ReleaseLedgerError("one-release ledger does not exist")
    raw_recipe = read_strict_json(recipe_path, label="global F02 recipe")
    ledger, _ = _load_ledger(ledger_path, raw_recipe)
    document, _ = _verify_authorized_release(
        ledger,
        recipe_path,
        catalog_path,
        repo_root=repo_root,
    )
    ledger, state = _load_ledger(ledger_path, document)
    _validate_release_marker(catalog_path, ledger_path, ledger["authorization"])
    return ledger, state, document


def _append_event(ledger: dict[str, Any], kind: str, payload: dict[str, Any]) -> None:
    ledger["events"].append(_make_event(ledger["events"], kind, payload))
    ledger["ledger_state_sha256"] = _ledger_digest(ledger["authorization"], ledger["events"])


def begin_attempt(
    recipe_path: str | Path,
    catalog_path: str | Path,
    ledger_path: str | Path,
    *,
    task_id: str,
    attempt_id: str,
    slurm_job_id: str,
    slurm_array_job_id: str,
    slurm_array_task_id: int,
    repo_root: str | Path = _RECIPE_ROOT,
) -> dict[str, Any]:
    """Reject v1 attempt creation before inspecting or mutating any supplied path."""

    _reject_release_mutation()
    ledger_file = Path(ledger_path)
    _require_external_ledger(ledger_file, Path(repo_root))
    with _ledger_lock(ledger_file):
        ledger, state, document = _load_for_mutation(
            ledger_file,
            recipe_path,
            catalog_path,
            repo_root=repo_root,
        )
        task_ids = {task["task_id"] for task in _recipe_tasks(document)}
        if task_id not in task_ids:
            raise ReleaseLedgerError("attempt task_id is not in the global 450-task recipe")
        if not _IDENTIFIER.fullmatch(attempt_id):
            raise ReleaseLedgerError("attempt_id is invalid")
        if attempt_id in state["attempts"]:
            raise ReleaseLedgerError("attempt_id has already been recorded")
        if state["sealed"]:
            raise ReleaseLedgerError("release is sealed; no further attempts are permitted")
        if task_id in state["active_by_task"]:
            raise ReleaseLedgerError("task already has an active attempt")
        if task_id in state["successful_tasks"]:
            raise ReleaseLedgerError("task already succeeded and cannot be released again")
        payload = {
            "attempt_id": attempt_id,
            "task_id": task_id,
            "slurm": {
                "job_id": slurm_job_id,
                "array_job_id": slurm_array_job_id,
                "array_task_id": slurm_array_task_id,
            },
        }
        _validate_slurm(payload["slurm"])
        _append_event(ledger, "attempt_started", payload)
        _load_ledger_document_in_memory(ledger, document)
        _write_atomic(ledger_file, ledger)
    return ledger["events"][-1]


def _load_ledger_document_in_memory(
    ledger: dict[str, Any],
    recipe_document: dict[str, Any],
) -> dict[str, Any]:
    authorization = _validate_authorization(ledger["authorization"])
    if ledger["ledger_state_sha256"] != _ledger_digest(authorization, ledger["events"]):
        raise ReleaseLedgerError("mutated ledger state SHA-256 is invalid")
    tasks = {task["task_id"]: task for task in _recipe_tasks(recipe_document)}
    return _validate_event_log(ledger["events"], tasks)


def _validate_success_attestation(
    attestation: dict[str, Any],
    *,
    attestation_file_sha256: str,
    result_file_sha256: str,
    task: dict[str, Any],
    bundle: dict[str, Any],
    authorization: dict[str, Any],
) -> dict[str, str]:
    if not _is_sha256(attestation_file_sha256) or not _is_sha256(result_file_sha256):
        raise ReleaseLedgerError("result snapshot SHA-256 is invalid")
    _require_exact_keys(
        attestation,
        {
            "schema_version",
            "task_identity",
            "bundle_hashes",
            "release",
            "result_file_sha256",
        },
        "result attestation",
    )
    if attestation["schema_version"] != RESULT_ATTESTATION_SCHEMA_VERSION:
        raise ReleaseLedgerError("unsupported result attestation schema_version")
    if attestation["task_identity"] != task:
        raise ReleaseLedgerError("result attestation task identity is mismatched")
    expected_bundle_hashes = {key: bundle["hashes"][key] for key in _BUNDLE_HASH_KEYS}
    if attestation["bundle_hashes"] != expected_bundle_hashes:
        raise ReleaseLedgerError("result attestation recipe bundle hashes are mismatched")
    expected_release = {
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
    }
    if attestation["release"] != expected_release:
        raise ReleaseLedgerError("result attestation release identity is mismatched")
    if attestation["result_file_sha256"] != result_file_sha256:
        raise ReleaseLedgerError("result attestation does not match result bytes")
    return {
        "result_file_sha256": result_file_sha256,
        "result_attestation_sha256": attestation_file_sha256,
    }


def _task_configuration_payload(
    recipe_document: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, Any]:
    methods = recipe_document["payload"]["methods"]
    method = next(
        (item for item in methods if item["method_id"] == task["method_id"]),
        None,
    )
    if method is None:
        raise ReleaseLedgerError("result task method is absent from the recipe")
    configuration_id = f"D{task['dimension']}-seed{task['seed']}"
    records = method["configuration"]["by_dimension_and_seed"]
    matches = [record for record in records if record["configuration_id"] == configuration_id]
    if len(matches) != 1 or matches[0]["sha256"] != task["method_configuration_sha256"]:
        raise ReleaseLedgerError("result task configuration is absent or mismatched")
    return matches[0]["payload"]


def _validate_internal_result(
    result: dict[str, Any],
    *,
    task: dict[str, Any],
    bundle: dict[str, Any],
    config_payload: dict[str, Any],
    recipe_document: dict[str, Any],
    authorization: dict[str, Any],
) -> None:
    required = {
        "schema_version",
        "status",
        "task_config",
        "training",
        "evaluation",
        "corpus",
        "arms",
        "catalog",
        "provenance",
    }
    if not required.issubset(result):
        raise ReleaseLedgerError("internal result is missing release-critical fields")
    if result["schema_version"] != "f02_internal_task_v1" or result["status"] != "complete":
        raise ReleaseLedgerError("internal result is not a complete supported result")
    if result["task_config"] != config_payload:
        raise ReleaseLedgerError("internal result task configuration is mismatched")
    training = result["training"]
    evaluation = result["evaluation"]
    corpus = result["corpus"]
    catalog = result["catalog"]
    arms = result["arms"]
    provenance = result["provenance"]
    if any(
        not isinstance(value, dict)
        for value in (training, evaluation, corpus, catalog, arms, provenance)
    ):
        raise ReleaseLedgerError("internal result release-critical fields must be objects")
    if (
        training.get("optimizer_updates") != task["optimizer_updates"]
        or evaluation.get("split") != "test"
        or evaluation.get("design") != "primary"
        or corpus.get("replica") != task["replica"]
        or corpus.get("dimension") != task["dimension"]
        or catalog.get("sha256") != authorization["catalog_sha256"]
        or catalog.get("task_index") != task["catalog_task_index"]
    ):
        raise ReleaseLedgerError("internal result task/corpus/catalog identity is mismatched")
    _require_exact_keys(
        catalog,
        {"path", "sha256", "generation_git_commit", "generation_git_tree", "task_index"},
        "internal result catalog",
    )
    recipe_catalog = recipe_document["payload"]["artifacts"]["catalog"]
    if catalog != {
        "path": recipe_catalog["path"],
        "sha256": authorization["catalog_sha256"],
        "generation_git_commit": recipe_catalog["generation_git_commit"],
        "generation_git_tree": recipe_catalog["generation_git_tree"],
        "task_index": task["catalog_task_index"],
    }:
        raise ReleaseLedgerError("internal result canonical catalog provenance is mismatched")
    expected_arms = {
        "TERA-50",
        "ORBIT-50",
        f"ORBIT-{task['orbit_resource_m']}",
        "value-only-conditional-50",
    }
    if set(arms) != expected_arms:
        raise ReleaseLedgerError("internal result arm set is incomplete or unexpected")
    git = provenance.get("git")
    data = provenance.get("data")
    dependencies = provenance.get("dependencies")
    submodules = provenance.get("submodules")
    if (
        not isinstance(git, dict)
        or not isinstance(data, dict)
        or not isinstance(dependencies, dict)
        or not isinstance(submodules, dict)
    ):
        raise ReleaseLedgerError("internal result provenance is incomplete")
    _require_exact_keys(
        data,
        {
            "dataset_path",
            "metadata_path",
            "manifest_path",
            "file_sha256",
            "manifest_sha256",
            "dataset_content_sha256",
            "generator_config",
        },
        "internal result provenance.data",
    )
    expected_file_hashes = {
        bundle["dataset_filename"]: bundle["hashes"]["dataset_file_sha256"],
        bundle["metadata_filename"]: bundle["hashes"]["metadata_file_sha256"],
    }
    if any(
        not isinstance(data[field], str)
        for field in ("dataset_path", "metadata_path", "manifest_path")
    ):
        raise ReleaseLedgerError("internal result bundle paths must be strings")
    if (
        Path(data["dataset_path"]).name != bundle["dataset_filename"]
        or Path(data["metadata_path"]).name != bundle["metadata_filename"]
        or Path(data["manifest_path"]).name != bundle["manifest_filename"]
        or data["file_sha256"] != expected_file_hashes
        or data["manifest_sha256"] != bundle["hashes"]["sha256_manifest_file_sha256"]
        or data["dataset_content_sha256"] != bundle["hashes"]["dataset_content_sha256"]
        or data["generator_config"] != bundle["generator_config"]
    ):
        raise ReleaseLedgerError("internal result recipe bundle provenance is mismatched")
    expected_dependencies = {
        item["path"]: {"sha256": item["sha256"]}
        for item in recipe_document["payload"]["artifacts"]["dependencies"]
    }
    if (
        git.get("commit") != authorization["execution_commit"]
        or git.get("tree") != authorization["execution_tree"]
        or git.get("status_porcelain") != []
        or dependencies != expected_dependencies
        or submodules.get("tera_gitlink")
        != recipe_document["payload"]["artifacts"]["tera_submodule"]["commit"]
    ):
        raise ReleaseLedgerError("internal result release provenance is mismatched or dirty")


def _validate_external_result_for_release(
    result: dict[str, Any],
    *,
    task: dict[str, Any],
    bundle: dict[str, Any],
    config_payload: dict[str, Any],
    recipe_document: dict[str, Any],
    authorization: dict[str, Any],
) -> None:
    try:
        from experiments.f02_external_adapter import validate_external_result

        validate_external_result(result)
    except (ImportError, RuntimeError, TypeError, ValueError) as error:
        raise ReleaseLedgerError("external result failed its strict adapter validator") from error
    expected_identity_fields = (
        "method_id",
        "bundle_id",
        "catalog_task_index",
        "replica",
        "n_particles",
        "dimension",
        "seed",
        "optimizer_updates",
        "evaluation_split",
        "evaluation_design",
    )
    expected_identity = {field: task[field] for field in expected_identity_fields}
    if (
        result.get("task_identity") != expected_identity
        or result.get("config") != config_payload
        or result.get("config_sha256") != task["method_configuration_sha256"]
    ):
        raise ReleaseLedgerError("external result task/config identity is mismatched")
    sources = result.get("sources")
    provenance = result.get("provenance")
    if not isinstance(sources, dict) or not isinstance(provenance, dict):
        raise ReleaseLedgerError("external result release provenance is incomplete")
    protocol_sha256 = recipe_document["payload"]["artifacts"]["protocol"]["sha256"]
    method = next(
        item
        for item in recipe_document["payload"]["methods"]
        if item["method_id"] == task["method_id"]
    )
    evidence = {item["role"]: item["sha256"] for item in method["selection_evidence"]}
    if (
        sources.get("catalog_sha256") != authorization["catalog_sha256"]
        or sources.get("dataset_content_sha256") != bundle["hashes"]["dataset_content_sha256"]
        or provenance.get("repo_commit") != authorization["execution_commit"]
        or provenance.get("repo_tree") != authorization["execution_tree"]
        or provenance.get("vendor_commit")
        != recipe_document["payload"]["artifacts"]["tera_submodule"]["commit"]
        or provenance.get("f02_protocol_sha256") != protocol_sha256
        or provenance.get("dependency_lock_sha256") != evidence["runtime_dependency_lock"]
    ):
        raise ReleaseLedgerError("external result source/release provenance is mismatched")


def _validate_result_for_release(
    result: dict[str, Any],
    *,
    task: dict[str, Any],
    recipe_document: dict[str, Any],
    authorization: dict[str, Any],
) -> None:
    config_payload = _task_configuration_payload(recipe_document, task)
    bundle = _recipe_bundle_for_task(recipe_document, task)
    if task["method_id"] == "internal-shared-fit":
        _validate_internal_result(
            result,
            task=task,
            bundle=bundle,
            config_payload=config_payload,
            recipe_document=recipe_document,
            authorization=authorization,
        )
    else:
        _validate_external_result_for_release(
            result,
            task=task,
            bundle=bundle,
            config_payload=config_payload,
            recipe_document=recipe_document,
            authorization=authorization,
        )


def finish_attempt(
    recipe_path: str | Path,
    catalog_path: str | Path,
    ledger_path: str | Path,
    *,
    task_id: str,
    attempt_id: str,
    outcome: str,
    exit_code: int,
    failure_code: str | None = None,
    result_path: str | Path | None = None,
    attestation_path: str | Path | None = None,
    repo_root: str | Path = _RECIPE_ROOT,
) -> dict[str, Any]:
    """Reject v1 attempt completion before inspecting any result or ledger path."""

    _reject_release_mutation()
    ledger_file = Path(ledger_path)
    _require_external_ledger(ledger_file, Path(repo_root))
    with _ledger_lock(ledger_file):
        ledger, state, document = _load_for_mutation(
            ledger_file,
            recipe_path,
            catalog_path,
            repo_root=repo_root,
        )
        attempt = state["attempts"].get(attempt_id)
        if (
            attempt is None
            or attempt["finished"]
            or attempt["task_id"] != task_id
            or state["active_by_task"].get(task_id) != attempt_id
        ):
            raise ReleaseLedgerError("finish does not match one active attempt")
        task_map = {task["task_id"]: task for task in _recipe_tasks(document)}
        if outcome == "succeeded":
            if exit_code != 0 or failure_code is not None:
                raise ReleaseLedgerError(
                    "successful finish requires exit_code=0 and no failure_code"
                )
            if result_path is None or attestation_path is None:
                raise ReleaseLedgerError("successful finish requires result and attestation files")
            result_document, result_file_sha256 = _read_json_snapshot(
                result_path,
                label="F02 confirmatory result",
            )
            attestation_document, attestation_file_sha256 = _read_json_snapshot(
                attestation_path,
                label="F02 result attestation",
            )
            _validate_result_for_release(
                result_document,
                task=task_map[task_id],
                recipe_document=document,
                authorization=ledger["authorization"],
            )
            bundle = _recipe_bundle_for_task(document, task_map[task_id])
            result = _validate_success_attestation(
                attestation_document,
                attestation_file_sha256=attestation_file_sha256,
                result_file_sha256=result_file_sha256,
                task=task_map[task_id],
                bundle=bundle,
                authorization=ledger["authorization"],
            )
        elif outcome == "failed":
            if not failure_code:
                raise ReleaseLedgerError("failed finish requires a nonempty failure_code")
            if attestation_path is not None:
                raise ReleaseLedgerError("failed finish does not accept a success attestation")
            if result_path is None:
                result = None
            else:
                _, result_file_sha256 = _read_regular_file_snapshot(
                    result_path,
                    label="failed F02 result",
                )
                result = {"result_file_sha256": result_file_sha256}
        else:
            raise ReleaseLedgerError("outcome must be succeeded or failed")
        payload = {
            "attempt_id": attempt_id,
            "task_id": task_id,
            "outcome": outcome,
            "exit_code": exit_code,
            "failure_code": failure_code,
            "result": result,
        }
        _append_event(ledger, "attempt_finished", payload)
        _load_ledger_document_in_memory(ledger, document)
        _write_atomic(ledger_file, ledger)
    return ledger["events"][-1]


def seal_release(
    recipe_path: str | Path,
    catalog_path: str | Path,
    ledger_path: str | Path,
    *,
    repo_root: str | Path = _RECIPE_ROOT,
) -> dict[str, Any]:
    """Reject v1 sealing before inspecting or mutating any supplied path."""

    _reject_release_mutation()
    ledger_file = Path(ledger_path)
    _require_external_ledger(ledger_file, Path(repo_root))
    with _ledger_lock(ledger_file):
        ledger, state, document = _load_for_mutation(
            ledger_file,
            recipe_path,
            catalog_path,
            repo_root=repo_root,
        )
        if state["sealed"]:
            raise ReleaseLedgerError("release is already sealed")
        if state["active_attempt_count"] or state["remaining_task_count"]:
            raise ReleaseLedgerError("cannot seal a partial or active confirmatory task grid")
        ordered_ids = [task["task_id"] for task in _recipe_tasks(document)]
        payload = {
            "completed_task_count": len(ordered_ids),
            "completed_task_ids_sha256": sha256_bytes(canonical_json_bytes(ordered_ids)),
        }
        _append_event(ledger, "release_sealed", payload)
        _load_ledger_document_in_memory(ledger, document)
        _write_atomic(ledger_file, ledger)
    return ledger["events"][-1]


def audit_release(
    recipe_path: str | Path,
    catalog_path: str | Path,
    ledger_path: str | Path,
    *,
    repo_root: str | Path = _RECIPE_ROOT,
) -> dict[str, Any]:
    """Inspect deterministic accounting in a manually supplied scaffold fixture."""

    ledger_file = Path(ledger_path)
    _require_external_ledger(ledger_file, Path(repo_root))
    ledger, state, _ = _load_for_mutation(
        ledger_file,
        recipe_path,
        catalog_path,
        repo_root=repo_root,
    )
    return {
        "structurally_consistent": True,
        "releasable": False,
        "release_scaffold_status": RECIPE_SCAFFOLD_STATUS,
        "release_mutations_enabled": RELEASE_MUTATIONS_ENABLED,
        "development_evidence_semantically_verified": (DEVELOPMENT_EVIDENCE_SEMANTICALLY_VERIFIED),
        "schema_version": LEDGER_SCHEMA_VERSION,
        "release_id": ledger["authorization"]["release_id"],
        "experiment_id": ledger["authorization"]["experiment_id"],
        "protocol_id": ledger["authorization"]["protocol_id"],
        "protocol_path": ledger["authorization"]["protocol_path"],
        "protocol_sha256": ledger["authorization"]["protocol_sha256"],
        "protocol_binding_sha256": ledger["authorization"]["protocol_binding_sha256"],
        "execution_commit": ledger["authorization"]["execution_commit"],
        "catalog_sha256": ledger["authorization"]["catalog_sha256"],
        "event_count": len(ledger["events"]),
        "attempt_count": state["attempt_count"],
        "failed_attempt_count": state["failed_attempt_count"],
        "active_attempt_count": state["active_attempt_count"],
        "successful_task_count": state["successful_task_count"],
        "remaining_task_count": state["remaining_task_count"],
        "sealed": state["sealed"],
        "ledger_state_sha256": ledger["ledger_state_sha256"],
        "confirmatory_execution_enabled_by_this_module": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--recipe", type=Path, required=True)
        command.add_argument("--catalog", type=Path, required=True)
        command.add_argument("--ledger", type=Path, required=True)
        command.add_argument("--repo-root", type=Path, default=_RECIPE_ROOT)

    authorize = subparsers.add_parser("authorize")
    common(authorize)

    begin = subparsers.add_parser("begin-attempt")
    common(begin)
    begin.add_argument("--task-id", required=True)
    begin.add_argument("--attempt-id", required=True)
    begin.add_argument("--slurm-job-id", required=True)
    begin.add_argument("--slurm-array-job-id", required=True)
    begin.add_argument("--slurm-array-task-id", type=int, required=True)

    finish = subparsers.add_parser("finish-attempt")
    common(finish)
    finish.add_argument("--task-id", required=True)
    finish.add_argument("--attempt-id", required=True)
    finish.add_argument("--outcome", choices=("succeeded", "failed"), required=True)
    finish.add_argument("--exit-code", type=int, required=True)
    finish.add_argument("--failure-code")
    finish.add_argument("--result", type=Path)
    finish.add_argument("--attestation", type=Path)

    seal = subparsers.add_parser("seal")
    common(seal)
    audit = subparsers.add_parser("audit")
    common(audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    common = {
        "recipe_path": args.recipe,
        "catalog_path": args.catalog,
        "ledger_path": args.ledger,
        "repo_root": args.repo_root,
    }
    try:
        if args.command != "audit":
            _reject_release_mutation()
        output = audit_release(**common)
    except ReleaseLedgerError as error:
        parser.error(str(error))
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LEDGER_SCHEMA_VERSION",
    "RELEASE_MUTATIONS_ENABLED",
    "RELEASE_MUTATION_DISABLED_REASON",
    "CATALOG_RELEASE_MARKER_SCHEMA_VERSION",
    "RESULT_ATTESTATION_SCHEMA_VERSION",
    "ReleaseLedgerError",
    "audit_release",
    "authorize_release",
    "begin_attempt",
    "catalog_ledger_path",
    "catalog_release_marker_path",
    "finish_attempt",
    "seal_release",
    "verify_recipe_only_release",
]
