"""Cluster launcher for F01 with runtime and exact simulated-data provenance.

This wrapper deliberately delegates all experiment logic and argument validation
to :mod:`experiments.f01_orbit_gp_sim`.  Its only responsibilities are to hash
the exact in-memory datasets consumed by F01, capture a non-secret runtime
manifest, and write outputs atomically into an array-task-specific directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import experiments.f01_orbit_gp_sim as f01

_TENSOR_FIELDS = (
    "X_train",
    "X_train_scaled",
    "X_eval",
    "X_eval_scaled",
    "lengthscale",
    "f_train_obs",
    "g_train_obs",
    "z_train_obs",
)
_SELECTED_ENVIRONMENT = (
    "SLURM_JOB_ID",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_JOB_NODELIST",
    "SLURM_CPUS_PER_TASK",
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PYTHONHASHSEED",
    "F01_SLURM_EXCLUSIVE_VERIFIED",
    "F01_SLURM_EXCLUSIVE_MODE",
)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tensor_fingerprint(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().to(device="cpu").contiguous()
    raw = value.numpy().tobytes(order="C")
    return {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "nbytes": len(raw),
        "sha256": _sha256_bytes(raw),
    }


def _dataset_fingerprint(data: Any, *, d: int, repeat: int) -> dict[str, Any]:
    tensors = [_tensor_fingerprint(name, getattr(data, name)) for name in _TENSOR_FIELDS]
    metadata_payload = {
        "d": d,
        "repeat": repeat,
        "kernel_name": data.kernel_name,
        "outputscale": float(data.outputscale),
        "sigma_f": float(data.sigma_f),
        "sigma_g": float(data.sigma_g),
        "sampling_backend": data.sampling_backend,
        "tensors": tensors,
    }
    canonical = json.dumps(metadata_payload, sort_keys=True, separators=(",", ":")).encode()
    return {**metadata_payload, "combined_sha256": _sha256_bytes(canonical)}


def _package_manifest() -> list[dict[str, str]]:
    packages: dict[str, tuple[str, str]] = {}
    for distribution in metadata.distributions():
        package_name = distribution.metadata.get("Name")
        if package_name:
            packages[package_name.lower()] = (package_name, distribution.version)
    return [
        {"name": original_name, "version": version}
        for _, (original_name, version) in sorted(packages.items())
    ]


def _runtime_manifest() -> dict[str, Any]:
    packages = _package_manifest()
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    return {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "node": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_devices": cuda_devices,
        },
        "selected_environment": {
            name: os.environ[name] for name in _SELECTED_ENVIRONMENT if name in os.environ
        },
        "packages": packages,
        "packages_sha256": _sha256_bytes(
            json.dumps(packages, sort_keys=True, separators=(",", ":")).encode()
        ),
    }


def _parse_arguments() -> tuple[argparse.Namespace, argparse.Namespace]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    launcher_args, experiment_argv = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *experiment_argv]
        experiment_args = f01.parse_args()
    finally:
        sys.argv = original_argv
    return launcher_args, experiment_args


def main() -> None:
    launcher_args, experiment_args = _parse_arguments()
    if os.environ.get("F01_REQUIRE_SLURM") == "1" and "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("F01_REQUIRE_SLURM=1 but no Slurm job environment is present")
    if experiment_args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("F01 requested CUDA, but PyTorch reports that CUDA is unavailable")

    output = Path(experiment_args.out)
    artifact_paths = (
        launcher_args.dataset_manifest.resolve(),
        launcher_args.runtime_manifest.resolve(),
        output.resolve(),
    )
    if len(set(artifact_paths)) != len(artifact_paths):
        raise RuntimeError("dataset, runtime, and result paths must be distinct")

    runtime_manifest = _runtime_manifest()
    _write_json_atomic(launcher_args.runtime_manifest, runtime_manifest)

    fingerprints: list[dict[str, Any]] = []
    original_simulate_dataset = f01.simulate_dataset

    def fingerprinting_simulate_dataset(cfg: Any, d: int, *, repeat: int):
        data = original_simulate_dataset(cfg, d=d, repeat=repeat)
        fingerprints.append(_dataset_fingerprint(data, d=d, repeat=repeat))
        return data

    f01.simulate_dataset = fingerprinting_simulate_dataset
    try:
        result = f01.run(experiment_args)
    finally:
        f01.simulate_dataset = original_simulate_dataset

    if len(fingerprints) != experiment_args.repeats:
        raise RuntimeError(
            f"captured {len(fingerprints)} datasets for {experiment_args.repeats} repeats"
        )

    dataset_manifest = {
        "schema_version": 1,
        "algorithm": "sha256",
        "definition": (
            "Hash of tensor dtype, shape, and raw CPU-contiguous bytes for the exact "
            "SimulatedDataset instances consumed by F01"
        ),
        "base_seed": experiment_args.seed,
        "datasets": fingerprints,
    }
    _write_json_atomic(launcher_args.dataset_manifest, dataset_manifest)

    result["cluster_provenance"] = {
        "dataset_manifest": str(launcher_args.dataset_manifest),
        "dataset_manifest_sha256": _sha256_bytes(launcher_args.dataset_manifest.read_bytes()),
        "runtime_manifest": str(launcher_args.runtime_manifest),
        "runtime_manifest_sha256": _sha256_bytes(launcher_args.runtime_manifest.read_bytes()),
        "packages_sha256": runtime_manifest["packages_sha256"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "exclusive_node_verified": os.environ.get("F01_SLURM_EXCLUSIVE_VERIFIED") == "1",
        "exclusive_node_verification_mode": os.environ.get("F01_SLURM_EXCLUSIVE_MODE"),
        "wall_time_is_inferential": False,
    }

    _write_json_atomic(output, result)
    for record in result["rows"]:
        print(json.dumps(record), flush=True)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
