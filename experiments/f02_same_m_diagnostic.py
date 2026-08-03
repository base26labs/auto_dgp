"""Development-only diagnosis for a failed F02 ORBIT/TERA same-m control.

The diagnostic deliberately exposes no test-split option.  It repeats the
registered optimizer-selection fit, records the local singular spectra, and
compares float32, tighter-CG float32, and prediction-only float64 paths.  It is
not an experimental result or a mechanism for changing the frozen gate.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

import torch

from data.load_nbody_confirmatory import load_prepared_confirmatory_bundle
from experiments.f02_internal_models import (
    FrozenTERAParameters,
    ScalarPrediction,
    fit_released_tera,
    freeze_tera_parameters,
    predict_orbit,
    predict_released_tera,
    prepared_split_to_tensors,
)
from experiments.f02_internal_task import (
    InternalTaskConfig,
    _authorize_evaluation_phase,
    _bundle_identity,
    _fit_kwargs,
    _preflight_bundle_identity,
    _selected_bundle,
    save_internal_result,
    validate_catalog_identity,
)

DIAGNOSTIC_SCHEMA_VERSION = "f02_same_m_diagnostic_v1"


def _comparison(reference: ScalarPrediction, candidate: ScalarPrediction) -> dict[str, Any]:
    mean_error = torch.abs(reference.mean - candidate.mean)
    variance_error = torch.abs(reference.latent_variance - candidate.latent_variance)
    return {
        "maxabs_mean": float(mean_error.max().detach().cpu()),
        "maxabs_latent_variance": float(variance_error.max().detach().cpu()),
        "per_target_abs_mean": mean_error.detach().cpu().tolist(),
        "per_target_abs_latent_variance": variance_error.detach().cpu().tolist(),
    }


def _cast_prediction(prediction: ScalarPrediction, dtype: torch.dtype) -> ScalarPrediction:
    return ScalarPrediction(
        mean=prediction.mean.to(dtype=dtype),
        latent_variance=prediction.latent_variance.to(dtype=dtype),
        observation_variance=prediction.observation_variance.to(dtype=dtype),
    )


def _scalar_scores(target: torch.Tensor, prediction: ScalarPrediction) -> dict[str, float]:
    error = prediction.mean - target
    variance = prediction.latent_variance
    return {
        "rmse": float(torch.sqrt(torch.mean(error * error)).detach().cpu()),
        "latent_gaussian_nll": float(
            torch.mean(0.5 * (torch.log(2.0 * torch.pi * variance) + error * error / variance))
            .detach()
            .cpu()
        ),
    }


def _orbit_solver(prediction: ScalarPrediction) -> dict[str, Any]:
    if prediction.details is None:
        raise RuntimeError("ORBIT diagnostic prediction lacks solver details")
    details = prediction.details
    return {
        "rank": details.ranks.detach().cpu().tolist(),
        "iterations": details.iterations.detach().cpu().tolist(),
        "operator_matvecs": details.operator_matvecs.detach().cpu().tolist(),
        "preconditioner_applications": (
            details.preconditioner_applications.detach().cpu().tolist()
        ),
        "fresh_relative_residual": details.relative_residuals.detach().cpu().tolist(),
        "converged": details.converged.detach().cpu().tolist(),
        "basis_exact": details.basis_exact.detach().cpu().tolist(),
    }


def _singular_spectra(
    x_train: torch.Tensor,
    x_eval: torch.Tensor,
    parameters: FrozenTERAParameters,
    *,
    m: int,
) -> list[dict[str, Any]]:
    lengthscale = parameters.lengthscale.to(device=x_train.device, dtype=x_train.dtype)
    if lengthscale.numel() == 1:
        train_scaled = x_train / lengthscale.reshape(1, 1)
        eval_scaled = x_eval / lengthscale.reshape(1, 1)
    else:
        train_scaled = x_train / lengthscale.reshape(1, -1)
        eval_scaled = x_eval / lengthscale.reshape(1, -1)
    neighbours = torch.topk(
        torch.cdist(eval_scaled, train_scaled),
        k=min(m, x_train.shape[0]),
        largest=False,
    ).indices
    records: list[dict[str, Any]] = []
    for target, indices in zip(x_eval, neighbours, strict=True):
        differences = (x_train[indices] - target.unsqueeze(0)).T
        if lengthscale.numel() == 1:
            differences = differences / lengthscale.reshape(1, 1)
        else:
            differences = differences / lengthscale.reshape(-1, 1)
        singular_values = torch.linalg.svdvals(differences)
        threshold = singular_values[0] * max(differences.shape) * torch.finfo(differences.dtype).eps
        positive = singular_values > 0.0
        condition_number = math.inf
        if bool(positive.any().item()):
            smallest_positive = singular_values[positive][-1]
            condition_number = float((singular_values[0] / smallest_positive).detach().cpu())
        records.append(
            {
                "singular_values": singular_values.detach().cpu().tolist(),
                "current_numerical_threshold": float(threshold.detach().cpu()),
                "current_retained_rank": int((singular_values > threshold).sum().item()),
                "algebraic_maximum_rank": min(differences.shape),
                "condition_number_over_positive_values": condition_number,
            }
        )
    return records


def _prediction_set(
    train: Any,
    evaluation: Any,
    parameters: FrozenTERAParameters,
    *,
    m: int,
    cg_tolerance: float,
) -> tuple[ScalarPrediction, ScalarPrediction, list[float]]:
    # The released helper does not expose which adaptive q-coordinate jitter
    # succeeded.  This diagnostic-only drop-in repeats its six-line algorithm
    # verbatim and restores the module global immediately after prediction.
    import md22_regression.models.tera as released_tera_module

    selected_jitters: list[float] = []
    original_cholesky = released_tera_module.cholesky_with_jitter

    def recording_cholesky(
        matrix: torch.Tensor,
        jitter0: float = 1e-8,
        jitter_max: float = 1e-1,
    ) -> torch.Tensor:
        jitter = jitter0
        identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
        while True:
            try:
                factor = torch.linalg.cholesky(matrix + jitter * identity)
                selected_jitters.append(float(jitter))
                return factor
            except Exception:
                jitter *= 10.0
                if jitter > jitter_max:
                    raise

    released_tera_module.cholesky_with_jitter = recording_cholesky
    try:
        tera = predict_released_tera(train, evaluation.X, parameters, m=m)
    finally:
        released_tera_module.cholesky_with_jitter = original_cholesky
    orbit = predict_orbit(
        train,
        evaluation.X,
        parameters,
        m=m,
        cg_tolerance=cg_tolerance,
        use_preconditioner=True,
        function_jitter=1e-8,
        reduced_jitter=1e-8,
    )
    if len(selected_jitters) != evaluation.X.shape[0]:
        raise RuntimeError("released reduced Cholesky jitter trace has the wrong row count")
    return tera, orbit, selected_jitters


def run_diagnostic(
    dataset_path: str | Path,
    *,
    catalog_path: str | Path,
    output_path: str | Path,
    train_steps: int = 20,
    seed: int = 11,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run the registered development-only same-m diagnostic."""

    config = InternalTaskConfig(
        train_steps=train_steps,
        seed=seed,
        candidate_m=(),
        cg_tolerance=1e-5,
        dtype="float32",
        device=device,
    )
    preflight_identity = _preflight_bundle_identity(dataset_path)
    authorization = validate_catalog_identity(catalog_path, preflight_identity)
    _authorize_evaluation_phase(authorization, "validation")
    bundle = load_prepared_confirmatory_bundle(dataset_path)
    if _bundle_identity(bundle) != preflight_identity:
        raise RuntimeError("bundle identity changed after preflight")
    selected = _selected_bundle(bundle, "validation", "optimizer_selection")
    train32 = prepared_split_to_tensors(selected, "train", dtype=torch.float32, device=device)
    evaluation32 = prepared_split_to_tensors(
        selected,
        "validation",
        dtype=torch.float32,
        device=device,
    )
    model = fit_released_tera(train32, **_fit_kwargs(config))
    parameters32 = freeze_tera_parameters(model)

    tera32, orbit32, tera32_jitters = _prediction_set(
        train32,
        evaluation32,
        parameters32,
        m=50,
        cg_tolerance=1e-5,
    )
    orbit32_tight = predict_orbit(
        train32,
        evaluation32.X,
        parameters32,
        m=50,
        cg_tolerance=1e-8,
        use_preconditioner=True,
        function_jitter=1e-8,
        reduced_jitter=1e-8,
    )

    train64 = prepared_split_to_tensors(selected, "train", dtype=torch.float64, device=device)
    evaluation64 = prepared_split_to_tensors(
        selected,
        "validation",
        dtype=torch.float64,
        device=device,
    )
    parameters64 = replace(parameters32, lengthscale=parameters32.lengthscale.double())
    tera64, orbit64, tera64_jitters = _prediction_set(
        train64,
        evaluation64,
        parameters64,
        m=50,
        cg_tolerance=1e-10,
    )

    result = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "scope": "development-validation optimizer-selection diagnostic only",
        "task": {
            "replica": bundle.loaded.dataset.config.replica,
            "dimension": bundle.loaded.dataset.config.state_dim,
            "train_steps": train_steps,
            "seed": seed,
            "m": 50,
        },
        "learned_parameters": {
            "lengthscale": parameters32.lengthscale.detach().cpu().tolist(),
            "outputscale": parameters32.outputscale,
            "sigma_f_variance": parameters32.sigma_f,
            "sigma_g_variance": parameters32.sigma_g,
        },
        "float32_default": {
            "comparison": _comparison(tera32, orbit32),
            "released_tera_selected_q_coordinate_jitter": tera32_jitters,
            "scores": {
                "TERA-50": _scalar_scores(evaluation32.value, tera32),
                "ORBIT-50": _scalar_scores(evaluation32.value, orbit32),
            },
            "solver": _orbit_solver(orbit32),
        },
        "float32_tighter_cg": {
            "comparison_to_float32_tera": _comparison(tera32, orbit32_tight),
            "solver": _orbit_solver(orbit32_tight),
        },
        "float64_prediction_only": {
            "comparison": _comparison(tera64, orbit64),
            "released_tera_selected_q_coordinate_jitter": tera64_jitters,
            "scores": {
                "TERA-50": _scalar_scores(evaluation64.value, tera64),
                "ORBIT-50": _scalar_scores(evaluation64.value, orbit64),
            },
            "solver": _orbit_solver(orbit64),
            "tera_float32_to_float64": _comparison(_cast_prediction(tera64, torch.float32), tera32),
            "orbit_float32_to_float64": _comparison(
                _cast_prediction(orbit64, torch.float32),
                orbit32,
            ),
            "tera_float32_to_orbit_float64": _comparison(
                _cast_prediction(orbit64, torch.float32),
                tera32,
            ),
        },
        "float32_local_singular_spectra": _singular_spectra(
            train32.X,
            evaluation32.X,
            parameters32,
            m=50,
        ),
        "float64_local_singular_spectra": _singular_spectra(
            train64.X,
            evaluation64.X,
            parameters64,
            m=50,
        ),
        "catalog": {
            "sha256": authorization.catalog_sha256,
            "task_index": authorization.bundle_entry.get("task_index"),
        },
        "registered_config": asdict(config),
    }
    save_internal_result(result, output_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, choices=(20, 50, 100), default=20)
    parser.add_argument("--seed", type=int, choices=(11, 29, 47), default=11)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_diagnostic(
        args.dataset,
        catalog_path=args.catalog,
        output_path=args.out,
        train_steps=args.train_steps,
        seed=args.seed,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
