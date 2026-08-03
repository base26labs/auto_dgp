"""Development-only diagnosis for a failed F02 ORBIT/TERA same-m control.

The diagnostic deliberately exposes no test-split option.  It repeats the
registered optimizer-selection fit, records the local singular spectra, and
compares float32, tighter-CG float32, and prediction-only float64 paths.  It is
not an experimental result or a mechanism for changing the frozen gate.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
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
    TensorConfirmatorySplit,
    build_released_tera_predictor,
    fit_released_tera,
    freeze_tera_parameters,
    predict_orbit,
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
from experiments.f02_support_dense_oracle import (
    RANK_RULE_NAME,
    SOURCE_DTYPE,
    predict_local_dense_support,
)
from gp.orbit import build_local_geometry_from_differences

DIAGNOSTIC_SCHEMA_VERSION = "f02_same_m_diagnostic_v3"


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


def _float32_scalar_to_float64(value: float, *, device: torch.device) -> float:
    return float(torch.as_tensor(value, dtype=torch.float32, device=device).to(torch.float64))


def _quantized_split_to_float64(
    split: TensorConfirmatorySplit,
) -> TensorConfirmatorySplit:
    """Promote an already quantized float32 split without revisiting raw arrays."""

    if split.X.dtype != torch.float32:
        raise TypeError("the source split must already be quantized to float32")
    return replace(
        split,
        X=split.X.to(dtype=torch.float64),
        value=split.value.to(dtype=torch.float64),
        gradient=split.gradient.to(dtype=torch.float64),
        time_value=split.time_value.to(dtype=torch.float64),
    )


def _quantized_parameters_to_float64(
    parameters: FrozenTERAParameters,
) -> FrozenTERAParameters:
    """Promote the exact learned float32 parameter values to float64."""

    if parameters.lengthscale.dtype != torch.float32:
        raise TypeError("the source parameters must contain a float32 lengthscale")
    device = parameters.lengthscale.device
    return replace(
        parameters,
        lengthscale=parameters.lengthscale.to(dtype=torch.float64),
        outputscale=_float32_scalar_to_float64(parameters.outputscale, device=device),
        sigma_f=_float32_scalar_to_float64(parameters.sigma_f, device=device),
        sigma_g=_float32_scalar_to_float64(parameters.sigma_g, device=device),
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


def _fixed_rank_geometry_records(
    train: TensorConfirmatorySplit,
    evaluation: TensorConfirmatorySplit,
    parameters: FrozenTERAParameters,
    *,
    m: int,
    neighbour_indices: torch.Tensor,
    rank_epsilon: float,
) -> list[dict[str, Any]]:
    """Record the actual ORBIT geometry under the fixed operational cutoff."""

    neighbours = _validate_fixed_neighbour_indices(
        train,
        evaluation,
        neighbour_indices,
        m=m,
    )
    lengthscale = parameters.lengthscale.to(device=train.X.device, dtype=train.X.dtype)
    records: list[dict[str, Any]] = []
    for target_index, (target, indices) in enumerate(zip(evaluation.X, neighbours, strict=True)):
        differences = (train.X[indices] - target.unsqueeze(0)).T
        if lengthscale.numel() == 1:
            differences = differences / lengthscale.reshape(1, 1)
        else:
            differences = differences / lengthscale.reshape(-1, 1)
        singular_values = torch.linalg.svdvals(differences)
        largest = singular_values[0]
        operational_threshold = largest * max(differences.shape) * rank_epsilon
        native_threshold = largest * max(differences.shape) * torch.finfo(differences.dtype).eps
        geometry = build_local_geometry_from_differences(
            differences,
            rank_epsilon=rank_epsilon,
        )
        projector = geometry.coordinates @ geometry.q_to_z.T
        rank = geometry.rank
        records.append(
            {
                "target_position": target_index,
                "target_source_index": int(evaluation.source_indices[target_index].item()),
                "neighbour_source_indices": (train.source_indices[indices].detach().cpu().tolist()),
                "singular_values": singular_values.detach().cpu().tolist(),
                "operational_rank_epsilon": rank_epsilon,
                "operational_rank_threshold": float(operational_threshold.detach().cpu()),
                "operational_retained_rank": rank,
                "native_compute_rank_threshold": float(native_threshold.detach().cpu()),
                "native_compute_retained_rank": int(
                    (singular_values > native_threshold).sum().item()
                ),
                "discarded_singular_value_energy": float(
                    geometry.discarded_eigenvalue_sum.detach().cpu()
                ),
                "discarded_modes_are_unresolvable_at_native_cutoff": geometry.is_exact,
                "q_coordinate_support_projector": projector.detach().cpu().tolist(),
            }
        )
    return records


def _projector_comparison(
    reference_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(reference_records) != len(candidate_records):
        raise ValueError("projector record sets must have the same length")
    errors: list[float] = []
    for reference, candidate in zip(reference_records, candidate_records, strict=True):
        reference_projector = torch.as_tensor(
            reference["q_coordinate_support_projector"],
            dtype=torch.float64,
        )
        candidate_projector = torch.as_tensor(
            candidate["q_coordinate_support_projector"],
            dtype=torch.float64,
        )
        if reference_projector.shape != candidate_projector.shape:
            errors.append(math.inf)
        else:
            errors.append(float(torch.max(torch.abs(reference_projector - candidate_projector))))
    return {
        "per_target_maxabs": errors,
        "maxabs": max(errors, default=0.0),
    }


def _neighbour_indices(
    x_train: torch.Tensor,
    x_eval: torch.Tensor,
    lengthscale: torch.Tensor,
    *,
    m: int,
) -> torch.Tensor:
    """Use the pinned released TERA implementation's exact KNN routine."""

    if m <= 0:
        raise ValueError("m must be positive")

    # Importing gp.tera installs the pinned vendor source on sys.path.  Calling
    # its own routine avoids a diagnostic copy drifting from the released
    # implementation whose neighbours are being frozen.
    importlib.import_module("gp.tera")
    from gp_sim_kl.ordering import knn_to_eval

    lengthscale = lengthscale.to(device=x_train.device, dtype=x_train.dtype)
    if lengthscale.numel() == 1:
        train_scaled = x_train / lengthscale.reshape(1, 1)
        eval_scaled = x_eval / lengthscale.reshape(1, 1)
    else:
        train_scaled = x_train / lengthscale.reshape(1, -1)
        eval_scaled = x_eval / lengthscale.reshape(1, -1)
    rows = knn_to_eval(train_scaled, eval_scaled, m)
    if not rows:
        return torch.empty(
            (0, min(m, x_train.shape[0])),
            dtype=torch.long,
            device=x_train.device,
        )
    return torch.stack(rows, dim=0)


def _validate_fixed_neighbour_indices(
    train: TensorConfirmatorySplit,
    evaluation: TensorConfirmatorySplit,
    neighbour_indices: torch.Tensor,
    *,
    m: int,
) -> torch.Tensor:
    """Reject any fixed-neighbour payload that could change comparison scope."""

    if not isinstance(neighbour_indices, torch.Tensor):
        raise TypeError("neighbour_indices must be a torch tensor")
    if neighbour_indices.dtype != torch.long:
        raise TypeError("neighbour_indices must have dtype torch.long")
    if neighbour_indices.device != train.X.device:
        raise ValueError("neighbour_indices must be on the training-data device")
    expected_shape = (evaluation.X.shape[0], min(m, train.X.shape[0]))
    if neighbour_indices.shape != expected_shape:
        raise ValueError(f"neighbour_indices must have shape {expected_shape}")
    if neighbour_indices.numel() > 0:
        if bool((neighbour_indices < 0).any().item()) or bool(
            (neighbour_indices >= train.X.shape[0]).any().item()
        ):
            raise ValueError("neighbour_indices contains an out-of-range row")
        if neighbour_indices.shape[1] > 1:
            ordered = torch.sort(neighbour_indices, dim=1).values
            if bool((ordered[:, 1:] == ordered[:, :-1]).any().item()):
                raise ValueError("each neighbour row must contain unique training rows")
    return neighbour_indices


def _cross_dtype_neighbour_identity(
    train32: TensorConfirmatorySplit,
    evaluation32: TensorConfirmatorySplit,
    parameters32: FrozenTERAParameters,
    train64: TensorConfirmatorySplit,
    evaluation64: TensorConfirmatorySplit,
    parameters64: FrozenTERAParameters,
    *,
    m: int,
) -> dict[str, Any]:
    """Record whether dtype-specific top-k calls select the same source rows."""

    neighbours32 = _neighbour_indices(
        train32.X,
        evaluation32.X,
        parameters32.lengthscale,
        m=m,
    )
    neighbours64 = _neighbour_indices(
        train64.X,
        evaluation64.X,
        parameters64.lengthscale,
        m=m,
    )
    sources32 = train32.source_indices[neighbours32].detach().cpu()
    sources64 = train64.source_indices[neighbours64].detach().cpu()
    same_order = torch.all(sources32 == sources64, dim=1)
    same_set = torch.all(
        torch.sort(sources32, dim=1).values == torch.sort(sources64, dim=1).values,
        dim=1,
    )
    return {
        "selection_rule": "pinned vendor knn_to_eval on scaled Euclidean distance",
        "canonical_selection_dtype": "float32",
        "canonical_neighbour_source_indices": sources32.tolist(),
        "native_float32_neighbour_source_indices": sources32.tolist(),
        "native_source_quantized_float64_neighbour_source_indices": sources64.tolist(),
        "native_recomputation_per_target_same_order": same_order.tolist(),
        "native_recomputation_all_targets_same_order": bool(same_order.all().item()),
        "native_recomputation_per_target_same_set": same_set.tolist(),
        "native_recomputation_all_targets_same_set": bool(same_set.all().item()),
        "fixed_comparisons_use_canonical_float32_indices": True,
    }


def _support64_prediction_set(
    train: TensorConfirmatorySplit,
    evaluation: TensorConfirmatorySplit,
    parameters: FrozenTERAParameters,
    *,
    m: int,
    neighbour_indices: torch.Tensor,
    function_jitters: list[float] | tuple[float, ...] | None = None,
    support_coordinate_jitters: list[float] | tuple[float, ...] | None = None,
) -> tuple[ScalarPrediction, list[dict[str, Any]]]:
    """Run the dense oracle on one caller-frozen neighbourhood per target."""

    if train.X.dtype != torch.float64 or evaluation.X.dtype != torch.float64:
        raise TypeError("support64 requires float32-quantized tensors promoted to float64")
    if parameters.lengthscale.dtype != torch.float64:
        raise TypeError("support64 parameters must be promoted to float64")
    neighbours = _validate_fixed_neighbour_indices(
        train,
        evaluation,
        neighbour_indices,
        m=m,
    )
    if function_jitters is None:
        f_jitters = [1e-8] * evaluation.X.shape[0]
    else:
        f_jitters = list(function_jitters)
        if len(f_jitters) != evaluation.X.shape[0]:
            raise ValueError("function_jitters must have one entry per target")
    if support_coordinate_jitters is None:
        q_jitters = [1e-8] * evaluation.X.shape[0]
    else:
        q_jitters = list(support_coordinate_jitters)
        if len(q_jitters) != evaluation.X.shape[0]:
            raise ValueError("support_coordinate_jitters must have one entry per target")
    means: list[torch.Tensor] = []
    variances: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    rows = zip(evaluation.X, neighbours, f_jitters, q_jitters, strict=True)
    for target_index, (target, indices, f_jitter, q_jitter) in enumerate(rows):
        prediction = predict_local_dense_support(
            train.X[indices],
            train.value[indices],
            train.gradient[indices],
            target.unsqueeze(0),
            lengthscale=parameters.lengthscale,
            outputscale=parameters.outputscale,
            value_noise_variance=parameters.sigma_f,
            gradient_noise_variance=parameters.sigma_g,
            kernel=parameters.kernel,
            gradient_noise_model=parameters.gradient_noise_model,
            function_jitter=f_jitter,
            support_coordinate_jitter=q_jitter,
        )
        means.append(prediction.mean)
        variances.append(prediction.latent_variance)
        records.append(
            {
                "target_position": target_index,
                "target_source_index": int(evaluation.source_indices[target_index].item()),
                "neighbour_source_indices": (train.source_indices[indices].detach().cpu().tolist()),
                "mean": float(prediction.mean.detach().cpu()),
                "latent_variance": float(prediction.latent_variance.detach().cpu()),
                "value_only_conditional_variance": float(
                    prediction.value_only_conditional_variance.detach().cpu()
                ),
                "gradient_variance_reduction": float(
                    prediction.gradient_variance_reduction.detach().cpu()
                ),
                "ambient_scaled_difference_support_projector": (
                    prediction.support_basis @ prediction.support_basis.T
                )
                .detach()
                .cpu()
                .tolist(),
                "q_coordinate_support_projector": (
                    prediction.support_coordinates @ prediction.tera_to_support.T
                )
                .detach()
                .cpu()
                .tolist(),
                "diagnostics": asdict(prediction.diagnostics),
            }
        )
    mean = torch.stack(means)
    latent_variance = torch.stack(variances)
    return (
        ScalarPrediction(
            mean=mean,
            latent_variance=latent_variance,
            observation_variance=latent_variance + latent_variance.new_tensor(parameters.sigma_f),
        ),
        records,
    )


def _released_tera_fixed_prediction_set(
    train: TensorConfirmatorySplit,
    evaluation: TensorConfirmatorySplit,
    parameters: FrozenTERAParameters,
    *,
    m: int,
    neighbour_indices: torch.Tensor,
) -> tuple[ScalarPrediction, list[float], list[float]]:
    """Call the pinned released one-target path on caller-frozen row indices."""

    neighbours = _validate_fixed_neighbour_indices(
        train,
        evaluation,
        neighbour_indices,
        m=m,
    )
    predictor = build_released_tera_predictor(train, parameters, m=m)
    predict_one = getattr(predictor, "_predict_one", None)
    if predict_one is None or tuple(inspect.signature(predict_one).parameters) != (
        "x_eval",
        "x_eval_scaled",
        "idx",
    ):
        raise RuntimeError("pinned released TERA _predict_one API changed")

    import gp_sim_kl.models.common as released_common_module
    import md22_regression.models.tera as released_tera_module
    from gp_sim_kl.utils import scale_inputs

    original_reduced_cholesky = released_tera_module.cholesky_with_jitter
    original_function_cholesky = released_common_module.cholesky_with_jitter
    if tuple(inspect.signature(original_reduced_cholesky).parameters) != (
        "K",
        "jitter0",
        "jitter_max",
    ):
        raise RuntimeError("pinned released TERA cholesky_with_jitter API changed")
    if tuple(inspect.signature(original_function_cholesky).parameters) != (
        "K",
        "jitter0",
        "jitter_max",
    ):
        raise RuntimeError("pinned released TERA function Cholesky API changed")
    lengthscale = parameters.lengthscale.to(device=train.X.device, dtype=train.X.dtype)
    evaluation_scaled = scale_inputs(evaluation.X, lengthscale)
    means: list[torch.Tensor] = []
    variances: list[torch.Tensor] = []
    selected_q_jitters: list[float] = []
    selected_function_jitters: list[float] = []
    function_jitter_by_neighbourhood: dict[tuple[int, ...], float] = {}

    def make_recording_cholesky(trace: list[float]):
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
                    trace.append(float(jitter))
                    return factor
                except Exception:
                    jitter *= 10.0
                    if jitter > jitter_max:
                        raise

        return recording_cholesky

    rows = zip(evaluation.X, evaluation_scaled, neighbours, strict=True)
    for target, target_scaled, indices in rows:
        target_q_jitters: list[float] = []
        target_function_jitters: list[float] = []
        released_tera_module.cholesky_with_jitter = make_recording_cholesky(target_q_jitters)
        released_common_module.cholesky_with_jitter = make_recording_cholesky(
            target_function_jitters
        )
        try:
            with torch.no_grad():
                mean, variance = predict_one(
                    x_eval=target.unsqueeze(0),
                    x_eval_scaled=target_scaled.unsqueeze(0),
                    idx=indices,
                )
        finally:
            released_tera_module.cholesky_with_jitter = original_reduced_cholesky
            released_common_module.cholesky_with_jitter = original_function_cholesky
        if len(target_q_jitters) != 1:
            raise RuntimeError(
                "released TERA must make exactly one reduced Cholesky call per target"
            )
        neighbourhood_key = tuple(int(index) for index in indices.detach().cpu().tolist())
        if len(target_function_jitters) == 1:
            function_jitter_by_neighbourhood[neighbourhood_key] = target_function_jitters[0]
        elif (
            len(target_function_jitters) == 0
            and neighbourhood_key in function_jitter_by_neighbourhood
        ):
            target_function_jitters.append(function_jitter_by_neighbourhood[neighbourhood_key])
        else:
            raise RuntimeError(
                "released TERA function cache made an unexpected number of Cholesky calls"
            )
        means.append(mean)
        variances.append(variance)
        selected_q_jitters.append(target_q_jitters[0])
        selected_function_jitters.append(target_function_jitters[0])

    mean = torch.stack(means).contiguous()
    latent_variance = torch.stack(variances).contiguous()
    variance_floor = torch.finfo(latent_variance.dtype).eps
    if bool((latent_variance == variance_floor).any().item()):
        raise RuntimeError(
            "released TERA variance equals its dtype epsilon clipping floor; "
            "unclipped variance positivity cannot be certified"
        )
    return (
        ScalarPrediction(
            mean=mean,
            latent_variance=latent_variance,
            observation_variance=latent_variance + latent_variance.new_tensor(parameters.sigma_f),
            released_variance_epsilon_floor=float(variance_floor),
            released_variance_epsilon_floor_inactive=True,
        ),
        selected_q_jitters,
        selected_function_jitters,
    )


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

    # Do not revisit the raw source-float64 arrays here.  These tensors and
    # parameters are the exact released float32 values promoted to float64.
    train64 = _quantized_split_to_float64(train32)
    evaluation64 = _quantized_split_to_float64(evaluation32)
    parameters64 = _quantized_parameters_to_float64(parameters32)
    if not torch.equal(train32.source_indices, train64.source_indices) or not torch.equal(
        evaluation32.source_indices,
        evaluation64.source_indices,
    ):
        raise RuntimeError("promoted splits changed source-row identity")

    # Every numerical arm receives this one positional index matrix.  It is
    # selected once by the pinned vendor KNN in source float32 and is never
    # recomputed as part of an N1/N2 comparison.
    fixed_neighbours = _neighbour_indices(
        train32.X,
        evaluation32.X,
        parameters32.lengthscale,
        m=50,
    )
    cross_dtype_neighbours = _cross_dtype_neighbour_identity(
        train32,
        evaluation32,
        parameters32,
        train64,
        evaluation64,
        parameters64,
        m=50,
    )
    tera32, tera32_q_jitters, tera32_function_jitters = _released_tera_fixed_prediction_set(
        train32,
        evaluation32,
        parameters32,
        m=50,
        neighbour_indices=fixed_neighbours,
    )

    source_q_jitter = float(
        torch.as_tensor(1e-8, dtype=SOURCE_DTYPE, device=train32.X.device).to(torch.float64)
    )
    support64, support64_targets = _support64_prediction_set(
        train64,
        evaluation64,
        parameters64,
        m=50,
        neighbour_indices=fixed_neighbours,
        support_coordinate_jitters=[source_q_jitter] * evaluation64.X.shape[0],
    )
    support64_matched_tera32_jitters, support64_matched_tera32_targets = _support64_prediction_set(
        train64,
        evaluation64,
        parameters64,
        m=50,
        neighbour_indices=fixed_neighbours,
        function_jitters=tera32_function_jitters,
        support_coordinate_jitters=tera32_q_jitters,
    )

    rank_epsilon = float(torch.finfo(SOURCE_DTYPE).eps)
    orbit32 = predict_orbit(
        train32,
        evaluation32.X,
        parameters32,
        m=50,
        neighbour_indices=fixed_neighbours,
        rank_epsilon=rank_epsilon,
        cg_tolerance=1e-5,
        use_preconditioner=True,
        function_jitter=source_q_jitter,
        reduced_jitter=source_q_jitter,
    )
    orbit32_tight = predict_orbit(
        train32,
        evaluation32.X,
        parameters32,
        m=50,
        neighbour_indices=fixed_neighbours,
        rank_epsilon=rank_epsilon,
        cg_tolerance=1e-8,
        use_preconditioner=True,
        function_jitter=source_q_jitter,
        reduced_jitter=source_q_jitter,
    )
    tera64, tera64_q_jitters, tera64_function_jitters = _released_tera_fixed_prediction_set(
        train64,
        evaluation64,
        parameters64,
        m=50,
        neighbour_indices=fixed_neighbours,
    )
    orbit64 = predict_orbit(
        train64,
        evaluation64.X,
        parameters64,
        m=50,
        neighbour_indices=fixed_neighbours,
        rank_epsilon=rank_epsilon,
        cg_tolerance=1e-10,
        use_preconditioner=True,
        function_jitter=source_q_jitter,
        reduced_jitter=source_q_jitter,
    )
    orbit64_n2_default = predict_orbit(
        train64,
        evaluation64.X,
        parameters64,
        m=50,
        neighbour_indices=fixed_neighbours,
        rank_epsilon=rank_epsilon,
        cg_tolerance=1e-5,
        use_preconditioner=True,
        function_jitter=source_q_jitter,
        reduced_jitter=source_q_jitter,
    )
    orbit64_n2_tight = predict_orbit(
        train64,
        evaluation64.X,
        parameters64,
        m=50,
        neighbour_indices=fixed_neighbours,
        rank_epsilon=rank_epsilon,
        cg_tolerance=1e-8,
        use_preconditioner=True,
        function_jitter=source_q_jitter,
        reduced_jitter=source_q_jitter,
    )

    support_ranks = torch.tensor(
        [record["diagnostics"]["numerical_rank"] for record in support64_targets],
        dtype=torch.long,
        device=train32.X.device,
    )
    orbit_predictions = (
        orbit32,
        orbit32_tight,
        orbit64,
        orbit64_n2_default,
        orbit64_n2_tight,
    )
    if any(prediction.details is None for prediction in orbit_predictions):
        raise RuntimeError("ORBIT fixed-geometry predictions lack solver details")
    rank_rule_aligned = all(
        torch.equal(prediction.details.ranks, support_ranks) for prediction in orbit_predictions
    )
    orbit32_geometry = _fixed_rank_geometry_records(
        train32,
        evaluation32,
        parameters32,
        m=50,
        neighbour_indices=fixed_neighbours,
        rank_epsilon=rank_epsilon,
    )
    orbit64_geometry = _fixed_rank_geometry_records(
        train64,
        evaluation64,
        parameters64,
        m=50,
        neighbour_indices=fixed_neighbours,
        rank_epsilon=rank_epsilon,
    )
    expected_physical_rank = max(0, min(fixed_neighbours.shape[1], train32.X.shape[1] - 6))
    physical_rank_aligned = bool((support_ranks == expected_physical_rank).all().item())

    result = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "scope": (
            "exploratory development-validation diagnostic only; thresholds are TBD and this "
            "artifact cannot constitute an F02b N0/N1/N2 pass"
        ),
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
        "comparison_geometry": {
            "neighbour_policy": (
                "one pinned-vendor KNN call in source float32; identical positional rows are "
                "injected into every TERA, ORBIT, and support64 arm"
            ),
            "rank_rule_name": RANK_RULE_NAME,
            "rank_epsilon": rank_epsilon,
            "support64_ranks": support_ranks.detach().cpu().tolist(),
            "orbit32_ranks": orbit32.details.ranks.detach().cpu().tolist(),
            "orbit64_ranks": orbit64.details.ranks.detach().cpu().tolist(),
            "orbit64_n2_default_ranks": (orbit64_n2_default.details.ranks.detach().cpu().tolist()),
            "orbit64_n2_tight_ranks": orbit64_n2_tight.details.ranks.detach().cpu().tolist(),
            "expected_physical_rank_min_m_d_minus_6": expected_physical_rank,
            "all_support64_ranks_match_physical_expectation": physical_rank_aligned,
            "all_arms_use_fixed_neighbours": True,
            "all_orbit_arms_match_support64_rank": rank_rule_aligned,
            "rank_rule_aligned_only_no_n1_n2_pass_claim": rank_rule_aligned,
            "support64_to_orbit32_q_projector": _projector_comparison(
                support64_targets,
                orbit32_geometry,
            ),
            "support64_to_orbit64_q_projector": _projector_comparison(
                support64_targets,
                orbit64_geometry,
            ),
            "cross_dtype_native_knn_sensitivity": cross_dtype_neighbours,
        },
        "float32_fixed_geometry": {
            "released_tera32_to_orbit32": _comparison(tera32, orbit32),
            "released_tera_selected_q_coordinate_jitter": tera32_q_jitters,
            "released_tera_selected_function_coordinate_jitter": tera32_function_jitters,
            "scores": {
                "TERA-50": _scalar_scores(evaluation32.value, tera32),
                "ORBIT-50": _scalar_scores(evaluation32.value, orbit32),
            },
            "solver": _orbit_solver(orbit32),
        },
        "float32_tighter_cg_fixed_geometry": {
            "comparison_to_float32_tera": _comparison(tera32, orbit32_tight),
            "solver": _orbit_solver(orbit32_tight),
        },
        "source_quantized_float64_prediction_only": {
            "input_construction": (
                "exact source-fp32-quantized train32/evaluation32/parameters32 values promoted "
                "to float64; raw source float64 arrays are not reloaded"
            ),
            "released_tera64_to_orbit64": _comparison(tera64, orbit64),
            "support64_to_released_tera64": _comparison(support64, tera64),
            "support64_to_orbit64": _comparison(support64, orbit64),
            "orbit32_to_support64": _comparison(
                support64,
                _cast_prediction(orbit32, torch.float64),
            ),
            "n2_same_stopping_tolerance_1e-5_orbit32_to_orbit64": {
                "comparison": _comparison(
                    orbit64_n2_default,
                    _cast_prediction(orbit32, torch.float64),
                ),
                "orbit32_solver": _orbit_solver(orbit32),
                "orbit64_solver": _orbit_solver(orbit64_n2_default),
            },
            "n2_same_stopping_tolerance_1e-8_orbit32_to_orbit64": {
                "comparison": _comparison(
                    orbit64_n2_tight,
                    _cast_prediction(orbit32_tight, torch.float64),
                ),
                "orbit32_solver": _orbit_solver(orbit32_tight),
                "orbit64_solver": _orbit_solver(orbit64_n2_tight),
            },
            "support64_matched_tera32_function_and_q_jitter_to_released_tera32": _comparison(
                support64_matched_tera32_jitters,
                _cast_prediction(tera32, torch.float64),
            ),
            "released_tera_selected_q_coordinate_jitter": tera64_q_jitters,
            "released_tera_selected_function_coordinate_jitter": tera64_function_jitters,
            "scores": {
                "TERA-50": _scalar_scores(evaluation64.value, tera64),
                "ORBIT-50": _scalar_scores(evaluation64.value, orbit64),
                "support64-50": _scalar_scores(evaluation64.value, support64),
            },
            "solver": _orbit_solver(orbit64),
            "support64_per_target": support64_targets,
            "support64_matched_tera32_function_and_q_jitter_per_target": (
                support64_matched_tera32_targets
            ),
            "tera_float32_to_float64": _comparison(
                tera64,
                _cast_prediction(tera32, torch.float64),
            ),
            "orbit32_default_to_orbit64_reference_confounded_by_stopping_tolerance": (
                _comparison(
                    orbit64,
                    _cast_prediction(orbit32, torch.float64),
                )
            ),
            "tera_float32_to_orbit_float64": _comparison(
                orbit64,
                _cast_prediction(tera32, torch.float64),
            ),
        },
        "fixed_source_rank_rule_float32_geometry": orbit32_geometry,
        "fixed_source_rank_rule_source_quantized_float64_geometry": orbit64_geometry,
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
