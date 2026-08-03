"""Shared internal-model adapters for the F02 confirmatory experiment.

The released TERA implementation uses ``sigma_f`` and ``sigma_g`` as noise
*variances* while fitting.  Its lower-level prediction data container is
asymmetric: ``sigma_f`` is stored there as a standard deviation and squared by
the function-conditioning cache, whereas ``sigma_g`` remains a variance.
This module is the single boundary that translates those conventions.  ORBIT
and the value-only control receive the fitted variances directly.

All inputs are already in the canonical train-normalized F02 coordinates.  No
row selection, reordering, or further normalization is performed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from data.load_nbody_confirmatory import PreparedConfirmatoryBundle
from gp.orbit import MarginalPredictions, predict_marginal_values


@dataclass(frozen=True, slots=True)
class TensorConfirmatorySplit:
    """One prepared split copied to torch without changing row order."""

    name: str
    source_indices: torch.Tensor
    X: torch.Tensor
    value: torch.Tensor
    gradient: torch.Tensor
    trajectory_id: torch.Tensor
    time_index: torch.Tensor
    time_value: torch.Tensor

    def __post_init__(self) -> None:
        if self.X.ndim != 2 or self.X.shape[0] == 0:
            raise ValueError("X must have nonzero shape (n, d)")
        n, dimension = self.X.shape
        expected_shapes = {
            "source_indices": (n,),
            "value": (n,),
            "gradient": (n, dimension),
            "trajectory_id": (n,),
            "time_index": (n,),
            "time_value": (n,),
        }
        for field, shape in expected_shapes.items():
            tensor = getattr(self, field)
            if tensor.shape != shape:
                raise ValueError(f"{field} must have shape {shape}")
            if tensor.device != self.X.device:
                raise ValueError("all split tensors must be on the same device")
        if not self.X.is_floating_point() or not self.value.is_floating_point():
            raise TypeError("X and value must use a floating dtype")
        if self.gradient.dtype != self.X.dtype or self.value.dtype != self.X.dtype:
            raise TypeError("X, value, and gradient must use the same dtype")

    @property
    def y(self) -> torch.Tensor:
        """Alias matching the released MD22 scalar-target name."""

        return self.value

    @property
    def g(self) -> torch.Tensor:
        """Alias matching the released MD22 gradient-target name."""

        return self.gradient


@dataclass(frozen=True, slots=True)
class FrozenTERAParameters:
    """Detached learned TERA parameters; both ``sigma`` fields are variances."""

    lengthscale: torch.Tensor
    outputscale: float
    sigma_f: float
    sigma_g: float
    kernel: str
    gradient_noise_model: str = "iid"

    def __post_init__(self) -> None:
        lengthscale = torch.as_tensor(self.lengthscale).detach().clone().reshape(-1).contiguous()
        if lengthscale.numel() == 0 or not lengthscale.is_floating_point():
            raise ValueError("lengthscale must be a nonempty floating tensor")
        if not bool(torch.isfinite(lengthscale).all()) or bool((lengthscale <= 0.0).any()):
            raise ValueError("lengthscale must be finite and strictly positive")
        object.__setattr__(self, "lengthscale", lengthscale)

        scalars = {
            "outputscale": float(self.outputscale),
            "sigma_f": float(self.sigma_f),
            "sigma_g": float(self.sigma_g),
        }
        if not math.isfinite(scalars["outputscale"]) or scalars["outputscale"] <= 0.0:
            raise ValueError("outputscale must be finite and strictly positive")
        for name in ("sigma_f", "sigma_g"):
            if not math.isfinite(scalars[name]) or scalars[name] < 0.0:
                raise ValueError(f"{name} variance must be finite and non-negative")
        for name, value in scalars.items():
            object.__setattr__(self, name, value)
        if self.kernel not in {"rbf", "matern52"}:
            raise ValueError("kernel must be 'rbf' or 'matern52'")
        if self.gradient_noise_model != "iid":
            raise ValueError("F02 internal TERA/ORBIT comparisons require iid gradient noise")

    @property
    def value_noise_variance(self) -> float:
        return self.sigma_f

    @property
    def gradient_noise_variance(self) -> float:
        return self.sigma_g


@dataclass(frozen=True, slots=True)
class ScalarPrediction:
    """Scalar predictive moments with unambiguous variance semantics."""

    mean: torch.Tensor
    latent_variance: torch.Tensor
    observation_variance: torch.Tensor
    details: Any | None = None
    released_variance_epsilon_floor: float | None = None
    released_variance_epsilon_floor_inactive: bool | None = None

    def __post_init__(self) -> None:
        if self.mean.ndim != 1 or self.mean.shape != self.latent_variance.shape:
            raise ValueError("mean and latent_variance must have matching vector shapes")
        if self.observation_variance.shape != self.mean.shape:
            raise ValueError("observation_variance must match mean")
        floor = self.released_variance_epsilon_floor
        inactive = self.released_variance_epsilon_floor_inactive
        if (floor is None) != (inactive is None):
            raise ValueError("released variance floor value/status must be supplied together")
        if floor is not None:
            if not math.isfinite(float(floor)) or float(floor) <= 0.0:
                raise ValueError("released variance epsilon floor must be finite and positive")
            if inactive is not True:
                raise ValueError(
                    "a returned prediction must certify the released floor is inactive"
                )


def _ensure_released_tera_available() -> None:
    # Importing gp.tera adds the frozen vendor's src directory to sys.path.
    from gp import tera as released_tera_wrapper

    if not released_tera_wrapper._VENDOR_SRC:  # pragma: no cover - defensive import guard
        raise RuntimeError("released TERA vendor source is unavailable")


def _copy_float_array(
    value: np.ndarray,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> torch.Tensor:
    # Prepared arrays are intentionally read-only.  Copying avoids exposing
    # them through a writable torch view while preserving their exact order.
    return torch.as_tensor(np.array(value, copy=True), dtype=dtype, device=device).contiguous()


def _copy_index_array(
    value: np.ndarray,
    *,
    device: torch.device | str | None,
) -> torch.Tensor:
    return torch.as_tensor(
        np.array(value, copy=True),
        dtype=torch.long,
        device=device,
    ).contiguous()


def prepared_split_to_tensors(
    bundle: PreparedConfirmatoryBundle,
    split: str,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> TensorConfirmatorySplit:
    """Copy one authoritative prepared-bundle split to torch in-place order."""

    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("dtype must be a floating torch dtype")
    prepared = bundle.prepared.split(split)
    return TensorConfirmatorySplit(
        name=prepared.name,
        source_indices=_copy_index_array(prepared.source_indices, device=device),
        X=_copy_float_array(prepared.X, dtype=dtype, device=device),
        value=_copy_float_array(prepared.E, dtype=dtype, device=device),
        gradient=_copy_float_array(prepared.F, dtype=dtype, device=device),
        trajectory_id=_copy_index_array(prepared.trajectory_id, device=device),
        time_index=_copy_index_array(prepared.time_index, device=device),
        time_value=_copy_float_array(prepared.time_value, dtype=dtype, device=device),
    )


def _to_released_md22_split(train: TensorConfirmatorySplit) -> Any:
    """Marshal normalized tensors into the released training container."""

    _ensure_released_tera_available()
    from md22_regression.data import EnergyForceScaler, MD22Split

    empty_x = train.X[:0]
    empty_value = train.value[:0]
    empty_gradient = train.gradient[:0]
    identity_scaler = EnergyForceScaler(
        energy_mean=train.value.new_zeros(()),
        energy_std=train.value.new_ones(()),
        x_scale=1.0,
    )
    return MD22Split(
        name=train.name,
        preprocessing_version="f02_canonical_train_normalized_v1",
        split_id=f"f02-{train.name}",
        X_train=train.X,
        y_train=train.value,
        g_train=train.gradient,
        # These fields are used only by optional logging metrics.  With the
        # mandatory log_every=0 they remain inert identity-unit placeholders.
        E_train=train.value,
        F_train=train.gradient,
        X_test=empty_x,
        y_test=empty_value,
        g_test=empty_gradient,
        E_test=empty_value,
        F_test=empty_gradient,
        scaler=identity_scaler,
        n_atoms=max(1, train.X.shape[1] // 3),
        train_indices=train.source_indices,
        test_indices=train.source_indices[:0],
    )


def fit_released_tera(
    train: TensorConfirmatorySplit,
    *,
    training_m: int = 20,
    train_steps: int = 0,
    train_epochs: int = 0,
    kernel: str = "rbf",
    outputscale: float = 1.0,
    sigma_f: float = 1e-3,
    sigma_g: float = 1e-3,
    lengthscale: float | list[float] | None = 1.0,
    lengthscale_init: str = "median",
    lengthscale_init_max_points: int = 2048,
    use_ard: bool = False,
    seed: int = 0,
    batch_size: int = 256,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    graph_refresh_epochs: int = 0,
    learn_lengthscale: bool = True,
    learn_outputscale: bool = True,
    learn_sigma_f: bool = True,
    learn_sigma_g: bool = True,
    min_sigma_f: float = 1e-6,
    min_sigma_g: float = 0.0,
) -> Any:
    """Fit the unmodified released ``TERAModel`` on canonical tensors.

    ``sigma_f`` and ``sigma_g`` are variances, matching the released MD22
    model's fitting API.  F02 fixes ``gradient_noise_model='iid'``.
    """

    if training_m <= 0:
        raise ValueError("training_m must be positive")
    if train_steps < 0:
        raise ValueError("train_steps must be non-negative")
    if train_epochs < 0:
        raise ValueError("train_epochs must be non-negative")
    if train_steps > 0 and train_epochs > 0:
        raise ValueError("set at most one of train_steps and train_epochs")
    _ensure_released_tera_available()
    from md22_regression.models.tera import TERAModel

    model = TERAModel(
        m=training_m,
        kernel=kernel,
        outputscale=outputscale,
        sigma_f=sigma_f,
        sigma_g=sigma_g,
        lengthscale=lengthscale,
        lengthscale_init=lengthscale_init,
        lengthscale_init_max_points=lengthscale_init_max_points,
        use_ard=use_ard,
        seed=seed,
        train_steps=train_steps,
        train_epochs=train_epochs,
        graph_refresh_epochs=graph_refresh_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        learn_lengthscale=learn_lengthscale,
        learn_outputscale=learn_outputscale,
        learn_sigma_f=learn_sigma_f,
        learn_sigma_g=learn_sigma_g,
        min_sigma_f=min_sigma_f,
        min_sigma_g=min_sigma_g,
        log_every=0,
        gradient_noise_model="iid",
    )
    model.fit(_to_released_md22_split(train))
    return model


def freeze_tera_parameters(model: Any) -> FrozenTERAParameters:
    """Snapshot the learned parameters needed by all three internal arms."""

    if model.lengthscale is None:
        raise RuntimeError("TERAModel must be fit before its parameters are frozen")
    return FrozenTERAParameters(
        lengthscale=model.lengthscale,
        outputscale=model.outputscale,
        sigma_f=model.sigma_f,
        sigma_g=model.sigma_g,
        kernel=model.kernel,
        gradient_noise_model=model.gradient_noise_model,
    )


def _prediction_data(
    train: TensorConfirmatorySplit,
    parameters: FrozenTERAParameters,
) -> Any:
    _ensure_released_tera_available()
    from gp_sim_kl.data import SimulatedDataset
    from gp_sim_kl.utils import scale_inputs

    lengthscale = parameters.lengthscale.to(device=train.X.device, dtype=train.X.dtype)
    x_scaled = scale_inputs(train.X, lengthscale)
    empty = train.X[:0]
    return SimulatedDataset(
        X_train=train.X,
        X_train_scaled=x_scaled,
        X_eval=empty,
        X_eval_scaled=empty,
        lengthscale=lengthscale,
        outputscale=parameters.outputscale,
        # FunctionConditioningCache squares sigma_f; sigma_g is added directly.
        sigma_f=math.sqrt(parameters.sigma_f),
        sigma_g=parameters.sigma_g,
        kernel_name=parameters.kernel,
        f_train_obs=train.value,
        g_train_obs=train.gradient,
        z_train_obs=torch.cat(
            [train.value.unsqueeze(1), train.gradient],
            dim=1,
        ).reshape(-1),
        sampling_backend="f02_confirmatory",
    )


def build_released_tera_predictor(
    train: TensorConfirmatorySplit,
    parameters: FrozenTERAParameters,
    *,
    m: int,
) -> Any:
    """Reconstruct the released private predictor at an arbitrary prediction ``m``."""

    if m <= 0:
        raise ValueError("m must be positive")
    _ensure_released_tera_available()
    from md22_regression.models.tera import _MD22TERAPredictor

    predictor = _MD22TERAPredictor(m=m, gradient_noise_model="iid")
    predictor.build(_prediction_data(train, parameters))
    return predictor


def _validated_eval(
    train: TensorConfirmatorySplit,
    x_eval: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(x_eval, torch.Tensor):
        raise TypeError("x_eval must be a torch tensor")
    if x_eval.ndim != 2 or x_eval.shape[1] != train.X.shape[1]:
        raise ValueError(f"x_eval must have shape (n, {train.X.shape[1]})")
    if x_eval.dtype != train.X.dtype or x_eval.device != train.X.device:
        raise TypeError("x_eval must match the training tensor dtype and device")
    return x_eval


def _scalar_prediction(
    mean: torch.Tensor,
    latent_variance: torch.Tensor,
    parameters: FrozenTERAParameters,
    *,
    details: Any | None = None,
    released_variance_epsilon_floor: float | None = None,
    released_variance_epsilon_floor_inactive: bool | None = None,
) -> ScalarPrediction:
    observation_variance = latent_variance + latent_variance.new_tensor(parameters.sigma_f)
    return ScalarPrediction(
        mean=mean,
        latent_variance=latent_variance,
        observation_variance=observation_variance,
        details=details,
        released_variance_epsilon_floor=released_variance_epsilon_floor,
        released_variance_epsilon_floor_inactive=released_variance_epsilon_floor_inactive,
    )


def _empty_prediction(
    x_eval: torch.Tensor,
    parameters: FrozenTERAParameters,
    *,
    details: Any | None = None,
) -> ScalarPrediction:
    empty = x_eval.new_empty((0,))
    return _scalar_prediction(empty, empty.clone(), parameters, details=details)


def predict_released_tera(
    train: TensorConfirmatorySplit,
    x_eval: torch.Tensor,
    parameters: FrozenTERAParameters,
    *,
    m: int,
) -> ScalarPrediction:
    """Predict released dense TERA marginals using frozen learned parameters."""

    x_eval = _validated_eval(train, x_eval)
    if x_eval.shape[0] == 0:
        return _empty_prediction(x_eval, parameters)
    predictor = build_released_tera_predictor(train, parameters, m=m)
    prediction = predictor.predict_f_marginals(x_eval)
    variance_floor = torch.finfo(prediction.var.dtype).eps
    if bool((prediction.var == variance_floor).any().item()):
        raise RuntimeError(
            "released TERA variance equals its dtype epsilon clipping floor; "
            "unclipped variance positivity cannot be certified"
        )
    return _scalar_prediction(
        prediction.mean,
        prediction.var,
        parameters,
        released_variance_epsilon_floor=float(variance_floor),
        released_variance_epsilon_floor_inactive=True,
    )


def predict_orbit(
    train: TensorConfirmatorySplit,
    x_eval: torch.Tensor,
    parameters: FrozenTERAParameters,
    *,
    m: int,
    rank: int | None = None,
    relative_rank_tolerance: float | None = None,
    cg_tolerance: float = 1e-10,
    cg_max_iterations: int | None = None,
    use_preconditioner: bool = True,
    function_jitter: float = 1e-8,
    reduced_jitter: float = 1e-8,
) -> ScalarPrediction:
    """Predict ORBIT marginals with exactly TERA's frozen kernel/noise state."""

    x_eval = _validated_eval(train, x_eval)
    if x_eval.shape[0] == 0:
        return _empty_prediction(x_eval, parameters)
    details: MarginalPredictions = predict_marginal_values(
        train.X,
        train.value,
        train.gradient,
        x_eval,
        m=m,
        lengthscale=parameters.lengthscale,
        outputscale=parameters.outputscale,
        value_noise_variance=parameters.sigma_f,
        gradient_noise_variance=parameters.sigma_g,
        kernel=parameters.kernel,
        gradient_noise_model="iid",
        rank=rank,
        relative_rank_tolerance=relative_rank_tolerance,
        cg_tolerance=cg_tolerance,
        cg_max_iterations=cg_max_iterations,
        use_preconditioner=use_preconditioner,
        function_jitter=function_jitter,
        reduced_jitter=reduced_jitter,
    )
    return _scalar_prediction(details.mean, details.variance, parameters, details=details)


def predict_value_only_local_gp(
    train: TensorConfirmatorySplit,
    x_eval: torch.Tensor,
    parameters: FrozenTERAParameters,
    *,
    m: int,
) -> ScalarPrediction:
    """Prediction-time value-only ablation with derivative-trained TERA parameters."""

    if m <= 0:
        raise ValueError("m must be positive")
    x_eval = _validated_eval(train, x_eval)
    if x_eval.shape[0] == 0:
        return _empty_prediction(x_eval, parameters)

    _ensure_released_tera_available()
    from gp_sim_kl.deroos import function_covariance
    from gp_sim_kl.ordering import knn_to_eval
    from gp_sim_kl.utils import cholesky_with_jitter, scale_inputs

    lengthscale = parameters.lengthscale.to(device=train.X.device, dtype=train.X.dtype)
    train_scaled = scale_inputs(train.X, lengthscale)
    eval_scaled = scale_inputs(x_eval, lengthscale)
    neighbourhoods = knn_to_eval(train_scaled, eval_scaled, m)
    cache: dict[tuple[int, ...], tuple[torch.Tensor, torch.Tensor]] = {}
    means: list[torch.Tensor] = []
    latent_variances: list[torch.Tensor] = []
    for target, indices in zip(x_eval, neighbourhoods, strict=True):
        key = tuple(int(index) for index in indices.detach().cpu().tolist())
        cached = cache.get(key)
        if cached is None:
            x_condition = train.X[indices]
            covariance = function_covariance(
                x_condition,
                x_condition,
                lengthscale,
                parameters.outputscale,
                parameters.kernel,
            )
            covariance = 0.5 * (covariance + covariance.T)
            covariance = covariance + parameters.sigma_f * torch.eye(
                indices.numel(),
                device=train.X.device,
                dtype=train.X.dtype,
            )
            factor = cholesky_with_jitter(covariance)
            value_weights = torch.cholesky_solve(
                train.value[indices].unsqueeze(1),
                factor,
            ).squeeze(1)
            cache[key] = factor, value_weights
        else:
            factor, value_weights = cached

        target_cross = function_covariance(
            train.X[indices],
            target.unsqueeze(0),
            lengthscale,
            parameters.outputscale,
            parameters.kernel,
        ).squeeze(1)
        target_weights = torch.cholesky_solve(target_cross.unsqueeze(1), factor).squeeze(1)
        means.append(torch.dot(target_cross, value_weights))
        latent_variances.append(
            target_cross.new_tensor(parameters.outputscale)
            - torch.dot(target_cross, target_weights)
        )

    return _scalar_prediction(
        torch.stack(means),
        torch.stack(latent_variances),
        parameters,
    )


__all__ = [
    "FrozenTERAParameters",
    "ScalarPrediction",
    "TensorConfirmatorySplit",
    "build_released_tera_predictor",
    "fit_released_tera",
    "freeze_tera_parameters",
    "predict_orbit",
    "predict_released_tera",
    "predict_value_only_local_gp",
    "prepared_split_to_tensors",
]
