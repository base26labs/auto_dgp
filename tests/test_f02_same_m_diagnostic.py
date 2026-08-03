from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from experiments.f02_internal_models import (
    FrozenTERAParameters,
    ScalarPrediction,
    TensorConfirmatorySplit,
)
from experiments.f02_same_m_diagnostic import (
    DIAGNOSTIC_SCHEMA_VERSION,
    _neighbour_indices,
    _quantized_parameters_to_float64,
    _quantized_split_to_float64,
    _scalar_scores,
    _singular_spectra,
    _support64_prediction_set,
)


def _split(
    name: str,
    x: torch.Tensor,
    values: torch.Tensor,
    gradients: torch.Tensor,
    *,
    source_offset: int = 0,
) -> TensorConfirmatorySplit:
    rows = x.shape[0]
    indices = torch.arange(
        source_offset,
        source_offset + rows,
        dtype=torch.long,
        device=x.device,
    )
    return TensorConfirmatorySplit(
        name=name,
        source_indices=indices,
        X=x,
        value=values,
        gradient=gradients,
        trajectory_id=torch.zeros(rows, dtype=torch.long, device=x.device),
        time_index=torch.arange(rows, dtype=torch.long, device=x.device),
        time_value=torch.arange(rows, dtype=x.dtype, device=x.device) / 10.0,
    )


def test_help_does_not_load_data_or_fit_a_model() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "experiments/f02_same_m_diagnostic.py", "--help"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "development-only" in result.stdout.lower()


def test_singular_spectrum_reports_current_numerical_rank() -> None:
    x_train = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        dtype=torch.float32,
    )
    x_eval = torch.tensor([[0.5, 0.0]], dtype=torch.float32)
    parameters = FrozenTERAParameters(
        lengthscale=torch.ones(1),
        outputscale=1.0,
        sigma_f=0.1,
        sigma_g=0.1,
        kernel="rbf",
    )
    records = _singular_spectra(x_train, x_eval, parameters, m=3)
    assert len(records) == 1
    assert records[0]["algebraic_maximum_rank"] == 2
    assert records[0]["current_retained_rank"] == 1


def test_scalar_scores_use_raw_latent_variance() -> None:
    prediction = ScalarPrediction(
        mean=torch.tensor([1.0, 3.0], dtype=torch.float64),
        latent_variance=torch.tensor([2.0, 2.0], dtype=torch.float64),
        observation_variance=torch.tensor([9.0, 9.0], dtype=torch.float64),
    )
    scores = _scalar_scores(torch.tensor([0.0, 1.0], dtype=torch.float64), prediction)
    assert scores["rmse"] == (2.5**0.5)
    expected_nll = 0.5 * torch.log(torch.tensor(4.0 * torch.pi, dtype=torch.float64)) + 0.625
    assert scores["latent_gaussian_nll"] == expected_nll.item()


def test_float64_diagnostic_inputs_only_promote_exact_float32_values() -> None:
    raw_x = torch.tensor(
        [[1.00000007, -0.33333334], [12345.67891, 0.10000001]],
        dtype=torch.float64,
    )
    split32 = _split(
        "train",
        raw_x.to(dtype=torch.float32),
        torch.tensor([0.123456789, -9.87654321], dtype=torch.float32),
        torch.tensor(
            [[0.20000003, -0.40000007], [0.60000011, -0.80000013]],
            dtype=torch.float32,
        ),
    )
    parameters32 = FrozenTERAParameters(
        lengthscale=torch.tensor([0.80000007], dtype=torch.float32),
        outputscale=1.200000071,
        sigma_f=0.0300000017,
        sigma_g=0.0400000031,
        kernel="rbf",
    )

    split64 = _quantized_split_to_float64(split32)
    parameters64 = _quantized_parameters_to_float64(parameters32)

    assert DIAGNOSTIC_SCHEMA_VERSION == "f02_same_m_diagnostic_v2"
    for source, promoted in (
        (split32.X, split64.X),
        (split32.value, split64.value),
        (split32.gradient, split64.gradient),
        (split32.time_value, split64.time_value),
    ):
        assert promoted.dtype == torch.float64
        torch.testing.assert_close(promoted, source.to(torch.float64), rtol=0.0, atol=0.0)
    assert split64.source_indices is split32.source_indices
    assert split64.trajectory_id is split32.trajectory_id
    assert split64.time_index is split32.time_index
    torch.testing.assert_close(
        parameters64.lengthscale,
        parameters32.lengthscale.to(torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    for name in ("outputscale", "sigma_f", "sigma_g"):
        expected = float(torch.tensor(getattr(parameters32, name), dtype=torch.float32))
        assert getattr(parameters64, name) == expected
    assert split64.X[0, 0] != raw_x[0, 0]


def test_support64_prediction_set_records_per_target_numerical_evidence() -> None:
    generator = torch.Generator().manual_seed(73)
    fixed = torch.tensor([0.2, -0.3, 0.5, -0.7, 0.9], dtype=torch.float32)
    train_intrinsic = torch.randn(6, 2, generator=generator, dtype=torch.float32)
    evaluation_intrinsic = torch.randn(2, 2, generator=generator, dtype=torch.float32)
    train32 = _split(
        "train",
        torch.cat([train_intrinsic, fixed.expand(6, -1)], dim=1),
        torch.randn(6, generator=generator, dtype=torch.float32),
        torch.randn(6, 7, generator=generator, dtype=torch.float32),
        source_offset=100,
    )
    evaluation32 = _split(
        "validation",
        torch.cat([evaluation_intrinsic, fixed.expand(2, -1)], dim=1),
        torch.randn(2, generator=generator, dtype=torch.float32),
        torch.randn(2, 7, generator=generator, dtype=torch.float32),
        source_offset=900,
    )
    parameters32 = FrozenTERAParameters(
        lengthscale=torch.tensor([0.8], dtype=torch.float32),
        outputscale=1.2,
        sigma_f=0.03,
        sigma_g=0.04,
        kernel="rbf",
    )
    train64 = _quantized_split_to_float64(train32)
    evaluation64 = _quantized_split_to_float64(evaluation32)
    parameters64 = _quantized_parameters_to_float64(parameters32)

    prediction, records = _support64_prediction_set(
        train64,
        evaluation64,
        parameters64,
        m=4,
    )

    expected_neighbours = _neighbour_indices(
        train64.X,
        evaluation64.X,
        parameters64.lengthscale,
        m=4,
    )
    assert prediction.mean.shape == (2,)
    assert prediction.latent_variance.shape == (2,)
    assert bool(torch.isfinite(prediction.mean).all())
    assert bool((prediction.latent_variance > 0.0).all())
    assert len(records) == 2
    required_diagnostics = {
        "numerical_rank",
        "singular_values",
        "function_jitter_used",
        "function_spectrum_before_jitter",
        "support_coordinate_jitter",
        "support_spectrum_before_jitter",
        "support_spectrum_after_jitter",
        "support_condition_number_after_jitter",
        "support_relative_solve_residual",
        "support_relative_solve_residual_tolerance",
    }
    for target_position, record in enumerate(records):
        assert record["target_position"] == target_position
        assert record["target_source_index"] == 900 + target_position
        assert record["neighbour_source_indices"] == (
            train64.source_indices[expected_neighbours[target_position]].tolist()
        )
        diagnostics = record["diagnostics"]
        assert required_diagnostics <= diagnostics.keys()
        assert diagnostics["numerical_rank"] == 2
        assert len(diagnostics["singular_values"]) == 4
        assert len(diagnostics["support_spectrum_before_jitter"]) == 8
        assert (
            diagnostics["support_relative_solve_residual"]
            <= diagnostics["support_relative_solve_residual_tolerance"]
        )
        ambient_projector = torch.tensor(
            record["ambient_scaled_difference_support_projector"],
            dtype=torch.float64,
        )
        q_projector = torch.tensor(
            record["q_coordinate_support_projector"],
            dtype=torch.float64,
        )
        assert ambient_projector.shape == (7, 7)
        assert q_projector.shape == (4, 4)
        for projector in (ambient_projector, q_projector):
            torch.testing.assert_close(projector, projector.T, rtol=0.0, atol=1e-12)
            torch.testing.assert_close(
                projector @ projector,
                projector,
                rtol=1e-11,
                atol=1e-11,
            )
            torch.testing.assert_close(
                torch.trace(projector),
                projector.new_tensor(2.0),
                rtol=1e-11,
                atol=1e-11,
            )
