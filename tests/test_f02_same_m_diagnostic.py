from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from experiments.f02_internal_models import (
    FrozenTERAParameters,
    ScalarPrediction,
    TensorConfirmatorySplit,
    build_released_tera_predictor,
)
from experiments.f02_same_m_diagnostic import (
    DIAGNOSTIC_SCHEMA_VERSION,
    _cross_dtype_neighbour_identity,
    _fixed_rank_geometry_records,
    _neighbour_indices,
    _quantized_parameters_to_float64,
    _quantized_split_to_float64,
    _released_tera_fixed_prediction_set,
    _scalar_scores,
    _singular_spectra,
    _support64_prediction_set,
    _validate_source_checkout,
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


def test_source_checkout_binding_requires_exact_clean_commit(monkeypatch, tmp_path: Path) -> None:
    commit = "a" * 40
    tree = "b" * 40

    def clean_git(_root, *arguments, **_kwargs):
        responses = {
            ("rev-parse", "HEAD"): f"{commit}\n",
            ("rev-parse", "HEAD^{tree}"): f"{tree}\n",
            ("status", "--porcelain=v1", "--untracked-files=all"): "",
        }
        return responses[arguments]

    monkeypatch.setattr("experiments.f02_same_m_diagnostic._run_git", clean_git)
    assert _validate_source_checkout(commit, repo_root=tmp_path) == {
        "commit": commit,
        "tree": tree,
    }

    with pytest.raises(ValueError, match="full lowercase"):
        _validate_source_checkout("A" * 40, repo_root=tmp_path)
    with pytest.raises(RuntimeError, match="expected"):
        _validate_source_checkout("c" * 40, repo_root=tmp_path)

    def dirty_git(_root, *arguments, **_kwargs):
        if arguments[0] == "status":
            return "?? unexpected.txt\n"
        return clean_git(_root, *arguments, **_kwargs)

    monkeypatch.setattr("experiments.f02_same_m_diagnostic._run_git", dirty_git)
    with pytest.raises(RuntimeError, match="globally clean"):
        _validate_source_checkout(commit, repo_root=tmp_path)


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

    assert DIAGNOSTIC_SCHEMA_VERSION == "f02_same_m_diagnostic_v3"
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


def test_near_ties_prove_native_cross_dtype_knn_is_not_a_comparison_contract() -> None:
    dimension = 12
    generator = torch.Generator().manual_seed(10_000 * dimension)
    target = torch.rand(1, dimension, generator=generator, dtype=torch.float32)
    directions = torch.randn(80, dimension, generator=generator, dtype=torch.float32)
    directions = directions / torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    x_train = target + directions
    train32 = _split(
        "train",
        x_train,
        torch.zeros(80, dtype=torch.float32),
        torch.zeros(80, dimension, dtype=torch.float32),
        source_offset=100,
    )
    evaluation32 = _split(
        "validation",
        target,
        torch.zeros(1, dtype=torch.float32),
        torch.zeros(1, dimension, dtype=torch.float32),
        source_offset=900,
    )
    parameters32 = FrozenTERAParameters(
        lengthscale=torch.ones(1, dtype=torch.float32),
        outputscale=1.0,
        sigma_f=0.01,
        sigma_g=0.01,
        kernel="rbf",
    )
    train64 = _quantized_split_to_float64(train32)
    evaluation64 = _quantized_split_to_float64(evaluation32)
    parameters64 = _quantized_parameters_to_float64(parameters32)

    identity = _cross_dtype_neighbour_identity(
        train32,
        evaluation32,
        parameters32,
        train64,
        evaluation64,
        parameters64,
        m=20,
    )

    assert identity["native_recomputation_all_targets_same_order"] is False
    assert identity["native_recomputation_all_targets_same_set"] is False
    assert identity["fixed_comparisons_use_canonical_float32_indices"] is True
    canonical = _neighbour_indices(
        train32.X,
        evaluation32.X,
        parameters32.lengthscale,
        m=20,
    )
    assert identity["canonical_neighbour_source_indices"] == (
        train32.source_indices[canonical].tolist()
    )


def test_fixed_rank_geometry_uses_source_epsilon_without_relabelling_weak_modes_exact() -> None:
    singular_values = torch.tensor(
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 1e-7, 9e-8, 8e-8, 7e-8, 6e-8, 5e-8],
        dtype=torch.float32,
    )
    x_train = torch.diag(singular_values)
    train32 = _split(
        "train",
        x_train,
        torch.zeros(12, dtype=torch.float32),
        torch.zeros(12, 12, dtype=torch.float32),
    )
    evaluation32 = _split(
        "validation",
        torch.zeros(1, 12, dtype=torch.float32),
        torch.zeros(1, dtype=torch.float32),
        torch.zeros(1, 12, dtype=torch.float32),
        source_offset=100,
    )
    parameters32 = FrozenTERAParameters(
        lengthscale=torch.ones(1, dtype=torch.float32),
        outputscale=1.0,
        sigma_f=0.01,
        sigma_g=0.01,
        kernel="rbf",
    )
    train64 = _quantized_split_to_float64(train32)
    evaluation64 = _quantized_split_to_float64(evaluation32)
    parameters64 = _quantized_parameters_to_float64(parameters32)
    neighbours = torch.arange(12, dtype=torch.long).reshape(1, 12)
    rank_epsilon = float(torch.finfo(torch.float32).eps)

    record32 = _fixed_rank_geometry_records(
        train32,
        evaluation32,
        parameters32,
        m=12,
        neighbour_indices=neighbours,
        rank_epsilon=rank_epsilon,
    )[0]
    record64 = _fixed_rank_geometry_records(
        train64,
        evaluation64,
        parameters64,
        m=12,
        neighbour_indices=neighbours,
        rank_epsilon=rank_epsilon,
    )[0]

    assert record32["operational_retained_rank"] == 6
    assert record64["operational_retained_rank"] == 6
    assert record32["native_compute_retained_rank"] == 6
    assert record64["native_compute_retained_rank"] == 12
    assert record32["discarded_modes_are_unresolvable_at_native_cutoff"] is True
    assert record64["discarded_modes_are_unresolvable_at_native_cutoff"] is False


def test_released_tera_fixed_path_calls_pinned_one_target_api_with_caller_rows() -> None:
    from gp_sim_kl.utils import scale_inputs

    generator = torch.Generator().manual_seed(119)
    train = _split(
        "train",
        torch.randn(6, 4, generator=generator, dtype=torch.float64),
        torch.randn(6, generator=generator, dtype=torch.float64),
        torch.randn(6, 4, generator=generator, dtype=torch.float64),
        source_offset=100,
    )
    evaluation = _split(
        "validation",
        torch.randn(2, 4, generator=generator, dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        torch.zeros(2, 4, dtype=torch.float64),
        source_offset=900,
    )
    parameters = FrozenTERAParameters(
        lengthscale=torch.tensor([0.8], dtype=torch.float64),
        outputscale=1.2,
        sigma_f=0.03,
        sigma_g=0.04,
        kernel="rbf",
    )
    fixed_neighbours = torch.tensor([[5, 3, 1], [0, 4, 2]], dtype=torch.long)

    prediction, q_jitters, function_jitters = _released_tera_fixed_prediction_set(
        train,
        evaluation,
        parameters,
        m=3,
        neighbour_indices=fixed_neighbours,
    )
    predictor = build_released_tera_predictor(train, parameters, m=3)
    evaluation_scaled = scale_inputs(evaluation.X, parameters.lengthscale)
    with torch.no_grad():
        expected = [
            predictor._predict_one(
                x_eval=target.unsqueeze(0),
                x_eval_scaled=target_scaled.unsqueeze(0),
                idx=indices,
            )
            for target, target_scaled, indices in zip(
                evaluation.X,
                evaluation_scaled,
                fixed_neighbours,
                strict=True,
            )
        ]

    torch.testing.assert_close(prediction.mean, torch.stack([item[0] for item in expected]))
    torch.testing.assert_close(
        prediction.latent_variance,
        torch.stack([item[1] for item in expected]),
    )
    assert len(q_jitters) == evaluation.X.shape[0]
    assert len(function_jitters) == evaluation.X.shape[0]


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

    expected_neighbours = _neighbour_indices(
        train32.X,
        evaluation32.X,
        parameters32.lengthscale,
        m=4,
    )
    prediction, records = _support64_prediction_set(
        train64,
        evaluation64,
        parameters64,
        m=4,
        neighbour_indices=expected_neighbours,
    )
    neighbour_identity = _cross_dtype_neighbour_identity(
        train32,
        evaluation32,
        parameters32,
        train64,
        evaluation64,
        parameters64,
        m=4,
    )
    assert neighbour_identity["native_recomputation_all_targets_same_order"] is True
    assert neighbour_identity["native_recomputation_all_targets_same_set"] is True
    assert neighbour_identity["fixed_comparisons_use_canonical_float32_indices"] is True
    assert neighbour_identity["canonical_neighbour_source_indices"] == (
        train32.source_indices[expected_neighbours].tolist()
    )
    assert neighbour_identity["native_source_quantized_float64_neighbour_source_indices"] == (
        train64.source_indices[expected_neighbours].tolist()
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
