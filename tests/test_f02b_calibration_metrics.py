"""Tests for pure, fail-closed F02b numerical calibration metrics."""

import json
import math

import pytest
import torch

from experiments.f02b_calibration_metrics import (
    CalibrationMetricInputError,
    moment_error_metrics,
    nbody_physical_constraint_residuals,
    projector_metrics,
    rank_boundary_metrics,
    select_geometry_strata,
)


def _assert_strict_json(value):
    encoded = json.dumps(value, allow_nan=False, sort_keys=True)
    assert json.loads(encoded) == value


def test_moment_errors_use_fixed_scales_and_are_json_safe():
    result = moment_error_metrics(
        torch.tensor([0.0, 2.0], dtype=torch.float64),
        torch.tensor([0.5, 1.0], dtype=torch.float64),
        torch.tensor([0.5, 2.0], dtype=torch.float64),
        torch.tensor([0.75, 1.0], dtype=torch.float64),
        outputscale=4.0,
        sigma_f=1.0,
    )

    assert result["n_targets"] == 2
    assert result["per_target"]["absolute_mean"] == pytest.approx([0.5, 1.0])
    assert result["per_target"]["mean_over_max_one_abs_reference"] == pytest.approx([0.5, 0.5])
    assert result["per_target"]["mean_over_sqrt_outputscale"] == pytest.approx([0.25, 0.5])
    assert result["per_target"]["absolute_variance"] == pytest.approx([0.25, 1.0])
    assert result["per_target"][
        "variance_over_max_sigma_f_abs_reference_variance"
    ] == pytest.approx([0.25, 0.5])
    assert result["per_target"]["variance_over_outputscale"] == pytest.approx([0.0625, 0.25])
    assert result["max"]["mean_over_max_one_abs_reference"] == pytest.approx(0.5)
    assert result["max"]["variance_over_outputscale"] == pytest.approx(0.25)
    _assert_strict_json(result)


@pytest.mark.parametrize("bad_variance", [0.0, -1.0, float("nan"), float("inf")])
def test_moment_errors_reject_invalid_raw_variances(bad_variance):
    reference_variance = torch.tensor([1.0, 2.0], dtype=torch.float64)
    candidate_variance = torch.tensor([1.0, bad_variance], dtype=torch.float64)
    with pytest.raises(CalibrationMetricInputError):
        moment_error_metrics(
            torch.zeros(2, dtype=torch.float64),
            torch.zeros(2, dtype=torch.float64),
            reference_variance,
            candidate_variance,
            outputscale=1.0,
            sigma_f=0.1,
        )


@pytest.mark.parametrize("name,value", [("outputscale", 0.0), ("sigma_f", -1.0)])
def test_moment_errors_require_positive_scales(name, value):
    kwargs = {"outputscale": 1.0, "sigma_f": 0.1}
    kwargs[name] = value
    with pytest.raises(CalibrationMetricInputError, match="strictly positive"):
        moment_error_metrics(
            torch.zeros(1, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
            **kwargs,
        )


def test_moment_errors_reject_shape_and_dtype_mismatches():
    with pytest.raises(CalibrationMetricInputError, match="identical shapes"):
        moment_error_metrics(
            torch.zeros(2, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            outputscale=1.0,
            sigma_f=0.1,
        )
    with pytest.raises(CalibrationMetricInputError, match="identical dtypes"):
        moment_error_metrics(
            torch.zeros(2, dtype=torch.float64),
            torch.zeros(2, dtype=torch.float32),
            torch.ones(2, dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            outputscale=1.0,
            sigma_f=0.1,
        )
    with pytest.raises(CalibrationMetricInputError, match="CPU float64"):
        moment_error_metrics(
            torch.zeros(2, dtype=torch.float32),
            torch.zeros(2, dtype=torch.float32),
            torch.ones(2, dtype=torch.float32),
            torch.ones(2, dtype=torch.float32),
            outputscale=1.0,
            sigma_f=0.1,
        )


def test_projector_metrics_equal_principal_angle_statistics():
    angle = math.pi / 6.0
    direction = torch.tensor([math.cos(angle), math.sin(angle)], dtype=torch.float64)
    reference = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64))
    candidate = direction[:, None] @ direction[None, :]

    result = projector_metrics(reference, candidate, reference_rank=1, candidate_rank=1)

    assert result["difference_spectral_norm"] == pytest.approx(math.sin(angle))
    assert result["difference_frobenius_normalized"] == pytest.approx(math.sin(angle))
    assert result["difference_frobenius_norm"] == pytest.approx(math.sqrt(2.0) * math.sin(angle))
    assert result["maxabs"] == pytest.approx(math.sin(angle) * math.cos(angle))
    assert result["reference_symmetry_spectral_error"] == 0.0
    assert result["candidate_symmetry_spectral_error"] == 0.0
    assert result["reference_idempotence_spectral_error"] == 0.0
    assert result["candidate_idempotence_spectral_error"] == pytest.approx(0.0, abs=1e-15)
    assert result["reference_trace_error_from_rank"] == 0.0
    assert result["candidate_trace_error_from_rank"] == pytest.approx(0.0, abs=1e-15)
    _assert_strict_json(result)


def test_projector_rank_zero_reports_undefined_normalized_frobenius_without_division():
    reference = torch.zeros((2, 2), dtype=torch.float64)
    candidate = torch.zeros((2, 2), dtype=torch.float64)

    result = projector_metrics(reference, candidate, reference_rank=0, candidate_rank=0)

    assert result["difference_spectral_norm"] == 0.0
    assert result["difference_frobenius_norm"] == 0.0
    assert result["difference_frobenius_normalized"] is None
    assert result["reference_trace_error_from_rank"] == 0.0
    assert result["candidate_trace_error_from_rank"] == 0.0
    _assert_strict_json(result)


def test_projector_metrics_reject_non_square_and_mixed_precision_inputs():
    with pytest.raises(CalibrationMetricInputError, match="square"):
        projector_metrics(
            torch.zeros((2, 3), dtype=torch.float64),
            torch.zeros((2, 3), dtype=torch.float64),
            reference_rank=1,
            candidate_rank=1,
        )
    with pytest.raises(CalibrationMetricInputError, match="identical dtypes"):
        projector_metrics(
            torch.eye(2, dtype=torch.float64),
            torch.eye(2, dtype=torch.float32),
            reference_rank=1,
            candidate_rank=1,
        )
    with pytest.raises(CalibrationMetricInputError, match="ranks must match"):
        projector_metrics(
            torch.zeros((2, 2), dtype=torch.float64),
            torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64)),
            reference_rank=0,
            candidate_rank=1,
        )


def test_rank_boundary_metrics_use_strict_cutoff_and_expected_boundary():
    result = rank_boundary_metrics(
        torch.tensor([8.0, 2.0, 0.25, 0.0], dtype=torch.float64),
        cutoff=1.0,
        expected_rank=2,
    )

    assert result["strict_selected_rank"] == 2
    assert result["rank_matches_expected"] is True
    assert result["keep_over_cutoff_ratio"] == 2.0
    assert result["cutoff_over_drop_ratio"] == 4.0
    assert result["log2_keep_guard_bits"] == 1.0
    assert result["log2_drop_guard_bits"] == 2.0
    assert result["minimum_log2_guard_bits"] == 1.0
    assert result["boundary_keep_over_drop_ratio"] == 8.0
    assert result["drop_is_exact_zero"] is False
    assert result["zero_spectrum"] is False
    assert result["total_singular_value_energy"] == pytest.approx(68.0625)
    assert result["selected_discarded_singular_value_energy"] == pytest.approx(0.0625)
    assert result["selected_discarded_energy_fraction"] == pytest.approx(0.0625 / 68.0625)
    _assert_strict_json(result)

    equality = rank_boundary_metrics(
        torch.tensor([2.0, 1.0, 0.5], dtype=torch.float64),
        cutoff=1.0,
        expected_rank=1,
    )
    assert equality["strict_selected_rank"] == 1
    assert equality["log2_drop_guard_bits"] == 0.0


def test_rank_boundary_metrics_return_json_null_for_absent_or_infinite_sides():
    all_dropped = rank_boundary_metrics(
        torch.tensor([8.0, 2.0], dtype=torch.float64),
        cutoff=1.0,
        expected_rank=0,
    )
    assert all_dropped["keep_singular_value"] is None
    assert all_dropped["keep_over_cutoff_ratio"] is None
    assert all_dropped["log2_keep_guard_bits"] is None
    assert all_dropped["boundary_keep_over_drop_ratio"] is None
    _assert_strict_json(all_dropped)

    exact_zero_drop = rank_boundary_metrics(
        torch.tensor([8.0, 2.0, 0.0], dtype=torch.float64),
        cutoff=1.0,
        expected_rank=2,
    )
    assert exact_zero_drop["drop_is_exact_zero"] is True
    assert exact_zero_drop["cutoff_over_drop_ratio"] is None
    assert exact_zero_drop["log2_drop_guard_bits"] is None
    assert exact_zero_drop["boundary_keep_over_drop_ratio"] is None
    assert exact_zero_drop["minimum_log2_guard_bits"] == pytest.approx(1.0)
    _assert_strict_json(exact_zero_drop)

    all_kept = rank_boundary_metrics(
        torch.tensor([8.0, 2.0], dtype=torch.float64),
        cutoff=1.0,
        expected_rank=2,
    )
    assert all_kept["drop_singular_value"] is None
    assert all_kept["cutoff_over_drop_ratio"] is None
    assert all_kept["log2_drop_guard_bits"] is None
    _assert_strict_json(all_kept)

    zero_spectrum = rank_boundary_metrics(
        torch.zeros(4, dtype=torch.float64),
        cutoff=0.0,
        expected_rank=0,
    )
    assert zero_spectrum["zero_spectrum"] is True
    assert zero_spectrum["strict_selected_rank"] == 0
    assert zero_spectrum["rank_matches_expected"] is True
    assert zero_spectrum["drop_is_exact_zero"] is True
    assert zero_spectrum["minimum_log2_guard_bits"] is None
    assert zero_spectrum["selected_discarded_energy_fraction"] is None
    _assert_strict_json(zero_spectrum)


@pytest.mark.parametrize(
    "singular_values,match",
    [
        (torch.tensor([1.0, 2.0], dtype=torch.float64), "sorted descending"),
        (torch.tensor([1.0, -0.1], dtype=torch.float64), "nonnegative"),
        (torch.tensor([1.0, float("nan")], dtype=torch.float64), "finite"),
    ],
)
def test_rank_boundary_metrics_reject_invalid_spectra(singular_values, match):
    with pytest.raises(CalibrationMetricInputError, match=match):
        rank_boundary_metrics(singular_values, cutoff=0.5, expected_rank=1)


def test_rank_boundary_metrics_reject_invalid_cutoff_and_dtype():
    values = torch.tensor([2.0, 1.0], dtype=torch.float64)
    with pytest.raises(CalibrationMetricInputError, match="zero only"):
        rank_boundary_metrics(values, cutoff=0.0, expected_rank=1)
    with pytest.raises(CalibrationMetricInputError, match="nonnegative"):
        rank_boundary_metrics(values, cutoff=-1.0, expected_rank=1)
    with pytest.raises(CalibrationMetricInputError, match="dtype"):
        rank_boundary_metrics(
            values,
            cutoff=torch.tensor(1.0, dtype=torch.float32),
            expected_rank=1,
        )


def _constrained_raw_differences(masses: torch.Tensor) -> torch.Tensor:
    first_q = torch.tensor(
        [[1.0, -2.0], [0.5, 1.5], [-0.25, 0.75]],
        dtype=masses.dtype,
    )
    second_q = -(masses[0] / masses[1]) * first_q
    first_p = torch.tensor(
        [[0.4, -0.3], [1.2, 0.2], [-0.8, 0.6]],
        dtype=masses.dtype,
    )
    second_p = -first_p
    return torch.cat((first_q, second_q, first_p, second_p), dim=0)


def test_nbody_constraint_residuals_use_mass_and_span_adjusted_coefficients():
    masses = torch.tensor([2.0, 3.0], dtype=torch.float64)
    x_span = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
        dtype=torch.float64,
    )
    standardized = _constrained_raw_differences(masses) / x_span[:, None]

    result = nbody_physical_constraint_residuals(standardized, masses, x_span)

    assert result["n_particles"] == 2
    assert result["dimension"] == 12
    assert result["n_differences"] == 2
    assert result["constraint_labels"] == [
        "position_x",
        "position_y",
        "position_z",
        "momentum_x",
        "momentum_y",
        "momentum_z",
    ]
    assert result["global_normalized_residual"] == pytest.approx(0.0, abs=1e-16)
    assert result["normalized_excess_over_roundoff_estimate"] == pytest.approx(0.0, abs=1e-16)
    assert result["matrix_product_gamma"] == pytest.approx(
        12 * torch.finfo(torch.float64).eps / (1 - 12 * torch.finfo(torch.float64).eps)
    )
    assert result["normalization_floor"] == torch.finfo(torch.float64).tiny
    assert result["per_row_normalized_residual"] == pytest.approx([0.0] * 6, abs=1e-16)
    constraint = torch.tensor(result["constraint_matrix"], dtype=torch.float64)
    assert constraint[0, 0] == pytest.approx(masses[0] * x_span[0])
    assert constraint[0, 3] == pytest.approx(masses[1] * x_span[3])
    assert constraint[3, 6] == pytest.approx(x_span[6])
    assert constraint[3, 9] == pytest.approx(x_span[9])
    expected_scale = float(
        torch.linalg.matrix_norm(constraint, ord=2) * torch.linalg.matrix_norm(standardized, ord=2)
    )
    expected_roundoff = result["matrix_product_gamma"] * float(
        torch.linalg.matrix_norm(torch.abs(constraint) @ torch.abs(standardized), ord=2)
    )
    assert result["normalization_scale_spectral"] == pytest.approx(expected_scale)
    assert result["roundoff_estimate_spectral"] == pytest.approx(expected_roundoff)
    assert result["global_normalized_residual"] == pytest.approx(
        result["global_residual_spectral"] / max(expected_scale, torch.finfo(torch.float64).tiny)
    )
    assert result["residual_over_roundoff_estimate"] == pytest.approx(
        result["global_residual_spectral"] / max(expected_roundoff, torch.finfo(torch.float64).tiny)
    )
    _assert_strict_json(result)

    perturbed = standardized.clone()
    perturbed[0, 0] += 0.1
    changed = nbody_physical_constraint_residuals(perturbed, masses, x_span)
    assert changed["per_row_residual_norm"][0] == pytest.approx(0.2)
    assert changed["per_row_normalized_residual"][0] > 0.0
    assert changed["per_row_residual_norm"][1:] == pytest.approx([0.0] * 5, abs=1e-16)
    assert changed["global_normalized_residual"] > 0.0
    assert changed["global_residual_spectral"] > 0.0
    assert changed["residual_over_roundoff_estimate"] > 1.0
    assert changed["normalized_excess_over_roundoff_estimate"] > 0.0
    _assert_strict_json(changed)


def test_nbody_zero_differences_are_valid_and_do_not_divide_by_zero():
    result = nbody_physical_constraint_residuals(
        torch.zeros((12, 3), dtype=torch.float64),
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        torch.ones(12, dtype=torch.float64),
    )
    assert result["difference_spectral_norm"] == 0.0
    assert result["global_normalized_residual"] == 0.0
    assert result["per_row_normalized_residual"] == [0.0] * 6
    assert result["residual_over_roundoff_estimate"] == 0.0
    assert result["normalized_excess_over_roundoff_estimate"] == 0.0
    _assert_strict_json(result)


def test_nbody_constraint_residuals_reject_layout_and_dtype_errors():
    with pytest.raises(CalibrationMetricInputError, match="promotion to float64"):
        nbody_physical_constraint_residuals(
            torch.zeros((12, 2), dtype=torch.float32),
            torch.ones(2, dtype=torch.float32),
            torch.ones(12, dtype=torch.float32),
        )
    with pytest.raises(CalibrationMetricInputError, match=r"6 \* n_particles"):
        nbody_physical_constraint_residuals(
            torch.zeros((11, 2), dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            torch.ones(12, dtype=torch.float64),
        )
    with pytest.raises(CalibrationMetricInputError, match="dtype"):
        nbody_physical_constraint_residuals(
            torch.zeros((12, 2), dtype=torch.float64),
            torch.ones(2, dtype=torch.float32),
            torch.ones(12, dtype=torch.float64),
        )
    with pytest.raises(CalibrationMetricInputError, match="strictly positive"):
        nbody_physical_constraint_residuals(
            torch.zeros((12, 2), dtype=torch.float64),
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            torch.ones(12, dtype=torch.float64),
        )


def _geometry_record(source_index, keep_guard, drop_guard, label):
    return {
        "target_source_index": source_index,
        "log2_keep_guard_bits": keep_guard,
        "log2_drop_guard_bits": drop_guard,
        "label_that_must_not_affect_selection": label,
    }


def test_geometry_strata_selection_is_guard_ordered_and_source_index_tied():
    records = [
        _geometry_record(9, 1.0, 4.0, "best label"),
        _geometry_record(3, 1.0, 5.0, "worst label"),
        _geometry_record(7, 2.0, 8.0, "middle"),
        _geometry_record(1, 3.0, 7.0, "x"),
        _geometry_record(5, 4.0, None, "y"),
    ]

    three = select_geometry_strata(records, count=3)
    assert [row["stratum"] for row in three["selected"]] == ["worst", "median", "best"]
    assert [row["target_source_index"] for row in three["selected"]] == [3, 7, 5]
    assert [row["minimum_log2_guard_bits"] for row in three["selected"]] == [1.0, 2.0, 4.0]
    _assert_strict_json(three)

    two = select_geometry_strata(records, count=2)
    assert [row["target_source_index"] for row in two["selected"]] == [3, 5]
    _assert_strict_json(two)

    even = select_geometry_strata(
        [
            _geometry_record(0, 0.0, 10.0, "unused"),
            _geometry_record(1, 1.0, 10.0, "unused"),
            _geometry_record(2, 2.0, 10.0, "unused"),
            _geometry_record(3, 3.0, 10.0, "unused"),
        ],
        count=3,
    )
    assert [row["target_source_index"] for row in even["selected"]] == [0, 2, 3]


def test_geometry_strata_selection_accepts_precomputed_minimum_guard():
    result = select_geometry_strata(
        [
            {"target_source_index": 4, "minimum_log2_guard_bits": -1.0},
            {"target_source_index": 2, "minimum_log2_guard_bits": 3.0},
        ],
        count=2,
    )
    assert [row["target_source_index"] for row in result["selected"]] == [4, 2]


@pytest.mark.parametrize(
    "records,count,match",
    [
        ([{"target_source_index": 1, "minimum_log2_guard_bits": 1.0}], 2, "at least"),
        (
            [
                {"target_source_index": 1, "minimum_log2_guard_bits": 1.0},
                {"target_source_index": 1, "minimum_log2_guard_bits": 2.0},
            ],
            2,
            "unique",
        ),
        (
            [
                {"target_source_index": 1, "minimum_log2_guard_bits": float("inf")},
                {"target_source_index": 2, "minimum_log2_guard_bits": 2.0},
            ],
            2,
            "finite",
        ),
        (
            [
                {
                    "target_source_index": 1,
                    "log2_keep_guard_bits": "label-derived",
                    "log2_drop_guard_bits": 2.0,
                },
                {"target_source_index": 2, "minimum_log2_guard_bits": 2.0},
            ],
            2,
            "real scalar",
        ),
        (
            [
                {"target_source_index": -1, "minimum_log2_guard_bits": 1.0},
                {"target_source_index": 2, "minimum_log2_guard_bits": 2.0},
            ],
            2,
            "nonnegative",
        ),
        (
            [
                {
                    "target_source_index": 1,
                    "minimum_log2_guard_bits": 3.0,
                    "log2_keep_guard_bits": 2.0,
                    "log2_drop_guard_bits": 4.0,
                },
                {"target_source_index": 2, "minimum_log2_guard_bits": 2.0},
            ],
            2,
            "inconsistent",
        ),
    ],
)
def test_geometry_strata_selection_fails_closed(records, count, match):
    with pytest.raises(CalibrationMetricInputError, match=match):
        select_geometry_strata(records, count=count)
