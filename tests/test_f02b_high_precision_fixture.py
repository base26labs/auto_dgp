from __future__ import annotations

import inspect
import json
from fractions import Fraction

import mpmath as mp
import numpy as np
import pytest
import torch

from experiments import f02b_high_precision_fixture
from experiments.f02_support_dense_oracle import predict_local_dense_support
from experiments.f02b_high_precision_fixture import (
    HighPrecisionFixtureError,
    build_high_precision_rbf_reference,
    rbf_kernel_blocks_exact_fp32,
)


def _mp_matrix(record: dict[str, object]) -> list[list[mp.mpf]]:
    values = record["values"]
    assert isinstance(values, list)
    return [[mp.mpf(value) for value in row] for row in values]


def _fixture_kwargs() -> dict[str, object]:
    return {
        "x_condition": [[-1.0], [1.0]],
        "value_condition": [0.5, -0.25],
        "gradient_condition": [[0.125], [-0.375]],
        "x_target": [0.0],
        "support_basis": [[1.0]],
        "support_coordinates": [[-1.0], [1.0]],
        "lengthscale": 1.0,
        "outputscale": 1.0,
        "value_noise_variance": 2.0**-4,
        "gradient_noise_variance": 2.0**-5,
        "gradient_noise_model": "iid",
        "function_jitter": 2.0**-20,
        "support_coordinate_jitter": 2.0**-18,
    }


def _scalar_blocks(first: float, second: float) -> dict[str, object]:
    return rbf_kernel_blocks_exact_fp32(
        [[first]],
        [[second]],
        first_directions=[[1.0]],
        second_directions=[[1.0]],
        lengthscale=1.0,
        outputscale=1.0,
    )


def test_analytic_rbf_blocks_have_exact_sign_shape_and_dyadic_source() -> None:
    result = rbf_kernel_blocks_exact_fp32(
        [[0.0]],
        [[0.5]],
        first_directions=[[1.0]],
        second_directions=[[1.0]],
        lengthscale=1.0,
        outputscale=2.0,
    )

    assert result["source_quantization"] == {
        "dtype": "float32",
        "exact_dyadic_decode": True,
        "longdouble_allowed": False,
        "accepted_materialized_scalars": ("binary64-or-lower float or integer <= 2**53"),
    }
    assert result["source_inputs"]["second_points_float32_hex"] == [["0x3f000000"]]
    assert result["blocks"]["Kff"]["shape"] == [1, 1]
    assert result["blocks"]["Kfg"]["shape"] == [1, 1]
    assert result["blocks"]["Kgf"]["shape"] == [1, 1]
    assert result["blocks"]["Kgg"]["shape"] == [1, 1]

    with mp.workprec(220):
        covariance = 2 * mp.exp(-mp.mpf(1) / 8)
        kff = _mp_matrix(result["blocks"]["Kff"])[0][0]
        kfg = _mp_matrix(result["blocks"]["Kfg"])[0][0]
        kgf = _mp_matrix(result["blocks"]["Kgf"])[0][0]
        kgg = _mp_matrix(result["blocks"]["Kgg"])[0][0]
        assert abs(kff - covariance) < mp.mpf("1e-45")
        assert abs(kfg - (-covariance / 2)) < mp.mpf("1e-45")
        assert abs(kgf - covariance / 2) < mp.mpf("1e-45")
        assert abs(kgg - 3 * covariance / 4) < mp.mpf("1e-45")


def test_non_dyadic_python_float_is_decoded_from_its_binary32_bits() -> None:
    result = rbf_kernel_blocks_exact_fp32(
        [[0.1]],
        [[0.0]],
        first_directions=[[1.0]],
        second_directions=[[1.0]],
        lengthscale=1.0,
        outputscale=1.0,
    )

    assert result["source_inputs"]["first_points_float32_hex"] == [["0x3dcccccd"]]
    with mp.workprec(220):
        exact_binary32_tenth = mp.mpf(13_421_773) * mp.ldexp(1, -27)
        expected = mp.exp(-(exact_binary32_tenth**2) / 2)
        observed = _mp_matrix(result["blocks"]["Kff"])[0][0]
        assert abs(observed - expected) < mp.mpf("1e-45")


def test_signed_zero_and_minimum_subnormal_source_words_are_preserved() -> None:
    result = rbf_kernel_blocks_exact_fp32(
        [[-0.0, 2.0**-149]],
        [[0.0, 0.0]],
        first_directions=[[1.0], [0.0]],
        second_directions=[[1.0], [0.0]],
        lengthscale=[1.0, 1.0],
        outputscale=1.0,
    )

    assert result["source_inputs"]["first_points_float32_hex"] == [["0x80000000", "0x00000001"]]


def test_gradient_blocks_match_finite_differences_and_joint_symmetry() -> None:
    first = -0.5
    second = 0.25
    step = 2.0**-11
    center = _scalar_blocks(first, second)
    second_plus = _scalar_blocks(first, second + step)
    second_minus = _scalar_blocks(first, second - step)
    first_plus = _scalar_blocks(first + step, second)
    first_minus = _scalar_blocks(first - step, second)

    with mp.workprec(220):
        kfg = _mp_matrix(center["blocks"]["Kfg"])[0][0]
        kgf = _mp_matrix(center["blocks"]["Kgf"])[0][0]
        second_difference = (
            _mp_matrix(second_plus["blocks"]["Kff"])[0][0]
            - _mp_matrix(second_minus["blocks"]["Kff"])[0][0]
        ) / (2 * mp.mpf(step))
        first_difference = (
            _mp_matrix(first_plus["blocks"]["Kff"])[0][0]
            - _mp_matrix(first_minus["blocks"]["Kff"])[0][0]
        ) / (2 * mp.mpf(step))
        assert abs(kfg - second_difference) < mp.mpf("6e-8")
        assert abs(kgf - first_difference) < mp.mpf("6e-8")
        assert kfg == -kgf

    symmetric = rbf_kernel_blocks_exact_fp32(
        [[-0.5, 0.25], [0.75, -1.0]],
        [[-0.5, 0.25], [0.75, -1.0]],
        first_directions=[[1.0, 0.0], [0.0, 1.0]],
        second_directions=[[1.0, 0.0], [0.0, 1.0]],
        lengthscale=[1.0, 2.0],
        outputscale=1.5,
    )
    with mp.workprec(220):
        kff = _mp_matrix(symmetric["blocks"]["Kff"])
        kfg = _mp_matrix(symmetric["blocks"]["Kfg"])
        kgf = _mp_matrix(symmetric["blocks"]["Kgf"])
        kgg = _mp_matrix(symmetric["blocks"]["Kgg"])
        assert all(kff[i][j] == kff[j][i] for i in range(2) for j in range(2))
        assert all(kgf[i][j] == kfg[j][i] for i in range(4) for j in range(2))
        assert all(kgg[i][j] == kgg[j][i] for i in range(4) for j in range(4))


def test_reference_is_json_safe_positive_and_stable_from_160_to_256_bits() -> None:
    artifact = build_high_precision_rbf_reference(**_fixture_kwargs())

    assert artifact["schema_version"] == "f02b_high_precision_rbf_fixture_v1"
    assert artifact["status"] == "complete"
    assert artifact["precision"]["primary_bits"] == 160
    assert artifact["precision"]["verification_bits"] == 256
    assert artifact["precision"]["stabilization_required_bits"] == 100
    assert artifact["precision"]["stabilized"] is True
    assert artifact["certificate_scope"] == {
        "conditional_given_caller_supplied_support": True,
        "support_selection_audited": False,
        "rank_rule_audited": False,
        "support64_float64_svd_geometry_audited": False,
        "support_geometry_representation": (
            "caller-supplied basis and coordinates independently quantized to float32"
        ),
        "required_companion_evidence": (
            "F02b N0 rank and projector evidence for the identical support geometry"
        ),
    }
    assert artifact["source_quantization"]["exact_dyadic_decode"] is True
    assert artifact["source_quantization"]["longdouble_allowed"] is False
    assert artifact["dimensions"] == {
        "condition_count": 2,
        "ambient_dimension": 1,
        "support_rank": 1,
        "observation_count": 4,
    }
    assert mp.mpf(artifact["moments"]["raw_latent_variance"]) > 0
    assert artifact["checks"]["raw_latent_variance_positive"] is True
    assert artifact["projected_system"]["observation_covariance"]["shape"] == [4, 4]
    assert artifact["projected_system"]["observations"]["shape"] == [4]
    serialized = json.dumps(artifact, allow_nan=False, sort_keys=True)
    assert json.loads(serialized) == artifact
    assert build_high_precision_rbf_reference(**_fixture_kwargs()) == artifact


def test_large_coordinates_cannot_relax_basis_orthonormality() -> None:
    kwargs = {
        **_fixture_kwargs(),
        "x_condition": [[50_000_000.0], [-50_000_000.0]],
        "support_basis": [[2.0]],
        "support_coordinates": [[100_000_000.0], [-100_000_000.0]],
    }
    with pytest.raises(HighPrecisionFixtureError, match="not orthonormal"):
        build_high_precision_rbf_reference(**kwargs)


def test_small_moment_stabilization_requires_true_relative_bits() -> None:
    with pytest.raises(HighPrecisionFixtureError, match="did not stabilize"):
        build_high_precision_rbf_reference(
            x_condition=[[2.0**-40]],
            value_condition=[0.2],
            gradient_condition=[[0.3]],
            x_target=[0.0],
            support_basis=[[1.0]],
            support_coordinates=[[2.0**-40]],
            lengthscale=1.0,
            outputscale=1.0,
            value_noise_variance=2.0**-149,
            gradient_noise_variance=0.0,
            gradient_noise_model="iid",
            function_jitter=0.0,
            support_coordinate_jitter=2.0**-149,
        )


def test_wider_than_binary64_numeric_sources_are_rejected_before_double_rounding() -> None:
    halfway_perturbation = Fraction(1, 1) + Fraction(1, 2**24) + Fraction(1, 2**54)
    with pytest.raises(HighPrecisionFixtureError, match="binary64-or-lower"):
        rbf_kernel_blocks_exact_fp32(
            [[halfway_perturbation]],
            [[0.0]],
            first_directions=[[1.0]],
            second_directions=[[1.0]],
            lengthscale=1.0,
            outputscale=1.0,
        )
    with pytest.raises(HighPrecisionFixtureError, match="exact binary64 range"):
        rbf_kernel_blocks_exact_fp32(
            [[2**53 + 1]],
            [[0.0]],
            first_directions=[[1.0]],
            second_directions=[[1.0]],
            lengthscale=1.0,
            outputscale=1.0,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"outputscale": -1.0}, "strictly positive"),
        ({"support_coordinate_jitter": 0.0}, "strictly positive"),
        ({"support_coordinates": [[-2.0], [2.0]]}, "do not match"),
        ({"gradient_condition": [[0.125], [float("nan")]]}, "remain finite"),
        ({"primary_precision_bits": 159}, "at least 160"),
        ({"verification_precision_bits": 255}, "at least 256"),
    ],
)
def test_reference_fails_closed_on_invalid_contract(
    change: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(HighPrecisionFixtureError, match=message):
        build_high_precision_rbf_reference(**{**_fixture_kwargs(), **change})


def test_numpy_longdouble_is_explicitly_rejected() -> None:
    with pytest.raises(HighPrecisionFixtureError, match="numpy.longdouble"):
        build_high_precision_rbf_reference(
            **{**_fixture_kwargs(), "outputscale": np.longdouble(1.0)}
        )


def test_final_moments_match_float64_dense_support_oracle_without_reusing_it() -> None:
    kwargs = _fixture_kwargs()
    artifact = build_high_precision_rbf_reference(**kwargs)
    x = torch.tensor(kwargs["x_condition"], dtype=torch.float64)
    values = torch.tensor(kwargs["value_condition"], dtype=torch.float64)
    gradients = torch.tensor(kwargs["gradient_condition"], dtype=torch.float64)
    target = torch.tensor([kwargs["x_target"]], dtype=torch.float64)
    oracle = predict_local_dense_support(
        x,
        values,
        gradients,
        target,
        lengthscale=torch.tensor([kwargs["lengthscale"]], dtype=torch.float64),
        outputscale=kwargs["outputscale"],
        value_noise_variance=kwargs["value_noise_variance"],
        gradient_noise_variance=kwargs["gradient_noise_variance"],
        kernel="rbf",
        gradient_noise_model=kwargs["gradient_noise_model"],
        function_jitter=kwargs["function_jitter"],
        support_coordinate_jitter=kwargs["support_coordinate_jitter"],
    )

    assert float(artifact["moments"]["mean"]) == pytest.approx(
        float(oracle.mean),
        rel=0.0,
        abs=2e-15,
    )
    assert float(artifact["moments"]["raw_latent_variance"]) == pytest.approx(
        float(oracle.latent_variance),
        rel=0.0,
        abs=2e-15,
    )

    source = inspect.getsource(f02b_high_precision_fixture)
    assert "from gp.orbit" not in source
    assert "import gp.orbit" not in source
    assert "f02_support_dense_oracle" not in source
    assert "open(" not in source


@pytest.mark.parametrize("gradient_noise_model", ["iid", "scaled"])
def test_multidimensional_ard_projected_system_matches_dense_support_final_output(
    gradient_noise_model: str,
) -> None:
    x = torch.tensor(
        [[-1.0, 0.25], [0.5, -0.75], [1.25, 0.5]],
        dtype=torch.float32,
    )
    values = torch.tensor([0.2, -0.4, 0.7], dtype=torch.float32)
    gradients = torch.tensor(
        [[0.1, -0.3], [0.4, 0.2], [-0.2, 0.6]],
        dtype=torch.float32,
    )
    target = torch.tensor([[0.125, -0.125]], dtype=torch.float32)
    lengthscale = torch.tensor([0.7, 1.3], dtype=torch.float32)
    oracle = predict_local_dense_support(
        x,
        values,
        gradients,
        target,
        lengthscale=lengthscale,
        outputscale=1.2,
        value_noise_variance=1e-3,
        gradient_noise_variance=2e-3,
        kernel="rbf",
        gradient_noise_model=gradient_noise_model,
        function_jitter=1e-7,
        support_coordinate_jitter=1e-6,
    )
    artifact = build_high_precision_rbf_reference(
        x,
        values,
        gradients,
        target[0],
        support_basis=oracle.support_basis,
        support_coordinates=oracle.support_coordinates,
        lengthscale=lengthscale,
        outputscale=1.2,
        value_noise_variance=1e-3,
        gradient_noise_variance=2e-3,
        gradient_noise_model=gradient_noise_model,
        function_jitter=oracle.diagnostics.function_jitter_used,
        support_coordinate_jitter=1e-6,
    )

    assert artifact["dimensions"]["ambient_dimension"] == 2
    assert artifact["dimensions"]["support_rank"] == 2
    assert float(artifact["moments"]["mean"]) == pytest.approx(
        float(oracle.mean),
        rel=0.0,
        abs=2e-14,
    )
    assert float(artifact["moments"]["raw_latent_variance"]) == pytest.approx(
        float(oracle.latent_variance),
        rel=0.0,
        abs=2e-14,
    )
