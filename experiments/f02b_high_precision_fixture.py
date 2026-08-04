"""Independent high-precision RBF reference for small F02b fixtures.

This module deliberately imports neither ORBIT nor the float64 support oracle.
It quantizes every numeric input to IEEE-754 binary32, decodes those bits as an
exact dyadic number, assembles analytic value/gradient RBF blocks with mpmath,
and solves one explicit dense joint covariance system.  It never reads data,
labels, configuration files, or other artifacts.

The caller supplies a fixed orthonormal basis in standardized coordinates and
the corresponding coordinates of ``(x_condition - x_target) / lengthscale``.
This keeps support selection outside the reference calculation.  The fixture
checks the supplied basis and coordinates, but never calls a production SVD.
Consequently it certifies the conditional calculation *given that support*,
not the support-selection rule itself; every real-data use requires separate
F02b N0 rank/projector evidence for the same geometry.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any

import mpmath as mp

SCHEMA_VERSION = "f02b_high_precision_rbf_fixture_v1"
MINIMUM_PRIMARY_PRECISION_BITS = 160
MINIMUM_VERIFICATION_PRECISION_BITS = 256
STABILIZATION_BITS = 100
SOURCE_EPSILON_EXPONENT = -23
SOURCE_CONSISTENCY_MULTIPLIER = 64


class HighPrecisionFixtureError(RuntimeError):
    """Raised when the independent reference cannot certify its result."""


@dataclass(frozen=True)
class _QuantizedProblem:
    x_condition: tuple[tuple[int, ...], ...]
    value_condition: tuple[int, ...]
    gradient_condition: tuple[tuple[int, ...], ...]
    x_target: tuple[int, ...]
    support_basis: tuple[tuple[int, ...], ...]
    support_coordinates: tuple[tuple[int, ...], ...]
    lengthscale: tuple[int, ...]
    outputscale: int
    value_noise_variance: int
    gradient_noise_variance: int
    function_jitter: int
    support_coordinate_jitter: int
    gradient_noise_model: str


@dataclass(frozen=True)
class _KernelBlocks:
    kff: Any
    kfg: Any
    kgf: Any
    kgg: Any


@dataclass(frozen=True)
class _Computation:
    blocks: _KernelBlocks
    support_noise_metric: Any
    support_jitter_metric: Any
    projected_gradient_observations: Any
    observation_covariance: Any
    observations: Any
    target_cross_covariance: Any
    mean: Any
    raw_latent_variance: Any
    basis_orthonormality_error: Any
    coordinate_consistency_error: Any
    basis_consistency_tolerance: Any
    coordinate_consistency_tolerance: Any
    kernel_transpose_error: Any
    kernel_symmetry_error: Any
    solve_relative_residual: Any


def _reject_numpy_longdouble(value: Any, label: str) -> None:
    value_type = type(value)
    module = value_type.__module__.split(".", maxsplit=1)[0]
    name = value_type.__name__.lower()
    if module == "numpy" and ("longdouble" in name or name in {"float96", "float128"}):
        raise HighPrecisionFixtureError(f"{label} may not use numpy.longdouble")

    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return
    dtype_module = type(dtype).__module__.split(".", maxsplit=1)[0]
    if dtype_module != "numpy" or getattr(dtype, "kind", None) != "f":
        return
    try:
        itemsize = int(dtype.itemsize)
    except (TypeError, ValueError):
        return
    if itemsize > 8:
        raise HighPrecisionFixtureError(f"{label} may not use numpy.longdouble")


def _materialize(value: Any, label: str) -> Any:
    _reject_numpy_longdouble(value, label)
    materialized = value
    detach = getattr(materialized, "detach", None)
    if callable(detach):
        materialized = detach()
    cpu = getattr(materialized, "cpu", None)
    if callable(cpu):
        materialized = cpu()
    tolist = getattr(materialized, "tolist", None)
    if callable(tolist) and not isinstance(materialized, (list, tuple)):
        materialized = tolist()
    return materialized


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _binary32_word(value: Any, label: str) -> int:
    value = _materialize(value, label)
    if _is_sequence(value) or isinstance(value, (bool, complex, str, bytes)):
        raise HighPrecisionFixtureError(f"{label} must be a real scalar")
    if type(value) not in {int, float}:
        raise HighPrecisionFixtureError(
            f"{label} must materialize as a binary64-or-lower float or an exact small integer"
        )
    if type(value) is int and abs(value) > 2**53:
        raise HighPrecisionFixtureError(
            f"{label} integer exceeds the exact binary64 range required before binary32 quantization"
        )
    try:
        numeric = float(value)
        packed = struct.pack(">f", numeric)
    except (OverflowError, TypeError, ValueError, struct.error) as error:
        raise HighPrecisionFixtureError(f"{label} cannot be represented as binary32") from error
    word = struct.unpack(">I", packed)[0]
    if (word >> 23) & 0xFF == 0xFF:
        raise HighPrecisionFixtureError(f"{label} must remain finite after binary32 quantization")
    return word


def _binary32_python_value(word: int) -> float:
    return struct.unpack(">f", struct.pack(">I", word))[0]


def _mp_from_binary32(word: int) -> Any:
    sign = -1 if word >> 31 else 1
    exponent_bits = (word >> 23) & 0xFF
    fraction = word & 0x7FFFFF
    if exponent_bits == 0:
        if fraction == 0:
            return mp.mpf("0")
        significand = fraction
        exponent = -149
    else:
        significand = (1 << 23) | fraction
        exponent = exponent_bits - 127 - 23
    return sign * mp.ldexp(mp.mpf(significand), exponent)


def _vector_words(value: Any, label: str) -> tuple[int, ...]:
    value = _materialize(value, label)
    if not _is_sequence(value) or not value:
        raise HighPrecisionFixtureError(f"{label} must be a non-empty vector")
    if any(_is_sequence(item) for item in value):
        raise HighPrecisionFixtureError(f"{label} must be one-dimensional")
    return tuple(_binary32_word(item, f"{label}[{index}]") for index, item in enumerate(value))


def _matrix_words(value: Any, label: str) -> tuple[tuple[int, ...], ...]:
    value = _materialize(value, label)
    if not _is_sequence(value) or not value:
        raise HighPrecisionFixtureError(f"{label} must be a non-empty matrix")
    rows: list[tuple[int, ...]] = []
    width: int | None = None
    for row_index, raw_row in enumerate(value):
        raw_row = _materialize(raw_row, f"{label}[{row_index}]")
        if not _is_sequence(raw_row) or not raw_row:
            raise HighPrecisionFixtureError(f"{label} must contain non-empty rows")
        if any(_is_sequence(item) for item in raw_row):
            raise HighPrecisionFixtureError(f"{label} must be two-dimensional")
        row = tuple(
            _binary32_word(item, f"{label}[{row_index}][{column_index}]")
            for column_index, item in enumerate(raw_row)
        )
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise HighPrecisionFixtureError(f"{label} must not be ragged")
        rows.append(row)
    return tuple(rows)


def _lengthscale_words(value: Any, dimension: int) -> tuple[int, ...]:
    materialized = _materialize(value, "lengthscale")
    if _is_sequence(materialized):
        words = _vector_words(materialized, "lengthscale")
        if len(words) == 1:
            words = words * dimension
        elif len(words) != dimension:
            raise HighPrecisionFixtureError(
                f"lengthscale must have length 1 or ambient dimension {dimension}"
            )
    else:
        words = (_binary32_word(materialized, "lengthscale"),) * dimension
    if any(_binary32_python_value(word) <= 0.0 for word in words):
        raise HighPrecisionFixtureError("lengthscale must be finite and strictly positive")
    return words


def _require_scalar_sign(word: int, label: str, *, strictly_positive: bool) -> None:
    value = _binary32_python_value(word)
    valid = value > 0.0 if strictly_positive else value >= 0.0
    if not valid:
        qualifier = "strictly positive" if strictly_positive else "non-negative"
        raise HighPrecisionFixtureError(f"{label} must be finite and {qualifier}")


def _prepare_problem(
    x_condition: Any,
    value_condition: Any,
    gradient_condition: Any,
    x_target: Any,
    *,
    support_basis: Any,
    support_coordinates: Any,
    lengthscale: Any,
    outputscale: Any,
    value_noise_variance: Any,
    gradient_noise_variance: Any,
    gradient_noise_model: str,
    function_jitter: Any,
    support_coordinate_jitter: Any,
) -> _QuantizedProblem:
    x_words = _matrix_words(x_condition, "x_condition")
    m = len(x_words)
    dimension = len(x_words[0])
    values_words = _vector_words(value_condition, "value_condition")
    gradients_words = _matrix_words(gradient_condition, "gradient_condition")
    target_words = _vector_words(x_target, "x_target")
    basis_words = _matrix_words(support_basis, "support_basis")
    coordinate_words = _matrix_words(support_coordinates, "support_coordinates")

    if len(values_words) != m:
        raise HighPrecisionFixtureError(f"value_condition must have shape ({m},)")
    if len(gradients_words) != m or any(len(row) != dimension for row in gradients_words):
        raise HighPrecisionFixtureError(f"gradient_condition must have shape ({m}, {dimension})")
    if len(target_words) != dimension:
        raise HighPrecisionFixtureError(f"x_target must have shape ({dimension},)")
    if len(basis_words) != dimension:
        raise HighPrecisionFixtureError(
            f"support_basis must have ambient dimension {dimension} rows"
        )
    support_rank = len(basis_words[0])
    if support_rank < 1 or any(len(row) != support_rank for row in basis_words):
        raise HighPrecisionFixtureError("support_basis must have a positive, fixed column count")
    if support_rank > min(m, dimension):
        raise HighPrecisionFixtureError("support rank may not exceed min(m, ambient dimension)")
    if len(coordinate_words) != m or any(len(row) != support_rank for row in coordinate_words):
        raise HighPrecisionFixtureError(
            f"support_coordinates must have shape ({m}, {support_rank})"
        )
    if gradient_noise_model not in {"iid", "scaled"}:
        raise HighPrecisionFixtureError("gradient_noise_model must be 'iid' or 'scaled'")

    outputscale_word = _binary32_word(outputscale, "outputscale")
    value_noise_word = _binary32_word(value_noise_variance, "value_noise_variance")
    gradient_noise_word = _binary32_word(
        gradient_noise_variance,
        "gradient_noise_variance",
    )
    function_jitter_word = _binary32_word(function_jitter, "function_jitter")
    support_jitter_word = _binary32_word(
        support_coordinate_jitter,
        "support_coordinate_jitter",
    )
    _require_scalar_sign(outputscale_word, "outputscale", strictly_positive=True)
    _require_scalar_sign(value_noise_word, "value_noise_variance", strictly_positive=False)
    _require_scalar_sign(
        gradient_noise_word,
        "gradient_noise_variance",
        strictly_positive=False,
    )
    _require_scalar_sign(function_jitter_word, "function_jitter", strictly_positive=False)
    _require_scalar_sign(
        support_jitter_word,
        "support_coordinate_jitter",
        strictly_positive=True,
    )

    return _QuantizedProblem(
        x_condition=x_words,
        value_condition=values_words,
        gradient_condition=gradients_words,
        x_target=target_words,
        support_basis=basis_words,
        support_coordinates=coordinate_words,
        lengthscale=_lengthscale_words(lengthscale, dimension),
        outputscale=outputscale_word,
        value_noise_variance=value_noise_word,
        gradient_noise_variance=gradient_noise_word,
        function_jitter=function_jitter_word,
        support_coordinate_jitter=support_jitter_word,
        gradient_noise_model=gradient_noise_model,
    )


def _mp_matrix(words: tuple[tuple[int, ...], ...]) -> Any:
    return mp.matrix([[_mp_from_binary32(word) for word in row] for row in words])


def _mp_vector(words: tuple[int, ...]) -> Any:
    return mp.matrix([_mp_from_binary32(word) for word in words])


def _max_abs_matrix(matrix: Any) -> Any:
    maximum = mp.mpf("0")
    for row in range(matrix.rows):
        for column in range(matrix.cols):
            maximum = max(maximum, abs(matrix[row, column]))
    return maximum


def _max_abs_vector(vector: Any) -> Any:
    maximum = mp.mpf("0")
    for index in range(vector.rows):
        maximum = max(maximum, abs(vector[index]))
    return maximum


def _assert_finite_matrix(matrix: Any, label: str) -> None:
    for row in range(matrix.rows):
        for column in range(matrix.cols):
            if not mp.isfinite(matrix[row, column]):
                raise HighPrecisionFixtureError(f"{label} contains a nonfinite value")


def _kernel_blocks_mp(
    first_points: Any,
    second_points: Any,
    first_directions: Any,
    second_directions: Any,
    lengthscale: Any,
    outputscale: Any,
) -> _KernelBlocks:
    first_count, dimension = first_points.rows, first_points.cols
    second_count = second_points.rows
    first_rank = first_directions.cols
    second_rank = second_directions.cols
    kff = mp.matrix(first_count, second_count)
    kfg = mp.matrix(first_count, second_count * second_rank)
    kgf = mp.matrix(first_count * first_rank, second_count)
    kgg = mp.matrix(first_count * first_rank, second_count * second_rank)
    inverse_lengthscale2 = [1 / (lengthscale[index] ** 2) for index in range(dimension)]

    for first_index in range(first_count):
        for second_index in range(second_count):
            difference = [
                first_points[first_index, coordinate] - second_points[second_index, coordinate]
                for coordinate in range(dimension)
            ]
            distance2 = mp.fsum(
                difference[coordinate] ** 2 * inverse_lengthscale2[coordinate]
                for coordinate in range(dimension)
            )
            covariance = outputscale * mp.exp(-distance2 / 2)
            kff[first_index, second_index] = covariance
            scaled_gradient = [
                difference[coordinate] * inverse_lengthscale2[coordinate]
                for coordinate in range(dimension)
            ]

            first_projections = [
                mp.fsum(
                    first_directions[coordinate, direction] * scaled_gradient[coordinate]
                    for coordinate in range(dimension)
                )
                for direction in range(first_rank)
            ]
            second_projections = [
                mp.fsum(
                    second_directions[coordinate, direction] * scaled_gradient[coordinate]
                    for coordinate in range(dimension)
                )
                for direction in range(second_rank)
            ]
            for second_direction in range(second_rank):
                # d/d(second) k(first, second) has the sign of first-second.
                kfg[first_index, second_index * second_rank + second_direction] = (
                    covariance * second_projections[second_direction]
                )
            for first_direction in range(first_rank):
                # d/d(first) k(first, second) has the opposite sign.
                kgf[first_index * first_rank + first_direction, second_index] = (
                    -covariance * first_projections[first_direction]
                )
                for second_direction in range(second_rank):
                    diagonal_term = mp.fsum(
                        first_directions[coordinate, first_direction]
                        * inverse_lengthscale2[coordinate]
                        * second_directions[coordinate, second_direction]
                        for coordinate in range(dimension)
                    )
                    mixed = diagonal_term - (
                        first_projections[first_direction] * second_projections[second_direction]
                    )
                    kgg[
                        first_index * first_rank + first_direction,
                        second_index * second_rank + second_direction,
                    ] = covariance * mixed

    for label, matrix in (("Kff", kff), ("Kfg", kfg), ("Kgf", kgf), ("Kgg", kgg)):
        _assert_finite_matrix(matrix, label)
    return _KernelBlocks(kff=kff, kfg=kfg, kgf=kgf, kgg=kgg)


def _matrix_difference_max(first: Any, second: Any) -> Any:
    if first.rows != second.rows or first.cols != second.cols:
        raise HighPrecisionFixtureError("internal matrix shapes changed between precision runs")
    maximum = mp.mpf("0")
    for row in range(first.rows):
        for column in range(first.cols):
            maximum = max(maximum, abs(first[row, column] - second[row, column]))
    return maximum


def _solve_spd(matrix: Any, rhs: Any, *, precision_bits: int, label: str) -> tuple[Any, Any]:
    try:
        mp.cholesky(matrix)
        solution = mp.cholesky_solve(matrix, rhs)
    except (ValueError, ZeroDivisionError) as error:
        raise HighPrecisionFixtureError(f"{label} is not positive definite") from error
    residual = matrix * solution - rhs
    relative_residual = _max_abs_vector(residual) / max(mp.mpf("1"), _max_abs_vector(rhs))
    tolerance = mp.ldexp(mp.mpf("1"), -max(64, precision_bits // 2))
    if not mp.isfinite(relative_residual) or relative_residual > tolerance:
        raise HighPrecisionFixtureError(
            f"{label} solve residual exceeds the precision-dependent tolerance"
        )
    return solution, relative_residual


def _spd_inverse(matrix: Any, *, precision_bits: int, label: str) -> Any:
    inverse = mp.matrix(matrix.rows, matrix.cols)
    for column in range(matrix.cols):
        unit = mp.matrix(matrix.rows, 1)
        unit[column] = 1
        solution, _ = _solve_spd(
            matrix,
            unit,
            precision_bits=precision_bits,
            label=label,
        )
        for row in range(matrix.rows):
            inverse[row, column] = solution[row]
    return (inverse + inverse.T) / 2


def _compute_reference(problem: _QuantizedProblem, precision_bits: int) -> _Computation:
    with mp.workprec(precision_bits):
        x_condition = _mp_matrix(problem.x_condition)
        values = _mp_vector(problem.value_condition)
        gradients = _mp_matrix(problem.gradient_condition)
        target = _mp_vector(problem.x_target)
        basis = _mp_matrix(problem.support_basis)
        coordinates = _mp_matrix(problem.support_coordinates)
        lengthscale = _mp_vector(problem.lengthscale)
        outputscale = _mp_from_binary32(problem.outputscale)
        value_noise = _mp_from_binary32(problem.value_noise_variance)
        gradient_noise = _mp_from_binary32(problem.gradient_noise_variance)
        function_jitter = _mp_from_binary32(problem.function_jitter)
        support_jitter = _mp_from_binary32(problem.support_coordinate_jitter)
        m, dimension, support_rank = x_condition.rows, x_condition.cols, basis.cols

        identity_rank = mp.eye(support_rank)
        basis_gram = basis.T * basis
        basis_error = _matrix_difference_max(basis_gram, identity_rank)
        expected_coordinates = mp.matrix(m, support_rank)
        for row in range(m):
            for direction in range(support_rank):
                expected_coordinates[row, direction] = mp.fsum(
                    ((x_condition[row, coordinate] - target[coordinate]) / lengthscale[coordinate])
                    * basis[coordinate, direction]
                    for coordinate in range(dimension)
                )
        coordinate_error = _matrix_difference_max(coordinates, expected_coordinates)
        source_epsilon = mp.ldexp(mp.mpf("1"), SOURCE_EPSILON_EXPONENT)
        basis_consistency_scale = max(
            mp.mpf("1"),
            _max_abs_matrix(basis_gram),
        )
        coordinate_consistency_scale = max(
            mp.mpf("1"),
            _max_abs_matrix(coordinates),
            _max_abs_matrix(expected_coordinates),
        )
        basis_consistency_tolerance = (
            SOURCE_CONSISTENCY_MULTIPLIER * source_epsilon * basis_consistency_scale
        )
        coordinate_consistency_tolerance = (
            SOURCE_CONSISTENCY_MULTIPLIER * source_epsilon * coordinate_consistency_scale
        )
        if basis_error > basis_consistency_tolerance:
            raise HighPrecisionFixtureError(
                "support_basis is not orthonormal within the fixed source-fp32 tolerance"
            )
        if coordinate_error > coordinate_consistency_tolerance:
            raise HighPrecisionFixtureError(
                "support_coordinates do not match the fixed basis and standardized offsets"
            )

        # A standardized basis direction B corresponds to the physical gradient
        # direction diag(lengthscale) B.
        physical_directions = mp.matrix(dimension, support_rank)
        for coordinate in range(dimension):
            for direction in range(support_rank):
                physical_directions[coordinate, direction] = (
                    lengthscale[coordinate] * basis[coordinate, direction]
                )

        blocks = _kernel_blocks_mp(
            x_condition,
            x_condition,
            physical_directions,
            physical_directions,
            lengthscale,
            outputscale,
        )
        transpose_error = _matrix_difference_max(blocks.kgf, blocks.kfg.T)
        kff_symmetry_error = _matrix_difference_max(blocks.kff, blocks.kff.T)
        kgg_symmetry_error = _matrix_difference_max(blocks.kgg, blocks.kgg.T)
        kernel_symmetry_error = max(kff_symmetry_error, kgg_symmetry_error)
        kernel_scale = max(
            mp.mpf("1"),
            _max_abs_matrix(blocks.kff),
            _max_abs_matrix(blocks.kgg),
        )
        kernel_tolerance = mp.ldexp(kernel_scale, -precision_bits + 16)
        if transpose_error > kernel_tolerance or kernel_symmetry_error > kernel_tolerance:
            raise HighPrecisionFixtureError("analytic RBF blocks failed symmetry checks")

        support_noise_metric = mp.matrix(support_rank, support_rank)
        for first_direction in range(support_rank):
            for second_direction in range(support_rank):
                support_noise_metric[first_direction, second_direction] = gradient_noise * mp.fsum(
                    physical_directions[coordinate, first_direction]
                    * (1 if problem.gradient_noise_model == "iid" else lengthscale[coordinate] ** 2)
                    * physical_directions[coordinate, second_direction]
                    for coordinate in range(dimension)
                )

        coordinate_gram = coordinates.T * coordinates
        inverse_coordinate_gram = _spd_inverse(
            coordinate_gram,
            precision_bits=precision_bits,
            label="support coordinate Gram matrix",
        )
        support_jitter_metric = support_jitter * inverse_coordinate_gram

        kff_observed = mp.matrix(blocks.kff)
        kgg_observed = mp.matrix(blocks.kgg)
        for row in range(m):
            kff_observed[row, row] += value_noise + function_jitter
            for first_direction in range(support_rank):
                for second_direction in range(support_rank):
                    kgg_observed[
                        row * support_rank + first_direction,
                        row * support_rank + second_direction,
                    ] += (
                        support_noise_metric[first_direction, second_direction]
                        + support_jitter_metric[first_direction, second_direction]
                    )

        projected_gradients = gradients * physical_directions
        observation_count = m + m * support_rank
        observations = mp.matrix(observation_count, 1)
        for row in range(m):
            observations[row] = values[row]
            for direction in range(support_rank):
                observations[m + row * support_rank + direction] = projected_gradients[
                    row,
                    direction,
                ]

        joint = mp.matrix(observation_count, observation_count)
        for row in range(m):
            for column in range(m):
                joint[row, column] = kff_observed[row, column]
        for row in range(m):
            for column in range(m * support_rank):
                joint[row, m + column] = blocks.kfg[row, column]
                joint[m + column, row] = blocks.kgf[column, row]
        for row in range(m * support_rank):
            for column in range(m * support_rank):
                joint[m + row, m + column] = kgg_observed[row, column]
        joint_asymmetry = _matrix_difference_max(joint, joint.T)
        if joint_asymmetry > kernel_tolerance:
            raise HighPrecisionFixtureError("joint observation covariance is not symmetric")
        joint = (joint + joint.T) / 2
        _assert_finite_matrix(joint, "joint observation covariance")

        target_points = mp.matrix(1, dimension)
        for coordinate in range(dimension):
            target_points[0, coordinate] = target[coordinate]
        target_blocks = _kernel_blocks_mp(
            target_points,
            x_condition,
            physical_directions,
            physical_directions,
            lengthscale,
            outputscale,
        )
        target_cross = mp.matrix(observation_count, 1)
        for column in range(m):
            target_cross[column] = target_blocks.kff[0, column]
        for column in range(m * support_rank):
            target_cross[m + column] = target_blocks.kfg[0, column]

        observation_weights, observation_residual = _solve_spd(
            joint,
            observations,
            precision_bits=precision_bits,
            label="joint observation covariance",
        )
        covariance_weights, covariance_residual = _solve_spd(
            joint,
            target_cross,
            precision_bits=precision_bits,
            label="joint observation covariance",
        )
        mean = (target_cross.T * observation_weights)[0]
        raw_latent_variance = outputscale - (target_cross.T * covariance_weights)[0]
        if not mp.isfinite(mean):
            raise HighPrecisionFixtureError("posterior mean is nonfinite")
        if not mp.isfinite(raw_latent_variance) or raw_latent_variance <= 0:
            raise HighPrecisionFixtureError(
                "raw latent posterior variance must be finite and positive"
            )

        return _Computation(
            blocks=blocks,
            support_noise_metric=support_noise_metric,
            support_jitter_metric=support_jitter_metric,
            projected_gradient_observations=projected_gradients,
            observation_covariance=joint,
            observations=observations,
            target_cross_covariance=target_cross,
            mean=mean,
            raw_latent_variance=raw_latent_variance,
            basis_orthonormality_error=basis_error,
            coordinate_consistency_error=coordinate_error,
            basis_consistency_tolerance=basis_consistency_tolerance,
            coordinate_consistency_tolerance=coordinate_consistency_tolerance,
            kernel_transpose_error=transpose_error,
            kernel_symmetry_error=kernel_symmetry_error,
            solve_relative_residual=max(observation_residual, covariance_residual),
        )


def _decimal(value: Any, precision_bits: int) -> str:
    if not mp.isfinite(value):
        raise HighPrecisionFixtureError("cannot serialize a nonfinite high-precision value")
    digits = int(math.ceil(precision_bits * math.log10(2.0))) + 3
    return mp.nstr(value, n=digits, strip_zeros=False)


def _matrix_record(matrix: Any, precision_bits: int) -> dict[str, Any]:
    return {
        "shape": [matrix.rows, matrix.cols],
        "values": [
            [_decimal(matrix[row, column], precision_bits) for column in range(matrix.cols)]
            for row in range(matrix.rows)
        ],
    }


def _vector_record(vector: Any, precision_bits: int) -> dict[str, Any]:
    return {
        "shape": [vector.rows],
        "values": [_decimal(vector[index], precision_bits) for index in range(vector.rows)],
    }


def _hex_word(word: int) -> str:
    return f"0x{word:08x}"


def _hex_matrix(words: tuple[tuple[int, ...], ...]) -> list[list[str]]:
    return [[_hex_word(word) for word in row] for row in words]


def _hex_vector(words: tuple[int, ...]) -> list[str]:
    return [_hex_word(word) for word in words]


def _relative_delta(first: Any, second: Any) -> Any:
    denominator = abs(second)
    if denominator == 0:
        return mp.mpf("0") if first == second else mp.inf
    return abs(first - second) / denominator


def _relative_matrix_delta(first: Any, second: Any) -> Any:
    delta = _matrix_difference_max(first, second)
    denominator = _max_abs_matrix(second)
    if denominator == 0:
        return mp.mpf("0") if delta == 0 else mp.inf
    return delta / denominator


def _validate_precision(primary_precision_bits: int, verification_precision_bits: int) -> None:
    if isinstance(primary_precision_bits, bool) or not isinstance(primary_precision_bits, int):
        raise HighPrecisionFixtureError("primary_precision_bits must be an integer")
    if isinstance(verification_precision_bits, bool) or not isinstance(
        verification_precision_bits,
        int,
    ):
        raise HighPrecisionFixtureError("verification_precision_bits must be an integer")
    if primary_precision_bits < MINIMUM_PRIMARY_PRECISION_BITS:
        raise HighPrecisionFixtureError(
            f"primary_precision_bits must be at least {MINIMUM_PRIMARY_PRECISION_BITS}"
        )
    if verification_precision_bits < MINIMUM_VERIFICATION_PRECISION_BITS:
        raise HighPrecisionFixtureError(
            f"verification_precision_bits must be at least {MINIMUM_VERIFICATION_PRECISION_BITS}"
        )
    if verification_precision_bits <= primary_precision_bits:
        raise HighPrecisionFixtureError("verification precision must exceed primary precision")


def rbf_kernel_blocks_exact_fp32(
    first_points: Any,
    second_points: Any,
    *,
    first_directions: Any,
    second_directions: Any,
    lengthscale: Any,
    outputscale: Any,
    precision_bits: int = MINIMUM_PRIMARY_PRECISION_BITS,
) -> dict[str, Any]:
    """Return independently assembled analytic RBF value/gradient blocks.

    ``Kfg`` differentiates with respect to the second input and ``Kgf`` with
    respect to the first input.  Direction matrices have shape ``(D, r)`` and
    contain physical-coordinate directional derivatives.
    """

    if isinstance(precision_bits, bool) or not isinstance(precision_bits, int):
        raise HighPrecisionFixtureError("precision_bits must be an integer")
    if precision_bits < MINIMUM_PRIMARY_PRECISION_BITS:
        raise HighPrecisionFixtureError(
            f"precision_bits must be at least {MINIMUM_PRIMARY_PRECISION_BITS}"
        )
    first_words = _matrix_words(first_points, "first_points")
    second_words = _matrix_words(second_points, "second_points")
    first_direction_words = _matrix_words(first_directions, "first_directions")
    second_direction_words = _matrix_words(second_directions, "second_directions")
    dimension = len(first_words[0])
    if any(len(row) != dimension for row in second_words):
        raise HighPrecisionFixtureError("first_points and second_points must share dimension")
    if len(first_direction_words) != dimension or len(second_direction_words) != dimension:
        raise HighPrecisionFixtureError("direction matrices must have one row per input dimension")
    first_rank = len(first_direction_words[0])
    second_rank = len(second_direction_words[0])
    if first_rank < 1 or second_rank < 1:
        raise HighPrecisionFixtureError("direction matrices must have at least one column")
    if any(len(row) != first_rank for row in first_direction_words) or any(
        len(row) != second_rank for row in second_direction_words
    ):
        raise HighPrecisionFixtureError("direction matrices must not be ragged")
    lengthscale_words = _lengthscale_words(lengthscale, dimension)
    outputscale_word = _binary32_word(outputscale, "outputscale")
    _require_scalar_sign(outputscale_word, "outputscale", strictly_positive=True)

    with mp.workprec(precision_bits):
        blocks = _kernel_blocks_mp(
            _mp_matrix(first_words),
            _mp_matrix(second_words),
            _mp_matrix(first_direction_words),
            _mp_matrix(second_direction_words),
            _mp_vector(lengthscale_words),
            _mp_from_binary32(outputscale_word),
        )
        return {
            "kernel": "rbf",
            "precision_bits": precision_bits,
            "source_quantization": {
                "dtype": "float32",
                "exact_dyadic_decode": True,
                "longdouble_allowed": False,
                "accepted_materialized_scalars": "binary64-or-lower float or integer <= 2**53",
            },
            "source_inputs": {
                "first_points_float32_hex": _hex_matrix(first_words),
                "second_points_float32_hex": _hex_matrix(second_words),
                "first_directions_float32_hex": _hex_matrix(first_direction_words),
                "second_directions_float32_hex": _hex_matrix(second_direction_words),
                "lengthscale_float32_hex": _hex_vector(lengthscale_words),
                "outputscale_float32_hex": _hex_word(outputscale_word),
            },
            "blocks": {
                "Kff": _matrix_record(blocks.kff, precision_bits),
                "Kfg": _matrix_record(blocks.kfg, precision_bits),
                "Kgf": _matrix_record(blocks.kgf, precision_bits),
                "Kgg": _matrix_record(blocks.kgg, precision_bits),
            },
        }


def build_high_precision_rbf_reference(
    x_condition: Any,
    value_condition: Any,
    gradient_condition: Any,
    x_target: Any,
    *,
    support_basis: Any,
    support_coordinates: Any,
    lengthscale: Any,
    outputscale: Any,
    value_noise_variance: Any,
    gradient_noise_variance: Any,
    gradient_noise_model: str = "iid",
    function_jitter: Any = 0.0,
    support_coordinate_jitter: Any = 1e-8,
    primary_precision_bits: int = MINIMUM_PRIMARY_PRECISION_BITS,
    verification_precision_bits: int = MINIMUM_VERIFICATION_PRECISION_BITS,
) -> dict[str, Any]:
    """Build a strict JSON-safe scalar posterior reference artifact.

    All arithmetic is repeated at the two requested binary precisions.  The
    call fails closed unless kernel assembly and both posterior moments agree
    to at least :data:`STABILIZATION_BITS` relative bits.
    """

    _validate_precision(primary_precision_bits, verification_precision_bits)
    problem = _prepare_problem(
        x_condition,
        value_condition,
        gradient_condition,
        x_target,
        support_basis=support_basis,
        support_coordinates=support_coordinates,
        lengthscale=lengthscale,
        outputscale=outputscale,
        value_noise_variance=value_noise_variance,
        gradient_noise_variance=gradient_noise_variance,
        gradient_noise_model=gradient_noise_model,
        function_jitter=function_jitter,
        support_coordinate_jitter=support_coordinate_jitter,
    )
    primary = _compute_reference(problem, primary_precision_bits)
    verification = _compute_reference(problem, verification_precision_bits)

    with mp.workprec(verification_precision_bits):
        mean_relative_delta = _relative_delta(primary.mean, verification.mean)
        variance_relative_delta = _relative_delta(
            primary.raw_latent_variance,
            verification.raw_latent_variance,
        )
        covariance_relative_delta = _relative_matrix_delta(
            primary.observation_covariance,
            verification.observation_covariance,
        )
        target_relative_delta = _relative_matrix_delta(
            primary.target_cross_covariance,
            verification.target_cross_covariance,
        )
        stabilization_tolerance = mp.ldexp(mp.mpf("1"), -STABILIZATION_BITS)
        stabilization_values = (
            mean_relative_delta,
            variance_relative_delta,
            covariance_relative_delta,
            target_relative_delta,
        )
        stabilized = all(value <= stabilization_tolerance for value in stabilization_values)
        if not stabilized:
            raise HighPrecisionFixtureError(
                "primary- and verification-precision calculations did not stabilize"
            )

        m = len(problem.x_condition)
        dimension = len(problem.x_condition[0])
        support_rank = len(problem.support_basis[0])
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "kernel": "rbf",
            "certificate_scope": {
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
            },
            "dimensions": {
                "condition_count": m,
                "ambient_dimension": dimension,
                "support_rank": support_rank,
                "observation_count": m * (support_rank + 1),
            },
            "precision": {
                "backend": "mpmath",
                "mpmath_version": str(mp.__version__),
                "primary_bits": primary_precision_bits,
                "verification_bits": verification_precision_bits,
                "minimum_primary_bits": MINIMUM_PRIMARY_PRECISION_BITS,
                "minimum_verification_bits": MINIMUM_VERIFICATION_PRECISION_BITS,
                "stabilization_required_bits": STABILIZATION_BITS,
                "stabilization_relative_tolerance": _decimal(
                    stabilization_tolerance,
                    verification_precision_bits,
                ),
                "stabilized": True,
                "mean_relative_delta": _decimal(
                    mean_relative_delta,
                    verification_precision_bits,
                ),
                "raw_latent_variance_relative_delta": _decimal(
                    variance_relative_delta,
                    verification_precision_bits,
                ),
                "joint_covariance_relative_delta": _decimal(
                    covariance_relative_delta,
                    verification_precision_bits,
                ),
                "target_cross_relative_delta": _decimal(
                    target_relative_delta,
                    verification_precision_bits,
                ),
            },
            "source_quantization": {
                "dtype": "float32",
                "encoding": "IEEE-754 binary32",
                "conversion": "exact significand times power-of-two dyadic decode",
                "exact_dyadic_decode": True,
                "longdouble_allowed": False,
                "accepted_materialized_scalars": "binary64-or-lower float or integer <= 2**53",
            },
            "source_inputs": {
                "x_condition_float32_hex": _hex_matrix(problem.x_condition),
                "value_condition_float32_hex": _hex_vector(problem.value_condition),
                "gradient_condition_float32_hex": _hex_matrix(problem.gradient_condition),
                "x_target_float32_hex": _hex_vector(problem.x_target),
                "support_basis_float32_hex": _hex_matrix(problem.support_basis),
                "support_coordinates_float32_hex": _hex_matrix(problem.support_coordinates),
                "lengthscale_float32_hex": _hex_vector(problem.lengthscale),
                "outputscale_float32_hex": _hex_word(problem.outputscale),
                "value_noise_variance_float32_hex": _hex_word(problem.value_noise_variance),
                "gradient_noise_variance_float32_hex": _hex_word(problem.gradient_noise_variance),
                "function_jitter_float32_hex": _hex_word(problem.function_jitter),
                "support_coordinate_jitter_float32_hex": _hex_word(
                    problem.support_coordinate_jitter
                ),
                "gradient_noise_model": problem.gradient_noise_model,
            },
            "analytic_blocks": {
                "Kff": _matrix_record(verification.blocks.kff, verification_precision_bits),
                "Kfg": _matrix_record(verification.blocks.kfg, verification_precision_bits),
                "Kgf": _matrix_record(verification.blocks.kgf, verification_precision_bits),
                "Kgg": _matrix_record(verification.blocks.kgg, verification_precision_bits),
            },
            "projected_system": {
                "support_noise_metric": _matrix_record(
                    verification.support_noise_metric,
                    verification_precision_bits,
                ),
                "support_jitter_metric": _matrix_record(
                    verification.support_jitter_metric,
                    verification_precision_bits,
                ),
                "projected_gradient_observations": _matrix_record(
                    verification.projected_gradient_observations,
                    verification_precision_bits,
                ),
                "observation_covariance": _matrix_record(
                    verification.observation_covariance,
                    verification_precision_bits,
                ),
                "observations": _vector_record(
                    verification.observations,
                    verification_precision_bits,
                ),
                "target_cross_covariance": _vector_record(
                    verification.target_cross_covariance,
                    verification_precision_bits,
                ),
            },
            "moments": {
                "mean": _decimal(verification.mean, verification_precision_bits),
                "raw_latent_variance": _decimal(
                    verification.raw_latent_variance,
                    verification_precision_bits,
                ),
                "primary_mean": _decimal(primary.mean, primary_precision_bits),
                "primary_raw_latent_variance": _decimal(
                    primary.raw_latent_variance,
                    primary_precision_bits,
                ),
            },
            "checks": {
                "basis_orthonormality_error": _decimal(
                    verification.basis_orthonormality_error,
                    verification_precision_bits,
                ),
                "coordinate_consistency_error": _decimal(
                    verification.coordinate_consistency_error,
                    verification_precision_bits,
                ),
                "basis_consistency_tolerance": _decimal(
                    verification.basis_consistency_tolerance,
                    verification_precision_bits,
                ),
                "coordinate_consistency_tolerance": _decimal(
                    verification.coordinate_consistency_tolerance,
                    verification_precision_bits,
                ),
                "kernel_transpose_error": _decimal(
                    verification.kernel_transpose_error,
                    verification_precision_bits,
                ),
                "kernel_symmetry_error": _decimal(
                    verification.kernel_symmetry_error,
                    verification_precision_bits,
                ),
                "solve_relative_residual": _decimal(
                    verification.solve_relative_residual,
                    verification_precision_bits,
                ),
                "raw_latent_variance_positive": True,
                "all_serialized_numerics_finite": True,
            },
        }
