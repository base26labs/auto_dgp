from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

import experiments.f02b_calibration_probe_execution as execution_module
from cluster.f02b_calibration_grid import probe_task_for_index
from experiments.f02_design import EVALUATION_TIME_INDICES
from experiments.f02_internal_models import FrozenTERAParameters, TensorConfirmatorySplit
from experiments.f02b_calibration_probe_core import (
    FP64_ONLY_TOLERANCE_SWEEP,
    PRIMARY_EVALUATION_ROW_COUNT,
    PRODUCTION_TOLERANCE,
    SHARED_TOLERANCE_SWEEP,
    ProbeEvaluationRows,
    build_probe_work_plan,
)
from experiments.f02b_calibration_probe_execution import (
    MATRIX_FREE_ROUNDOFF_QUALIFICATION,
    OPERATOR_ACTION_PROVENANCE,
    REPLAY_ACTION_PROVENANCE,
    RESIDUAL_PROVENANCE,
    LabelFreeEvaluationTensors,
    ProbeExecutionEvidenceError,
    ProbeExecutionInputError,
    RegisteredOrbitArmInputs,
    RegisteredOrbitStrata,
    RegisteredSourceGeometry,
    build_source_orbit_arm_inputs,
    evaluation_rows_to_tensors,
    execute_registered_orbit_target,
    promote_evaluation_to_float64,
    promote_parameters_to_float64,
    promote_registered_orbit_arm_to_float64,
    promote_training_split_to_float64,
    registered_orbit_tolerances,
    scan_registered_source_geometry,
    select_registered_orbit_strata,
)
from gp.orbit import (
    LocalGeometry,
    build_local_geometry,
    build_local_geometry_from_differences,
    build_local_value_system,
    solve_local_value_system,
)


def _source_promoted(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return value.to(dtype=torch.float32).to(dtype=dtype)


def _synthetic_arm(dtype: torch.dtype = torch.float32) -> RegisteredOrbitArmInputs:
    generator = torch.Generator().manual_seed(20260803)
    train_count = 24
    dimension = 12
    latent_rank = 6

    train_latent = torch.randn(train_count, latent_rank, generator=generator)
    eval_latent = torch.randn(PRIMARY_EVALUATION_ROW_COUNT, latent_rank, generator=generator)
    train_x32 = torch.cat(
        [train_latent, torch.zeros(train_count, dimension - latent_rank)],
        dim=1,
    ).to(torch.float32)
    evaluation_x32 = torch.cat(
        [
            eval_latent,
            torch.zeros(PRIMARY_EVALUATION_ROW_COUNT, dimension - latent_rank),
        ],
        dim=1,
    ).to(torch.float32)
    values32 = torch.randn(train_count, generator=generator, dtype=torch.float32)
    gradients32 = torch.randn(
        train_count,
        dimension,
        generator=generator,
        dtype=torch.float32,
    )
    train_sources = torch.arange(train_count, dtype=torch.long)
    train = TensorConfirmatorySplit(
        name="train",
        source_indices=train_sources,
        X=_source_promoted(train_x32, torch.float32),
        value=_source_promoted(values32, torch.float32),
        gradient=_source_promoted(gradients32, torch.float32),
        trajectory_id=torch.arange(train_count, dtype=torch.long),
        time_index=torch.zeros(train_count, dtype=torch.long),
        time_value=_source_promoted(
            torch.arange(train_count, dtype=torch.float32),
            torch.float32,
        ),
    )

    evaluation_sources = torch.arange(
        1000,
        1000 + PRIMARY_EVALUATION_ROW_COUNT,
        dtype=torch.long,
    )
    evaluation = LabelFreeEvaluationTensors(
        source_indices=evaluation_sources,
        X=_source_promoted(evaluation_x32, torch.float32),
        trajectory_id=torch.arange(20, dtype=torch.long).repeat_interleave(5),
        time_index=torch.tensor(EVALUATION_TIME_INDICES, dtype=torch.long).repeat(20),
        time_value=_source_promoted(
            torch.arange(PRIMARY_EVALUATION_ROW_COUNT, dtype=torch.float32),
            torch.float32,
        ),
    )
    work_plan = build_probe_work_plan(probe_task_for_index(45))
    parameters = FrozenTERAParameters(
        lengthscale=torch.tensor([1.0], dtype=torch.float32),
        outputscale=float(torch.tensor(1.25, dtype=torch.float32)),
        sigma_f=float(torch.tensor(1e-3, dtype=torch.float32)),
        sigma_g=float(torch.tensor(2e-3, dtype=torch.float32)),
        kernel="rbf",
    )
    source_arm = build_source_orbit_arm_inputs(
        train32=train,
        evaluation32=evaluation,
        parameters32=parameters,
        work_plan=work_plan,
    )
    if dtype == torch.float32:
        return source_arm
    return promote_registered_orbit_arm_to_float64(source_arm)


def _geometry_and_strata(arm32: RegisteredOrbitArmInputs):
    geometries = tuple(
        scan_registered_source_geometry(arm32, position)
        for position in range(PRIMARY_EVALUATION_ROW_COUNT)
    )
    return geometries, select_registered_orbit_strata(arm32, geometries)


def test_label_free_evaluation_copy_has_no_label_surface() -> None:
    count = PRIMARY_EVALUATION_ROW_COUNT
    source_indices = np.arange(1000, 1000 + count, dtype=np.int64)
    X = np.arange(count * 12, dtype=np.float32).reshape(count, 12)
    trajectory_id = np.arange(20, dtype=np.int64).repeat(5)
    time_index = np.tile(np.array(EVALUATION_TIME_INDICES, dtype=np.int64), 20)
    time_value = np.arange(count, dtype=np.float32)
    rows = ProbeEvaluationRows(
        source_indices=source_indices,
        X=X,
        trajectory_id=trajectory_id,
        time_index=time_index,
        time_value=time_value,
    )

    copied = evaluation_rows_to_tensors(rows, dtype=torch.float32)
    source_indices[0] = 999999
    X[0, 0] = 999999.0

    assert int(copied.source_indices[0]) == 1000
    assert float(copied.X[0, 0]) == 0.0
    assert not hasattr(copied, "value")
    assert not hasattr(copied, "gradient")
    assert not hasattr(copied, "E")
    assert not hasattr(copied, "F")
    with pytest.raises(ProbeExecutionInputError, match="source float32"):
        evaluation_rows_to_tensors(rows, dtype=torch.float64)


def test_registered_tolerances_are_dtype_and_role_exact() -> None:
    assert registered_orbit_tolerances(
        torch.float32,
        include_stratum_sweep=False,
    ) == (PRODUCTION_TOLERANCE,)
    assert registered_orbit_tolerances(
        torch.float64,
        include_stratum_sweep=False,
    ) == (PRODUCTION_TOLERANCE,)
    assert registered_orbit_tolerances(
        torch.float32,
        include_stratum_sweep=True,
    ) == SHARED_TOLERANCE_SWEEP
    assert registered_orbit_tolerances(
        torch.float64,
        include_stratum_sweep=True,
    ) == SHARED_TOLERANCE_SWEEP + FP64_ONLY_TOLERANCE_SWEEP
    with pytest.raises(ProbeExecutionInputError, match="must be bool"):
        registered_orbit_tolerances(
            torch.float32,
            include_stratum_sweep=1,  # type: ignore[arg-type]
        )


def test_label_free_tensor_contract_rejects_noncanonical_row_design() -> None:
    arm = _synthetic_arm()
    changed_times = arm.evaluation.time_index.clone()
    changed_times[0], changed_times[1] = changed_times[1].clone(), changed_times[0].clone()
    with pytest.raises(ProbeExecutionInputError, match="canonical trajectory blocks"):
        replace(arm.evaluation, time_index=changed_times)


def test_direct_svd_geometry_retains_the_evidence_used_for_selection() -> None:
    differences = torch.diag(
        torch.tensor([4.0, 2.0, 1.0, 0.0], dtype=torch.float64)
    )
    epsilon = float(torch.finfo(torch.float32).eps)
    geometry = build_local_geometry_from_differences(
        differences,
        rank_epsilon=epsilon,
    )

    assert geometry.singular_values is not None
    assert geometry.operational_singular_value_cutoff is not None
    assert geometry.native_singular_value_cutoff is not None
    assert geometry.rank_epsilon_used is not None
    torch.testing.assert_close(
        geometry.singular_values,
        torch.tensor([4.0, 2.0, 1.0, 0.0], dtype=torch.float64),
    )
    assert float(geometry.operational_singular_value_cutoff) == pytest.approx(
        4.0 * 4 * epsilon
    )
    assert geometry.rank == 3

    gram_geometry = build_local_geometry(differences.T @ differences)
    assert gram_geometry.singular_values is None
    legacy = LocalGeometry(
        geometry.coordinates,
        geometry.q_to_z,
        geometry.eigenvalues,
        geometry.discarded_eigenvalue_sum,
        geometry.is_exact,
    )
    assert legacy.singular_values is None


def test_orbit_target_builds_once_and_reuses_production_sweep_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm = _synthetic_arm(torch.float32)
    geometries, strata = _geometry_and_strata(arm)
    target_position = strata.selected_target_positions[0]
    build_modes: list[bool] = []
    solve_modes: list[bool] = []
    original_build = execution_module._build_local_value_system_from_registered_geometry
    original_solve = execution_module.solve_local_value_system

    def recording_build(*args, **kwargs):
        build_modes.append(torch.is_inference_mode_enabled())
        return original_build(*args, **kwargs)

    def recording_solve(*args, **kwargs):
        solve_modes.append(torch.is_inference_mode_enabled())
        return original_solve(*args, **kwargs)

    monkeypatch.setattr(
        execution_module,
        "_build_local_value_system_from_registered_geometry",
        recording_build,
    )
    monkeypatch.setattr(execution_module, "solve_local_value_system", recording_solve)

    result = execute_registered_orbit_target(
        arm,
        geometries[target_position],
        strata,
    )

    assert build_modes == [True]
    assert solve_modes == [True] * len(SHARED_TOLERANCE_SWEEP)
    assert tuple(item.requested_tolerance for item in result.solves) == SHARED_TOLERANCE_SWEEP
    production_index = SHARED_TOLERANCE_SWEEP.index(PRODUCTION_TOLERANCE)
    assert result.production_solve is result.solves[production_index]
    assert result.system.geometry.rank == arm.work_plan.physical_rank == 6
    assert result.system.geometry is geometries[target_position].geometry
    assert result.source_rank_grid_sha256 == strata.source_rank_grid_sha256
    assert result.strata_selection_sha256 == strata.selection_sha256
    assert result.rank_boundary["rank_matches_expected"] is True
    assert result.rank_boundary["native_compute_strict_selected_rank"] == 6
    assert result.rank_boundary["native_compute_rank_matches_expected"] is True
    assert result.rank_boundary["rank_evidence_source"] == (
        "same_direct_svd_used_by_orbit_system"
    )
    assert result.function_cholesky_error["compute_dtype"] == "float32"
    assert result.function_cholesky_error["residual_kind"] == (
        "cholesky_factorization_residual"
    )

    solution_ptrs: set[int] = set()
    for evidence in result.solves:
        solve = evidence.prediction.solve
        solution_ptrs.add(solve.solution.data_ptr())
        assert torch.equal(evidence.verified_operator_action, solve.operator_action)
        assert torch.equal(evidence.verified_residual, solve.residual)
        assert evidence.matrix_free_error is not None
        assert evidence.matrix_free_error_unavailable_reason is None
        assert evidence.matrix_free_error["residual_provenance"] == "caller_claimed"
        assert evidence.operator_action_provenance == OPERATOR_ACTION_PROVENANCE
        assert evidence.residual_provenance == RESIDUAL_PROVENANCE
        assert evidence.replay_action_provenance == REPLAY_ACTION_PROVENANCE
        assert set(evidence.replay_consistency) == {
            "operator_action_maxabs_difference",
            "operator_action_norm2_difference",
            "residual_maxabs_difference",
            "residual_norm2_difference",
        }
        assert all(value >= 0.0 for value in evidence.replay_consistency.values())
        assert evidence.operator_norm_upper_bound_provenance == (
            result.system.operator_norm_upper_bound_provenance
        )
        assert evidence.roundoff_qualification == MATRIX_FREE_ROUNDOFF_QUALIFICATION
        assert evidence.diagnostic_operator_matvecs == 1
        assert evidence.prediction.mean.grad_fn is None
        assert evidence.prediction.variance.grad_fn is None
    assert len(solution_ptrs) == len(result.solves)


def test_production_only_target_executes_one_registered_solve() -> None:
    arm = _synthetic_arm(torch.float32)
    geometries, strata = _geometry_and_strata(arm)
    target_position = next(
        position
        for position in range(PRIMARY_EVALUATION_ROW_COUNT)
        if position not in strata.selected_target_positions
    )

    result = execute_registered_orbit_target(
        arm,
        geometries[target_position],
        strata,
    )

    assert len(result.solves) == 1
    assert result.production_solve is result.solves[0]
    assert result.production_solve.requested_tolerance == PRODUCTION_TOLERANCE
    assert result.production_solve.prediction.solve.max_iterations == (
        arm.work_plan.max_iterations
    )


def test_float64_production_only_target_executes_one_registered_solve() -> None:
    arm32 = _synthetic_arm(torch.float32)
    geometries, strata = _geometry_and_strata(arm32)
    target_position = next(
        position
        for position in range(PRIMARY_EVALUATION_ROW_COUNT)
        if position not in strata.selected_target_positions
    )
    arm64 = promote_registered_orbit_arm_to_float64(arm32)

    result = execute_registered_orbit_target(
        arm64,
        geometries[target_position],
        strata,
    )

    assert result.compute_dtype == torch.float64
    assert len(result.solves) == 1
    assert result.production_solve is result.solves[0]
    assert result.production_solve.requested_tolerance == PRODUCTION_TOLERANCE


def test_float64_target_consumes_the_exact_source_fp32_cutoff() -> None:
    arm32 = _synthetic_arm(torch.float32)
    geometries, strata = _geometry_and_strata(arm32)
    target_position = strata.selected_target_positions[0]
    source_result = execute_registered_orbit_target(
        arm32,
        geometries[target_position],
        strata,
    )
    arm64 = promote_registered_orbit_arm_to_float64(arm32)

    result64 = execute_registered_orbit_target(
        arm64,
        geometries[target_position],
        strata,
    )

    assert tuple(item.requested_tolerance for item in result64.solves) == (
        SHARED_TOLERANCE_SWEEP + FP64_ONLY_TOLERANCE_SWEEP
    )
    assert result64.production_solve is result64.solves[
        (SHARED_TOLERANCE_SWEEP + FP64_ONLY_TOLERANCE_SWEEP).index(
            PRODUCTION_TOLERANCE
        )
    ]
    source_cutoff = float(
        source_result.system.geometry.operational_singular_value_cutoff
    )
    assert result64.rank_boundary["source_fp32_operational_cutoff"] == source_cutoff
    assert float(result64.system.geometry.operational_singular_value_cutoff) == (
        source_cutoff
    )
    assert result64.system.geometry.rank_epsilon_used is None
    assert result64.rank_boundary["operational_cutoff_source"] == (
        "caller_supplied_absolute_singular_value_cutoff"
    )
    assert result64.rank_boundary["compute_singular_value_dtype"] == "float64"
    native_cutoff = result64.system.geometry.native_singular_value_cutoff
    singular_values = result64.system.geometry.singular_values
    assert native_cutoff is not None
    assert singular_values is not None
    expected_native_cutoff = singular_values[0] * max(
        geometries[target_position].standardized_differences.shape
    ) * torch.finfo(torch.float64).eps
    assert float(native_cutoff) == float(expected_native_cutoff)
    assert float(native_cutoff) < source_cutoff
    assert result64.source_arm_binding_sha256 == arm32.source_arm_binding_sha256
    assert result64.source_rank_reference_sha256 == (
        source_result.source_rank_reference_sha256
    )

    with pytest.raises(ProbeExecutionInputError, match="source_geometry"):
        execute_registered_orbit_target(
            arm64,
            None,  # type: ignore[arg-type]
            strata,
        )


def test_float64_arm_rejects_post_factory_mutation() -> None:
    arm = _synthetic_arm(torch.float64)
    assert arm.train.X.dtype == torch.float64

    arm.evaluation.X[0, 0] = torch.tensor(0.1, dtype=torch.float64)
    with pytest.raises(ProbeExecutionInputError, match="changed after"):
        arm.assert_unchanged()


def test_source_content_hash_rejects_data_mutation_without_version_change() -> None:
    arm = _synthetic_arm(torch.float32)
    version = arm.train.value._version
    arm.train.value.data[0] = 123.0

    assert arm.train.value._version == version
    with pytest.raises(ProbeExecutionInputError, match="content changed"):
        arm.assert_unchanged()


def test_promoted_content_hash_rejects_numpy_mutation_without_version_change() -> None:
    arm = _synthetic_arm(torch.float64)
    version = arm.train.gradient._version
    arm.train.gradient.numpy()[0, 0] = 123.0

    assert arm.train.gradient._version == version
    with pytest.raises(ProbeExecutionInputError, match="content changed"):
        arm.assert_unchanged()


def test_float64_arm_rejects_an_exactly_representable_forged_source() -> None:
    arm = _synthetic_arm(torch.float64)
    changed_x = arm.train.X.clone()
    changed_x[0, 0] = 123.0
    changed_train = replace(arm.train, X=changed_x)

    with pytest.raises(ProbeExecutionInputError, match="bound source-fp32 SHA-256"):
        replace(arm, train=changed_train)


def test_public_float64_promotion_path_preserves_only_source_fp32_values() -> None:
    arm32 = _synthetic_arm(torch.float32)

    train64 = promote_training_split_to_float64(arm32.train)
    evaluation64 = promote_evaluation_to_float64(arm32.evaluation)
    parameters64 = promote_parameters_to_float64(arm32.parameters)
    arm64 = promote_registered_orbit_arm_to_float64(arm32)

    assert torch.equal(train64.X, arm32.train.X.to(torch.float64))
    assert torch.equal(train64.value, arm32.train.value.to(torch.float64))
    assert torch.equal(evaluation64.X, arm32.evaluation.X.to(torch.float64))
    assert torch.equal(
        parameters64.lengthscale,
        arm32.parameters.lengthscale.to(torch.float64),
    )
    assert not hasattr(arm64.evaluation, "value")
    with pytest.raises(ProbeExecutionInputError, match="requires a float32"):
        promote_training_split_to_float64(train64)
    with pytest.raises(ProbeExecutionInputError, match="requires float32"):
        promote_evaluation_to_float64(evaluation64)
    with pytest.raises(ProbeExecutionInputError, match="requires float32"):
        promote_parameters_to_float64(parameters64)


def test_source_factory_owns_private_tensor_snapshots() -> None:
    source = _synthetic_arm(torch.float32)
    snapshot = build_source_orbit_arm_inputs(
        train32=source.train,
        evaluation32=source.evaluation,
        parameters32=source.parameters,
        work_plan=source.work_plan,
    )
    expected_train = snapshot.train.X.clone()
    expected_evaluation = snapshot.evaluation.X.clone()

    source.train.X.add_(10.0)
    source.evaluation.X.sub_(10.0)

    assert torch.equal(snapshot.train.X, expected_train)
    assert torch.equal(snapshot.evaluation.X, expected_evaluation)


def test_geometry_scan_and_strata_require_complete_factory_bound_population() -> None:
    arm = _synthetic_arm(torch.float32)
    geometries, strata = _geometry_and_strata(arm)

    assert len(geometries) == PRIMARY_EVALUATION_ROW_COUNT
    assert all(type(item) is RegisteredSourceGeometry for item in geometries)
    assert type(strata) is RegisteredOrbitStrata
    assert len(strata.source_rank_reference_sha256) == PRIMARY_EVALUATION_ROW_COUNT
    assert len(strata.source_rank_grid_sha256) == 64
    assert len(strata.selected_target_positions) == arm.work_plan.support_target_count
    assert len(set(strata.selected_target_positions)) == len(
        strata.selected_target_positions
    )
    assert strata.selection_record["selected_count"] == arm.work_plan.support_target_count
    assert len(strata.selection_sha256) == 64
    with pytest.raises(ProbeExecutionInputError, match="complete 100-target"):
        select_registered_orbit_strata(arm, geometries[:-1])
    with pytest.raises(ProbeExecutionInputError, match="out of target order"):
        select_registered_orbit_strata(arm, tuple(reversed(geometries)))
    with pytest.raises(ProbeExecutionInputError, match="audited scan"):
        replace(geometries[0], _construction_token=object())
    with pytest.raises(ProbeExecutionInputError, match="audited selector"):
        replace(strata, _construction_token=object())
    mismatched_grid = replace(strata, source_rank_grid_sha256="0" * 64)
    with pytest.raises(ProbeExecutionEvidenceError, match="rank-grid SHA-256"):
        execute_registered_orbit_target(arm, geometries[0], mismatched_grid)


def test_source_geometry_hash_detects_coordinate_and_rank_record_mutation() -> None:
    arm = _synthetic_arm(torch.float32)
    geometries, strata = _geometry_and_strata(arm)
    target_position = strata.selected_target_positions[0]
    source_geometry = geometries[target_position]

    with torch.inference_mode():
        source_geometry.geometry.coordinates[0, 0].add_(1.0)
    with pytest.raises(ProbeExecutionEvidenceError, match="SHA-256"):
        execute_registered_orbit_target(arm, source_geometry, strata)

    arm = _synthetic_arm(torch.float32)
    geometries, strata = _geometry_and_strata(arm)
    target_position = strata.selected_target_positions[0]
    source_geometry = geometries[target_position]
    source_geometry.rank_boundary["rank_matches_expected"] = False
    with pytest.raises(ProbeExecutionEvidenceError, match="rank record"):
        execute_registered_orbit_target(arm, source_geometry, strata)


def test_registered_strata_hash_detects_selection_record_mutation() -> None:
    arm = _synthetic_arm(torch.float32)
    geometries, strata = _geometry_and_strata(arm)
    strata.selection_record["selection_rule"] = "tampered"

    with pytest.raises(ProbeExecutionEvidenceError, match="strata SHA-256"):
        execute_registered_orbit_target(
            arm,
            geometries[strata.selected_target_positions[0]],
            strata,
        )


def test_execution_revalidates_source_geometry_after_all_solves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm = _synthetic_arm(torch.float32)
    geometries, strata = _geometry_and_strata(arm)
    target_position = next(
        position
        for position in range(PRIMARY_EVALUATION_ROW_COUNT)
        if position not in strata.selected_target_positions
    )
    source_geometry = geometries[target_position]
    original_solve = execution_module.solve_local_value_system

    def mutating_solve(*args, **kwargs):
        prediction = original_solve(*args, **kwargs)
        source_geometry.rank_boundary["rank_matches_expected"] = False
        return prediction

    monkeypatch.setattr(execution_module, "solve_local_value_system", mutating_solve)
    with pytest.raises(ProbeExecutionEvidenceError, match="rank record"):
        execute_registered_orbit_target(arm, source_geometry, strata)


def test_nonconvergence_remains_valid_fresh_numerical_evidence() -> None:
    arm = _synthetic_arm(torch.float32)
    geometries, strata = _geometry_and_strata(arm)
    target_position = next(
        position
        for position in range(PRIMARY_EVALUATION_ROW_COUNT)
        if position not in strata.selected_target_positions
    )
    execution = execute_registered_orbit_target(
        arm,
        geometries[target_position],
        strata,
    )
    tolerance = 1e-30
    prediction = solve_local_value_system(
        execution.system,
        tolerance=tolerance,
        max_iterations=1,
        use_preconditioner=True,
    )

    assert not prediction.solve.converged
    assert prediction.solve.termination_reason == "maximum_iterations"
    evidence = execution_module._verified_solve_evidence(
        execution.system,
        prediction,
        tolerance=tolerance,
        max_iterations=1,
    )
    assert evidence.prediction.solve.converged is False
    assert evidence.prediction.solve.termination_reason == "maximum_iterations"
    assert evidence.matrix_free_error is not None


def test_nonzero_operator_replay_difference_is_recorded_not_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm = _synthetic_arm(torch.float32)
    geometries, strata = _geometry_and_strata(arm)
    target_position = next(
        position
        for position in range(PRIMARY_EVALUATION_ROW_COUNT)
        if position not in strata.selected_target_positions
    )
    execution = execute_registered_orbit_target(
        arm,
        geometries[target_position],
        strata,
    )
    system = execution.system
    assert system.operator is not None
    tolerance = PRODUCTION_TOLERANCE
    prediction = solve_local_value_system(
        system,
        tolerance=tolerance,
        max_iterations=arm.work_plan.max_iterations,
        use_preconditioner=True,
    )
    original_matmul = system.operator.matmul

    def one_ulp_replay(value: torch.Tensor) -> torch.Tensor:
        result = original_matmul(value).clone()
        result[0] = torch.nextafter(result[0], result.new_tensor(float("inf")))
        return result

    monkeypatch.setattr(system.operator, "matmul", one_ulp_replay)
    evidence = execution_module._verified_solve_evidence(
        system,
        prediction,
        tolerance=tolerance,
        max_iterations=arm.work_plan.max_iterations,
    )

    assert torch.equal(evidence.verified_operator_action, prediction.solve.operator_action)
    assert torch.equal(evidence.verified_residual, prediction.solve.residual)
    assert evidence.replay_consistency["operator_action_maxabs_difference"] > 0.0
    assert evidence.replay_consistency["residual_maxabs_difference"] > 0.0


def test_rank_zero_probe_evidence_has_no_matrix_free_solve_metric() -> None:
    dtype = torch.float64
    x_condition = torch.zeros(4, 3, dtype=dtype)
    x_target = torch.zeros(1, 3, dtype=dtype)
    system = build_local_value_system(
        x_condition,
        torch.tensor([0.2, -0.1, 0.4, 0.7], dtype=dtype),
        torch.zeros(4, 3, dtype=dtype),
        x_target,
        lengthscale=torch.tensor([1.3], dtype=dtype),
        outputscale=1.2,
        value_noise_variance=0.05,
        gradient_noise_variance=0.02,
        kernel="matern52",
    )
    tolerance = 1e-8
    prediction = solve_local_value_system(
        system,
        tolerance=tolerance,
        max_iterations=20,
        use_preconditioner=True,
    )

    evidence = execution_module._verified_solve_evidence(
        system,
        prediction,
        tolerance=tolerance,
        max_iterations=20,
    )
    assert system.geometry.rank == 0
    assert evidence.diagnostic_operator_matvecs == 0
    assert evidence.matrix_free_error is None
    assert evidence.matrix_free_error_unavailable_reason == (
        "rank_zero_no_reduced_system"
    )
    assert all(value == 0.0 for value in evidence.replay_consistency.values())


def test_arm_rejects_forged_neighbour_source_identity() -> None:
    arm = _synthetic_arm()
    arm.fixed_neighbours.source_indices[0, 0] += 1
    with pytest.raises(ProbeExecutionInputError, match="changed after"):
        arm.assert_unchanged()


def test_arm_rejects_non_authoritative_work_plan() -> None:
    arm = _synthetic_arm()
    changed_plan = replace(arm.work_plan, max_iterations=arm.work_plan.max_iterations - 1)
    with pytest.raises(ProbeExecutionInputError, match="does not match"):
        build_source_orbit_arm_inputs(
            train32=arm.train,
            evaluation32=arm.evaluation,
            parameters32=arm.parameters,
            work_plan=changed_plan,
        )


def test_registered_arm_direct_construction_is_rejected() -> None:
    arm = _synthetic_arm()
    with pytest.raises(ProbeExecutionInputError, match="audited factory"):
        RegisteredOrbitArmInputs(
            train=arm.train,
            evaluation=arm.evaluation,
            parameters=arm.parameters,
            fixed_neighbours=arm.fixed_neighbours,
            work_plan=arm.work_plan,
            source_arm_binding_sha256=arm.source_arm_binding_sha256,
            binding_kind=arm.binding_kind,
            _construction_token=object(),
        )
