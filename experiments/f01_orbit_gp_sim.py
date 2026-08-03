"""F01 preregistration: ORBIT equivalence, solver behaviour, and neighbourhood headroom.

Hypothesis
----------
TERA's target-specific projected gradients can be expressed in an orthonormal
basis of the same local difference span.  A matrix-free solve in that basis
will (H1) reproduce TERA's scalar posterior at the same conditioning set size
and (H2) make larger conditioning sets feasible, reducing marginal KL to the
exact derivative-GP posterior.

Falsification gates
-------------------
* H1 fails if, in float64 with TERA's matched 1e-8 coordinate jitter, ORBIT differs
  from TERA by more than 1e-6 in either mean or variance at the same ``m``.
* The iterative mechanism fails if any solve misses its declared residual
  tolerance or produces non-positive/non-finite variance.
* H2 fails if increasing ``m`` does not reduce mean marginal KL over at least
  three repeats.  A runtime-only result is not an accuracy finding.
* A same-``m`` tie is an implementation control, not a win.

Confound controls
-----------------
The data are sampled from the exact derivative GP, kernel parameters are known
and shared, evaluation inputs and observations are paired, and no model learns
hyperparameters.  This isolates posterior approximation from task structure,
optimizer budget, and data leakage.  TERA's official predictor is unmodified.
Wall time is descriptive only and must be interpreted on an exclusive node;
solver residual, iteration count, rank, and live CUDA allocation are recorded.

Usage (small CPU smoke):
    python experiments/f01_orbit_gp_sim.py --n-train 40 --n-eval 20 --d 8 \
        --m-values 5,10 --tera-max-m 10 --repeats 1 --sampling dense
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_SRC = REPO_ROOT / "gp" / "tera" / "vendor" / "src"
for source_root in (REPO_ROOT, VENDOR_SRC):
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)

import torch
from gp_sim_kl.config import ExperimentConfig
from gp_sim_kl.metrics import average_marginal_kl
from gp_sim_kl.models.exact_dgp import ExactDerivativeGPPredictor
from gp_sim_kl.models.exact_dgp_deroos import ExactDerivativeGPDeroosPredictor
from gp_sim_kl.models.tera import TERAPredictor
from gp_sim_kl.simulation import simulate_dataset

from gp.orbit import predict_marginal_values

_SAME_M_TOLERANCE = 1e-6


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(device: torch.device, function):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    start = time.perf_counter()
    result = function()
    _synchronize(device)
    elapsed = time.perf_counter() - start
    peak_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return result, elapsed, peak_bytes


def _reference_predictor(sampling: str):
    if sampling == "deroos":
        return ExactDerivativeGPDeroosPredictor()
    return ExactDerivativeGPPredictor()


def _config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="f01_orbit_gp_sim",
        kernel=args.kernel,
        use_ard=False,
        lengthscale=None,
        outputscale=1.0,
        sigma_f=args.sigma_f,
        sigma_g=0.0,
        n_train=args.n_train,
        n_eval=args.n_eval,
        m=min(args.m_values),
        d_values=[args.d],
        repeats=args.repeats,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        methods=[],
        target_median_correlation=args.target_median_correlation,
        design="sobol",
        sampling=args.sampling,
        dense_sampling_max_obs_dim=args.dense_sampling_max_obs_dim,
        deroos_sampling_max_n2=args.deroos_sampling_max_n2,
        exact_dense_max_d=args.d,
        exact_dense_max_obs_dim=args.dense_sampling_max_obs_dim,
        exact_deroos_max_n=args.n_train,
    )


def _predict_tera(data, m: int):
    predictor = TERAPredictor(m=m)
    predictor.build(data)
    return predictor.predict_f_marginals(data.X_eval)


def _predict_orbit(data, m: int, args: argparse.Namespace):
    return predict_marginal_values(
        data.X_train,
        data.f_train_obs,
        data.g_train_obs,
        data.X_eval,
        m=m,
        lengthscale=data.lengthscale,
        outputscale=data.outputscale,
        value_noise_variance=data.sigma_f**2,
        gradient_noise_variance=data.sigma_g**2,
        kernel=data.kernel_name,
        gradient_noise_model="scaled",
        cg_tolerance=args.cg_tolerance,
        cg_max_iterations=args.cg_max_iterations,
        use_preconditioner=not args.no_preconditioner,
        function_jitter=args.function_jitter,
        reduced_jitter=args.reduced_jitter,
    )


def _valid_variance(variance: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(variance) & (variance > 0.0)


def _average_kl_or_none(reference, mean: torch.Tensor, variance: torch.Tensor) -> float | None:
    valid = _valid_variance(reference.var) & _valid_variance(variance)
    if not bool(valid.all().item()):
        return None
    return float(average_marginal_kl(reference.mean, reference.var, mean, variance))


def _finite_max_or_none(values: torch.Tensor) -> float | None:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return None
    return float(finite.max())


def _orbit_matmul_flops_proxy(m: int, rank: int) -> int:
    """Leading GEMM and triangular-solve FLOPs for one reduced-operator action."""

    return 10 * m * m * rank + 2 * m * rank * rank + 2 * m * m


def _preconditioner_flops_proxy(m: int, rank: int) -> int:
    """Leading GEMM FLOPs for one Kronecker preconditioner application."""

    return 4 * m * m * rank + 4 * m * rank * rank


def _summarize_hypotheses(
    rows: list[dict],
    repeats: int,
    tera_max_m: int,
    n_train: int,
) -> dict:
    orbit_rows = [row for row in rows if row["method"] == "ORBIT-exact"]
    paired_rows = [row for row in orbit_rows if "maxabs_mean_to_same_m_tera" in row]
    same_m_pass = bool(paired_rows) and all(
        row["maxabs_mean_to_same_m_tera"] <= _SAME_M_TOLERANCE
        and row["maxabs_variance_to_same_m_tera"] <= _SAME_M_TOLERANCE
        for row in paired_rows
    )
    solver_pass = bool(orbit_rows) and all(
        row["cg_converged_fraction"] == 1.0
        and row["variance_valid_fraction"] == 1.0
        and row["basis_exact_fraction"] == 1.0
        for row in orbit_rows
    )

    paired_m = sorted({row["m"] for row in paired_rows})
    reference_m = max((m for m in paired_m if m <= tera_max_m), default=None)
    candidate_m = max((row["m"] for row in orbit_rows), default=None)
    baseline_kls = [
        row["avg_marginal_kl"]
        for row in orbit_rows
        if row["m"] == reference_m and row["avg_marginal_kl"] is not None
    ]
    candidate_kls = [
        row["avg_marginal_kl"]
        for row in orbit_rows
        if row["m"] == candidate_m and row["avg_marginal_kl"] is not None
    ]
    enough_repeats = repeats >= 3 and len(baseline_kls) == repeats and len(candidate_kls) == repeats
    larger_m = reference_m is not None and candidate_m is not None and candidate_m > reference_m
    nontrivial_candidate = candidate_m is not None and candidate_m < n_train
    mean_baseline = sum(baseline_kls) / len(baseline_kls) if baseline_kls else None
    mean_candidate = sum(candidate_kls) / len(candidate_kls) if candidate_kls else None
    headroom_pass = bool(
        enough_repeats
        and larger_m
        and nontrivial_candidate
        and mean_baseline is not None
        and mean_candidate is not None
        and mean_candidate < mean_baseline
    )
    return {
        "h1_same_m_equivalence_pass": same_m_pass,
        "iterative_validity_pass": solver_pass,
        "h2_larger_m_headroom_pass": headroom_pass,
        "h2_reference_m": reference_m,
        "h2_candidate_m": candidate_m,
        "h2_reference_mean_kl": mean_baseline,
        "h2_candidate_mean_kl": mean_candidate,
        "h2_requires_at_least_three_repeats": True,
        "h2_requires_candidate_m_below_n_train": True,
        "h2_candidate_is_nontrivial": nontrivial_candidate,
        "scope": "mechanism experiment only; not a SOTA performance claim",
    }


def run(args: argparse.Namespace) -> dict:
    cfg = _config(args)
    device = torch.device(args.device)
    rows = []
    for repeat in range(args.repeats):
        data = simulate_dataset(cfg, d=args.d, repeat=repeat)
        reference_model = _reference_predictor(args.sampling)
        _, reference_build_seconds, reference_build_peak = _measure(
            device,
            lambda reference_model=reference_model, data=data: reference_model.build(data),
        )
        reference, reference_seconds, reference_peak = _measure(
            device,
            lambda reference_model=reference_model, data=data: reference_model.predict_f_marginals(
                data.X_eval
            ),
        )

        tera_by_m = {}
        for m in args.m_values:
            if m > args.tera_max_m:
                continue
            tera, elapsed, peak_bytes = _measure(
                device,
                lambda m=m, data=data: _predict_tera(data, m),
            )
            tera_by_m[m] = tera
            rows.append(
                {
                    "repeat": repeat,
                    "method": "TERA",
                    "m": m,
                    "avg_marginal_kl": _average_kl_or_none(reference, tera.mean, tera.var),
                    "maxabs_mean_to_reference": float((tera.mean - reference.mean).abs().max()),
                    "maxabs_variance_to_reference": float((tera.var - reference.var).abs().max()),
                    "seconds_descriptive": elapsed,
                    "peak_allocated_bytes": peak_bytes,
                    "reduced_system_dimension": m * m,
                    "explicit_reduced_covariance_elements_per_target": m**4,
                    "reduced_cholesky_leading_flops_per_target": (m**6) / 3.0,
                    "variance_valid_fraction": float(_valid_variance(tera.var).float().mean()),
                }
            )

        for m in args.m_values:
            orbit, elapsed, peak_bytes = _measure(
                device,
                lambda m=m, data=data: _predict_orbit(data, m, args),
            )
            row = {
                "repeat": repeat,
                "method": "ORBIT-exact",
                "m": m,
                "avg_marginal_kl": _average_kl_or_none(
                    reference,
                    orbit.mean,
                    orbit.variance,
                ),
                "maxabs_mean_to_reference": float((orbit.mean - reference.mean).abs().max()),
                "maxabs_variance_to_reference": float((orbit.variance - reference.var).abs().max()),
                "seconds_descriptive": elapsed,
                "peak_allocated_bytes": peak_bytes,
                "rank_min": int(orbit.ranks.min()),
                "rank_mean": float(orbit.ranks.float().mean()),
                "rank_max": int(orbit.ranks.max()),
                "cg_iterations_min": int(orbit.iterations.min()),
                "cg_iterations_mean": float(orbit.iterations.float().mean()),
                "cg_iterations_max": int(orbit.iterations.max()),
                "cg_relative_residual_max": float(orbit.relative_residuals.max()),
                "cg_converged_fraction": float(orbit.converged.float().mean()),
                "variance_min_raw": float(orbit.variance.min()),
                "variance_valid_fraction": float(_valid_variance(orbit.variance).float().mean()),
                "exact_arithmetic_certificate_fraction": float(
                    orbit.exact_arithmetic_certified.float().mean()
                ),
                "floating_point_rigorous_certificate_fraction": float(
                    orbit.floating_point_rigorous.float().mean()
                ),
                "basis_exact_fraction": float(orbit.basis_exact.float().mean()),
                "finite_precision_variance_correction_maxabs": float(
                    orbit.finite_precision_variance_corrections.abs().max()
                ),
                "expected_kl_upper_bound_max": _finite_max_or_none(orbit.expected_kl_upper_bounds),
                "reduced_system_dimension_max": m * int(orbit.ranks.max()),
                "explicit_reduced_covariance_elements_per_target": 0,
                "operator_core_elements_proxy_per_target_max": int(
                    3 * m * m
                    + 2 * m * int(orbit.ranks.max())
                    + int(orbit.ranks.max()) ** 2
                    + int(orbit.ranks.max())
                ),
                "cg_operator_flops_proxy_total": int(
                    sum(
                        (int(iterations) + 1) * _orbit_matmul_flops_proxy(m, int(rank))
                        for iterations, rank in zip(
                            orbit.iterations.tolist(),
                            orbit.ranks.tolist(),
                            strict=True,
                        )
                    )
                ),
                "cg_preconditioner_flops_proxy_total": int(
                    0
                    if args.no_preconditioner
                    else sum(
                        int(iterations) * _preconditioner_flops_proxy(m, int(rank))
                        for iterations, rank in zip(
                            orbit.iterations.tolist(),
                            orbit.ranks.tolist(),
                            strict=True,
                        )
                    )
                ),
            }
            if m in tera_by_m:
                tera = tera_by_m[m]
                row["maxabs_mean_to_same_m_tera"] = float((orbit.mean - tera.mean).abs().max())
                row["maxabs_variance_to_same_m_tera"] = float(
                    (orbit.variance - tera.var).abs().max()
                )
            rows.append(row)

        rows.append(
            {
                "repeat": repeat,
                "method": "Exact dGP reference",
                "m": None,
                "avg_marginal_kl": 0.0,
                "build_seconds_descriptive": reference_build_seconds,
                "prediction_seconds_descriptive": reference_seconds,
                "seconds_descriptive": reference_build_seconds + reference_seconds,
                "peak_allocated_bytes": (
                    None
                    if reference_build_peak is None
                    else max(reference_build_peak, reference_peak)
                ),
            }
        )

    hypotheses = _summarize_hypotheses(
        rows,
        args.repeats,
        args.tera_max_m,
        args.n_train,
    )
    return {
        "preregistration": {
            "same_m_equivalence_tolerance": _SAME_M_TOLERANCE,
            "cg_tolerance": args.cg_tolerance,
            "wall_time_is_inferential": False,
            "nonpositive_or_nonfinite_variance_is_failure": True,
            "certificate_arithmetic_scope": (
                "exact-arithmetic bound for represented operator; reported floating-point "
                "scalars are qualified by dtype-scale roundoff"
            ),
            "cost_model_note": (
                "FLOP fields count leading dense GEMM/triangular-solve terms only; "
                "exclusive-node measurements remain descriptive corroboration"
            ),
        },
        "config": vars(args),
        "hypothesis_tests": hypotheses,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F01 ORBIT GP-simulation experiment")
    parser.add_argument("--n-train", type=int, default=40)
    parser.add_argument("--n-eval", type=int, default=20)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--m-values", default="5,10,20")
    parser.add_argument("--tera-max-m", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--kernel", choices=["rbf", "matern52"], default="matern52")
    parser.add_argument("--sigma-f", type=float, default=1e-3)
    parser.add_argument("--target-median-correlation", type=float, default=0.3)
    parser.add_argument("--sampling", choices=["dense", "deroos"], default="dense")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--cg-tolerance", type=float, default=1e-8)
    parser.add_argument("--cg-max-iterations", type=int)
    parser.add_argument("--no-preconditioner", action="store_true")
    parser.add_argument("--function-jitter", type=float, default=1e-8)
    parser.add_argument("--reduced-jitter", type=float, default=1e-8)
    parser.add_argument("--dense-sampling-max-obs-dim", type=int, default=20000)
    parser.add_argument("--deroos-sampling-max-n2", type=int, default=6400)
    parser.add_argument("--out", default="runs/f01_orbit_gp_sim.json")
    args = parser.parse_args()
    args.m_values = [int(value) for value in args.m_values.split(",")]
    if args.m_values != sorted(set(args.m_values)):
        parser.error("m-values must be strictly increasing with no duplicates")
    if any(value <= 0 or value > args.n_train for value in args.m_values):
        parser.error("all m-values must lie in [1, n-train]")
    if args.tera_max_m <= 0:
        parser.error("tera-max-m must be positive")
    if args.n_train <= 1 or args.n_eval <= 0 or args.d <= 0 or args.repeats <= 0:
        parser.error("n-train must exceed 1 and n-eval, d, and repeats must be positive")
    if args.sigma_f < 0.0:
        parser.error("sigma-f must be non-negative")
    if not 0.0 < args.target_median_correlation < 1.0:
        parser.error("target-median-correlation must lie in (0, 1)")
    if args.cg_max_iterations is not None and args.cg_max_iterations <= 0:
        parser.error("cg-max-iterations must be positive")
    if args.function_jitter <= 0.0 or args.reduced_jitter < 0.0:
        parser.error("function-jitter must be positive and reduced-jitter non-negative")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments)
    output = Path(arguments.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    for record in result["rows"]:
        print(json.dumps(record), flush=True)
    print(f"wrote {output}")
