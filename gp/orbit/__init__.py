"""ORBIT: orthonormal reduced-basis iterative target conditionals.

The package is an experimental, matrix-free reformulation of TERA's local
target-specific gradient reduction.  It lives outside :mod:`gp.tera` so the
official baseline remains frozen.
"""

from gp.orbit.budgeted import (
    BudgetedGuardedMarginals,
    predict_budgeted_guarded_marginals,
)
from gp.orbit.operator import (
    CGResult,
    LocalGeometry,
    OrthonormalReducedOperator,
    PosteriorCertificate,
    ReducedKroneckerPreconditioner,
    build_local_geometry,
    build_local_geometry_from_differences,
    compute_posterior_certificate,
    solve_reduced_cg,
)
from gp.orbit.predictor import (
    LocalPrediction,
    LocalValueGradientPrediction,
    LocalValueSystem,
    MarginalPredictions,
    build_local_value_system,
    differentiate_solved_local_value_system,
    predict_local_value,
    predict_local_value_and_mean_gradient,
    predict_marginal_values,
    solve_local_value_system,
)

__all__ = [
    "CGResult",
    "BudgetedGuardedMarginals",
    "LocalGeometry",
    "LocalPrediction",
    "LocalValueGradientPrediction",
    "LocalValueSystem",
    "MarginalPredictions",
    "OrthonormalReducedOperator",
    "PosteriorCertificate",
    "ReducedKroneckerPreconditioner",
    "build_local_geometry",
    "build_local_geometry_from_differences",
    "build_local_value_system",
    "compute_posterior_certificate",
    "differentiate_solved_local_value_system",
    "predict_local_value",
    "predict_budgeted_guarded_marginals",
    "predict_local_value_and_mean_gradient",
    "predict_marginal_values",
    "solve_reduced_cg",
    "solve_local_value_system",
]
