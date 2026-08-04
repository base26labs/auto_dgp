"""ORBIT: orthonormal reduced-basis iterative target conditionals.

The package is an experimental, matrix-free reformulation of TERA's local
target-specific gradient reduction.  It lives outside :mod:`gp.tera` so the
official baseline remains frozen.
"""

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
    LocalValueSystem,
    MarginalPredictions,
    build_local_value_system,
    predict_local_value,
    predict_marginal_values,
    solve_local_value_system,
)

__all__ = [
    "CGResult",
    "LocalGeometry",
    "LocalPrediction",
    "LocalValueSystem",
    "MarginalPredictions",
    "OrthonormalReducedOperator",
    "PosteriorCertificate",
    "ReducedKroneckerPreconditioner",
    "build_local_geometry",
    "build_local_geometry_from_differences",
    "build_local_value_system",
    "compute_posterior_certificate",
    "predict_local_value",
    "predict_marginal_values",
    "solve_reduced_cg",
    "solve_local_value_system",
]
