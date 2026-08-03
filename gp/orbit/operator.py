"""Matrix-free orthonormal form of TERA's reduced-gradient conditional.

For one scalar target and ``m`` conditioning locations, TERA projects each
observed gradient onto the ``m`` target-to-neighbour differences.  The stacked
projected state therefore has size ``m**2`` and TERA explicitly factorizes its
conditional covariance at an ``O(m**6)`` cost.

Let the scaled target-centred difference matrix be ``Delta`` and write its thin
SVD as ``Delta = U S V.T`` with rank ``r``.  Conditioning on

    q_a = Delta.T @ g_a = V S (U.T @ g_a)

is equivalent to conditioning on the orthonormal coordinates
``z_a = U.T @ g_a``.  The latter have only ``r`` entries per neighbour and a
much simpler covariance.  This module applies that covariance without ever
forming its ``(m*r) x (m*r)`` matrix.

For a stationary radial kernel, define ``A[a,b]`` and ``B[a,b]`` so that the
scaled gradient covariance is

    Cov(g_a, g_b) = A[a,b] I + B[a,b] d_ab d_ab.T.

If ``C[a] = U.T @ (x_a - x_target)``, the unconditioned projected-gradient
blocks are

    G[a,b] = A[a,b] I_r + B[a,b] (C[a]-C[b]) (C[a]-C[b]).T

plus block-diagonal projected observation noise.  Conditioning on neighbour
function values gives ``K_delta = G - Q K_ff^-1 Q.T``.  Each multiplication by
``K_delta`` costs ``O(m**2*r)`` and stores only ``O(m*r + m**2)`` entries.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LocalGeometry:
    """Orthonormal coordinates for one target-centred local geometry.

    ``coordinates`` contains the coordinates of the ``m`` scaled differences
    in an orthonormal basis, shape ``(m, r)``.  ``q_to_z`` maps row-wise TERA
    statistics ``q = D.T @ g`` to the orthonormal statistics ``z``.
    """

    coordinates: torch.Tensor
    q_to_z: torch.Tensor
    eigenvalues: torch.Tensor
    discarded_eigenvalue_sum: torch.Tensor
    is_exact: bool

    @property
    def rank(self) -> int:
        return int(self.coordinates.shape[1])


def build_local_geometry(
    gram: torch.Tensor,
    *,
    rank: int | None = None,
    relative_tolerance: float | None = None,
) -> LocalGeometry:
    """Build an exact or truncated orthonormal basis from ``H = Delta.T Delta``.

    With neither ``rank`` nor ``relative_tolerance`` supplied, only numerical
    null directions are discarded.  Supplying either option enables the
    separately labelled approximate-rank mode; it must not be described as an
    exact TERA reformulation.  Forming ``H`` squares the condition number, so
    prediction code should prefer :func:`build_local_geometry_from_differences`
    when the original difference matrix is available.
    """

    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must be a square matrix")
    if rank is not None and relative_tolerance is not None:
        raise ValueError("set at most one of rank and relative_tolerance")
    if rank is not None and not 1 <= rank <= gram.shape[0]:
        raise ValueError("rank must lie between 1 and gram.shape[0]")
    if relative_tolerance is not None and not 0.0 <= relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must lie in [0, 1)")

    sym = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(sym)
    order = torch.arange(eigenvalues.numel() - 1, -1, -1, device=gram.device)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    largest = eigenvalues[0].clamp_min(0.0)
    numerical_tolerance = largest * gram.shape[0] * torch.finfo(gram.dtype).eps
    if rank is not None:
        keep = torch.arange(eigenvalues.numel(), device=gram.device) < rank
        keep = keep & (eigenvalues > numerical_tolerance)
    else:
        threshold = numerical_tolerance
        if relative_tolerance is not None:
            threshold = torch.maximum(threshold, largest * relative_tolerance)
        keep = eigenvalues > threshold

    discarded = eigenvalues[~keep].clamp_min(0.0).sum()
    is_exact = not bool((eigenvalues[~keep] > numerical_tolerance).any().item())
    if not bool(keep.any().item()):
        empty = gram.new_empty((gram.shape[0], 0))
        return LocalGeometry(
            coordinates=empty,
            q_to_z=empty.clone(),
            eigenvalues=gram.new_empty((0,)),
            discarded_eigenvalue_sum=discarded,
            is_exact=is_exact,
        )

    kept_values = eigenvalues[keep]
    kept_vectors = eigenvectors[:, keep]
    singular_values = torch.sqrt(kept_values)
    coordinates = kept_vectors * singular_values.unsqueeze(0)
    q_to_z = kept_vectors / singular_values.unsqueeze(0)
    return LocalGeometry(
        coordinates=coordinates,
        q_to_z=q_to_z,
        eigenvalues=kept_values,
        discarded_eigenvalue_sum=discarded,
        is_exact=is_exact,
    )


def build_local_geometry_from_differences(
    differences: torch.Tensor,
    *,
    rank: int | None = None,
    relative_tolerance: float | None = None,
) -> LocalGeometry:
    """Build local coordinates from a stable thin SVD of scaled differences.

    Forming ``differences.T @ differences`` squares its condition number and
    can discard resolvable small singular directions.  Prediction therefore
    uses this direct-SVD path.  The ambient left singular vectors are only a
    transient decomposition output; covariance state uses ``V`` and ``S``.
    """

    if differences.ndim != 2 or min(differences.shape) == 0:
        raise ValueError("differences must have nonzero shape (d, m)")
    if rank is not None and relative_tolerance is not None:
        raise ValueError("set at most one of rank and relative_tolerance")
    maximum_rank = min(differences.shape)
    if rank is not None and not 1 <= rank <= maximum_rank:
        raise ValueError("rank must lie between 1 and min(d, m)")
    if relative_tolerance is not None and not 0.0 <= relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must lie in [0, 1)")

    _, singular_values, right_transpose = torch.linalg.svd(
        differences,
        full_matrices=False,
    )
    largest = singular_values[0]
    numerical_tolerance = largest * max(differences.shape) * torch.finfo(differences.dtype).eps
    if rank is not None:
        keep = torch.arange(singular_values.numel(), device=differences.device) < rank
        keep = keep & (singular_values > numerical_tolerance)
    else:
        threshold = numerical_tolerance
        if relative_tolerance is not None:
            threshold = torch.maximum(threshold, largest * relative_tolerance)
        keep = singular_values > threshold

    discarded = (singular_values[~keep] ** 2).sum()
    is_exact = not bool((singular_values[~keep] > numerical_tolerance).any().item())
    m = differences.shape[1]
    if not bool(keep.any().item()):
        empty = differences.new_empty((m, 0))
        return LocalGeometry(
            coordinates=empty,
            q_to_z=empty.clone(),
            eigenvalues=differences.new_empty((0,)),
            discarded_eigenvalue_sum=discarded,
            is_exact=is_exact,
        )

    kept_singular_values = singular_values[keep]
    kept_right_vectors = right_transpose[keep].T
    coordinates = kept_right_vectors * kept_singular_values.unsqueeze(0)
    q_to_z = kept_right_vectors / kept_singular_values.unsqueeze(0)
    return LocalGeometry(
        coordinates=coordinates,
        q_to_z=q_to_z,
        eigenvalues=kept_singular_values**2,
        discarded_eigenvalue_sum=discarded,
        is_exact=is_exact,
    )


class OrthonormalReducedOperator:
    """Apply ``K_delta = G - Q K_ff^-1 Q.T + jitter I`` matrix-free."""

    def __init__(
        self,
        coordinates: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        function_cholesky: torch.Tensor,
        gradient_noise: torch.Tensor | float = 0.0,
        *,
        jitter: float = 1e-8,
    ) -> None:
        if coordinates.ndim != 2:
            raise ValueError("coordinates must have shape (m, r)")
        m, r = coordinates.shape
        if alpha.shape != (m, m) or beta.shape != (m, m):
            raise ValueError("alpha and beta must have shape (m, m)")
        if function_cholesky.shape != (m, m):
            raise ValueError("function_cholesky must have shape (m, m)")
        if jitter < 0.0:
            raise ValueError("jitter must be non-negative")

        noise = torch.as_tensor(
            gradient_noise,
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        if noise.ndim == 0:
            noise = noise * torch.eye(r, device=coordinates.device, dtype=coordinates.dtype)
        if noise.shape != (r, r):
            raise ValueError("gradient_noise must be scalar or have shape (r, r)")

        self.coordinates = coordinates
        self.alpha = 0.5 * (alpha + alpha.T)
        self.beta = 0.5 * (beta + beta.T)
        self.function_cholesky = function_cholesky
        self.gradient_noise = 0.5 * (noise + noise.T)
        self.jitter = float(jitter)

    @property
    def m(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def rank(self) -> int:
        return int(self.coordinates.shape[1])

    @property
    def size(self) -> int:
        return self.m * self.rank

    @property
    def eigenvalue_lower_bound(self) -> float:
        """Return a conservative lower bound for the jittered operator.

        The signal part is a Gaussian conditional covariance and therefore
        positive semidefinite.  Projected observation noise and the explicit
        coordinate jitter are the only generally available strict lower bound.
        """

        minimum_noise = torch.linalg.eigvalsh(self.gradient_noise).min()
        numerical_scale = torch.linalg.norm(self.gradient_noise, ord=2).clamp_min(1.0)
        tolerance = 100.0 * torch.finfo(self.gradient_noise.dtype).eps * numerical_scale
        if float(minimum_noise) < -float(tolerance):
            raise ValueError("gradient_noise must be positive semidefinite")
        return max(0.0, float(minimum_noise)) + self.jitter

    def _matrix(self, value: torch.Tensor) -> torch.Tensor:
        if value.numel() != self.size:
            raise ValueError(f"expected a vector with {self.size} entries")
        return value.reshape(self.m, self.rank)

    def q_matmul(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the value/projected-gradient cross block ``Q``."""

        if value.ndim != 1 or value.numel() != self.m:
            raise ValueError(f"expected a vector with {self.m} entries")
        weighted = self.alpha * value.unsqueeze(0)
        out = -(weighted.sum(dim=1, keepdim=True) * self.coordinates)
        out = out + weighted @ self.coordinates
        return out.reshape(-1)

    def q_t_matmul(self, value: torch.Tensor) -> torch.Tensor:
        """Apply ``Q.T`` without instantiating the ``(m*r) x m`` matrix."""

        matrix = self._matrix(value)
        cross = matrix @ self.coordinates.T
        self_dot = torch.diagonal(cross).unsqueeze(1)
        return -(self.alpha * (self_dot - cross)).sum(dim=0)

    def unconditional_matmul(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the unconditioned projected-gradient covariance ``G``."""

        matrix = self._matrix(value)
        out = self.alpha @ matrix

        cross = self.coordinates @ matrix.T
        pair_dot = cross - torch.diagonal(cross).unsqueeze(0)
        weights = self.beta * pair_dot
        out = out + weights.sum(dim=1, keepdim=True) * self.coordinates
        out = out - weights @ self.coordinates

        out = out + matrix @ self.gradient_noise.T
        return out.reshape(-1)

    def matmul(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the conditional covariance ``K_delta``."""

        q_t_value = self.q_t_matmul(value)
        conditioned = torch.cholesky_solve(
            q_t_value.unsqueeze(1),
            self.function_cholesky,
        ).squeeze(1)
        out = self.unconditional_matmul(value) - self.q_matmul(conditioned)
        if self.jitter:
            out = out + self.jitter * value
        return out

    def conditional_cross(
        self,
        target_alpha: torch.Tensor,
        function_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``k_delta`` for the scalar target after conditioning on values.

        ``target_alpha[a]`` is the stationary gradient coefficient between the
        target and neighbour ``a``; ``function_weights`` is
        ``K_ff^-1 k_f,target``.
        """

        if target_alpha.shape != (self.m,) or function_weights.shape != (self.m,):
            raise ValueError("target_alpha and function_weights must have shape (m,)")
        unconditioned = -target_alpha.unsqueeze(1) * self.coordinates
        return (unconditioned.reshape(-1) - self.q_matmul(function_weights)).contiguous()

    def dense(self) -> torch.Tensor:
        """Materialize the operator for tests and small diagnostics only."""

        eye = torch.eye(self.size, device=self.coordinates.device, dtype=self.coordinates.dtype)
        return torch.stack([self.matmul(eye[:, j]) for j in range(self.size)], dim=1)


class ReducedKroneckerPreconditioner:
    """Kronecker-sum preconditioner for the dominant ``A kron I + I kron N``.

    The radial outer-product term and function-value Schur correction are left
    to CG.  Application costs ``O(m**2*r + m*r**2)`` after two small symmetric
    eigendecompositions.
    """

    def __init__(self, operator: OrthonormalReducedOperator, *, floor: float = 1e-10) -> None:
        if floor <= 0.0:
            raise ValueError("floor must be positive")
        alpha_values, alpha_vectors = torch.linalg.eigh(operator.alpha)
        noise_values, noise_vectors = torch.linalg.eigh(operator.gradient_noise)
        denominator = (
            alpha_values.unsqueeze(1) + noise_values.unsqueeze(0) + operator.jitter
        ).clamp_min(floor)
        self.alpha_vectors = alpha_vectors
        self.noise_vectors = noise_vectors
        self.denominator = denominator
        self.m = operator.m
        self.rank = operator.rank

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        if value.numel() != self.m * self.rank:
            raise ValueError(f"expected a vector with {self.m * self.rank} entries")
        matrix = value.reshape(self.m, self.rank)
        transformed = self.alpha_vectors.T @ matrix @ self.noise_vectors
        transformed = transformed / self.denominator
        out = self.alpha_vectors @ transformed @ self.noise_vectors.T
        return out.reshape(-1)


@dataclass(frozen=True)
class CGResult:
    solution: torch.Tensor
    residual: torch.Tensor
    iterations: int
    relative_residual: float
    residual_norm: float
    rhs_norm: float
    converged: bool


@dataclass(frozen=True)
class PosteriorCertificate:
    """Residual certificate for an anytime scalar Gaussian conditional.

    ``variance_error_upper_bound`` bounds the gap between the conservative
    variance at the current Galerkin/PCG iterate and the exactly solved
    variance *within the selected geometric basis*.  ``solve_certified``
    reports that narrower guarantee.  ``exact_arithmetic_certified``
    additionally requires the basis to retain every non-null geometric mode;
    spectral truncation needs a separate error analysis and is never hidden
    inside the PCG certificate.  ``floating_point_rigorous`` remains false
    unless a future implementation supplies verified residual and scalar
    evaluation enclosures.
    The inequalities are exact-arithmetic statements for the represented
    operator.  Reported floating-point scalars are subject to dtype-scale
    roundoff and are not interval-arithmetic certificates.
    """

    variance_error_upper_bound: float
    expected_kl_upper_bound: float
    operator_eigenvalue_lower_bound: float
    exact_arithmetic_certified: bool
    solve_certified: bool
    basis_is_exact: bool
    floating_point_rigorous: bool


def _cg_result(
    operator: OrthonormalReducedOperator,
    rhs: torch.Tensor,
    solution: torch.Tensor,
    iterations: int,
    tolerance: float,
) -> CGResult:
    """Build a result from a recomputed, rather than recursive, residual."""

    true_residual = rhs - operator.matmul(solution)
    rhs_norm = float(torch.linalg.norm(rhs))
    residual_norm = float(torch.linalg.norm(true_residual))
    relative_residual = residual_norm / rhs_norm if rhs_norm else 0.0
    finite = math.isfinite(relative_residual)
    return CGResult(
        solution=solution,
        residual=true_residual,
        iterations=iterations,
        relative_residual=relative_residual,
        residual_norm=residual_norm,
        rhs_norm=rhs_norm,
        converged=finite and relative_residual <= tolerance,
    )


def compute_posterior_certificate(
    operator: OrthonormalReducedOperator,
    solve: CGResult,
    conservative_variance: torch.Tensor | float,
    *,
    basis_is_exact: bool = True,
) -> PosteriorCertificate:
    """Certify scalar-posterior solve error from the recomputed residual.

    If ``lambda_min(operator) >= lambda_0 > 0``, the Galerkin energy error is
    at most ``||residual||**2 / lambda_0``.  For a scalar Gaussian target this
    energy error is exactly the excess (conservative) posterior variance.  If
    the bound is smaller than that variance, it also yields an expected KL
    upper bound ``0.5 * log(v / (v - bound))``.  These are exact-arithmetic
    inequalities for the stored operator data; finite-precision evaluation is
    not a directed-rounding or interval certificate.
    """

    lower_bound = operator.eigenvalue_lower_bound
    variance = float(torch.as_tensor(conservative_variance))
    if lower_bound <= 0.0 or not math.isfinite(variance) or variance <= 0.0:
        return PosteriorCertificate(
            variance_error_upper_bound=math.inf,
            expected_kl_upper_bound=math.inf,
            operator_eigenvalue_lower_bound=lower_bound,
            exact_arithmetic_certified=False,
            solve_certified=False,
            basis_is_exact=basis_is_exact,
            floating_point_rigorous=False,
        )

    variance_error = solve.residual_norm**2 / lower_bound
    if not math.isfinite(variance_error) or variance_error >= variance:
        return PosteriorCertificate(
            variance_error_upper_bound=variance_error,
            expected_kl_upper_bound=math.inf,
            operator_eigenvalue_lower_bound=lower_bound,
            exact_arithmetic_certified=False,
            solve_certified=False,
            basis_is_exact=basis_is_exact,
            floating_point_rigorous=False,
        )

    expected_kl = 0.5 * math.log(variance / (variance - variance_error))
    return PosteriorCertificate(
        variance_error_upper_bound=variance_error,
        expected_kl_upper_bound=expected_kl,
        operator_eigenvalue_lower_bound=lower_bound,
        exact_arithmetic_certified=basis_is_exact,
        solve_certified=True,
        basis_is_exact=basis_is_exact,
        floating_point_rigorous=False,
    )


def solve_reduced_cg(
    operator: OrthonormalReducedOperator,
    rhs: torch.Tensor,
    *,
    tolerance: float = 1e-6,
    max_iterations: int | None = None,
    preconditioner: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> CGResult:
    """Solve one reduced conditional system with (preconditioned) CG."""

    if rhs.ndim != 1 or rhs.numel() != operator.size:
        raise ValueError(f"rhs must have shape ({operator.size},)")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if max_iterations is None:
        max_iterations = operator.size
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    rhs_norm = torch.linalg.norm(rhs)
    if float(rhs_norm) == 0.0:
        return CGResult(solution, rhs.clone(), 0, 0.0, 0.0, 0.0, True)

    z = residual if preconditioner is None else preconditioner(residual)
    direction = z.clone()
    rz = torch.dot(residual, z)
    relative_residual = 1.0
    for iteration in range(1, max_iterations + 1):
        image = operator.matmul(direction)
        curvature = torch.dot(direction, image)
        if not bool(torch.isfinite(curvature).item()) or float(curvature) <= 0.0:
            return _cg_result(operator, rhs, solution, iteration - 1, tolerance)
        step = rz / curvature
        solution = solution + step * direction
        residual = residual - step * image
        relative_residual = float(torch.linalg.norm(residual) / rhs_norm)
        if relative_residual <= tolerance:
            checked = _cg_result(operator, rhs, solution, iteration, tolerance)
            if checked.converged or iteration == max_iterations:
                return checked

            # Recursive CG residuals can drift just below the stopping
            # threshold while a fresh operator application remains above it.
            # Replace the residual and reliably restart instead of returning a
            # result whose own convergence flag contradicts the stop decision.
            residual = checked.residual
            z = residual if preconditioner is None else preconditioner(residual)
            rz = torch.dot(residual, z)
            if not bool(torch.isfinite(rz).item()) or float(rz) <= 0.0:
                return checked
            direction = z.clone()
            continue
        next_z = residual if preconditioner is None else preconditioner(residual)
        next_rz = torch.dot(residual, next_z)
        if not bool(torch.isfinite(next_rz).item()) or math.isclose(float(rz), 0.0):
            return _cg_result(operator, rhs, solution, iteration, tolerance)
        direction = next_z + (next_rz / rz) * direction
        rz = next_rz
    return _cg_result(operator, rhs, solution, max_iterations, tolerance)
