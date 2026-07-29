"""N-body benchmark: the base exact derivative GP (starting point) vs TERA (best approximate GP).

Both predict the GRADIENT of the target value at held-out N-body states; we report gradient RMSE
(mean +- sd over seeds). The exact GP also reports value RMSE; TERA additionally reports value NLL,
since it returns a predictive variance (the exact GP's mean-only path does not).

This is the STARTING scaffold, deliberately simple: a validation-grid search over the exact GP's
lengthscale AND its (value, derivative) noise pair (so it self-selects hyperparameters like TERA
does, on the clock), the base unpreconditioned CG solve, and a modest N. Generate the data first
(see docs/STARTUP.md); it writes data/nbody/*.npz.

Usage:  uv run python experiments/nbody_benchmark.py
"""

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, ".")

import numpy as np
import torch

from gp.cg import solve_cg
from gp.common import noise_vector, stack_targets
from gp.exact import gaussian_nll, predict_grad, predict_value
from gp.kernels import MaternDerivKernel, RBFDerivKernel
from gp.metrics import rmse
from gp.tera import run_tera

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float32

# D (state dimension = 2 * n_particles * n_dims) -> dataset file written by data/get_nbody.py
DATASETS = {
    12: "nbody_n2_d3.npz",
    60: "nbody_n10_d3.npz",
}
SEEDS = [0, 1, 2]
NTRAIN = 1500
NTEST = 500
CG_ITERS = 1000
CG_TOL = 1e-3

# Exact-GP hyperparameter selection: sweep the lengthscale over these multipliers of the median
# heuristic, and the (value noise, derivative noise) pair over these grids, scored on a held-out
# fraction of the training data (see select_hypers).
LS_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
NOISE_GRID = (1e-3, 1e-2, 1e-1)
DERIV_NOISE_GRID = (1e-3, 1e-2, 1e-1)
VAL_FRAC = 0.2
HP_SUBSAMPLE = 4000

# Kernel for BOTH arms. RBF is the classical starting kernel (and TERA's upstream default); Matern-5/2
# ("matern52") is better-conditioned for the CG solve and a natural next step.
KERNEL = "rbf"
_KERNELS = {"rbf": RBFDerivKernel, "matern52": MaternDerivKernel}

# Starting point / fallback for the exact-GP noise on the (K+Lambda) diagonal: (value, gradient).
# These are where the coordinate search begins, and what it falls back to if no cell converges --
# they are NOT the values used unless the search selects them.
EXACT_NOISE = 1e-2
EXACT_DERIV_NOISE = 1e-2

# TERA (approximate baseline) config, mirroring its upstream defaults.
TERA_M = 20
TERA_EPOCHS = 1


def prep(path: str, seed: int) -> list[torch.Tensor]:
    """Load, split, and standardize one dataset. X -> per-dim [0,1]; value -> zero-mean/unit-std;
    gradients rescaled by the same (span / value-std) so the standardization is consistent. The npz
    stores the value under key `E` and the gradient under key `F`."""
    rec = np.load(f"data/nbody/{path}")
    X = rec["X"].astype(np.float64)
    val = rec["E"].astype(np.float64).reshape(-1)
    grad = rec["F"].astype(np.float64)
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed)).numpy()
    tr, te = perm[:NTRAIN], perm[NTRAIN : NTRAIN + NTEST]

    Xtr, Xte = torch.tensor(X[tr], dtype=DT), torch.tensor(X[te], dtype=DT)
    vtr_raw, vte_raw = torch.tensor(val[tr], dtype=DT), torch.tensor(val[te], dtype=DT)
    gtr, gte = torch.tensor(grad[tr], dtype=DT), torch.tensor(grad[te], dtype=DT)

    lo, hi = Xtr.min(0).values, Xtr.max(0).values
    span = (hi - lo).clamp_min(1e-12)
    Xtr, Xte = (Xtr - lo) / span, (Xte - lo) / span
    mu, sig = vtr_raw.mean(), vtr_raw.std().clamp_min(1e-8)
    vtr, vte = (vtr_raw - mu) / sig, (vte_raw - mu) / sig
    gtr, gte = gtr * span / sig, gte * span / sig
    return [t.to(DEV) for t in (Xtr, vtr, gtr, Xte, vte, gte)]


def median_lengthscale(X: torch.Tensor, n: int = 1500) -> float:
    idx = torch.randperm(X.shape[0])[:n]
    return float(torch.pdist(X[idx].double()).median())


# ============================================================================
# Exact-GP hyperparameter selection (validation-grid search: lengthscale + noise pair)
# ============================================================================


def _score_cell(
    Xt: torch.Tensor,
    y: torch.Tensor,
    Xv: torch.Tensor,
    vv: torch.Tensor,
    gv: torch.Tensor,
    osv: torch.Tensor,
    cell: tuple[float, float, float],
) -> float:
    """Fit one (lengthscale, value noise, derivative noise) cell on the HP train sub-split and return
    value+gradient RMSE on the val split. Returns inf if the CG solve did not converge -- that cell's
    score would be meaningless, and small noise makes (K+Lambda) harder to solve, so this must be
    checked per cell now that the noise is itself being searched."""
    ntr, D = Xt.shape
    ls, noise, deriv_noise = cell
    kernel = _KERNELS[KERNEL](torch.full((D,), ls, device=DEV, dtype=DT), osv)
    lam = noise_vector(ntr, D, noise, deriv_noise, DEV, DT)
    alpha = solve_cg(Xt, y, lam, kernel, max_iter=CG_ITERS, tol=CG_TOL)
    rel = float(
        torch.linalg.norm(kernel.mvm(Xt, Xt, alpha) + lam * alpha - y) / torch.linalg.norm(y)
    )
    if rel > 10 * CG_TOL:
        return float("inf")
    return rmse(predict_value(Xv, Xt, alpha, kernel), vv) + rmse(
        predict_grad(Xv, Xt, alpha, kernel), gv
    )


def select_hypers(
    Xtr: torch.Tensor,
    vtr: torch.Tensor,
    gtr: torch.Tensor,
) -> tuple[float, float, float]:
    """Pick the exact GP's (lengthscale, value noise, derivative noise) by validation search: fit on a
    train sub-split, score value+gradient RMSE on a held-out val split, keep the best cell.

    The noise pair is searched because TERA learns its sigma_f and sigma_g as INDEPENDENT parameters;
    an exact arm with the two nailed together at one hard-coded constant is not equally equipped, and
    a TERA win could then be tuning asymmetry rather than method. Value and derivative noise trade off
    against each other, so the pair is swept JOINTLY (full NOISE_GRID x DERIV_NOISE_GRID product).

    Search is COORDINATE, not the full three-way product: lengthscale, then the noise pair, then the
    lengthscale again at the selected noise (19 cells, vs 45 for the product) -- HP selection is
    counted in fit_s, so the budget is real. That means an interaction strong enough to need a joint
    lengthscale/noise sweep can be missed. Falls back to (median heuristic, EXACT_NOISE,
    EXACT_DERIV_NOISE) if no cell's CG solve converges."""
    nsub = min(Xtr.shape[0], HP_SUBSAMPLE)
    ntr = int(nsub * (1.0 - VAL_FRAC))
    Xt, vt, gt = Xtr[:ntr], vtr[:ntr], gtr[:ntr]
    Xv, vv, gv = Xtr[ntr:nsub], vtr[ntr:nsub], gtr[ntr:nsub]

    base = median_lengthscale(Xt)
    osv = torch.tensor(float(vt.var()), device=DEV, dtype=DT)
    y = stack_targets(vt, gt)

    best_cell = (base, EXACT_NOISE, EXACT_DERIV_NOISE)
    best_score = float("inf")

    def sweep(cells: list[tuple[float, float, float]]) -> None:
        nonlocal best_cell, best_score
        for cell in cells:
            score = _score_cell(Xt, y, Xv, vv, gv, osv, cell)
            if score < best_score:
                best_score, best_cell = score, cell

    # stage 1: lengthscale at the starting noise
    sweep([(base * mult, EXACT_NOISE, EXACT_DERIV_NOISE) for mult in LS_GRID])
    # stage 2: the noise PAIR, jointly, at the selected lengthscale
    ls = best_cell[0]
    sweep([(ls, n, dn) for n in NOISE_GRID for dn in DERIV_NOISE_GRID])
    # stage 3: lengthscale again at the selected noise pair
    _, noise, deriv_noise = best_cell
    sweep([(base * mult, noise, deriv_noise) for mult in LS_GRID])
    return best_cell


# ============================================================================
# The two arms
# ============================================================================


def exact_arm(
    Xtr: torch.Tensor,
    vtr: torch.Tensor,
    gtr: torch.Tensor,
    Xte: torch.Tensor,
    vte: torch.Tensor,
    gte: torch.Tensor,
) -> dict[str, float]:
    """Base exact derivative GP: the KERNEL kernel, validation-grid selection of the lengthscale and
    the (value, derivative) noise pair, base CG solve. Selection AND the final fit are both on the
    clock (fair against TERA's self-tuning, which also learns sigma_f and sigma_g)."""
    D = Xtr.shape[1]
    t = time.time()
    ls, noise, deriv_noise = select_hypers(Xtr, vtr, gtr)
    osv = torch.tensor(float(vtr.var()), device=DEV, dtype=DT)
    kernel = _KERNELS[KERNEL](torch.full((D,), ls, device=DEV, dtype=DT), osv)
    y = stack_targets(vtr, gtr)
    lam = noise_vector(Xtr.shape[0], D, noise, deriv_noise, DEV, DT)
    alpha = solve_cg(Xtr, y, lam, kernel, max_iter=CG_ITERS, tol=CG_TOL)
    resid = kernel.mvm(Xtr, Xtr, alpha) + lam * alpha - y
    rel = float(torch.linalg.norm(resid) / torch.linalg.norm(y))
    fit_s = time.time() - t

    return dict(
        val_rmse=rmse(predict_value(Xte, Xtr, alpha, kernel), vte),
        grad_rmse=rmse(predict_grad(Xte, Xtr, alpha, kernel), gte),
        lengthscale=ls,
        noise=noise,
        deriv_noise=deriv_noise,
        cg_rel_resid=rel,
        fit_s=fit_s,
    )


def tera_arm(
    Xtr: torch.Tensor,
    vtr: torch.Tensor,
    gtr: torch.Tensor,
    Xte: torch.Tensor,
    vte: torch.Tensor,
    gte: torch.Tensor,
) -> dict[str, float]:
    """TERA, the best approximate derivative GP: self-learns its hyperparameters and fits."""
    v_mean, v_var, grad_rmse, fit_s = run_tera(
        Xtr, vtr, gtr, Xte, vte, gte, m=TERA_M, kernel=KERNEL, train_epochs=TERA_EPOCHS
    )
    return dict(
        val_rmse=rmse(v_mean, vte),
        val_nll=gaussian_nll(vte, v_mean, v_var),
        grad_rmse=grad_rmse,
        fit_s=fit_s,
    )


# ============================================================================
# Driver
# ============================================================================


def config() -> dict:
    """Every parameter / hyperparameter / choice for this benchmark, recorded alongside the results."""
    return {
        "device": str(DEV),
        "dtype": str(DT).replace("torch.", ""),
        "seeds": list(SEEDS),
        "n_train": NTRAIN,
        "n_test": NTEST,
        "kernel": KERNEL,
        "exact": {
            "cg_iters": CG_ITERS,
            "cg_tol": CG_TOL,
            "noise_init": EXACT_NOISE,
            "deriv_noise_init": EXACT_DERIV_NOISE,
            "ls_grid": list(LS_GRID),
            "noise_grid": list(NOISE_GRID),
            "deriv_noise_grid": list(DERIV_NOISE_GRID),
            "val_frac": VAL_FRAC,
            "hp_subsample": HP_SUBSAMPLE,
        },
        "tera": {"m": TERA_M, "epochs": TERA_EPOCHS},
    }


def main(datasets: dict, out_path: str) -> None:
    print(f"n-body benchmark: base exact DGP vs TERA, {len(SEEDS)} seeds  dev={DEV}")
    header = f"{'D':>4}{'N':>7}{'seed':>5} | {'exact_grad':>11}{'exact_val':>11}{'resid':>9} | {'tera_grad':>11}{'tera_val':>11}{'tera_nll':>10}"
    print(header)

    rows = []
    for D, path in datasets.items():
        for seed in SEEDS:
            torch.manual_seed(seed)
            data = prep(path, seed)
            ex = exact_arm(*data)
            te = tera_arm(*data)
            rows.append({"D": D, "dataset": path, "seed": seed, "exact": ex, "tera": te})
            print(
                f"{D:>4}{NTRAIN:>7}{seed:>5} | {ex['grad_rmse']:>11.4f}{ex['val_rmse']:>11.4f}"
                f"{ex['cg_rel_resid']:>9.1e} | {te['grad_rmse']:>11.4f}{te['val_rmse']:>11.4f}"
                f"{te['val_nll']:>10.3f}",
                flush=True,
            )

    print("\n=== summary (gradient RMSE, mean +- sd over seeds) ===")
    print(f"{'D':>4} | {'exact_grad':>16}{'tera_grad':>16}")
    summary = {}
    for D in datasets:
        eg = np.array([r["exact"]["grad_rmse"] for r in rows if r["D"] == D])
        tg = np.array([r["tera"]["grad_rmse"] for r in rows if r["D"] == D])
        summary[D] = {
            "exact_grad_mean": float(eg.mean()),
            "exact_grad_std": float(eg.std()),
            "tera_grad_mean": float(tg.mean()),
            "tera_grad_std": float(tg.std()),
        }
        print(f"{D:>4} | {eg.mean():>7.4f}±{eg.std():<8.4f}{tg.mean():>7.4f}±{tg.std():<8.4f}")

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": config(), "results": rows, "summary": summary}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="N-body benchmark: base exact DGP vs TERA")
    parser.add_argument(
        "--datasets",
        default=",".join(str(d) for d in DATASETS),
        help="comma-separated state dims D to run (default: all)",
    )
    parser.add_argument(
        "--out", default="runs/nbody_benchmark.json", help="JSON results output path"
    )
    args = parser.parse_args()
    chosen = {int(d): DATASETS[int(d)] for d in args.datasets.split(",")}
    main(chosen, args.out)
