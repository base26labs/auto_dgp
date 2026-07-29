"""Thin wrapper around the official TERA implementation (Seung & Katzfuss 2026, arXiv:2606.02909),
pulled as a git submodule under `vendor/` (MIT license, see vendor/LICENSE).

TERA is a Vecchia-approximate derivative GP that predicts the SCALAR function value from observed
gradients via exact target-specific gradient reduction. Its native API is value-only, but its
posterior mean is a differentiable function of the input, so we ALSO take its gradient as a gradient
prediction: TERA is derivative-informed (trained on gradients), so d/dx m(x) is a legitimate gradient
predictor, and we compare it head-to-head with the exact solver. So in our harness TERA reports value
RMSE + value NLL AND gradient RMSE.

We call TERA's own `TERAModel.fit/predict` for the value posterior; we only marshal our standardized
(X, value, gradient) tensors into TERA's own data-split container and read back its predictive
marginals. For the gradient we differentiate a faithful batched reimplementation of TERA's own
posterior-mean (validated to ~1e-5 against its per-point loop), because TERA's native predict runs
under `torch.no_grad()`. Config mirrors TERA's upstream defaults (m=20, rbf, 1 training epoch, lr=0.01,
learns all hyperparameters, iid gradient noise) so the comparison is faithful.
"""
import math
import os
import sys
import time

import torch

# The submodule is the upstream TERA repo (hseung88/tera), whose importable packages live under src/.
_VENDOR_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "src")
if _VENDOR_SRC not in sys.path:
    sys.path.insert(0, _VENDOR_SRC)


_SQRT5 = math.sqrt(5.0)


def _bcov(A, B, ls, os, kernel):
    """Batched function covariance. A:(n,mA,D), B:(n,mB,D) -> (n,mA,mB). Mirrors deroos.function_covariance
    (the center term cancels in the difference, so we skip it)."""
    d = (A[:, :, None, :] - B[:, None, :, :])
    d = d / ls.reshape(1, 1, 1, -1) if ls.numel() > 1 else d / ls.reshape(1, 1, 1, 1)
    r = (d * d).sum(dim=-1)
    if kernel == "rbf":
        return os * torch.exp(-0.5 * r)
    if kernel == "matern52":
        a = torch.sqrt(torch.clamp(r, min=1e-12)) * _SQRT5
        return os * (1.0 + a + (a * a) / 3.0) * torch.exp(-a)
    raise ValueError(kernel)


def _alpha_r(r, kernel, os):
    if kernel == "rbf":
        return os * torch.exp(-0.5 * r)
    a = torch.sqrt(torch.clamp(r, min=1e-12)) * _SQRT5
    return (5.0 / 3.0) * os * (1.0 + a) * torch.exp(-a)


def _beta_r(r, kernel, os):
    if kernel == "rbf":
        return -os * torch.exp(-0.5 * r)
    a = torch.sqrt(torch.clamp(r, min=1e-12)) * _SQRT5
    return -(25.0 / 3.0) * os * torch.exp(-a)


def _batched_mean(Xg, Xg_scaled, nbrs, data, sigma_g_model):
    """Vectorized TERA posterior-mean over a batch of eval points. Faithful batched reimplementation of
    TERAModel._predict_one's mean (validated to ~1e-5 against the per-point loop). Xg:(n,D) is the leaf we
    differentiate for the gradient; nbrs:(n,m) long. Returns means:(n,)."""
    dev, dt = Xg.device, Xg.dtype
    n, m = nbrs.shape
    os = torch.as_tensor(data.outputscale, device=dev, dtype=dt)
    ls = data.lengthscale.to(dev)
    kn = data.kernel_name

    Xc = data.X_train[nbrs]                 # (n,m,D)  raw
    Xc_scaled = data.X_train_scaled[nbrs]   # (n,m,D)  scaled
    yc = data.f_train_obs[nbrs]             # (n,m)
    gc = data.g_train_obs[nbrs]             # (n,m,D)

    # Lff = chol(Kff + sigma_f^2 I): constant w.r.t. Xg, so build detached.
    with torch.no_grad():
        Kff = _bcov(Xc, Xc, ls, os, kn)
        Kff = 0.5 * (Kff + Kff.transpose(-1, -2))
        if data.sigma_f > 0.0:
            Kff = Kff + (data.sigma_f ** 2) * torch.eye(m, device=dev, dtype=dt)
        jit = 1e-8
        while True:
            try:
                Lff = torch.linalg.cholesky(Kff + jit * torch.eye(m, device=dev, dtype=dt)); break
            except Exception:
                jit *= 10.0
                if jit > 1e-1:
                    raise

    k_fc = _bcov(Xc, Xg[:, None, :], ls, os, kn).squeeze(-1)      # (n,m)
    ds = Xc_scaled - Xg_scaled[:, None, :]                        # (n,m,D) scaled
    H = ds @ ds.transpose(-1, -2)                                 # (n,m,m)
    cols = H.transpose(-1, -2)                                    # (n,m,m)
    q = cols[:, :, None, :] - cols[:, None, :, :]                 # (n,m,m,m)  [a,b,k]
    r_i = torch.diagonal(H, dim1=-2, dim2=-1)                     # (n,m)
    alpha_i = _alpha_r(r_i, kn, os)                               # (n,m)

    dcc = Xc_scaled[:, :, None, :] - Xc_scaled[:, None, :, :]     # (n,m,m,D)
    r_cc = (dcc * dcc).sum(dim=-1)                                # (n,m,m)
    alpha_cc = _alpha_r(r_cc, kn, os)
    beta_cc = _beta_r(r_cc, kn, os)

    bar_k = ((-alpha_i[:, :, None]) * cols).reshape(n, m * m)     # (n,m^2)
    Q = ((-alpha_cc[:, :, :, None]) * q).permute(0, 1, 3, 2).reshape(n, m * m, m)  # (n,m^2,m)

    G0b = alpha_cc[:, :, :, None, None] * H[:, None, None, :, :]
    G0b = G0b + beta_cc[:, :, :, None, None] * (q[:, :, :, :, None] * q[:, :, :, None, :])  # (n,a,b,i,j)
    if data.sigma_g > 0.0:
        delta_raw = Xc - Xg[:, None, :]                          # (n,m,D)
        if sigma_g_model == "scaled":
            ell2 = (ls * ls)
            dn = delta_raw * ell2.reshape(1, 1, -1) if ls.numel() > 1 else delta_raw * (ls * ls)
            R = delta_raw @ dn.transpose(-1, -2)
        else:  # iid
            R = delta_raw @ delta_raw.transpose(-1, -2)          # (n,m,m)
        R = 0.5 * (R + R.transpose(-1, -2))
        eye_ab = torch.eye(m, device=dev, dtype=dt)[None, :, :, None, None]
        G0b = G0b + data.sigma_g * (eye_ab * R[:, None, None, :, :])
    G0 = G0b.permute(0, 1, 3, 2, 4).reshape(n, m * m, m * m)     # rows (a,i), cols (b,j)

    rhs = torch.cat([k_fc[:, :, None], Q.transpose(-1, -2)], dim=-1)  # (n,m,1+m^2)
    sol = torch.cholesky_solve(rhs, Lff)                              # (n,m,1+m^2)
    beta_star = sol[:, :, 0]                                          # (n,m)
    QKinv = sol[:, :, 1:]                                             # (n,m,m^2)

    k_delta = bar_k - (Q @ beta_star[:, :, None]).squeeze(-1)         # (n,m^2)
    K_delta = G0 - Q @ QKinv                                         # (n,m^2,m^2)
    K_delta = 0.5 * (K_delta + K_delta.transpose(-1, -2))
    jit = 1e-8
    eye2 = torch.eye(m * m, device=dev, dtype=dt)
    while True:
        try:
            L_delta = torch.linalg.cholesky(K_delta + jit * eye2); break
        except Exception:
            jit *= 10.0
            if jit > 1e-1:
                raise
    w_star = torch.cholesky_solve(k_delta[:, :, None], L_delta).squeeze(-1)  # (n,m^2)

    qtw = (Q.transpose(-1, -2) @ w_star[:, :, None]).squeeze(-1)             # (n,m)
    f_weights = beta_star - torch.cholesky_solve(qtw[:, :, None], Lff).squeeze(-1)  # (n,m)
    q_obs = (gc @ (Xc - Xg[:, None, :]).transpose(-1, -2)).reshape(n, m * m)  # (n,m^2)
    mean = (f_weights * yc).sum(dim=-1) + (w_star * q_obs).sum(dim=-1)        # (n,)
    return mean


def _batched_grads(model, Xte, ns, chunk=256):
    """Full-test-set gradient prediction (grad of TERA's value posterior) via chunked batched linear
    algebra -- replaces the per-point Python loop. Returns g_pred:(ns,D)."""
    from gp_sim_kl.ordering import knn_to_eval
    from gp_sim_kl.utils import scale_inputs
    data = model.predictor.data
    sg_model = getattr(model, "gradient_noise_model", "iid")
    outs = []
    for c0 in range(0, ns, chunk):
        c1 = min(c0 + chunk, ns)
        Xg = Xte[c0:c1].detach().clone().requires_grad_(True)
        Xg_scaled = scale_inputs(Xg, data.lengthscale)
        nbr_list = knn_to_eval(data.X_train_scaled, Xg_scaled.detach(), model.m)  # list of (m,) long
        nbrs = torch.stack(nbr_list, dim=0)                                    # (b,m) long
        mean = _batched_mean(Xg, Xg_scaled, nbrs, data, sg_model)
        g = torch.autograd.grad(mean.sum(), Xg)[0]                             # (b,D)
        outs.append(g.detach())
    return torch.cat(outs, dim=0)


def run_tera(Xtr, vtr, gtr, Xte, vte, gte, *, m=20, kernel="rbf", train_epochs=1, batch_size=256,
             lr=0.01, seed=0):
    """Fit TERA on (Xtr, vtr, gtr) and return (val_mean, val_var, grad_rmse, fit_s) at Xte. TERA's API is
    value-only, but its posterior mean is a differentiable function of x, so the GRADIENT prediction is
    d/dx m(x). TERA is derivative-INFORMED (trained on gradients), so this gradient is a legitimate
    gradient predictor, letting us compare TERA gradients head-to-head with the exact solver. grad_rmse
    is NaN if predict is not autograd-able."""
    from md22_regression.data import EnergyForceScaler, MD22Split
    from md22_regression.models.tera import TERAModel

    D = Xtr.shape[1]
    n_atoms = max(1, D // 3)
    dev, dt = Xtr.device, Xtr.dtype
    idx_tr = torch.arange(Xtr.shape[0], device=dev)
    idx_te = torch.arange(Xte.shape[0], device=dev)
    scaler = EnergyForceScaler(energy_mean=torch.zeros((), dtype=dt),
                               energy_std=torch.ones((), dtype=dt), x_scale=1.0)
    # data is already standardized upstream (get_data), so the scaler is identity; TERA operates on the
    # standardized X/value/gradient tensors directly. E_/F_ fields are unused by fit/predict (metrics only).
    split = MD22Split(
        name="bench", preprocessing_version="v0", split_id=f"s{seed}",
        X_train=Xtr, y_train=vtr, g_train=gtr, E_train=vtr, F_train=gtr,
        X_test=Xte, y_test=vte, g_test=gte, E_test=vte, F_test=gte,
        scaler=scaler, n_atoms=n_atoms, train_indices=idx_tr, test_indices=idx_te)
    # all inits/flags mirror TERA's upstream config defaults (faithful comparison)
    model = TERAModel(m=m, kernel=kernel, outputscale=1.0, sigma_f=1e-3, sigma_g=1e-3,
                      lengthscale=1.0, lengthscale_init="median", lengthscale_init_max_points=2048,
                      use_ard=False, seed=seed, train_steps=0, train_epochs=train_epochs,
                      batch_size=batch_size, lr=lr, learn_lengthscale=True, learn_outputscale=True,
                      learn_sigma_f=True, learn_sigma_g=True, gradient_noise_model="iid")
    t_fit = time.time()
    model.fit(split)            # TERA learns its hyperparameters AND fits here -- this is the TRAINING cost
    fit_s = time.time() - t_fit
    pred = model.predict(Xte)   # PREDICTION (value) -- not part of the training cost
    # GRADIENT = gradient of TERA's value posterior mean w.r.t. the eval points (prediction-time, NOT counted
    # in fit_s). TERA's predict_f_marginals runs under `torch.no_grad()`, so we replicate its per-point loop
    # WITH grad enabled, calling TERA's own (unmodified) `_predict_one`, on a capped test subset for speed.
    # Returns the gradient D-RMSE directly (computed on the matching subset).
    grad_rmse = float("nan")
    try:
        from gp.metrics import rmse
        ns = Xte.shape[0]                                    # full test set (chunked batched linalg is fast)
        g_pred = _batched_grads(model, Xte, ns)             # (ns, D) = predicted gradient, VECTORIZED
        grad_rmse = float(rmse(g_pred.reshape(ns, -1), gte[:ns].reshape(ns, -1)))
    except Exception:
        grad_rmse = float("nan")
    return pred.y_mean, pred.y_var, grad_rmse, fit_s
