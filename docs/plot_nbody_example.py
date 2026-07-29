"""Generate docs/nbody_example.png — an illustration of what the derivative GP fits.

Left: one 2-body trajectory in position space, with force arrows (the position part of the gradient
∇H the GP observes at each state). Right: sampled phase-space states colored by their Hamiltonian
value H (the scalar the GP fits), projected to 2 principal components.

Run:  uv run python docs/plot_nbody_example.py
"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
from get_nbody import generate_nbody_dataset, hamiltonian_dynamics, hamiltonian_gradients

BLUE, AQUA = "#2a78d6", "#1baf7a"  # validated categorical slots 1, 2
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BLUES = LinearSegmentedColormap.from_list("brand_blues", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])

# ---- Panel A: one 2-body trajectory (d=3), positions projected to the xy-plane ----
np.random.seed(7)
n, d = 2, 3
masses = np.array([1.6, 1.0])
q0 = np.array([[-1.2, 0.0, 0.0], [1.0, 0.2, 0.0]])
p0 = np.array([[0.0, 0.55, 0.0], [0.0, -0.80, 0.0]])
q0 -= np.average(q0, weights=masses, axis=0)
p0 -= np.average(p0, weights=masses, axis=0)
s0 = np.concatenate([q0.ravel(), p0.ravel()])
sol = solve_ivp(
    hamiltonian_dynamics, (0, 7.0), s0, args=(masses, 1.0, 0.1),
    t_eval=np.linspace(0, 7.0, 500), method="DOP853", rtol=1e-10, atol=1e-12,
)
Q = sol.y[: n * d].reshape(n, d, -1)  # positions (n, d, T)
force = np.zeros((sol.t.size, n, d))
for k in range(sol.t.size):
    q = sol.y[: n * d, k].reshape(n, d)
    p = sol.y[n * d :, k].reshape(n, d)
    dHdq, _ = hamiltonian_gradients(q, p, masses)
    force[k] = -dHdq  # force = -dH/dq

# ---- Panel B: a small dataset, PCA-projected states colored by energy ----
X, E, _, _ = generate_nbody_dataset(
    n_particles=2, n_dims=3, n_samples=400, n_trajectories=40, steps_per_trajectory=60, seed=1
)
Xc = X - X.mean(0)
_, _, Vt = np.linalg.svd(Xc, full_matrices=False)
P2 = Xc @ Vt[:2].T

# ---- draw ----
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor="white")
fig.suptitle(
    "What the derivative GP fits: value H(x) and gradient ∇H(x) at phase-space states x = [q, p]",
    fontsize=12, color=INK, y=1.02,
)

for i, (c, lab) in enumerate([(BLUE, "particle 1"), (AQUA, "particle 2")]):
    axA.plot(Q[i, 0], Q[i, 1], color=c, lw=2, alpha=0.9, label=lab, zorder=2)
idx = np.linspace(20, sol.t.size - 20, 9).astype(int)
span = max(np.ptp(Q[:, 0]), np.ptp(Q[:, 1]))
for i, c in enumerate([BLUE, AQUA]):
    fx, fy = force[idx, i, 0], force[idx, i, 1]
    nrm = np.hypot(fx, fy) + 1e-9
    axA.quiver(
        Q[i, 0, idx], Q[i, 1, idx], fx / nrm * 0.12 * span, fy / nrm * 0.12 * span,
        color=c, alpha=0.55, angles="xy", scale_units="xy", scale=1, width=0.006, headwidth=4, zorder=3,
    )
    axA.scatter(Q[i, 0, idx], Q[i, 1, idx], s=14, color=c, edgecolor="white", linewidth=0.5, zorder=4)
axA.set_title("gradient ∇H observed at each state (forces →)", fontsize=10.5, color=INK)
axA.set_xlabel("position x", color=MUTED)
axA.set_ylabel("position y", color=MUTED)
axA.set_aspect("equal")
axA.legend(frameon=False, fontsize=9, labelcolor=INK)

sc = axB.scatter(P2[:, 0], P2[:, 1], c=E, cmap=BLUES, s=16, edgecolor="white", linewidth=0.3)
cb = fig.colorbar(sc, ax=axB, fraction=0.046, pad=0.03)
cb.set_label("Hamiltonian value H (energy)", color=INK, fontsize=9)
cb.outline.set_edgecolor(GRID)
axB.set_title("scalar value H the GP fits, over phase space", fontsize=10.5, color=INK)
axB.set_xlabel("state PC 1", color=MUTED)
axB.set_ylabel("state PC 2", color=MUTED)

for ax in (axA, axB):
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nbody_example.png")
fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
