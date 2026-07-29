# References

Prior work this repo builds on, pulls in, or compares against. Each entry notes where in the code it
shows up.

## The research loop (methodology / inspiration)

The loop this repo is built to study — an LLM agent that edits code, runs a time-boxed experiment,
measures it, keeps or discards the change, and repeats — is the *autoresearch* pattern popularized by:
- Karpathy, A. (2026). *autoresearch.* GitHub. <https://github.com/karpathy/autoresearch>

## Baselines shipped in the repo

**TERA** — `gp/tera/` (pulled as a git submodule under `gp/tera/vendor/`, MIT license, © 2026 Hyunseok
Seung). The paper baseline the loop starts from: a Vecchia-approximate derivative GP that predicts the
scalar function value from observed gradients via target-specific gradient reduction.
- Seung, H. & Katzfuss, M. (2026). *Scalable Derivative Gaussian Processes via Exact Gradient
  Reduction (TERA).* arXiv:2606.02909. <https://arxiv.org/abs/2606.02909>

**Exact derivative GP** — `gp/exact/`, `gp/cg/`, `gp/kernels/`. The ground-truth dense/iterative solver
for the derivative dual `(K + Λ) α = y`, used as the oracle. Derivative (value + gradient) observations
in GPs are classical:
- Solak, E., Murray-Smith, R., Leithead, W. E., Leith, D. J. & Rasmussen, C. E. (2003). *Derivative
  Observations in Gaussian Process Models of Dynamic Systems.* NeurIPS.
  <https://proceedings.neurips.cc/paper/2002/hash/5b8e4fd39d9786228649a8a8bec4e008-Abstract.html>
- Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning,* §9.4.
  MIT Press. <https://gaussianprocess.org/gpml/>

## Core methods

**Stationary derivative-kernel algebra / the factored ("blocked") matvec** — `gp/kernels/base.py`
(`DerivKernel.mvm`). This is the structure that reduces every stationary derivative kernel to three
scalar functions of the scaled squared distance — `K0` (value–value), `A` (value–grad / grad–grad
identity part), `B` (grad–grad outer-product part) — so the matvec is `O(N²D)` and never forms the
`O(D²)` gradient–gradient block. TERA uses the same algebra internally.
- de Roos, F., Gessner, A. & Hennig, P. (2021). *High-Dimensional Gaussian Process Inference with
  Derivatives.* ICML. <https://arxiv.org/abs/2102.07542>

**Matrix-free conjugate gradients for kernel systems** — `gp/cg/` (base, unpreconditioned CG).
- Hestenes, M. R. & Stiefel, E. (1952). *Methods of Conjugate Gradients for Solving Linear Systems.*
  J. Res. Natl. Bur. Stand. <https://doi.org/10.6028/jres.049.044>
- Gardner, J. R., Pleiss, G., Bindel, D., Weinberger, K. Q. & Wilson, A. G. (2018). *GPyTorch:
  Blackbox Matrix–Matrix Gaussian Process Inference with GPU Acceleration.* NeurIPS (matrix-free
  kernel CG / BBMM). <https://arxiv.org/abs/1809.11165>

**Vecchia approximation** — the basis of the TERA baseline.
- Vecchia, A. V. (1988). *Estimation and Model Identification for Continuous Spatial Processes.*
  J. R. Stat. Soc. B. <https://doi.org/10.1111/j.2517-6161.1988.tb01729.x>
- Katzfuss, M. & Guinness, J. (2021). *A General Framework for Vecchia Approximations of Gaussian
  Processes.* Statistical Science. <https://arxiv.org/abs/1708.06302>

## Data

**N-body gravitational Hamiltonian** — `data/get_nbody.py`, `data/nbody/*.npz`. The synthetic substrate:
`H(q,p) = Σ|p_i|²/2m_i − G Σ_{i<j} m_i m_j / √(r² + ε²)`, with Plummer softening `ε` and true gradients.
The toy n-body derivative-observation benchmark this data follows is from:
- Huang, D. (2026). *Scaling Gaussian Process Regression with Full Derivative Observations (DSoftKI).*
  TMLR. <https://arxiv.org/abs/2505.09134>
- Plummer, H. C. (1911). *On the Problem of Distribution in Globular Star Clusters.* MNRAS (the
  softening technique). <https://doi.org/10.1093/mnras/71.5.460>
