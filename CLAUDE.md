# auto_dgp — a testbed for studying the research capabilities of LLMs

This repo is a self-contained testbed for **the research capabilities of LLMs**: the object of study is
how well an LLM-driven research loop (an *autoresearch* loop, inspired by Karpathy's) actually works.
The loop is human-steered: it runs unattended within a session, but a human sets the direction.
**Derivative-observation Gaussian processes** (GPs that observe function *values and gradients*) are
the substrate — a concrete, checkable domain the loop operates in so its progress can be measured —
not the research goal in themselves. The substrate task is a Hamiltonian N-body system: predict the
target value (and, where the method supports it, its gradient) from gradient observations.

Two reference points ship in the repo as **frozen starting baselines**:

- **TERA** (`gp/tera/`) — the paper baseline the loop starts from: a Vecchia-approximate
  derivative GP that predicts the scalar value from observed gradients.
- **Exact DGP** (`gp/exact/` + `gp/cg/`) — the ground-truth dense/iterative solver: the base
  conjugate-gradient method with **no preconditioner**. The oracle new methods are measured against.

Within the substrate the loop's concrete objective is to beat these on value RMSE **and** NLL under a
fixed budget — but what we are actually evaluating is the research process itself. Everything
needed is in this directory; the doc below is complete on its own.

## Status

Just starting. Data, both baselines (TERA, exact DGP), and a first benchmark
(`experiments/nbody_benchmark.py`, which loads + standardizes the data and scores both arms) are in
place. `results.tsv` and `docs/RESEARCH_LOG.md` exist but are **empty** (header row / no entries);
`docs/learnings.md` does not exist yet. No baseline has been recorded, so no number is yet
interpretable. The exact arm selects its lengthscale **and** its (value, derivative) noise pair on a
validation split, so it is equipped comparably to TERA's learned `sigma_f`/`sigma_g`.
**Next: record the baselines** (run the benchmark, log the numbers) before any experiments.

## Layout

| Path | What | Editable? |
|------|------|-----------|
| `data/nbody/*.npz` | Hamiltonian N-body datasets (see below) | frozen data — never regenerate silently |
| `data/get_nbody.py` | dataset generator | frozen |
| `gp/tera/__init__.py` | `run_tera(...)` wrapper around the official TERA implementation | external — do not edit |
| `gp/tera/vendor/` | official TERA implementation, unmodified (MIT) | **NEVER edit** |
| `gp/kernels/` | `DerivKernel` base + RBF / Matern-5/2 derivative kernels (shared math core) | yes, but it's load-bearing for every method |
| `gp/cg/` | base (unpreconditioned) conjugate-gradient solver `solve_cg` | yes |
| `gp/exact/` | dense `solve_exact`, posterior mean/grad prediction, `gaussian_nll` | yes |
| `gp/common/` | target/noise assembly (`stack_targets`, `noise_vector`) | yes |
| `gp/metrics.py` | shared metrics | frozen once harness exists |
| `experiments/fNN_*.py` | one file per experiment, registered in `experiments/expt_registry.py` (created with the first experiment) | yes — this is where work happens |

**Data format** (`np.load`): `X` — state `[q, p]` flattened, shape `(N, 2·n·d)`; `E` — the scalar
target value `(N,)`; `F` — its full gradient `dH/dx` `(N, 2·n·d)`. Files:
`nbody_n{2,4,6,8,10}_d3.npz` (~9.5k rows) and `*_100k.npz` (100k rows).

**Derivative-GP system.** Methods solve the dual `(K + Λ) α = y`, where `y = stack_targets(E, F)`
is the per-point `[value, grad…]` stack (length `N·(D+1)`), `Λ = noise_vector(...)` is the
diagonal `[noise, deriv_noise·D]` per point, and `K` is the derivative kernel from `gp/kernels`.
`gp/exact.solve_exact` forms the dense system (small N only); `gp/cg.solve_cg` solves it
matrix-free via the factored `O(N²D)` kernel matvec. They agree to ~1e-10 on well-conditioned
toy problems — that equivalence is the regression test for any kernel/solver change.

## Environment

- **uv only.** Run everything as `uv run python ...`; add deps by editing `pyproject.toml`
  then `uv lock`. Never `pip install` ad hoc, and do not add dependencies mid-loop (comparability
  includes the software stack).
- **Python 3.12**, modern type annotations (`list[str]`, `X | None`, no `typing.Optional`).
- **Lint before commit:** `uv run ruff check .` (all of `gp/tera` is excluded from linting).

## The loop

1. **Fixed, comparable budget.** Every run gets the same wall-clock/epoch/data budget, decided
   once when the harness is built and then never changed. You can never "win" by quietly spending
   more.
2. **Baselines first.** The first runs are the unmodified TERA and exact-DGP baselines. Every
   later number is read against them. No baseline recorded → no result is interpretable.
3. **Frozen scorer.** The evaluation path — the data loading + standardization in
   `experiments/nbody_benchmark.py` and the metrics in `gp/metrics.py` — is read-only. Editing the
   scorer is the easiest way to manufacture a fake win. If it is genuinely broken, fix it, say so
   loudly, and re-run the baselines — all prior numbers are then void.
4. **Log every run to `results.tsv`** (tab-separated, never commas — they break in descriptions;
   keep the file untracked by git):
   `commit  model  val_rmse  val_nll  dataset  status  description`
   with `status ∈ {keep, discard, crash}`; use `0.000000` for crashes. Check **both** accuracy and
   calibration — an RMSE win with blown-up NLL is overconfidence, not a win.
5. **Git workflow:** commit the change *before* running; if the result improves, advance the
   branch; if equal/worse, `git reset` back. Redirect run output to a log file
   (`> runs/run.log 2>&1`) and `grep` the metrics out — never let training output flood context.
   Rewind across multiple commits only very sparingly.
6. **Crashes & timeouts:** trivial bug (typo, shape, import) → fix and re-run. Fundamentally broken
   idea (OOM, numerically unstable) → log `crash`, move on. Kill any run at ~2× its budget and
   treat it as a failure.
7. **Simplicity criterion.** A tiny gain that adds ugly complexity is not worth keeping;
   equal-or-better results from *deleting* code is a first-class win.
8. **`docs/learnings.md` is long-term memory.** Distill every ~10 experiments: what works (with
   evidence), what doesn't, open questions, current-best table. *Distill, don't append* — remove
   superseded findings. Read it at the start of every session before choosing the next experiment.
9. **Never stop, but never flog a dead hypothesis.** Once looping, don't pause to ask "should I
   continue?" — the human may be asleep. But autonomy is not license to re-attack a dead idea in a
   new hat: after 2–3 honest ties/negatives, the verdict is in — pivot or distill the negative.
   Out of ideas → literature search, re-read the in-scope files, combine near-misses; don't rerun
   the graveyard.

## Before believing any number — research hygiene

A toy result is a *hypothesis*, not a finding, until it survives controls at sufficient scale.
The traps this repo is most likely to fall into:

- **Smoke runs are not findings.** Run the full budget, **≥3 seeds, report mean ± spread**, before
  anything is called a result. Numbers that only appear small/short/single-seed are artifacts until
  they survive the full run.
- **Headroom & necessity.** A fair baseline must be below ceiling *and* actually need the thing
  you're testing. The N-body Hamiltonian is **separable** (`H = T(p) + V(q)`) with an analytically
  trivial kinetic part — if a baseline already nails the task by exploiting that structure, the
  task can't show your effect. A tie at ceiling proves nothing.
- **Structure learned, not handed.** No computing the answer in Python; no fixed
  homomorphism/relation credited as a learned capability. Diff what the docstring claims against
  what `forward`/the solver actually does.
- **Fair, strong baseline.** TERA's config mirrors its own defaults (documented in the wrapper
  docstring) — don't detune it and then beat it. TERA's native API is value-only, but since its
  posterior mean is differentiable and it is trained on gradients, we take the gradient of its value
  posterior as a legitimate gradient prediction — so the gradient RMSE head-to-head against the exact
  DGP is fair. TERA also reports value NLL (from its predictive variance); the base exact DGP is
  mean-only, so its NLL column is N/A until a predictive-variance path is added.
- **Ablate what you credit.** Freeze/remove the component you claim is doing the work; if the score
  barely moves, it's decorative. Add a shuffle/permutation null and an untrained-model control
  before reporting a positive.
- **Trust the "that shouldn't work" reflex.** A mechanistically backwards result is a red flag, not
  a discovery — build the control that would expose the artifact *before* writing it up.

Honest negatives are the deliverable. A retraction caught by a control is a success of this
process, not a failure.

## Conventions

- `docs/RESEARCH_LOG.md` is the single narrative doc; every number in it is real output. Retractions are
  reported in-place, loudly, naming the check that failed.
- `docs/REFERENCES.md` cites the prior work this repo builds on, pulls in, or compares against (TERA, the
  De Roos derivative-kernel algebra behind the factored matvec, Vecchia, the data). Add to it when you
  introduce a method from the literature — know the literature before claiming something is new.
- Experiments are `experiments/fNN_short_name.py` with a pre-registration docstring written
  **before** running: *if this wins, is it real, or is it (i) answer-structure built in / denied to
  the baseline, (ii) task too easy / capability not needed, (iii) a strawman baseline, (iv) a
  smoke/single-seed fluke, (v) a degeneracy, (vi) a component I'm not crediting doing the work, or
  (vii) confounded measurement?* If you can't rule all seven out, the experiment is rigged — fix it
  before trusting the number.
- Commit messages end with:
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
