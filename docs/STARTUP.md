# Startup — first-time setup for a fresh clone

Read this before doing anything else. This repo is **self-contained**: an LLM-driven research
(*autoresearch*) loop on derivative Gaussian processes over an N-body Hamiltonian system, with two
reference points —
the **base exact derivative GP** (the ground-truth starting point) and **TERA** (the best
approximate GP baseline). A fresh clone does **not** contain the datasets or the TERA source;
you generate the data locally and pull TERA as a git submodule. Do the steps below in order.

## 1. Pull the TERA submodule

TERA is not vendored into this repo — it is a git submodule at `gp/tera/vendor` pointing at the
upstream repo `https://github.com/hseung88/tera.git` (see `.gitmodules`), pinned to a specific commit.
Populate it:

```bash
git submodule update --init --recursive
```

(Or clone with `git clone --recursive <this repo>` in the first place.) After this,
`gp/tera/vendor/` contains the upstream TERA repo (importable packages under `src/`) and
`from gp.tera import run_tera` works. The thin wrapper `gp/tera/__init__.py` stays in this repo (it
depends on `gp.metrics` and adds `gp/tera/vendor/src` to `sys.path`); only the upstream TERA source
lives in the submodule. Nothing needs to be created or hosted — it pulls straight from upstream.

## 2. Install the environment

```bash
uv sync            # installs runtime deps + the dev group (ruff, pytest) from uv.lock
```

Everything runs through uv: `uv run python ...`, `uv run pytest`, `uv run ruff check .`.
Do not `pip install` ad hoc or add dependencies mid-loop.

## 3. Generate the N-body data

The datasets are large and are **not** committed (`data/nbody/*.npz` is gitignored). Generate them
with the frozen generator, which writes `data/nbody/nbody_n{n}_d3.npz`. It writes to `nbody/`
relative to the working directory, so run it from `data/`:

```bash
cd data
for n in 2 4 6 8 10; do
    uv run python get_nbody.py --n_particles $n --n_dims 3 --n_samples 10000
done
cd ..
```

The starter benchmark only needs `nbody_n2_d3.npz` (D=12) and `nbody_n10_d3.npz` (D=60); generating
all five gives the full D-sweep. Each file holds `X` (state `[q,p]`), `E` (value), `F` (gradient).

## 4. Verify the install

```bash
uv run pytest -q                                   # 17 tests: the factored/blocked matvec == dense, CG == exact
uv run python experiments/nbody_benchmark.py       # base exact DGP vs TERA, grad RMSE + val RMSE/NLL
```

If the tests pass and the benchmark prints a table for both arms, the setup is correct.

## What is frozen / external

- `gp/tera/` — the TERA baseline (submodule under `vendor/`, plus a thin wrapper). **Never edit.**
- `data/get_nbody.py` — the dataset generator. Frozen.
- `gp/kernels/`, `gp/cg/`, `gp/exact/`, `gp/common/` — the base exact derivative GP. This is the
  code the research loop develops against; see `CLAUDE.md` for the loop rules and `docs/REFERENCES.md`
  for the prior work each piece builds on.
