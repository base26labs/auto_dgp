# Repository Guidelines

## Project Structure & Module Organization

Core derivative-GP code lives in `gp/`: kernels are in `gp/kernels/`, iterative and dense solvers
in `gp/cg/` and `gp/exact/`, and the physics-informed SPARK model in `gp/spark/`. Put runnable
studies in `experiments/`, dataset generators in `data/`, and correctness tests in `tests/`.
Committed benchmark artifacts belong under `evidence/`; narrative results belong in
`docs/RESEARCH_LOG.md`. Treat `gp/tera/`, legacy `data/get_nbody.py`, and established scoring paths
as frozen unless a documented correction reruns all affected baselines.

## Build, Test, and Development Commands

- `git submodule update --init --recursive` fetches the pinned TERA baseline.
- `uv sync` creates the Python 3.12 environment from `uv.lock`.
- `uv run pytest -q` runs the complete correctness and evidence suite.
- `uv run ruff check .` checks lint and imports; `uv run ruff format .` formats Python code.
- `uv run python experiments/f01_spark_nd_sweep.py --help` lists the frozen F01 workflow.

Do not install packages ad hoc with `pip`; update `pyproject.toml` and `uv.lock` deliberately.

## Coding Style & Naming Conventions

Use four-space indentation, a 100-character line target, absolute imports, and modern Python 3.12
annotations such as `list[str]` and `X | None`. Ruff enforces `E`, `F`, `W`, import sorting, Python
upgrades, bugbear, and tidy-import rules. Use `snake_case` for modules and functions, `PascalCase`
for classes, and descriptive uppercase constants. Name new experiments `fNN_short_name.py` and
begin each with a preregistration docstring.

## Testing Guidelines

Use pytest files and functions named `test_*.py` and `test_*`. Parametrize RBF and Matérn coverage
where applicable. Numerical changes must compare matrix-free behavior with float64 dense references
at tight tolerances. Protect shapes, residuals, symmetry, diagonals, and chunking invariance. Add an
evidence-integrity test when committing benchmark summaries.

## Commit & Pull Request Guidelines

Use concise, imperative subjects. End commits with
`Co-Authored-By: GPT-5.6 Sol <noreply@openai.com>`. Pull requests should state the hypothesis or fix,
list validation commands, identify changed baselines, and link issues. Report benchmark claims with
full configuration and at least three seeds (mean plus spread); update `docs/RESEARCH_LOG.md` and
commit the supporting evidence. Include screenshots only for changed figures or visual docs.
