# F01 SPARK–TERA evidence

This directory is the committed evidence for protocol `f01_spark_nd_sweep_v1`. The run used source
commit `a177883a6751372ce5fa8818b0442944f3cafa4e` and pinned TERA commit
`b2382e10a045abca3d653ad58c4a2a9c1ca73458`.

- `results/` contains 54 canonical JSON records: two paired arms for each of 27 fixed-system
  datasets. Each record includes metrics, configuration, resource diagnostics, runtime versions,
  source hashes, and the dataset hash.
- `summary.json` is the canonical preregistered aggregation over three systems in each of nine
  `(n_particles, spatial_dims)` cells.
- `data_sha256.tsv` binds the ignored, reproducible NPZ datasets without committing 19 MB of binary
  data.
- `slurm_generate.tsv` and `slurm_benchmark.tsv` show that all shared eight-CPU array tasks completed
  with exit code zero. Arrays were `2814307` and `2814338`, using the unique run root
  `f01_nd_a177883_njDXyZ`.

The strict outcome is negative: `summary.json` records `all_configurations_pass: false` because
SPARK's mean NLL is worse than TERA for `(n=4,d=3)`. Run `uv run pytest -q` to recompute the summary
from these raw records and verify its byte-for-byte identity. Reproduce the ignored datasets and
method records with the commands in `docs/STARTUP.md`; `_write_new` deliberately refuses to
overwrite existing artifacts.
