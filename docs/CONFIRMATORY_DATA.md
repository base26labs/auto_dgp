# Confirmatory N-body data

`data/get_nbody.py` and its `data/nbody/*.npz` outputs are the frozen legacy benchmark. Keep them for
reproduction and smoke tests, but do not use them for a generalization claim: that generator draws a
different unobserved mass vector for every trajectory, omits both masses and trajectory identifiers,
and the legacy benchmark randomly splits individual states.

The companion module `data.generate_nbody_confirmatory` creates a separate, versioned corpus for
confirmatory evaluation without changing the legacy generator. One replica is one well-defined
Hamiltonian: it draws one fixed mass vector, uses it for every trajectory, and saves that vector with
the complete generation configuration.

## Generate a replica

Run from the repository root:

```bash
uv run python -m data.generate_nbody_confirmatory \
  --output-dir data/nbody_confirmatory \
  --n-particles 2 \
  --n-dims 3 \
  --n-trajectories 100 \
  --steps-per-trajectory 100 \
  --replica 0 \
  --mass-seed 1729 \
  --trajectory-seed 2718 \
  --split-seed 31415
```

Generate different task replicas by changing `--replica`; the replica is mixed independently into
mass, initial-condition, split, and validation random streams. Never pool replicas into one GP fit,
because each fixed mass vector defines a different Hamiltonian.

Each run writes:

- `<stem>.npz`: raw `X`, `E`, and `F`, fixed `masses`, `trajectory_id`, `time_index`, physical time,
  sample and trajectory split manifests, full config JSON, and train-only normalization arrays;
- `<stem>.metadata.json`: readable configuration, masses, shapes, split groups, normalization, and
  validation results;
- `<stem>.sha256.json`: SHA-256 hashes for both data and metadata. Verify it with
  `verify_sha256_manifest(...)` before a run.

For model experiments, use `data.load_nbody_confirmatory.load_prepared_confirmatory_bundle(...)`.
It fails closed unless the checksum manifest, NPZ, metadata, config, split groups, train-only
normalization, and physical validation report agree, and it returns read-only normalized arrays with
source row and trajectory/time identifiers intact.

## Frozen evaluation invariants

- Splits are deterministic and group-disjoint: no trajectory occurs in more than one of train,
  validation, and test.
- Every requested time point is retained. There is no energy- or gradient-based filtering.
- Input and target normalization is computed from training samples only. For
  `X'=(X-x_min)/x_span` and `E'=(E-energy_mean)/energy_std`, use the saved
  `gradient_scale=x_span/energy_std` so `F'=F*gradient_scale`.
- Generation fails on incomplete integration, nonfinite arrays, excessive energy drift, malformed
  split partitions, or finite-difference disagreement with the analytic gradient.
- Freeze the NPZ, metadata, checksum manifest, and split seeds before inspecting confirmatory test
  scores. The legacy corpus remains unchanged and should be reported separately.

F02's predeclared, label-independent temporal design and predictive claim gates are fixed in
`docs/F02_NBODY_PROTOCOL.md`.  The loader retains the complete corpus; the experiment layer selects
the same declared `time_index` values for every method and preserves the corresponding source IDs.
