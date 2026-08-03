# F02 global confirmatory release scaffold

`f02_global_confirmatory_recipe_v1` and `f02_one_release_ledger_v1` are
**non-executable scaffolds**. They can check structural identities, but they are not release
authorization and must not be used to inspect or score confirmatory labels.

The source-committed status is `non_executable_scaffold_v1`,
`development_evidence_semantically_verified=false`, and
`RELEASE_MUTATIONS_ENABLED=false`. Every public ledger mutation (`authorize`, `begin-attempt`,
`finish-attempt`, and `seal`) fails before reading or creating its arguments with:

> F02 release mutations are disabled: actual runner, semantic evidence validators, immutable
> result store not integrated

There is no public bypass and no legal v1 ledger. `audit` is read-only format/fixture inspection; it
cannot make a manually supplied document releasable. Enabling any mutation requires a reviewed code
change and schema-version bump, not a command-line flag or environment variable.

The original F02 development pilot also failed its float32 same-neighbour gate: released TERA32 and
ORBIT32 differed materially although their float64 paths agreed. The original protocol is terminated
and not releasable. `experiments/f02_internal_task.py` independently remains hard-disabled for
`evaluation_split=test`.

## What the structural recipe binds

A structurally valid recipe contains one complete 450-task grid:

- all 50 confirmatory corpora (`replica=101..110`, `n_particles={2,4,6,8,10}`);
- optimizer seeds `{11,29,47}`;
- `internal-shared-fit`, `dsoftki-512`, and `ddsvgp-512`, with 150 tasks each;
- a complete 15-entry `(D, seed)` configuration grid per method;
- the five-entry `D -> ORBIT m` schedule for the internal method;
- source commit/tree, TERA gitlink, protocol identity/blob, dependencies, and catalog identity; and
- the four hashes of each corpus bundle: dataset file, metadata file, manifest file, and semantic
  dataset content.

The catalog must be the regular, non-symlink file at the exact absolute path
`input.run_root/catalog.json`; `input.run_root` must be within `provenance.repo_root`. A byte-identical
copy or renamed catalog is rejected.

That is only a structural path check, not a global corpus identity or a trust anchor. Both roots are
self-reported by the catalog: a copied wrapper can report a new `input.run_root` while preserving the
same 50 bundle identities. The dormant sibling marker is likewise an ordinary deletable file. Neither
mechanism provides a one-release guarantee, and neither may be used to enable v1 execution.

Every recipe has distinct `experiment_id` and `protocol_id` fields. The protocol ID, canonical
repository path, and Git-blob SHA are combined into the protocol binding. Both
`docs/F02_NBODY_PROTOCOL.md` and `docs/F02B_NBODY_PROTOCOL.md` are blocked. The terminated protocol's
blob SHA-256, `6a103772e99e953d71a13f9655faea91f532613d750610b8358b6a1cc2bb2df8`, is also
blocked at every other path, and any protocol blob containing a case-insensitive `DRAFT` marker is
rejected.

`f02_method_selection_v1.selection_evidence` currently stores role names and hashes only. A matching
hash proves byte identity, not that an artifact contains the right development-only rows, selection
rule, candidate grid, tie-break, or winning configuration. Consequently structural recipe validation
returns `releasable=false`; it never calls these hashes semantically verified.

The JSON Schema files are descriptive scaffolds, not authorization contracts. They constrain selected
outer fields but do not prove hash relationships or all nested bundle, task, event, and result
semantics. A schema-only consumer must never treat a document as validated or releasable; the Python
structural validator is stricter, and even it deliberately returns `releasable=false` in v1.

Recipe construction and validation remain safe structural operations. Their output explicitly says
that execution is disabled:

```bash
python experiments/f02_global_release.py validate \
  --recipe releases/f02_global_confirmatory_recipe.json \
  --catalog /protected/f02/<run>/catalog.json
```

Running the following is an intentional fail-closed check, not an authorization procedure:

```bash
python cluster/f02_confirmatory_ledger.py authorize \
  --recipe releases/f02_global_confirmatory_recipe.json \
  --catalog /protected/f02/<run>/catalog.json \
  --ledger /protected/f02-release-ledgers/catalog.one-release-ledger.json
```

## Required successor integration

A successor version must not enable mutations until all of the following are implemented and
reviewed together.

### Global corpus identity, protected registry, and stable snapshots

The one-release key must be derived from the exact ordered corpus-set identity, including the four
registered hashes for every confirmatory bundle. It must not be keyed by the catalog pathname or by a
self-reported `input.run_root`. Authorization must reserve that identity in a service-owned,
non-deletable or write-once global registry. A writable sibling marker beside `catalog.json` is not an
authorization primitive, even when created with `O_EXCL`.

Recipe and catalog files must each be opened once with `O_NOFOLLOW`, verified as stable regular files
with `fstat`, and parsed and hashed from the same captured bytes. Separate path reads for validation
and hashing permit symlink/replacement races and are insufficient. The protected registry must bind
those snapshot hashes and the corpus-set identity atomically before any test artifact can be opened.

### Positive frozen-protocol authorization

Blocking known paths, one terminated blob hash, and text containing `DRAFT` is fail-closed defense in
depth, not positive protocol approval. A successor must require an explicit allowlist or signed/frozen
approval record that binds the exact protocol ID, repository path, Git blob, review state, and release
schema. A renamed or one-byte-modified old protocol, and any document marked `BLOCKED`, `TBD`, or
otherwise unapproved, must fail even if it avoids the literal word `DRAFT`.

### Semantic development-evidence validators

The successor schema must bind both canonical path and SHA-256 for the actual artifacts. The planned
repository-relative paths are:

- `releases/f02/evidence/internal-shared-fit/optimizer_selection_report.json`;
- `releases/f02/evidence/internal-shared-fit/orbit_resource_selection_report.json`;
- `releases/f02/evidence/dsoftki-512/optimizer_selection_report.json`;
- `releases/f02/evidence/dsoftki-512/runtime_dependency_lock.json`;
- `releases/f02/evidence/ddsvgp-512/optimizer_selection_report.json`; and
- `releases/f02/evidence/ddsvgp-512/runtime_dependency_lock.json`.

Those files are future required artifacts, not present release evidence. Their successor validators
must, at minimum:

- reconstruct the exact 15 development bundles and reject confirmatory replicas or test labels;
- verify all preregistered update budgets, all `(D, seed)` coordinates, the aggregation rule, and
  deterministic tie-break before accepting the selected update budget;
- reconstruct the internal ORBIT resource frontier over every allowed `m` and verify the selected
  five-dimension schedule; and
- parse each external dependency lock, resolve the audited package/vendor identities, and prove that
  its runtime configuration is exactly the one embedded in every corresponding task.

These must be semantic validators over strict artifact schemas, not file-existence or hash checks.
Only after they succeed may a successor recipe record
`development_evidence_semantically_verified=true`.

### Actual runner and immutable result storage

The cluster runner must accept only a task identity and configuration already present in the single
global recipe, obtain an authorized attempt token before opening a test artifact, and emit strict
internal/external result schemas. A successful result and attestation must each be read once through
one `O_NOFOLLOW` file descriptor; `fstat` must prove a stable regular-file snapshot, and parsing and
SHA-256 must use those same bytes.

Each result must bind its exact recipe bundle. Internal results need `provenance.data` containing the
three canonical filenames, dataset/metadata/manifest file hashes, generator configuration, and
`dataset_content_sha256`. The current internal result schema omits that semantic content hash and is
therefore intentionally rejected by the scaffold validator. External results must bind at least
`sources.dataset_content_sha256`. Every result attestation must repeat all four recipe bundle hashes.

Hash and identity matching alone are not result semantics. Success validation must also enforce exact
top-level and per-arm schemas, finite metrics and predictive outputs, convergence/termination records,
the validated test gate, attempt and scheduler identity, and method-specific invariants. Arbitrary or
empty arm payloads must never count as success. The current private scaffold helpers are not a trusted
runner attestation and must be replaced or independently hardened before mutations are enabled.

Accepted snapshots must then be copied with create-if-absent semantics into a protected,
content-addressed, non-overwritable store, for example:

- `/protected/f02-confirmatory/results/sha256/<first-two-hex>/<result-sha256>.json`; and
- `/protected/f02-confirmatory/attestations/sha256/<first-two-hex>/<attestation-sha256>.json`.

The store needs service-only writes, regular-file/no-symlink checks, fsync of file and parent,
immutable or write-once retention, and collision rejection when existing bytes differ. Ledger events
must reference only these protected content addresses, never mutable worker paths.

Sealing must reopen all 450 protected result and attestation objects by their recorded content
addresses, take fresh no-follow snapshots, recompute hashes, rerun strict schema and semantic bundle
validation, verify the complete task set and absence of active attempts, and only then append a seal.
Counting prior success events without this protected-store reread is insufficient.

Until that successor is reviewed and its schema version changes, the v1 code remains a structural
research artifact only.
