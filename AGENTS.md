# CoPTER experiment rules

These rules apply to the whole repository. More specific `AGENTS.md` files may
add constraints but must not weaken these experiment-integrity requirements.

## ABA protocol

- The default curriculum is A1 100 epochs, B 100 epochs, then A2 100 epochs.
- Each task has exactly one canonical configuration and one canonical flow file.
  Their SHA256 hashes are recorded in the protocol manifest and must not change
  silently after a run has started.
- ACC and SOR must use the same topology, traffic, seed, action space, reward,
  neural-network dimensions, optimizer settings, and training budget. Only the
  algorithm, replay policy, and SOR regularizers may differ.
- ACC and SOR use independent ports, model directories, logs, and raw-output
  directories.
- The top-level scheduler owns the epoch loop. `simulation/run_100.sh` and
  `copter/run_100.sh` execute exactly one epoch per invocation.
- Epsilon follows the global schedule and is never reset at A1 -> B or B -> A2.
- Target networks synchronize by global optimizer step, never by an
  episode-local counter.
- A phase transition preserves policy, target, optimizer, replay, RNG state,
  epsilon state, and global environment/training counters.
- Never silently cold-start replay, substitute a missing checkpoint, or skip a
  failed epoch.
- The registered replication route is C1 100 epochs, D 100 epochs, then C2
  100 epochs. ABA and CDC are independent experiments: models, replay, RNG,
  ports, runtime configs, raw output, and reports must never be shared between
  the routes.
- When ABA and CDC run concurrently, the four arms are ABA-ACC, ABA-SOR,
  CDC-ACC, and CDC-SOR. Port assignments must be unique across all four arms.

## Analysis and reporting

- `tools/analysis/metrics_core.py` is the authoritative implementation of FCT,
  slowdown, completion, queue, ECN, throughput, and PFC statistics.
- Formal report scripts must import the authoritative analysis API. They must
  not reimplement percentiles or metric formulas.
- A failed analysis must preserve all raw output and must not write an epoch
  completion marker.
- Large raw files from non-boundary epochs may be deleted only after successful
  analysis. A1/B/A2 boundary raw output and full checkpoints are retained.
- This ABA protocol has no frozen evaluation. Reports may describe training,
  task adaptation, and recovery after returning to A, but must not claim that
  cross-task training-reward differences alone prove catastrophic forgetting.
- PFC values without a valid port mapping are `n/a`, never zero.
- Every report records analysis provenance, input hashes, flow/config hashes,
  percentile method, warm-up handling, metric version, and generation time.

## Validation and repository safety

- Changes to this pipeline require metric-fixture, smoke, and resume tests.
- Preserve existing user changes in a dirty worktree. Never reset or overwrite
  unrelated work.
- Runtime configuration files and generated output belong below their declared
  run directory; do not introduce machine-specific absolute paths into tracked
  source files.
