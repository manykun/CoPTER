# ABA experiment guide

This pipeline runs the matched curriculum
`realistic_webserver -> realistic_cachefollower -> realistic_webserver` for
ACC and SOR.  It records one row of network metrics per method and epoch and
keeps port-level measurements separately.

A second independent route is also registered:
`realistic_hadoop -> realistic_alistorage -> realistic_hadoop`. Running both
routes produces four isolated arms: ABA-ACC, ABA-SOR, CDC-ACC, and CDC-SOR.
Both routes keep the same topology, offered load, duration, action space,
reward, and optimizer settings; the traffic CDF is the intended workload
change.

## Canonical inputs

- Experiment configuration: `configs/aba/webserver_cachefollower.yaml`
- Task A configuration: `simulation/mix/aba/webserver_cachefollower/task_a_webserver.conf`
- Task A traffic: `simulation/mix/aba/webserver_cachefollower/task_a_webserver.flow`
- Task B configuration: `simulation/mix/aba/webserver_cachefollower/task_b_cachefollower.conf`
- Task B traffic: `simulation/mix/aba/webserver_cachefollower/task_b_cachefollower.flow`
- Immutable input hashes: `simulation/mix/aba/webserver_cachefollower/manifest.json`
- CDC experiment configuration: `configs/aba/hadoop_alistorage.yaml`
- Task C/D canonical inputs: `simulation/mix/aba/hadoop_alistorage/`

Do not edit canonical inputs after a run has been prepared.  Use a new run ID
for a scientifically different protocol.

## Recommended execution order

Run from the repository root in the configured Python environment:

```bash
conda activate m3

# Formula fixtures and scheduler/resume safety tests.
python -m unittest \
  tools.analysis.tests.test_metrics_core \
  tools.analysis.tests.test_aba_scheduler -v

# Six-epoch A1(2) -> B(2) -> A2(2) ACC/SOR integration test.
bash scripts/continual_validation/run_aba.sh \
  --config configs/aba/webserver_cachefollower.yaml \
  --stage smoke \
  --run-id aba_web_cache_smoke_s1

# Formal A100 -> B100 -> A100 run.
bash scripts/continual_validation/run_aba.sh \
  --config configs/aba/webserver_cachefollower.yaml \
  --stage all \
  --run-id aba_web_cache_s1
```

To smoke-test or run ABA and CDC concurrently (ACC and SOR within each route):

```bash
bash scripts/continual_validation/run_aba_cdc.sh \
  --stage smoke \
  --aba-run-id aba_web_cache_smoke_s1 \
  --cdc-run-id cdc_hadoop_alistorage_smoke_s1

bash scripts/continual_validation/run_aba_cdc.sh \
  --stage all \
  --aba-run-id aba_web_cache_s1 \
  --cdc-run-id cdc_hadoop_alistorage_s1
```

The four simulator/Agent pairs use ports 7300, 7301, 7310, and 7311. Run all
four concurrently only when the host has enough CPU, memory, and disk I/O.

ACC and SOR run concurrently by default on ports 7300 and 7301.  To resume an
interrupted formal run without cold-starting replay:

```bash
bash scripts/continual_validation/run_aba.sh \
  --config configs/aba/webserver_cachefollower.yaml \
  --stage run \
  --run-id aba_web_cache_s1 \
  --resume

bash scripts/continual_validation/run_aba.sh \
  --config configs/aba/webserver_cachefollower.yaml \
  --stage analyze \
  --run-id aba_web_cache_s1
```

Resume accepts only the last fully completed epoch, or an interrupted epoch for
which the terminal Agent record and complete raw NS-3 output prove completion.
An ambiguous checkpoint is stopped for manual inspection instead of being
trained twice.

## Output layout

- Protocol and method state: `experiments/aba/<run_id>/`
- Models and exact replay/RNG snapshots: `experiments/aba/<run_id>/<method>/models/`
- A1/B/A2 boundary checkpoints: `experiments/aba/<run_id>/<method>/checkpoints/`
- Logs: `experiments/aba/<run_id>/<method>/logs/`
- Portable CSV copies: `experiments/aba/<run_id>/summaries/`
- Final report and figures: `experiments/aba/<run_id>/REPORT.md` and `figures/`
- Runtime configurations: `simulation/mix/aba/webserver_cachefollower/runtime/<run_id>/`
- Raw and method-level CSV output: `simulation/output/aba/<run_id>/<method>/`

The network CSV has one row per epoch. In the port CSV, `physical` rows combine
queue, throughput, ECN, and mapped PFC values using the simulator's common
`switch-peer` key. Agent `watch` rows remain separate and carry the Agent port
index plus its physical identifier. Legacy PFC files without a peer-id column
are reported as `n/a`, never joined by guessing.

## Analysis provenance

`tools/analysis/metrics_core.py` is the only authoritative implementation of
FCT, slowdown, completion, queue, ECN, throughput, and PFC formulas.
`analyze_aba_epoch.py` applies those formulas to one epoch;
`build_aba_report.py` only combines the resulting rows and draws figures.

Every retained epoch contains `analysis_provenance.json`.  The final report
directory also contains aggregate provenance with metric version, percentile
method, warm-up buckets, timestamps, and input hashes.  Because this protocol
does not perform frozen evaluation, the report describes adaptation and
return-to-A recovery rather than claiming a strict proof of catastrophic
forgetting.
