# Validation record

Executed on 2026-09-06, Python 3.12.13, Linux x86_64 CPU.

## What ran

| Check | Result |
|---|---|
| Pytest complete invocation | **46 passed, 4 skipped**, 5.31 seconds |
| LSTM reference forward and analytical gradients | Passed |
| LSTM finite-difference gradient check | Passed |
| Small normal-batch learning check | Passed |
| Synthetic end-to-end training | Completed, 4,000 rows, 8 epochs |
| PyTorch → ONNX export and numerical parity | Passed |
| XGBoost → ONNX export and numerical parity | Passed |
| Single-batch and extreme-input ONNX parity | Passed |
| Offline/online features and latest-row loss | Passed |
| Warmup, sentinel, anomaly and bypass branches | Passed |
| Malformed, nonfinite, negative, wrong-size input | Passed |
| Tied times and out-of-order state protection | Passed |
| Feature-schema and artifact integrity checks | Tested checksum rejection and invalid inputs |
| Kafka commit order and failure contracts | Passed using a fake consumer/store |
| API validation, conflicts, dependency failure | Passed using injected model/store |
| PostgreSQL duplicate, rollback, restart, quarantine tests | **4 skipped**; no database server |
| Docker Compose / real Kafka full-stack test | **Not executed**; Docker unavailable |
| GitHub Actions workflow | Written, not executed on GitHub |
| Ruff lint, Python compile, YAML parsing | Passed |
| Release gate on synthetic evaluation | Correctly failed: synthetic evidence and insufficient recall |

The one pytest warning is a Starlette/AnyIO deprecation warning, not a test failure.
The environment could not launch native PostgreSQL because its user-management operations were
unavailable. No database transactional behavior or real Kafka delivery guarantee is claimed as
locally verified. The supplied CI workflow runs those tests in a disposable Docker stack.
`all-tests.xml` is the raw final pytest evidence.

## Synthetic held-out comparison

The test partition has 800 rows and 20 fraud labels. These are artificial patterns with a 2.5%
fraud prevalence; they are not the real ULB dataset and must not be represented as such.

| Model | Average precision | Recall | Precision | Classifier fraction | Illustrative cost |
|---|---:|---:|---:|---:|---:|
| LSTM + XGBoost | 0.6978 | 65.00% | 68.42% | 13.63% | 625.91 |
| XGBoost only | 0.7250 | 65.00% | 39.39% | 100.00% | 653.91 |
| Logistic regression | 0.6919 | 70.00% | 50.00% | 100.00% | 488.13 |

The gate's test recall was 75%, despite selecting for 98% empirical policy recall. The cascade
and XGBoost baseline both recalled 65% at their selected operating thresholds; the cascade reduced
false-positive reviews from 20 to 6. Logistic regression recalled 70% and had the lowest
illustrative cost. This supports further baseline comparison, not a claim that the hybrid wins.

The autoencoder normal-validation MSE decreased from 0.98076 to 0.95282 over eight epochs.
That is evidence the optimizer learns this demo distribution, not evidence of fraud-detection
improvement. Hyperparameter choices still need controlled real-data ablations.

## Export evidence

- autoencoder: maximum absolute difference 1.192092896e-07 on export parity inputs.
- xgboost: maximum absolute difference 8.381903172e-08 on export parity inputs.

Both exports passed ONNX checker. Additional unit tests cover batch size one and large positive
and negative inputs. The initial export failed with protobuf 7 due to a converter boolean/int
attribute incompatibility. Pinning protobuf 5.29.3 resolved the observed failure; that compatibility
constraint is recorded in pyproject.toml. No failed export artifacts are delivered.

## Local benchmark, not end-to-end

- Complete local serial scoring: mean **0.273 ms**, p95 **0.476 ms**, p99 **0.990 ms** across 800 events.
- Inverse mean local service time: 3665.0 events/s. This is **not stack throughput**.
- XGBoost ONNX-session-only mean: 0.0073 ms. This excludes preprocessing and is not a matched whole-pipeline baseline.
- ONNX intra-op threads: one. CPU identifier reported by this environment: x86_64.

These timings exclude Kafka, HTTP, PostgreSQL, producer pacing, network delays, and queueing.
A single run on shared hardware is not a stable service-level objective. Benchmark scope and
platform are saved in `benchmark.json`. The original README's 14.2 ms full-pipeline result remains
unverified.
