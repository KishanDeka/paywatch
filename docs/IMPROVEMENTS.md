# Improvements over the supplied README

## Changes already implemented

| Original premise | Repository decision | Why it matters |
|---|---|---|
| Ten transactions imply customer history | Explicit global stream and one partition | Avoids inventing an entity relationship absent from the data |
| Only anomalies get classified | Warmup + anomaly + deterministic sentinel routes | Exposes some low-anomaly fraud and makes cold starts explicit |
| About 97–99% of traffic should be filtered | Recall-constrained policy split, measured route fraction | Compute savings cannot justify silently discarding most fraud |
| Fraud threshold fixed at 0.5 | Calibrated score and cost-selected threshold | Imbalance and business consequences make 0.5 arbitrary |
| Asynchronous background audit | Awaited transactional decision and window checkpoint | Prevents success responses for unpersisted decisions |
| In-memory sequence state | Durable PostgreSQL state, event deduplication | Retries/restarts preserve the actual model input history |
| ZooKeeper dependency | Kafka KRaft | Removes an unnecessary service for this local setup |
| Reported latency and AUC without evidence | Synthetic evidence, exact scope, generated reports | Makes resume claims defensible |
| 'Zero-day detection' from reconstruction alone | Anomaly signal with explicit limitations | Novel deviations are not proof of novel fraud detection |

## What I would do next, in order

### 1. Let the baseline win when it should

Run the complete real-data pipeline before adding infrastructure. Compare XGBoost-only and the
cascade under the same chronological splits, costs, and review budget. Measure complete local and
durable latency for both. If the LSTM has no incremental value, use XGBoost for every transaction
and run the autoencoder in shadow for investigation. A well-supported negative result is a stronger
portfolio story than an unjustified two-stage system.

The supplied code evaluates both options but the serving workflow implements the requested cascade.
A shadow serving mode is a recommended next extension, not an already implemented capability.

### 2. Find data with real sequence identity

Choose a transaction dataset with documented customer/account IDs and timestamps, or build a
clearly labeled simulator with multiple accounts and behavioral attack episodes. Add per-entity
windows, causal rolling count/amount features, and consistent key-based Kafka partitioning.
Only then benchmark whether a recurrent model adds value beyond strong tabular velocity features.
Do not invent card IDs for the anonymous Kaggle rows to imply real behavior.

### 3. Improve the experimental design before the architecture

Use several rolling-origin development folds. Tune hidden size, sequence length (1, 5, 10, 20),
weight decay, normal-window inclusion rules, sentinel fraction, and target gate recall there.
Include a shuffled-context diagnostic, a simple non-recurrent autoencoder, and a cheap robust
z-score anomaly baseline. These are future experiments, not completed ablations in this archive.
Reserve a fresh final test period. Investigate fraud missed specifically by the gate and how often
reconstruction anomalies persist after one unusual transaction.

For gate selection, require enough policy positives to estimate recall meaningfully and consider
one-sided binomial uncertainty bounds under explicit independence assumptions. Temporal dependence
complicates those bounds; larger time blocks and repeated periods are preferable to treating every
adjacent window as independent evidence.

### 4. Model the review operation

Add an analyst capacity or top-k review constraint and compare precision at fixed review volume.
Use delayed fraud labels to estimate realized cost, recovery, and customer friction. Store model
outputs and eventual outcomes separately; joining them by event ID enables unbiased monitoring.
Sentinel sampling alone does not provide truth labels. Consider an independently labeled audit
sample to measure what the gate misses.

### 5. Turn reliability claims into failure experiments

Run the included Compose and integration suite. Then kill the consumer after database commit but
before Kafka commit, inject database disconnects, and test repeated partition reassignments. Add
consumer-generation fencing and durable source-offset progression before multi-instance operation.
Introduce schema migrations, retention/compaction for the offset ledger, authentication, TLS,
and a secrets manager if moving outside a localhost demo. Add a transactional outbox before sending
real alerts; do not append a best-effort webhook to the inference path.

### 6. Make drift monitoring actionable

The repository produces train-reference feature PSI and SQL/Prometheus observability. Add a bounded
rolling online feature monitor and delayed-label performance alarms next. PSI is distributional
change, not proof of degraded model quality. Keep thresholds fixed during evaluation and require a
reviewed training run before changing them. Automatic threshold adaptation can hide an attack by
learning that the attack is normal.

## Portfolio presentation

Show three figures first: the gate recall–classifier fraction trade-off, full-test precision–recall
curves against XGBoost alone, and amount-weighted decision cost. The generated `evaluation.png`
contains these. Follow with the atomic processing design and an edge-case test table.

A suitable description without invented performance numbers:

> Built a reproducible fraud-triage system with a hand-implemented LSTM, calibrated XGBoost,
> ONNX inference, and transactional Kafka/PostgreSQL processing. Evaluated gate-induced missed
> fraud against simpler baselines and tested gradient correctness, replay idempotency contracts,
> schema failures, and restart-state consistency.

Before claiming real database replay idempotency testing on your resume, execute the integration
suite. Add numbers only from a saved real-data evaluation or a measured full-stack benchmark, with
its scope and hardware. Do not use the synthetic 65% recall as a result on the credit-card dataset.

A personal angle suited to a scientific computing background is the emphasis on uncertainty,
reproducible pipelines, controlled comparisons, and explaining why an experiment failed. This is
more distinctive than merely listing Kafka, Docker, and two ML models.
