# Design decisions and their rationale

## 1. Start with what the data can actually support

The input dataset contains anonymous principal-component features, amount, elapsed time, and a
fraud label. It does not identify a cardholder. Treating ten successive rows as one customer's
history would invent a relationship. PayWatch instead models a global ordered stream. This is a
limited temporal experiment, and the XGBoost-only baseline is central to deciding whether it helps.

One Kafka partition is a deliberate modeling constraint, not a scaling achievement. Splitting
these global windows over multiple partitions changes their meaning. With a future entity-aware
dataset, partition by customer/card ID and maintain independent windows; do not just increase the
partition count on this implementation.

The producer assigns IDs before stable sorting by time. Equal-time records keep original order.
A full dataset checksum changes IDs when the file changes. This makes exact-file replay stable,
although logically identical transactions in two differently serialized files are not deduplicated.

## 2. LSTM implemented directly

The cell computes four affine projections together for efficiency. For input x, previous hidden
state h, and previous cell state c:

```math
a = W_x x + W_h h + b
```

Split a into input, forget, candidate, and output blocks. Then:

```math
i = sigmoid(a_i),\quad f = sigmoid(a_f),\quad g = tanh(a_g),\quad o = sigmoid(a_o)
```

```math
c' = f \odot c + i \odot g,\qquad h' = o \odot tanh(c')
```

`lstm.py` contains these operations and both time loops. It does not use a prebuilt recurrent
layer. PyTorch supplies tensors, linear algebra, automatic differentiation, and optimization;
this is not a manually coded backpropagation engine. The tests compare hidden/cell outputs,
input gradients, recurrent/input-weight gradients, and bias gradients with `torch.nn.LSTMCell`.
A separate finite-difference gradient check protects against a shared reference wiring mistake.

The default hidden size is 24 for 30 inputs and 10 time steps. The encoder compresses the whole
window into its final hidden vector. A separate decoder receives that repeated vector at each
step and predicts all input features. There is no direct input-to-output skip connection or
teacher forcing, so reconstruction must use the bottleneck. Zero initial hidden/cell states make
each window's computation independent of earlier inference calls; only the input window is durable.

Input weights use Xavier initialization; each recurrent gate block is orthogonal. Forget biases
start at one to encourage retaining early information. AdamW, weight decay, clipping the total
gradient norm to one, and normal-validation early stopping improve optimization stability. These
are informed defaults, not experimentally proven superior hyperparameters on the real dataset.

The training objective is mean squared error over the complete window. Inference scores only
the newest row's reconstruction error: a single unusual incoming transaction is not diluted by
nine ordinary rows. An earlier fraud can still affect the latent context for subsequent events;
that is a reason to measure false-positive bursts, not to call every reconstruction anomaly fraud.

## 3. Preprocessing and contamination controls

Features are the ordered V1–V28 values, log(1 + amount), and log(1 + elapsed interarrival time).
Amount and gaps have long tails, so the logarithms reduce scale imbalance without discarding
large values. Time is a local gap rather than raw absolute time, reducing reliance on which hour
of this very short dataset a transaction occurred. The first row of each stream has gap zero.
Tied timestamps are legal; negative gaps are not. API and training share the same transformation.

Standardization uses only the training partition, including its fraud rows. The autoencoder
optimizer itself only sees fully normal windows. Including all training rows in the scaler is a
small, explicit distinction from claiming that every fitted preprocessing statistic is normal-only.
Features with near-zero variance use scale one, avoiding division by zero. We do not clip
standardized extremes: clipping could erase the anomaly signal. Raw numerical limits reject
unreasonable values before a model can overflow. Missing or nonfinite input is quarantined, not
silently filled with zero. The manifest fixes the feature order.

Critically, fraud rows are not deleted before creating sequences. Instead we build windows in
original order and exclude any window containing a positive label. Deleting rows first would join
transactions that were never contiguous. Sliding-window views avoid an additional full-copy
intermediate; selected windows are materialized for simple CPU minibatches. Full Kaggle-scale
training can still require hundreds of MB. Larger workloads should use indexed lazy datasets.

## 4. Four chronological partitions

Training gets 60%, calibration gets 10%, policy selection gets 10%, and the final test gets 20%.
Boundary timestamps move together. All window and interarrival state resets at a boundary. The
first nine events therefore use the same supervised warmup path in offline and online scoring.
This costs some context and creates a few artificial cold starts, but avoids overlapping windows
or fitted statistics leaking across evaluation partitions.

The calibration partition has two disclosed uses: autoencoder early stopping on normal windows
and sigmoid calibration of XGBoost probabilities using its labels. The policy partition chooses
the anomaly gate and decision thresholds. Neither is an unbiased final evaluation set. The test
partition is used only after choices are frozen. Training aborts if any partition lacks one class;
quietly reporting undefined ROC-AUC would hide an unsuitable evaluation split.

Repeatedly inspecting the test and changing hyperparameters would still leak information. For
serious experiments, use rolling chronological folds for development and reserve a final untouched
period. This two-day dataset provides weak evidence for long-term generalization.

## 5. XGBoost, calibration, and decision costs

The classifier trains on all training events, including anomalies and ordinary transactions.
This gives it support for sentinel and warmup traffic, which can be low-anomaly fraud. Restricting
its training to whatever one autoencoder happened to route would add selection bias and instability.

A compact depth-three, 150-tree histogram model with regularization, row/column subsampling, and
inverse-frequency positive weighting is the starting point. It is deliberately inexpensive enough
to benchmark directly against the entire cascade. Class weighting changes the training objective;
its outputs should not automatically be presented as calibrated fraud probabilities.

We fit a regularized logistic mapping of XGBoost log odds on the natural-prevalence calibration
partition, without class weighting at that stage. Coefficient and intercept are plain manifest
numbers, applied after ONNX scoring. Brier score on the test measures calibration error. This
mapping is not guaranteed to remain calibrated after prevalence drift or within every routed subset.

The policy threshold minimizes an illustrative cost:

```math
C = c_{review} N_{review} + f_{loss} \sum_{missed\ fraud} Amount
```

Defaults are a review cost of two and a missed-loss fraction of one. This assumes review prevents
the relevant loss and ignores customer friction, recovery, limited analyst capacity, and varying
review outcomes. It is a transparent scenario, not a validated financial estimate. Both the cascade
and XGBoost-only baseline choose their own threshold on the same policy partition. A value just
above one represents 'review none' when reviewing would cost more. The search sorts scores once
and evaluates tied-score groups together, rather than looping over a quadratic grid.

## 6. The gate is a hypothesis to test

The requested gate recall is 98% on the policy partition. We select the largest threshold that
retains enough observed fraud, counting warmup and sentinel routes. This reduces classifier calls
subject to that empirical constraint; it cannot guarantee test recall. There is deliberately no
fixed 97% traffic-filtering target. Few policy fraud examples can make any threshold unreliable.

A BLAKE2 digest of each event ID selects a stable 5% sentinel sample. The same event follows the
same route after restart. The sample sends some otherwise bypassed transactions to XGBoost and
can support monitoring of blind spots. It supplies neither ground-truth labels nor complete recall:
a supervised score on a sampled event is not evidence that an unsampled event was safe.

The synthetic demonstration is intentionally not polished into a success story: the gate achieves
its policy target but retains only 75% of test fraud. A release check rejects synthetic results and
can fail real candidates whose recall or decision cost does not meet explicit criteria. This is
useful evidence of disciplined model selection. If the gate harms results, score every transaction
with the baseline and keep anomaly scoring in an experimental shadow workflow.

## 7. Export and runtime performance

The time length and feature dimension are fixed in ONNX, while batch size is dynamic. Unrolling
ten hand-written recurrent steps produces ordinary ONNX operators, avoiding recurrent-operator
export surprises. Both ONNX graphs are checked, and numerical parity is tested against the fitted
PyTorch/XGBoost models. An observed protobuf 7 incompatibility with the converter led to an explicit
protobuf 5.29.3 compatibility pin. The chosen versions form a tested baseline, not a claim to be latest.

Serving imports ONNX Runtime rather than PyTorch or XGBoost. One intra-op and inter-op thread avoids
oversubscribing tiny single-event workloads. Blocking model calls run outside the async event loop.
Database operations use SQLAlchemy's async interface and asyncpg.

Two-stage inference does not inherently reduce latency. The LSTM runs for virtually every full
window and can cost more than the classifier calls it saves. The benchmark records complete local
scoring and a separately labeled XGBoost-session microbenchmark. Neither includes Kafka or durable
storage; the latter is not a fair end-to-end baseline comparison by itself.

## 8. Durable streaming semantics

The database transaction locks the stream checkpoint, checks for an existing event, runs inference,
and writes the decision and next window state together. Source offset auditing is part of that
transaction. A duplicate with the same normalized payload returns the old decision without moving
the window. The same event ID with a changed payload is a conflict.

The Kafka offset is committed only after database success or durable quarantine. A crash after
database commit but before Kafka commit causes replay; the unique stream/event key prevents double
state advancement. A model or database failure stops the worker without committing the message.
There is no unbounded in-memory audit queue and no best-effort FastAPI background-task persistence.
Awaiting durable writes trades some throughput for a more credible audit trail.

This provides at-least-once Kafka delivery with idempotent database effects. It is not a distributed
exactly-once transaction across Kafka, PostgreSQL, and external actions. Multi-instance rebalances,
network partitions, and failover races need additional fencing and fault-injection validation.
The supported portfolio configuration is one service instance, one worker, and one Kafka partition.
