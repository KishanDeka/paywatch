# PayWatch — two-stage fraud/anomaly detection

**Purpose:** A portfolio project that utilises : a hand-built LSTM autoencoder, calibrated XGBoost, and a durable Kafka → FastAPI → PostgreSQL inference workflow.

The problem statement is **whether an anomaly gate saves enough classifier work to justify
its missed fraud and latency**. PayWatch measures that trade-off instead of assuming that a more complicated pipeline is better. Its scientific emphasis is uncertainty, controlled comparisons, reproducibility, and diagnosing failure modes.

**Status:** implemented and locally tested with synthetic data. Both ONNX exports were verified.
Real credit-card metrics and full Docker/Kafka/PostgreSQL integration are not yet measured.
See [validation evidence](reports/VALIDATION.md). The original README's 14.2 ms, 250 TPS,
0.89 ROC-AUC, 0.86 PR-AUC, and 97% filtering claims are not treated as measured results.

## What different choices are made here

- An LSTM cell implemented with explicit input, forget, candidate, and output gates; no
  `nn.LSTM` or `nn.LSTMCell` in the model. Forward and backward results are checked against
  a reference cell, including a numerical gradient check.
- A gate selected on a chronological policy partition to meet a requested empirical fraud
  recall. A deterministic 5% sample of otherwise bypassed events also gets classified.
- A business-cost decision threshold, probability calibration, amount-weighted missed fraud,
  and block-bootstrap recall intervals. Business costs are explicit assumptions.
- Atomic decision, window-state, and offset auditing in PostgreSQL; event IDs make retries
  idempotent. A failed database write never produces a successful durable acknowledgment.
- A baseline-first evaluation: calibrated XGBoost alone and logistic regression are included.
  A release-check script rejects synthetic evidence and insufficient real-data performance.

## Workflow

```mermaid
flowchart TD
  A[Chronological replay] --> B[Kafka: one ordered partition]
  B --> C[Schema and ordering checks]
  C -->|Invalid| Q[Durable quarantine]
  C --> D[Shared feature transform]
  D --> E[Rolling LSTM reconstruction]
  E -->|Warmup, anomaly, or sentinel sample| F[Calibrated XGBoost]
  E -->|Below gate| G[Pass with unscored probability]
  F --> H[Review or pass]
  G --> I[Atomic audit and window checkpoint]
  H --> I
  Q --> J[Commit Kafka offset]
  I --> J
```

`review` is a recorded recommendation. No transaction is actually blocked and no alert is
sent externally. An unscored transaction has `fraud_probability=null`, never a fabricated zero.
The HTTP endpoint uses an independent demo stream so HTTP requests cannot change Kafka windows.

## Dataset used

Use the [ULB/Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud):
`Time`, `Amount`, `V1`–`V28`, and `Class`. Download it manually into `data/creditcard.csv`.
No real dataset is redistributed here. The included demo is generated synthetic data.

The data has no card/customer ID. A window of ten rows therefore describes a **global arrival
stream**, not one customer's spending sequence. The project enforces a single Kafka partition.
Do not claim per-customer velocity, user behavior modeling, or demonstrated zero-day detection.
See [design choices](docs/DESIGN.md) before interpreting the LSTM.

## Repository guide

```text
paywatch/
├── src/paywatch/
│   ├── lstm.py           # Explicit recurrent cell and autoencoder
│   ├── train.py          # Train, calibrate, choose policy, export, evaluate
│   ├── data.py           # CSV contract, time splits, replay adapter
│   ├── features.py       # Same transformation in training and serving
│   ├── policy.py         # Gate, sentinel sample, calibration, cost threshold
│   ├── evaluation.py     # Metrics, uncertainty, drift calculations
│   ├── inference.py      # ONNX scoring without training frameworks
│   ├── database.py       # Transactional state and audit persistence
│   ├── worker.py         # Kafka validation, processing, manual commits
│   ├── api.py            # FastAPI lifecycle, HTTP scoring, health, metrics
│   ├── replay.py         # Rate-controlled CSV producer
│   └── schema.py         # Strict event schema
├── training/train.py     # Convenient training entry point
├── producer/producer.py  # Convenient producer entry point
├── consumer/main.py      # Convenient ASGI entry point
├── db/init.sql           # Tables and analytical views
├── scripts/              # Demo, plots, benchmark, release check, stack test
├── tests/                # Model, pipeline, export, API, database tests
├── docs/                 # Decisions, model card, runbook, improvements
├── data/                 # Local CSV files; ignored by Git
├── models/               # Versioned training outputs; ignored by Git
├── reports/              # Measured evidence and figures
├── .github/workflows/ci.yml
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```

All reusable logic is in one importable package. The three entry-point folders contain no
duplicated model or feature logic. One Docker image supports both consumer and producer.

## Quick start

Use Python 3.11 or 3.12. Tested locally on Python 3.12/Linux CPU. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[train,dev]'
python scripts/make_demo_data.py
python -m paywatch.train --data data/demo.csv --output models/my-demo --dataset-kind synthetic --epochs 8
PAYWATCH_TEST_MODELS=models/my-demo python -m pytest -q -m 'not integration'
python scripts/plot_results.py --models models/my-demo
python scripts/benchmark.py --models models/my-demo
```

On Windows activate `.venv\Scripts\activate`; use the appropriate PyTorch installation for your
platform. Set environment variables with your shell's syntax. Run commands from the root,
not from `training/`. Each training output must be a new or empty directory.

The delivered archive includes `models/demo/`, `data/demo.csv`, and evidence from a completed
synthetic run. These are conveniences for review, not real-data results. They are ignored when
you initialize Git. `make demo` is intended for a fresh checkout without those generated files;
use the new-directory command above when working from this archive.

## Train on the real dataset

```bash
python -m paywatch.train --data data/creditcard.csv --output models/run-001 --dataset-kind creditcard --epochs 30
python scripts/plot_results.py --models models/run-001 --output reports/real-run-001
python scripts/benchmark.py --models models/run-001 --data data/creditcard.csv --output reports/real-run-001/benchmark.json
python scripts/check_release.py models/run-001/evaluation.json
```

Splits are chronological: 60% training, 10% calibration/early stopping, 10% policy selection,
20% final test. Timestamp ties stay together. Each split needs both classes; short or invalid
splits fail explicitly. Scaler statistics come only from training. Windows reset at every
boundary, so none straddles a partition. The classifier sees all training rows; the autoencoder
sees only originally contiguous windows containing no labeled fraud.

The gate aims for 98% empirical recall on the policy split, with no promise that only 1–3% of
traffic gets scored. Read `evaluation.json` to see what the data supports. The default illustrative
cost is 2 currency units per review plus the full amount of any missed fraud. Adjust
`--review-cost` and `--loss-fraction` before evaluating a deployment decision.

## Start the streaming stack

Requires Docker Compose v2. Images are version-pinned, with Kafka in KRaft mode (no ZooKeeper).

```bash
cp .env.example .env
# Edit .env: choose an alphanumeric local password and MODEL_DIR=./models/demo
# Use MODEL_DIR=./models/run-001 for a real-data model.
docker compose up --build --wait
curl http://localhost:8000/health/ready
docker compose --profile replay run --rm producer
```

For real data:

```bash
docker compose --profile replay run --rm producer --data /app/data/creditcard.csv --bootstrap kafka:29092 --split test
```

Producer replay defaults to the held-out test partition, preserves sorted order, sends no label,
and derives stable event IDs from dataset hash plus original row number. Replaying the same file
has no duplicate database effect. A different model or unrelated dataset should use a fresh
stream/database environment; do not append old timestamps to a used stream.

API docs: `http://localhost:8000/docs`. A durable HTTP example:

```bash
python - <<'PYTHON'
import httpx
payload = {"event_id": "manual-001", "time": 1, "amount": 12.5, "v": [0.0] * 28}
r = httpx.post("http://localhost:8000/score", json=payload)
r.raise_for_status()
print(r.json())
PYTHON
```

Inspect SQL metrics:

```bash
docker compose exec postgres psql -U paywatch -d paywatch -c 'SELECT * FROM view_hourly_anomaly_metrics;'
docker compose exec postgres psql -U paywatch -d paywatch -c 'SELECT * FROM view_stream_velocity LIMIT 10;'
```

`/health/live` checks process availability. `/health/ready` checks the database, loaded models,
and worker status. `/metrics` exposes Prometheus counters and processing-latency histograms.
See [operations and failure handling](docs/RUNBOOK.md).

## Tests and evidence

```bash
python -m ruff check .
python -m pytest -q
# Against an explicitly disposable PostgreSQL database:
PAYWATCH_TEST_DATABASE_URL=postgresql+asyncpg://paywatch:YOUR_PASSWORD@localhost:5432/paywatch python -m pytest -q -m integration
```

Database tests skip when their URL is not set. ONNX inference tests skip if their model directory
does not exist. CI trains its own synthetic artifact before testing and starts a disposable
Compose stack for PostgreSQL and Kafka tests. The workflow is included; it has not been run on
GitHub in this session.

![Synthetic evaluation](reports/evaluation.png)

**Synthetic evidence only:** the gate retained 75% of held-out fraud, and the cascade recalled
65%. This is a documented failure of the validation gate to generalize, not a production-quality
fraud detector. The measured local p95 scoring time is in `reports/benchmark.json`; it excludes
Kafka, HTTP, database latency, and queueing.

## Read more

- [Detailed model and engineering choices](docs/DESIGN.md)
- [Model card and evaluation limitations](docs/MODEL_CARD.md)
- [Recommended improvements and portfolio narrative](docs/IMPROVEMENTS.md)
- [Operation, restart, failure, and measurement guide](docs/RUNBOOK.md)
- [Executed tests and measured results](reports/VALIDATION.md)
- [Primary technical references](docs/REFERENCES.md)

MIT licensed for the code. Dataset terms remain separate.
