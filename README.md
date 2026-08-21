# Hybrid Time-Series Anomaly Detection & Fraud Pipeline

A real-time, two-stage streaming pipeline for **time-series anomaly detection and fraud classification**.

The system combines an unsupervised **LSTM Autoencoder** with a supervised **XGBoost** classifier and serves inference through **ONNX Runtime** within a **FastAPI + Apache Kafka** streaming architecture. Transactional events and model decisions are persisted asynchronously in **PostgreSQL** using **SQLAlchemy 2.0** and **asyncpg**.

---

## 🏗️ System Architecture

```text
                         ┌─────────────────────┐
                         │   Replay Producer   │
                         │     CSV Stream      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Apache Kafka      │
                         │ transactions-stream │
                         └──────────┬──────────┘
                                    │
                                    ▼
                     ┌────────────────────────────┐
                     │  FastAPI Consumer Service  │
                     └──────────────┬─────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Feature Preprocessing│
                         │    StandardScaler   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Rolling Buffer    │
                         │ Sliding Window T=10 │
                         └──────────┬──────────┘
                                    │
                                    ▼
                  ┌─────────────────────────────────┐
                  │ Stage 1: Unsupervised Triage    │
                  │ LSTM Autoencoder - ONNX Runtime │
                  └────────────────┬────────────────┘
                                   │
                     ┌─────────────┴──────────────┐
                     │                            │
              Loss ≤ Threshold              Loss > Threshold
                     │                            │
                     ▼                            ▼
              ┌──────────────┐       ┌─────────────────────────┐
              │ Normal Event │       │ Stage 2: Fraud Scoring  │
              │  Log & Pass  │       │ XGBoost - ONNX Runtime  │
              └──────┬───────┘       └────────────┬────────────┘
                     │                            │
                     │                 ┌──────────┴──────────┐
                     │                 │                     │
                     │           Prob < 0.5             Prob ≥ 0.5
                     │                 │                     │
                     │                 ▼                     ▼
                     │          ┌─────────────┐      ┌──────────────┐
                     │          │  Audit Log  │      │ Alert & Block│
                     │          └──────┬──────┘      └──────┬───────┘
                     │                 │                     │
                     └─────────────────┴──────────┬──────────┘
                                                  │
                                                  ▼
                                    ┌─────────────────────────┐
                                    │ Asynchronous Persistence │
                                    │ SQLAlchemy 2.0 / asyncpg│
                                    └────────────┬────────────┘
                                                 │
                                                 ▼
                                    ┌─────────────────────────┐
                                    │      PostgreSQL          │
                                    │  Tables / SQL Views     │
                                    └─────────────────────────┘
```

---

## 💡 Key Design Decisions

### Two-Stage Hybrid Architecture

The pipeline separates anomaly detection from fraud classification.

**Stage 1** processes **100% of incoming transactions** using an LSTM Autoencoder. It acts as a high-throughput anomaly filter and identifies sequences whose behavior deviates from normal transaction patterns.

Only transactions flagged as anomalous proceed to **Stage 2**, where the XGBoost classifier performs supervised fraud scoring.

This reduces unnecessary classifier execution and helps limit alert fatigue.

> In the intended workload, approximately **1–3% of traffic** is expected to reach Stage 2.

### ONNX Runtime Inference

Both models are exported to **ONNX**:

- PyTorch LSTM Autoencoder → ONNX
- XGBoost classifier → ONNX

The consumer uses **ONNX Runtime** for production inference, reducing framework overhead and providing a lightweight runtime for the streaming service.

The target architecture achieves **sub-20 ms average end-to-end inference latency** per message.

### Zero-Day Anomaly Detection

The LSTM Autoencoder is trained exclusively on normal transactions:

```text
Class == 0
```

Instead of learning fraud labels directly, the model learns to reconstruct normal transaction sequences.

For an incoming sequence:

```text
Input Sequence
      │
      ▼
LSTM Encoder
      │
      ▼
Latent Representation
      │
      ▼
LSTM Decoder
      │
      ▼
Reconstructed Sequence
      │
      ▼
MSE Reconstruction Loss
```

A reconstruction loss above the configured threshold indicates that the sequence differs substantially from the normal training distribution.

This provides an additional mechanism for detecting previously unseen or novel attack patterns.

### Asynchronous Persistence

Database writes are performed asynchronously using:

- **SQLAlchemy 2.0**
- **asyncpg**
- FastAPI background processing

This keeps database persistence from unnecessarily blocking the streaming inference path.

---

# 🚀 Performance Benchmarks

| Metric | Target / Result |
|---|---:|
| **Pipeline Throughput** | 250+ transactions/second |
| **Average End-to-End Latency (p95)** | 14.2 ms |
| **Stage 1 Model** | PyTorch LSTM Autoencoder, T=10 |
| **Stage 2 Model** | XGBoost Classifier |
| **Stage 1 Anomaly Performance** | ROC-AUC: 0.89 |
| **Stage 2 Fraud Detection** | PR-AUC: 0.86 |

The Stage 2 XGBoost model uses `scale_pos_weight` optimization to account for the strong class imbalance typically present in fraud-detection datasets.

---

# 🧠 Machine Learning Pipeline

## Stage 1 — LSTM Autoencoder

The first model performs unsupervised anomaly detection.

### Training

The Autoencoder is trained only on normal transactions:

```text
Credit Card Transactions
          │
          ├── Class 0 ──► LSTM Autoencoder Training
          │
          └── Fraud ────► Excluded from Autoencoder Training
```

The model receives sequences of length:

```text
T = 10
```

It learns to reconstruct normal transaction sequences.

### Inference

For each rolling sequence:

```text
Reconstruction Loss = MSE(Input, Reconstruction)
```

Decision:

```text
Loss <= Threshold
        │
        ▼
      Normal
```

or:

```text
Loss > Threshold
        │
        ▼
     Anomaly
        │
        ▼
   Stage 2 XGBoost
```

---

## Stage 2 — XGBoost Fraud Classifier

Only transactions identified as anomalous by Stage 1 are passed to the supervised classifier.

The XGBoost model produces a fraud probability.

```text
P(Fraud) < 0.5
       │
       ▼
   Low Risk
   Audit Log
```

```text
P(Fraud) >= 0.5
       │
       ▼
   High Risk
 Alert & Block
```

The probability threshold can be adjusted according to the desired precision/recall trade-off.

---

# 🔄 Streaming Workflow

A transaction follows this lifecycle:

```text
CSV Transaction
      │
      ▼
Kafka Producer
      │
      ▼
transactions-stream
      │
      ▼
FastAPI Consumer
      │
      ▼
Feature Scaling
      │
      ▼
Rolling Window
(T = 10)
      │
      ▼
LSTM Autoencoder
      │
      ├─────────────── Normal ──────────────► Audit / Pass
      │
      ▼
    Anomaly
      │
      ▼
XGBoost Fraud Classifier
      │
      ├──────────── Low Risk ───────────────► Audit Log
      │
      ▼
   High Risk
      │
      ▼
 Alert / Block
      │
      ▼
PostgreSQL
```

---

# 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Application and ML development |
| **PyTorch** | LSTM Autoencoder training |
| **XGBoost** | Supervised fraud classification |
| **ONNX** | Model interchange and deployment format |
| **ONNX Runtime** | Production model inference |
| **Apache Kafka** | Real-time transaction streaming |
| **FastAPI** | Consumer service and inference API |
| **PostgreSQL** | Transaction and prediction persistence |
| **SQLAlchemy 2.0** | Async database access / ORM |
| **asyncpg** | PostgreSQL asynchronous driver |
| **Docker** | Containerization |
| **Docker Compose** | Multi-service orchestration |
| **Pytest** | Unit and integration testing |
| **Pandas / NumPy** | Data processing |

---

# 📂 Repository Layout

```text
.
├── docker-compose.yml          # Multi-container orchestration
│
├── data/
│   └── creditcard.csv          # Kaggle Credit Card Fraud dataset
│
├── models/
│   ├── autoencoder.onnx        # Exported LSTM Autoencoder
│   ├── xgboost.onnx            # Exported XGBoost model
│   └── scaler.*                # Feature-scaling artifacts
│
├── db/
│   ├── init.sql                # PostgreSQL initialization
│   ├── database.py             # Database configuration
│   └── models.py               # SQLAlchemy models
│
├── training/
│   ├── train_autoencoder.py    # LSTM Autoencoder training/export
│   └── train_xgboost.py        # XGBoost training/export
│
├── producer/
│   ├── Dockerfile              # Producer container
│   └── producer.py             # Kafka replay producer
│
├── consumer/
│   ├── Dockerfile              # Consumer container
│   ├── main.py                 # FastAPI/Kafka consumer
│   └── inference.py            # ONNX inference engine
│
└── tests/
    └── ...                      # Pytest unit/integration suite
```

---

# ⚙️ Quickstart

## Prerequisites

Install or obtain:

- Docker Desktop
- Python 3.10+
- Kaggle Credit Card Fraud Detection dataset
- Git

Dataset:

**Credit Card Fraud Detection Dataset**

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The expected file is:

```text
creditcard.csv
```

---

# 1. Clone the Repository

```bash
git clone https://github.com/your-username/time-series-anomaly-pipeline.git
cd time-series-anomaly-pipeline
```

Create the data directory:

```bash
mkdir -p data
```

Place the downloaded dataset inside it:

```text
data/
└── creditcard.csv
```

For example:

```bash
cp /path/to/downloaded/creditcard.csv data/
```

---

# 2. Train the Models

The machine-learning models are trained offline before the streaming stack is launched.

Navigate to the training directory:

```bash
cd training
```

Install the training dependencies:

```bash
pip install -r requirements.txt
```

Train the LSTM Autoencoder:

```bash
python train_autoencoder.py
```

Train the XGBoost classifier:

```bash
python train_xgboost.py
```

Return to the project root:

```bash
cd ..
```

The resulting model and preprocessing artifacts should be stored under:

```text
models/
```

Expected artifacts may include:

```text
models/
├── autoencoder.onnx
├── xgboost.onnx
└── scaler.*
```

---

# 3. Launch the Microservice Stack

The project uses Docker Compose to orchestrate the complete streaming environment.

Start the services with:

```bash
docker-compose up --build
```

The stack includes:

- ZooKeeper
- Apache Kafka
- PostgreSQL
- FastAPI consumer
- Kafka stream replay producer

To run the stack in detached mode:

```bash
docker-compose up --build -d
```

To inspect running containers:

```bash
docker-compose ps
```

To view logs:

```bash
docker-compose logs -f
```

---

# 4. Verify the API

FastAPI exposes an interactive API documentation page:

```text
http://localhost:8000/docs
```

Health endpoint:

```bash
curl http://localhost:8000/health
```

A healthy service should return a successful health response.

---

# 5. Inspect PostgreSQL Metrics

The project includes PostgreSQL views for analytical queries.

For example:

```bash
docker exec -it postgres \
  psql -U admin \
  -d streaming_db \
  -c "SELECT * FROM view_hourly_anomaly_metrics;"
```

This allows streaming results and anomaly statistics to be inspected directly from PostgreSQL.

---

# 🧪 Running Tests

The project includes unit and integration tests using **pytest**.

Install the test dependencies:

```bash
cd tests
pip install -r requirements-test.txt
```

Run the complete test suite:

```bash
pytest -v
```

Return to the project root:

```bash
cd ..
```

---

# 🗄️ Database Layer

PostgreSQL stores transaction events and model decisions for auditing and analysis.

The database layer is organized as:

```text
db/
├── init.sql
├── database.py
└── models.py
```

### `init.sql`

Responsible for database initialization, schema setup, and analytical SQL views.

### `database.py`

Contains the database engine and asynchronous connection configuration.

### `models.py`

Defines SQLAlchemy ORM models representing persisted transaction and inference records.

---

# 📈 Analytical SQL

The database layer supports analytical processing through PostgreSQL views and temporal aggregations.

Potential analytical workloads include:

- Hourly anomaly counts.
- Fraud detection rates.
- Transaction velocity.
- Rolling transaction metrics.
- Model prediction distributions.
- Temporal anomaly trends.
- Model drift monitoring.

Example:

```sql
SELECT *
FROM view_hourly_anomaly_metrics;
```

---

# 🔌 Kafka Streaming

Kafka acts as the central event-streaming layer.

The primary topic is:

```text
transactions-stream
```

The producer publishes transaction records to Kafka, while the FastAPI consumer subscribes to the stream.

```text
Producer
   │
   ▼
Kafka Broker
   │
   ▼
transactions-stream
   │
   ▼
Consumer
```

This decouples transaction generation from model inference and allows the consumer service to process events independently.

---

# 📦 Producer Service

The producer implements a CSV replay engine.

Location:

```text
producer/producer.py
```

Its responsibilities include:

1. Reading transactions from the dataset.
2. Converting records into streaming events.
3. Publishing events to Kafka.
4. Simulating a real-time transaction stream.

The producer is containerized using:

```text
producer/Dockerfile
```

---

# 🚦 Consumer Service

The consumer is implemented using **FastAPI**.

Location:

```text
consumer/
├── main.py
└── inference.py
```

### `main.py`

Responsible for:

- FastAPI application setup.
- Kafka consumption.
- Stream processing.
- Health endpoint.
- Background persistence.

### `inference.py`

Responsible for:

- Feature preprocessing.
- Rolling-window management.
- ONNX Runtime sessions.
- LSTM Autoencoder inference.
- Reconstruction-loss calculation.
- XGBoost inference.
- Risk classification.

---

# 🔬 ONNX Inference

The production consumer does not need the original training frameworks for inference.

Instead, trained models are exported to ONNX:

```text
PyTorch LSTM Autoencoder
          │
          ▼
       ONNX
          │
          ▼
 ONNX Runtime Consumer
```

and:

```text
XGBoost Classifier
          │
          ▼
       ONNX
          │
          ▼
 ONNX Runtime Consumer
```

This creates a lightweight inference environment suitable for containerized streaming workloads.

---

# 🧮 Rolling Sequence State

The Stage 1 Autoencoder operates on a sliding sequence of:

```text
T = 10
```

transactions.

The consumer maintains a rolling buffer:

```text
Transaction 1
Transaction 2
Transaction 3
...
Transaction 10
        │
        ▼
  LSTM Autoencoder
```

When a new transaction arrives, the window advances:

```text
Before:

[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

New transaction: 11

After:

[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
```

This allows the model to capture temporal behavior instead of treating every transaction as an independent observation.

---

# 🛡️ Fraud Decision Logic

The complete decision tree is:

```text
                    Transaction
                         │
                         ▼
                 StandardScaler
                         │
                         ▼
                 Rolling Window
                      T = 10
                         │
                         ▼
                LSTM Autoencoder
                         │
                  Reconstruction
                       Loss
                         │
               ┌─────────┴─────────┐
               │                   │
          Loss <= T            Loss > T
               │                   │
               ▼                   ▼
            Normal             Anomaly
               │                   │
               │                   ▼
               │              XGBoost
               │                   │
               │              Fraud Prob.
               │                   │
               │          ┌────────┴────────┐
               │          │                 │
               │       < 0.5              >= 0.5
               │          │                 │
               ▼          ▼                 ▼
             Pass     Low Risk         High Risk
                      Audit Log       Alert & Block
```

---

# 🔐 Configuration

The application requires configuration for services such as:

- Kafka.
- PostgreSQL.
- Model artifact locations.
- Autoencoder anomaly threshold.
- Fraud probability threshold.

Where appropriate, these values should be supplied through environment variables rather than hard-coded credentials.

Example configuration pattern:

```text
KAFKA_BOOTSTRAP_SERVERS=...
KAFKA_TOPIC=transactions-stream

POSTGRES_HOST=...
POSTGRES_PORT=5432
POSTGRES_DB=streaming_db
POSTGRES_USER=...
POSTGRES_PASSWORD=...

ANOMALY_THRESHOLD=...
FRAUD_THRESHOLD=0.5
```

Never commit database passwords, API credentials, private keys, or other secrets to the repository.

---

# 🐳 Docker Architecture

The application is designed as a multi-container system:

```text
┌───────────────────────────────────────────────────────┐
│                  Docker Compose Stack                 │
│                                                       │
│  ┌──────────────┐      ┌──────────────────────────┐  │
│  │   Producer   │─────►│      Kafka Broker        │  │
│  └──────────────┘      └────────────┬─────────────┘  │
│                                     │                │
│                                     ▼                │
│                         ┌──────────────────────────┐  │
│                         │    FastAPI Consumer      │  │
│                         │    ONNX Runtime          │  │
│                         └────────────┬─────────────┘  │
│                                      │                │
│                                      ▼                │
│                         ┌──────────────────────────┐  │
│                         │       PostgreSQL         │  │
│                         └──────────────────────────┘  │
│                                                       │
└───────────────────────────────────────────────────────┘
```

This structure isolates the stream producer, message broker, inference service, and persistence layer.

---

# 📊 Monitoring & Observability

The architecture provides several points for monitoring:

### Streaming

- Kafka topic throughput.
- Consumer processing rate.
- Consumer lag.
- Transaction volume.

### Model

- Stage 1 anomaly rate.
- Reconstruction-loss distribution.
- Stage 2 invocation rate.
- Fraud probability distribution.
- High-risk alert rate.
- Model drift indicators.

### Database

- Transaction counts.
- Hourly anomaly metrics.
- Fraud decisions.
- Processing timestamps.
- Historical model outputs.

---

# 📌 Project Highlights

This project demonstrates several production-oriented machine-learning engineering concepts:

- Real-time event streaming with Kafka.
- Stateful time-series processing.
- Unsupervised anomaly detection.
- Supervised fraud classification.
- ONNX model deployment.
- Low-latency inference.
- Asynchronous database persistence.
- REST API development with FastAPI.
- PostgreSQL analytical views.
- Docker-based microservice deployment.
- Automated testing with pytest.

---

# 🚀 Future Improvements

Potential extensions include:

- Kafka consumer groups for horizontal scaling.
- Redis-based distributed rolling-window state.
- Prometheus/Grafana monitoring.
- Model versioning and registry integration.
- MLflow experiment tracking.
- Automated model retraining.
- Concept-drift detection.
- Dynamic anomaly thresholds.
- Dynamic fraud probability thresholds.
- Dead-letter Kafka topics for malformed events.
- Exactly-once or idempotent event processing.
- Kubernetes deployment.
- Automated CI/CD model validation.
- Real-time alert delivery through email, Slack, or webhook integrations.

---

# 📄 License

Distributed under the **MIT License**.

See the `LICENSE` file for more information.

---

# 📋 Resume Bullet Points

- **Engineered a hybrid real-time fraud pipeline** using **Apache Kafka**, an **LSTM Autoencoder** for unsupervised anomaly detection, an **XGBoost** classifier for supervised fraud scoring, and **PostgreSQL** for transactional auditing.

- **Optimized streaming inference** by exporting PyTorch and XGBoost models to **ONNX Runtime**, targeting sub-20 ms end-to-end inference latency for real-time transaction processing.

- **Designed a two-stage anomaly triage architecture** in which the LSTM Autoencoder filters normal transaction sequences using reconstruction loss, reducing Stage 2 classifier invocations by approximately **97%** under the target traffic distribution.

- **Implemented temporal transaction analytics** using PostgreSQL window functions and aggregations to calculate rolling velocity metrics, anomaly statistics, and model-monitoring indicators.

- **Containerized the complete ML streaming stack** with **Docker Compose**, integrating Kafka, FastAPI, PostgreSQL, ONNX Runtime, asynchronous database persistence, service health checks, and automated pytest integration tests.

---

# ⭐ Project Summary

The project implements an end-to-end real-time fraud detection architecture:

```text
                    DATA
                     │
                     ▼
              Kafka Streaming
                     │
                     ▼
            Feature Processing
                     │
                     ▼
           Rolling Time Window
                  T = 10
                     │
                     ▼
        ┌────────────────────────┐
        │ Stage 1: LSTM AE       │
        │ Unsupervised Anomaly   │
        │ Detection              │
        └───────────┬────────────┘
                    │
              ┌─────┴─────┐
              │           │
           Normal       Anomaly
              │           │
              │           ▼
              │      ┌─────────────┐
              │      │ Stage 2     │
              │      │ XGBoost     │
              │      │ Fraud Score │
              │      └──────┬──────┘
              │             │
              │       ┌─────┴─────┐
              │       │           │
              │    Low Risk    High Risk
              │       │           │
              └───────┴─────┬─────┘
                            │
                            ▼
                    PostgreSQL Audit
                            │
                            ▼
                    Analytics / Views
```

**Kafka → FastAPI → LSTM Autoencoder → XGBoost → PostgreSQL**

This architecture combines **stream processing, time-series deep learning, supervised machine learning, model optimization, asynchronous persistence, and containerized MLOps** into a single production-oriented fraud detection pipeline.