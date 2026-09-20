# Operations and failure handling

## Supported deployment

One local Kafka broker in KRaft mode, one input partition, one consumer process, one PostgreSQL
instance, and a fixed versioned model directory. Compose binds exposed ports to localhost. No
authentication layer or TLS is implemented; this is not an internet-facing service configuration.
Use an alphanumeric local PostgreSQL password because it is embedded in a connection URL.

The API and Kafka share the same model and feature functions but have different stream IDs.
HTTP requests are serialized by the HTTP stream row lock. Because concurrent arrivals can be
out of timestamp order, replay ordered sequences through Kafka for reproducible experiments.

## Event contract

```json
{"event_id":"dataset-hash:row-index","time":123.0,"amount":14.5,"v":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}
```

No labels or extra keys are accepted. Time/amount must be nonnegative and finite. The V vector
must have 28 finite, bounded numbers. The schema permits normal JSON numeric coercion through
Pydantic; it is strict about fields, lengths, bounds, and finiteness, not Python scalar types.
Transaction amounts are floating point model features, not a settlement ledger.

## Outcomes

| Condition | Behavior |
|---|---|
| First nine events | XGBoost route with reason `warmup`; no invented reconstruction score |
| Loss above/equal gate | Classifier route `anomaly` |
| Low loss selected by stable hash | Classifier route `sentinel` |
| Low loss otherwise | `gate_pass`, null probability |
| Probability above/equal threshold | Record `review` |
| Duplicate ID and identical normalized payload | Return original decision, no state advance |
| Duplicate ID with different payload | HTTP 409 or durable Kafka quarantine |
| Older event time | HTTP 409 or durable Kafka quarantine, no state change |
| Equal timestamp | Accept in arrival order |
| Malformed Kafka message | Store payload hash/reason in quarantine, then commit offset |
| Model or database exception | No Kafka commit; worker stops and readiness fails |
| Model version differs from persisted state | Refuse scoring; start a new isolated stream environment |
| Database commit succeeds, Kafka commit fails | Replay and deduplicate; no distributed exactly-once claim |

Quarantine deliberately stores a payload hash and reason rather than an unlimited raw payload.
Raw-message recovery depends on Kafka retention and source offsets. The table is a durable
quarantine ledger, not a Kafka dead-letter topic. Retention and replay tooling are future work.

## Restart and model changes

Use `docker compose restart consumer` for the same model and dataset. State is loaded from
PostgreSQL inside each transaction, so no in-memory checkpoint restoration is needed.
A worker failure makes readiness return 503; inspect `docker compose logs consumer`, fix the
cause, and restart. The service does not attempt infinite retries around a broken model.

For an unrelated dataset or new model, use a separate Compose project/environment with fresh
volumes and ports or explicitly clear a disposable demo environment. `docker compose down` keeps
volumes. `docker compose down -v` deletes all local audit and Kafka data; use it only when you
intend to discard that demo. Reusing old state with a new scaler/model is not supported.
PostgreSQL's init SQL runs on a newly initialized data volume. An existing deployment needs an
explicit migration when schema changes; this project does not run ad hoc DDL at API startup.

## Metrics and interpretation

`paywatch_processed_total` counts processing attempts, including duplicate replays. Use the SQL
`decisions` table for deduplicated transaction counts. Histogram values cover feature/model/database
processing; they exclude time waiting in Kafka and time before the request entered the service.
`view_hourly_anomaly_metrics` groups by ingestion hour. `view_stream_velocity` uses event-relative
seconds; it represents global stream velocity. Its RANGE frame includes timestamp peers and it is
an offline analytics view, not a causal feature supplied to the model.

The model's `top_deviations` names standardized extremes. They are diagnostic breadcrumbs, not
SHAP, feature attribution, or a business explanation for anonymous PCA components.

The training run stores per-feature test PSI against training bins in `evaluation.json`. No online
PSI daemon, Grafana dashboard, automatic retraining, or Prometheus server is included. The service
exposes metrics for a collector that you can add later.

## Benchmark correctly

`python scripts/benchmark.py` records serial local scoring p50/p95/p99 and mean, after warmup.
It also measures a separately labeled XGBoost ONNX session call. The benchmark cannot establish
full-stack TPS. For that, record producer send timestamps, source arrival rate, broker lag,
durable completion timestamps, resource use, errors, and percentile latency at steady state.
Measure multiple loads until lag grows. Define whether latency starts before producer send,
at Kafka append, or at consumer receipt. Never call p95 an average.

## Integration checks

The four PostgreSQL tests use unique stream namespaces and require a disposable test database URL.
They validate concurrent duplicate serialization, transaction rollback on inference failure,
durable state across Store instances, and quarantine deduplication. They leave their small test
records in that disposable database.

`python scripts/stack_smoke.py` requires an empty disposable Compose stream and
`PAYWATCH_TEST_DATABASE_URL`. It sends a valid event twice and a malformed event, then checks the
real database. Its large synthetic timestamp means ordinary earlier replay should not follow it
in the same stream. The GitHub workflow runs this only against a fresh disposable stack.

The worker unit tests separately verify commit-after-persistence order, no commit on database
failure, durable quarantine before commit, invalid model state stopping, and single-partition
validation. These mocks do not establish actual broker behavior or PostgreSQL transaction semantics.
