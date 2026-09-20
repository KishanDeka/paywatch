CREATE TABLE IF NOT EXISTS stream_state (
    stream_id TEXT PRIMARY KEY,
    state JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS decisions (
    stream_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    decision JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (stream_id, event_id)
);
CREATE INDEX IF NOT EXISTS decisions_created_at_idx ON decisions(created_at);
CREATE TABLE IF NOT EXISTS processed_offsets (
    topic TEXT NOT NULL,
    partition_id INTEGER NOT NULL,
    offset_id BIGINT NOT NULL,
    PRIMARY KEY(topic, partition_id, offset_id)
);
CREATE TABLE IF NOT EXISTS quarantine (
    topic TEXT NOT NULL,
    partition_id INTEGER NOT NULL,
    offset_id BIGINT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY(topic, partition_id, offset_id)
);
CREATE OR REPLACE VIEW view_hourly_anomaly_metrics AS
SELECT date_trunc('hour', created_at) AS ingestion_hour,
       decision->>'model_version' AS model_version,
       count(*) AS transactions,
       count(*) FILTER (WHERE decision->>'route_reason' = 'anomaly') AS anomalies,
       count(*) FILTER (WHERE decision->>'route_reason' <> 'gate_pass') AS classified,
       count(*) FILTER (WHERE decision->>'action' = 'review') AS reviews,
       avg((decision->>'inference_ms')::double precision) AS mean_inference_ms,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY (decision->>'inference_ms')::double precision) AS p95_inference_ms
FROM decisions GROUP BY 1, 2;

CREATE OR REPLACE VIEW view_stream_velocity AS
SELECT stream_id, event_id,
       (decision->>'event_time')::double precision AS event_time,
       count(*) OVER (PARTITION BY stream_id ORDER BY (decision->>'event_time')::double precision RANGE BETWEEN 60 PRECEDING AND CURRENT ROW) AS transactions_last_60_seconds,
       sum((decision->>'amount')::double precision) OVER (PARTITION BY stream_id ORDER BY (decision->>'event_time')::double precision RANGE BETWEEN 60 PRECEDING AND CURRENT ROW) AS amount_last_60_seconds
FROM decisions;
