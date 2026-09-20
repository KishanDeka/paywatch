import asyncio
import hashlib
import json
import time
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


class EventConflict(ValueError):
    pass


def payload_hash(event):
    return hashlib.sha256(event.model_dump_json().encode()).hexdigest()


class Store:
    def __init__(self, url):
        self.engine = create_async_engine(url, pool_pre_ping=True, pool_size=5)

    async def ready(self):
        async with self.engine.connect() as conn:
            await conn.execute(text("SELECT 1 FROM stream_state LIMIT 1"))

    async def close(self):
        await self.engine.dispose()

    async def process(self, model, event, stream, source=None):
        start = time.perf_counter()
        fingerprint = payload_hash(event)
        async with self.engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO stream_state (stream_id, state) VALUES (:stream, '{}'::jsonb) ON CONFLICT DO NOTHING"
                ),
                {"stream": stream},
            )
            row = (
                await conn.execute(
                    text("SELECT state FROM stream_state WHERE stream_id=:stream FOR UPDATE"),
                    {"stream": stream},
                )
            ).one()
            existing = (
                await conn.execute(
                    text(
                        "SELECT payload_hash, decision FROM decisions WHERE stream_id=:stream AND event_id=:id"
                    ),
                    {"stream": stream, "id": event.event_id},
                )
            ).first()
            if existing:
                if existing.payload_hash != fingerprint:
                    raise EventConflict("event ID reused with a different payload")
                decision = dict(existing.decision)
            else:
                state = row.state or None
                decision, state = await asyncio.to_thread(model.score, event, state)
                await conn.execute(
                    text(
                        "INSERT INTO decisions (stream_id, event_id, payload_hash, decision) VALUES (:stream, :id, :hash, CAST(:decision AS jsonb))"
                    ),
                    {
                        "stream": stream,
                        "id": event.event_id,
                        "hash": fingerprint,
                        "decision": json.dumps(decision, allow_nan=False),
                    },
                )
                await conn.execute(
                    text(
                        "UPDATE stream_state SET state=CAST(:state AS jsonb), updated_at=now() WHERE stream_id=:stream"
                    ),
                    {"stream": stream, "state": json.dumps(state, allow_nan=False)},
                )
            if source:
                await conn.execute(
                    text(
                        "INSERT INTO processed_offsets (topic, partition_id, offset_id) VALUES (:topic, :partition, :offset) ON CONFLICT DO NOTHING"
                    ),
                    source,
                )
        return decision, (time.perf_counter() - start) * 1000

    async def quarantine(self, source, raw, reason):
        async with self.engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO quarantine (topic, partition_id, offset_id, payload_sha256, reason) VALUES (:topic, :partition, :offset, :hash, :reason) ON CONFLICT DO NOTHING"
                ),
                {**source, "hash": hashlib.sha256(raw).hexdigest(), "reason": reason[:200]},
            )
            await conn.execute(
                text(
                    "INSERT INTO processed_offsets (topic, partition_id, offset_id) VALUES (:topic, :partition, :offset) ON CONFLICT DO NOTHING"
                ),
                source,
            )
