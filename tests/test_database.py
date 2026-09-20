import asyncio
import os
from pathlib import Path
import uuid
import pytest
from sqlalchemy import text
from paywatch.database import Store, EventConflict
from paywatch.schema import Transaction

pytestmark = pytest.mark.integration


class Model:
    def score(self, event, state):
        state = state or {"count": 0}
        return {"event_id": event.event_id, "count": state["count"] + 1}, {
            "count": state["count"] + 1
        }


@pytest.fixture
async def store():
    url = os.getenv("PAYWATCH_TEST_DATABASE_URL")
    if not url:
        pytest.skip("set PAYWATCH_TEST_DATABASE_URL to a disposable PostgreSQL database")
    store = Store(url)
    async with store.engine.begin() as conn:
        raw = await conn.get_raw_connection()
        await raw.driver_connection.execute(Path("db/init.sql").read_text())
    yield store
    await store.close()


def event(amount=1):
    return Transaction(event_id="same", time=1, amount=amount, v=[0] * 28)


async def test_concurrent_duplicate_committed_once(store):
    stream = f"test:{uuid.uuid4()}"
    results = await asyncio.gather(*[store.process(Model(), event(), stream) for _ in range(5)])
    assert all(x[0]["count"] == 1 for x in results)
    async with store.engine.connect() as conn:
        assert (
            await conn.execute(
                text("SELECT count(*) FROM decisions WHERE stream_id=:s"), {"s": stream}
            )
        ).scalar() == 1
    with pytest.raises(EventConflict):
        await store.process(Model(), event(2), stream)


async def test_failure_rolls_back_window_and_decision(store):
    class Broken:
        def score(self, event, state):
            raise RuntimeError("model failed")

    stream = f"test:{uuid.uuid4()}"
    with pytest.raises(RuntimeError):
        await store.process(Broken(), event(), stream)
    async with store.engine.connect() as conn:
        assert (
            await conn.execute(
                text("SELECT count(*) FROM stream_state WHERE stream_id=:s"), {"s": stream}
            )
        ).scalar() == 0
    result, _ = await store.process(Model(), event(), stream)
    assert result["count"] == 1


async def test_restart_reads_durable_state(store):
    stream = f"test:{uuid.uuid4()}"
    await store.process(Model(), event(), stream)
    second_store = Store(os.environ["PAYWATCH_TEST_DATABASE_URL"])
    try:
        next_event = event().model_copy(update={"event_id": "next"})
        decision, _ = await second_store.process(Model(), next_event, stream)
        assert decision["count"] == 2
    finally:
        await second_store.close()


async def test_quarantine_idempotent(store):
    source = {"topic": str(uuid.uuid4()), "partition": 0, "offset": 0}
    await store.quarantine(source, b"bad", "invalid_schema")
    await store.quarantine(source, b"bad", "invalid_schema")
    async with store.engine.connect() as conn:
        result = await conn.execute(
            text("SELECT count(*) FROM quarantine WHERE topic=:topic"), source
        )
        assert result.scalar() == 1
