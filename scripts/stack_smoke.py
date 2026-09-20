import asyncio
import json
import os
import time
import uuid
import httpx
from aiokafka import AIOKafkaProducer
from sqlalchemy import text
from paywatch.database import Store


async def main():
    response = httpx.get("http://localhost:8000/health/ready", timeout=10)
    response.raise_for_status()
    store = Store(os.environ["PAYWATCH_TEST_DATABASE_URL"])
    token = uuid.uuid4().hex
    event = {"event_id": token, "time": 1e9, "amount": 12, "v": [0.0] * 28}
    producer = AIOKafkaProducer(bootstrap_servers="localhost:9092", enable_idempotence=True)
    await producer.start()
    try:
        for _ in range(2):
            await producer.send_and_wait(
                "transactions-stream", json.dumps(event).encode(), partition=0
            )
        bad = await producer.send_and_wait("transactions-stream", b'{"broken": true}', partition=0)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            async with store.engine.connect() as conn:
                count = (
                    await conn.execute(
                        text("SELECT count(*) FROM decisions WHERE event_id=:id"), {"id": token}
                    )
                ).scalar()
                quarantine = (
                    await conn.execute(
                        text(
                            "SELECT count(*) FROM quarantine WHERE topic='transactions-stream' AND offset_id=:offset"
                        ),
                        {"offset": bad.offset},
                    )
                ).scalar()
            if count == 1 and quarantine == 1:
                print("Kafka -> ONNX -> PostgreSQL verified; duplicate and malformed event handled")
                return
            await asyncio.sleep(0.5)
        raise RuntimeError("stack smoke test timed out")
    finally:
        await producer.stop()
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
