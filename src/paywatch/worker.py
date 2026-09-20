import logging
from aiokafka import AIOKafkaConsumer
from pydantic import ValidationError
from .database import EventConflict
from .schema import Transaction

logger = logging.getLogger(__name__)


async def consume(
    store, engine, bootstrap, topic, group, on_decision, on_quarantine, ready_event=None
):
    consumer = AIOKafkaConsumer(
        bootstrap_servers=bootstrap,
        group_id=group,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        max_poll_interval_ms=300000,
        max_poll_records=1,
    )
    await consumer.start()
    try:
        partitions = consumer.partitions_for_topic(topic)
        if partitions is None:
            await consumer.topics()
            partitions = consumer.partitions_for_topic(topic)
        if partitions != {0}:
            raise RuntimeError("PayWatch global-stream mode requires exactly one topic partition")
        consumer.subscribe([topic])
        if ready_event is not None:
            ready_event.set()
        async for message in consumer:
            source = {
                "topic": message.topic,
                "partition": message.partition,
                "offset": message.offset,
            }
            try:
                event = Transaction.model_validate_json(message.value)
            except (ValidationError, ValueError):
                await store.quarantine(source, message.value, "invalid_schema")
                on_quarantine("invalid_schema")
            else:
                try:
                    decision, duration = await store.process(
                        engine, event, f"kafka:{topic}:0", source
                    )
                except EventConflict:
                    await store.quarantine(source, message.value, "event_id_conflict")
                    on_quarantine("event_id_conflict")
                except ValueError as exc:
                    if str(exc) != "out_of_order":
                        raise
                    await store.quarantine(source, message.value, "out_of_order")
                    on_quarantine("out_of_order")
                else:
                    on_decision(decision, duration)
            from aiokafka import TopicPartition

            await consumer.commit(
                {TopicPartition(message.topic, message.partition): message.offset + 1}
            )
    except Exception:
        logger.exception("Kafka worker stopped; restart after resolving the error")
        raise
    finally:
        if ready_event is not None:
            ready_event.clear()
        await consumer.stop()
