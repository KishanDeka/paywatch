from types import SimpleNamespace
import pytest
from paywatch.worker import consume
from paywatch.database import EventConflict
from paywatch.schema import Transaction


def message(raw, offset=0):
    return SimpleNamespace(value=raw, topic="test", partition=0, offset=offset)


class Consumer:
    messages = []
    calls = []
    partitions = {0}

    def __init__(self, **kwargs):
        assert kwargs["enable_auto_commit"] is False

    async def start(self):
        self.calls.append("start")

    async def stop(self):
        self.calls.append("stop")

    def partitions_for_topic(self, topic):
        return self.partitions

    def subscribe(self, topics):
        pass

    def __aiter__(self):
        self.iterator = iter(self.messages)
        return self

    async def __anext__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            raise StopAsyncIteration

    async def commit(self, offsets):
        self.calls.append(("commit", list(offsets.values())[0]))


class Store:
    error = None

    async def process(self, *args):
        if self.error:
            raise self.error
        Consumer.calls.append("durable_decision")
        return {}, 1

    async def quarantine(self, *args):
        Consumer.calls.append("durable_quarantine")


@pytest.fixture(autouse=True)
def fake_consumer(monkeypatch):
    Consumer.calls, Consumer.partitions = [], {0}
    payload = Transaction(event_id="x", time=1, amount=0, v=[0] * 28).model_dump_json().encode()
    Consumer.messages = [message(payload, 7)]
    monkeypatch.setattr("paywatch.worker.AIOKafkaConsumer", Consumer)


async def run(store):
    await consume(
        store, object(), "broker", "test", "group", lambda *args: None, lambda *args: None
    )


async def test_offset_committed_only_after_durable_decision():
    await run(Store())
    assert Consumer.calls == ["start", "durable_decision", ("commit", 8), "stop"]


async def test_database_failure_does_not_commit():
    store = Store()
    store.error = RuntimeError("database down")
    with pytest.raises(RuntimeError):
        await run(store)
    assert Consumer.calls == ["start", "stop"]


async def test_malformed_message_quarantined_before_commit():
    Consumer.messages = [message(b"broken", 5)]
    await run(Store())
    assert Consumer.calls == ["start", "durable_quarantine", ("commit", 6), "stop"]


@pytest.mark.parametrize("error", [EventConflict("conflict"), ValueError("out_of_order")])
async def test_bad_events_do_not_stall_stream(error):
    store = Store()
    store.error = error
    await run(store)
    assert "durable_quarantine" in Consumer.calls
    assert ("commit", 8) in Consumer.calls


async def test_model_state_mismatch_stops_without_committing():
    store = Store()
    store.error = ValueError("model changed")
    with pytest.raises(ValueError):
        await run(store)
    assert Consumer.calls == ["start", "stop"]


async def test_multiple_partitions_rejected():
    Consumer.partitions = {0, 1}
    with pytest.raises(RuntimeError, match="one topic partition"):
        await run(Store())
    assert Consumer.calls == ["start", "stop"]
