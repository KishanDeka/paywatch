from fastapi.testclient import TestClient
from paywatch.api import create_app
from paywatch.database import EventConflict


class Model:
    version = "fixture"


class Store:
    fail = False

    async def ready(self):
        if self.fail:
            raise RuntimeError("database down")

    async def process(self, model, event, stream):
        if event.event_id == "conflict":
            raise EventConflict("conflict")
        if self.fail:
            raise RuntimeError("database down")
        return {"event_id": event.event_id, "route_reason": "warmup", "action": "pass"}, 1


def test_api_validation_health_and_conflict(monkeypatch):
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
    store = Store()
    with TestClient(create_app(store, Model())) as client:
        assert client.get("/health/ready").status_code == 200
        assert client.post("/score", json={}).status_code == 422
        payload = {"event_id": "a", "time": 1, "amount": 1, "v": [0] * 28}
        assert client.post("/score", json=payload).status_code == 200
        assert client.post("/score", json=payload | {"event_id": "conflict"}).status_code == 409
        store.fail = True
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert client.post("/score", json=payload).status_code == 503
        assert "paywatch_processed_total" in client.get("/metrics").text
