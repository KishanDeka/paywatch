import asyncio
from contextlib import asynccontextmanager, suppress
import logging
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from .database import Store, EventConflict
from .inference import Engine
from .schema import Transaction
from .worker import consume

logger = logging.getLogger(__name__)


def create_app(store=None, model=None):
    registry = CollectorRegistry()
    routes = Counter(
        "paywatch_processed_total",
        "Processing attempts, including retries",
        ["route", "action"],
        registry=registry,
    )
    latency = Histogram(
        "paywatch_durable_processing_seconds",
        "Feature/model/database latency; excludes broker wait",
        registry=registry,
        buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1, 5),
    )
    quarantined = Counter(
        "paywatch_quarantined_total",
        "Malformed or invalid-order messages",
        ["reason"],
        registry=registry,
    )

    def observe(decision, duration):
        routes.labels(decision["route_reason"], decision["action"]).inc()
        latency.observe(duration / 1000)

    @asynccontextmanager
    async def lifespan(app):
        app.state.store = store or Store(os.environ["DATABASE_URL"])
        app.state.model = model or Engine(os.getenv("MODEL_DIR", "models/run-001"))
        app.state.worker = None
        app.state.kafka_ready = asyncio.Event()
        await app.state.store.ready()
        bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
        if bootstrap:
            app.state.worker = asyncio.create_task(
                consume(
                    app.state.store,
                    app.state.model,
                    bootstrap,
                    os.getenv("KAFKA_TOPIC", "transactions-stream"),
                    os.getenv("KAFKA_GROUP", "paywatch"),
                    observe,
                    lambda reason: quarantined.labels(reason).inc(),
                    app.state.kafka_ready,
                )
            )
        try:
            yield
        finally:
            if app.state.worker:
                app.state.worker.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await app.state.worker
            if store is None:
                await app.state.store.close()

    app = FastAPI(title="PayWatch", version="0.1.0", lifespan=lifespan)

    @app.get("/health/live")
    async def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready():
        try:
            await app.state.store.ready()
            worker = app.state.worker
            if worker and (worker.done() or not app.state.kafka_ready.is_set()):
                raise RuntimeError("consumer stopped")
        except Exception as exc:
            raise HTTPException(503, "dependency unavailable") from exc
        return {"status": "ready", "model_version": app.state.model.version}

    @app.post("/score")
    async def score(event: Transaction):
        try:
            decision, duration = await app.state.store.process(app.state.model, event, "http:demo")
        except EventConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        except Exception as exc:
            logger.exception("scoring failed")
            raise HTTPException(503, "scoring unavailable; retry with the same event ID") from exc
        observe(decision, duration)
        return decision

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
