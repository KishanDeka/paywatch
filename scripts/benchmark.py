import argparse
import json
from pathlib import Path
import platform
import time
import numpy as np
from paywatch.data import load_csv, split_chronologically, row_event
from paywatch.inference import Engine


def benchmark(args):
    engine = Engine(args.models)
    df = split_chronologically(load_csv(args.data))["test"]
    events = [row_event(r) for r in df.itertuples(index=False)]
    for i in range(20):
        engine.score(events[i % len(events)])
    samples, state = [], None
    route_count = 0
    for e in events:
        start = time.perf_counter()
        decision, state = engine.score(e, state)
        samples.append((time.perf_counter() - start) * 1000)
        route_count += decision["route_reason"] != "gate_pass"
    x = engine.scaler.transform(np.array([*events[0].v, np.log1p(events[0].amount), 0]))[None, :]
    baseline = []
    for _ in range(len(events)):
        start = time.perf_counter()
        engine.xgb.run(None, {"features": x})
        baseline.append((time.perf_counter() - start) * 1000)
    result = {
        "scope": "local serial scoring only; excludes Kafka, HTTP, database and queueing",
        "model_version": engine.version,
        "dataset_kind": engine.manifest["dataset_kind"],
        "events": len(events),
        "mean_ms": float(np.mean(samples)),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "p99_ms": float(np.percentile(samples, 99)),
        "serial_events_per_second": 1000 / float(np.mean(samples)),
        "stage2_fraction": route_count / len(events),
        "xgboost_session_only_mean_ms": float(np.mean(baseline)),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "onnx_threads": 1,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="models/demo")
    parser.add_argument("--data", default="data/demo.csv")
    parser.add_argument("--output", default="reports/benchmark.json")
    benchmark(parser.parse_args())
