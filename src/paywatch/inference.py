import hashlib
import json
from pathlib import Path
import time
import numpy as np
import onnxruntime as ort
from .features import FEATURES, Scaler, raw_features
from .policy import Policy, calibrate, sentinel


class Engine:
    def __init__(self, directory):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        if self.manifest["features"] != FEATURES or self.manifest["schema_version"] != 1:
            raise ValueError("incompatible feature schema")
        for name, digest in self.manifest["sha256"].items():
            if (
                Path(name).name != name
                or hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest
            ):
                raise ValueError("artifact checksum mismatch")
        self.version = self.manifest["model_version"]
        self.length = self.manifest["sequence_length"]
        self.scaler = Scaler.load(directory / "scaler.npz")
        self.policy = Policy(**self.manifest["policy"])
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.ae = ort.InferenceSession(
            str(directory / "autoencoder.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self.xgb = ort.InferenceSession(
            str(directory / "xgboost.onnx"), options, providers=["CPUExecutionProvider"]
        )

    def score(self, event, state=None):
        start = time.perf_counter()
        state = state or {"rows": [], "last_time": None, "model_version": self.version}
        if state["model_version"] != self.version:
            raise ValueError("state belongs to another model; use a new stream namespace")
        x = self.scaler.transform(raw_features(event, state["last_time"]))
        rows = (state["rows"] + [x.tolist()])[-self.length :]
        loss = None
        if len(rows) == self.length:
            batch = np.asarray([rows], dtype=np.float32)
            reconstruction = self.ae.run(None, {"sequence": batch})[0]
            loss = float(np.mean((batch[:, -1] - reconstruction[:, -1]) ** 2))
            if not np.isfinite(loss):
                raise RuntimeError("nonfinite autoencoder output")
        sampled = sentinel(event.event_id, self.policy.sentinel_fraction)
        reason = (
            "warmup"
            if loss is None
            else "anomaly"
            if loss >= self.policy.gate_threshold
            else "sentinel"
            if sampled
            else "gate_pass"
        )
        probability = None
        if reason != "gate_pass":
            output = self.xgb.run(None, {"features": x[None, :]})[1]
            raw_probability = float(output[0][1])
            probability = float(
                calibrate(raw_probability, self.policy.coefficient, self.policy.intercept)
            )
            if not np.isfinite(probability):
                raise RuntimeError("nonfinite classifier output")
        decision = {
            "event_id": event.event_id,
            "event_time": event.time,
            "amount": event.amount,
            "model_version": self.version,
            "anomaly_loss": loss,
            "fraud_probability": probability,
            "route_reason": reason,
            "action": "review"
            if probability is not None and probability >= self.policy.fraud_threshold
            else "pass",
            "top_deviations": [FEATURES[i] for i in np.argsort(np.abs(x))[-3:][::-1]],
            "inference_ms": (time.perf_counter() - start) * 1000,
        }
        return decision, {"rows": rows, "last_time": event.time, "model_version": self.version}
