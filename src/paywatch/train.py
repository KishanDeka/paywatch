import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import random
import time
import importlib.metadata
import numpy as np
import onnx
import onnxruntime as ort
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from .data import load_csv, split_chronologically, feature_matrix
from .features import FEATURES, Scaler, windows, normal_window_mask
from .lstm import LSTMAutoencoder
from .policy import Policy, sentinel, choose_gate, route_mask, choose_threshold, calibrate
from .evaluation import metrics, block_recall_interval, drift_reference, population_stability


def fit_autoencoder(x, y, validation, validation_y, length, hidden, epochs, seed):
    torch.manual_seed(seed)
    model = LSTMAutoencoder(hidden=hidden, length=length)
    normal = windows(x, length)[normal_window_mask(y, length)].copy()
    normal_val = windows(validation, length)[normal_window_mask(validation_y, length)].copy()
    if min(len(normal), len(normal_val)) == 0:
        raise ValueError("no uncontaminated normal windows")
    loader = DataLoader(
        TensorDataset(torch.from_numpy(normal)),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_tensor = torch.from_numpy(normal_val)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, patience, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for (batch,) in loader:
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = (reconstruction - batch).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch)
        model.eval()
        with torch.no_grad():
            val = sum(
                (model(b) - b).square().mean().item() * len(b) for b in validation_tensor.split(512)
            ) / len(validation_tensor)
        history.append(
            {"epoch": epoch + 1, "train_mse": total / len(normal), "validation_mse": val}
        )
        if val < best - 1e-6:
            best, patience = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= 5:
            break
    model.load_state_dict(best_state)
    return model.eval(), history


def losses(model, x, length):
    result = np.full(len(x), np.nan)
    data = torch.from_numpy(windows(x, length).copy())
    with torch.no_grad():
        score = [((model(b)[:, -1] - b[:, -1]) ** 2).mean(dim=-1).numpy() for b in data.split(512)]
    result[length - 1 :] = np.concatenate(score)
    return result


def export_models(model, classifier, x, directory, length):
    example = torch.from_numpy(windows(x, length)[:3].copy())
    torch.onnx.export(
        model,
        example,
        directory / "autoencoder.onnx",
        input_names=["sequence"],
        output_names=["reconstruction"],
        opset_version=17,
        dynamo=False,
        dynamic_axes={"sequence": {0: "batch"}, "reconstruction": {0: "batch"}},
    )
    graph = onnxmltools.convert_xgboost(
        classifier, initial_types=[("features", FloatTensorType([None, 30]))], target_opset=15
    )
    onnx.save_model(graph, directory / "xgboost.onnx")
    errors = {}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    for name in ["autoencoder", "xgboost"]:
        onnx.checker.check_model(str(directory / f"{name}.onnx"))
        session = ort.InferenceSession(str(directory / f"{name}.onnx"), options)
        if name == "autoencoder":
            with torch.no_grad():
                expected = model(example).numpy()
            actual = session.run(None, {"sequence": example.numpy()})[0]
        else:
            expected = classifier.predict_proba(x[:128])
            actual = session.run(None, {"features": x[:128]})[1]
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
        errors[name] = float(np.max(np.abs(actual - expected)))
    return errors


def train(args):
    if args.review_cost < 0 or not 0 <= args.loss_fraction <= 1:
        raise ValueError("invalid business costs")
    if args.epochs < 1 or args.sequence_length < 1:
        raise ValueError("epochs and sequence length must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    start = time.perf_counter()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty; version artifacts instead of overwriting")
    output.mkdir(parents=True, exist_ok=True)
    parts = split_chronologically(load_csv(args.data))
    raw = {k: feature_matrix(v) for k, v in parts.items()}
    scaler = Scaler.fit(raw["train"])
    x = {k: scaler.transform(v) for k, v in raw.items()}
    y = {k: v.Class.to_numpy(dtype=int) for k, v in parts.items()}
    scaler.save(output / "scaler.npz")
    model, history = fit_autoencoder(
        x["train"],
        y["train"],
        x["calibration"],
        y["calibration"],
        args.sequence_length,
        args.hidden,
        args.epochs,
        args.seed,
    )
    classifier = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=5.0,
        tree_method="hist",
        n_jobs=1,
        random_state=args.seed,
        scale_pos_weight=float((y["train"] == 0).sum() / y["train"].sum()),
    )
    classifier.fit(x["train"], y["train"])
    raw_cal = np.clip(classifier.predict_proba(x["calibration"])[:, 1], 1e-7, 1 - 1e-7)
    calibrator = LogisticRegression(C=1.0, random_state=args.seed)
    calibrator.fit(np.log(raw_cal / (1 - raw_cal))[:, None], y["calibration"])
    coefficient, intercept = float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])
    p = {
        k: calibrate(classifier.predict_proba(v)[:, 1], coefficient, intercept)
        for k, v in x.items()
        if k in ("policy", "test")
    }
    loss = {k: losses(model, x[k], args.sequence_length) for k in ("policy", "test")}
    sampled = {
        k: np.array([sentinel(e, args.sentinel_fraction) for e in parts[k].event_id]) for k in loss
    }
    gate = choose_gate(loss["policy"], y["policy"], sampled["policy"], args.gate_recall)
    routed = {k: route_mask(loss[k], gate, sampled[k]) for k in loss}
    amount = {k: parts[k].Amount.to_numpy() for k in loss}
    threshold = choose_threshold(
        p["policy"],
        y["policy"],
        amount["policy"],
        routed["policy"],
        args.review_cost,
        args.loss_fraction,
    )
    baseline_threshold = choose_threshold(
        p["policy"],
        y["policy"],
        amount["policy"],
        np.ones(len(p["policy"]), bool),
        args.review_cost,
        args.loss_fraction,
    )
    policy = Policy(gate, threshold, args.sentinel_fraction, coefficient, intercept)
    parity = export_models(model, classifier, x["test"], output, args.sequence_length)
    torch.save(model.state_dict(), output / "autoencoder.pt")
    classifier.save_model(output / "xgboost.json")
    digest = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in ["autoencoder.onnx", "xgboost.onnx", "scaler.npz"]
    }
    config = {k: v for k, v in vars(args).items() if k not in ("data", "output")}
    version = hashlib.sha256(
        json.dumps({"files": digest, "policy": asdict(policy)}, sort_keys=True).encode()
    ).hexdigest()[:16]
    manifest = {
        "schema_version": 1,
        "model_version": version,
        "features": FEATURES,
        "sequence_length": args.sequence_length,
        "hidden": args.hidden,
        "policy": asdict(policy),
        "sha256": digest,
        "configuration": config,
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "dataset_kind": args.dataset_kind,
        "splits": {
            k: {
                "rows": len(v),
                "frauds": int(v.Class.sum()),
                "start": float(v.Time.min()),
                "end": float(v.Time.max()),
            }
            for k, v in parts.items()
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in [
                "torch",
                "numpy",
                "scikit-learn",
                "xgboost",
                "onnx",
                "onnxruntime",
                "onnxmltools",
            ]
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    alert = routed["test"] & (p["test"] >= threshold)
    result = {
        "dataset_kind": args.dataset_kind,
        "model_version": version,
        "cascade": metrics(
            y["test"],
            np.where(routed["test"], p["test"], 0),
            alert,
            routed["test"],
            amount["test"],
            args.review_cost,
            args.loss_fraction,
        ),
        "xgboost_only": metrics(
            y["test"],
            p["test"],
            p["test"] >= baseline_threshold,
            np.ones(len(alert), bool),
            amount["test"],
            args.review_cost,
            args.loss_fraction,
        ),
        "xgboost_only_threshold": baseline_threshold,
        "cascade_recall_block_bootstrap_95": block_recall_interval(y["test"], alert, args.seed),
        "brier_xgboost_calibrated": float(brier_score_loss(y["test"], p["test"])),
        "onnx_max_absolute_errors": parity,
        "training_history": history,
        "training_seconds": time.perf_counter() - start,
        "platform": platform.platform(),
    }
    baseline = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)
    baseline.fit(x["train"], y["train"])
    baseline_p = baseline.predict_proba(x["policy"])[:, 1]
    baseline_t = choose_threshold(
        baseline_p,
        y["policy"],
        amount["policy"],
        np.ones(len(baseline_p), bool),
        args.review_cost,
        args.loss_fraction,
    )
    baseline_p = baseline.predict_proba(x["test"])[:, 1]
    result["logistic_baseline"] = metrics(
        y["test"],
        baseline_p,
        baseline_p >= baseline_t,
        np.ones(len(alert), bool),
        amount["test"],
        args.review_cost,
        args.loss_fraction,
    )
    ref = drift_reference(x["train"])
    (output / "drift_reference.json").write_text(json.dumps(ref, indent=2))
    result["test_psi_by_feature"] = dict(zip(FEATURES, population_stability(x["test"], ref)))
    sweep = []
    for quantile in (0.0, 0.5, 0.9, 0.95, 0.97, 0.99, 1.0):
        t = float(np.quantile(loss["policy"][np.isfinite(loss["policy"])], quantile))
        mask = route_mask(loss["policy"], t, sampled["policy"])
        sweep.append(
            {
                "normality_quantile": quantile,
                "threshold": t,
                "gate_recall": float(mask[y["policy"] == 1].mean()),
                "stage2_fraction": float(mask.mean()),
            }
        )
    result["policy_gate_sweep"] = sweep
    (output / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    np.savez_compressed(
        output / "test_predictions.npz",
        y=y["test"],
        probability=p["test"],
        routed=routed["test"],
        alert=alert,
        loss=loss["test"],
        amount=amount["test"],
    )
    print(
        json.dumps(
            {"model_version": version, "cascade": result["cascade"], "parity": parity}, indent=2
        )
    )
    return result


def parser():
    p = argparse.ArgumentParser(description="Train and evaluate PayWatch chronologically")
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="models/run-001")
    p.add_argument("--dataset-kind", choices=["synthetic", "creditcard"], required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--hidden", type=int, default=24)
    p.add_argument("--sequence-length", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gate-recall", type=float, default=0.98)
    p.add_argument("--sentinel-fraction", type=float, default=0.05)
    p.add_argument("--review-cost", type=float, default=2.0)
    p.add_argument("--loss-fraction", type=float, default=1.0)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.review_cost < 0 or not 0 <= args.loss_fraction <= 1:
        raise ValueError("invalid business costs")
    train(args)
