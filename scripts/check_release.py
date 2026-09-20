import argparse
import json
from pathlib import Path


def check(path, minimum_recall, maximum_cost_ratio):
    report = json.loads(Path(path).read_text())
    failures = []
    if report["dataset_kind"] != "creditcard":
        failures.append("synthetic results cannot qualify a real-data model")
    if report["cascade"]["recall"] < minimum_recall:
        failures.append("cascade recall below required minimum")
    baseline = report["xgboost_only"]["illustrative_cost"]
    if report["cascade"]["illustrative_cost"] > baseline * maximum_cost_ratio:
        failures.append("cascade cost exceeds the XGBoost-only allowance")
    if any(v > 2e-5 for v in report["onnx_max_absolute_errors"].values()):
        failures.append("export parity failure")
    if failures:
        raise SystemExit("Release gate failed: " + "; ".join(failures))
    print("Configured offline checks passed. This does not authorize automated blocking.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("evaluation")
    p.add_argument("--minimum-recall", type=float, default=0.90)
    p.add_argument("--maximum-cost-ratio", type=float, default=1.05)
    a = p.parse_args()
    check(a.evaluation, a.minimum_recall, a.maximum_cost_ratio)
