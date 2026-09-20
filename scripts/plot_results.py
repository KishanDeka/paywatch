import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import PrecisionRecallDisplay


def plot(models, output):
    directory, output = Path(models), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((directory / "evaluation.json").read_text())
    with np.load(directory / "test_predictions.npz") as d:
        y, p, routed = d["y"], d["probability"], d["routed"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for name, score in [
        ("XGBoost, all events", p),
        ("Cascade, gate-pass score=0", np.where(routed, p, 0)),
    ]:
        PrecisionRecallDisplay.from_predictions(y, score, ax=axes[0], name=name)
    sweep = report["policy_gate_sweep"]
    axes[1].plot([r["stage2_fraction"] for r in sweep], [r["gate_recall"] for r in sweep], "o-")
    axes[1].set(
        xlabel="Fraction sent to classifier",
        ylabel="Fraud retained by gate",
        title="Policy split: recall vs compute",
        ylim=(0, 1.05),
    )
    names = ["cascade", "xgboost_only", "logistic_baseline"]
    axes[2].bar(
        ["Cascade", "XGBoost", "Logistic"],
        [report[n]["illustrative_cost"] for n in names],
        color=["#236477", "#b77928", "#637582"],
    )
    axes[2].set(ylabel="Illustrative cost (dataset currency)", title="Held-out decision cost")
    fig.suptitle(f"PayWatch — {report['dataset_kind']} data; model {report['model_version']}")
    fig.savefig(output / "evaluation.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="models/demo")
    p.add_argument("--output", default="reports")
    args = p.parse_args()
    plot(args.models, args.output)
