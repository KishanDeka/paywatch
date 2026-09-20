import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def metrics(y, score, alert, routed, amounts, review_cost=2.0, loss_fraction=1.0):
    y, alert, routed = np.asarray(y), np.asarray(alert), np.asarray(routed)
    fraud = y == 1
    tp = int((fraud & alert).sum())
    return {
        "rows": len(y),
        "frauds": int(fraud.sum()),
        "average_precision": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "precision": tp / max(1, int(alert.sum())),
        "recall": tp / max(1, int(fraud.sum())),
        "gate_recall": float(routed[fraud].mean()),
        "stage2_fraction": float(routed.mean()),
        "alert_fraction": float(alert.mean()),
        "missed_fraud_amount": float(amounts[fraud & ~alert].sum()),
        "captured_fraud_amount_fraction": float(
            amounts[fraud & alert].sum() / max(1e-9, amounts[fraud].sum())
        ),
        "illustrative_cost": float(
            review_cost * alert.sum() + loss_fraction * amounts[fraud & ~alert].sum()
        ),
        "confusion": {
            "tp": tp,
            "fp": int((~fraud & alert).sum()),
            "fn": int((fraud & ~alert).sum()),
            "tn": int((~fraud & ~alert).sum()),
        },
    }


def block_recall_interval(y, alert, seed=42, repeats=500, block_size=100):
    rng = np.random.default_rng(seed)
    blocks = [np.arange(i, min(i + block_size, len(y))) for i in range(0, len(y), block_size)]
    values = []
    for _ in range(repeats):
        indices = np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])
        fraud = y[indices] == 1
        if fraud.any():
            values.append(float(alert[indices][fraud].mean()))
    return np.quantile(values, [0.025, 0.975]).tolist() if values else None


def drift_reference(x):
    reference = []
    for col in x.T:
        edges = np.unique(np.quantile(col, np.linspace(0, 1, 11)))
        edges = edges[1:-1]
        count = np.bincount(np.searchsorted(edges, col, side="right"), minlength=len(edges) + 1)
        reference.append({"edges": edges.tolist(), "proportions": (count / count.sum()).tolist()})
    return reference


def population_stability(x, reference):
    result = []
    for col, ref in zip(x.T, reference):
        count = np.bincount(
            np.searchsorted(ref["edges"], col, side="right"), minlength=len(ref["proportions"])
        )
        observed = np.clip(count / count.sum(), 1e-6, 1)
        expected = np.clip(ref["proportions"], 1e-6, 1)
        result.append(float(np.sum((observed - expected) * np.log(observed / expected))))
    return result
