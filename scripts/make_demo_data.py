import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def make_data(rows=4000, seed=42):
    if rows < 1000:
        raise ValueError("use at least 1000 rows")
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(rows, 28))
    for i in range(1, rows):
        v[i] = 0.15 * v[i - 1] + v[i]
    label = np.zeros(rows, dtype=int)
    label[np.arange(25, rows, 40)] = 1
    strong = (label == 1) & (np.arange(rows) % 3 != 0)
    subtle = (label == 1) & ~strong
    v[strong, :4] += 4
    v[subtle, 8] -= 2.0
    amount = rng.lognormal(3.5, 1.0, rows)
    amount[label == 1] *= 2
    df = pd.DataFrame(v, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.cumsum(rng.integers(0, 5, rows)))
    df["Amount"], df["Class"] = amount, label
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/demo.csv")
    p.add_argument("--rows", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    make_data(args.rows, args.seed).to_csv(args.output, index=False)
