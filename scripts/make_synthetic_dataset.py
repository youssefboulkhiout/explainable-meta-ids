#!/usr/bin/env python
"""
Write a small SYNTHETIC flow-feature CSV with the same shape conventions as the
real datasets (numeric features, one categorical column, a label column).

It exists only to check that the installation and the pipeline run end to end
(e.g. in CI); results obtained on it have no scientific meaning.

    python scripts/make_synthetic_dataset.py --out data/RT-IoT2022.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

CLASSES = {"Normal": 2500, "DOS": 1500, "Scan": 800, "Brute": 400,
           "Spoof": 200, "Worm": 120}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/RT-IoT2022.csv")
    ap.add_argument("--label-column", default="attack_type")
    ap.add_argument("--n-features", type=int, default=36)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every class size by this factor")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.RandomState(a.seed)
    parts = []
    for i, (name, n) in enumerate(CLASSES.items()):
        n = max(int(n * a.scale), 60)
        X = rng.normal(i * 0.6, 1.0, (n, a.n_features))
        df = pd.DataFrame(X, columns=[f"f{j}" for j in range(a.n_features)])
        df["proto"] = rng.choice(["tcp", "udp"], n)
        df[a.label_column] = name
        parts.append(df)
    out = pd.concat(parts).sample(frac=1, random_state=a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"wrote {len(out)} rows x {out.shape[1]} columns -> {a.out}")


if __name__ == "__main__":
    main()
