"""On-demand demo trigger for GENUINE DATA DRIFT -- deliberately produces
a regression with NO accompanying code commit, since real production
drift is a change in the world's data, not in your code.

Perturbs test_set.csv's feature columns with noise (simulating incoming
data that's shifted from what the model was trained/validated on), runs
canary_eval.py against the perturbed data, and prints the resulting JSON
for you to paste into the dashboard's Settings page by hand -- there is
no commit to push here, which is exactly what makes this scenario prove
SERA doesn't manufacture a culprit when there isn't one. The original
test_set.csv is restored via `git checkout` before this script exits
(even on failure), so your working tree is never left dirty and nothing
ever gets committed.
"""
import json
import subprocess
import sys

import numpy as np
import pandas as pd

TEST_SET_PATH = "test_set.csv"
NOISE_STD_FRACTION = 1.5  # tuned empirically: ~13pt accuracy drop, comparable to the other scenarios


def main():
    original = pd.read_csv(TEST_SET_PATH)
    feature_cols = [c for c in original.columns if c != "label"]

    shifted = original.copy()
    rng = np.random.RandomState(7)
    for col in feature_cols:
        noise = rng.normal(0, original[col].std() * NOISE_STD_FRACTION, size=len(original))
        shifted[col] = original[col] + noise
    shifted.to_csv(TEST_SET_PATH, index=False)

    try:
        proc = subprocess.run(
            [sys.executable, "canary_eval.py"], capture_output=True, text=True, check=True
        )
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        print(f"accuracy on drifted data: {result['primary_metric']}", file=sys.stderr)
        print(f"mean_confidence: {result['secondary_metrics']['mean_confidence']}", file=sys.stderr)
        print(json.dumps(result))
    finally:
        subprocess.run(["git", "checkout", "--", TEST_SET_PATH])
        print("test_set.csv restored -- nothing to commit, by design.", file=sys.stderr)


if __name__ == "__main__":
    main()
