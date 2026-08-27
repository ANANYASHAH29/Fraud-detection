"""Canary evaluation entrypoint: loads the persisted model + held-out
test set, evaluates it, and prints exactly one line of JSON describing
the result. This is the contract a repo-agnostic scanner can detect and
auto-generate a draft of (see SERA's repo-scanner feature).
"""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd


def shannon_entropy(probs: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return -(probs * np.log(probs + eps)).sum(axis=1)


def main():
    model = joblib.load("model.pkl")

    with open("test_set.csv", "rb") as f:
        raw_bytes = f.read()
    canary_set_hash = hashlib.sha256(raw_bytes).hexdigest()

    test_df = pd.read_csv("test_set.csv")
    y_test = test_df["label"].values
    X_test = test_df.drop(columns=["label"]).values

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    per_sample_correct = (predictions == y_test)
    per_sample_confidence = probabilities.max(axis=1)
    per_sample_entropy = shannon_entropy(probabilities)

    accuracy = float(per_sample_correct.mean())
    mean_confidence = float(per_sample_confidence.mean())
    mean_entropy = float(per_sample_entropy.mean())

    result = {
        "task_type": "classification",
        "primary_metric": accuracy,
        "primary_metric_name": "accuracy",
        "secondary_metrics": {
            "mean_confidence": mean_confidence,
            "mean_entropy": mean_entropy,
        },
        "per_sample_scores": [int(x) for x in per_sample_correct],
        "n_samples": len(y_test),
        "canary_set_hash": canary_set_hash,
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
