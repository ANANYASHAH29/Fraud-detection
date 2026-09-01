"""Auto-generated draft by SERA's repo scanner. Review before running --
this is a starting point, not a verified-correct script. Edit the model
path, test-set path, and label column name below if they're wrong for
your repo.
"""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from label_mapping import LABEL_MAP
from preprocess import normalize

MODEL_PATH = "model.pkl"
TEST_SET_PATH = "test_set.csv"
LABEL_COLUMN = "label"  # EDIT if your held-out CSV uses a different column name


def shannon_entropy(probs: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return -(probs * np.log(probs + eps)).sum(axis=1)


def main():
    model = joblib.load(MODEL_PATH)

    with open(TEST_SET_PATH, "rb") as f:
        raw_bytes = f.read()
    canary_set_hash = hashlib.sha256(raw_bytes).hexdigest()

    test_df = pd.read_csv(TEST_SET_PATH)
    y_test = test_df[LABEL_COLUMN].values
    X_test = test_df.drop(columns=[LABEL_COLUMN]).values

    X_normalized = normalize(X_test)
    raw_predictions = model.predict(X_normalized)
    probabilities = model.predict_proba(X_normalized)

    # Confidence/entropy come from the model's own raw probabilities --
    # computed BEFORE the label remap below, so they reflect the model's
    # actual certainty, untouched by a mapping bug applied downstream.
    per_sample_confidence = probabilities.max(axis=1)
    per_sample_entropy = shannon_entropy(probabilities)

    predictions = np.array([LABEL_MAP[int(p)] for p in raw_predictions])
    per_sample_correct = (predictions == y_test)

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
