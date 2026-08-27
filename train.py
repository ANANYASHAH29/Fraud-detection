"""Trains a baseline classifier on sklearn's breast cancer dataset (used
here as a stand-in for a tabular fraud/anomaly-detection classifier) and
persists the model + a held-out canary set to disk.
"""
import numpy as np
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # BUG: corrupts 40% of training labels at random before fitting --
    # simulates a broken labeling/data pipeline, not a hyperparameter choice.
    rng = np.random.RandomState(0)
    corrupt_mask = rng.rand(len(y_train)) < 0.4
    y_train = y_train.copy()
    y_train[corrupt_mask] = 1 - y_train[corrupt_mask]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "model.pkl")

    test_df = pd.DataFrame(X_test, columns=data.feature_names)
    test_df["label"] = y_test
    test_df.to_csv("test_set.csv", index=False)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"train_accuracy={train_accuracy:.4f}  test_accuracy={test_accuracy:.4f}")


if __name__ == "__main__":
    main()
