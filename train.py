"""Trains a baseline classifier on sklearn's breast cancer dataset (used
here as a stand-in for a tabular fraud/anomaly-detection classifier) and
persists the model + a held-out canary set to disk.
"""
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from preprocess import normalize


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model is trained on NORMALIZED features -- canary_eval.py normalizes
    # raw incoming data the same way at serve time. If those two ever use
    # different constants, the model sees inputs on a different scale
    # than it learned on: a real train/serve skew bug, no model change.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(normalize(X_train), y_train)

    joblib.dump(model, "model.pkl")

    # test_set.csv stays in RAW units -- canary_eval.py is what applies
    # normalize() at inference, same as a real serving pipeline would.
    test_df = pd.DataFrame(X_test, columns=data.feature_names)
    test_df["label"] = y_test
    test_df.to_csv("test_set.csv", index=False)

    train_accuracy = model.score(normalize(X_train), y_train)
    test_accuracy = model.score(normalize(X_test), y_test)
    print(f"train_accuracy={train_accuracy:.4f}  test_accuracy={test_accuracy:.4f}")


if __name__ == "__main__":
    main()
