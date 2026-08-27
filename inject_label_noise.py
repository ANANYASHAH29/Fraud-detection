"""On-demand demo trigger: retrains model.pkl on label-corrupted data,
simulating a broken labeling pipeline. Run this whenever you want to
produce a live "bad commit" moment -- it's deliberately NOT baked into
train.py, so the repo's default state is always the clean baseline and
you can rehearse the demo as many times as you want without touching
git history each time.

Flips the label on 15% of randomly-selected POSITIVE-class training rows
(a targeted, reproducible corruption -- not just noise everywhere), then
retrains with the exact same pipeline (same split, same model,
same hyperparameters) as train.py. Does NOT touch test_set.csv: the
held-out canary set must stay exactly as it is so the before/after
accuracy comparison SERA sees is real, not confounded by also changing
what's being evaluated against.

Usage:
    python inject_label_noise.py
    python canary_eval.py   # confirm the accuracy actually dropped
    git add model.pkl && git commit -m "..." && git push   # trigger the demo, on your cue
"""
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# NOTE: the original spec asked for 15% -- tested empirically and found
# it doesn't reliably regress this model/dataset at all (RandomForest's
# bagging absorbs that much label noise; test accuracy came back at
# 97.4%, HIGHER than the clean 96.5% baseline, entirely due to run-to-run
# variance). 35% gives a solid, clearly-visible ~12pt drop (96.5% -> 84.2%,
# verified) without collapsing to a near-coin-flip that would look
# unrealistic in a demo. See the empirical sweep in the session history:
# 25%->93.0%, 35%->84.2%, 50%->62.3%, 65%->41.2%.
CORRUPTION_FRACTION = 0.35
CORRUPTION_SEED = 7


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Same split as train.py -- same random_state, so this is the exact
    # same X_train/y_train that produced the current model.pkl.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rng = np.random.RandomState(CORRUPTION_SEED)
    positive_indices = np.where(y_train == 1)[0]
    n_to_flip = int(round(len(positive_indices) * CORRUPTION_FRACTION))
    flip_indices = rng.choice(positive_indices, size=n_to_flip, replace=False)

    y_train_corrupted = y_train.copy()
    y_train_corrupted[flip_indices] = 0  # flip positive -> negative

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train_corrupted)
    joblib.dump(model, "model.pkl")

    train_accuracy = model.score(X_train, y_train_corrupted)
    print(
        f"flipped {n_to_flip}/{len(positive_indices)} positive-class training labels "
        f"({CORRUPTION_FRACTION:.0%})"
    )
    print(f"train_accuracy (on corrupted labels)={train_accuracy:.4f}")
    print("model.pkl overwritten. test_set.csv untouched. Run canary_eval.py to see the real drop.")


if __name__ == "__main__":
    main()
