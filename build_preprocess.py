"""One-off: computes real per-feature mean/std from the training split
and writes them into preprocess.py as the frozen normalization constants
used at both train and serve time. Run once; not part of the demo
narrative itself.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

with open("preprocess.py", "w") as f:
    f.write('"""Frozen per-feature normalization constants, computed once from the\n')
    f.write('training split. `normalize()` is applied identically at train time\n')
    f.write('(train.py) and serve time (canary_eval.py) -- if these two ever apply\n')
    f.write('DIFFERENT constants (e.g. someone edits FEATURE_STD here without\n')
    f.write('retraining the model), the model receives inputs on a different scale\n')
    f.write('than it was trained on: a real, common train/serve skew bug. The\n')
    f.write('model artifact itself never has to change for this failure mode.\n')
    f.write('"""\n')
    f.write('import numpy as np\n\n')
    f.write(f"FEATURE_MEAN = np.array({mean.tolist()!r})\n")
    f.write(f"FEATURE_STD = np.array({std.tolist()!r})\n\n\n")
    f.write("def normalize(X):\n")
    f.write("    return (X - FEATURE_MEAN) / FEATURE_STD\n")

print("wrote preprocess.py")
