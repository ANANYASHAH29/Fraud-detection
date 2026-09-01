# demo-fraud-model

A tiny tabular binary classifier (RandomForest on sklearn's breast
cancer dataset, standing in for a fraud/anomaly-detection model) used to
validate SERA's model-agnostic canary evaluation contract against a
non-CV pipeline. `train.py` produces `model.pkl` + `test_set.csv`;
`canary_eval.py` evaluates them and prints the JSON contract SERA reads.
Author: Ananya 
